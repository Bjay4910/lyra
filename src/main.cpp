#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include "onnx.proto3.pb.h"
#include "tensor.h"
#include "ops.h"

std::string format_shape(const onnx::ValueInfoProto& value_info) {
    const auto& tensor_type = value_info.type().tensor_type();
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < tensor_type.shape().dim_size(); ++i) {
        const auto& dim = tensor_type.shape().dim(i);
        if (dim.has_dim_value()) {
            oss << dim.dim_value();
        } else if (dim.has_dim_param()) {
            oss << dim.dim_param();
        } else {
            oss << "?";
        }
        if (i != tensor_type.shape().dim_size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

// An initializer's bytes may be stored inline in the .onnx file (raw_data) or in a
// side-car file referenced by external_data {location, offset, length}. Protobuf has a
// 2GB message limit, so exporters (including torch.onnx.export) spill any tensor above a
// size threshold -- typically 1KB -- into a .onnx.data blob. For those, raw_data() is
// EMPTY and the real bytes must be read from disk. This loader resolves both cases.
class ExternalDataLoader {
public:
    explicit ExternalDataLoader(std::filesystem::path model_dir)
        : model_dir_(std::move(model_dir)) {}

    // Returns {pointer, byte_count} for the tensor's raw bytes. The pointer stays valid
    // for the lifetime of this loader (external) / the ModelProto (inline).
    std::pair<const char*, size_t> raw_bytes(const onnx::TensorProto& proto) {
        if (proto.data_location() != onnx::TensorProto::EXTERNAL) {
            return {proto.raw_data().data(), proto.raw_data().size()};
        }

        std::string location;
        size_t offset = 0;
        size_t length = 0;
        bool has_length = false;
        for (const auto& kv : proto.external_data()) {
            if (kv.key() == "location") location = kv.value();
            else if (kv.key() == "offset") offset = std::stoull(kv.value());
            else if (kv.key() == "length") { length = std::stoull(kv.value()); has_length = true; }
        }
        if (location.empty()) {
            throw std::runtime_error("initializer '" + proto.name() +
                                     "' is EXTERNAL but declares no 'location'");
        }

        const std::string& blob = load_file(location);
        if (offset > blob.size()) {
            throw std::runtime_error("initializer '" + proto.name() + "': offset " +
                                     std::to_string(offset) + " past end of " + location);
        }
        // Per the ONNX spec an absent 'length' means "read to end of file".
        if (!has_length) length = blob.size() - offset;
        if (offset + length > blob.size()) {
            throw std::runtime_error("initializer '" + proto.name() +
                                     "': external range out of bounds in " + location);
        }
        return {blob.data() + offset, length};
    }

private:
    std::filesystem::path model_dir_;
    std::map<std::string, std::string> cache_;  // node-stable: returned refs stay valid

    const std::string& load_file(const std::string& location) {
        auto it = cache_.find(location);
        if (it != cache_.end()) return it->second;

        std::filesystem::path full = model_dir_ / location;
        std::ifstream f(full, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("failed to open external data file: " + full.string());
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        auto inserted = cache_.emplace(location, ss.str()).first;
        std::cout << "Loaded external data file " << full.filename().string()
                  << " (" << inserted->second.size() << " bytes)" << std::endl;
        return inserted->second;
    }
};

Tensor tensor_from_proto(const onnx::TensorProto& proto, ExternalDataLoader& loader) {
    std::vector<int> shape;
    for (int i = 0; i < proto.dims_size(); ++i) {
        shape.push_back(static_cast<int>(proto.dims(i)));
    }

    Tensor t(shape);

    // FLOAT initializers use either the typed float_data field or packed raw bytes.
    if (proto.float_data_size() > 0) {
        if (static_cast<size_t>(proto.float_data_size()) != t.size()) {
            throw std::runtime_error("initializer '" + proto.name() + "': shape implies " +
                                     std::to_string(t.size()) + " floats but float_data holds " +
                                     std::to_string(proto.float_data_size()));
        }
        for (int i = 0; i < proto.float_data_size(); ++i) t.raw_data()[i] = proto.float_data(i);
        return t;
    }

    auto [bytes, num_bytes] = loader.raw_bytes(proto);
    size_t num_floats = num_bytes / sizeof(float);
    // Loud failure beats a silently zero-filled weight tensor.
    if (num_floats != t.size()) {
        throw std::runtime_error("initializer '" + proto.name() + "': shape implies " +
                                 std::to_string(t.size()) + " floats but data holds " +
                                 std::to_string(num_floats));
    }
    std::memcpy(t.raw_data().data(), bytes, num_bytes);

    return t;
}

// std::map::operator[] default-constructs on a miss, which turns a typo or an unloaded
// initializer into a silently empty tensor. Every read goes through here instead.
const Tensor& get_tensor(const std::map<std::string, Tensor>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("tensor '" + name + "' was never produced or loaded");
    }
    return it->second;
}

const onnx::AttributeProto* find_attribute(const onnx::NodeProto& node, const std::string& name) {
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name) {
            return &attr;
        }
    }
    return nullptr;
}

int get_int_attribute(const onnx::NodeProto& node, const std::string& name, int default_value) {
    const auto* attr = find_attribute(node, name);
    if (attr == nullptr) return default_value;
    return static_cast<int>(attr->i());
}

int get_list_attribute_first(const onnx::NodeProto& node, const std::string& name, int default_value) {
    const auto* attr = find_attribute(node, name);
    if (attr == nullptr || attr->ints_size() == 0) return default_value;
    return static_cast<int>(attr->ints(0));
}

std::vector<int64_t> read_int64_data(const onnx::TensorProto& proto, ExternalDataLoader& loader) {
    std::vector<int64_t> result;
    if (proto.int64_data_size() > 0) {
        for (int i = 0; i < proto.int64_data_size(); ++i) result.push_back(proto.int64_data(i));
        return result;
    }
    auto [bytes, num_bytes] = loader.raw_bytes(proto);
    size_t count = num_bytes / sizeof(int64_t);
    result.resize(count);
    // memcpy, not a reinterpret_cast deref: the byte range has no alignment guarantee.
    if (count > 0) std::memcpy(result.data(), bytes, count * sizeof(int64_t));
    return result;
}

int run(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_file>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    std::ifstream input(model_path, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open file: " << model_path << std::endl;
        return 1;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    std::string file_contents = buffer.str();

    onnx::ModelProto model;
    if (!model.ParseFromString(file_contents)) {
        std::cerr << "Failed to parse ONNX model." << std::endl;
        return 1;
    }

    std::cout << "Successfully parsed ONNX model!" << std::endl;
    std::cout << "IR version: " << model.ir_version() << std::endl;

    const onnx::GraphProto& graph = model.graph();
    std::cout << "\nGraph name: " << graph.name() << std::endl;

    std::cout << "\n--- Inputs ---" << std::endl;
    for (const auto& input_info : graph.input()) {
        std::cout << "  " << input_info.name() << " " << format_shape(input_info) << std::endl;
    }

    std::cout << "\n--- Outputs ---" << std::endl;
    for (const auto& output_info : graph.output()) {
        std::cout << "  " << output_info.name() << " " << format_shape(output_info) << std::endl;
    }

    std::cout << "\n--- Nodes (" << graph.node_size() << " total) ---" << std::endl;
    for (int i = 0; i < graph.node_size(); ++i) {
        const onnx::NodeProto& node = graph.node(i);
        std::cout << "Node " << i << ": " << node.op_type();

        std::cout << "  inputs: [";
        for (int j = 0; j < node.input_size(); ++j) {
            std::cout << node.input(j);
            if (j != node.input_size() - 1) std::cout << ", ";
        }
        std::cout << "]  outputs: [";
        for (int j = 0; j < node.output_size(); ++j) {
            std::cout << node.output(j);
            if (j != node.output_size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\n--- Stage 5: Executor ---" << std::endl;

    ExternalDataLoader ext_loader(std::filesystem::path(model_path).parent_path());

    std::map<std::string, Tensor> tensors;
    std::map<std::string, onnx::TensorProto> raw_initializers;
    int num_external = 0;
    for (const auto& init : graph.initializer()) {
        if (init.data_location() == onnx::TensorProto::EXTERNAL) ++num_external;
        if (init.data_type() == onnx::TensorProto::FLOAT) {
            tensors[init.name()] = tensor_from_proto(init, ext_loader);
        }
        raw_initializers[init.name()] = init;
    }
    std::cout << "Loaded " << tensors.size() << " float initializers ("
              << num_external << " of " << graph.initializer_size()
              << " stored externally)" << std::endl;

    std::vector<int> input_shape;
    const auto& model_input_dims = graph.input(0).type().tensor_type().shape();
    for (int i = 0; i < model_input_dims.dim_size(); ++i) {
        const auto& dim = model_input_dims.dim(i);
        input_shape.push_back(dim.has_dim_value() ? static_cast<int>(dim.dim_value()) : 1);
    }

    Tensor input_tensor(input_shape);
    if (input_shape == std::vector<int>{1, 4}) {
        input_tensor.raw_data() = {1.0f, 2.0f, 3.0f, 4.0f};
    } else {
        std::ifstream bin_file("input.bin", std::ios::binary);
        if (bin_file.is_open()) {
            bin_file.read(reinterpret_cast<char*>(input_tensor.raw_data().data()),
                          input_tensor.size() * sizeof(float));
            std::cout << "Loaded real input from input.bin" << std::endl;
        } else {
            std::cerr << "Warning: input.bin not found, using placeholder 0.5 values" << std::endl;
            for (size_t i = 0; i < input_tensor.size(); ++i) input_tensor.raw_data()[i] = 0.5f;
        }
    }
    tensors["input"] = input_tensor;

    for (int i = 0; i < graph.node_size(); ++i) {
        const onnx::NodeProto& node = graph.node(i);

        if (node.op_type() == "Gemm") {
            const Tensor& x = get_tensor(tensors, node.input(0));
            const Tensor& w = get_tensor(tensors, node.input(1));
            const Tensor& bias = get_tensor(tensors, node.input(2));

            Tensor w_t({w.shape()[1], w.shape()[0]});
            for (int r = 0; r < w.shape()[0]; ++r) {
                for (int c = 0; c < w.shape()[1]; ++c) {
                    w_t.at({c, r}) = w.at({r, c});
                }
            }

            Tensor product = matmul(x, w_t);

            Tensor result(product.shape());
            for (int r = 0; r < product.shape()[0]; ++r) {
                for (int c = 0; c < product.shape()[1]; ++c) {
                    result.at({r, c}) = product.at({r, c}) + bias.at({c});
                }
            }

            tensors[node.output(0)] = result;
        }
        else if (node.op_type() == "Relu") {
            tensors[node.output(0)] = relu(get_tensor(tensors, node.input(0)));
        }
        else if (node.op_type() == "Conv") {
            const Tensor& x = get_tensor(tensors, node.input(0));
            const Tensor& w = get_tensor(tensors, node.input(1));
            Tensor zero_bias({w.shape()[0]});
            const Tensor& bias = node.input_size() > 2 ? get_tensor(tensors, node.input(2))
                                                       : zero_bias;

            int stride = get_list_attribute_first(node, "strides", 1);
            int padding = get_list_attribute_first(node, "pads", 0);
            int groups = get_int_attribute(node, "group", 1);

            tensors[node.output(0)] = conv2d(x, w, bias, stride, padding, groups);
        }
        else if (node.op_type() == "BatchNormalization") {
            const Tensor& x = get_tensor(tensors, node.input(0));
            const Tensor& gamma = get_tensor(tensors, node.input(1));
            const Tensor& beta = get_tensor(tensors, node.input(2));
            const Tensor& mean = get_tensor(tensors, node.input(3));
            const Tensor& var = get_tensor(tensors, node.input(4));

            float epsilon = 1e-5f;
            const auto* eps_attr = find_attribute(node, "epsilon");
            if (eps_attr != nullptr) epsilon = eps_attr->f();

            tensors[node.output(0)] = batch_norm(x, gamma, beta, mean, var, epsilon);
        }
        else if (node.op_type() == "MaxPool") {
            const Tensor& x = get_tensor(tensors, node.input(0));

            int kernel_size = get_list_attribute_first(node, "kernel_shape", 2);
            int stride = get_list_attribute_first(node, "strides", kernel_size);

            tensors[node.output(0)] = max_pool(x, kernel_size, stride);
        }
        else if (node.op_type() == "Softmax") {
            tensors[node.output(0)] = softmax(get_tensor(tensors, node.input(0)));
        }
        else if (node.op_type() == "Add") {
            tensors[node.output(0)] = add(get_tensor(tensors, node.input(0)),
                                          get_tensor(tensors, node.input(1)));
        }
        else if (node.op_type() == "Clip") {
            float min_val = get_tensor(tensors, node.input(1)).raw_data()[0];
            float max_val = get_tensor(tensors, node.input(2)).raw_data()[0];
            tensors[node.output(0)] = clip(get_tensor(tensors, node.input(0)), min_val, max_val);
        }
        else if (node.op_type() == "ReduceMean") {
            tensors[node.output(0)] = reduce_mean_hw(get_tensor(tensors, node.input(0)));
        }
        else if (node.op_type() == "Reshape") {
            auto shape_init = raw_initializers.find(node.input(1));
            if (shape_init == raw_initializers.end()) {
                throw std::runtime_error("Reshape: shape operand '" + node.input(1) +
                                         "' is not an initializer");
            }
            std::vector<int64_t> shape_data = read_int64_data(shape_init->second, ext_loader);
            long long known_product = 1;
            int infer_idx = -1;
            for (size_t i = 0; i < shape_data.size(); ++i) {
                if (shape_data[i] == -1) infer_idx = static_cast<int>(i);
                else known_product *= shape_data[i];
            }
            if (infer_idx >= 0) {
                shape_data[infer_idx] =
                    static_cast<int64_t>(get_tensor(tensors, node.input(0)).size() / known_product);
            }
            std::vector<int> new_shape(shape_data.begin(), shape_data.end());
            tensors[node.output(0)] = reshape(get_tensor(tensors, node.input(0)), new_shape);
        }
    }

    const Tensor& final_output = get_tensor(tensors, graph.output(0).name());
    std::cout << "\nLyra output shape: " << format_shape(graph.output(0)) << std::endl;

    const std::vector<float>& scores = final_output.raw_data();
    int preview = std::min<int>(5, static_cast<int>(scores.size()));
    std::cout << "Lyra output[0..4]: [";
    for (int i = 0; i < preview; ++i) {
        std::cout << scores[i];
        if (i + 1 < preview) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Rank the class logits so a MobileNetV2 run reads as predictions, not raw floats.
    int top_k = std::min<int>(5, static_cast<int>(scores.size()));
    std::vector<int> ranked(scores.size());
    for (size_t i = 0; i < ranked.size(); ++i) ranked[i] = static_cast<int>(i);
    std::partial_sort(ranked.begin(), ranked.begin() + top_k, ranked.end(),
                      [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::cout << "\nLyra top " << top_k << " predictions (class index : score):" << std::endl;
    for (int i = 0; i < top_k; ++i) {
        std::cout << "  class " << ranked[i] << ": "
                  << std::fixed << std::setprecision(4) << scores[ranked[i]]
                  << std::defaultfloat << std::endl;
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}