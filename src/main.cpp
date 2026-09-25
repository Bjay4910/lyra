#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <chrono>
#include <cmath>
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

// Stage 6 scratch harness: compares conv2d() against conv2d_im2col() on the REAL
// tensors flowing through the graph (a node's actual input activation, weight and bias
// at the moment it executes), not synthetic data. Enabled with --conv-check.
struct ConvCheck {
    bool enabled = false;
    std::vector<int> node_indices;   // empty => every Conv node
    int failures = 0;

    bool wants(int node_index) const {
        if (!enabled) return false;
        if (node_indices.empty()) return true;
        return std::find(node_indices.begin(), node_indices.end(), node_index) !=
               node_indices.end();
    }

    void print_header() const {
        std::cout << "\n--- Stage 6: conv2d() vs conv2d_im2col() on real nodes ---" << std::endl;
        std::cout << std::left << std::setw(6) << "NODE"
                  << std::setw(20) << "INPUT"
                  << std::setw(20) << "WEIGHT"
                  << std::setw(7) << "GRP"
                  << std::setw(5) << "S"
                  << std::setw(5) << "P"
                  << std::right << std::setw(12) << "NAIVE(ms)"
                  << std::setw(12) << "IM2COL(ms)"
                  << std::setw(10) << "SPEEDUP"
                  << std::setw(13) << "MAX |DIFF|"
                  << std::setw(8) << "RESULT" << std::endl;
    }

    // Returns the im2col output so the caller can keep executing the graph with it.
    Tensor compare(int node_index, const Tensor& x, const Tensor& w, const Tensor& bias,
                   int stride, int padding, int groups, float tolerance) {
        using Clock = std::chrono::high_resolution_clock;

        auto t0 = Clock::now();
        Tensor naive = conv2d(x, w, bias, stride, padding, groups);
        auto t1 = Clock::now();
        Tensor fast = conv2d_im2col(x, w, bias, stride, padding, groups);
        auto t2 = Clock::now();

        double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fast_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        bool ok = naive.shape() == fast.shape();
        float max_diff = 0.0f;
        if (ok) {
            for (size_t i = 0; i < naive.size(); ++i) {
                max_diff = std::max(max_diff,
                                    std::fabs(naive.raw_data()[i] - fast.raw_data()[i]));
            }
            ok = max_diff <= tolerance;
        }
        if (!ok) ++failures;

        auto shape_str = [](const std::vector<int>& s) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < s.size(); ++i) {
                oss << s[i];
                if (i + 1 < s.size()) oss << "x";
            }
            oss << "]";
            return oss.str();
        };

        std::cout << std::left << std::setw(6) << node_index
                  << std::setw(20) << shape_str(x.shape())
                  << std::setw(20) << shape_str(w.shape())
                  << std::setw(7) << (groups == w.shape()[0] && groups > 1
                                          ? "DW(" + std::to_string(groups) + ")"
                                          : std::to_string(groups))
                  << std::setw(5) << stride
                  << std::setw(5) << padding
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << naive_ms
                  << std::setw(12) << fast_ms
                  << std::setw(9) << (fast_ms > 0.0 ? naive_ms / fast_ms : 0.0) << "x"
                  << std::scientific << std::setprecision(3) << std::setw(13) << max_diff
                  << std::defaultfloat
                  << std::setw(8) << (ok ? "PASS" : "FAIL") << std::endl;

        return fast;
    }
};

// One class label per line, ordered by class index. Returns empty if the file is missing.
std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        labels.push_back(line);
    }
    return labels;
}

int run(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_file> [--conv-check[=n,n,...]]"
                  << std::endl;
        return 1;
    }

    auto run_start = std::chrono::high_resolution_clock::now();
    std::string model_path = argv[1];

    ConvCheck conv_check;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--conv-check", 0) != 0) continue;
        conv_check.enabled = true;
        if (arg.size() > 13 && arg[12] == '=') {
            std::stringstream ss(arg.substr(13));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) conv_check.node_indices.push_back(std::stoi(tok));
            }
        }
    }

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

    // Per-op-TYPE wall-clock accounting. Keyed by op_type so all 52 Conv nodes roll up
    // into one number -- the question is "which kind of op owns the runtime", not
    // "which individual node".
    std::map<std::string, double> time_by_op_type_ms;
    std::map<std::string, int> count_by_op_type;
    using Clock = std::chrono::high_resolution_clock;

    if (conv_check.enabled) conv_check.print_header();

    auto graph_start = Clock::now();

    for (int i = 0; i < graph.node_size(); ++i) {
        const onnx::NodeProto& node = graph.node(i);
        auto node_start = Clock::now();

        if (node.op_type() == "Gemm") {
            const Tensor& x = get_tensor(tensors, node.input(0));
            const Tensor& w = get_tensor(tensors, node.input(1));
            const Tensor& bias = get_tensor(tensors, node.input(2));

            // ONNX Gemm stores the weight as [out_features, in_features]; matmul() needs
            // [in_features, out_features]. Raw pointer arithmetic rather than at(): the
            // old version constructed a std::vector<int> per element, which for the
            // 1000x1280 classifier weight meant ~1.28M heap allocations -- the same
            // pattern that used to dominate matmul(). This is pure data movement, no
            // arithmetic, so the transposed buffer is bit-identical either way.
            int w_rows = w.shape()[0];
            int w_cols = w.shape()[1];
            Tensor w_t({w_cols, w_rows});
            const float* w_src = w.raw_data().data();
            float* w_dst = w_t.raw_data().data();
            for (int r = 0; r < w_rows; ++r) {
                const float* src_row = w_src + static_cast<size_t>(r) * w_cols;
                for (int c = 0; c < w_cols; ++c) {
                    w_dst[static_cast<size_t>(c) * w_rows + r] = src_row[c];
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

            // Stage 6: im2col + GEMM is the production path. conv2d() remains in ops.cpp
            // as the naive reference the --conv-check harness validates against.
            tensors[node.output(0)] =
                conv_check.wants(i)
                    ? conv_check.compare(i, x, w, bias, stride, padding, groups, 1e-4f)
                    : conv2d_im2col(x, w, bias, stride, padding, groups);
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

        std::chrono::duration<double, std::milli> node_ms = Clock::now() - node_start;
        time_by_op_type_ms[node.op_type()] += node_ms.count();
        count_by_op_type[node.op_type()] += 1;
    }

    std::chrono::duration<double, std::milli> graph_ms = Clock::now() - graph_start;

    if (conv_check.enabled) {
        std::cout << (conv_check.failures == 0
                          ? "conv-check: ALL NODES MATCH within 1e-4"
                          : "conv-check: " + std::to_string(conv_check.failures) + " NODE(S) FAILED")
                  << std::endl;
        if (conv_check.failures != 0) return 1;
    }

    {
        std::vector<std::pair<std::string, double>> sorted(time_by_op_type_ms.begin(),
                                                           time_by_op_type_ms.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        double total_op_ms = 0.0;
        for (const auto& kv : sorted) total_op_ms += kv.second;

        std::cout << "\n--- Profile: time by op type (slowest first) ---" << std::endl;
        std::cout << std::left << std::setw(22) << "OP TYPE"
                  << std::right << std::setw(7) << "NODES"
                  << std::setw(14) << "TOTAL (ms)"
                  << std::setw(14) << "AVG (ms)"
                  << std::setw(10) << "% OPS" << std::endl;
        for (const auto& [op_type, ms] : sorted) {
            int n = count_by_op_type[op_type];
            std::cout << std::left << std::setw(22) << op_type
                      << std::right << std::setw(7) << n
                      << std::fixed << std::setprecision(3)
                      << std::setw(14) << ms
                      << std::setw(14) << (ms / n)
                      << std::setprecision(2)
                      << std::setw(9) << (total_op_ms > 0.0 ? 100.0 * ms / total_op_ms : 0.0) << "%"
                      << std::defaultfloat << std::endl;
        }
        std::cout << std::left << std::setw(22) << "TOTAL"
                  << std::right << std::setw(7) << graph.node_size()
                  << std::fixed << std::setprecision(3) << std::setw(14) << total_op_ms
                  << std::defaultfloat << std::endl;
        std::cout << "\nGraph execution wall clock: "
                  << std::fixed << std::setprecision(3) << graph_ms.count() << " ms"
                  << std::defaultfloat << std::endl;
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

    // Numerical diff against PyTorch's raw logits (pre-softmax), read with the same
    // flat-float32 layout as input.bin. Optional: models without a reference file skip it.
    {
        std::cout << "\n--- Validation vs PyTorch ---" << std::endl;
        std::ifstream ref_file("pytorch_output.bin", std::ios::binary);
        if (!ref_file.is_open()) {
            std::cout << "pytorch_output.bin not found, skipping numerical diff" << std::endl;
        } else {
            std::vector<float> reference(scores.size());
            ref_file.read(reinterpret_cast<char*>(reference.data()),
                          reference.size() * sizeof(float));
            // The file is looked up in the working directory, so it may belong to a
            // different model (e.g. MobileNet's 1000 logits when running tiny_mlp). A
            // size mismatch in either direction means "not our reference": skip rather
            // than diff against zero-filled or truncated data.
            bool size_ok = static_cast<size_t>(ref_file.gcount()) == reference.size() * sizeof(float) &&
                           ref_file.peek() == std::ifstream::traits_type::eof();
            if (!size_ok) {
                std::cout << "pytorch_output.bin does not hold " << scores.size()
                          << " floats (belongs to a different model?), skipping numerical diff"
                          << std::endl;
            } else {
                const float tolerance = 1e-3f;
                float max_diff = 0.0f;
                int max_diff_idx = 0;
                for (size_t i = 0; i < scores.size(); ++i) {
                    float d = std::fabs(scores[i] - reference[i]);
                    if (d > max_diff) {
                        max_diff = d;
                        max_diff_idx = static_cast<int>(i);
                    }
                }
                std::cout << "Max |Lyra - PyTorch|: " << std::scientific << std::setprecision(3)
                          << max_diff << std::defaultfloat << " at class " << max_diff_idx
                          << " (Lyra " << scores[max_diff_idx] << ", PyTorch "
                          << reference[max_diff_idx] << ")" << std::endl;
                std::cout << (max_diff <= tolerance ? "VALIDATION PASSED" : "VALIDATION FAILED")
                          << " (tolerance 1e-3)" << std::endl;
            }
        }
    }

    // Softmax is a display-only step: logits -> probabilities for the top-k printout.
    Tensor probs_tensor = softmax(final_output);
    const std::vector<float>& probs = probs_tensor.raw_data();

    int top_k = std::min<int>(5, static_cast<int>(scores.size()));
    auto rank_top_k = [top_k](const std::vector<float>& v) {
        std::vector<int> idx(v.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                          [&v](int a, int b) { return v[a] > v[b]; });
        idx.resize(top_k);
        return idx;
    };
    std::vector<int> ranked = rank_top_k(probs);

    // Softmax is monotonic, so the ranking must match the logit ranking exactly.
    if (ranked != rank_top_k(scores)) {
        throw std::runtime_error("softmax reordered the top-" + std::to_string(top_k) +
                                 " classes; softmax() or the ranking is broken");
    }

    // Labels are optional: only used if the file exists and has one line per output class
    // (so a non-ImageNet model like tiny_mlp just gets bare indices).
    std::vector<std::string> labels = load_labels("imagenet_classes.txt");
    if (!labels.empty() && labels.size() != scores.size()) {
        std::cout << "\nimagenet_classes.txt has " << labels.size() << " labels but the model has "
                  << scores.size() << " outputs; printing indices only" << std::endl;
        labels.clear();
    }

    std::cout << "\nLyra top " << top_k << " predictions (class index: label - confidence):"
              << std::endl;
    for (int i = 0; i < top_k; ++i) {
        std::cout << "  " << ranked[i] << ": ";
        if (!labels.empty()) std::cout << labels[ranked[i]] << " - ";
        std::cout << std::fixed << std::setprecision(1) << 100.0f * probs[ranked[i]] << "%"
                  << std::defaultfloat << std::endl;
    }

    std::chrono::duration<double, std::milli> run_ms =
        std::chrono::high_resolution_clock::now() - run_start;
    std::cout << "\nTotal wall clock (parse + load + execute): "
              << std::fixed << std::setprecision(3) << run_ms.count() << " ms"
              << std::defaultfloat << std::endl;

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