#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>

class Tensor {
public:
    Tensor() : shape_({}), strides_({}) {}

    Tensor(const std::vector<int>& shape)
        : shape_(shape)
    {
        strides_ = compute_strides(shape_);
        size_t total = 1;
        for (int dim : shape_) {
            total *= static_cast<size_t>(dim);
        }
        data_.resize(total, 0.0f);
    }

    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    size_t size() const { return data_.size(); }

    float& at(const std::vector<int>& indices) {
        return data_[flat_index(indices)];
    }

    float at(const std::vector<int>& indices) const {
        return data_[flat_index(indices)];
    }

    std::vector<float>& raw_data() { return data_; }
    const std::vector<float>& raw_data() const { return data_; }

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    std::vector<float> data_;

    static std::vector<int> compute_strides(const std::vector<int>& shape) {
        std::vector<int> strides(shape.size());
        int running = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = running;
            running *= shape[i];
        }
        return strides;
    }

    size_t flat_index(const std::vector<int>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("Tensor::at - wrong number of indices");
        }
        size_t idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            idx += static_cast<size_t>(indices[i]) * static_cast<size_t>(strides_[i]);
        }
        return idx;
    }
};
