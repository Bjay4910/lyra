#pragma once

#include "tensor.h"

Tensor add(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& x);
Tensor softmax(const Tensor& x);
Tensor batch_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& running_mean, const Tensor& running_var, float epsilon);
Tensor max_pool(const Tensor& x, int kernel_size, int stride);
// Naive baseline: direct 7-deep loop nest. Kept for correctness/speed comparison
// against conv2d_im2col(); the executor no longer calls this.
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride, int padding, int groups);

// Same semantics as conv2d(), implemented as im2col + GEMM.
Tensor conv2d_im2col(const Tensor& x, const Tensor& weight, const Tensor& bias,
                     int stride, int padding, int groups);
Tensor clip(const Tensor& x, float min_val, float max_val);
Tensor reduce_mean_hw(const Tensor& x);
Tensor reshape(const Tensor& x, const std::vector<int>& new_shape);