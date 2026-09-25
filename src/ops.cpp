#include "ops.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("add: shape mismatch");
    }

    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.raw_data()[i] = a.raw_data()[i] + b.raw_data()[i];
    }
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("matmul: only 2D tensors supported");
    }

    int M = a_shape[0];
    int K = a_shape[1];
    int K2 = b_shape[0];
    int N = b_shape[1];

    if (K != K2) {
        throw std::runtime_error("matmul: inner dimensions do not match");
    }

    Tensor result({M, N});

    // i-k-j order with raw pointers. Two reasons this beats the textbook i-j-k form:
    //   1. at({i, k}) builds a std::vector<int> per element access, so the old inner
    //      loop did two heap allocations per multiply-accumulate. That overhead, not
    //      the arithmetic, was the cost.
    //   2. i-k-j walks both b and result along contiguous rows, so the inner loop is
    //      unit-stride and vectorizable.
    // Each result[i][j] still accumulates its K terms in ascending k, so the output is
    // bit-identical to the i-j-k version -- the accumulations are just interleaved.
    const float* a_data = a.raw_data().data();
    const float* b_data = b.raw_data().data();
    float* r_data = result.raw_data().data();

    for (int i = 0; i < M; ++i) {
        const float* a_row = a_data + static_cast<size_t>(i) * K;
        float* r_row = r_data + static_cast<size_t>(i) * N;
        for (int k = 0; k < K; ++k) {
            float a_ik = a_row[k];
            const float* b_row = b_data + static_cast<size_t>(k) * N;
            for (int j = 0; j < N; ++j) {
                r_row[j] += a_ik * b_row[j];
            }
        }
    }

    return result;
}

Tensor relu(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.raw_data()[i] = std::max(0.0f, x.raw_data()[i]);
    }
    return result;
}

Tensor softmax(const Tensor& x) {
    const auto& shape = x.shape();
    int batch = shape[0];
    int num_classes = shape[1];

    Tensor result(shape);

    for (int b = 0; b < batch; ++b) {
        float max_val = x.at({b, 0});
        for (int c = 1; c < num_classes; ++c) {
            max_val = std::max(max_val, x.at({b, c}));
        }

        float sum = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float exp_val = std::exp(x.at({b, c}) - max_val);
            result.at({b, c}) = exp_val;
            sum += exp_val;
        }

        for (int c = 0; c < num_classes; ++c) {
            result.at({b, c}) = result.at({b, c}) / sum;
        }
    }

    return result;
}

Tensor batch_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& running_mean, const Tensor& running_var, float epsilon) {
    const auto& shape = x.shape();
    int batch = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];

    Tensor result(shape);

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float mean = running_mean.at({c});
            float var = running_var.at({c});
            float g = gamma.at({c});
            float be = beta.at({c});
            float denom = std::sqrt(var + epsilon);

            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float val = x.at({b, c, h, w});
                    result.at({b, c, h, w}) = ((val - mean) / denom) * g + be;
                }
            }
        }
    }

    return result;
}

Tensor max_pool(const Tensor& x, int kernel_size, int stride) {
    const auto& shape = x.shape();
    int batch = shape[0];
    int channels = shape[1];
    int in_height = shape[2];
    int in_width = shape[3];

    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;

    Tensor result({batch, channels, out_height, out_width});

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int h_start = oh * stride;
                    int w_start = ow * stride;

                    float max_val = x.at({b, c, h_start, w_start});
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            float val = x.at({b, c, h_start + kh, w_start + kw});
                            max_val = std::max(max_val, val);
                        }
                    }

                    result.at({b, c, oh, ow}) = max_val;
                }
            }
        }
    }

    return result;
}

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride, int padding, int groups) {
    const auto& x_shape = x.shape();
    const auto& w_shape = weight.shape(); // [out_channels, in_channels/groups, kH, kW]

    int batch = x_shape[0];
    int in_height = x_shape[2];
    int in_width = x_shape[3];

    int out_channels = w_shape[0];
    int in_channels_per_group = w_shape[1];
    int kernel_h = w_shape[2];
    int kernel_w = w_shape[3];
    int out_channels_per_group = out_channels / groups;

    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

    Tensor result({batch, out_channels, out_height, out_width});

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            int group_idx = oc / out_channels_per_group;
            int in_channel_start = group_idx * in_channels_per_group;

            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;

                    for (int icg = 0; icg < in_channels_per_group; ++icg) {
                        int ic = in_channel_start + icg;
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) continue;

                                float in_val = x.at({b, ic, ih, iw});
                                float w_val = weight.at({oc, icg, kh, kw});
                                sum += in_val * w_val;
                            }
                        }
                    }

                    sum += bias.at({oc});
                    result.at({b, oc, oh, ow}) = sum;
                }
            }
        }
    }

    return result;
}

// im2col + GEMM formulation of conv2d(). Identical signature and semantics to the
// naive conv2d() above, just a different route to the same numbers.
//
// The idea: a convolution is a matrix multiply in disguise. For each output spatial
// position we gather the flattened input patch the kernel would touch into one COLUMN
// of a [icpg*kH*kW, out_h*out_w] buffer. The weights for a group are already laid out
// contiguously as [ocpg, icpg*kH*kW], so one GEMM produces every output channel at
// every spatial position for that group at once.
//
// Grouped conv (including MobileNetV2's depthwise, where groups == in_channels and
// icpg == 1) is handled by running the whole build + GEMM once per group: group g reads
// input channels [g*icpg, (g+1)*icpg) and writes output channels [g*ocpg, (g+1)*ocpg).
Tensor conv2d_im2col(const Tensor& x, const Tensor& weight, const Tensor& bias,
                     int stride, int padding, int groups) {
    const auto& x_shape = x.shape();
    const auto& w_shape = weight.shape(); // [out_channels, in_channels/groups, kH, kW]

    int batch = x_shape[0];
    int in_channels = x_shape[1];
    int in_height = x_shape[2];
    int in_width = x_shape[3];

    int out_channels = w_shape[0];
    int in_channels_per_group = w_shape[1];
    int kernel_h = w_shape[2];
    int kernel_w = w_shape[3];

    if (groups <= 0 || out_channels % groups != 0) {
        throw std::runtime_error("conv2d_im2col: out_channels not divisible by groups");
    }
    if (in_channels != in_channels_per_group * groups) {
        throw std::runtime_error("conv2d_im2col: input channels inconsistent with weight/groups");
    }
    int out_channels_per_group = out_channels / groups;

    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;

    int patch_size = in_channels_per_group * kernel_h * kernel_w; // GEMM's K
    int num_patches = out_height * out_width;                     // GEMM's N

    Tensor result({batch, out_channels, out_height, out_width});
    float* out_data = result.raw_data().data();
    const float* x_data = x.raw_data().data();
    const float* w_data = weight.raw_data().data();

    // Weight rows for a group are a contiguous slice of the [oc, icpg, kH, kW] buffer,
    // so the "reshape to 2D" is a straight copy -- no transposing or gathering needed.
    Tensor w_mat({out_channels_per_group, patch_size});
    // Allocated once and refilled per (batch, group); every cell is written each pass,
    // including the explicit zeros for padding, so nothing stale leaks between groups.
    Tensor col({patch_size, num_patches});
    float* col_data = col.raw_data().data();

    for (int b = 0; b < batch; ++b) {
        for (int g = 0; g < groups; ++g) {
            std::copy(w_data + static_cast<size_t>(g) * out_channels_per_group * patch_size,
                      w_data + static_cast<size_t>(g + 1) * out_channels_per_group * patch_size,
                      w_mat.raw_data().data());

            for (int icg = 0; icg < in_channels_per_group; ++icg) {
                int ic = g * in_channels_per_group + icg;
                const float* channel = x_data +
                    (static_cast<size_t>(b) * in_channels + ic) * in_height * in_width;

                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int row = (icg * kernel_h + kh) * kernel_w + kw;
                        float* col_row = col_data + static_cast<size_t>(row) * num_patches;

                        for (int oh = 0; oh < out_height; ++oh) {
                            int ih = oh * stride - padding + kh;
                            float* col_out = col_row + static_cast<size_t>(oh) * out_width;

                            if (ih < 0 || ih >= in_height) {
                                std::fill(col_out, col_out + out_width, 0.0f);
                                continue;
                            }
                            const float* in_row = channel + static_cast<size_t>(ih) * in_width;

                            for (int ow = 0; ow < out_width; ++ow) {
                                int iw = ow * stride - padding + kw;
                                col_out[ow] = (iw < 0 || iw >= in_width) ? 0.0f : in_row[iw];
                            }
                        }
                    }
                }
            }

            // [ocpg, patch_size] x [patch_size, num_patches] -> [ocpg, num_patches]
            Tensor product = matmul(w_mat, col);
            const float* prod_data = product.raw_data().data();

            for (int r = 0; r < out_channels_per_group; ++r) {
                int oc = g * out_channels_per_group + r;
                float b_val = bias.at({oc});
                const float* prod_row = prod_data + static_cast<size_t>(r) * num_patches;
                float* dst = out_data +
                    (static_cast<size_t>(b) * out_channels + oc) * num_patches;

                for (int p = 0; p < num_patches; ++p) {
                    dst[p] = prod_row[p] + b_val;
                }
            }
        }
    }

    return result;
}

Tensor clip(const Tensor& x, float min_val, float max_val) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.raw_data()[i] = std::min(std::max(x.raw_data()[i], min_val), max_val);
    }
    return result;
}

Tensor reduce_mean_hw(const Tensor& x) {
    const auto& shape = x.shape();
    int batch = shape[0], channels = shape[1], height = shape[2], width = shape[3];
    Tensor result({batch, channels, 1, 1});

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w)
                    sum += x.at({b, c, h, w});
            result.at({b, c, 0, 0}) = sum / (height * width);
        }
    }
    return result;
}

Tensor reshape(const Tensor& x, const std::vector<int>& new_shape) {
    Tensor result(new_shape);
    result.raw_data() = x.raw_data();
    return result;
}