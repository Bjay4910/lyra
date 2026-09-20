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

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a.at({i, k}) * b.at({k, j});
            }
            result.at({i, j}) = sum;
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