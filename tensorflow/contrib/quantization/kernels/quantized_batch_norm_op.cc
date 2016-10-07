/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/contrib/quantization/kernels/quantization_utils.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace {

// A slow but straightforward implementation of batch normalization.
template <typename T1, typename T2>
void ReferenceBatchNorm(const Tensor& input, const float input_min,
                        const float input_max, const Tensor& mean,
                        float mean_min, float mean_max, const Tensor& var,
                        float var_min, float var_max, const Tensor& beta,
                        float beta_min, float beta_max, const Tensor& gamma,
                        float gamma_min, float gamma_max,
                        float variance_epsilon, bool scale_after_normalization,
                        Tensor* output, float* output_min, float* output_max) {
  auto input_flat = input.flat<T1>();
  auto mean_flat = mean.flat<T1>();
  auto var_flat = var.flat<T1>();
  auto beta_flat = beta.flat<T1>();
  auto gamma_flat = gamma.flat<T1>();
  auto output_flat = output->flat<T2>();

  const int depth = mean.dim_size(0);
  const int row_count = input_flat.size() / depth;

  *output_min = std::numeric_limits<float>::max();
  *output_max = std::numeric_limits<float>::lowest();
  for (int pass = 0; pass < 2; ++pass) {
    const bool is_range_pass = (pass == 0);
    for (int row_index = 0; row_index < row_count; ++row_index) {
      for (int channel = 0; channel < depth; ++channel) {
        const int input_index = (row_index * depth) + channel;
        const float input_value =
            QuantizedToFloat(input_flat(input_index), input_min, input_max);
        const float mean_value =
            QuantizedToFloat(mean_flat(channel), mean_min, mean_max);
        const float var_value =
            QuantizedToFloat(var_flat(channel), var_min, var_max);
        const float beta_value =
            QuantizedToFloat(beta_flat(channel), beta_min, beta_max);
        const float gamma_value =
            QuantizedToFloat(gamma_flat(channel), gamma_min, gamma_max);
        float output_value;
        if (scale_after_normalization) {
          output_value = (((input_value - mean_value) /
                           sqrtf(var_value + variance_epsilon)) *
                          gamma_value) +
                         beta_value;
        } else {
          output_value = ((input_value - mean_value) /
                          sqrtf(var_value + variance_epsilon)) +
                         beta_value;
        }
        if (is_range_pass) {
          *output_min = std::min(output_value, *output_min);
          *output_max = std::max(output_value, *output_max);
        } else {
          output_flat(input_index) =
              FloatToQuantized<T2>(output_value, *output_min, *output_max);
        }
      }
    }
  }
}

// An implementation of batch normalization that does the main calculations
// using only fixed-point arithmetic. There's a prologue with some floating
// calculations, but assuming the weights are constant these could be hoisted to
// an offline process, or baked into the weights.
template <typename T1, typename T2>
void FixedPointBatchNorm(const Tensor& input, const float input_min,
                         const float input_max, const Tensor& mean,
                         float mean_min, float mean_max, const Tensor& var,
                         float var_min, float var_max, const Tensor& beta,
                         float beta_min, float beta_max, const Tensor& gamma,
                         float gamma_min, float gamma_max,
                         float variance_epsilon, bool scale_after_normalization,
                         Tensor* output, float* output_min, float* output_max) {
  auto input_flat = input.flat<T1>();
  auto mean_flat = mean.flat<T1>();
  auto var_flat = var.flat<T1>();
  auto beta_flat = beta.flat<T1>();
  auto gamma_flat = gamma.flat<T1>();
  auto output_flat = output->flat<T2>();

  const int depth = mean.dim_size(0);
  const int row_count = input_flat.size() / depth;

  // The range here is chosen so that typical input values fit in without any
  // overflow or loss of precision, going from +1m to -1m with 10 bits of fixed
  // point precision.
  *output_min = -(1 << 20);
  *output_max = (1 << 20);

  Tensor scale_tensor(DataTypeToEnum<T2>::v(), {depth});
  auto scale_flat = scale_tensor.flat<T2>();
  Tensor offset_tensor(DataTypeToEnum<T2>::v(), {depth});
  auto offset_flat = offset_tensor.flat<T2>();
  for (int channel = 0; channel < depth; ++channel) {
    const float mean_value =
        QuantizedToFloat(mean_flat(channel), mean_min, mean_max);
    const float var_value =
        QuantizedToFloat(var_flat(channel), var_min, var_max);
    const float beta_value =
        QuantizedToFloat(beta_flat(channel), beta_min, beta_max);
    const float gamma_value =
        QuantizedToFloat(gamma_flat(channel), gamma_min, gamma_max);
    float scale_value;
    if (scale_after_normalization) {
      scale_value = (1.0f / sqrtf(var_value + variance_epsilon)) * gamma_value;
    } else {
      scale_value = (1.0f / sqrtf(var_value + variance_epsilon));
    }
    const float offset_value = (-mean_value * scale_value) + beta_value;
    scale_flat(channel) =
        FloatToQuantized<T2>(scale_value, *output_min, *output_max);
    offset_flat(channel) =
        FloatToQuantized<T2>(offset_value, *output_min, *output_max);
  }

  const T2 one_in_output_space =
      FloatToQuantized<T2>(1.0f, *output_min, *output_max);
  for (int row_index = 0; row_index < row_count; ++row_index) {
    for (int channel = 0; channel < depth; ++channel) {
      const int input_index = (row_index * depth) + channel;
      const T2 input_value =
          RequantizeInNewRange<T1, T2>(input_flat(input_index), input_min,
                                       input_max, *output_min, *output_max);
      const T2 scale_value = scale_flat(channel);
      const T2 offset_value = offset_flat(channel);
      const T2 output_value =
          ((input_value * scale_value) / one_in_output_space) + offset_value;
      output_flat(input_index) = output_value;
    }
  }
}

}  // namespace

template <typename T1, typename T2>
class QuantizedBatchNormOp : public OpKernel {
 public:
  explicit QuantizedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("variance_epsilon", &variance_epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("scale_after_normalization",
                                             &scale_after_normalization_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const float input_min = context->input(1).flat<float>()(0);
    const float input_max = context->input(2).flat<float>()(0);
    const Tensor& mean = context->input(3);
    const float mean_min = context->input(4).flat<float>()(0);
    const float mean_max = context->input(5).flat<float>()(0);
    const Tensor& var = context->input(6);
    const float var_min = context->input(7).flat<float>()(0);
    const float var_max = context->input(8).flat<float>()(0);
    const Tensor& beta = context->input(9);
    const float beta_min = context->input(10).flat<float>()(0);
    const float beta_max = context->input(11).flat<float>()(0);
    const Tensor& gamma = context->input(12);
    const float gamma_min = context->input(13).flat<float>()(0);
    const float gamma_max = context->input(14).flat<float>()(0);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, mean.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        mean.shape().DebugString()));
    OP_REQUIRES(context, var.dims() == 1,
                errors::InvalidArgument("var must be 1-dimensional",
                                        var.shape().DebugString()));
    OP_REQUIRES(context, beta.dims() == 1,
                errors::InvalidArgument("beta must be 1-dimensional",
                                        beta.shape().DebugString()));
    OP_REQUIRES(context, gamma.dims() == 1,
                errors::InvalidArgument("gamma must be 1-dimensional",
                                        gamma.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    float output_min;
    float output_max;
    FixedPointBatchNorm<T1, T2>(input, input_min, input_max, mean, mean_min,
                                mean_max, var, var_min, var_max, beta, beta_min,
                                beta_max, gamma, gamma_min, gamma_max,
                                variance_epsilon_, scale_after_normalization_,
                                output, &output_min, &output_max);

    Tensor* output_min_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {}, &output_min_tensor));
    output_min_tensor->flat<float>()(0) = output_min;

    Tensor* output_max_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, {}, &output_max_tensor));
    output_max_tensor->flat<float>()(0) = output_max;
  }

 private:
  float variance_epsilon_;
  bool scale_after_normalization_;
};

REGISTER_KERNEL_BUILDER(Name("QuantizedBatchNormWithGlobalNormalization")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint32>("out_type"),
                        QuantizedBatchNormOp<quint8, qint32>);

}  // namespace tensorflow
