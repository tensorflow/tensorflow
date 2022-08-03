/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

template <typename Tin, typename Tout>
void EvalPerTensorDequantize(const Tensor& input, float scale,
                             int32_t zero_point, Tensor& output) {
  DCHECK(input.IsSameSize(output));
  AffineDequantize(input.flat<Tin>(), scale, zero_point, output.flat<Tout>());
}

template <typename Tin, typename Tout>
void EvalPerChannelDequantize(const Tensor& input, const Tensor& scales,
                              const Tensor& zero_points, int quantization_axis,
                              Tensor& output) {
  DCHECK(input.IsSameSize(output));
  const float* scales_data = scales.flat<float>().data();
  const int32_t* zero_points_data = zero_points.flat<int32_t>().data();
  const int64_t quantization_dim_size = output.dim_size(quantization_axis);

  auto input_tensor =
      input.template flat_inner_outer_dims<Tin, 3>(quantization_axis - 1);

  int64_t pre_dim_size = 1;
  for (int i = 0; i < quantization_axis; ++i) {
    pre_dim_size *= output.dim_size(i);
  }
  int64_t post_dim_size = 1;
  for (int i = quantization_axis + 1; i < output.dims(); ++i) {
    post_dim_size *= output.dim_size(i);
  }
  auto output_tensor = output.template bit_casted_shaped<Tout, 3>(
      {pre_dim_size, quantization_dim_size, post_dim_size});

  for (int i = 0; i < quantization_dim_size; ++i) {
    AffineDequantize(input_tensor.template chip<1>(i), scales_data[i],
                     zero_points_data[i], output_tensor.template chip<1>(i));
  }
}

template <typename Tin, typename Tout>
void EvalDequantize(const Tensor& input, const Tensor& scales,
                    const Tensor& zero_points, int quantization_axis,
                    Tensor& output) {
  if (quantization_axis >= 0) {
    EvalPerChannelDequantize<Tin, Tout>(input, scales, zero_points,
                                        quantization_axis, output);
  } else {
    EvalPerTensorDequantize<Tin, Tout>(input, scales.scalar<float>()(),
                                       zero_points.scalar<int32>()(), output);
  }
}

}  // namespace

template <typename Tin, typename Tout>
class UniformDequantizeOp : public OpKernel {
 public:
  explicit UniformDequantizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("quantization_axis", &quantization_axis_));

    OP_REQUIRES(context,
                (std::is_same<Tin, qint8>() || std::is_same<Tin, qint32>()),
                InvalidArgument("Unsupported input type."));
    OP_REQUIRES(context, (std::is_same<Tout, float>()),
                InvalidArgument("Unsupported output type."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& scales = context->input(1);
    const Tensor& zero_points = context->input(2);

    OP_REQUIRES_OK(context, QuantizationAxisAndShapeValid(
                                input.shape(), scales.shape(),
                                zero_points.shape(), quantization_axis_));
    OP_REQUIRES(context, AllElementsPositive<float>(scales),
                InvalidArgument("rhs scales elements must be all positive."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    EvalDequantize<Tin, Tout>(input, scales, zero_points, quantization_axis_,
                              *output);
  }

 private:
  int quantization_axis_;
};

REGISTER_KERNEL_BUILDER(Name("UniformDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tin")
                            .TypeConstraint<float>("Tout"),
                        UniformDequantizeOp<qint8, float>);

REGISTER_KERNEL_BUILDER(Name("UniformDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tin")
                            .TypeConstraint<float>("Tout"),
                        UniformDequantizeOp<qint32, float>);

}  // namespace tensorflow
