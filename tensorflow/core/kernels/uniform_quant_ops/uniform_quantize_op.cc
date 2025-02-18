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
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

template <typename Tin, typename Tout>
void EvalPerTensorQuantize(const Tensor& input, float scale, int32_t zero_point,
                           int32_t quantization_min_val,
                           int32_t quantization_max_val, Tensor& output) {
  const float inv_scale = 1.0f / scale;
  AffineQuantize(input.flat<Tin>(), inv_scale, zero_point, quantization_min_val,
                 quantization_max_val, output.flat<Tout>());
}

template <typename Tin, typename Tout>
void EvalPerChannelQuantize(const Tensor& input, const Tensor& scales,
                            const Tensor& zero_points, int quantization_axis,
                            int32_t quantization_min_val,
                            int32_t quantization_max_val, Tensor& output) {
  DCHECK(input.IsSameSize(output));
  const float* scales_data = scales.flat<float>().data();
  const int32_t* zero_points_data = zero_points.flat<int32_t>().data();

  auto input_tensor =
      input.template flat_inner_outer_dims<Tin, 3>(quantization_axis - 1);
  auto output_tensor =
      output.template flat_inner_outer_dims<Tout, 3>(quantization_axis - 1);

  for (int i = 0; i < output.dim_size(quantization_axis); ++i) {
    const float inv_scale = 1.0f / scales_data[i];
    AffineQuantize(input_tensor.template chip<1>(i), inv_scale,
                   zero_points_data[i], quantization_min_val,
                   quantization_max_val, output_tensor.template chip<1>(i));
  }
}

template <typename Tin, typename Tout>
void EvalQuantize(const Tensor& input, const Tensor& scales,
                  const Tensor& zero_points, int quantization_axis,
                  int32_t quantization_min_val, int32_t quantization_max_val,
                  Tensor& output) {
  if (quantization_axis >= 0) {
    EvalPerChannelQuantize<Tin, Tout>(input, scales, zero_points,
                                      quantization_axis, quantization_min_val,
                                      quantization_max_val, output);
  } else {
    EvalPerTensorQuantize<Tin, Tout>(
        input, scales.scalar<float>()(), zero_points.scalar<int32>()(),
        quantization_min_val, quantization_max_val, output);
  }
}

}  // namespace

class UniformQuantizeOp : public OpKernel {
 public:
  explicit UniformQuantizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Tin", &tin_));
    OP_REQUIRES(context, tin_ == DataType::DT_FLOAT,
                InvalidArgument("Unsupported input type."));
    OP_REQUIRES_OK(context, context->GetAttr("Tout", &tout_));
    OP_REQUIRES(context,
                tout_ == DataType::DT_QINT8 || tout_ == DataType::DT_QUINT8 ||
                    tout_ == DataType::DT_QINT32,
                InvalidArgument("Unsupported output type."));

    OP_REQUIRES_OK(context, context->GetAttr("quantization_min_val",
                                             &quantization_min_val_));
    OP_REQUIRES_OK(context, context->GetAttr("quantization_max_val",
                                             &quantization_max_val_));

    OP_REQUIRES_OK(context,
                   context->GetAttr("quantization_axis", &quantization_axis_));
    OP_REQUIRES(context, (quantization_axis_ >= -1),
                InvalidArgument("quantization_axis must be >= -1, given: ",
                                quantization_axis_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& scales = context->input(1);
    const Tensor& zero_points = context->input(2);

    OP_REQUIRES_OK(context, (QuantizationAxisAndShapeValid(
                                input.shape(), scales.shape(),
                                zero_points.shape(), quantization_axis_)));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (tout_ == DataType::DT_QINT8) {
      EvalQuantize<float, qint8>(input, scales, zero_points, quantization_axis_,
                                 quantization_min_val_, quantization_max_val_,
                                 *output);
    } else if (tout_ == DataType::DT_QUINT8) {
      EvalQuantize<float, quint8>(input, scales, zero_points,
                                  quantization_axis_, quantization_min_val_,
                                  quantization_max_val_, *output);
    } else {
      EvalQuantize<float, qint32>(input, scales, zero_points,
                                  quantization_axis_, quantization_min_val_,
                                  quantization_max_val_, *output);
    }
  }

 private:
  DataType tin_, tout_;
  int quantization_axis_;
  int32_t quantization_min_val_;
  int32_t quantization_max_val_;
};

REGISTER_KERNEL_BUILDER(Name("UniformQuantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Tin")
                            .TypeConstraint("Tout",
                                            {DT_QINT8, DT_QUINT8, DT_QINT32}),
                        UniformQuantizeOp);

}  // namespace tensorflow
