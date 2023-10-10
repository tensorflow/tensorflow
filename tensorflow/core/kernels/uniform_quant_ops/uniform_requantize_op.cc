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
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

using tensorflow::errors::InvalidArgument;

// Changing input_quantization_min/max_val is no-op for this kernel.
template <typename Tin, typename Tout>
class UniformRequantizeOp : public OpKernel {
 public:
  explicit UniformRequantizeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context,
                (std::is_same<Tin, qint32>() || std::is_same<Tin, qint8>()),
                InvalidArgument("Unsupported input type."));
    OP_REQUIRES(context, (std::is_same<Tout, qint8>()),
                InvalidArgument("Unsupported output type."));

    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_min_val",
                                             &output_quantization_min_val_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_max_val",
                                             &output_quantization_max_val_));

    OP_REQUIRES_OK(context, context->GetAttr("input_quantization_axis",
                                             &input_quantization_axis_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_axis",
                                             &output_quantization_axis_));
    OP_REQUIRES(
        context, (input_quantization_axis_ >= -1),
        InvalidArgument("input_quantization_axis must be >= -1, given: ",
                        input_quantization_axis_));
    OP_REQUIRES(
        context, (output_quantization_axis_ >= -1),
        InvalidArgument("output_quantization_axis must be >= -1, given: ",
                        output_quantization_axis_));
    OP_REQUIRES(
        context,
        (!(input_quantization_axis_ >= 0 && output_quantization_axis_ >= 0) ||
         input_quantization_axis_ == output_quantization_axis_),
        InvalidArgument("If input and output is both per-axis quantized, the "
                        "quantization axis must be same."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& input_scales = context->input(1);
    const Tensor& input_zero_points = context->input(2);
    const Tensor& output_scales = context->input(3);
    const Tensor& output_zero_points = context->input(4);

    OP_REQUIRES_OK(context,
                   (QuantizationAxisAndShapeValid(
                       input.shape(), input_scales.shape(),
                       input_zero_points.shape(), input_quantization_axis_)));
    OP_REQUIRES_OK(context,
                   (QuantizationAxisAndShapeValid(
                       input.shape(), output_scales.shape(),
                       output_zero_points.shape(), output_quantization_axis_)));

    OP_REQUIRES(
        context,
        (AllElementsPositive<float>(input_scales) &&
         AllElementsPositive<float>(output_scales)),
        InvalidArgument("input/output scales elements must be all positive."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    OP_REQUIRES_OK(
        context,
        EvalRequantize<Tin, Tout>(
            context, input, input_scales, input_zero_points, output_scales,
            output_zero_points, input_quantization_axis_,
            output_quantization_axis_, output_quantization_min_val_,
            output_quantization_max_val_, *output));
  }

 private:
  int input_quantization_axis_;
  int32_t input_quantization_min_val_;
  int32_t input_quantization_max_val_;
  int output_quantization_axis_;
  int32_t output_quantization_min_val_;
  int32_t output_quantization_max_val_;
};

REGISTER_KERNEL_BUILDER(Name("UniformRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tin")
                            .TypeConstraint<qint8>("Tout"),
                        UniformRequantizeOp<qint8, qint8>);

REGISTER_KERNEL_BUILDER(Name("UniformRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tin")
                            .TypeConstraint<qint8>("Tout"),
                        UniformRequantizeOp<qint32, qint8>);

}  // namespace tensorflow
