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
namespace {

using tensorflow::errors::InvalidArgument;

// Requantize from per-tensor to per-tensor.
template <typename Tin, typename Tout>
Status PerTensorToPerTensorRequantize(
    const Tensor& input, float input_scale, int32_t input_zero_point,
    float output_scale, int32_t output_zero_point, int32_t quantization_min_val,
    int32_t quantization_max_val, Tensor& output) {
  const double effective_multiplier =
      static_cast<double>(input_scale) / output_scale;
  int32_t effective_quantized_multiplier;
  int32_t effective_shift;
  TF_RETURN_IF_ERROR(QuantizeMultiplier(
      effective_multiplier, effective_quantized_multiplier, effective_shift));

  output.flat<Tout>() = input.flat<Tin>().unaryExpr(
      [effective_quantized_multiplier, effective_shift, input_zero_point,
       output_zero_point, quantization_min_val,
       quantization_max_val](Tin input_val) {
        return AffineRequantizeWithQuantizedMultiplierAndShift<Tin, Tout>(
            input_val, effective_quantized_multiplier, effective_shift,
            input_zero_point, output_zero_point, quantization_min_val,
            quantization_max_val);
      });
  return OkStatus();
}

// Requantize where the input or output contains any per-axis quantized cases.
// - From per-tensor to per-axis.
// - From per-axis to per-tensor.
// - From per-axis to per-axis.
template <typename Tin, typename Tout>
Status PerAxisRequantize(OpKernelContext* context, const Tensor& input,
                         const Tensor& input_scales,
                         const Tensor& input_zero_points,
                         const Tensor& output_scales,
                         const Tensor& output_zero_points,
                         int quantization_axis, int32_t quantization_min_val,
                         int32_t quantization_max_val, Tensor& output) {
  const bool input_per_axis_quantization = input_scales.dims() == 1;
  const bool output_per_axis_quantization = output_scales.dims() == 1;
  const auto& per_axis_scales_shape = input_per_axis_quantization
                                          ? input_scales.shape()
                                          : output_scales.shape();

  Tensor effective_quantized_multipliers;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, per_axis_scales_shape,
                                            &effective_quantized_multipliers));
  Tensor effective_shifts;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, per_axis_scales_shape,
                                            &effective_shifts));

  const float* input_scales_data = input_scales.flat<float>().data();
  const float* output_scales_data = output_scales.flat<float>().data();
  int32_t* effective_quantized_multipliers_data =
      effective_quantized_multipliers.flat<int32_t>().data();
  int32_t* effective_shifts_data = effective_shifts.flat<int32_t>().data();

  const int64_t quantization_dim_size = output.dim_size(quantization_axis);

  for (int64_t i = 0; i < quantization_dim_size; ++i) {
    const double effective_multiplier =
        static_cast<double>(
            input_scales_data[input_per_axis_quantization ? i : 0]) /
        output_scales_data[output_per_axis_quantization ? i : 0];
    TF_RETURN_IF_ERROR(QuantizeMultiplier(
        effective_multiplier, effective_quantized_multipliers_data[i],
        effective_shifts_data[i]));
  }

  const int32* input_zero_points_data = input_zero_points.flat<int32>().data();
  const int32* output_zero_points_data =
      output_zero_points.flat<int32>().data();

  auto input_tensor =
      input.template flat_inner_outer_dims<Tin, 3>(quantization_axis - 1);
  auto output_tensor =
      output.template flat_inner_outer_dims<Tout, 3>(quantization_axis - 1);

  for (int i = 0; i < quantization_dim_size; ++i) {
    output_tensor.template chip<1>(i) =
        input_tensor.template chip<1>(i).unaryExpr(
            [effective_quantized_multipliers_data, effective_shifts_data,
             input_zero_points_data, output_zero_points_data,
             quantization_min_val, quantization_max_val,
             input_per_axis_quantization, output_per_axis_quantization,
             i](Tin input_val) {
              return AffineRequantizeWithQuantizedMultiplierAndShift<Tin, Tout>(
                  input_val, effective_quantized_multipliers_data[i],
                  effective_shifts_data[i],
                  input_zero_points_data[input_per_axis_quantization ? i : 0],
                  output_zero_points_data[output_per_axis_quantization ? i : 0],
                  quantization_min_val, quantization_max_val);
            });
  }
  return OkStatus();
}

template <typename Tin, typename Tout>
Status EvalRequantize(OpKernelContext* context, const Tensor& input,
                      const Tensor& input_scales,
                      const Tensor& input_zero_points,
                      const Tensor& output_scales,
                      const Tensor& output_zero_points,
                      int input_quantization_axis, int output_quantization_axis,
                      int32_t quantization_min_val,
                      int32_t quantization_max_val, Tensor& output) {
  if (input_quantization_axis == -1 && output_quantization_axis == -1) {
    return PerTensorToPerTensorRequantize<Tin, Tout>(
        input, input_scales.scalar<float>()(),
        input_zero_points.scalar<int32>()(), output_scales.scalar<float>()(),
        output_zero_points.scalar<int32>()(), quantization_min_val,
        quantization_max_val, output);
  } else {
    const int quantization_axis = input_quantization_axis >= 0
                                      ? input_quantization_axis
                                      : output_quantization_axis;
    return PerAxisRequantize<Tin, Tout>(
        context, input, input_scales, input_zero_points, output_scales,
        output_zero_points, quantization_axis, quantization_min_val,
        quantization_max_val, output);
  }
}

}  // namespace

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
