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
#ifndef TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_MATH_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_MATH_UTILS_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

namespace internal {

// Multiply by the effective quantized multiplier and shift.
// Caller is responsible for guaranteeing:
// quantized_multiplier >= 0
// shift >= -31 && shift <= 30
// The usage of this function is restricted to "multiply by quantized_multiplier
// and shift which were calcluated from QuantizeMultiplier() function below",
// so the conditions are expected to be met.
//
// Reference (TFLite MultiplyByQuantizedMultiplier with TFLITE_SINGLE_ROUNDING):
// https://github.com/tensorflow/tensorflow/blob/47c640a961874f644cd071752835c7b792450bb8/tensorflow/lite/kernels/internal/common.h#L145
// Above implementation refers from ruy MultiplyByQuantizedMultiplier
// (https://github.com/google/ruy/blob/97ebb72aa0655c0af98896b317476a5d0dacad9c/ruy/apply_multiplier.cc)
//
// After mutiplying fixed point quantized_multiplier, apply single rounding
// operation (addition of 'round' to result and then shift right by
// total_shift). where round=(1 << (30 - shift)) and total_shift=(31 - shift)
inline int32_t MultiplyByQuantizedMultiplier(int32_t x,
                                             int32_t quantized_multiplier,
                                             int shift) {
  const int64_t total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  int64_t result = x * static_cast<int64_t>(quantized_multiplier) + round;
  result = result >> total_shift;

  result = std::clamp(
      result, static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  return static_cast<int32_t>(result);
}

}  // namespace internal

// Quantize eigen Tensor input_tensor using given inv_scale and zero_point,
// using the formula:
// quantized_val = floor(input_val * inv_scale + 0.5f) + zero_point
//
// The caller is reponsible for the validity of the inv_scale (Avoid precision
// loss from taking inverse, and ensure that inv_scale is a finite number.)
template <typename ConstTensorTin, typename TensorTout>
void AffineQuantize(const ConstTensorTin& input_tensor, float inv_scale,
                    int32_t zero_point, int32_t quantization_min_val,
                    int32_t quantization_max_val, TensorTout quantized_tensor) {
  quantized_tensor = ((input_tensor.template cast<float>() * inv_scale + 0.5f)
                          .floor()
                          .template cast<int32_t>() +
                      zero_point)
                         .cwiseMin(quantization_max_val)
                         .cwiseMax(quantization_min_val)
                         .template cast<typename TensorTout::Scalar>();
}

// Dequantize eigen Tensor input_tensor using given scale and zero_point, using
// the formula:
// dequantized_val = (input_val - zero_point) * scale
template <typename ConstTensorTin, typename TensorTout>
void AffineDequantize(const ConstTensorTin& input_tensor, float scale,
                      int32_t zero_point, TensorTout dequantized_tensor) {
  dequantized_tensor = (((input_tensor.template cast<int32_t>() - zero_point))
                            .template cast<float>() *
                        scale)
                           .template cast<typename TensorTout::Scalar>();
}

// Given a portion of input float tensor, quantizes the data and writes output
// to the corresponding portion in quantized_tensor. The quantization scale and
// zero_point is calculated using the input data min and max.
// This function is used for dynamic range quantization in hybrid (float x qint)
// kernels.
//
// This function behavior aligns with TFLite AsymmetricQuantize()
// (https://github.com/tensorflow/tensorflow/blob/779d3824c8b38a622773940011ced0388697b951/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc#L72)
// to achieve feature parity with TFLite which is required since supporting
// mobile executions is the one of the major use cases. The behavior is same
// except for following difference: TFLite AsymmetricQuantize() uses round(input
// / scale + zero_point), while AffineQuantize() uses floor(input_val *
// (1./scale) + 0.5) + zero_point
template <typename ConstTensorTin, typename TensorTout>
absl::Status AsymmetricQuantize(const ConstTensorTin& input_tensor,
                                int32_t quantization_min_val,
                                int32_t quantization_max_val, float& scale,
                                int32& zero_point,
                                TensorTout quantized_tensor) {
  if (quantization_min_val >= quantization_max_val) {
    // NOLINTNEXTLINE
    return errors::InvalidArgument(
        "quantization_min_val must be smaller than quantization_max_val. "
        "Given ",
        quantization_min_val, ", ", quantization_max_val);
  }

  Eigen::Tensor<float, 0, Eigen::RowMajor> input_tensor_min =
      input_tensor.minimum();
  Eigen::Tensor<float, 0, Eigen::RowMajor> input_tensor_max =
      input_tensor.maximum();
  const double rmin = static_cast<double>(std::min(0.0f, input_tensor_min()));
  const double rmax = static_cast<double>(std::max(0.0f, input_tensor_max()));
  const double qmin_double = quantization_min_val;
  const double qmax_double = quantization_max_val;

  float inv_scale = 0;
  scale = (rmax - rmin) / (qmax_double - qmin_double);
  if (rmax - rmin != 0) {
    // Re-calculate the inverse instead of using (1./scale), to avoid loss of
    // precision.
    inv_scale = (qmax_double - qmin_double) / (rmax - rmin);
  }
  if (scale == 0 || !std::isfinite(inv_scale)) {
    quantized_tensor.setZero();
    scale = 1.0;
    zero_point = 0;
    return absl::OkStatus();
  }

  // Using the scale calculated from the quantization range and data range,
  // calculate zero point from quantization min and quantization max.
  // Among those two, choose the zero point that has smaller error.
  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);
  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  int8_t nudged_zero_point = 0;
  if (zero_point_double <= qmin_double) {
    nudged_zero_point = quantization_min_val;
  } else if (zero_point_double >= qmax_double) {
    nudged_zero_point = quantization_max_val;
  } else {
    nudged_zero_point = static_cast<int8_t>(round(zero_point_double));
  }
  zero_point = nudged_zero_point;

  AffineQuantize(input_tensor, inv_scale, zero_point, quantization_min_val,
                 quantization_max_val, quantized_tensor);
  return absl::OkStatus();
}

// Given double_multiplier, quantize it where it is represented by two int32_t,
// quantized_multiplier and shift.
//
// double_multiplier must be a positive finite number. Otherwise returns
// InvalidArgument.
//
// Output quantized_multiplier is clamped to range [0, INT32_MAX],
// and shift is clamped to range [-31, 30].
absl::Status QuantizeMultiplier(double double_multiplier,
                                int32_t& quantized_multiplier, int32_t& shift);

// Requantize input_val given quantized effective_muliplier|shift and
// input|output zero_point.
// Effective multiplier and shift should be calculated from effective scale
// which is:
// (product of input scales) / (product of output scales).
template <typename Tin, typename Tout>
Tout AffineRequantizeWithQuantizedMultiplierAndShift(
    Tin input_val, int32_t effective_quantized_multiplier, int effective_shift,
    int32_t input_zero_point, int32_t output_zero_point,
    int32_t quantization_min_val, int32_t quantization_max_val) {
  const int32_t input = static_cast<int32_t>(input_val) - input_zero_point;

  const int32_t unclamped =
      internal::MultiplyByQuantizedMultiplier(
          input, effective_quantized_multiplier, effective_shift) +
      output_zero_point;

  // Clamp with [quantization_min_val, quantization_max_val].
  return static_cast<Tout>(
      std::max<int32_t>(std::min<int32_t>(unclamped, quantization_max_val),
                        quantization_min_val));
}

namespace internal {

// Requantize from per-tensor to per-tensor.
template <typename Tin, typename Tout>
absl::Status PerTensorToPerTensorRequantize(
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
  return absl::OkStatus();
}

// Requantize where the input or output contains any per-axis quantized cases.
// - From per-tensor to per-axis.
// - From per-axis to per-tensor.
// - From per-axis to per-axis.
template <typename Tin, typename Tout>
absl::Status PerAxisRequantize(OpKernelContext* context, const Tensor& input,
                               const Tensor& input_scales,
                               const Tensor& input_zero_points,
                               const Tensor& output_scales,
                               const Tensor& output_zero_points,
                               int quantization_axis,
                               int32_t quantization_min_val,
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
  return absl::OkStatus();
}

}  // namespace internal

template <typename Tin, typename Tout>
absl::Status EvalRequantize(
    OpKernelContext* context, const Tensor& input, const Tensor& input_scales,
    const Tensor& input_zero_points, const Tensor& output_scales,
    const Tensor& output_zero_points, int input_quantization_axis,
    int output_quantization_axis, int32_t quantization_min_val,
    int32_t quantization_max_val, Tensor& output) {
  if (input_quantization_axis == -1 && output_quantization_axis == -1) {
    return internal::PerTensorToPerTensorRequantize<Tin, Tout>(
        input, input_scales.scalar<float>()(),
        input_zero_points.scalar<int32>()(), output_scales.scalar<float>()(),
        output_zero_points.scalar<int32>()(), quantization_min_val,
        quantization_max_val, output);
  } else {
    const int quantization_axis = input_quantization_axis >= 0
                                      ? input_quantization_axis
                                      : output_quantization_axis;
    return internal::PerAxisRequantize<Tin, Tout>(
        context, input, input_scales, input_zero_points, output_scales,
        output_zero_points, quantization_axis, quantization_min_val,
        quantization_max_val, output);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_MATH_UTILS_H_
