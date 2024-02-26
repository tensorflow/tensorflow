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

#include <algorithm>
#include <limits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_params.h"

namespace tensorflow {
namespace {

using tensorflow::errors::InvalidArgument;

// Given dimension numbers and total dims of lhs, returns the permutation of lhs
// dimension indices that reorders lhs to the order:
// (batch, feature, spatial_0, ..., spatial_dims-3).
std::vector<int32_t> LhsTransposePerm(
    const UniformQuantizedConvolutionDimensionNumbersAttr& dimension_numbers,
    const int dims) {
  // dims > 2 is guaranteed by
  // ConvolutionParams::ValidateOrFillParamsAndValidateShape().
  std::vector<int32_t> lhs_perm(dims);
  lhs_perm[0] = dimension_numbers.input_batch_dimension();
  lhs_perm[1] = dimension_numbers.input_feature_dimension();
  std::copy(dimension_numbers.input_spatial_dimensions().begin(),
            dimension_numbers.input_spatial_dimensions().end(),
            lhs_perm.begin() + 2);
  return lhs_perm;
}

// Given dimension numbers and total dims of rhs, returns the permutation of rhs
// dimension indices that reorders rhs to the order:
// (output_feature, input_feature, spatial_0, ..., spatial_dims-3).
std::vector<int32_t> RhsTransposePerm(
    const UniformQuantizedConvolutionDimensionNumbersAttr& dimension_numbers,
    const int dims) {
  // dims > 2 is guaranteed by
  // ConvolutionParams::ValidateOrFillParamsAndValidateShape().
  std::vector<int32_t> rhs_perm(dims);
  rhs_perm[0] = dimension_numbers.kernel_output_feature_dimension();
  rhs_perm[1] = dimension_numbers.kernel_input_feature_dimension();
  std::copy(dimension_numbers.kernel_spatial_dimensions().begin(),
            dimension_numbers.kernel_spatial_dimensions().end(),
            rhs_perm.begin() + 2);
  return rhs_perm;
}

// Given dimension numbers and total dims of output, returns the permutation of
// output dimension indices that reorders output to order:
// (batch, feature, spatial_0, ..., spatial_dims-3).
std::vector<int32_t> OutTransposePerm(
    const UniformQuantizedConvolutionDimensionNumbersAttr& dimension_numbers,
    const int dims) {
  // dims > 2 is guaranteed by
  // ConvolutionParams::ValidateOrFillParamsAndValidateShape().
  std::vector<int32_t> out_perm(dims);
  out_perm[0] = dimension_numbers.output_batch_dimension();
  out_perm[1] = dimension_numbers.output_feature_dimension();
  std::copy(dimension_numbers.output_spatial_dimensions().begin(),
            dimension_numbers.output_spatial_dimensions().end(),
            out_perm.begin() + 2);
  return out_perm;
}

// Given the permutation returned from OutTransposePerm, returns the permutation
// that reorders "transposed output" to the "original ordered output".
std::vector<int32_t> OutBackTransposePerm(absl::Span<const int32_t> out_perm) {
  std::vector<int32_t> out_perm_back(out_perm.size());
  for (int i = 0; i < out_perm.size(); ++i) {
    out_perm_back[out_perm[i]] = i;
  }
  return out_perm_back;
}

// Below *TransposedLhs*() functions work on transposed lhs by LhsTransposePerm,
// transposed rhs by RhsTransposePerm, and transposed out by OutTransposePerm.
// Since they are transposed to [batch_dim][feature_dim][spatial_dims ... ]
// order,
// their ith dim corresponds to window_strides[i - 2], lhs_dilation[i - 2],
// rhs_dilation[i - 2],
// and {padding_list[2 * (i - 2)], padding_list[2 * (i - 2) + 1]}.

// Given in_shape of transposed lhs by LhsTransposePerm, and convolution_params,
// returns padded and dilated out_shape of lhs.
TensorShape PaddedAndDilatedTransposedLhsShape(
    const TensorShape& in_shape,
    const UniformQuantizedConvolutionParams& convolution_params) {
  TensorShape out_shape = in_shape;
  for (int i = 2; i < in_shape.dims(); ++i) {
    // lhs dim (i) -> use (i - 2) for lhs_dilation and padding_list.
    const int64_t lhs_size_dilated =
        UniformQuantizedConvolutionParams::DilatedSize(
            in_shape.dim_size(i), convolution_params.lhs_dilation()[i - 2]);
    const int64_t out_lhs_size =
        lhs_size_dilated + convolution_params.padding_list()[2 * (i - 2)] +
        convolution_params.padding_list()[2 * (i - 2) + 1];
    out_shape.set_dim(i, out_lhs_size);
  }

  return out_shape;
}

// Given in_spatial_idx of transposed lhs, returns the corresponding spatial_idx
// of padded and dilated transposed lhs.
int64_t PaddedAndDilatedTransposedLhsSpatialIdx(
    const UniformQuantizedConvolutionParams& convolution_params,
    const TensorShape& lhs_in_shape, const TensorShape& lhs_out_shape,
    int64_t in_spatial_idx) {
  int64_t out_spatial_idx = 0;
  int64_t out_spatial_inner_dim_size = 1;
  for (int dim = lhs_in_shape.dims() - 1; dim >= 2; --dim) {
    // lhs dim (dim) -> use (dim - 2) for lhs_dilation and padding_list.
    const int64_t in_spatial_idx_of_dim =
        in_spatial_idx % lhs_in_shape.dim_size(dim);

    const int64_t out_spatial_idx_of_dim =
        convolution_params.padding_list()[2 * (dim - 2)] +
        convolution_params.lhs_dilation()[dim - 2] * in_spatial_idx_of_dim;
    out_spatial_idx += out_spatial_idx_of_dim * out_spatial_inner_dim_size;

    in_spatial_idx /= lhs_in_shape.dim_size(dim);
    out_spatial_inner_dim_size *= lhs_out_shape.dim_size(dim);
  }
  return out_spatial_idx;
}

// Given rhs_spatial_idx of transposed rhs and out_spatial_idx of transposed
// out, returns corresponding lhs_spatial_idx of padded and dilated transposed
// lhs.
int64_t ConvolutionTransposedLhsSpatialIdx(
    const UniformQuantizedConvolutionParams& convolution_params,
    const TensorShape& lhs_shape, const TensorShape& rhs_shape,
    const TensorShape& out_shape, int64_t rhs_spatial_idx,
    int64_t out_spatial_idx) {
  int64_t lhs_spatial_idx = 0;
  int64_t lhs_spatial_inner_dim_size = 1;
  for (int dim = lhs_shape.dims() - 1; dim >= 2; --dim) {
    // (dim) -> use (dim - 2) for window_strides and rhs_dilation.
    const int64_t rhs_spatial_idx_of_dim =
        rhs_spatial_idx % rhs_shape.dim_size(dim);
    const int64_t out_spatial_idx_of_dim =
        out_spatial_idx % out_shape.dim_size(dim);

    const int64_t lhs_spatial_idx_of_dim =
        out_spatial_idx_of_dim * convolution_params.window_strides()[dim - 2] +
        rhs_spatial_idx_of_dim * convolution_params.rhs_dilation()[dim - 2];
    lhs_spatial_idx += lhs_spatial_idx_of_dim * lhs_spatial_inner_dim_size;

    rhs_spatial_idx /= rhs_shape.dim_size(dim);
    out_spatial_idx /= out_shape.dim_size(dim);
    lhs_spatial_inner_dim_size *= lhs_shape.dim_size(dim);
  }
  return lhs_spatial_idx;
}

// Given transposed lhs, dilate using lhs_dilation and pad with lhs_zero_points,
// on the assumption that lhs_out is already allocated with a correct shape.
template <typename Tlhs>
void PadAndDilateTransposedLhs(
    const Tensor& lhs_in,
    const UniformQuantizedConvolutionParams& convolution_params,
    const Tensor& lhs_zero_points, Tensor& lhs_out) {
  auto lhs_in_tensor = lhs_in.flat_outer_dims<Tlhs, 3>();
  auto lhs_out_tensor = lhs_out.flat_outer_dims<Tlhs, 3>();
  const int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();
  const bool is_lhs_zero_points_scalar = lhs_zero_points.dims() == 0;

  for (int64_t batch_idx = 0; batch_idx < lhs_in.dim_size(0); ++batch_idx) {
    lhs_out_tensor.template chip<0>(batch_idx).setConstant(
        lhs_zero_points_data[is_lhs_zero_points_scalar ? 0 : batch_idx]);
    for (int64_t feature_idx = 0; feature_idx < lhs_in.dim_size(1);
         ++feature_idx) {
      for (int64_t in_spatial_idx = 0;
           in_spatial_idx < lhs_in_tensor.dimension(2); ++in_spatial_idx) {
        const int64_t out_spatial_idx = PaddedAndDilatedTransposedLhsSpatialIdx(
            convolution_params, lhs_in.shape(), lhs_out.shape(),
            in_spatial_idx);
        lhs_out_tensor(batch_idx, feature_idx, out_spatial_idx) =
            lhs_in_tensor(batch_idx, feature_idx, in_spatial_idx);
      }
    }
  }
}

// Quantized Conv on padded and dilated transposed lhs and transposed rhs, given
// acc_f to calculate each value to accumulate, and out_f to calculate output
// value on each accumulated value.
template <typename Tlhs, typename Trhs, typename Tout, typename AccF,
          typename OutF>
void ConvWithAccFunctionAndOutFunction(
    const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params, Tensor& out,
    const AccF& acc_f, const OutF& out_f) {
  const int64_t out_feature_group_size_by_feature_group_count =
      out.dim_size(1) / convolution_params.feature_group_count();
  const int64_t out_feature_group_size_by_batch_group_count =
      out.dim_size(1) / convolution_params.batch_group_count();

  auto lhs_tensor = lhs.flat_outer_dims<Tlhs, 3>();
  auto rhs_tensor = rhs.flat_outer_dims<Trhs, 3>();
  auto out_tensor = out.flat_outer_dims<Tout, 3>();

  // Iter out batch.
  for (int64_t out_batch_idx = 0; out_batch_idx < out_tensor.dimension(0);
       ++out_batch_idx) {
    // Iter out feature.
    for (int64_t out_feature_idx = 0; out_feature_idx < out_tensor.dimension(1);
         ++out_feature_idx) {
      const int64_t lhs_batch_idx =
          (out_feature_idx / out_feature_group_size_by_batch_group_count) *
              out_tensor.dimension(0) +
          out_batch_idx;

      // Iter out spatial.
      for (int out_spatial_idx = 0; out_spatial_idx < out_tensor.dimension(2);
           ++out_spatial_idx) {
        int32_t acc = 0;

        // Iter rhs input feature.
        for (int64_t rhs_in_feature_idx = 0;
             rhs_in_feature_idx < rhs_tensor.dimension(1);
             ++rhs_in_feature_idx) {
          const int64_t lhs_feature_idx =
              (out_feature_idx /
               out_feature_group_size_by_feature_group_count) *
                  rhs_tensor.dimension(1) +
              rhs_in_feature_idx;

          // Iter rhs spatial.
          for (int64_t rhs_spatial_idx = 0;
               rhs_spatial_idx < rhs_tensor.dimension(2); ++rhs_spatial_idx) {
            const int64_t lhs_spatial_idx = ConvolutionTransposedLhsSpatialIdx(
                convolution_params, lhs.shape(), rhs.shape(), out.shape(),
                rhs_spatial_idx, out_spatial_idx);
            const Tlhs lhs_val =
                lhs_tensor(lhs_batch_idx, lhs_feature_idx, lhs_spatial_idx);
            const Trhs rhs_val = rhs_tensor(out_feature_idx, rhs_in_feature_idx,
                                            rhs_spatial_idx);
            acc += acc_f(lhs_val, rhs_val, lhs_batch_idx, out_feature_idx);
          }
        }

        out_tensor(out_batch_idx, out_feature_idx, out_spatial_idx) =
            out_f(acc, lhs_batch_idx, out_feature_idx);
      }
    }
  }
}

// Quantized Conv on per-tensor quantized padded and dilated transposed lhs and
// per-tensor quantized transposed rhs.
template <typename Tin, typename Tout>
Status EvalLhsPerTensorAndRhsPerTensorQuantizedConv(
    const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params,
    const float lhs_scale, const int32_t lhs_zero_point, const float rhs_scale,
    const int32_t rhs_zero_point, const float output_scale,
    const int32_t output_zero_point, const int output_quantization_min_val,
    const int output_quantization_max_val, Tensor& out) {
  const double effective_multiplier =
      static_cast<double>(lhs_scale) * rhs_scale / output_scale;
  int32_t effective_quantized_multiplier;
  int effective_shift;
  TF_RETURN_IF_ERROR(QuantizeMultiplier(
      effective_multiplier, effective_quantized_multiplier, effective_shift));

  ConvWithAccFunctionAndOutFunction<Tin, Tin, Tout>(
      lhs, rhs, convolution_params, out,
      /*acc_f=*/
      [lhs_zero_point, rhs_zero_point](Tin lhs_val, Tin rhs_val,
                                       int64_t lhs_batch_idx,
                                       int64_t out_feature_idx) {
        return (static_cast<int32_t>(lhs_val) - lhs_zero_point) *
               (static_cast<int32_t>(rhs_val) - rhs_zero_point);
      },
      /*out_f=*/
      [effective_quantized_multiplier, effective_shift, output_zero_point,
       output_quantization_min_val, output_quantization_max_val](
          int32_t acc, int64_t lhs_batch_idx, int64_t out_feature_idx) {
        return AffineRequantizeWithQuantizedMultiplierAndShift<int32_t, Tout>(
            acc, effective_quantized_multiplier, effective_shift,
            /*input_zero_point=*/0, output_zero_point,
            output_quantization_min_val, output_quantization_max_val);
      });
  return absl::OkStatus();
}

// Quantized Conv on per-tensor quantized padded and dilated transposed lhs and
// per-channel quantized transposed rhs.
template <typename Tin, typename Tout>
Status EvalLhsPerTensorAndRhsPerChannelQuantizedConv(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params,
    const float lhs_scale, const int32_t lhs_zero_point,
    const Tensor& rhs_scales, const Tensor& rhs_zero_points,
    const Tensor& output_scales, const Tensor& output_zero_points,
    const int output_quantization_min_val,
    const int output_quantization_max_val, Tensor& out) {
  const int64_t out_feature_size = out.dim_size(1);
  const float* rhs_scales_data = rhs_scales.flat<float>().data();
  const int32_t* rhs_zero_points_data = rhs_zero_points.flat<int32_t>().data();

  Tensor effective_quantized_multipliers;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, rhs_scales.shape(),
                                            &effective_quantized_multipliers));
  Tensor effective_shifts;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DT_INT32, rhs_scales.shape(), &effective_shifts));
  int32_t* effective_quantized_multipliers_data =
      effective_quantized_multipliers.flat<int32_t>().data();
  int32_t* effective_shifts_data = effective_shifts.flat<int32_t>().data();

  const bool is_output_scales_scalar = output_scales.dims() == 0;

  if (!is_output_scales_scalar) {
    const float* output_scales_data = output_scales.flat<float>().data();
    for (int64_t out_feature_idx = 0; out_feature_idx < out_feature_size;
         ++out_feature_idx) {
      const double effective_multiplier = static_cast<double>(lhs_scale) *
                                          rhs_scales_data[out_feature_idx] /
                                          output_scales_data[out_feature_idx];
      TF_RETURN_IF_ERROR(QuantizeMultiplier(
          effective_multiplier,
          effective_quantized_multipliers_data[out_feature_idx],
          effective_shifts_data[out_feature_idx]));
    }
  } else {
    const float output_scale = output_scales.scalar<float>()();
    for (int64_t out_feature_idx = 0; out_feature_idx < out_feature_size;
         ++out_feature_idx) {
      const double effective_multiplier = static_cast<double>(lhs_scale) *
                                          rhs_scales_data[out_feature_idx] /
                                          output_scale;
      TF_RETURN_IF_ERROR(QuantizeMultiplier(
          effective_multiplier,
          effective_quantized_multipliers_data[out_feature_idx],
          effective_shifts_data[out_feature_idx]));
    }
  }

  const int32_t* output_zero_points_data =
      output_zero_points.flat<int32_t>().data();
  ConvWithAccFunctionAndOutFunction<Tin, Tin, Tout>(
      lhs, rhs, convolution_params, out,
      /*acc_f=*/
      [lhs_zero_point, rhs_zero_points_data](Tin lhs_val, Tin rhs_val,
                                             int64_t lhs_batch_idx,
                                             int64_t out_feature_idx) {
        return (static_cast<int32_t>(lhs_val) - lhs_zero_point) *
               (static_cast<int32_t>(rhs_val) -
                rhs_zero_points_data[out_feature_idx]);
      },
      /*out_f=*/
      [effective_quantized_multipliers_data, effective_shifts_data,
       output_zero_points_data, output_quantization_min_val,
       output_quantization_max_val, is_output_scales_scalar](
          int32_t acc, int64_t lhs_batch_idx, int64_t out_feature_idx) {
        return AffineRequantizeWithQuantizedMultiplierAndShift<int32_t, Tout>(
            acc, effective_quantized_multipliers_data[out_feature_idx],
            effective_shifts_data[out_feature_idx],
            /*input_zero_point=*/0,
            output_zero_points_data[is_output_scales_scalar ? 0
                                                            : out_feature_idx],
            output_quantization_min_val, output_quantization_max_val);
      });
  return absl::OkStatus();
}

// Quantized Conv on per-batch quantized padded and dilated transposed lhs and
// per-tensor quantized transposed rhs.
template <typename Tlhs, typename Trhs>
void EvalLhsPerBatchAndRhsPerTensorQuantizedConv(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params,
    const Tensor& lhs_scales, const Tensor& lhs_zero_points,
    const float rhs_scale, const int32_t rhs_zero_point, Tensor& out) {
  const float* lhs_scales_data = lhs_scales.flat<float>().data();
  const int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();

  ConvWithAccFunctionAndOutFunction<Tlhs, Trhs, float>(
      lhs, rhs, convolution_params, out,
      /*acc_f=*/
      [lhs_zero_points_data, rhs_zero_point](Tlhs lhs_val, Trhs rhs_val,
                                             int64_t lhs_batch_idx,
                                             int64_t out_feature_idx) {
        return (static_cast<int32_t>(lhs_val) -
                lhs_zero_points_data[lhs_batch_idx]) *
               (static_cast<int32_t>(rhs_val) - rhs_zero_point);
      },
      /*out_f=*/
      [lhs_scales_data, rhs_scale](int32_t acc, int64_t lhs_batch_idx,
                                   int64_t out_feature_idx) {
        return acc * lhs_scales_data[lhs_batch_idx] * rhs_scale;
      });
}

// Quantized Conv on per-batch quantized padded and dilated transposed lhs and
// per-channel quantized transposed rhs.
template <typename Tlhs, typename Trhs>
void EvalLhsPerBatchAndRhsPerChannelQuantizedConv(
    const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params,
    const Tensor& lhs_scales, const Tensor& lhs_zero_points,
    const Tensor& rhs_scales, const Tensor& rhs_zero_points, Tensor& out) {
  const float* lhs_scales_data = lhs_scales.flat<float>().data();
  const int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();
  const float* rhs_scales_data = rhs_scales.flat<float>().data();
  const int32_t* rhs_zero_points_data = rhs_zero_points.flat<int32_t>().data();

  ConvWithAccFunctionAndOutFunction<Tlhs, Trhs, float>(
      lhs, rhs, convolution_params, out,
      /*acc_f=*/
      [lhs_zero_points_data, rhs_zero_points_data](Tlhs lhs_val, Trhs rhs_val,
                                                   int64_t lhs_batch_idx,
                                                   int64_t out_feature_idx) {
        return (static_cast<int32_t>(lhs_val) -
                lhs_zero_points_data[lhs_batch_idx]) *
               (static_cast<int32_t>(rhs_val) -
                rhs_zero_points_data[out_feature_idx]);
      },
      /*out_f=*/
      [lhs_scales_data, rhs_scales_data](int32_t acc, int64_t lhs_batch_idx,
                                         int64_t out_feature_idx) {
        return acc * lhs_scales_data[lhs_batch_idx] *
               rhs_scales_data[out_feature_idx];
      });
}

// Given quantized `lhs` and quantized `rhs`, performs quantized convolution and
// writes to `out`. Assumes that `out` is already allocated with correct size.
template <typename Tin, typename Tout>
Status EvalQuantizedConv(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params,
    const Tensor& lhs_scales, const Tensor& lhs_zero_points,
    const Tensor& rhs_scales, const Tensor& rhs_zero_points,
    const Tensor& output_scales, const Tensor& output_zero_points,
    int output_quantization_min_val, int output_quantization_max_val,
    Tensor& out) {
  const auto& dimension_numbers = convolution_params.dimension_numbers();
  // Transpose lhs.
  const auto& lhs_perm = LhsTransposePerm(dimension_numbers, lhs.dims());
  Tensor lhs_transposed;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      lhs.dtype(), TransposedShape(lhs.shape(), lhs_perm), &lhs_transposed));
  Transpose<Tin>(lhs, lhs_perm, lhs_transposed);
  // Transpose rhs.
  const auto& rhs_perm = RhsTransposePerm(dimension_numbers, rhs.dims());
  Tensor rhs_transposed;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      rhs.dtype(), TransposedShape(rhs.shape(), rhs_perm), &rhs_transposed));
  Transpose<Tin>(rhs, rhs_perm, rhs_transposed);
  // Allocate tranposed_out.
  const auto& out_perm = OutTransposePerm(dimension_numbers, out.dims());
  Tensor out_transposed;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      out.dtype(), TransposedShape(out.shape(), out_perm), &out_transposed));

  Tensor lhs_padded_and_dilated;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(lhs_transposed.dtype(),
                             PaddedAndDilatedTransposedLhsShape(
                                 lhs_transposed.shape(), convolution_params),
                             &lhs_padded_and_dilated));
  PadAndDilateTransposedLhs<Tin>(lhs_transposed, convolution_params,
                                 lhs_zero_points, lhs_padded_and_dilated);

  const float lhs_scale = lhs_scales.scalar<float>()();
  const int32_t lhs_zero_point = lhs_zero_points.scalar<int32_t>()();
  if (rhs_scales.dims() != 0) {
    TF_RETURN_IF_ERROR(EvalLhsPerTensorAndRhsPerChannelQuantizedConv<Tin, Tout>(
        context, lhs_padded_and_dilated, rhs_transposed, convolution_params,
        lhs_scale, lhs_zero_point, rhs_scales, rhs_zero_points, output_scales,
        output_zero_points, output_quantization_min_val,
        output_quantization_max_val, out_transposed));
  } else {
    DCHECK_EQ(output_scales.dims(), 0);
    const float rhs_scale = rhs_scales.scalar<float>()();
    const int32_t rhs_zero_point = rhs_zero_points.scalar<int32_t>()();
    const float output_scale = output_scales.scalar<float>()();
    const int32_t output_zero_point = output_zero_points.scalar<int32_t>()();
    TF_RETURN_IF_ERROR(EvalLhsPerTensorAndRhsPerTensorQuantizedConv<Tin, Tout>(
        lhs_padded_and_dilated, rhs_transposed, convolution_params, lhs_scale,
        lhs_zero_point, rhs_scale, rhs_zero_point, output_scale,
        output_zero_point, output_quantization_min_val,
        output_quantization_max_val, out_transposed));
  }

  // Transpose transposed_out back to out.
  const auto& out_perm_back = OutBackTransposePerm(out_perm);
  Transpose<Tout>(out_transposed, out_perm_back, out);
  return absl::OkStatus();
}

// Given float `lhs` and quantized `rhs`, performs per-batch dynamic range
// quantization on `lhs`, and then performs quantized convolution on
// quantized_lhs and `rhs`, and writes to `out`. Assumes that `out` is already
// allocated with correct size.
// For more details on `lhs` quantization policy, refer to the comment of class
// UniformQuantizedConvolutionHybridOp below.
template <typename Trhs>
Status EvalHybridConv(
    OpKernelContext* context, const Tensor& lhs, const Tensor& rhs,
    const UniformQuantizedConvolutionParams& convolution_params,
    const Tensor& rhs_scales, const Tensor& rhs_zero_points, Tensor& out) {
  using TlhsQuant = Trhs;
  DataType lhs_quant_dtype = DataTypeToEnum<TlhsQuant>::v();

  const auto& dimension_numbers = convolution_params.dimension_numbers();
  // Transpose lhs.
  const auto& lhs_perm = LhsTransposePerm(dimension_numbers, lhs.dims());
  Tensor lhs_transposed;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_FLOAT, TransposedShape(lhs.shape(), lhs_perm), &lhs_transposed));
  Transpose<float>(lhs, lhs_perm, lhs_transposed);
  // Transpose rhs.
  const auto& rhs_perm = RhsTransposePerm(dimension_numbers, rhs.dims());
  Tensor rhs_transposed;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      rhs.dtype(), TransposedShape(rhs.shape(), rhs_perm), &rhs_transposed));
  Transpose<Trhs>(rhs, rhs_perm, rhs_transposed);
  // Allocate tranposed_out.
  const auto& out_perm = OutTransposePerm(dimension_numbers, out.dims());
  Tensor out_transposed;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_FLOAT, TransposedShape(out.shape(), out_perm), &out_transposed));

  const int64_t lhs_batch_size = lhs_transposed.dim_size(0);
  Tensor lhs_quantized;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      lhs_quant_dtype, lhs_transposed.shape(), &lhs_quantized));
  Tensor lhs_scales;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DT_FLOAT, {lhs_batch_size}, &lhs_scales));
  Tensor lhs_zero_points;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(DT_INT32, {lhs_batch_size}, &lhs_zero_points));
  float* lhs_scales_data = lhs_scales.flat<float>().data();
  int32_t* lhs_zero_points_data = lhs_zero_points.flat<int32_t>().data();

  auto lhs_tensor = lhs_transposed.template flat_outer_dims<float, 2>();
  auto lhs_quantized_tensor =
      lhs_quantized.template flat_outer_dims<TlhsQuant, 2>();
  for (int64_t b = 0; b < lhs_batch_size; ++b) {
    TF_RETURN_IF_ERROR(AsymmetricQuantize(
        lhs_tensor.template chip<0>(b),
        /*quantization_min_val=*/std::numeric_limits<TlhsQuant>::lowest(),
        /*quantization_max_val=*/std::numeric_limits<TlhsQuant>::max(),
        lhs_scales_data[b], lhs_zero_points_data[b],
        lhs_quantized_tensor.template chip<0>(b)));
  }

  Tensor lhs_padded_and_dilated;
  TF_RETURN_IF_ERROR(
      context->allocate_temp(lhs_quant_dtype,
                             PaddedAndDilatedTransposedLhsShape(
                                 lhs_quantized.shape(), convolution_params),
                             &lhs_padded_and_dilated));
  PadAndDilateTransposedLhs<TlhsQuant>(lhs_quantized, convolution_params,
                                       lhs_zero_points, lhs_padded_and_dilated);

  if (rhs_scales.dims() != 0) {
    EvalLhsPerBatchAndRhsPerChannelQuantizedConv<TlhsQuant, Trhs>(
        lhs_padded_and_dilated, rhs_transposed, convolution_params, lhs_scales,
        lhs_zero_points, rhs_scales, rhs_zero_points, out_transposed);
  } else {
    EvalLhsPerBatchAndRhsPerTensorQuantizedConv<TlhsQuant, Trhs>(
        context, lhs_padded_and_dilated, rhs_transposed, convolution_params,
        lhs_scales, lhs_zero_points, rhs_scales.scalar<float>()(),
        rhs_zero_points.scalar<int32_t>()(), out_transposed);
  }

  // Transpose transposed_out back to out.
  const auto& out_perm_back = OutBackTransposePerm(out_perm);
  Transpose<float>(out_transposed, out_perm_back, out);
  return absl::OkStatus();
}

}  // namespace

template <typename Tin, typename Tout>
class UniformQuantizedConvolutionOp : public OpKernel {
 public:
  explicit UniformQuantizedConvolutionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, convolution_params_.LoadFromAttrs(*context));

    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_min_val",
                                             &output_quantization_min_val_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_max_val",
                                             &output_quantization_max_val_));

    int lhs_quantization_axis;
    OP_REQUIRES_OK(context, context->GetAttr("lhs_quantization_axis",
                                             &lhs_quantization_axis));
    OP_REQUIRES(
        context, (lhs_quantization_axis == -1),
        InvalidArgument("lhs_quantization_axis Attr must be -1 (per-tensor)."));
    OP_REQUIRES_OK(context, context->GetAttr("rhs_quantization_axis",
                                             &rhs_quantization_axis_));
    OP_REQUIRES_OK(context, context->GetAttr("output_quantization_axis",
                                             &output_quantization_axis_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& lhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& lhs_scales = context->input(2);
    const Tensor& lhs_zero_points = context->input(3);
    const Tensor& rhs_scales = context->input(4);
    const Tensor& rhs_zero_points = context->input(5);
    const Tensor& output_scales = context->input(6);
    const Tensor& output_zero_points = context->input(7);

    OP_REQUIRES(context, (AllElementsPositive<float>(lhs_scales)),
                InvalidArgument("lhs scales elements must be all positive."));
    OP_REQUIRES(context, (AllElementsPositive<float>(rhs_scales)),
                InvalidArgument("rhs scales elements must be all positive."));
    OP_REQUIRES(
        context, (AllElementsPositive<float>(output_scales)),
        InvalidArgument("output scales elements must be all positive."));

    OP_REQUIRES_OK(context,
                   convolution_params_.ValidateOrFillParamsAndValidateShape(
                       lhs.shape(), rhs.shape()));

    // Check lhs scales/zero_points shapes.
    OP_REQUIRES(
        context,
        (lhs_scales.IsSameSize(lhs_zero_points) && lhs_scales.dims() == 0),
        InvalidArgument(
            "lhs scales/zero_points must be all scalar tensors. Given: ",
            lhs_scales.shape().DebugString(),
            lhs_zero_points.shape().DebugString()));

    // Check rhs axis.
    OP_REQUIRES(
        context,
        (rhs_quantization_axis_ == -1 ||
         rhs_quantization_axis_ == convolution_params_.dimension_numbers()
                                       .kernel_output_feature_dimension()),
        InvalidArgument("rhs_quantization_axis Attr must be -1 (per-tensor) or "
                        "dimension_numbers.kernel_output_feature_dimension "
                        "(per-channel)."));
    // Check rhs scales/zero_points shapes.
    OP_REQUIRES_OK(
        context, QuantizationAxisAndShapeValid(rhs.shape(), rhs_scales.shape(),
                                               rhs_zero_points.shape(),
                                               rhs_quantization_axis_));

    // Check output axis.
    OP_REQUIRES(
        context,
        (output_quantization_axis_ == -1 ||
         output_quantization_axis_ == convolution_params_.dimension_numbers()
                                          .output_feature_dimension()),
        InvalidArgument(
            "output_quantization_axis Attr must be -1 (per-tensor) or "
            "dimension_numbers.output_feature_dimension (per-channel)."));

    auto output_shape =
        convolution_params_.CalculateOutputShape(lhs.shape(), rhs.shape());
    OP_REQUIRES_OK(context, output_shape.status());
    // Check output scales/zero_points shapes.
    OP_REQUIRES_OK(context,
                   QuantizationAxisAndShapeValid(
                       output_shape.value(), output_scales.shape(),
                       output_zero_points.shape(), output_quantization_axis_));
    OP_REQUIRES(
        context, (rhs_scales.dims() > 0 || output_scales.dims() == 0),
        InvalidArgument(
            "If rhs is per-tensor quantized, output must be also per-tensor "
            "quantized. Given output scales/zero_points of rank ",
            output_scales.dims()));

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape.value(), &output));

    OP_REQUIRES_OK(
        context,
        EvalQuantizedConv<Tin, Tout>(
            context, lhs, rhs, convolution_params_, lhs_scales, lhs_zero_points,
            rhs_scales, rhs_zero_points, output_scales, output_zero_points,
            output_quantization_min_val_, output_quantization_max_val_,
            *output));
  }

 private:
  UniformQuantizedConvolutionParams convolution_params_;
  int rhs_quantization_axis_;
  int output_quantization_axis_;
  int output_quantization_min_val_;
  int output_quantization_max_val_;
};

// This kernel internally quantizes lhs with following conditions, which aligns
// with current TFLite behavior.
// - lhs_quantization_min = -128 (narrow_range = false)
// - lhs_quantization_max = 127
// - lhs_quantization_axis = dimension_numbers.lhs_spec.batch_dimension
// (per-batch quantization)
// - lhs_asymmetric_quantize = true
template <typename Tlhs, typename Trhs, typename Tout>
class UniformQuantizedConvolutionHybridOp : public OpKernel {
 public:
  explicit UniformQuantizedConvolutionHybridOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rhs_quantization_axis",
                                             &rhs_quantization_axis_));
    OP_REQUIRES_OK(context, convolution_params_.LoadFromAttrs(*context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& lhs = context->input(0);
    const Tensor& rhs = context->input(1);
    const Tensor& rhs_scales = context->input(2);
    const Tensor& rhs_zero_points = context->input(3);

    OP_REQUIRES(context, (AllElementsPositive<float>(rhs_scales)),
                InvalidArgument("rhs scales elements must be all positive."));
    OP_REQUIRES_OK(context,
                   convolution_params_.ValidateOrFillParamsAndValidateShape(
                       lhs.shape(), rhs.shape()));
    OP_REQUIRES(
        context,
        (rhs_quantization_axis_ == -1 ||
         rhs_quantization_axis_ == convolution_params_.dimension_numbers()
                                       .kernel_output_feature_dimension()),
        InvalidArgument("rhs_quantization_axis Attr must be -1 (per-tensor) or "
                        "dimension_numbers.kernel_output_feature_dimension "
                        "(per-channel)."));
    // Check rhs scales/zero_points shapes.
    OP_REQUIRES_OK(
        context, QuantizationAxisAndShapeValid(rhs.shape(), rhs_scales.shape(),
                                               rhs_zero_points.shape(),
                                               rhs_quantization_axis_));

    Tensor* output;
    auto output_shape =
        convolution_params_.CalculateOutputShape(lhs.shape(), rhs.shape());
    OP_REQUIRES_OK(context, output_shape.status());
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape.value(), &output));

    OP_REQUIRES_OK(context,
                   EvalHybridConv<Trhs>(context, lhs, rhs, convolution_params_,
                                        rhs_scales, rhs_zero_points, *output));
  }

 private:
  UniformQuantizedConvolutionParams convolution_params_;
  int rhs_quantization_axis_;
};

REGISTER_KERNEL_BUILDER(Name("UniformQuantizedConvolution")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tin")
                            .TypeConstraint<qint32>("Tout"),
                        UniformQuantizedConvolutionOp<qint8, qint32>);

REGISTER_KERNEL_BUILDER(
    Name("UniformQuantizedConvolutionHybrid")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("Tlhs")
        .TypeConstraint<qint8>("Trhs")
        .TypeConstraint<float>("Tout"),
    UniformQuantizedConvolutionHybridOp<float, qint8, float>);

}  // namespace tensorflow
