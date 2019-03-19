/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_CONV_H_

#ifdef GEMMLOWP_NEON

#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "public/map.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

struct GemmlowpOutputPipelineFixedPointPCLhs {
  typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
      ColVectorMap;
  typedef std::tuple<gemmlowp::OutputStageBiasAddition<ColVectorMap>,
                     gemmlowp::OutputStageScaleInt32ByFixedPointAndExponentPC<
                         gemmlowp::VectorShape::Col>,
                     gemmlowp::OutputStageClamp,
                     gemmlowp::OutputStageSaturatingCastToInt8>
      Pipeline;
  static Pipeline MakeExp(const int32* bias_data, int output_rows,
                          const int32 output_offset,
                          const int32* output_multiplier,
                          const int* output_left_shift,
                          int32 output_activation_min,
                          int32 output_activation_max) {
    ColVectorMap bias_vector(bias_data, output_rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;

    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponentPC<
        gemmlowp::VectorShape::Col>
        quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_offset;
    quantize_down_stage.result_fixedpoint_multiplier =
        ColVectorMap(output_multiplier, output_rows);
    quantize_down_stage.result_exponent =
        ColVectorMap(output_left_shift, output_rows);

    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = output_activation_min;
    clamp_stage.max = output_activation_max;
    gemmlowp::OutputStageSaturatingCastToInt8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage,
                           clamp_stage, saturating_cast_stage);
  }
};

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape, int8* output_data,
    const RuntimeShape& im2col_shape, int8* im2col_data,
    gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("Conv/8bit");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int32 input_offset = params.input_offset;
  const int32 output_offset = params.output_offset;
  // Set min and max value of the output.
  static constexpr int32 output_activation_min =
      std::numeric_limits<int8_t>::min();
  static constexpr int32 output_activation_max =
      std::numeric_limits<int8_t>::max();
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int8* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  const int8 input_zero_point = -input_offset;
  TFLITE_DCHECK_GE(input_zero_point, output_activation_min);
  TFLITE_DCHECK_LE(input_zero_point, output_activation_max);
  const uint8 zero_point_byte =
      *reinterpret_cast<const uint8*>(&input_zero_point);
  if (need_dilated_im2col) {
    TFLITE_DCHECK(im2col_data);
    optimized_ops::DilatedIm2col(params, zero_point_byte, input_shape,
                                 input_data, filter_shape, output_shape,
                                 im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    optimized_ops::Im2col(params, filter_height, filter_width, zero_point_byte,
                          input_shape, input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_rows = gemm_input_shape->Dims(3);
  const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int filter_rows = filter_shape.Dims(0);
  const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
  const int output_rows = output_shape.Dims(3);
  // See b/79927784.
  // const int output_cols = FlatSizeSkipDim(output_shape, 3);
  const int output_cols =
      output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, gemm_input_cols);
  TFLITE_DCHECK_EQ(filter_cols, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);
  gemmlowp::MatrixMap<const int8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, filter_rows, filter_cols);
  gemmlowp::MatrixMap<const int8, gemmlowp::MapOrder::ColMajor> input_matrix(
      gemm_input_data, gemm_input_rows, gemm_input_cols);
  gemmlowp::MatrixMap<int8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, output_cols);

  const auto& output_pipeline = GemmlowpOutputPipelineFixedPointPCLhs::MakeExp(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);

  gemmlowp::GemmWithOutputPipeline<
      int8, int8, gemmlowp::SignedL8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix,
      /*filter_offset*/ 0, input_offset, output_pipeline);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // GEMMLOWP_NEON

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_CONV_H_
