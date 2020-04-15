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

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const int8* input_data, const RuntimeShape& filter_shape,
    const int8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape, int8* output_data,
    const RuntimeShape& im2col_shape, int8* im2col_data,
    CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("Conv/8bit");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int32 input_offset = params.input_offset;
  const int32 output_offset = params.output_offset;
  // Set min and max value of the output.
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
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

  cpu_backend_gemm::MatrixParams<int8> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = 0;  // filter is symmetric-quantized
  cpu_backend_gemm::MatrixParams<int8> rhs_params;
  rhs_params.rows = gemm_input_rows;
  rhs_params.cols = gemm_input_cols;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  cpu_backend_gemm::MatrixParams<int8> dst_params;
  dst_params.rows = output_rows;
  dst_params.cols = output_cols;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<
      int32, int8,
      cpu_backend_gemm::QuantizationFlavor::kIntegerWithPerRowMultiplier>
      gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint_perchannel = output_multiplier;
  gemm_params.multiplier_exponent_perchannel = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_CONV_H_
