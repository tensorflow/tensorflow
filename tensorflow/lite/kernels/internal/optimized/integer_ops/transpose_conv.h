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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_TRANSPOSE_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_TRANSPOSE_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace optimized_integer_ops {

// TransposeConvV2 expect the weights in HWOI order.
template <typename InputScalar, typename DestinationScalar>
inline void TransposeConvV2(
    const ConvParams& params, const int32* output_multiplier,
    const int32* output_shift, const RuntimeShape& input_shape,
    const InputScalar* input_data,
    const RuntimeShape& hwoi_ordered_filter_shape,
    const int8_t* hwoi_ordered_filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    DestinationScalar* output_data, const RuntimeShape& col2im_shape,
    int32_t* col2im_data, int32_t* scratch_data,
    CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("TransposeConvV2/int8");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(hwoi_ordered_filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK(col2im_data);
  TFLITE_DCHECK(hwoi_ordered_filter_data);

  const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_image_size = input_shape.Dims(1) * input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_image_size = output_height * output_width;
  const int input_depth =
      MatchingDim(input_shape, 3, hwoi_ordered_filter_shape, 3);
  const int output_depth =
      MatchingDim(output_shape, 3, hwoi_ordered_filter_shape, 2);
  const int input_offset = input_image_size * input_depth;
  const int output_offset = output_image_size * output_depth;

  const int filter_height = hwoi_ordered_filter_shape.Dims(0);
  const int filter_width = hwoi_ordered_filter_shape.Dims(1);
  const int padding_top = params.padding_values.height;
  const int padding_bottom =
      params.padding_values.height + params.padding_values.height_offset;
  const int padding_left = params.padding_values.width;
  const int padding_right =
      params.padding_values.width + params.padding_values.width_offset;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;

  const int hwoi_ordered_filter_total_size =
      filter_height * filter_width * output_depth;

  cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = hwoi_ordered_filter_total_size;
  lhs_params.cols = input_depth;
  // Since our weight is symmetric quantized, the zp will always be 0.
  lhs_params.zero_point = 0;

  int32_t* scratch_data_p = scratch_data;
  std::fill_n(scratch_data, output_offset * batch_size, static_cast<int32>(0));
  for (int i = 0; i < batch_size; ++i) {
    cpu_backend_gemm::MatrixParams<InputScalar> rhs_params;
    rhs_params.order = cpu_backend_gemm::Order::kColMajor;
    rhs_params.rows = input_depth;
    rhs_params.cols = input_image_size;
    rhs_params.zero_point = -params.input_offset;

    cpu_backend_gemm::MatrixParams<int32_t> dst_params;
    dst_params.order = cpu_backend_gemm::Order::kColMajor;
    dst_params.rows = hwoi_ordered_filter_total_size;
    dst_params.cols = input_image_size;

    cpu_backend_gemm::GemmParams<int32_t, int32_t> gemm_params;
    cpu_backend_gemm::Gemm(lhs_params, hwoi_ordered_filter_data, rhs_params,
                           input_data + input_offset * i, dst_params,
                           col2im_data, gemm_params, cpu_backend_context);

    optimized_ops::Col2im(
        col2im_data, output_depth, output_height, output_width, filter_height,
        filter_width, padding_top, padding_left, padding_bottom, padding_right,
        stride_height, stride_width, scratch_data_p);

    scratch_data_p += output_offset;
  }
  scratch_data_p = scratch_data;
  optimized_ops::BiasAdd(scratch_data_p, bias_data, batch_size, output_height,
                         output_width, output_depth);

  optimized_ops::Quantize(output_multiplier, output_shift, output_depth,
                          output_shape.FlatSize(), params.output_offset,
                          output_activation_min, output_activation_max,
                          scratch_data, output_data);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_TRANSPOSE_CONV_H_
