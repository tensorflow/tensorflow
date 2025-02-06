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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_FULLY_CONNECTED_H_

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

template <typename InputScalar, typename DstScalar>
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const InputScalar* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    DstScalar* output_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnectedInt8/8bit");

  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(b/62193649): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
  const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
  TFLITE_DCHECK_EQ(filter_shape.FlatSize(), filter_rows * filter_cols);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);
  }
  const bool use_caching =
      (cpu_backend_context != nullptr) && cpu_backend_context->use_caching();

  cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = 0;
  lhs_params.cache_policy =
      use_caching ? cpu_backend_gemm::DefaultCachePolicy(params.lhs_cacheable)
                  : cpu_backend_gemm::CachePolicy::kNeverCache;
  cpu_backend_gemm::MatrixParams<InputScalar> rhs_params;
  rhs_params.rows = filter_cols;
  rhs_params.cols = batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  rhs_params.cache_policy =
      use_caching ? cpu_backend_gemm::DefaultCachePolicy(params.rhs_cacheable)
                  : cpu_backend_gemm::CachePolicy::kNeverCache;
  cpu_backend_gemm::MatrixParams<DstScalar> dst_params;
  dst_params.rows = filter_rows;
  dst_params.cols = batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<
      int32_t, DstScalar,
      cpu_backend_gemm::QuantizationFlavor::kIntegerWithPerRowMultiplier>
      gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint_perchannel = output_multiplier;
  gemm_params.multiplier_exponent_perchannel = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

template <typename InputScalar, typename DstScalar>
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const InputScalar* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    DstScalar* output_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnectedInt8/8bit");

  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(b/62193649): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
  const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
  TFLITE_DCHECK_EQ(filter_shape.FlatSize(), filter_rows * filter_cols);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);
  }
  const bool use_caching =
      (cpu_backend_context != nullptr) && cpu_backend_context->use_caching();

  cpu_backend_gemm::MatrixParams<int8_t> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  lhs_params.cache_policy =
      use_caching ? cpu_backend_gemm::DefaultCachePolicy(params.lhs_cacheable)
                  : cpu_backend_gemm::CachePolicy::kNeverCache;
  cpu_backend_gemm::MatrixParams<InputScalar> rhs_params;
  rhs_params.rows = filter_cols;
  rhs_params.cols = batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  rhs_params.cache_policy =
      use_caching ? cpu_backend_gemm::DefaultCachePolicy(params.rhs_cacheable)
                  : cpu_backend_gemm::CachePolicy::kNeverCache;
  cpu_backend_gemm::MatrixParams<DstScalar> dst_params;
  dst_params.rows = filter_rows;
  dst_params.cols = batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<int32_t, DstScalar> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_FULLY_CONNECTED_H_
