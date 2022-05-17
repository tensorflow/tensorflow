/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/resize_nearest_neighbor.h"

#if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/transpose_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

#if __aarch64__ && __clang__
#define TFLITE_SOFTMAX_USE_UINT16_LUT
#endif

namespace tflite {
namespace optimized_ops {

// Unoptimized reference ops:
using reference_ops::Broadcast4DSlowGreater;
using reference_ops::Broadcast4DSlowGreaterEqual;
using reference_ops::Broadcast4DSlowGreaterEqualWithScaling;
using reference_ops::Broadcast4DSlowGreaterWithScaling;
using reference_ops::Broadcast4DSlowLess;
using reference_ops::Broadcast4DSlowLessEqual;
using reference_ops::Broadcast4DSlowLessEqualWithScaling;
using reference_ops::Broadcast4DSlowLessWithScaling;
using reference_ops::BroadcastAdd4DSlow;
using reference_ops::BroadcastMul4DSlow;
using reference_ops::BroadcastSub16POTSlow;
using reference_ops::BroadcastSubSlow;
using reference_ops::Concatenation;
using reference_ops::ConcatenationWithScaling;
using reference_ops::DepthConcatenation;
using reference_ops::Div;
using reference_ops::Elu;
using reference_ops::FakeQuant;
using reference_ops::Fill;
using reference_ops::Gather;
using reference_ops::Greater;
using reference_ops::GreaterEqual;
using reference_ops::GreaterEqualWithScaling;
using reference_ops::GreaterWithScaling;
using reference_ops::LeakyRelu;
using reference_ops::Less;
using reference_ops::LessEqual;
using reference_ops::LessEqualWithScaling;
using reference_ops::LessWithScaling;
using reference_ops::Mean;
using reference_ops::ProcessBroadcastShapes;
using reference_ops::RankOneSelect;
using reference_ops::Relu0To1;  // NOLINT
using reference_ops::Relu1;
using reference_ops::Relu6;
using reference_ops::ReluX;
using reference_ops::Round;
using reference_ops::Select;
using reference_ops::SpaceToBatchND;
using reference_ops::Split;
using reference_ops::Sub16;

// TODO(b/80247582) Remove this constant.
// This will be phased out as the shifts are revised with more thought. Use of a
// constant enables us to track progress on this work.
//
// Used to convert from old-style shifts (right) to new-style (left).
static constexpr int kReverseShift = -1;

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen vector expression. The std::conditional here is to
// construct the suitable Eigen type for the constness of the
// data. Indeed, for const data, we need to produce
//    Eigen::Map<const Eigen::Matrix<float, ...>>
// and not the more straightforward
//    Eigen::Map<Eigen::Matrix<const float, ...>>
template <typename Scalar>
using VectorMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type,
                                   Eigen::Dynamic, 1>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>>::type;

template <typename Scalar>
VectorMap<Scalar> MapAsVector(Scalar* data, const RuntimeShape& shape) {
  const int size = shape.FlatSize();
  return VectorMap<Scalar>(data, size, 1);
}

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen matrix expression. The same explanation as for VectorMap
// above also applies here.
template <typename Scalar>
using MatrixMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type,
                                   Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsRows(Scalar* data,
                                               const RuntimeShape& shape) {
  const int dims_count = shape.DimensionsCount();
  const int rows = shape.Dims(dims_count - 1);
  const int cols = FlatSizeSkipDim(shape, dims_count - 1);
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsCols(Scalar* data,
                                                const RuntimeShape& shape) {
  const int cols = shape.Dims(0);
  const int rows = FlatSizeSkipDim(shape, 0);
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar>
using ArrayMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Array<typename std::remove_const<Scalar>::type,
                                  Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar>
ArrayMap<Scalar> MapAsArrayWithLastDimAsRows(Scalar* data,
                                             const RuntimeShape& shape) {
  const int dims_count = shape.DimensionsCount();
  const int rows = shape.Dims(dims_count - 1);
  const int cols = FlatSizeSkipDim(shape, dims_count - 1);
  return ArrayMap<Scalar>(data, rows, cols);
}

// Copied from tensorflow/core/framework/tensor_types.h
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>>
      UnalignedConstMatrix;
};

// TODO(b/62193649): this function is only needed as long
// as we have the --variable_batch hack.
template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithGivenNumberOfRows(Scalar* data,
                                                   const RuntimeShape& shape,
                                                   int rows) {
  const int flatsize = shape.FlatSize();
  TFLITE_DCHECK_EQ(flatsize % rows, 0);
  const int cols = flatsize / rows;
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename ElementwiseF, typename ScalarBroadcastF, typename T>
inline void BinaryBroadcastFiveFold(const ArithmeticParams& unswitched_params,
                                    const RuntimeShape& unswitched_input1_shape,
                                    const T* unswitched_input1_data,
                                    const RuntimeShape& unswitched_input2_shape,
                                    const T* unswitched_input2_data,
                                    const RuntimeShape& output_shape,
                                    T* output_data, ElementwiseF elementwise_f,
                                    ScalarBroadcastF scalar_broadcast_f) {
  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input1_multiplier = unswitched_params.input2_multiplier;
  switched_params.input1_shift = unswitched_params.input2_shift;
  switched_params.input2_offset = unswitched_params.input1_offset;
  switched_params.input2_multiplier = unswitched_params.input1_multiplier;
  switched_params.input2_shift = unswitched_params.input1_shift;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const T* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const T* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  T* output_data_ptr = output_data;
  const T* input1_data_ptr = input1_data;
  const T* input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for
  // input 2. Put another way, input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1) {
    // General fivefold pattern, with y4 > 1 so there is a non-broadcast inner
    // dimension.
    for (int i0 = 0; i0 < y0; ++i0) {
      const T* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          for (int i3 = 0; i3 < y3; ++i3) {
            elementwise_f(y4, params, input1_data_ptr, input2_data_ptr,
                          output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          // We have broadcast y4 of input1 data y3 times, and now move on.
          input1_data_ptr += y4;
        }
      }
      // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
      input2_data_reset = input2_data_ptr;
    }
  } else if (input1_data_ptr != nullptr) {
    // Special case of y4 == 1, in which the innermost loop is a single
    // element and can be combined with the next (y3) as an inner broadcast.
    //
    // Note that this handles the case of pure scalar broadcast when
    // y0 == y1 == y2 == 1. With low overhead it handles cases such as scalar
    // broadcast with batch (as y2 > 1).
    //
    // NOTE The process is the same as the above general case except
    // simplified for y4 == 1 and the loop over y3 is contained within the
    // AddScalarBroadcast function.
    for (int i0 = 0; i0 < y0; ++i0) {
      const T* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          scalar_broadcast_f(y3, params, *input1_data_ptr, input2_data_ptr,
                             output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          input1_data_ptr += 1;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT

// Looks up each element of <indices> in <table>, returns them in a vector.
inline uint8x16_t aarch64_lookup_vector(const uint8x16x4_t table[4],
                                        uint8x16_t indices) {
  // Look up in 1st quarter of the table: top 2 bits of indices == 00
  uint8x16_t output1 = vqtbl4q_u8(table[0], indices);
  // Look up in 2nd quarter of the table: top 2 bits of indices == 01
  uint8x16_t output2 =
      vqtbl4q_u8(table[1], veorq_u8(indices, vdupq_n_u8(0x40)));
  // Look up in 3rd quarter of the table: top 2 bits of indices == 10
  uint8x16_t output3 =
      vqtbl4q_u8(table[2], veorq_u8(indices, vdupq_n_u8(0x80)));
  // Look up in 4th quarter of the table: top 2 bits of indices == 11
  uint8x16_t output4 =
      vqtbl4q_u8(table[3], veorq_u8(indices, vdupq_n_u8(0xc0)));

  // Combine result of the 4 lookups.
  return vorrq_u8(vorrq_u8(output1, output2), vorrq_u8(output3, output4));
}

#endif

inline void AddBiasAndEvalActivationFunction(float output_activation_min,
                                             float output_activation_max,
                                             const RuntimeShape& bias_shape,
                                             const float* bias_data,
                                             const RuntimeShape& array_shape,
                                             float* array_data) {
  BiasAndClamp(output_activation_min, output_activation_max,
               bias_shape.FlatSize(), bias_data, array_shape.FlatSize(),
               array_data);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& bias_shape,
    const float* optional_bias_data, const RuntimeShape& output_shape,
    float* output_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnected");
  const int dims_count = weights_shape.DimensionsCount();
  const int input_rows = weights_shape.Dims(dims_count - 1);
  cpu_backend_gemm::MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = input_rows;
  rhs_params.cols = input_shape.FlatSize() / input_rows;
  rhs_params.cache_policy =
      cpu_backend_gemm::DefaultCachePolicy(params.rhs_cacheable);
  TFLITE_DCHECK_EQ(input_shape.FlatSize(), rhs_params.rows * rhs_params.cols);
  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.cols = weights_shape.Dims(dims_count - 1);
  lhs_params.rows = FlatSizeSkipDim(weights_shape, dims_count - 1);
  lhs_params.cache_policy =
      cpu_backend_gemm::DefaultCachePolicy(params.lhs_cacheable);
  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = output_shape.Dims(output_shape.DimensionsCount() - 1);
  dst_params.cols =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  cpu_backend_gemm::GemmParams<float, float> gemm_params;
  gemm_params.bias = optional_bias_data;
  gemm_params.clamp_min = params.float_activation_min;
  gemm_params.clamp_max = params.float_activation_max;
  cpu_backend_gemm::Gemm(lhs_params, weights_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnected/8bit");
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
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

  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  lhs_params.cache_policy =
      cpu_backend_gemm::DefaultCachePolicy(params.lhs_cacheable);
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = filter_cols;
  rhs_params.cols = batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  rhs_params.cache_policy =
      cpu_backend_gemm::DefaultCachePolicy(params.rhs_cacheable);
  cpu_backend_gemm::MatrixParams<uint8> dst_params;
  dst_params.rows = filter_rows;
  dst_params.cols = batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<int32, uint8> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data_int32, const RuntimeShape& output_shape,
    int16* output_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("FullyConnected/Uint8Int16");
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(output_offset, 0);
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
  const int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = output_depth;
  lhs_params.cols = accum_depth;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  lhs_params.cache_policy =
      cpu_backend_gemm::DefaultCachePolicy(params.lhs_cacheable);
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = accum_depth;
  rhs_params.cols = batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  rhs_params.cache_policy =
      cpu_backend_gemm::DefaultCachePolicy(params.rhs_cacheable);
  cpu_backend_gemm::MatrixParams<int16> dst_params;
  dst_params.rows = output_depth;
  dst_params.cols = batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = 0;
  cpu_backend_gemm::GemmParams<int32, int16> gemm_params;
  gemm_params.bias = bias_data_int32;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

// Internal function doing the actual arithmetic work for
// ShuffledFullyConnected.
// May be called either directly by it (single-threaded case) or may be used
// as the 'task' for worker threads to run (multi-threaded case, see
// ShuffledFullyConnectedWorkerTask below).
inline void ShuffledFullyConnectedWorkerImpl(
    const uint8* shuffled_input_workspace_data,
    const int8* shuffled_weights_data, int batches, int output_depth,
    int output_stride, int accum_depth, const int32* bias_data,
    int32 output_multiplier, int output_shift, int16* output_data) {
#if defined USE_NEON
  const int8* shuffled_weights_ptr = shuffled_weights_data;
  if (batches == 1) {
    const int right_shift = output_shift > 0 ? 0 : -output_shift;
    const int left_shift = output_shift > 0 ? output_shift : 0;
    for (int c = 0; c < output_depth; c += 4) {
      // Accumulation loop.
      int32x4_t row_accum0 = vdupq_n_s32(0);
      int32x4_t row_accum1 = vdupq_n_s32(0);
      int32x4_t row_accum2 = vdupq_n_s32(0);
      int32x4_t row_accum3 = vdupq_n_s32(0);
      for (int d = 0; d < accum_depth; d += 16) {
        int8x16_t weights0 = vld1q_s8(shuffled_weights_ptr + 0);
        int8x16_t weights1 = vld1q_s8(shuffled_weights_ptr + 16);
        int8x16_t weights2 = vld1q_s8(shuffled_weights_ptr + 32);
        int8x16_t weights3 = vld1q_s8(shuffled_weights_ptr + 48);
        shuffled_weights_ptr += 64;
        int8x16_t input =
            vreinterpretq_s8_u8(vld1q_u8(shuffled_input_workspace_data + d));
        int16x8_t local_accum0 =
            vmull_s8(vget_low_s8(weights0), vget_low_s8(input));
        int16x8_t local_accum1 =
            vmull_s8(vget_low_s8(weights1), vget_low_s8(input));
        int16x8_t local_accum2 =
            vmull_s8(vget_low_s8(weights2), vget_low_s8(input));
        int16x8_t local_accum3 =
            vmull_s8(vget_low_s8(weights3), vget_low_s8(input));
        local_accum0 =
            vmlal_s8(local_accum0, vget_high_s8(weights0), vget_high_s8(input));
        local_accum1 =
            vmlal_s8(local_accum1, vget_high_s8(weights1), vget_high_s8(input));
        local_accum2 =
            vmlal_s8(local_accum2, vget_high_s8(weights2), vget_high_s8(input));
        local_accum3 =
            vmlal_s8(local_accum3, vget_high_s8(weights3), vget_high_s8(input));
        row_accum0 = vpadalq_s16(row_accum0, local_accum0);
        row_accum1 = vpadalq_s16(row_accum1, local_accum1);
        row_accum2 = vpadalq_s16(row_accum2, local_accum2);
        row_accum3 = vpadalq_s16(row_accum3, local_accum3);
      }
      // Horizontally reduce accumulators
      int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,
          pairwise_reduced_acc_2, pairwise_reduced_acc_3;
      pairwise_reduced_acc_0 =
          vpadd_s32(vget_low_s32(row_accum0), vget_high_s32(row_accum0));
      pairwise_reduced_acc_1 =
          vpadd_s32(vget_low_s32(row_accum1), vget_high_s32(row_accum1));
      pairwise_reduced_acc_2 =
          vpadd_s32(vget_low_s32(row_accum2), vget_high_s32(row_accum2));
      pairwise_reduced_acc_3 =
          vpadd_s32(vget_low_s32(row_accum3), vget_high_s32(row_accum3));
      const int32x2_t reduced_lo =
          vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
      const int32x2_t reduced_hi =
          vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
      int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
      // Add bias values.
      int32x4_t bias_vec = vld1q_s32(bias_data + c);
      reduced = vaddq_s32(reduced, bias_vec);
      reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));
      // Multiply by the fixed-point multiplier.
      reduced = vqrdmulhq_n_s32(reduced, output_multiplier);
      // Rounding-shift-right.
      using gemmlowp::RoundingDivideByPOT;
      reduced = RoundingDivideByPOT(reduced, right_shift);
      // Narrow values down to 16 bit signed.
      const int16x4_t res16 = vqmovn_s32(reduced);
      vst1_s16(output_data + c, res16);
    }
  } else if (batches == 4) {
    const int right_shift = output_shift > 0 ? 0 : -output_shift;
    const int left_shift = output_shift > 0 ? output_shift : 0;
    for (int c = 0; c < output_depth; c += 4) {
      const int8* shuffled_input_ptr =
          reinterpret_cast<const int8*>(shuffled_input_workspace_data);
      // Accumulation loop.
      int32x4_t row_accum00 = vdupq_n_s32(0);
      int32x4_t row_accum10 = vdupq_n_s32(0);
      int32x4_t row_accum20 = vdupq_n_s32(0);
      int32x4_t row_accum30 = vdupq_n_s32(0);
      int32x4_t row_accum01 = vdupq_n_s32(0);
      int32x4_t row_accum11 = vdupq_n_s32(0);
      int32x4_t row_accum21 = vdupq_n_s32(0);
      int32x4_t row_accum31 = vdupq_n_s32(0);
      int32x4_t row_accum02 = vdupq_n_s32(0);
      int32x4_t row_accum12 = vdupq_n_s32(0);
      int32x4_t row_accum22 = vdupq_n_s32(0);
      int32x4_t row_accum32 = vdupq_n_s32(0);
      int32x4_t row_accum03 = vdupq_n_s32(0);
      int32x4_t row_accum13 = vdupq_n_s32(0);
      int32x4_t row_accum23 = vdupq_n_s32(0);
      int32x4_t row_accum33 = vdupq_n_s32(0);
      for (int d = 0; d < accum_depth; d += 16) {
        int8x16_t weights0 = vld1q_s8(shuffled_weights_ptr + 0);
        int8x16_t weights1 = vld1q_s8(shuffled_weights_ptr + 16);
        int8x16_t weights2 = vld1q_s8(shuffled_weights_ptr + 32);
        int8x16_t weights3 = vld1q_s8(shuffled_weights_ptr + 48);
        shuffled_weights_ptr += 64;
        int8x16_t input0 = vld1q_s8(shuffled_input_ptr + 0);
        int8x16_t input1 = vld1q_s8(shuffled_input_ptr + 16);
        int8x16_t input2 = vld1q_s8(shuffled_input_ptr + 32);
        int8x16_t input3 = vld1q_s8(shuffled_input_ptr + 48);
        shuffled_input_ptr += 64;
        int16x8_t local_accum0, local_accum1, local_accum2, local_accum3;
#define TFLITE_SHUFFLED_FC_ACCUM(B)                                           \
  local_accum0 = vmull_s8(vget_low_s8(weights0), vget_low_s8(input##B));      \
  local_accum1 = vmull_s8(vget_low_s8(weights1), vget_low_s8(input##B));      \
  local_accum2 = vmull_s8(vget_low_s8(weights2), vget_low_s8(input##B));      \
  local_accum3 = vmull_s8(vget_low_s8(weights3), vget_low_s8(input##B));      \
  local_accum0 =                                                              \
      vmlal_s8(local_accum0, vget_high_s8(weights0), vget_high_s8(input##B)); \
  local_accum1 =                                                              \
      vmlal_s8(local_accum1, vget_high_s8(weights1), vget_high_s8(input##B)); \
  local_accum2 =                                                              \
      vmlal_s8(local_accum2, vget_high_s8(weights2), vget_high_s8(input##B)); \
  local_accum3 =                                                              \
      vmlal_s8(local_accum3, vget_high_s8(weights3), vget_high_s8(input##B)); \
  row_accum0##B = vpadalq_s16(row_accum0##B, local_accum0);                   \
  row_accum1##B = vpadalq_s16(row_accum1##B, local_accum1);                   \
  row_accum2##B = vpadalq_s16(row_accum2##B, local_accum2);                   \
  row_accum3##B = vpadalq_s16(row_accum3##B, local_accum3);

        TFLITE_SHUFFLED_FC_ACCUM(0)
        TFLITE_SHUFFLED_FC_ACCUM(1)
        TFLITE_SHUFFLED_FC_ACCUM(2)
        TFLITE_SHUFFLED_FC_ACCUM(3)

#undef TFLITE_SHUFFLED_FC_ACCUM
      }
      // Horizontally reduce accumulators

#define TFLITE_SHUFFLED_FC_STORE(B)                                           \
  {                                                                           \
    int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,                 \
        pairwise_reduced_acc_2, pairwise_reduced_acc_3;                       \
    pairwise_reduced_acc_0 =                                                  \
        vpadd_s32(vget_low_s32(row_accum0##B), vget_high_s32(row_accum0##B)); \
    pairwise_reduced_acc_1 =                                                  \
        vpadd_s32(vget_low_s32(row_accum1##B), vget_high_s32(row_accum1##B)); \
    pairwise_reduced_acc_2 =                                                  \
        vpadd_s32(vget_low_s32(row_accum2##B), vget_high_s32(row_accum2##B)); \
    pairwise_reduced_acc_3 =                                                  \
        vpadd_s32(vget_low_s32(row_accum3##B), vget_high_s32(row_accum3##B)); \
    const int32x2_t reduced_lo =                                              \
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);            \
    const int32x2_t reduced_hi =                                              \
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);            \
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);                 \
    int32x4_t bias_vec = vld1q_s32(bias_data + c);                            \
    reduced = vaddq_s32(reduced, bias_vec);                                   \
    reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));                    \
    reduced = vqrdmulhq_n_s32(reduced, output_multiplier);                    \
    using gemmlowp::RoundingDivideByPOT;                                      \
    reduced = RoundingDivideByPOT(reduced, right_shift);                      \
    const int16x4_t res16 = vqmovn_s32(reduced);                              \
    vst1_s16(output_data + c + B * output_stride, res16);                     \
  }

      TFLITE_SHUFFLED_FC_STORE(0);
      TFLITE_SHUFFLED_FC_STORE(1);
      TFLITE_SHUFFLED_FC_STORE(2);
      TFLITE_SHUFFLED_FC_STORE(3);

#undef TFLITE_SHUFFLED_FC_STORE
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }
#else
  if (batches == 1) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4] = {0};
      // Accumulation loop.
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 16; j++) {
            int8 input_val = shuffled_input_data[d + j];
            int8 weights_val = *shuffled_weights_ptr++;
            accum[i] += weights_val * input_val;
          }
        }
      }
      for (int i = 0; i < 4; i++) {
        // Add bias value
        int acc = accum[i] + bias_data[c + i];
        // Down-scale the final int32 accumulator to the scale used by our
        // (16-bit, typically 3 integer bits) fixed-point format. The quantized
        // multiplier and shift here have been pre-computed offline
        // (e.g. by toco).
        acc =
            MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
        // Saturate, cast to int16, and store to output array.
        acc = std::max(acc, -32768);
        acc = std::min(acc, 32767);
        output_ptr[c + i] = acc;
      }
    }
  } else if (batches == 4) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      const int8* shuffled_input_ptr = shuffled_input_data;
      // Accumulation loop.
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4][4];
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          accum[i][b] = 0;
        }
      }
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 16; j++) {
              int8 input_val = shuffled_input_ptr[16 * b + j];
              int8 weights_val = shuffled_weights_ptr[16 * i + j];
              accum[i][b] += weights_val * input_val;
            }
          }
        }
        shuffled_input_ptr += 64;
        shuffled_weights_ptr += 64;
      }
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          // Add bias value
          int acc = accum[i][b] + bias_data[c + i];
          // Down-scale the final int32 accumulator to the scale used by our
          // (16-bit, typically 3 integer bits) fixed-point format. The
          // quantized multiplier and shift here have been pre-computed offline
          // (e.g. by toco).
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          // Saturate, cast to int16, and store to output array.
          acc = std::max(acc, -32768);
          acc = std::min(acc, 32767);
          output_ptr[b * output_stride + c + i] = acc;
        }
      }
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }
#endif
}

// Wraps ShuffledFullyConnectedWorkerImpl into a Task class
// to allow using gemmlowp's threadpool.
struct ShuffledFullyConnectedWorkerTask : cpu_backend_threadpool::Task {
  ShuffledFullyConnectedWorkerTask(const uint8* input_data,
                                   const int8* shuffled_weights_data,
                                   int batches, int output_depth,
                                   int output_stride, int accum_depth,
                                   const int32* bias_data,
                                   int32 output_multiplier, int output_shift,
                                   int16* output_data)
      : input_data_(input_data),
        shuffled_weights_data_(shuffled_weights_data),
        batches_(batches),
        output_depth_(output_depth),
        output_stride_(output_stride),
        accum_depth_(accum_depth),
        bias_data_(bias_data),
        output_multiplier_(output_multiplier),
        output_shift_(output_shift),
        output_data_(output_data) {}

  void Run() override {
    ShuffledFullyConnectedWorkerImpl(
        input_data_, shuffled_weights_data_, batches_, output_depth_,
        output_stride_, accum_depth_, bias_data_, output_multiplier_,
        output_shift_, output_data_);
  }

  const uint8* input_data_;
  const int8* shuffled_weights_data_;
  int batches_;
  int output_depth_;
  int output_stride_;
  int accum_depth_;
  const int32* bias_data_;
  int32 output_multiplier_;
  int output_shift_;
  int16* output_data_;
};

inline void ShuffledFullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& weights_shape,
    const uint8* shuffled_weights_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, uint8* shuffled_input_workspace_data,
    CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("ShuffledFullyConnected/8bit");
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_EQ(output_activation_min, -32768);
  TFLITE_DCHECK_EQ(output_activation_max, 32767);
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(b/62193649): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);
  TFLITE_DCHECK((accum_depth % 16) == 0);
  TFLITE_DCHECK((output_depth % 4) == 0);
  // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
  // so that just reinterpreting them as int8 values is equivalent to
  // subtracting 128 from them, thus implementing for free the subtraction of
  // the zero_point value 128.
  const int8* int8_shuffled_weights_data =
      reinterpret_cast<const int8*>(shuffled_weights_data);

  // Shuffling and xoring of input activations into the workspace buffer
  if (batches == 1) {
#ifdef USE_NEON
    const uint8x16_t signbit = vdupq_n_u8(0x80);
    for (int i = 0; i < accum_depth; i += 16) {
      uint8x16_t val = vld1q_u8(input_data + i);
      val = veorq_u8(val, signbit);
      vst1q_u8(shuffled_input_workspace_data + i, val);
    }
#else
    for (int i = 0; i < accum_depth; i++) {
      shuffled_input_workspace_data[i] = input_data[i] ^ 0x80;
    }
#endif
  } else if (batches == 4) {
    uint8* shuffled_input_workspace_ptr = shuffled_input_workspace_data;
    int c = 0;
#ifdef USE_NEON
    const uint8x16_t signbit = vdupq_n_u8(0x80);
    for (c = 0; c < accum_depth; c += 16) {
      const uint8* src_data_ptr = input_data + c;
      uint8x16_t val0 = vld1q_u8(src_data_ptr + 0 * accum_depth);
      uint8x16_t val1 = vld1q_u8(src_data_ptr + 1 * accum_depth);
      uint8x16_t val2 = vld1q_u8(src_data_ptr + 2 * accum_depth);
      uint8x16_t val3 = vld1q_u8(src_data_ptr + 3 * accum_depth);
      val0 = veorq_u8(val0, signbit);
      val1 = veorq_u8(val1, signbit);
      val2 = veorq_u8(val2, signbit);
      val3 = veorq_u8(val3, signbit);
      vst1q_u8(shuffled_input_workspace_ptr + 0, val0);
      vst1q_u8(shuffled_input_workspace_ptr + 16, val1);
      vst1q_u8(shuffled_input_workspace_ptr + 32, val2);
      vst1q_u8(shuffled_input_workspace_ptr + 48, val3);
      shuffled_input_workspace_ptr += 64;
    }
#else
    for (c = 0; c < accum_depth; c += 16) {
      for (int b = 0; b < 4; b++) {
        const uint8* src_data_ptr = input_data + b * accum_depth + c;
        for (int j = 0; j < 16; j++) {
          uint8 src_val = *src_data_ptr++;
          // Flip the sign bit, so that the kernel will only need to
          // reinterpret these uint8 values as int8, getting for free the
          // subtraction of the zero_point value 128.
          uint8 dst_val = src_val ^ 0x80;
          *shuffled_input_workspace_ptr++ = dst_val;
        }
      }
    }
#endif
  } else {
    TFLITE_DCHECK(false);
    return;
  }

  static constexpr int kKernelRows = 4;
  const int thread_count =
      LegacyHowManyThreads<kKernelRows>(cpu_backend_context->max_num_threads(),
                                        output_depth, batches, accum_depth);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    ShuffledFullyConnectedWorkerImpl(
        shuffled_input_workspace_data, int8_shuffled_weights_data, batches,
        output_depth, output_depth, accum_depth, bias_data, output_multiplier,
        output_shift, output_data);
    return;
  }

  // Multi-threaded case: use the gemmlowp context's threadpool.
  TFLITE_DCHECK_GT(thread_count, 1);
  std::vector<ShuffledFullyConnectedWorkerTask> tasks;
  // TODO(b/131746020) don't create new heap allocations every time.
  // At least we make it a single heap allocation by using reserve().
  tasks.reserve(thread_count);
  const int kRowsPerWorker =
      RoundUp<kKernelRows>(CeilQuotient(output_depth, thread_count));
  int row_start = 0;
  for (int i = 0; i < thread_count; i++) {
    int row_end = std::min(output_depth, row_start + kRowsPerWorker);
    tasks.emplace_back(shuffled_input_workspace_data,
                       int8_shuffled_weights_data + row_start * accum_depth,
                       batches, row_end - row_start, output_depth, accum_depth,
                       bias_data + row_start, output_multiplier, output_shift,
                       output_data + row_start);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_depth);
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
}

#ifdef USE_NEON

inline int32x4_t RoundToNearest(const float32x4_t input) {
#if defined(__aarch64__) || defined(__SSSE3__)
  // Note: vcvtnq_s32_f32 is not available in ARMv7
  return vcvtnq_s32_f32(input);
#else
  static const float32x4_t zero_val_dup = vdupq_n_f32(0.0f);
  static const float32x4_t point5_val_dup = vdupq_n_f32(0.5f);
  static const float32x4_t minus_point5_val_dup = vdupq_n_f32(-0.5f);

  const uint32x4_t mask = vcltq_f32(input, zero_val_dup);
  const float32x4_t round =
      vbslq_f32(mask, minus_point5_val_dup, point5_val_dup);
  return vcvtq_s32_f32(vaddq_f32(input, round));
#endif  // defined(__aarch64__) || defined(__SSSE3__)
}

inline uint32x4_t RoundToNearestUnsigned(const float32x4_t input) {
#if defined(__aarch64__)
  // Note that vcvtnq_u32_f32 is not available in ARMv7 or in arm_neon_sse.h.
  return vcvtnq_u32_f32(input);
#else
  static const float32x4_t point5_val_dup = vdupq_n_f32(0.5f);

  return vcvtq_u32_f32(vaddq_f32(input, point5_val_dup));
#endif  // defined(__aarch64__)
}

#endif  // USE_NEON

inline void MeanImpl(const tflite::MeanParams& op_params,
                     const RuntimeShape& input_shape, const uint8_t* input_data,
                     int32 multiplier, int32 shift, int32 bias,
                     const RuntimeShape& output_shape, uint8_t* output_data,
                     int start_depth, int end_depth) {
  ruy::profiler::ScopeLabel label("Mean4D/Uint8/MeanImpl");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr int32_t kMinValue = std::numeric_limits<uint8_t>::min();
  constexpr int32_t kMaxValue = std::numeric_limits<uint8_t>::max();

#ifdef USE_NEON
  const int32x4_t bias_dup = vdupq_n_s32(bias);
  const int32x4_t min_dup = vdupq_n_s32(kMinValue);
  const int32x4_t max_dup = vdupq_n_s32(kMaxValue);
#endif  // USE_NEON

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    int out_d = start_depth;
#ifdef USE_NEON

    for (; out_d <= end_depth - 16; out_d += 16) {
      int32x4x4_t temp_sum;
      temp_sum.val[0] = vdupq_n_s32(0);
      temp_sum.val[1] = vdupq_n_s32(0);
      temp_sum.val[2] = vdupq_n_s32(0);
      temp_sum.val[3] = vdupq_n_s32(0);
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const uint8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          uint8x16_t input_data_val = vld1q_u8(input_data_ptr);

          int16x8_t input_data_low_shift =
              vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_data_val)));
          int16x8_t input_data_high_shift =
              vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_data_val)));

          int32x4_t input_low_low =
              vmovl_s16(vget_low_s16(input_data_low_shift));
          int32x4_t input_high_low =
              vmovl_s16(vget_high_s16(input_data_low_shift));
          int32x4_t input_low_high =
              vmovl_s16(vget_low_s16(input_data_high_shift));
          int32x4_t input_high_high =
              vmovl_s16(vget_high_s16(input_data_high_shift));

          temp_sum.val[0] = vaddq_s32(temp_sum.val[0], input_low_low);
          temp_sum.val[1] = vaddq_s32(temp_sum.val[1], input_high_low);
          temp_sum.val[2] = vaddq_s32(temp_sum.val[2], input_low_high);
          temp_sum.val[3] = vaddq_s32(temp_sum.val[3], input_high_high);
        }
      }

      temp_sum =
          MultiplyByQuantizedMultiplier4Rows(temp_sum, multiplier, shift);

      temp_sum.val[0] = vaddq_s32(temp_sum.val[0], bias_dup);
      temp_sum.val[1] = vaddq_s32(temp_sum.val[1], bias_dup);
      temp_sum.val[2] = vaddq_s32(temp_sum.val[2], bias_dup);
      temp_sum.val[3] = vaddq_s32(temp_sum.val[3], bias_dup);

      temp_sum.val[0] = vminq_s32(vmaxq_s32(temp_sum.val[0], min_dup), max_dup);
      temp_sum.val[1] = vminq_s32(vmaxq_s32(temp_sum.val[1], min_dup), max_dup);
      temp_sum.val[2] = vminq_s32(vmaxq_s32(temp_sum.val[2], min_dup), max_dup);
      temp_sum.val[3] = vminq_s32(vmaxq_s32(temp_sum.val[3], min_dup), max_dup);

      uint16x4_t narrowed_low_low =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[0]));
      uint16x4_t narrowed_high_low =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[1]));
      uint16x4_t narrowed_low_high =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[2]));
      uint16x4_t narrowed_high_high =
          vmovn_u32(vreinterpretq_u32_s32(temp_sum.val[3]));

      uint16x8_t combined_low =
          vcombine_u16(narrowed_low_low, narrowed_high_low);
      uint16x8_t combined_high =
          vcombine_u16(narrowed_low_high, narrowed_high_high);

      uint8x8_t narrowed_low = vmovn_u16(combined_low);
      uint8x8_t narrowed_high = vmovn_u16(combined_high);

      uint8x16_t combined_output = vcombine_u8(narrowed_low, narrowed_high);

      uint8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      vst1q_u8(output_data_ptr, combined_output);
    }
#endif  // USE_NEON

    for (; out_d < end_depth; ++out_d) {
      int acc = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          acc += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }

      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += bias;
      acc = std::min(std::max(acc, kMinValue), kMaxValue);
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          static_cast<uint8_t>(acc);
    }
  }
}

struct MeanWorkerTask : cpu_backend_threadpool::Task {
  MeanWorkerTask(const tflite::MeanParams& op_params,
                 const RuntimeShape& input_shape, const uint8_t* input_data,
                 int32 multiplier, int32 shift, int32 bias,
                 const RuntimeShape& output_shape, uint8_t* output_data,
                 int start_height, int end_height)
      : op_params(op_params),
        input_shape(input_shape),
        input_data(input_data),
        multiplier(multiplier),
        shift(shift),
        bias(bias),
        output_shape(output_shape),
        output_data(output_data),
        start_height(start_height),
        end_height(end_height) {}

  void Run() override {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, start_height, end_height);
  }

 private:
  const tflite::MeanParams& op_params;
  const RuntimeShape& input_shape;
  const uint8_t* input_data;
  int32 multiplier;
  int32 shift;
  int32 bias;
  const RuntimeShape& output_shape;
  uint8_t* output_data;
  int start_height;
  int end_height;
};

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const uint8_t* input_data, int32 input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 uint8_t* output_data, int32 output_zero_point,
                 float output_scale, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("Mean4D/Uint8");
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const float num_elements_in_axis = input_width * input_height;

  float temp = input_zero_point * input_scale / output_scale;
  temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
  int32_t bias = output_zero_point - static_cast<int32_t>(temp);
  float real_scale = input_scale / (num_elements_in_axis * output_scale);

  int32 multiplier, shift;
  QuantizeMultiplier(real_scale, &multiplier, &shift);

  constexpr int kMinDepthPerThread = 8;
  int thread_count = output_depth / kMinDepthPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      std::min(thread_count, cpu_backend_context->max_num_threads());

  if (capped_thread_count == 1) {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, 0, output_depth);
  } else {
    // Instead parallel for batch, we loop for the output_depth since batch
    // is typical 1.
    std::vector<MeanWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int depth_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int depth_end = depth_start +
                      (output_depth - depth_start) / (capped_thread_count - i);
      tasks.emplace_back(op_params, input_shape, input_data, multiplier, shift,
                         bias, output_shape, output_data, depth_start,
                         depth_end);
      depth_start = depth_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

template <typename T, typename U>
inline bool MeanGeneral(const T* input_data, const int* input_dims,
                        const int input_num_dims, T* output_data,
                        const int* output_dims, const int output_num_dims,
                        const int* axis, const int num_axis_dimensions,
                        bool keep_dims, int* temp_index, int* resolved_axis,
                        int* normalized_dims, U* temp_sum) {
  return reference_ops::Mean(input_data, input_dims, input_num_dims,
                             output_data, output_dims, output_num_dims, axis,
                             num_axis_dimensions, keep_dims, temp_index,
                             resolved_axis, normalized_dims, temp_sum);
}

template <>
inline bool MeanGeneral<float, float>(
    const float* input_data, const int* input_dims, const int input_num_dims,
    float* output_data, const int* output_dims, const int output_num_dims,
    const int* axis, const int num_axis_dimensions, bool keep_dims,
    int* temp_index, int* resolved_axis, int* normalized_dims,
    float* temp_sum) {
  // Handle reduce_mean for the last dimensions.
  if (num_axis_dimensions == 1 && axis[0] == (input_num_dims - 1)) {
    ruy::profiler::ScopeLabel label("MeanLastDim/Float");
    int output_size = 1;
    for (int i = 0; i < input_num_dims - 1; ++i) {
      output_size *= input_dims[i];
    }
    const int last_input_dim = input_dims[axis[0]];

    // TODO(b/152563685): Consider use eigen to cover more general cases.
    const MatrixMap<const float> in_mat(input_data, last_input_dim,
                                        output_size);
    VectorMap<float> out(output_data, output_size, 1);
    out = (in_mat.array().colwise().sum()) / static_cast<float>(last_input_dim);
    return true;
  }

  return reference_ops::Mean(input_data, input_dims, input_num_dims,
                             output_data, output_dims, output_num_dims, axis,
                             num_axis_dimensions, keep_dims, temp_index,
                             resolved_axis, normalized_dims, temp_sum);
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data, CpuBackendContext* cpu_backend_context) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  ruy::profiler::ScopeLabel label("Conv");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col) {
    DilatedIm2col(params, float_zero_byte, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col(params, filter_height, filter_width, float_zero_byte, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

#if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
  // The following code computes matrix multiplication c = a * transponse(b)
  // with CBLAS, where:
  // * `a` is a matrix with dimensions (m, k).
  // * `b` is a matrix with dimensions (n, k), so transpose(b) is (k, n).
  // * `c` is a matrix with dimensions (m, n).
  // The naming of variables are aligned with CBLAS specification here.
  const float* a = gemm_input_data;
  const float* b = filter_data;
  float* c = output_data;
  // The stride of matrix a, b and c respectively.
  int stride_a = k;
  int stride_b = k;
  int stride_c = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a,
              stride_a, b, stride_b, 0.0f, c, stride_c);
  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max, bias_shape, bias_data,
      output_shape, output_data);
#else
  // When an optimized CBLAS implementation is not available, fall back
  // to using cpu_backend_gemm.
  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = n;
  lhs_params.cols = k;
  cpu_backend_gemm::MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = k;
  rhs_params.cols = m;
  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;
  cpu_backend_gemm::GemmParams<float, float> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
#endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
}

inline void HybridConv(const ConvParams& params, float* scaling_factors_ptr,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& filter_shape,
                       const int8_t* filter_data,
                       const RuntimeShape& bias_shape, const float* bias_data,
                       const RuntimeShape& accum_scratch_shape,
                       int32_t* accum_scratch, const RuntimeShape& output_shape,
                       float* output_data, const RuntimeShape& im2col_shape,
                       int8_t* im2col_data, CpuBackendContext* context) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batch_size = input_shape.Dims(0);
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);

  const int input_zero_point = 0;
  const int8_t* gemm_input_data = nullptr;
  int num_input;
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  if (need_dilated_im2col) {
    DilatedIm2col(params, input_zero_point, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    num_input = im2col_shape.FlatSize();
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    // symmetric quantization assumes zero point of 0.

    Im2col(params, filter_height, filter_width, input_zero_point, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    num_input = im2col_shape.FlatSize();
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    num_input = input_shape.FlatSize();
  }

  // Flatten 4D matrices into 2D matrices for matrix multiplication.

  // Flatten so that each filter has its own row.
  const int filter_rows = filter_shape.Dims(0);
  const int filter_cols = FlatSizeSkipDim(filter_shape, 0);

  // In MatrixBatchVectorMultiplyAccumulate, each output value is the
  // dot product of one row of the first matrix with one row of the second
  // matrix. Therefore, the number of cols in each matrix are equivalent.
  //
  // After Im2Col, each input patch becomes a row.
  const int gemm_input_cols = filter_cols;
  const int gemm_input_rows = num_input / gemm_input_cols;

  const int output_cols = output_shape.Dims(3);
  const int output_rows = FlatSizeSkipDim(output_shape, 3);
  TFLITE_DCHECK_EQ(output_cols, filter_rows);
  TFLITE_DCHECK_EQ(output_rows, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_cols);

  // MatrixBatchVectorMultiplyAccumulate assumes that each row of the second
  // input matrix has its own scale factor. This code duplicates the scale
  // factors for each row in the same batch.
  const int rows_per_batch = gemm_input_rows / batch_size;
  for (int i = gemm_input_rows - 1; i >= 0; --i) {
    scaling_factors_ptr[i] = scaling_factors_ptr[i / rows_per_batch];
  }

  std::fill_n(output_data, output_rows * output_cols, 0.0f);

  // The scratch buffer must have the same size as the output.
  TFLITE_DCHECK_EQ(accum_scratch_shape.FlatSize(), output_shape.FlatSize());
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      filter_data, filter_rows, filter_cols, gemm_input_data,
      scaling_factors_ptr, /*n_batch=*/gemm_input_rows, accum_scratch,
      output_data, context);
  AddBiasAndEvalActivationFunction(output_activation_min, output_activation_max,
                                   bias_shape, bias_data, output_shape,
                                   output_data);
}

inline void HybridConvPerChannel(
    const ConvParams& params, float* scaling_factors_ptr,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    const RuntimeShape& im2col_shape, int8_t* im2col_data,
    const float* per_channel_scale, int32_t* input_offset,
    const RuntimeShape& scratch_shape, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("ConvHybridPerChannel");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
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

  const int batch_size = input_shape.Dims(0);

  if (need_dilated_im2col) {
    TFLITE_DCHECK(im2col_data);
    optimized_ops::DilatedIm2col(params, input_shape, input_data, filter_shape,
                                 output_shape, im2col_data, input_offset,
                                 batch_size);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    Im2col(params, filter_height, filter_width, input_offset, batch_size,
           input_shape, input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int filter_rows = filter_shape.Dims(0);
  const int filter_cols = FlatSizeSkipDim(filter_shape, 0);

  const int gemm_input_rows = gemm_input_shape->Dims(3);
  const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int output_rows = output_shape.Dims(3);
  const int output_cols =
      output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);

  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, gemm_input_cols);
  TFLITE_DCHECK_EQ(filter_cols, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);
  TFLITE_DCHECK_EQ(scratch_shape.FlatSize(), output_shape.FlatSize());
  if (!compute_row_sums || *compute_row_sums) {
    tensor_utils::ReductionSumVector(filter_data, row_sums, filter_rows,
                                     filter_cols);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
  }

  cpu_backend_gemm::MatrixParams<int8> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;

  cpu_backend_gemm::MatrixParams<int8> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = gemm_input_rows;
  rhs_params.cols = gemm_input_cols;

  cpu_backend_gemm::MatrixParams<int32> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = output_rows;
  dst_params.cols = output_cols;

  // TODO(b/149003801): Use hybrid gemm once supported in Ruy.
  cpu_backend_gemm::GemmParams<int32_t, int32_t> gemm_params;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,
                         dst_params, scratch, gemm_params, cpu_backend_context);

  MatrixMap<float> out_mat(output_data, filter_rows, output_cols);
  MatrixMap<int32_t> in_mat(scratch, filter_rows, output_cols);
  VectorMap<const float> bias_data_vec(bias_data, filter_rows, 1);
  VectorMap<int32_t> row_sums_vec(row_sums, filter_rows, 1);
  VectorMap<const float> per_channel_scale_vec(per_channel_scale, filter_rows,
                                               1);
  const int cols_per_batch = output_cols / batch_size;
  for (int c = 0; c < output_cols; c++) {
    const int b = c / cols_per_batch;
    const float input_scale = scaling_factors_ptr[b];
    const int32_t zero_point = input_offset[b];
    out_mat.col(c) =
        (((in_mat.col(c) - (row_sums_vec * zero_point))
              .cast<float>()
              .cwiseProduct((per_channel_scale_vec * input_scale))) +
         bias_data_vec)
            .cwiseMin(params.float_activation_max)
            .cwiseMax(params.float_activation_min);
  }
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8* input_data, const RuntimeShape& filter_shape,
                 const uint8* filter_data, const RuntimeShape& bias_shape,
                 const int32* bias_data, const RuntimeShape& output_shape,
                 uint8* output_data, const RuntimeShape& im2col_shape,
                 uint8* im2col_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("Conv/8bit");

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const uint8* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    DilatedIm2col(params, input_zero_point, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    Im2col(params, filter_height, filter_width, input_zero_point, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_rows = gemm_input_shape->Dims(3);
  // Using FlatSizeSkipDim causes segfault in some contexts (see b/79927784).
  // The root cause has not yet been identified though. Same applies below for
  // the other calls commented out. This is a partial rollback of cl/196819423.
  // const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int gemm_input_cols = gemm_input_shape->Dims(0) *
                              gemm_input_shape->Dims(1) *
                              gemm_input_shape->Dims(2);
  const int filter_rows = filter_shape.Dims(0);
  // See b/79927784.
  // const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
  const int filter_cols =
      filter_shape.Dims(1) * filter_shape.Dims(2) * filter_shape.Dims(3);
  const int output_rows = output_shape.Dims(3);
  // See b/79927784.
  // const int output_cols = FlatSizeSkipDim(output_shape, 3);
  const int output_cols =
      output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, gemm_input_cols);
  TFLITE_DCHECK_EQ(filter_cols, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);

  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = gemm_input_rows;
  rhs_params.cols = gemm_input_cols;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  cpu_backend_gemm::MatrixParams<uint8> dst_params;
  dst_params.rows = output_rows;
  dst_params.cols = output_cols;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<int32, uint8> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

template <typename T>
inline void DepthToSpace(const tflite::DepthToSpaceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  ruy::profiler::ScopeLabel label("DepthToSpace");

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);

  const int output_depth = output_shape.Dims(3);
  const int batch_size = output_shape.Dims(0);

  // Number of continuous values that we can copy in one interation.
  const int stride = op_params.block_size * output_depth;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const T* input_ptr = input_data + Offset(input_shape, batch, in_h, 0, 0);
      for (int offset_h = 0; offset_h < op_params.block_size; ++offset_h) {
        const T* src = input_ptr;
        for (int in_w = 0; in_w < input_width; ++in_w) {
          memcpy(output_data, src, stride * sizeof(T));
          output_data += stride;
          src += input_depth;
        }
        input_ptr += stride;
      }
    }
  }
}

template <typename T>
inline void SpaceToDepth(const tflite::SpaceToDepthParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  ruy::profiler::ScopeLabel label("SpaceToDepth");

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  const int input_depth = input_shape.Dims(3);
  const int batch_size = input_shape.Dims(0);

  // Number of continuous values that we can copy in one interation.
  const int stride = op_params.block_size * input_depth;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      T* output_ptr = output_data + Offset(output_shape, batch, out_h, 0, 0);
      for (int offset_h = 0; offset_h < op_params.block_size; ++offset_h) {
        T* dst = output_ptr;
        for (int out_w = 0; out_w < output_width; ++out_w) {
          memcpy(dst, input_data, stride * sizeof(T));
          input_data += stride;
          dst += output_depth;
        }
        output_ptr += stride;
      }
    }
  }
}

inline void Relu(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Relu (not fused)");

  const auto input = MapAsVector(input_data, input_shape);
  auto output = MapAsVector(output_data, output_shape);
  output = input.cwiseMax(0.0f);
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const float* input_data,
                            const RuntimeShape& output_shape,
                            float* output_data, float epsilon = 1e-6) {
  ruy::profiler::ScopeLabel label("L2Normalization");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i) {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c) {
      const float val = input_data[c];
      squared_l2_norm += val * val;
    }
    float l2_norm = std::sqrt(squared_l2_norm);
    l2_norm = std::max(l2_norm, epsilon);
    for (int c = 0; c < depth; ++c) {
      *output_data = *input_data / l2_norm;
      ++output_data;
      ++input_data;
    }
  }
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const uint8* input_data,
                            const RuntimeShape& output_shape,
                            uint8* output_data) {
  ruy::profiler::ScopeLabel label("L2Normalization/8bit");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int32 input_zero_point = op_params.input_zero_point;
  for (int i = 0; i < outer_size; ++i) {
    int32 square_l2_norm = 0;
    for (int c = 0; c < depth; c++) {
      // Note that input_data advances by depth in the second pass below.
      int32 diff = input_data[c] - input_zero_point;
      square_l2_norm += diff * diff;
    }
    // TODO(b/29395854): add clamping to TOCO and TF Lite kernel
    // for all zero tensors in the input_data
    int32 inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(square_l2_norm, kReverseShift,
                                     &inv_l2norm_multiplier, &inv_l2norm_shift);

    for (int c = 0; c < depth; c++) {
      int32 diff = *input_data - input_zero_point;
      int32 rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
      int32 unclamped_output_val = 128 + rescaled_diff;
      int32 output_val = std::min(255, std::max(0, unclamped_output_val));
      *output_data = static_cast<uint8>(output_val);
      ++input_data;
      ++output_data;
    }
  }
}

inline void AddElementwise(int size, const ArithmeticParams& params,
                           const float* input1_data, const float* input2_data,
                           float* output_data) {
  int i = 0;

#ifdef USE_NEON
  const auto activation_min = vdupq_n_f32(params.float_activation_min);
  const auto activation_max = vdupq_n_f32(params.float_activation_max);
  for (; i <= size - 16; i += 16) {
    auto a10 = vld1q_f32(input1_data + i);
    auto a11 = vld1q_f32(input1_data + i + 4);
    auto a12 = vld1q_f32(input1_data + i + 8);
    auto a13 = vld1q_f32(input1_data + i + 12);
    auto a20 = vld1q_f32(input2_data + i);
    auto a21 = vld1q_f32(input2_data + i + 4);
    auto a22 = vld1q_f32(input2_data + i + 8);
    auto a23 = vld1q_f32(input2_data + i + 12);
    auto x0 = vaddq_f32(a10, a20);
    auto x1 = vaddq_f32(a11, a21);
    auto x2 = vaddq_f32(a12, a22);
    auto x3 = vaddq_f32(a13, a23);
    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4) {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vaddq_f32(a1, a2);
    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);
    vst1q_f32(output_data + i, x);
  }
#endif  // NEON

  for (; i < size; i++) {
    auto x = input1_data[i] + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(
        x, params.float_activation_min, params.float_activation_max);
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Add");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const uint8* input1_data, const uint8* input2_data,
                           uint8* output_data) {
  ruy::profiler::ScopeLabel label("AddElementwise/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
#ifdef USE_NEON
  const uint8x8_t output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const uint8x8_t output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);
  for (; i <= size - 8; i += 8) {
    const uint8x8_t input1_val_original = vld1_u8(input1_data + i);
    const uint8x8_t input2_val_original = vld1_u8(input2_data + i);
    const int16x8_t input1_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const int16x8_t input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const int16x8_t input1_val =
        vaddq_s16(input1_val_s16, vdupq_n_s16(params.input1_offset));
    const int16x8_t input2_val =
        vaddq_s16(input2_val_s16, vdupq_n_s16(params.input2_offset));
    const int16x4_t input1_val_high = vget_high_s16(input1_val);
    const int16x4_t input1_val_low = vget_low_s16(input1_val);
    const int16x4_t input2_val_high = vget_high_s16(input2_val);
    const int16x4_t input2_val_low = vget_low_s16(input2_val);
    int32x4_t x11 = vmovl_s16(input1_val_low);
    int32x4_t x12 = vmovl_s16(input1_val_high);
    int32x4_t x21 = vmovl_s16(input2_val_low);
    int32x4_t x22 = vmovl_s16(input2_val_high);
    const int32x4_t left_shift_dup = vdupq_n_s32(params.left_shift);
    x11 = vshlq_s32(x11, left_shift_dup);
    x12 = vshlq_s32(x12, left_shift_dup);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x11 = vqrdmulhq_n_s32(x11, params.input1_multiplier);
    x12 = vqrdmulhq_n_s32(x12, params.input1_multiplier);
    x21 = vqrdmulhq_n_s32(x21, params.input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, params.input2_multiplier);
    const int32x4_t input1_shift_dup = vdupq_n_s32(params.input1_shift);
    const int32x4_t input2_shift_dup = vdupq_n_s32(params.input2_shift);
    x11 = vshlq_s32(x11, input1_shift_dup);
    x12 = vshlq_s32(x12, input1_shift_dup);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    int32x4_t s1 = vaddq_s32(x11, x21);
    int32x4_t s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, params.output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    s1 = RoundingDivideByPOT(s1, -params.output_shift);
    s2 = RoundingDivideByPOT(s2, -params.output_shift);
    const int16x4_t s1_narrowed = vmovn_s32(s1);
    const int16x4_t s2_narrowed = vmovn_s32(s2);
    const int16x8_t s = vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed),
                                  vdupq_n_s16(params.output_offset));
    const uint8x8_t clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(s)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32 scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32 raw_sum = scaled_input1_val + scaled_input2_val;
    const int32 raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcast(int size, const ArithmeticParams& params,
                               uint8 input1_data, const uint8* input2_data,
                               uint8* output_data) {
  using gemmlowp::RoundingDivideByPOT;

  ruy::profiler::ScopeLabel label("AddScalarBroadcast/8bit");
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

  int i = 0;

#ifdef USE_NEON
  const int32x4_t left_shift_dup = vdupq_n_s32(params.left_shift);
  const uint8x8_t output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const uint8x8_t output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);

  // Process broadcast scalar.
  const uint8x8_t input1_val_original = vdup_n_u8(input1_data);
  const int16x8_t input1_val_s16 =
      vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
  const int16x8_t input1_val =
      vaddq_s16(input1_val_s16, vdupq_n_s16(params.input1_offset));
  const int16x4_t input1_val_high = vget_high_s16(input1_val);
  const int16x4_t input1_val_low = vget_low_s16(input1_val);
  int32x4_t x11 = vmovl_s16(input1_val_low);
  int32x4_t x12 = vmovl_s16(input1_val_high);
  x11 = vshlq_s32(x11, left_shift_dup);
  x12 = vshlq_s32(x12, left_shift_dup);
  x11 = vqrdmulhq_n_s32(x11, params.input1_multiplier);
  x12 = vqrdmulhq_n_s32(x12, params.input1_multiplier);
  const int32x4_t input1_shift_dup = vdupq_n_s32(params.input1_shift);
  x11 = vshlq_s32(x11, input1_shift_dup);
  x12 = vshlq_s32(x12, input1_shift_dup);

  for (; i <= size - 8; i += 8) {
    const uint8x8_t input2_val_original = vld1_u8(input2_data + i);
    const int16x8_t input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const int16x8_t input2_val =
        vaddq_s16(input2_val_s16, vdupq_n_s16(params.input2_offset));
    const int16x4_t input2_val_high = vget_high_s16(input2_val);
    const int16x4_t input2_val_low = vget_low_s16(input2_val);
    int32x4_t x21 = vmovl_s16(input2_val_low);
    int32x4_t x22 = vmovl_s16(input2_val_high);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x21 = vqrdmulhq_n_s32(x21, params.input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, params.input2_multiplier);
    const int32x4_t input2_shift_dup = vdupq_n_s32(params.input2_shift);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    int32x4_t s1 = vaddq_s32(x11, x21);
    int32x4_t s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, params.output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, params.output_multiplier);
    s1 = RoundingDivideByPOT(s1, -params.output_shift);
    s2 = RoundingDivideByPOT(s2, -params.output_shift);
    const int16x4_t s1_narrowed = vmovn_s32(s1);
    const int16x4_t s2_narrowed = vmovn_s32(s2);
    const int16x8_t s = vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed),
                                  vdupq_n_s16(params.output_offset));
    const uint8x8_t clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(s)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  if (i < size) {
    // Process broadcast scalar.
    const int32 input1_val = params.input1_offset + input1_data;
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);

    for (; i < size; ++i) {
      const int32 input2_val = params.input2_offset + input2_data[i];
      const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
      const int32 scaled_input2_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_input2_val, params.input2_multiplier,
              params.input2_shift);
      const int32 raw_sum = scaled_input1_val + scaled_input2_val;
      const int32 raw_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_sum, params.output_multiplier, params.output_shift) +
          params.output_offset;
      const int32 clamped_output =
          std::min(params.quantized_activation_max,
                   std::max(params.quantized_activation_min, raw_output));
      output_data[i] = static_cast<uint8>(clamped_output);
    }
  }
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcast(int size, const ArithmeticParams& params,
                               float broadcast_value, const float* input2_data,
                               float* output_data) {
  int i = 0;
#ifdef USE_NEON
  const float32x4_t output_activation_min_vector =
      vdupq_n_f32(params.float_activation_min);
  const float32x4_t output_activation_max_vector =
      vdupq_n_f32(params.float_activation_max);
  const float32x4_t broadcast_value_dup = vdupq_n_f32(broadcast_value);
  for (; i <= size - 4; i += 4) {
    const float32x4_t input2_val_original = vld1q_f32(input2_data + i);

    const float32x4_t output =
        vaddq_f32(input2_val_original, broadcast_value_dup);

    const float32x4_t clamped =
        vmaxq_f32(output_activation_min_vector,
                  vminq_f32(output_activation_max_vector, output));
    vst1q_f32(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    auto x = broadcast_value + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(
        x, params.float_activation_min, params.float_activation_max);
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8* input1_data,
                const RuntimeShape& input2_shape, const uint8* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("Add/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  ruy::profiler::ScopeLabel label("Add/Int16");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int input1_shift = params.input1_shift;
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  const int16 output_activation_min = params.quantized_activation_min;
  const int16 output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK(input1_shift == 0 || params.input2_shift == 0);
  TFLITE_DCHECK_LE(input1_shift, 0);
  TFLITE_DCHECK_LE(params.input2_shift, 0);
  const int16* not_shift_input = input1_shift == 0 ? input1_data : input2_data;
  const int16* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_right_shift =
      input1_shift == 0 ? -params.input2_shift : -input1_shift;

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
    F0 scaled_input = F0::FromRaw(
        gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
    F0 result = gemmlowp::SaturatingAdd(scaled_input, input_ready_scaled);
    const int16 raw_output = result.raw();
    const int16 clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = clamped_output;
  }
}

template <typename T>
inline typename std::enable_if<is_int32_or_int64<T>::value, void>::type Add(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Add/int32or64");

  T activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);

  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  if (input1_shape == input2_shape) {
    output_map.array() = (input1_map.array() + input2_map.array())
                             .cwiseMax(activation_min)
                             .cwiseMin(activation_max);
  } else if (input2_shape.FlatSize() == 1) {
    auto scalar = input2_data[0];
    output_map.array() = (input1_map.array() + scalar)
                             .cwiseMax(activation_min)
                             .cwiseMin(activation_max);
  } else if (input1_shape.FlatSize() == 1) {
    auto scalar = input1_data[0];
    output_map.array() = (scalar + input2_map.array())
                             .cwiseMax(activation_min)
                             .cwiseMin(activation_max);
  } else {
    reference_ops::BroadcastAdd4DSlow<T>(params, input1_shape, input1_data,
                                         input2_shape, input2_data,
                                         output_shape, output_data);
  }
}

template <typename T>
inline void BroadcastAddDispatch(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return BroadcastAdd4DSlow(params, input1_shape, input1_data, input2_shape,
                              input2_data, output_shape, output_data);
  }

  BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data,
      static_cast<void (*)(int, const ArithmeticParams&, const T*, const T*,
                           T*)>(AddElementwise),
      static_cast<void (*)(int, const ArithmeticParams&, T, const T*, T*)>(
          AddScalarBroadcast));
}

inline void BroadcastAddFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
  BroadcastAddDispatch(unswitched_params, unswitched_input1_shape,
                       unswitched_input1_data, unswitched_input2_shape,
                       unswitched_input2_data, output_shape, output_data);
}

inline void BroadcastAddFivefold(const ArithmeticParams& params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const float* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const float* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 float* output_data) {
  BroadcastAddDispatch(params, unswitched_input1_shape, unswitched_input1_data,
                       unswitched_input2_shape, unswitched_input2_data,
                       output_shape, output_data);
}

inline void MulElementwise(int size, const ArithmeticParams& params,
                           const float* input1_data, const float* input2_data,
                           float* output_data) {
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  int i = 0;
#ifdef USE_NEON
  const auto activation_min = vdupq_n_f32(output_activation_min);
  const auto activation_max = vdupq_n_f32(output_activation_max);
  for (; i <= size - 16; i += 16) {
    auto a10 = vld1q_f32(input1_data + i);
    auto a11 = vld1q_f32(input1_data + i + 4);
    auto a12 = vld1q_f32(input1_data + i + 8);
    auto a13 = vld1q_f32(input1_data + i + 12);
    auto a20 = vld1q_f32(input2_data + i);
    auto a21 = vld1q_f32(input2_data + i + 4);
    auto a22 = vld1q_f32(input2_data + i + 8);
    auto a23 = vld1q_f32(input2_data + i + 12);
    auto x0 = vmulq_f32(a10, a20);
    auto x1 = vmulq_f32(a11, a21);
    auto x2 = vmulq_f32(a12, a22);
    auto x3 = vmulq_f32(a13, a23);

    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);

    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4) {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vmulq_f32(a1, a2);

    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);

    vst1q_f32(output_data + i, x);
  }
#endif  // NEON

  for (; i < size; i++) {
    auto x = input1_data[i] * input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(x, output_activation_min,
                                                  output_activation_max);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Mul");

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int32* input1_data,
                const RuntimeShape& input2_shape, const int32* input2_data,
                const RuntimeShape& output_shape, int32* output_data) {
  ruy::profiler::ScopeLabel label("Mul/int32/activation");

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] * input2_data[i], output_activation_min,
        output_activation_max);
  }
}

inline void MulNoActivation(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const int32* input1_data,
                            const RuntimeShape& input2_shape,
                            const int32* input2_data,
                            const RuntimeShape& output_shape,
                            int32* output_data) {
  ruy::profiler::ScopeLabel label("Mul/int32");

  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  if (input1_shape == input2_shape) {
    output_map.array() = input1_map.array() * input2_map.array();
  } else if (input2_shape.FlatSize() == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() * scalar;
  } else if (input1_shape.FlatSize() == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar * input2_map.array();
  } else {
    reference_ops::BroadcastMul4DSlow(params, input1_shape, input1_data,
                                      input2_shape, input2_data, output_shape,
                                      output_data);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16/NoActivation");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    output_data[i] = unclamped_result.raw();
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16Uint8");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int32 output_offset = params.output_offset;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16 rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16 clamped_result =
        std::min<int16>(output_activation_max - output_offset, rescaled_result);
    clamped_result =
        std::max<int16>(output_activation_min - output_offset, clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

// Element-wise mul that can often be used for inner loop of broadcast Mul as
// well as the non-broadcast Mul.
inline void MulElementwise(int size, const ArithmeticParams& params,
                           const uint8* input1_data, const uint8* input2_data,
                           uint8* output_data) {
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input1_offset_vector = vdupq_n_s16(params.input1_offset);
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8) {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input1_val_original = vld1_u8(input1_data + i);
    const auto input2_val_original = vld1_u8(input2_data + i);
    const auto input1_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const auto input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const auto input1_val = vaddq_s16(input1_val_s16, input1_offset_vector);
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input1_val_low = vget_low_s16(input1_val);
    const auto input1_val_high = vget_high_s16(input1_val);
    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_s16(input2_val_low, input1_val_low);
    auto p2 = vmull_s16(input2_val_high, input1_val_high);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p =
        vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(p)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
inline void MulSimpleBroadcast(int size, const ArithmeticParams& params,
                               const uint8 broadcast_value,
                               const uint8* input2_data, uint8* output_data) {
  const int16 input1_val = params.input1_offset + broadcast_value;

  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8) {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input2_val_original = vld1_u8(input2_data + i);
    const auto input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_n_s16(input2_val_low, input1_val);
    auto p2 = vmull_n_s16(input2_val_high, input1_val);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vmovn_s32(p1);
    const auto p2_narrowed = vmovn_s32(p2);
    const auto p =
        vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(p)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
// This function will handle scalar_value (LHS) * vector_values (RHS).
// Since it's a float function, input params does not matter here.
inline void MulSimpleBroadcast(int size, const ArithmeticParams& params,
                               const float broadcast_value,
                               const float* input2_data, float* output_data) {
  int i = 0;
#ifdef USE_NEON
  const float32x4_t output_activation_min_vector =
      vdupq_n_f32(params.float_activation_min);
  const float32x4_t output_activation_max_vector =
      vdupq_n_f32(params.float_activation_max);
  const float32x4_t broadcast_value_dup = vdupq_n_f32(broadcast_value);
  for (; i <= size - 4; i += 4) {
    const float32x4_t input2_val_original = vld1q_f32(input2_data + i);

    const float32x4_t output =
        vmulq_f32(input2_val_original, broadcast_value_dup);

    const float32x4_t clamped =
        vmaxq_f32(output_activation_min_vector,
                  vminq_f32(output_activation_max_vector, output));
    vst1q_f32(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    float x = broadcast_value * input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(
        x, params.float_activation_min, params.float_activation_max);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8* input1_data,
                const RuntimeShape& input2_shape, const uint8* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("Mul/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T>
inline void BroadcastMulDispatch(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return BroadcastMul4DSlow(params, input1_shape, input1_data, input2_shape,
                              input2_data, output_shape, output_data);
  }

  BinaryBroadcastFiveFold(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data,
      static_cast<void (*)(int, const ArithmeticParams&, const T*, const T*,
                           T*)>(MulElementwise),
      static_cast<void (*)(int, const ArithmeticParams&, T, const T*, T*)>(
          MulSimpleBroadcast));
}

inline void BroadcastMulFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
  BroadcastMulDispatch(unswitched_params, unswitched_input1_shape,
                       unswitched_input1_data, unswitched_input2_shape,
                       unswitched_input2_data, output_shape, output_data);
}

inline void BroadcastMulFivefold(const ArithmeticParams& params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const float* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const float* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 float* output_data) {
  BroadcastMulDispatch(params, unswitched_input1_shape, unswitched_input1_data,
                       unswitched_input2_shape, unswitched_input2_data,
                       output_shape, output_data);
}

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastDiv is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T, int N = 5>
void BroadcastDivSlow(const ArithmeticParams& params,
                      const RuntimeShape& unextended_input1_shape,
                      const T* input1_data,
                      const RuntimeShape& unextended_input2_shape,
                      const T* input2_data,
                      const RuntimeShape& unextended_output_shape,
                      T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastDivSlow");
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  auto div_func = [&](int indexes[N]) {
    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] /
                input2_data[SubscriptToIndex(desc2, indexes)],
            output_activation_min, output_activation_max);
  };
  NDOpsHelper<N>(output_desc, div_func);
}

// BroadcastDiv is intentionally duplicated from reference_ops.h.
// For more details see the comment above the generic version of
// BroadcastDivSlow.
template <int N = 5>
inline void BroadcastDivSlow(const ArithmeticParams& params,
                             const RuntimeShape& unextended_input1_shape,
                             const uint8* input1_data,
                             const RuntimeShape& unextended_input2_shape,
                             const uint8* input2_data,
                             const RuntimeShape& unextended_output_shape,
                             uint8* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                 &output_desc);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);

  auto div_func = [&](int indexes[N]) {
    int32 input1_val =
        params.input1_offset + input1_data[SubscriptToIndex(desc1, indexes)];
    int32 input2_val =
        params.input2_offset + input2_data[SubscriptToIndex(desc2, indexes)];
    TFLITE_DCHECK_NE(input2_val, 0);
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;
    const int32 input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32 unscaled_quotient = MultiplyByQuantizedMultiplierGreaterThanOne(
        input1_val, input2_inv, headroom);
    const int total_shift = params.output_shift - recip_shift - headroom;
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            unscaled_quotient, params.output_multiplier, total_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[SubscriptToIndex(output_desc, indexes)] =
        static_cast<uint8>(clamped_output);
  };
  NDOpsHelper<N>(output_desc, div_func);
}

template <typename T>
inline void SubWithActivation(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("SubWithActivation_optimized");
  TFLITE_DCHECK_EQ(input1_shape.FlatSize(), input2_shape.FlatSize());
  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  T activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);
  output_map.array() = (input1_map.array() - input2_map.array())
                           .cwiseMin(activation_max)
                           .cwiseMax(activation_min);
}

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const float* input1_data,
                            const RuntimeShape& input2_shape,
                            const float* input2_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  ruy::profiler::ScopeLabel label("SubNonBroadcast");
  SubWithActivation<float>(params, input1_shape, input1_data, input2_shape,
                           input2_data, output_shape, output_data);
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  ruy::profiler::ScopeLabel label("Sub");

  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  if (input1_shape == input2_shape) {
    output_map.array() = input1_map.array() - input2_map.array();
  } else if (input1_shape.FlatSize() == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar - input2_map.array();
  } else if (input2_shape.FlatSize() == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() - scalar;
  } else {
    BroadcastSubSlow(params, input1_shape, input1_data, input2_shape,
                     input2_data, output_shape, output_data);
  }
}

inline void LstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const float* input_data, const RuntimeShape& unextended_prev_activ_shape,
    const float* prev_activ_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& unextended_bias_shape,
    const float* bias_data, const RuntimeShape& unextended_prev_state_shape,
    const float* prev_state_data,
    const RuntimeShape& unextended_output_state_shape, float* output_state_data,
    const RuntimeShape& unextended_output_activ_shape, float* output_activ_data,
    const RuntimeShape& unextended_concat_temp_shape, float* concat_temp_data,
    const RuntimeShape& unextended_activ_temp_shape, float* activ_temp_data,
    CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("LstmCell");
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  const int weights_dim_count = weights_shape.DimensionsCount();
  MatchingDim(  // batches
      input_shape, 0, prev_activ_shape, 0, prev_state_shape, 0,
      output_state_shape, 0, output_activ_shape, 0);
  MatchingDim(  // height
      input_shape, 1, prev_activ_shape, 1, prev_state_shape, 1,
      output_state_shape, 1, output_activ_shape, 1);
  MatchingDim(  // width
      input_shape, 2, prev_activ_shape, 2, prev_state_shape, 2,
      output_state_shape, 2, output_activ_shape, 2);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);

  // Concatenate prev_activ and input data together
  std::vector<float const*> concat_input_arrays_data;
  std::vector<RuntimeShape const*> concat_input_arrays_shapes;
  concat_input_arrays_data.push_back(input_data);
  concat_input_arrays_data.push_back(prev_activ_data);
  concat_input_arrays_shapes.push_back(&input_shape);
  concat_input_arrays_shapes.push_back(&prev_activ_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = concat_input_arrays_data.size();
  Concatenation(concat_params, &(concat_input_arrays_shapes[0]),
                &(concat_input_arrays_data[0]), concat_temp_shape,
                concat_temp_data);

  // Fully connected
  tflite::FullyConnectedParams fc_params;
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  fc_params.lhs_cacheable = false;
  fc_params.rhs_cacheable = false;
  FullyConnected(fc_params, concat_temp_shape, concat_temp_data, weights_shape,
                 weights_data, bias_shape, bias_data, activ_temp_shape,
                 activ_temp_data, cpu_backend_context);

  // Map raw arrays to Eigen arrays so we can use Eigen's optimized array
  // operations.
  ArrayMap<float> activ_temp_map =
      MapAsArrayWithLastDimAsRows(activ_temp_data, activ_temp_shape);
  auto input_gate_sm = activ_temp_map.block(0 * output_depth, 0, output_depth,
                                            activ_temp_map.cols());
  auto new_input_sm = activ_temp_map.block(1 * output_depth, 0, output_depth,
                                           activ_temp_map.cols());
  auto forget_gate_sm = activ_temp_map.block(2 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  auto output_gate_sm = activ_temp_map.block(3 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  ArrayMap<const float> prev_state_map =
      MapAsArrayWithLastDimAsRows(prev_state_data, prev_state_shape);
  ArrayMap<float> output_state_map =
      MapAsArrayWithLastDimAsRows(output_state_data, output_state_shape);
  ArrayMap<float> output_activ_map =
      MapAsArrayWithLastDimAsRows(output_activ_data, output_activ_shape);

  // Combined memory state and final output calculation
  ruy::profiler::ScopeLabel label2("MemoryStateAndFinalOutput");
  output_state_map =
      input_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
          new_input_sm.tanh() +
      forget_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
          prev_state_map;
  output_activ_map =
      output_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
      output_state_map.tanh();
}

template <int StateIntegerBits>
inline void LstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const uint8* input_data_uint8,
    const RuntimeShape& unextended_prev_activ_shape,
    const uint8* prev_activ_data_uint8, const RuntimeShape& weights_shape,
    const uint8* weights_data_uint8, const RuntimeShape& unextended_bias_shape,
    const int32* bias_data_int32,
    const RuntimeShape& unextended_prev_state_shape,
    const int16* prev_state_data_int16,
    const RuntimeShape& unextended_output_state_shape,
    int16* output_state_data_int16,
    const RuntimeShape& unextended_output_activ_shape,
    uint8* output_activ_data_uint8,
    const RuntimeShape& unextended_concat_temp_shape,
    uint8* concat_temp_data_uint8,
    const RuntimeShape& unextended_activ_temp_shape,
    int16* activ_temp_data_int16, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label(
      "LstmCell/quantized (8bit external, 16bit internal)");
  int32 weights_zero_point = params.weights_zero_point;
  int32 accum_multiplier = params.accum_multiplier;
  int accum_shift = params.accum_shift;
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  // Gather dimensions information, and perform consistency checks.
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int outer_size = MatchingFlatSizeSkipDim(
      input_shape, 3, prev_activ_shape, prev_state_shape, output_state_shape,
      output_activ_shape);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);
  const int fc_batches = FlatSizeSkipDim(activ_temp_shape, 3);
  const int fc_output_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, activ_temp_shape, 3);
  const int fc_accum_depth = total_input_depth;
  TFLITE_DCHECK_EQ(fc_output_depth, 4 * output_depth);

  // Depth-concatenate prev_activ and input data together.
  uint8 const* concat_input_arrays_data[2] = {input_data_uint8,
                                              prev_activ_data_uint8};
  const RuntimeShape* concat_input_arrays_shapes[2] = {&input_shape,
                                                       &prev_activ_shape};
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = 2;
  Concatenation(concat_params, concat_input_arrays_shapes,
                concat_input_arrays_data, concat_temp_shape,
                concat_temp_data_uint8);

  // Implementation of the fully connected node inside the LSTM cell.
  // The operands are 8-bit integers, the accumulators are internally 32bit
  // integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = fc_output_depth;
  lhs_params.cols = fc_accum_depth;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = weights_zero_point;
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = fc_accum_depth;
  rhs_params.cols = fc_batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = 128;
  cpu_backend_gemm::MatrixParams<int16> dst_params;
  dst_params.rows = fc_output_depth;
  dst_params.cols = fc_batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = 0;
  cpu_backend_gemm::GemmParams<int32, int16> gemm_params;
  gemm_params.bias = bias_data_int32;
  gemm_params.multiplier_fixedpoint = accum_multiplier;
  gemm_params.multiplier_exponent = accum_shift;
  cpu_backend_gemm::Gemm(
      lhs_params, weights_data_uint8, rhs_params, concat_temp_data_uint8,
      dst_params, activ_temp_data_int16, gemm_params, cpu_backend_context);

  // Rest of the LSTM cell: tanh and logistic math functions, and some adds
  // and muls, all done in 16-bit fixed-point.
  const int16* input_gate_input_ptr = activ_temp_data_int16;
  const int16* input_modulation_gate_input_ptr =
      activ_temp_data_int16 + output_depth;
  const int16* forget_gate_input_ptr = activ_temp_data_int16 + 2 * output_depth;
  const int16* output_gate_input_ptr = activ_temp_data_int16 + 3 * output_depth;
  const int16* prev_state_ptr = prev_state_data_int16;
  int16* output_state_data_ptr = output_state_data_int16;
  uint8* output_activ_data_ptr = output_activ_data_uint8;

  for (int b = 0; b < outer_size; ++b) {
    int c = 0;
#ifdef GEMMLOWP_NEON
    for (; c <= output_depth - 8; c += 8) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<int16x8_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(vld1q_s16(input_gate_input_ptr));
      input_gate_input_ptr += 8;
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input =
          F3::FromRaw(vld1q_s16(input_modulation_gate_input_ptr));
      input_modulation_gate_input_ptr += 8;
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(vld1q_s16(forget_gate_input_ptr));
      forget_gate_input_ptr += 8;
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(vld1q_s16(output_gate_input_ptr));
      output_gate_input_ptr += 8;
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(vld1q_s16(prev_state_ptr));
      prev_state_ptr += 8;
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      vst1q_s16(output_state_data_ptr, new_state.raw());
      output_state_data_ptr += 8;
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16x8_t rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int8x8_t int8_output_activ = vqmovn_s16(rescaled_output_activ);
      uint8x8_t uint8_output_activ =
          vadd_u8(vdup_n_u8(128), vreinterpret_u8_s8(int8_output_activ));
      vst1_u8(output_activ_data_ptr, uint8_output_activ);
      output_activ_data_ptr += 8;
    }
#endif
    for (; c < output_depth; ++c) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<std::int16_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(*input_gate_input_ptr++);
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input =
          F3::FromRaw(*input_modulation_gate_input_ptr++);
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(*forget_gate_input_ptr++);
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(*output_gate_input_ptr++);
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(*prev_state_ptr++);
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      *output_state_data_ptr++ = new_state.raw();
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16 rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int16 clamped_output_activ =
          std::max<int16>(-128, std::min<int16>(127, rescaled_output_activ));
      *output_activ_data_ptr++ = 128 + clamped_output_activ;
    }
    input_gate_input_ptr += 3 * output_depth;
    input_modulation_gate_input_ptr += 3 * output_depth;
    forget_gate_input_ptr += 3 * output_depth;
    output_gate_input_ptr += 3 * output_depth;
  }
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline bool AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("AveragePool");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  if (stride_height == 0) return false;
  if (stride_width == 0) return false;

  // TODO(benoitjacob) make this a proper reference impl without Eigen!
  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // TODO(benoitjacob) get rid of the dynamic memory allocation here!
  Eigen::VectorXf out_count(out_mat.cols());
  out_count.setZero();
  // Prefill the output to 0.
  out_mat.setZero();
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + params.padding_values.height;
        int wpad = w + params.padding_values.width;
        int h_start = (hpad < params.filter_height)
                          ? 0
                          : (hpad - params.filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < params.filter_width)
                          ? 0
                          : (wpad - params.filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) +=
                in_mat.col(NodeOffset(b, h, w, input_height, input_width));
            out_count(out_offset)++;
          }
        }
      }
    }
  }
  // Divide the output by the actual number of elements being averaged over
  TFLITE_DCHECK_GT(out_count.minCoeff(), 0);
  out_mat.array().rowwise() /= out_count.transpose().array();

  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i],
                                                  params.float_activation_min,
                                                  params.float_activation_max);
  }

  return true;
}

inline bool AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const uint8* input_data,
                        const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("AveragePool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  uint32 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          if (filter_count == 0) return false;
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const uint8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const uint8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const uint8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                uint16x4_t acc_reg[4];
                uint8x16_t input_reg = vld1q_u8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg[0] = vget_low_u16(vmovl_u8(vget_low_u8(input_reg)));
                acc_reg[1] = vget_high_u16(vmovl_u8(vget_low_u8(input_reg)));
                acc_reg[2] = vget_low_u16(vmovl_u8(vget_high_u8(input_reg)));
                acc_reg[3] = vget_high_u16(vmovl_u8(vget_high_u8(input_reg)));
                for (int i = 0; i < 4; i++) {
                  vst1q_u32(
                      acc + channel + 4 * i,
                      vaddw_u16(vld1q_u32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
              for (; channel <= tranche_depth - 8; channel += 8) {
                uint16x4_t acc_reg[2];
                uint16x8_t input_reg = vmovl_u8(vld1_u8(input_channel_ptr));
                input_channel_ptr += 8;
                acc_reg[0] = vget_low_u16(input_reg);
                acc_reg[1] = vget_high_u16(input_reg);
                for (int i = 0; i < 2; i++) {
                  vst1q_u32(
                      acc + channel + 4 * i,
                      vaddw_u16(vld1q_u32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] += *input_channel_ptr++;
              }
              input_row_ptr += depth;
            }
          }
          uint8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                   out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
#define AVGPOOL_DIVIDING_BY(FILTER_COUNT)                               \
  if (filter_count == FILTER_COUNT) {                                   \
    for (; channel <= tranche_depth - 8; channel += 8) {                \
      uint16 buf[8];                                                    \
      for (int i = 0; i < 8; i++) {                                     \
        buf[i] = (acc[channel + i] + FILTER_COUNT / 2) / FILTER_COUNT;  \
      }                                                                 \
      uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));                      \
      buf8 = vmin_u8(buf8, vdup_n_u8(params.quantized_activation_max)); \
      buf8 = vmax_u8(buf8, vdup_n_u8(params.quantized_activation_min)); \
      vst1_u8(output_ptr + channel, buf8);                              \
    }                                                                   \
  }
          AVGPOOL_DIVIDING_BY(9)
          AVGPOOL_DIVIDING_BY(15)
#undef AVGPOOL_DIVIDING_BY
          for (; channel <= tranche_depth - 8; channel += 8) {
            uint16 buf[8];
            for (int i = 0; i < 8; i++) {
              buf[i] = (acc[channel + i] + filter_count / 2) / filter_count;
            }
            uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));
            buf8 = vmin_u8(buf8, vdup_n_u8(params.quantized_activation_max));
            buf8 = vmax_u8(buf8, vdup_n_u8(params.quantized_activation_min));
            vst1_u8(output_ptr + channel, buf8);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            uint16 a = (acc[channel] + filter_count / 2) / filter_count;
            a = std::max<uint16>(a, params.quantized_activation_min);
            a = std::min<uint16>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<uint8>(a);
          }
        }
      }
    }
  }
  return true;
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  ruy::profiler::ScopeLabel label("MaxPool");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // Prefill the output to minimum representable float value
  out_mat.setConstant(std::numeric_limits<float>::lowest());
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + params.padding_values.height;
        int wpad = w + params.padding_values.width;
        int h_start = (hpad < params.filter_height)
                          ? 0
                          : (hpad - params.filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < params.filter_width)
                          ? 0
                          : (wpad - params.filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) =
                out_mat.col(out_offset)
                    .cwiseMax(in_mat.col(
                        NodeOffset(b, h, w, input_height, input_width)));
          }
        }
      }
    }
  }
  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i],
                                                  params.float_activation_min,
                                                  params.float_activation_max);
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const uint8* input_data, const RuntimeShape& output_shape,
                    uint8* output_data) {
  ruy::profiler::ScopeLabel label("MaxPool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  uint8 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const uint8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const uint8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const uint8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                uint8x16_t acc_reg = vld1q_u8(acc + channel);
                uint8x16_t input_reg = vld1q_u8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg = vmaxq_u8(acc_reg, input_reg);
                vst1q_u8(acc + channel, acc_reg);
              }

              for (; channel <= tranche_depth - 8; channel += 8) {
                uint8x8_t acc_reg = vld1_u8(acc + channel);
                uint8x8_t input_reg = vld1_u8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vmax_u8(acc_reg, input_reg);
                vst1_u8(acc + channel, acc_reg);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] = std::max(acc[channel], *input_channel_ptr++);
              }
              input_row_ptr += depth;
            }
          }
          uint8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                   out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 16; channel += 16) {
            uint8x16_t a = vld1q_u8(acc + channel);
            a = vminq_u8(a, vdupq_n_u8(params.quantized_activation_max));
            a = vmaxq_u8(a, vdupq_n_u8(params.quantized_activation_min));
            vst1q_u8(output_ptr + channel, a);
          }
          for (; channel <= tranche_depth - 8; channel += 8) {
            uint8x8_t a = vld1_u8(acc + channel);
            a = vmin_u8(a, vdup_n_u8(params.quantized_activation_max));
            a = vmax_u8(a, vdup_n_u8(params.quantized_activation_min));
            vst1_u8(output_ptr + channel, a);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            uint8 a = acc[channel];
            a = std::max<uint8>(a, params.quantized_activation_min);
            a = std::min<uint8>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<uint8>(a);
          }
        }
      }
    }
  }
}

inline void L2Pool(const PoolParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& output_shape,
                   float* output_data) {
  ruy::profiler::ScopeLabel label("L2Pool");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  // Actually carry out L2 Pool. Code is written in forward mode: we go through
  // the input values once, and write to all the pooled regions that it maps to.
  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  Eigen::VectorXf in_square(in_mat.rows());
  Eigen::VectorXf out_count(out_mat.cols());
  out_count.setZero();
  // Prefill the output to 0.
  out_mat.setZero();
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        const int hpad = h + params.padding_values.height;
        const int wpad = w + params.padding_values.width;
        const int h_start =
            (hpad < params.filter_height)
                ? 0
                : (hpad - params.filter_height) / stride_height + 1;
        const int h_end = std::min(hpad / stride_height + 1, output_height);
        const int w_start =
            (wpad < params.filter_width)
                ? 0
                : (wpad - params.filter_width) / stride_width + 1;
        const int w_end = std::min(wpad / stride_width + 1, output_width);
        // pre-compute square
        const int in_offset = w + input_width * (h + input_height * b);
        in_square =
            in_mat.col(in_offset).array() * in_mat.col(in_offset).array();
        // compute elementwise sum of squares
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            const int out_offset = pw + output_width * (ph + output_height * b);
            out_mat.col(out_offset) += in_square;
            out_count(out_offset)++;
          }
        }
      }
    }
  }

  out_count = out_count.array().inverse();
  out_mat =
      (out_mat.array().rowwise() * out_count.transpose().array()).cwiseSqrt();

  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i],
                                                  params.float_activation_min,
                                                  params.float_activation_max);
  }
}

inline void LocalResponseNormalization(
    const tflite::LocalResponseNormalizationParams& op_params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("LocalResponseNormalization");
  MatchingFlatSize(input_shape, output_shape);

  const auto data_in = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto data_out = MapAsMatrixWithLastDimAsRows(output_data, output_shape);

  // Carry out local response normalization, vector by vector.
  // Since the data are stored column major, making row-wise operation
  // probably not memory efficient anyway, we do an explicit for loop over
  // the columns.
  const int double_range = op_params.range * 2;
  Eigen::VectorXf padded_square(data_in.rows() + double_range);
  padded_square.setZero();
  const float bias = op_params.bias;
  for (int r = 0; r < data_in.cols(); ++r) {
    // Do local response normalization for data_in(:, r)
    // first, compute the square and store them in buffer for repeated use
    padded_square.block(op_params.range, 0, data_in.rows(), 1) =
        data_in.col(r).cwiseProduct(data_in.col(r)) * op_params.alpha;
    // Then, compute the scale and writes them to data_out
    float accumulated_scale = 0;
    for (int i = 0; i < double_range; ++i) {
      accumulated_scale += padded_square(i);
    }
    for (int i = 0; i < data_in.rows(); ++i) {
      accumulated_scale += padded_square(i + double_range);
      data_out(i, r) = bias + accumulated_scale;
      accumulated_scale -= padded_square(i);
    }
  }

  // In a few cases, the pow computation could benefit from speedups.
  if (op_params.beta == 1) {
    data_out.array() = data_in.array() * data_out.array().inverse();
  } else if (op_params.beta == 0.5f) {
    data_out.array() = data_in.array() * data_out.array().sqrt().inverse();
  } else {
    data_out.array() = data_in.array() * data_out.array().pow(-op_params.beta);
  }
}

inline void SoftmaxImpl(const SoftmaxParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data,
                        int start_batch, int end_batch) {
  ruy::profiler::ScopeLabel label("Softmax/Impl");
  MatchingFlatSize(input_shape, output_shape);

  const int logit_size = input_shape.Dims(input_shape.DimensionsCount() - 1);
  const MatrixMap<const float> in_mat(input_data + logit_size * start_batch,
                                      logit_size, end_batch - start_batch);
  MatrixMap<float> out_mat(output_data + logit_size * start_batch, logit_size,
                           end_batch - start_batch);
  // Compute the exponential first, removing the max coefficient for numerical
  // stability.
  out_mat =
      (in_mat.rowwise() - in_mat.colwise().maxCoeff()).array() * params.beta;
  // We are separating out the exp function so that exp can be vectorized.
  out_mat = out_mat.array().exp();
  // Normalize to get the activations.
  Eigen::Array<float, 1, Eigen::Dynamic> scale =
      out_mat.array().colwise().sum().inverse();
  out_mat.array().rowwise() *= scale;
}

struct SoftmaxWorkerTask : cpu_backend_threadpool::Task {
  SoftmaxWorkerTask(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& output_shape, float* output_data,
                    int start_batch, int end_batch)
      : params(params),
        input_shape(input_shape),
        input_data(input_data),
        output_shape(output_shape),
        output_data(output_data),
        start_batch(start_batch),
        end_batch(end_batch) {}
  void Run() override {
    SoftmaxImpl(params, input_shape, input_data, output_shape, output_data,
                start_batch, end_batch);
  }

 private:
  const tflite::SoftmaxParams& params;
  const RuntimeShape& input_shape;
  const float* input_data;
  const RuntimeShape& output_shape;
  float* output_data;
  int start_batch;
  int end_batch;
};

inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& output_shape, float* output_data,
                    CpuBackendContext* cpu_backend_context = nullptr) {
  ruy::profiler::ScopeLabel label("Softmax");

  // We picture softmax input as a 2-D matrix while the last dim is the logit
  // dim, and the rest dims will be the batch dim for the 2-D matrix.
  const int batch_size =
      FlatSizeSkipDim(input_shape, input_shape.DimensionsCount() - 1);
  constexpr int kMinBatchPerThread = 8;
  int thread_count = batch_size / kMinBatchPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      cpu_backend_context == nullptr
          ? 1
          : std::min(thread_count, cpu_backend_context->max_num_threads());
  if (capped_thread_count == 1) {
    SoftmaxImpl(params, input_shape, input_data, output_shape, output_data, 0,
                batch_size);
  } else {
    std::vector<SoftmaxWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int batch_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int batch_end =
          batch_start + (batch_size - batch_start) / (capped_thread_count - i);
      tasks.emplace_back(params, input_shape, input_data, output_shape,
                         output_data, batch_start, batch_end);
      batch_start = batch_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

template <typename T>
inline int32_t QuantizeSoftmaxOutput(float prob_rescaled, int32_t zero_point) {
  const int32_t prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
  return prob_rnd + zero_point;
}

#if !__aarch64__
// With ARM64, rounding is faster than add + truncation.
template <>
inline int32_t QuantizeSoftmaxOutput<uint8_t>(float prob_rescaled,
                                              int32_t zero_point) {
  return static_cast<int32_t>(prob_rescaled + 0.5f);
}
#endif

inline void PopulateSoftmaxLookupTable(SoftmaxParams* data, float input_scale,
                                       float beta) {
  const float scale = -input_scale * beta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  for (int32_t val = 0; val <= max_uint8; ++val) {
    data->table[max_uint8 - val] = expf(scale * val);
  }
}

template <typename In, typename Out>
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const In* input_data,
                    const RuntimeShape& output_shape, Out* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int excluding_last_dim =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int last_dim =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t clamp_max = std::numeric_limits<Out>::max();
  const int32_t clamp_min = std::numeric_limits<Out>::min();
  for (int i = 0; i < excluding_last_dim; ++i) {
    int32_t max_val = std::numeric_limits<In>::min();
    // Find max quantized value.
    for (int j = 0; j < last_dim; ++j) {
      max_val = std::max(max_val, static_cast<int32_t>(input_data[j]));
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    const float* table_offset = &params.table[max_uint8 - max_val];
    // Calculate normalizer sum(exp(x)).
    for (int j = 0; j < last_dim; ++j) {
      sum_exp += table_offset[input_data[j]];
    }

    const float inv_sum_exp = 1.0f / (sum_exp * params.scale);
    // Normalize and quantize probabilities.
    for (int j = 0; j < last_dim; ++j) {
      const float prob_rescaled = table_offset[input_data[j]] * inv_sum_exp;
      const int32_t prob_quantized =
          QuantizeSoftmaxOutput<Out>(prob_rescaled, params.zero_point);
      output_data[j] = static_cast<Out>(
          std::max(std::min(clamp_max, prob_quantized), clamp_min));
    }
    input_data += last_dim;
    output_data += last_dim;
  }
}

// Here's the softmax LUT optimization strategy:
// For softmax, we can do some mathmetically equivalent transformation:
//
// softmax(x) = e^x / sum(e^x, 0...n)  ===> equals to
// softmax(x) = e^(x - CONST) / sum(e^(x - CONST), 0...n)
//
// For quantization, `x` in our case is (input_q - input_zp) * input_s
// For uint8 case (int8 can be handled similarly), the range is [0, 255]
//
// so if we let
// CONST = (255 - input_zp) * input_s
// then we will have:
// softmax(x) = e^((input_q - 255) * input_s) --------- (1)
//         /
// sum(e^(input_q - 255) * input_s, 0...n)   -------- (2)
//
// the good thing about (1) is it's within the range of (0, 1), so we can
// approximate its result with uint16.
//  (1) = uint8_out * 1 / 2^16.
//
// so (1) is lookup_uint8_table(input_zp) * 1 / 2^16.
// then (2) is essentially the following:
// sum(lookup_uint8_table(input_zp), 0...n) / 2^16.
//
// since (output_q - output_zp) * output_s = softmax(x)
// output_q = lookup_uint8_table(input_zp)
//            /
// (sum(lookup_uint8_table(input_zp), 0...n) * output_s)
//             +
//   output_zp
//
// We can actually further improve the performance by using uint8 instead of
// uint16. But that we may lose some accuracy, so we need to pay attention
// to that.
inline void PopulateSoftmaxUInt8LookupTable(SoftmaxParams* data,
                                            float input_scale, float beta) {
  const float scale = input_scale * beta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  const int32_t max_uint16 = std::numeric_limits<uint16_t>::max();

  for (int32_t val = 0; val <= max_uint8; ++val) {
    float input_to_exp = scale * (val - max_uint8);
    int32_t temp = static_cast<int>(expf(input_to_exp) * max_uint16 + 0.5);
    temp = std::min(max_uint16, temp);
    uint8_t part1 = temp >> 8;
    uint8_t part2 = temp & 0xff;
    data->uint8_table1[val] = static_cast<uint8_t>(part1);
    data->uint8_table2[val] = static_cast<uint8_t>(part2);
  }
}

inline int FindMaxValue(int size, const uint8_t* input_data, uint8_t offset) {
  int32_t max_val = std::numeric_limits<uint8_t>::min();
  int j = 0;
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  uint8x16_t max_val_dup = vdupq_n_u8(max_val);
  uint8x16_t offset_dup = vdupq_n_u8(offset);
  for (; j <= size - 16; j += 16) {
    uint8x16_t input_value = vld1q_u8(input_data + j);
    input_value = veorq_u8(input_value, offset_dup);
    max_val_dup = vmaxq_u8(input_value, max_val_dup);
  }
  max_val = std::max(max_val, static_cast<int32>(vmaxvq_u8(max_val_dup)));
#endif

  for (; j < size; ++j) {
    max_val = std::max(max_val, static_cast<int32_t>(input_data[j] ^ offset));
  }
  return max_val;
}

#ifdef USE_NEON
// Value_to_store layout:
// [high_high, high_low, low_high, low_low].
inline void StoreValue(int32x4x4_t value_to_store, int8_t* output) {
  const int16x8_t result_1 = vcombine_s16(vqmovn_s32(value_to_store.val[1]),
                                          vqmovn_s32(value_to_store.val[0]));
  const int16x8_t result_2 = vcombine_s16(vqmovn_s32(value_to_store.val[3]),
                                          vqmovn_s32(value_to_store.val[2]));
  const int8x16_t result =
      vcombine_s8(vqmovn_s16(result_2), vqmovn_s16(result_1));
  vst1q_s8(output, result);
}

// Value_to_store layout:
// [high_high, high_low, low_high, low_low].
inline void StoreValue(int32x4x4_t value_to_store, uint8_t* output) {
  const uint16x8_t result_1 =
      vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[1])),
                   vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[0])));
  const uint16x8_t result_2 =
      vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[3])),
                   vqmovn_u32(vreinterpretq_u32_s32(value_to_store.val[2])));
  const uint8x16_t result =
      vcombine_u8(vqmovn_u16(result_2), vqmovn_u16(result_1));
  vst1q_u8(output, result);
}

#endif

template <typename In, typename Out>
inline void SoftmaxInt8LUT(const SoftmaxParams& params,
                           const RuntimeShape& input_shape,
                           const In* input_data,
                           const RuntimeShape& output_shape, Out* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int excluding_last_dim =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int last_dim =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t clamp_max = std::numeric_limits<Out>::max();
  const int32_t clamp_min = std::numeric_limits<Out>::min();

  // Offset is used to interpret the input data "correctly".
  // If the input is uint8, the data will be unchanged.
  // If the input is int8, since it will be reinterpret as uint8.
  // e.g.,
  // int8 127 will be applied "offset" to become 255 in uint8.
  uint8_t offset = 0;
  if (std::is_same<In, int8>::value) {
    offset = 0x80;
  }

  const uint8_t* input_data_uint = reinterpret_cast<const uint8_t*>(input_data);

#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  // This code uses ARM64-only instructions.
  // TODO(b/143709993): Port to ARMv7

  // Load the tables into registers. (4*4 128-bit registers)
  uint8x16x4_t table1[4];
  table1[0] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 0);
  table1[1] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 1);
  table1[2] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 2);
  table1[3] = vld1q_u8_x4(params.uint8_table1 + 16 * 4 * 3);

  uint8x16x4_t table2[4];
  table2[0] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 0);
  table2[1] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 1);
  table2[2] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 2);
  table2[3] = vld1q_u8_x4(params.uint8_table2 + 16 * 4 * 3);
#endif

  for (int i = 0; i < excluding_last_dim; ++i) {
    // Find max quantized value.
    int32_t max_val = FindMaxValue(last_dim, input_data_uint, offset);

    int32 sum_exp = 0;
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    const uint8_t table_offset = max_uint8 - max_val;

    // Calculate normalizer sum(exp(x)).
    int sum_j = 0;
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
    uint8x16_t table_offset_dup = vdupq_n_u8(table_offset);
    uint8x16_t offset_dup = vdupq_n_u8(offset);
    uint32x4_t sum_4 = vdupq_n_u32(0);
    const int multiplier_shift = 8;
    for (; sum_j <= last_dim - 16; sum_j += 16) {
      uint8x16_t input_value = vld1q_u8(input_data_uint + sum_j);
      input_value = veorq_u8(input_value, offset_dup);
      input_value = vaddq_u8(input_value, table_offset_dup);

      const uint8x16_t output1 = aarch64_lookup_vector(table1, input_value);
      const uint8x16_t output2 = aarch64_lookup_vector(table2, input_value);

      uint16x8_t exp_value1 =
          vshll_n_u8(vget_high_u8(output1), multiplier_shift);
      uint16x8_t exp_value2 =
          vshll_n_u8(vget_low_u8(output1), multiplier_shift);

      exp_value1 = vaddw_u8(exp_value1, vget_high_u8(output2));
      exp_value2 = vaddw_u8(exp_value2, vget_low_u8(output2));

      sum_4 = vpadalq_u16(sum_4, exp_value1);
      sum_4 = vpadalq_u16(sum_4, exp_value2);
    }
    int temp = vgetq_lane_u32(sum_4, 0) + vgetq_lane_u32(sum_4, 1) +
               vgetq_lane_u32(sum_4, 2) + vgetq_lane_u32(sum_4, 3);
    sum_exp += temp;

#endif
    for (; sum_j < last_dim; ++sum_j) {
      const uint8_t index = (input_data_uint[sum_j] ^ offset) + table_offset;

      uint8_t part1 = params.uint8_table1[index];
      uint8_t part2 = params.uint8_table2[index];
      sum_exp += ((part1 << 8) + part2);
    }

    const float inv_sum_exp = 1.0f / (sum_exp * params.scale);

    int32 multiplier, shift;
    QuantizeMultiplier(inv_sum_exp, &multiplier, &shift);

    // Normalize and quantize probabilities.
    int j = 0;
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
    const int32x4_t output_zp_dup = vdupq_n_s32(params.zero_point);
    const int32x4_t max_val_dup = vdupq_n_s32(clamp_max);
    const int32x4_t min_val_dup = vdupq_n_s32(clamp_min);

    for (; j <= last_dim - 16; j += 16) {
      uint8x16_t input_value = vld1q_u8(input_data_uint + j);
      input_value = veorq_u8(input_value, offset_dup);
      input_value = vaddq_u8(input_value, table_offset_dup);

      const uint8x16_t output1 = aarch64_lookup_vector(table1, input_value);
      const uint8x16_t output2 = aarch64_lookup_vector(table2, input_value);

      uint16x8_t exp_value1 =
          vshll_n_u8(vget_high_u8(output1), multiplier_shift);
      uint16x8_t exp_value2 =
          vshll_n_u8(vget_low_u8(output1), multiplier_shift);

      exp_value1 = vaddw_u8(exp_value1, vget_high_u8(output2));
      exp_value2 = vaddw_u8(exp_value2, vget_low_u8(output2));

      int32x4x4_t output_value;
      output_value.val[0] =
          vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(exp_value1)));
      output_value.val[1] =
          vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(exp_value1)));
      output_value.val[2] =
          vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(exp_value2)));
      output_value.val[3] =
          vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(exp_value2)));

      int32x4x4_t temp_val =
          MultiplyByQuantizedMultiplier4Rows(output_value, multiplier, shift);

      temp_val.val[0] = vaddq_s32(temp_val.val[0], output_zp_dup);
      temp_val.val[1] = vaddq_s32(temp_val.val[1], output_zp_dup);
      temp_val.val[2] = vaddq_s32(temp_val.val[2], output_zp_dup);
      temp_val.val[3] = vaddq_s32(temp_val.val[3], output_zp_dup);

      temp_val.val[0] =
          vmaxq_s32(vminq_s32(temp_val.val[0], max_val_dup), min_val_dup);
      temp_val.val[1] =
          vmaxq_s32(vminq_s32(temp_val.val[1], max_val_dup), min_val_dup);
      temp_val.val[2] =
          vmaxq_s32(vminq_s32(temp_val.val[2], max_val_dup), min_val_dup);
      temp_val.val[3] =
          vmaxq_s32(vminq_s32(temp_val.val[3], max_val_dup), min_val_dup);

      StoreValue(temp_val, output_data + j);
    }
#endif
    for (; j < last_dim; ++j) {
      const uint8_t index = (input_data_uint[j] ^ offset) + table_offset;
      const uint8_t part1 = params.uint8_table1[index];
      const uint8_t part2 = params.uint8_table2[index];
      const int32_t exp_value = (part1 << 8) + part2;
      const int32_t output_value =
          MultiplyByQuantizedMultiplier(exp_value, multiplier, shift);

      output_data[j] = static_cast<Out>(std::max(
          std::min(clamp_max, output_value + params.zero_point), clamp_min));
    }
    input_data_uint += last_dim;
    output_data += last_dim;
  }
}

inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const float* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("LogSoftmax");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    VectorMap<const float> block_input(input_data + i * depth, depth, 1);
    VectorMap<float> block_output(output_data + i * depth, depth, 1);
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    const float max = block_input.maxCoeff();
    const float log_sum = std::log((block_input.array() - max).exp().sum());
    block_output = block_input.array() - max - log_sum;
  }
}

// Backwards compatibility. Less optimized than below version.
inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const uint8* input_data,
                       const RuntimeShape& output_shape, uint8* output_data) {
  reference_ops::LogSoftmax(params, input_shape, input_data, output_shape,
                            output_data);
}

// Compute LogSoftmax as (x - x_max) - ln(sum(e^(x_i - x_max)...)
// as done in tf.nn.log_softmax to prevent underflow and overflow.
// This is in contrast to just log(softmax(x))
//
// To handle quantization, first dequantize the inputs (from doing
// e^(input scale * val) where we ignore the zero point since it cancels
// out during subtraction due to the ln) and do a rescale at the end to int8.
//
// Notably this makes use of float and is intended as the optimized
// form for quantized execution on CPU. For a fully integer version,
// see the reference op.
//
// TODO(tflite): notes for optimization:
// 1) See if e^ is also bottleneck in the reference fully-integer
// version and apply lookup there and compare.
template <typename T>
inline void LogSoftmax(const SoftmaxParams& params, float input_scale,
                       const RuntimeShape& input_shape, const T* input_data,
                       const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("LogSoftmax");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int excluding_last_dim =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int last_dim =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t clamp_max = std::numeric_limits<T>::max();
  const int32_t clamp_min = std::numeric_limits<T>::min();

  for (int i = 0; i < excluding_last_dim; ++i) {
    T max_val = std::numeric_limits<T>::min();
    // Find max quantized value.
    for (int j = 0; j < last_dim; ++j) {
      max_val = std::max(max_val, input_data[j]);
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8>::max();
    // Offset into table to compute exp(scale*(x - xmax)) instead of
    // exp(scale*(x)) to prevent overflow.
    const float* table_offset = &params.table[max_uint8 - max_val];
    // Calculate sum(exp(scale*(x - x_max))).
    for (int j = 0; j < last_dim; ++j) {
      sum_exp += table_offset[input_data[j]];
    }
    const float log_sum_exp = std::log(sum_exp);

    // params.scale is the output scale.
    const float scale = input_scale / params.scale;
    const float precomputed =
        (input_scale * max_val + log_sum_exp) / params.scale;
    for (int j = 0; j < last_dim; ++j) {
      // Equivalent to (input_scale * (input_data[j] - max_val) - log_sum_exp) /
      // output_scale.
      const float log_prob = scale * input_data[j] - precomputed;

      // TODO(tflite): look into better solution.
      // Use std::rint over std::round (which is used in
      // FakeQuant) since it's multiple times faster on tested arm32.
      const int32_t prob_quantized = std::rint(log_prob) + params.zero_point;
      output_data[j] = static_cast<T>(
          std::max(std::min(clamp_max, prob_quantized), clamp_min));
    }
    input_data += last_dim;
    output_data += last_dim;
  }
}

inline void Logistic(const RuntimeShape& input_shape, const float* input_data,
                     const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Logistic");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() =
      input_map.array().unaryExpr(Eigen::internal::scalar_logistic_op<float>());
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Logistic(const LogisticParams&, const RuntimeShape& input_shape,
                     const float* input_data, const RuntimeShape& output_shape,
                     float* output_data) {
  // Drop params: not needed.
  Logistic(input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const LogisticParams& params,
                     const RuntimeShape& input_shape, const int16* input_data,
                     const RuntimeShape& output_shape, int16* output_data) {
  ruy::profiler::ScopeLabel label("Logistic/Int16");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
  }

  int c = 0;
  const int16* input_data_ptr = input_data;
  int16* output_data_ptr = output_data;
#ifdef GEMMLOWP_NEON
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;

    for (; c <= flat_size - 16; c += 16) {
      F3 input0 = F3::FromRaw(vld1q_s16(input_data_ptr));
      F3 input1 = F3::FromRaw(vld1q_s16(input_data_ptr + 8));
      F0 output0 = gemmlowp::logistic(input0);
      F0 output1 = gemmlowp::logistic(input1);
      vst1q_s16(output_data_ptr, output0.raw());
      vst1q_s16(output_data_ptr + 8, output1.raw());

      input_data_ptr += 16;
      output_data_ptr += 16;
    }
    for (; c <= flat_size - 8; c += 8) {
      F3 input = F3::FromRaw(vld1q_s16(input_data_ptr));
      F0 output = gemmlowp::logistic(input);
      vst1q_s16(output_data_ptr, output.raw());

      input_data_ptr += 8;
      output_data_ptr += 8;
    }
  }
#endif
#ifdef GEMMLOWP_SSE4
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<gemmlowp::int16x8_m128i, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<gemmlowp::int16x8_m128i, 3>;

    for (; c <= flat_size - 16; c += 16) {
      F3 input0 = F3::FromRaw(gemmlowp::to_int16x8_m128i(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_data_ptr))));
      F3 input1 = F3::FromRaw(gemmlowp::to_int16x8_m128i(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(input_data_ptr + 8))));
      F0 output0 = gemmlowp::logistic(input0);
      F0 output1 = gemmlowp::logistic(input1);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr),
                       output0.raw().v);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr + 8),
                       output1.raw().v);
      input_data_ptr += 16;
      output_data_ptr += 16;
    }
    for (; c <= flat_size - 8; c += 8) {
      F3 input = F3::FromRaw(gemmlowp::to_int16x8_m128i(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_data_ptr))));
      F0 output = gemmlowp::logistic(input);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr),
                       output.raw().v);
      input_data_ptr += 8;
      output_data_ptr += 8;
    }
  }
#endif

  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    for (; c < flat_size; ++c) {
      F3 input = F3::FromRaw(*input_data_ptr);
      F0 output = gemmlowp::logistic(input);
      *output_data_ptr = output.raw();

      ++input_data_ptr;
      ++output_data_ptr;
    }
  }
}

inline void Tanh(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Tanh");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = input_map.array().tanh();
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Tanh(const TanhParams&, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& output_shape,
                 float* output_data) {
  // Drop params: not needed.
  Tanh(input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const TanhParams& params, const RuntimeShape& input_shape,
                 const int16* input_data, const RuntimeShape& output_shape,
                 int16* output_data) {
  ruy::profiler::ScopeLabel label("Tanh/Int16");
  const int input_left_shift = params.input_left_shift;
  // Support for shifts is limited until we have a parameterized version of
  // SaturatingRoundingMultiplyByPOT().
  TFLITE_DCHECK_GE(input_left_shift, 0);
  TFLITE_DCHECK_LE(input_left_shift, 1);

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  const int16* input_data_ptr = input_data;
  int16* output_data_ptr = output_data;
#ifdef GEMMLOWP_NEON
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;

    if (input_left_shift == 0) {
      for (; c <= flat_size - 16; c += 16) {
        F3 input0 = F3::FromRaw(vld1q_s16(input_data_ptr));
        F3 input1 = F3::FromRaw(vld1q_s16(input_data_ptr + 8));
        F0 output0 = gemmlowp::tanh(input0);
        F0 output1 = gemmlowp::tanh(input1);
        vst1q_s16(output_data_ptr, output0.raw());
        vst1q_s16(output_data_ptr + 8, output1.raw());

        input_data_ptr += 16;
        output_data_ptr += 16;
      }
      for (; c <= flat_size - 8; c += 8) {
        F3 input = F3::FromRaw(vld1q_s16(input_data_ptr));
        F0 output = gemmlowp::tanh(input);
        vst1q_s16(output_data_ptr, output.raw());

        input_data_ptr += 8;
        output_data_ptr += 8;
      }
    } else {
      for (; c <= flat_size - 16; c += 16) {
        F3 input0 = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            vld1q_s16(input_data_ptr)));
        F3 input1 = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            vld1q_s16(input_data_ptr + 8)));
        F0 output0 = gemmlowp::tanh(input0);
        F0 output1 = gemmlowp::tanh(input1);
        vst1q_s16(output_data_ptr, output0.raw());
        vst1q_s16(output_data_ptr + 8, output1.raw());

        input_data_ptr += 16;
        output_data_ptr += 16;
      }
      for (; c <= flat_size - 8; c += 8) {
        F3 input = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            vld1q_s16(input_data_ptr)));
        F0 output = gemmlowp::tanh(input);
        vst1q_s16(output_data_ptr, output.raw());

        input_data_ptr += 8;
        output_data_ptr += 8;
      }
    }
  }
#endif
#ifdef GEMMLOWP_SSE4
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<gemmlowp::int16x8_m128i, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<gemmlowp::int16x8_m128i, 3>;

    if (input_left_shift == 0) {
      for (; c <= flat_size - 16; c += 16) {
        F3 input0 = F3::FromRaw(gemmlowp::to_int16x8_m128i(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_data_ptr))));
        F3 input1 = F3::FromRaw(gemmlowp::to_int16x8_m128i(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(input_data_ptr + 8))));
        F0 output0 = gemmlowp::tanh(input0);
        F0 output1 = gemmlowp::tanh(input1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr),
                         output0.raw().v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr + 8),
                         output1.raw().v);

        input_data_ptr += 16;
        output_data_ptr += 16;
      }
      for (; c <= flat_size - 8; c += 8) {
        F3 input = F3::FromRaw(gemmlowp::to_int16x8_m128i(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_data_ptr))));
        F0 output = gemmlowp::tanh(input);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr),
                         output.raw().v);
        input_data_ptr += 8;
        output_data_ptr += 8;
      }
    } else {
      for (; c <= flat_size - 16; c += 16) {
        F3 input0 = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            gemmlowp::to_int16x8_m128i(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(input_data_ptr)))));
        F3 input1 = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            gemmlowp::to_int16x8_m128i(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(input_data_ptr + 8)))));
        F0 output0 = gemmlowp::tanh(input0);
        F0 output1 = gemmlowp::tanh(input1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr),
                         output0.raw().v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr + 8),
                         output1.raw().v);

        input_data_ptr += 16;
        output_data_ptr += 16;
      }
      for (; c <= flat_size - 8; c += 8) {
        F3 input = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            gemmlowp::to_int16x8_m128i(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(input_data_ptr)))));
        F0 output = gemmlowp::tanh(input);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_data_ptr),
                         output.raw().v);
        input_data_ptr += 8;
        output_data_ptr += 8;
      }
    }
  }
#endif

  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    if (input_left_shift == 0) {
      for (; c < flat_size; ++c) {
        F3 input = F3::FromRaw(*input_data_ptr);
        F0 output = gemmlowp::tanh(input);
        *output_data_ptr = output.raw();

        ++input_data_ptr;
        ++output_data_ptr;
      }
    } else {
      for (; c < flat_size; ++c) {
        F3 input = F3::FromRaw(
            gemmlowp::SaturatingRoundingMultiplyByPOT<1>(*input_data_ptr));
        F0 output = gemmlowp::tanh(input);
        *output_data_ptr = output.raw();

        ++input_data_ptr;
        ++output_data_ptr;
      }
    }
  }
}

template <typename SrcT, typename DstT>
inline void Cast(const RuntimeShape& input_shape, const SrcT* input_data,
                 const RuntimeShape& output_shape, DstT* output_data) {
  ruy::profiler::ScopeLabel label("Cast");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = input_map.array().template cast<DstT>();
}

inline void Floor(const RuntimeShape& input_shape, const float* input_data,
                  const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Floor");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = Eigen::floor(input_map.array());
}

inline void Ceil(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Ceil");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = Eigen::ceil(input_map.array());
}

// Helper methods for BatchToSpaceND.
// `spatial_index_dim` specifies post-crop offset index in this spatial
// dimension, i.e. spatial offset introduced by flattening batch to spatial
// dimension minus the crop size at beginning. `block_shape_dim` is the block
// size in current dimension. `input_dim` and `output_dim` are input and output
// size of BatchToSpaceND operation in current dimension.
// Output start index is inclusive and end index is exclusive.
inline void GetIndexRange(int spatial_index_dim, int block_shape_dim,
                          int input_dim, int output_dim, int* start_index,
                          int* end_index) {
  // (*start_index) * block_shape_dim is effectively rounded up to the next
  // multiple of block_shape_dim by the integer division.
  *start_index =
      std::max(0, (-spatial_index_dim + block_shape_dim - 1) / block_shape_dim);
  // Similarly, (*end_index) * block_shape_dim is rounded up too (note that
  // end_index is exclusive).
  *end_index = std::min(
      input_dim,
      (output_dim - spatial_index_dim + block_shape_dim - 1) / block_shape_dim);
}

template <typename T>
inline void BatchToSpaceND(
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const int32* block_shape_data,
    const RuntimeShape& unextended_input3_shape, const int32* crops_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BatchToSpaceND");

  TFLITE_DCHECK_GE(unextended_input1_shape.DimensionsCount(), 3);
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(unextended_input1_shape.DimensionsCount(),
                   unextended_output_shape.DimensionsCount());

  // Extends the input/output shape from 3D to 4D if needed, NHC -> NH1C.
  auto extend_shape = [](const RuntimeShape& shape) {
    if (shape.DimensionsCount() == 4) {
      return shape;
    }
    RuntimeShape new_shape(4, 1);
    new_shape.SetDim(0, shape.Dims(0));
    new_shape.SetDim(1, shape.Dims(1));
    new_shape.SetDim(3, shape.Dims(2));
    return new_shape;
  };
  const RuntimeShape input1_shape = extend_shape(unextended_input1_shape);
  const RuntimeShape output_shape = extend_shape(unextended_output_shape);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width =
      unextended_input1_shape.DimensionsCount() == 4 ? block_shape_data[1] : 1;
  const int crops_top = crops_data[0];
  const int crops_left =
      unextended_input1_shape.DimensionsCount() == 4 ? crops_data[2] : 0;

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;

    int in_h_start = 0;
    int in_h_end = 0;
    // GetIndexRange ensures start and end indices are in [0, output_height).
    GetIndexRange(spatial_offset / block_shape_width - crops_top,
                  block_shape_height, input_height, output_height, &in_h_start,
                  &in_h_end);

    for (int in_h = in_h_start; in_h < in_h_end; ++in_h) {
      const int out_h = in_h * block_shape_height +
                        spatial_offset / block_shape_width - crops_top;
      TFLITE_DCHECK_GE(out_h, 0);
      TFLITE_DCHECK_LT(out_h, output_height);

      int in_w_start = 0;
      int in_w_end = 0;
      // GetIndexRange ensures start and end indices are in [0, output_width).
      GetIndexRange(spatial_offset % block_shape_width - crops_left,
                    block_shape_width, input_width, output_width, &in_w_start,
                    &in_w_end);

      for (int in_w = in_w_start; in_w < in_w_end; ++in_w) {
        const int out_w = in_w * block_shape_width +
                          spatial_offset % block_shape_width - crops_left;
        TFLITE_DCHECK_GE(out_w, 0);
        TFLITE_DCHECK_LT(out_w, output_width);
        T* out = output_data + Offset(output_shape, out_batch, out_h, out_w, 0);
        const T* in =
            input1_data + Offset(input1_shape, in_batch, in_h, in_w, 0);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

template <typename T>
void TypedMemset(void* ptr, T value, size_t num) {
  // Optimization for common cases where memset() will suffice.
  if (value == 0 || std::is_same<T, uint8_t>::value) {
    memset(ptr, value, num * sizeof(T));
  } else {
    // Default implementation for cases where memset() will not preserve the
    // bytes, e.g., typically when sizeof(T) > sizeof(uint8_t).
    char* pos = static_cast<char*>(ptr);
    for (size_t i = 0; i < num; ++i) {
      memcpy(pos, &value, sizeof(T));
      pos = pos + sizeof(T);
    }
  }
}

// This makes heavy use of Offset, along with conditional branches. There may be
// opportunities for improvement.
//
// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImpl(const tflite::PadParams& op_params,
                    const RuntimeShape& input_shape, const T* input_data,
                    const P* pad_value_ptr, const RuntimeShape& output_shape,
                    T* output_data) {
  ruy::profiler::ScopeLabel label("PadImpl");
  const int max_supported_dims = 5;
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(max_supported_dims, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(max_supported_dims, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, max_supported_dims);
  TFLITE_DCHECK_LE(op_params.right_padding_count, max_supported_dims);

  // Pad kernels are limited to max 4 dimensions. Copy inputs so we can pad them
  // to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(max_supported_dims, 0);
  const int left_padding_extend =
      max_supported_dims - op_params.left_padding_count;
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[left_padding_extend + i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(max_supported_dims, 0);
  const int right_padding_extend =
      max_supported_dims - op_params.right_padding_count;
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[right_padding_extend + i] = op_params.right_padding[i];
  }

  const int output_batch = ext_output_shape.Dims(0);
  const int output_spatial_dim1 = ext_output_shape.Dims(1);
  const int output_spatial_dim2 = ext_output_shape.Dims(2);
  const int output_spatial_dim3 = ext_output_shape.Dims(3);
  const int output_channel = ext_output_shape.Dims(4);

  const int left_b_padding = left_padding_copy[0];
  const int left_s1_padding = left_padding_copy[1];
  const int left_s2_padding = left_padding_copy[2];
  const int left_s3_padding = left_padding_copy[3];
  const int left_c_padding = left_padding_copy[4];

  const int right_b_padding = right_padding_copy[0];
  const int right_s1_padding = right_padding_copy[1];
  const int right_s2_padding = right_padding_copy[2];
  const int right_s3_padding = right_padding_copy[3];
  const int right_c_padding = right_padding_copy[4];

  const int input_depth = ext_input_shape.Dims(4);
  const T pad_value = *pad_value_ptr;

  if (left_b_padding != 0) {
    TypedMemset<T>(output_data, pad_value,
                   left_b_padding * output_spatial_dim1 * output_spatial_dim2 *
                       output_spatial_dim3 * output_channel);
  }
  for (int out_b = left_b_padding; out_b < output_batch - right_b_padding;
       ++out_b) {
    if (left_s1_padding != 0) {
      TypedMemset<T>(output_data + Offset(ext_output_shape, out_b, 0, 0, 0, 0),
                     pad_value,
                     left_s1_padding * output_spatial_dim2 *
                         output_spatial_dim3 * output_channel);
    }
    for (int out_p = left_s1_padding;
         out_p < output_spatial_dim1 - right_s1_padding; ++out_p) {
      if (left_s2_padding != 0) {
        TypedMemset<T>(
            output_data + Offset(ext_output_shape, out_b, out_p, 0, 0, 0),
            pad_value, left_s2_padding * output_spatial_dim3 * output_channel);
      }
      for (int out_h = left_s2_padding;
           out_h < output_spatial_dim2 - right_s2_padding; ++out_h) {
        if (left_s3_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(ext_output_shape, out_b, out_p, out_h, 0, 0),
              pad_value, left_s3_padding * output_channel);
        }
        for (int out_w = left_s3_padding;
             out_w < output_spatial_dim3 - right_s3_padding; ++out_w) {
          if (left_c_padding != 0) {
            TypedMemset<T>(output_data + Offset(ext_output_shape, out_b, out_p,
                                                out_h, out_w, 0),
                           pad_value, left_c_padding);
          }

          T* out = output_data + Offset(ext_output_shape, out_b, out_p, out_h,
                                        out_w, left_c_padding);
          const T* in = input_data +
                        Offset(ext_input_shape, out_b - left_b_padding,
                               out_p - left_s1_padding, out_h - left_s2_padding,
                               out_w - left_s3_padding, 0);
          memcpy(out, in, input_depth * sizeof(T));

          if (right_c_padding != 0) {
            TypedMemset<T>(
                output_data + Offset(ext_output_shape, out_b, out_p, out_h,
                                     out_w, output_channel - right_c_padding),
                pad_value, right_c_padding);
          }
        }
        if (right_s3_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(ext_output_shape, out_b, out_p, out_h,
                                   output_spatial_dim3 - right_s3_padding, 0),
              pad_value, right_s3_padding * output_channel);
        }
      }
      if (right_s2_padding != 0) {
        TypedMemset<T>(
            output_data + Offset(ext_output_shape, out_b, out_p,
                                 output_spatial_dim2 - right_s2_padding, 0, 0),
            pad_value, right_s2_padding * output_spatial_dim3 * output_channel);
      }
    }
    if (right_s1_padding != 0) {
      TypedMemset<T>(
          output_data + Offset(ext_output_shape, out_b,
                               output_spatial_dim1 - right_s1_padding, 0, 0, 0),
          pad_value,
          right_s1_padding * output_spatial_dim2 * output_spatial_dim3 *
              output_channel);
    }
  }
  if (right_b_padding != 0) {
    TypedMemset<T>(
        output_data + Offset(ext_output_shape, output_batch - right_b_padding,
                             0, 0, 0, 0),
        pad_value,
        right_b_padding * output_spatial_dim1 * output_spatial_dim2 *
            output_spatial_dim3 * output_channel);
  }
}

template <typename T, typename P>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const P* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// The second (pad-value) input can be int32 when, say, the first is uint8.
template <typename T>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  const T converted_pad_value = static_cast<T>(*pad_value_ptr);
  PadImpl(op_params, input_shape, input_data, &converted_pad_value,
          output_shape, output_data);
}

// This version avoids conflicting template matching.
template <>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const int32* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                int32* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// TODO(b/117643175): Optimize. (This is an introductory copy of standard Pad.)
//
// This pad requires that (a) left and right paddings are in the 4D patterns
// {0, h_pad, w_pad, 0}, and (b) memset can be used: *pad_value_ptr == 0 and/or
// T is uint8.
//
// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImageStyleMemset(const tflite::PadParams& op_params,
                                const RuntimeShape& input_shape,
                                const T* input_data, const P* pad_value_ptr,
                                const RuntimeShape& output_shape,
                                T* output_data) {
  ruy::profiler::ScopeLabel label("PadImageStyle");
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(4, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, 4);
  TFLITE_DCHECK_LE(op_params.right_padding_count, 4);

  // Pad kernels are limited to max 4 dimensions. Copy inputs so we can pad them
  // to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(4, 0);
  const int left_padding_extend = 4 - op_params.left_padding_count;
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[left_padding_extend + i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(4, 0);
  const int right_padding_extend = 4 - op_params.right_padding_count;
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[right_padding_extend + i] = op_params.right_padding[i];
  }
  // The following padding restrictions are contractual requirements, and
  // embody what it means for a padding op to be "image-style".
  TFLITE_DCHECK_EQ(left_padding_copy[0], 0);
  TFLITE_DCHECK_EQ(left_padding_copy[3], 0);
  TFLITE_DCHECK_EQ(right_padding_copy[0], 0);
  TFLITE_DCHECK_EQ(right_padding_copy[3], 0);

  const int batch = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int output_height = ext_output_shape.Dims(1);
  const int output_width = ext_output_shape.Dims(2);
  const int input_height = ext_input_shape.Dims(1);
  const int input_width = ext_input_shape.Dims(2);
  const int depth = MatchingDim(ext_input_shape, 3, ext_output_shape, 3);

  const int left_h_padding = left_padding_copy[1];
  const int left_w_padding = left_padding_copy[2];
  const int right_h_padding = right_padding_copy[1];
  const int right_w_padding = right_padding_copy[2];

  TFLITE_DCHECK_EQ(output_height,
                   input_height + left_h_padding + right_h_padding);
  TFLITE_DCHECK_EQ(output_width,
                   input_width + left_w_padding + right_w_padding);

  const T pad_value = *pad_value_ptr;
  const int top_block_size = left_h_padding * output_width * depth;
  const size_t num_top_block_bytes = top_block_size * sizeof(T);
  const int bottom_block_size = right_h_padding * output_width * depth;
  const size_t num_bottom_block_bytes = bottom_block_size * sizeof(T);
  const int left_blocks_size = left_w_padding * depth;
  const size_t num_left_block_bytes = left_blocks_size * sizeof(T);
  const int right_blocks_size = right_w_padding * depth;
  const size_t num_right_block_bytes = right_blocks_size * sizeof(T);
  const int inner_line_size = input_width * depth;
  const size_t num_inner_line_bytes = inner_line_size * sizeof(T);

  if (input_height == 0) {
    memset(output_data, pad_value,
           num_top_block_bytes + num_bottom_block_bytes);
  } else {
    for (int i = 0; i < batch; ++i) {
      // For each image in the batch, apply the top padding, then iterate
      // through rows, then apply the bottom padding.
      //
      // By unwinding one iteration, we can combine the first left-margin
      // padding with the top padding, and the last right-margin padding with
      // the bottom padding.
      memset(output_data, pad_value,
             num_top_block_bytes + num_left_block_bytes);
      output_data += top_block_size + left_blocks_size;
      memcpy(output_data, input_data, num_inner_line_bytes);
      input_data += inner_line_size;
      output_data += inner_line_size;
      // One iteration unwound.
      // Unwinding this loop affords the opportunity to reorder the loop work
      // and hence combine memset() calls.
      //
      // Before unwinding:
      // for (int j = 0; j < input_height; ++j) {
      //   // Pad on left, copy central data, pad on right.
      //   memset(output_data, pad_value, num_left_block_bytes);
      //   output_data += left_blocks_size;
      //   memcpy(output_data, input_data, num_inner_line_bytes);
      //   input_data += inner_line_size;
      //   output_data += inner_line_size;
      //   memset(output_data, pad_value, num_right_block_bytes);
      //   output_data += right_blocks_size;
      // }
      for (int j = 1; j < input_height; ++j) {
        memset(output_data, pad_value,
               num_right_block_bytes + num_left_block_bytes);
        output_data += right_blocks_size + left_blocks_size;
        memcpy(output_data, input_data, num_inner_line_bytes);
        input_data += inner_line_size;
        output_data += inner_line_size;
      }
      memset(output_data, pad_value,
             num_right_block_bytes + num_bottom_block_bytes);
      output_data += right_blocks_size + bottom_block_size;
    }
  }
}

template <typename T, typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const P* pad_value_ptr,
                          const RuntimeShape& output_shape, T* output_data) {
  reference_ops::PadImageStyle(op_params, input_shape, input_data,
                               pad_value_ptr, output_shape, output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const uint8* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          uint8* output_data) {
  PadImageStyleMemset(op_params, input_shape, input_data, pad_value_ptr,
                      output_shape, output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const float* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          float* output_data) {
  const float converted_pad_value = static_cast<float>(*pad_value_ptr);
  if (converted_pad_value == 0.0f) {
    PadImageStyleMemset(op_params, input_shape, input_data, pad_value_ptr,
                        output_shape, output_data);
  } else {
    PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
            output_data);
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape,
                  const RuntimeShape& output_shape,
                  SequentialTensorWriter<T>* writer) {
  ruy::profiler::ScopeLabel label("Slice");
  const RuntimeShape ext_shape = RuntimeShape::ExtendedShape(5, input_shape);
  TFLITE_DCHECK_LE(op_params.begin_count, 5);
  TFLITE_DCHECK_LE(op_params.size_count, 5);
  const int begin_count = op_params.begin_count;
  const int size_count = op_params.size_count;
  // We front-pad the begin and size vectors.
  std::array<int, 5> start;
  std::array<int, 5> stop;
  for (int i = 0; i < 5; ++i) {
    int padded_i = 5 - i;
    start[i] =
        begin_count < padded_i ? 0 : op_params.begin[begin_count - padded_i];
    stop[i] =
        (size_count < padded_i || op_params.size[size_count - padded_i] == -1)
            ? ext_shape.Dims(i)
            : start[i] + op_params.size[size_count - padded_i];
  }

  for (int i0 = start[0]; i0 < stop[0]; ++i0) {
    for (int i1 = start[1]; i1 < stop[1]; ++i1) {
      for (int i2 = start[2]; i2 < stop[2]; ++i2) {
        for (int i3 = start[3]; i3 < stop[3]; ++i3) {
          const int len = stop[4] - start[4];
          if (len > 0)
            writer->WriteN(Offset(ext_shape, i0, i1, i2, i3, start[4]), len);
        }
      }
    }
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  SequentialTensorWriter<T> writer(input_data, output_data);
  return Slice(op_params, input_shape, output_shape, &writer);
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const TfLiteTensor* input,
                  const RuntimeShape& output_shape, TfLiteTensor* output) {
  SequentialTensorWriter<T> writer(input, output);
  return Slice(op_params, input_shape, output_shape, &writer);
}

// Note: This implementation is only optimized for the case where the inner
// stride == 1.
template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const RuntimeShape& unextended_output_shape,
                         SequentialTensorWriter<T>* writer) {
  using strided_slice::LoopCondition;
  using strided_slice::StartForAxis;
  using strided_slice::StopForAxis;

  ruy::profiler::ScopeLabel label("StridedSlice");

  // Note that the output_shape is not used herein.
  tflite::StridedSliceParams params_copy = op_params;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 5);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(5, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(5, unextended_output_shape);

  // Reverse and pad to 5 dimensions because that is what the runtime code
  // requires (ie. all shapes must be 5D and are given backwards).
  strided_slice::StridedSlicePadIndices(&params_copy, 5);

  const int start_0 = StartForAxis(params_copy, input_shape, 0);
  const int stop_0 = StopForAxis(params_copy, input_shape, 0, start_0);
  const int start_1 = StartForAxis(params_copy, input_shape, 1);
  const int stop_1 = StopForAxis(params_copy, input_shape, 1, start_1);
  const int start_2 = StartForAxis(params_copy, input_shape, 2);
  const int stop_2 = StopForAxis(params_copy, input_shape, 2, start_2);
  const int start_3 = StartForAxis(params_copy, input_shape, 3);
  const int stop_3 = StopForAxis(params_copy, input_shape, 3, start_3);
  const int start_4 = StartForAxis(params_copy, input_shape, 4);
  const int stop_4 = StopForAxis(params_copy, input_shape, 4, start_4);
  const bool inner_stride_is_1 = params_copy.strides[4] == 1;

  for (int offset_0 = start_0 * input_shape.Dims(1),
           end_0 = stop_0 * input_shape.Dims(1),
           step_0 = params_copy.strides[0] * input_shape.Dims(1);
       !LoopCondition(offset_0, end_0, params_copy.strides[0]);
       offset_0 += step_0) {
    for (int offset_1 = (offset_0 + start_1) * input_shape.Dims(2),
             end_1 = (offset_0 + stop_1) * input_shape.Dims(2),
             step_1 = params_copy.strides[1] * input_shape.Dims(2);
         !LoopCondition(offset_1, end_1, params_copy.strides[1]);
         offset_1 += step_1) {
      for (int offset_2 = (offset_1 + start_2) * input_shape.Dims(3),
               end_2 = (offset_1 + stop_2) * input_shape.Dims(3),
               step_2 = params_copy.strides[2] * input_shape.Dims(3);
           !LoopCondition(offset_2, end_2, params_copy.strides[2]);
           offset_2 += step_2) {
        for (int offset_3 = (offset_2 + start_3) * input_shape.Dims(4),
                 end_3 = (offset_2 + stop_3) * input_shape.Dims(4),
                 step_3 = params_copy.strides[3] * input_shape.Dims(4);
             !LoopCondition(offset_3, end_3, params_copy.strides[3]);
             offset_3 += step_3) {
          // When the stride is 1, the inner loop is equivalent to the
          // optimized slice inner loop. Otherwise, it is identical to the
          // strided_slice reference implementation inner loop.
          if (inner_stride_is_1) {
            const int len = stop_4 - start_4;
            if (len > 0) {
              writer->WriteN(offset_3 + start_4, len);
            }
          } else {
            for (int offset_4 = offset_3 + start_4, end_4 = offset_3 + stop_4;
                 !LoopCondition(offset_4, end_4, params_copy.strides[4]);
                 offset_4 += params_copy.strides[4]) {
              writer->Write(offset_4);
            }
          }
        }
      }
    }
  }
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  SequentialTensorWriter<T> writer(input_data, output_data);
  StridedSlice<T>(op_params, unextended_input_shape, unextended_output_shape,
                  &writer);
}

template <typename T>
inline void StridedSlice(const tflite::StridedSliceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const TfLiteTensor* input,
                         const RuntimeShape& unextended_output_shape,
                         TfLiteTensor* output) {
  SequentialTensorWriter<T> writer(input, output);
  StridedSlice<T>(op_params, unextended_input_shape, unextended_output_shape,
                  &writer);
}

template <typename T>
void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  ruy::profiler::ScopeLabel label("TensorFlowMinimum");
  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  auto min_value = input2_data[0];
  output_map.array() = input1_map.array().min(min_value);
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Minimum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  ruy::profiler::ScopeLabel label("TensorFlowMaximum");
  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  auto max_value = input2_data[0];
  output_map.array() = input1_map.array().max(max_value);
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Maximum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void TransposeIm2col(const ConvParams& params, uint8 zero_byte,
                     const RuntimeShape& input_shape, const T* input_data,
                     const RuntimeShape& filter_shape,
                     const RuntimeShape& output_shape, T* im2col_data) {
  ruy::profiler::ScopeLabel label("TransposeIm2col");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK(im2col_data);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  MatchingDim(output_shape, 3, filter_shape, 0);  // output_depth

  // Construct the MxN sized im2col matrix.
  // The rows M, are sub-ordered B x H x W
  const RuntimeShape row_shape({1, batches, output_height, output_width});
  // The columns, N, are sub-ordered Kh x Kw x Din
  const RuntimeShape col_shape({1, filter_height, filter_width, input_depth});
  // Use dimensions M and N to construct dims for indexing directly into im2col
  const RuntimeShape im2col_shape(
      {1, 1, row_shape.FlatSize(), col_shape.FlatSize()});

  // Build the im2col matrix by looping through all the input pixels,
  // computing their influence on the output, rather than looping through all
  // the output pixels. We therefore must initialize the im2col array to zero.
  // This is potentially inefficient because we subsequently overwrite bytes
  // set here. However, in practice memset is very fast and costs negligible.
  memset(im2col_data, zero_byte, im2col_shape.FlatSize() * sizeof(T));

  // Loop through the output batches
  for (int batch = 0; batch < batches; ++batch) {
    // Loop through input pixels one at a time.
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        // Loop through the output pixels it will influence
        const int out_x_origin = (in_x * stride_width) - pad_width;
        const int out_y_origin = (in_y * stride_height) - pad_height;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int out_y = out_y_origin + filter_y;
          // Is output pixel within height bounds?
          if ((out_y >= 0) && (out_y < output_height)) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int out_x = out_x_origin + filter_x;
              // Is output pixel within width bounds?
              if ((out_x >= 0) && (out_x < output_width)) {
                // Copy the input elements of this pixel
                T const* src =
                    input_data + Offset(input_shape, batch, in_y, in_x, 0);
                int row_offset = Offset(row_shape, 0, batch, out_y, out_x);
                int col_offset = Offset(col_shape, 0, filter_y, filter_x, 0);
                T* dst = im2col_data +
                         Offset(im2col_shape, 0, 0, row_offset, col_offset);
                memcpy(dst, src, input_depth * sizeof(T));
              }
            }
          }
        }
      }
    }
  }
}

// Returns in 'im_data' (assumes to be zero-initialized) image patch in storage
// order (height, width, depth), constructed from patches in 'col_data', which
// is required to be in storage order (out_height * out_width, filter_height,
// filter_width, in_depth).  Implementation by Yangqing Jia (jiayq).
// Copied from //tensorflow/core/kernels/conv_grad_input_ops.cc
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* im_data) {
  ruy::profiler::ScopeLabel label("Col2im");
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      T* im_patch_data = im_data + (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            // TODO(andydavis) Vectorize this loop (if compiler does not).
            for (int i = 0; i < depth; ++i) {
              im_patch_data[i] += col_data[i];
            }
          }
          im_patch_data += depth;
          col_data += depth;
        }
        // Jump over remaining number of depth.
        im_patch_data += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

// TODO(b/188008864) Optimize this function by combining outer loops.
template <typename T>
void BiasAdd(T* im_data, const T* bias_data, const int batch_size,
             const int height, const int width, const int depth) {
  if (bias_data) {
    for (int n = 0; n < batch_size; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (int d = 0; d < depth; ++d) {
            im_data[d] += bias_data[d];
          }
          im_data += depth;
        }
      }
    }
  }
}

// TransposeConvV2 expect the weights in HWOI order.
inline void TransposeConvV2(
    const ConvParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& hwoi_ordered_filter_shape,
    const float* hwoi_ordered_filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* const output_data, const RuntimeShape& col2im_shape,
    float* col2im_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("TransposeConvV2/float");
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

  const int hwoi_ordered_filter_total_size =
      filter_height * filter_width * output_depth;

  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = hwoi_ordered_filter_total_size;
  lhs_params.cols = input_depth;
  float* output_data_p = output_data;
  std::fill_n(output_data, output_offset * batch_size, 0.0f);
  for (int i = 0; i < batch_size; ++i) {
    cpu_backend_gemm::MatrixParams<float> rhs_params;
    rhs_params.order = cpu_backend_gemm::Order::kColMajor;
    rhs_params.rows = input_depth;
    rhs_params.cols = input_image_size;
    cpu_backend_gemm::MatrixParams<float> dst_params;
    dst_params.order = cpu_backend_gemm::Order::kColMajor;
    dst_params.rows = hwoi_ordered_filter_total_size;
    dst_params.cols = input_image_size;
    cpu_backend_gemm::GemmParams<float, float> gemm_params;
    cpu_backend_gemm::Gemm(lhs_params, hwoi_ordered_filter_data, rhs_params,
                           input_data + input_offset * i, dst_params,
                           col2im_data, gemm_params, cpu_backend_context);

    Col2im(col2im_data, output_depth, output_height, output_width,
           filter_height, filter_width, padding_top, padding_left,
           padding_bottom, padding_right, stride_height, stride_width,
           output_data_p);
    output_data_p += output_offset;
  }
  output_data_p = output_data;
  BiasAdd(output_data_p, bias_data, batch_size, output_height, output_width,
          output_depth);
}

inline void Quantize(int32_t multiplier, int32_t shift, int32_t total_size,
                     int32_t output_zp, int32_t* scratch, uint8_t* output) {
  ruy::profiler::ScopeLabel label("Quantize/uint8");
  int i = 0;
  const int32_t output_min = std::numeric_limits<uint8_t>::min();
  const int32_t output_max = std::numeric_limits<uint8_t>::max();

#ifdef USE_NEON
  const int32x4_t output_zp_dup = vdupq_n_s32(output_zp);
  const int32x4_t max_val_dup = vdupq_n_s32(output_max);
  const int32x4_t min_val_dup = vdupq_n_s32(output_min);

  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;

  for (; i <= total_size - 16; i += 16) {
    int32x4x4_t scratch_val;
    scratch_val.val[0] = vld1q_s32(scratch + i);
    scratch_val.val[1] = vld1q_s32(scratch + i + 4);
    scratch_val.val[2] = vld1q_s32(scratch + i + 8);
    scratch_val.val[3] = vld1q_s32(scratch + i + 12);

    int32x4x4_t temp_val =
        MultiplyByQuantizedMultiplier4Rows(scratch_val, multiplier, shift);

    temp_val.val[0] = vaddq_s32(temp_val.val[0], output_zp_dup);
    temp_val.val[1] = vaddq_s32(temp_val.val[1], output_zp_dup);
    temp_val.val[2] = vaddq_s32(temp_val.val[2], output_zp_dup);
    temp_val.val[3] = vaddq_s32(temp_val.val[3], output_zp_dup);

    temp_val.val[0] =
        vmaxq_s32(vminq_s32(temp_val.val[0], max_val_dup), min_val_dup);
    temp_val.val[1] =
        vmaxq_s32(vminq_s32(temp_val.val[1], max_val_dup), min_val_dup);
    temp_val.val[2] =
        vmaxq_s32(vminq_s32(temp_val.val[2], max_val_dup), min_val_dup);
    temp_val.val[3] =
        vmaxq_s32(vminq_s32(temp_val.val[3], max_val_dup), min_val_dup);

    const uint16x8_t result_1 =
        vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(temp_val.val[0])),
                     vqmovn_u32(vreinterpretq_u32_s32(temp_val.val[1])));
    const uint16x8_t result_2 =
        vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(temp_val.val[2])),
                     vqmovn_u32(vreinterpretq_u32_s32(temp_val.val[3])));
    const uint8x16_t result =
        vcombine_u8(vqmovn_u16(result_1), vqmovn_u16(result_2));
    vst1q_u8(output + i, result);
  }
#endif
  for (; i < total_size; ++i) {
    int32_t temp = MultiplyByQuantizedMultiplier(scratch[i], multiplier, shift);
    temp += output_zp;
    if (temp > output_max) {
      temp = output_max;
    }
    if (temp < output_min) {
      temp = output_min;
    }
    output[i] = static_cast<uint8_t>(temp);
  }
}

// Single-rounding MultiplyByQuantizedMultiplier
#if TFLITE_SINGLE_ROUNDING
inline void Quantize(const int32_t* multiplier, const int32_t* shift,
                     int32_t channel_size, int32_t total_size,
                     int32_t output_zp, int32_t output_min, int32_t output_max,
                     int32_t* scratch, int8_t* output) {
  ruy::profiler::ScopeLabel label("Quantize/int8");

  // Here we're trying to quantize the raw accumulators:
  //        output_channels
  //       data data data data data
  // rows  data data data data data
  //       data data data data data
  //          ....
  //
  // In order to minimize the reload of the multipliers & shifts, once we load
  // the multipliers & shifts, we load & quantize the raw accumulators for every
  // row.
#ifdef USE_NEON
  const int32x4_t output_offset_vec = vdupq_n_s32(output_zp);
  const int32x4_t output_activation_min_vec = vdupq_n_s32(output_min);
  const int32x4_t output_activation_max_vec = vdupq_n_s32(output_max);
  const int32x4_t minus_ones = vdupq_n_s32(-1);
#endif

  TFLITE_DCHECK_EQ(total_size % channel_size, 0);
  const int32_t rows = total_size / channel_size;

  int c = 0;

#ifdef USE_NEON
  for (; c <= channel_size - 8; c += 8) {
    int32x4_t out_shift_1 = vld1q_s32(shift + c);
    int32x4_t out_shift_2 = vld1q_s32(shift + c + 4);

    int32x4_t right_shift_1 = vminq_s32(out_shift_1, minus_ones);
    int32x4_t right_shift_2 = vminq_s32(out_shift_2, minus_ones);

    int32x4_t left_shift_1 = vsubq_s32(out_shift_1, right_shift_1);
    int32x4_t left_shift_2 = vsubq_s32(out_shift_2, right_shift_2);

    int32x4_t out_mul_1 = vld1q_s32(multiplier + c);
    int32x4_t out_mul_2 = vld1q_s32(multiplier + c + 4);
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32x4_t acc_1 = vld1q_s32(scratch + loc);
      int32x4_t acc_2 = vld1q_s32(scratch + loc + 4);

      // Saturating Doubling High Mul.
      acc_1 = vshlq_s32(acc_1, left_shift_1);
      acc_1 = vqdmulhq_s32(acc_1, out_mul_1);
      acc_2 = vshlq_s32(acc_2, left_shift_2);
      acc_2 = vqdmulhq_s32(acc_2, out_mul_2);

      // Rounding Dividing By POT.
      acc_1 = vrshlq_s32(acc_1, right_shift_1);
      acc_2 = vrshlq_s32(acc_2, right_shift_2);

      // Add the output offset.
      acc_1 = vaddq_s32(acc_1, output_offset_vec);
      acc_2 = vaddq_s32(acc_2, output_offset_vec);

      // Apply the activation function.
      acc_1 = vmaxq_s32(acc_1, output_activation_min_vec);
      acc_1 = vminq_s32(acc_1, output_activation_max_vec);
      acc_2 = vmaxq_s32(acc_2, output_activation_min_vec);
      acc_2 = vminq_s32(acc_2, output_activation_max_vec);

      // Saturating cast to int8 and store to destination.
      const int16x4_t acc_s16_1 = vqmovn_s32(acc_1);
      const int16x4_t acc_s16_2 = vqmovn_s32(acc_2);
      const int16x8_t res_s16 = vcombine_s16(acc_s16_1, acc_s16_2);
      const int8x8_t res_s8 = vqmovn_s16(res_s16);
      vst1_s8(output + loc, res_s8);
    }
  }

#endif  // USE_NEON
  // Handle leftover values, one by one. This is very slow.
  for (; c < channel_size; c++) {
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32 acc = scratch[loc];
      acc = MultiplyByQuantizedMultiplier(acc, multiplier[c], shift[c]);
      acc += output_zp;
      acc = std::max(acc, output_min);
      acc = std::min(acc, output_max);
      output[loc] = static_cast<int8>(acc);
    }
  }
}

inline void Quantize(const int32_t* multiplier, const int32_t* shift,
                     int32_t channel_size, int32_t total_size,
                     int32_t output_zp, int32_t output_min, int32_t output_max,
                     int32_t* scratch, int16_t* output) {
  ruy::profiler::ScopeLabel label("Quantize(Single-rounding)/int16");

  // Here we're trying to quantize the raw accumulators:
  //        output_channels
  //       data data data data data
  // rows  data data data data data
  //       data data data data data
  //          ....
  //
  // In order to minimize the reload of the multipliers & shifts, once we load
  // the multipliers & shifts, we load & quantize the raw accumulators for every
  // row.
#ifdef USE_NEON
  const int32x4_t output_offset_vec = vdupq_n_s32(output_zp);
  const int32x4_t output_activation_min_vec = vdupq_n_s32(output_min);
  const int32x4_t output_activation_max_vec = vdupq_n_s32(output_max);
  const int32x4_t minus_ones = vdupq_n_s32(-1);
#endif

  TFLITE_DCHECK_EQ(total_size % channel_size, 0);
  const int32_t rows = total_size / channel_size;

  int c = 0;

#ifdef USE_NEON
  for (; c <= channel_size - 8; c += 8) {
    int32x4_t out_shift_1 = vld1q_s32(shift + c);
    int32x4_t out_shift_2 = vld1q_s32(shift + c + 4);

    int32x4_t right_shift_1 = vminq_s32(out_shift_1, minus_ones);
    int32x4_t right_shift_2 = vminq_s32(out_shift_2, minus_ones);

    int32x4_t left_shift_1 = vsubq_s32(out_shift_1, right_shift_1);
    int32x4_t left_shift_2 = vsubq_s32(out_shift_2, right_shift_2);

    int32x4_t out_mul_1 = vld1q_s32(multiplier + c);
    int32x4_t out_mul_2 = vld1q_s32(multiplier + c + 4);
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32x4_t acc_1 = vld1q_s32(scratch + loc);
      int32x4_t acc_2 = vld1q_s32(scratch + loc + 4);

      // Saturating Doubling High Mul.
      acc_1 = vshlq_s32(acc_1, left_shift_1);
      acc_1 = vqdmulhq_s32(acc_1, out_mul_1);
      acc_2 = vshlq_s32(acc_2, left_shift_2);
      acc_2 = vqdmulhq_s32(acc_2, out_mul_2);

      // Rounding Dividing By POT.
      acc_1 = vrshlq_s32(acc_1, right_shift_1);
      acc_2 = vrshlq_s32(acc_2, right_shift_2);

      // Add the output offset.
      acc_1 = vaddq_s32(acc_1, output_offset_vec);
      acc_2 = vaddq_s32(acc_2, output_offset_vec);

      // Apply the activation function.
      acc_1 = vmaxq_s32(acc_1, output_activation_min_vec);
      acc_1 = vminq_s32(acc_1, output_activation_max_vec);
      acc_2 = vmaxq_s32(acc_2, output_activation_min_vec);
      acc_2 = vminq_s32(acc_2, output_activation_max_vec);

      // Saturating cast to int16 and store to destination.
      const int16x4_t acc_s16_1 = vqmovn_s32(acc_1);
      const int16x4_t acc_s16_2 = vqmovn_s32(acc_2);
      vst1_s16(reinterpret_cast<int16_t*>(output) + loc, acc_s16_1);
      vst1_s16(reinterpret_cast<int16_t*>(output) + loc + 4, acc_s16_2);
    }
  }

#endif  // USE_NEON
  // Handle leftover values, one by one. This is very slow.
  for (; c < channel_size; c++) {
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32 acc = scratch[loc];
      acc = MultiplyByQuantizedMultiplier(acc, multiplier[c], shift[c]);
      acc += output_zp;
      acc = std::max(acc, output_min);
      acc = std::min(acc, output_max);
      output[loc] = static_cast<int16>(acc);
    }
  }
}
// Double-rounding MultiplyByQuantizedMultiplier
#else
inline void Quantize(const int32_t* multiplier, const int32_t* shift,
                     int32_t channel_size, int32_t total_size,
                     int32_t output_zp, int32_t output_min, int32_t output_max,
                     int32_t* scratch, int8_t* output) {
  ruy::profiler::ScopeLabel label("Quantize/int8");

  // Here we're trying to quantize the raw accumulators:
  //        output_channels
  //       data data data data data
  // rows  data data data data data
  //       data data data data data
  //          ....
  //
  // In order to minimize the reload of the multipliers & shifts, once we load
  // the multipliers & shifts, we load & quantize the raw accumulators for every
  // row.
#ifdef USE_NEON
  const int32x4_t output_offset_vec = vdupq_n_s32(output_zp);
  const int32x4_t output_activation_min_vec = vdupq_n_s32(output_min);
  const int32x4_t output_activation_max_vec = vdupq_n_s32(output_max);
  const int32x4_t zeros = vdupq_n_s32(0);
#endif

  TFLITE_DCHECK_EQ(total_size % channel_size, 0);
  const int32_t rows = total_size / channel_size;

  int c = 0;

#ifdef USE_NEON
  using gemmlowp::RoundingDivideByPOT;
  for (; c <= channel_size - 8; c += 8) {
    int32x4_t out_shift_1 = vld1q_s32(shift + c);
    int32x4_t out_shift_2 = vld1q_s32(shift + c + 4);
    int32x4_t left_shift_1 = vmaxq_s32(out_shift_1, zeros);
    int32x4_t left_shift_2 = vmaxq_s32(out_shift_2, zeros);

    // Right shift will be performed as left shift with negative values.
    int32x4_t right_shift_1 = vminq_s32(out_shift_1, zeros);
    int32x4_t right_shift_2 = vminq_s32(out_shift_2, zeros);

    int32x4_t out_mul_1 = vld1q_s32(multiplier + c);
    int32x4_t out_mul_2 = vld1q_s32(multiplier + c + 4);
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32x4_t acc_1 = vld1q_s32(scratch + loc);
      int32x4_t acc_2 = vld1q_s32(scratch + loc + 4);

      // Saturating Rounding Doubling High Mul.
      acc_1 = vshlq_s32(acc_1, left_shift_1);
      acc_1 = vqrdmulhq_s32(acc_1, out_mul_1);
      acc_2 = vshlq_s32(acc_2, left_shift_2);
      acc_2 = vqrdmulhq_s32(acc_2, out_mul_2);

      // Rounding Dividing By POT.
      acc_1 = vrshlq_s32(acc_1, right_shift_1);
      acc_2 = vrshlq_s32(acc_2, right_shift_2);

      // Add the output offset.
      acc_1 = vaddq_s32(acc_1, output_offset_vec);
      acc_2 = vaddq_s32(acc_2, output_offset_vec);

      // Apply the activation function.
      acc_1 = vmaxq_s32(acc_1, output_activation_min_vec);
      acc_1 = vminq_s32(acc_1, output_activation_max_vec);
      acc_2 = vmaxq_s32(acc_2, output_activation_min_vec);
      acc_2 = vminq_s32(acc_2, output_activation_max_vec);

      // Saturating cast to int8 and store to destination.
      const int16x4_t acc_s16_1 = vqmovn_s32(acc_1);
      const int16x4_t acc_s16_2 = vqmovn_s32(acc_2);
      const int16x8_t res_s16 = vcombine_s16(acc_s16_1, acc_s16_2);
      const int8x8_t res_s8 = vqmovn_s16(res_s16);
      vst1_s8(output + loc, res_s8);
    }
  }

#endif  // USE_NEON
  // Handle leftover values, one by one. This is very slow.
  for (; c < channel_size; c++) {
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32 acc = scratch[loc];
      acc = MultiplyByQuantizedMultiplier(acc, multiplier[c], shift[c]);
      acc += output_zp;
      acc = std::max(acc, output_min);
      acc = std::min(acc, output_max);
      output[loc] = static_cast<int8>(acc);
    }
  }
}

inline void Quantize(const int32_t* multiplier, const int32_t* shift,
                     int32_t channel_size, int32_t total_size,
                     int32_t output_zp, int32_t output_min, int32_t output_max,
                     int32_t* scratch, int16_t* output) {
  ruy::profiler::ScopeLabel label("Quantize(Double-rounding)/int16");

  // Here we're trying to quantize the raw accumulators:
  //        output_channels
  //       data data data data data
  // rows  data data data data data
  //       data data data data data
  //          ....
  //
  // In order to minimize the reload of the multipliers & shifts, once we load
  // the multipliers & shifts, we load & quantize the raw accumulators for every
  // row.
#ifdef USE_NEON
  const int32x4_t output_offset_vec = vdupq_n_s32(output_zp);
  const int32x4_t output_activation_min_vec = vdupq_n_s32(output_min);
  const int32x4_t output_activation_max_vec = vdupq_n_s32(output_max);
  const int32x4_t zeros = vdupq_n_s32(0);
#endif

  TFLITE_DCHECK_EQ(total_size % channel_size, 0);
  const int32_t rows = total_size / channel_size;

  int c = 0;

#ifdef USE_NEON
  using gemmlowp::RoundingDivideByPOT;
  for (; c <= channel_size - 8; c += 8) {
    int32x4_t out_shift_1 = vld1q_s32(shift + c);
    int32x4_t out_shift_2 = vld1q_s32(shift + c + 4);
    int32x4_t left_shift_1 = vmaxq_s32(out_shift_1, zeros);
    int32x4_t left_shift_2 = vmaxq_s32(out_shift_2, zeros);

    // Right shift will be performed as left shift with negative values.
    int32x4_t right_shift_1 = vminq_s32(out_shift_1, zeros);
    int32x4_t right_shift_2 = vminq_s32(out_shift_2, zeros);

    int32x4_t out_mul_1 = vld1q_s32(multiplier + c);
    int32x4_t out_mul_2 = vld1q_s32(multiplier + c + 4);
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32x4_t acc_1 = vld1q_s32(scratch + loc);
      int32x4_t acc_2 = vld1q_s32(scratch + loc + 4);

      // Saturating Rounding Doubling High Mul.
      acc_1 = vshlq_s32(acc_1, left_shift_1);
      acc_1 = vqrdmulhq_s32(acc_1, out_mul_1);
      acc_2 = vshlq_s32(acc_2, left_shift_2);
      acc_2 = vqrdmulhq_s32(acc_2, out_mul_2);

      // Rounding Dividing By POT.
      acc_1 = vrshlq_s32(acc_1, right_shift_1);
      acc_2 = vrshlq_s32(acc_2, right_shift_2);

      // Add the output offset.
      acc_1 = vaddq_s32(acc_1, output_offset_vec);
      acc_2 = vaddq_s32(acc_2, output_offset_vec);

      // Apply the activation function.
      acc_1 = vmaxq_s32(acc_1, output_activation_min_vec);
      acc_1 = vminq_s32(acc_1, output_activation_max_vec);
      acc_2 = vmaxq_s32(acc_2, output_activation_min_vec);
      acc_2 = vminq_s32(acc_2, output_activation_max_vec);

      // Saturating cast to int16 and store to destination.
      const int16x4_t acc_s16_1 = vqmovn_s32(acc_1);
      const int16x4_t acc_s16_2 = vqmovn_s32(acc_2);
      vst1_s16(reinterpret_cast<int16_t*>(output) + loc, acc_s16_1);
      vst1_s16(reinterpret_cast<int16_t*>(output) + loc + 4, acc_s16_2);
    }
  }

#endif  // USE_NEON
  // Handle leftover values, one by one. This is very slow.
  for (; c < channel_size; c++) {
    for (int n = 0; n < rows; ++n) {
      int loc = n * channel_size + c;
      int32 acc = scratch[loc];
      acc = MultiplyByQuantizedMultiplier(acc, multiplier[c], shift[c]);
      acc += output_zp;
      acc = std::max(acc, output_min);
      acc = std::min(acc, output_max);
      output[loc] = static_cast<int16>(acc);
    }
  }
}
#endif  // TFLITE_SINGLE_ROUNDING

// TransposeConvV2 expect the weights in HWOI order.
inline void TransposeConvV2(
    const ConvParams& params, const RuntimeShape& input_shape,
    const uint8_t* input_data, const RuntimeShape& hwoi_ordered_filter_shape,
    const uint8_t* hwoi_ordered_filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8_t* output_data, const RuntimeShape& col2im_shape,
    int32_t* col2im_data, int32_t* scratch_data,
    CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("TransposeConvV2/uint8");
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

  const int hwoi_ordered_filter_total_size =
      filter_height * filter_width * output_depth;

  cpu_backend_gemm::MatrixParams<uint8_t> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = hwoi_ordered_filter_total_size;
  lhs_params.cols = input_depth;
  lhs_params.zero_point = -params.weights_offset;

  int32_t* scratch_data_p = scratch_data;
  std::fill_n(scratch_data, output_offset * batch_size, static_cast<int32>(0));
  for (int i = 0; i < batch_size; ++i) {
    cpu_backend_gemm::MatrixParams<uint8_t> rhs_params;
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

    Col2im(col2im_data, output_depth, output_height, output_width,
           filter_height, filter_width, padding_top, padding_left,
           padding_bottom, padding_right, stride_height, stride_width,
           scratch_data_p);

    scratch_data_p += output_offset;
  }
  scratch_data_p = scratch_data;
  BiasAdd(scratch_data_p, bias_data, batch_size, output_height, output_width,
          output_depth);

  Quantize(params.output_multiplier, params.output_shift,
           output_shape.FlatSize(), params.output_offset, scratch_data,
           output_data);
}

// Integer-only version of ResizeNearestNeighbor. Since scales are represented
// in fixed-point and thus approximated, |in_x| or |in_y| may differ from the
// reference version. Debug checks are in place to test if this occurs.
// NOTE: If align_corners or half_pixel_centers is true, we use the reference
// version.
inline void ResizeNearestNeighbor(
    const tflite::ResizeNearestNeighborParams& op_params,
    const RuntimeShape& unextended_input_shape, const uint8* input_data,
    const RuntimeShape& output_size_shape, const int32* output_size_data,
    const RuntimeShape& unextended_output_shape, uint8* output_data) {
  if (op_params.align_corners || op_params.half_pixel_centers) {
    // TODO(b/149823713): Add support for align_corners & half_pixel_centers in
    // this kernel.
    reference_ops::ResizeNearestNeighbor(
        op_params, unextended_input_shape, input_data, output_size_shape,
        output_size_data, unextended_output_shape, output_data);
    return;
  }
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);

  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  // The Tensorflow version of this op allows resize on the width and height
  // axis only.
  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  // Convert scales to fixed-point with 16 fractional bits. We add 1 as an
  // error factor and to avoid zero scales. For example, with input_height = 1,
  // output_height = 3, the float scaling factor would be non-zero at 1/3.
  // With fixed-point, this is zero.
  int32 height_scale = (input_height << 16) / output_height + 1;
  int32 width_scale = (input_width << 16) / output_width + 1;

  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const uint8* input_ptr = input_data;
  uint8* output_ptr = output_data;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32 in_y = std::min((y * height_scale) >> 16, input_height - 1);
      // Check offset calculation is the same as the reference version. See
      // function comment for details. We check using a non-float version of:
      // TFLITE_DCHECK_EQ(in_y, std::floor(y * (static_cast<float>(input_height)
      //                                            / output_height)));
      TFLITE_DCHECK_LT(y * input_height, output_height + in_y * output_height);
      TFLITE_DCHECK_GE(y * input_height, in_y * output_height);
      const uint8* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        int32 in_x = std::min((x * width_scale) >> 16, input_width - 1);
        // Check offset calculation is the same as the reference version. See
        // function comment for details. We check using a non-float version of:
        // TFLITE_DCHECK_EQ(in_y,
        //                  std::floor(y * (static_cast<float>(input_width)
        //                                      / output_width)));
        TFLITE_DCHECK_LT(x * input_width, output_width + in_x * output_width);
        TFLITE_DCHECK_GE(x * input_width, in_x * output_width);
        const uint8* x_input_ptr = y_input_ptr + in_x * col_offset;
        memcpy(output_ptr, x_input_ptr, depth);
        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}

template <typename input_type, typename output_type>
inline void Requantize(const input_type* input_data, int32_t size,
                       int32_t effective_scale_multiplier,
                       int32_t effective_scale_shift, int32_t input_zeropoint,
                       int32_t output_zeropoint, output_type* output_data) {
  reference_ops::Requantize(input_data, size, effective_scale_multiplier,
                            effective_scale_shift, input_zeropoint,
                            output_zeropoint, output_data);
}

template <>
inline void Requantize<int8_t, uint8_t>(const int8_t* input_data, int32_t size,
                                        int32_t effective_scale_multiplier,
                                        int32_t effective_scale_shift,
                                        int32_t input_zeropoint,
                                        int32_t output_zeropoint,
                                        uint8_t* output_data) {
  ruy::profiler::ScopeLabel label("Requantize/Int8ToUint8");

  static constexpr int32_t kMinOutput = std::numeric_limits<uint8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  for (; i <= size - 16; i += 16) {
    const int8x16_t input_vec = vld1q_s8(input_data + i);
    const int16x8_t first_half = vmovl_s8(vget_low_s8(input_vec));
    const int16x8_t second_half = vmovl_s8(vget_high_s8(input_vec));
    int32x4x4_t input;
    input.val[0] = vmovl_s16(vget_low_s16(first_half));
    input.val[1] = vmovl_s16(vget_high_s16(first_half));
    input.val[2] = vmovl_s16(vget_low_s16(second_half));
    input.val[3] = vmovl_s16(vget_high_s16(second_half));
    input.val[0] = vaddq_s32(input.val[0], input_zero_point_dup);
    input.val[1] = vaddq_s32(input.val[1], input_zero_point_dup);
    input.val[2] = vaddq_s32(input.val[2], input_zero_point_dup);
    input.val[3] = vaddq_s32(input.val[3], input_zero_point_dup);

    int32x4x4_t result = MultiplyByQuantizedMultiplier4Rows(
        input, effective_scale_multiplier, effective_scale_shift);

    result.val[0] = vaddq_s32(result.val[0], output_zero_point_dup);
    result.val[1] = vaddq_s32(result.val[1], output_zero_point_dup);
    result.val[2] = vaddq_s32(result.val[2], output_zero_point_dup);
    result.val[3] = vaddq_s32(result.val[3], output_zero_point_dup);
    result.val[0] =
        vmaxq_s32(vminq_s32(result.val[0], max_val_dup), min_val_dup);
    result.val[1] =
        vmaxq_s32(vminq_s32(result.val[1], max_val_dup), min_val_dup);
    result.val[2] =
        vmaxq_s32(vminq_s32(result.val[2], max_val_dup), min_val_dup);
    result.val[3] =
        vmaxq_s32(vminq_s32(result.val[3], max_val_dup), min_val_dup);

    const uint32x4_t result_val_1_unsigned =
        vreinterpretq_u32_s32(result.val[0]);
    const uint32x4_t result_val_2_unsigned =
        vreinterpretq_u32_s32(result.val[1]);
    const uint32x4_t result_val_3_unsigned =
        vreinterpretq_u32_s32(result.val[2]);
    const uint32x4_t result_val_4_unsigned =
        vreinterpretq_u32_s32(result.val[3]);

    const uint16x4_t narrowed_val_1 = vqmovn_u32(result_val_1_unsigned);
    const uint16x4_t narrowed_val_2 = vqmovn_u32(result_val_2_unsigned);
    const uint16x4_t narrowed_val_3 = vqmovn_u32(result_val_3_unsigned);
    const uint16x4_t narrowed_val_4 = vqmovn_u32(result_val_4_unsigned);
    const uint16x8_t output_first_half =
        vcombine_u16(narrowed_val_1, narrowed_val_2);
    const uint16x8_t output_second_half =
        vcombine_u16(narrowed_val_3, narrowed_val_4);
    const uint8x8_t narrowed_first_half = vqmovn_u16(output_first_half);
    const uint8x8_t narrowed_second_half = vqmovn_u16(output_second_half);
    const uint8x16_t narrowed_result =
        vcombine_u8(narrowed_first_half, narrowed_second_half);
    vst1q_u8(output_data + i, narrowed_result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

template <>
inline void Requantize<uint8_t, int8_t>(const uint8_t* input_data, int32_t size,
                                        int32_t effective_scale_multiplier,
                                        int32_t effective_scale_shift,
                                        int32_t input_zeropoint,
                                        int32_t output_zeropoint,
                                        int8_t* output_data) {
  ruy::profiler::ScopeLabel label("Requantize/Uint8ToInt8");

  static constexpr int32_t kMinOutput = std::numeric_limits<int8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  for (; i <= size - 16; i += 16) {
    const uint8x16_t input_vec = vld1q_u8(input_data + i);
    const uint16x8_t first_half = vmovl_u8(vget_low_u8(input_vec));
    const uint16x8_t second_half = vmovl_u8(vget_high_u8(input_vec));
    int32x4x4_t input;
    input.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(first_half)));
    input.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(first_half)));
    input.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(second_half)));
    input.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(second_half)));
    input.val[0] = vaddq_s32(input.val[0], input_zero_point_dup);
    input.val[1] = vaddq_s32(input.val[1], input_zero_point_dup);
    input.val[2] = vaddq_s32(input.val[2], input_zero_point_dup);
    input.val[3] = vaddq_s32(input.val[3], input_zero_point_dup);

    int32x4x4_t result = MultiplyByQuantizedMultiplier4Rows(
        input, effective_scale_multiplier, effective_scale_shift);

    result.val[0] = vaddq_s32(result.val[0], output_zero_point_dup);
    result.val[1] = vaddq_s32(result.val[1], output_zero_point_dup);
    result.val[2] = vaddq_s32(result.val[2], output_zero_point_dup);
    result.val[3] = vaddq_s32(result.val[3], output_zero_point_dup);
    result.val[0] =
        vmaxq_s32(vminq_s32(result.val[0], max_val_dup), min_val_dup);
    result.val[1] =
        vmaxq_s32(vminq_s32(result.val[1], max_val_dup), min_val_dup);
    result.val[2] =
        vmaxq_s32(vminq_s32(result.val[2], max_val_dup), min_val_dup);
    result.val[3] =
        vmaxq_s32(vminq_s32(result.val[3], max_val_dup), min_val_dup);

    const int16x4_t narrowed_val_1 = vqmovn_s32(result.val[0]);
    const int16x4_t narrowed_val_2 = vqmovn_s32(result.val[1]);
    const int16x4_t narrowed_val_3 = vqmovn_s32(result.val[2]);
    const int16x4_t narrowed_val_4 = vqmovn_s32(result.val[3]);
    const int16x8_t output_first_half =
        vcombine_s16(narrowed_val_1, narrowed_val_2);
    const int16x8_t output_second_half =
        vcombine_s16(narrowed_val_3, narrowed_val_4);
    const int8x8_t narrowed_first_half = vqmovn_s16(output_first_half);
    const int8x8_t narrowed_second_half = vqmovn_s16(output_second_half);
    const int8x16_t narrowed_result =
        vcombine_s8(narrowed_first_half, narrowed_second_half);
    vst1q_s8(output_data + i, narrowed_result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

template <>
inline void Requantize<int8_t, int8_t>(const int8_t* input_data, int32_t size,
                                       int32_t effective_scale_multiplier,
                                       int32_t effective_scale_shift,
                                       int32_t input_zeropoint,
                                       int32_t output_zeropoint,
                                       int8_t* output_data) {
  ruy::profiler::ScopeLabel label("Requantize/Int8ToInt8");

  static constexpr int32_t kMinOutput = std::numeric_limits<int8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  for (; i <= size - 16; i += 16) {
    const int8x16_t input_vec = vld1q_s8(input_data + i);
    const int16x8_t first_half = vmovl_s8(vget_low_s8(input_vec));
    const int16x8_t second_half = vmovl_s8(vget_high_s8(input_vec));
    int32x4x4_t input;
    input.val[0] = vmovl_s16(vget_low_s16(first_half));
    input.val[1] = vmovl_s16(vget_high_s16(first_half));
    input.val[2] = vmovl_s16(vget_low_s16(second_half));
    input.val[3] = vmovl_s16(vget_high_s16(second_half));

    input.val[0] = vaddq_s32(input.val[0], input_zero_point_dup);
    input.val[1] = vaddq_s32(input.val[1], input_zero_point_dup);
    input.val[2] = vaddq_s32(input.val[2], input_zero_point_dup);
    input.val[3] = vaddq_s32(input.val[3], input_zero_point_dup);

    int32x4x4_t result = MultiplyByQuantizedMultiplier4Rows(
        input, effective_scale_multiplier, effective_scale_shift);

    result.val[0] = vaddq_s32(result.val[0], output_zero_point_dup);
    result.val[1] = vaddq_s32(result.val[1], output_zero_point_dup);
    result.val[2] = vaddq_s32(result.val[2], output_zero_point_dup);
    result.val[3] = vaddq_s32(result.val[3], output_zero_point_dup);
    result.val[0] =
        vmaxq_s32(vminq_s32(result.val[0], max_val_dup), min_val_dup);
    result.val[1] =
        vmaxq_s32(vminq_s32(result.val[1], max_val_dup), min_val_dup);
    result.val[2] =
        vmaxq_s32(vminq_s32(result.val[2], max_val_dup), min_val_dup);
    result.val[3] =
        vmaxq_s32(vminq_s32(result.val[3], max_val_dup), min_val_dup);

    const int16x4_t narrowed_val_1 = vqmovn_s32(result.val[0]);
    const int16x4_t narrowed_val_2 = vqmovn_s32(result.val[1]);
    const int16x4_t narrowed_val_3 = vqmovn_s32(result.val[2]);
    const int16x4_t narrowed_val_4 = vqmovn_s32(result.val[3]);
    const int16x8_t output_first_half =
        vcombine_s16(narrowed_val_1, narrowed_val_2);
    const int16x8_t output_second_half =
        vcombine_s16(narrowed_val_3, narrowed_val_4);
    const int8x8_t narrowed_first_half = vqmovn_s16(output_first_half);
    const int8x8_t narrowed_second_half = vqmovn_s16(output_second_half);
    const int8x16_t narrowed_result =
        vcombine_s8(narrowed_first_half, narrowed_second_half);
    vst1q_s8(output_data + i, narrowed_result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

template <>
inline void Requantize<uint8_t, uint8_t>(
    const uint8_t* input_data, int32_t size, int32_t effective_scale_multiplier,
    int32_t effective_scale_shift, int32_t input_zeropoint,
    int32_t output_zeropoint, uint8_t* output_data) {
  ruy::profiler::ScopeLabel label("Requantize/Uint8ToUint8");

  static constexpr int32_t kMinOutput = std::numeric_limits<uint8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  for (; i <= size - 16; i += 16) {
    const uint8x16_t input_vec = vld1q_u8(input_data + i);
    const uint16x8_t first_half = vmovl_u8(vget_low_u8(input_vec));
    const uint16x8_t second_half = vmovl_u8(vget_high_u8(input_vec));
    int32x4x4_t input;
    input.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(first_half)));
    input.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(first_half)));
    input.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(second_half)));
    input.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(second_half)));
    input.val[0] = vaddq_s32(input.val[0], input_zero_point_dup);
    input.val[1] = vaddq_s32(input.val[1], input_zero_point_dup);
    input.val[2] = vaddq_s32(input.val[2], input_zero_point_dup);
    input.val[3] = vaddq_s32(input.val[3], input_zero_point_dup);

    int32x4x4_t result = MultiplyByQuantizedMultiplier4Rows(
        input, effective_scale_multiplier, effective_scale_shift);

    result.val[0] = vaddq_s32(result.val[0], output_zero_point_dup);
    result.val[1] = vaddq_s32(result.val[1], output_zero_point_dup);
    result.val[2] = vaddq_s32(result.val[2], output_zero_point_dup);
    result.val[3] = vaddq_s32(result.val[3], output_zero_point_dup);
    result.val[0] =
        vmaxq_s32(vminq_s32(result.val[0], max_val_dup), min_val_dup);
    result.val[1] =
        vmaxq_s32(vminq_s32(result.val[1], max_val_dup), min_val_dup);
    result.val[2] =
        vmaxq_s32(vminq_s32(result.val[2], max_val_dup), min_val_dup);
    result.val[3] =
        vmaxq_s32(vminq_s32(result.val[3], max_val_dup), min_val_dup);

    const uint32x4_t result_val_1_unsigned =
        vreinterpretq_u32_s32(result.val[0]);
    const uint32x4_t result_val_2_unsigned =
        vreinterpretq_u32_s32(result.val[1]);
    const uint32x4_t result_val_3_unsigned =
        vreinterpretq_u32_s32(result.val[2]);
    const uint32x4_t result_val_4_unsigned =
        vreinterpretq_u32_s32(result.val[3]);

    const uint16x4_t narrowed_val_1 = vqmovn_u32(result_val_1_unsigned);
    const uint16x4_t narrowed_val_2 = vqmovn_u32(result_val_2_unsigned);
    const uint16x4_t narrowed_val_3 = vqmovn_u32(result_val_3_unsigned);
    const uint16x4_t narrowed_val_4 = vqmovn_u32(result_val_4_unsigned);
    const uint16x8_t output_first_half =
        vcombine_u16(narrowed_val_1, narrowed_val_2);
    const uint16x8_t output_second_half =
        vcombine_u16(narrowed_val_3, narrowed_val_4);
    const uint8x8_t narrowed_first_half = vqmovn_u16(output_first_half);
    const uint8x8_t narrowed_second_half = vqmovn_u16(output_second_half);
    const uint8x16_t narrowed_result =
        vcombine_u8(narrowed_first_half, narrowed_second_half);
    vst1q_u8(output_data + i, narrowed_result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void HardSwish(const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("HardSwish/Float");
  auto size = MatchingFlatSize(input_shape, output_shape);
  int i = 0;
#ifdef USE_NEON
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t three = vdupq_n_f32(3.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  const float32x4_t one_sixth = vdupq_n_f32(1.0f / 6.0f);

  for (; i <= size - 16; i += 16) {
    // 4x partially unrolled version of the loop below. Refer to its comments.
    const float32x4_t in_0 = vld1q_f32(input_data + i + 0);
    const float32x4_t in_1 = vld1q_f32(input_data + i + 4);
    const float32x4_t in_2 = vld1q_f32(input_data + i + 8);
    const float32x4_t in_3 = vld1q_f32(input_data + i + 12);
    const float32x4_t in_scaled_0 = vmulq_f32(in_0, one_sixth);
    const float32x4_t in_scaled_1 = vmulq_f32(in_1, one_sixth);
    const float32x4_t in_scaled_2 = vmulq_f32(in_2, one_sixth);
    const float32x4_t in_scaled_3 = vmulq_f32(in_3, one_sixth);
    const float32x4_t in_reluish_0 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_0, three)));
    const float32x4_t in_reluish_1 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_1, three)));
    const float32x4_t in_reluish_2 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_2, three)));
    const float32x4_t in_reluish_3 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_3, three)));
    const float32x4_t product_0 = vmulq_f32(in_scaled_0, in_reluish_0);
    const float32x4_t product_1 = vmulq_f32(in_scaled_1, in_reluish_1);
    const float32x4_t product_2 = vmulq_f32(in_scaled_2, in_reluish_2);
    const float32x4_t product_3 = vmulq_f32(in_scaled_3, in_reluish_3);
    vst1q_f32(output_data + i + 0, product_0);
    vst1q_f32(output_data + i + 4, product_1);
    vst1q_f32(output_data + i + 8, product_2);
    vst1q_f32(output_data + i + 12, product_3);
  }
  for (; i <= size - 4; i += 4) {
    // The expression to be computed is:
    //   out = one_sixth * in * min(six, max(zero, (in + three)))
    // We structure the AST to have two roughly balanced, independent branches:
    //  - Multiplication: in_scaled = one_sixth * in.
    //  - Addition and clamping: in_reluish = min(six, max(zero, (in + three))).
    // Then the remaining multiplication at the root of the tree.
    const float32x4_t in = vld1q_f32(input_data + i);
    const float32x4_t in_scaled = vmulq_f32(in, one_sixth);
    const float32x4_t in_reluish =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in, three)));
    const float32x4_t product = vmulq_f32(in_scaled, in_reluish);
    vst1q_f32(output_data + i, product);
  }
#endif
  for (; i < size; i++) {
    const float in = input_data[i];
    output_data[i] =
        in * std::min(6.0f, std::max(0.0f, in + 3.0f)) * (1.0f / 6.0f);
  }
}

#ifdef USE_NEON
inline void SaturateAndStore(int16x8_t src, std::uint8_t* dst) {
  // Narrow values down to 8 bit unsigned, saturating.
  uint8x8_t res8 = vqmovun_s16(src);
  // Store results to destination.
  vst1_u8(dst, res8);
}

inline void SaturateAndStore(int16x8_t src, std::int8_t* dst) {
  // Narrow values down to 8 bit unsigned, saturating.
  int8x8_t res8 = vqmovn_s16(src);
  // Store results to destination.
  vst1_s8(dst, res8);
}
#endif

template <typename T>
inline void HardSwish(const HardSwishParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("HardSwish/Quantized");

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
  // This code heavily uses NEON saturating left shifts (vqshl*) with shift
  // amounts that can be zero, in which case we rely on the correct behavior
  // of a left shift by zero returning just its first operand unmodified.
  // Unfortunately, the Intel arm_neon_sse.h implementation of vqshl* is
  // buggy in the case of zero shift amounts, see b/137199585. That is why
  // this NEON code path is restricted to true ARM NEON, excluding
  // arm_neon_sse.h. Anyway, the arm_neon_sse.h implementation of saturating
  // left shifts is slow scalar code, so there may not be much benefit in
  // running that over just plain reference code.
  //
  // TODO(b/137199585): revisit when this is fixed.
#ifdef __ARM_NEON
  const int16x8_t positive_reluish_multiplier_exponent_minus_one =
      vdupq_n_s16(std::max(0, params.reluish_multiplier_exponent - 1));
  const int16x8_t positive_reluish_multiplier_exponent_last_bit =
      vdupq_n_s16(params.reluish_multiplier_exponent > 0 ? 1 : 0);
  const int16x8_t negative_reluish_multiplier_exponent =
      vdupq_n_s16(std::min(0, params.reluish_multiplier_exponent));
  const int16x8_t constant_32767 = vdupq_n_s16(32767);
  const int16x8_t output_multiplier_exponent =
      vdupq_n_s16(params.output_multiplier_exponent);
  const int16x8_t output_zero_point = vdupq_n_s16(params.output_zero_point);
  // 4x unrolled version of the below NEON loop. Read that first.
  for (; i <= flat_size - 32; i += 32) {
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_value_0_1 =
        Load16AndSubtractZeroPoint(input_data + i, params.input_zero_point);
    const int16x8x2_t input_value_2_3 = Load16AndSubtractZeroPoint(
        input_data + i + 16, params.input_zero_point);
    const int16x8_t input_value_on_hires_input_scale_0 =
        vshlq_n_s16(input_value_0_1.val[0], 7);
    const int16x8_t input_value_on_hires_input_scale_1 =
        vshlq_n_s16(input_value_0_1.val[1], 7);
    const int16x8_t input_value_on_hires_input_scale_2 =
        vshlq_n_s16(input_value_2_3.val[0], 7);
    const int16x8_t input_value_on_hires_input_scale_3 =
        vshlq_n_s16(input_value_2_3.val[1], 7);
    const int16x8_t input_value_on_preshift_output_scale_0 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_0,
                        params.output_multiplier_fixedpoint_int16);
    const int16x8_t input_value_on_preshift_output_scale_1 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_1,
                        params.output_multiplier_fixedpoint_int16);
    const int16x8_t input_value_on_preshift_output_scale_2 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_2,
                        params.output_multiplier_fixedpoint_int16);
    const int16x8_t input_value_on_preshift_output_scale_3 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_3,
                        params.output_multiplier_fixedpoint_int16);
    int16x8_t reluish_value_0 = input_value_on_hires_input_scale_0;
    int16x8_t reluish_value_1 = input_value_on_hires_input_scale_1;
    int16x8_t reluish_value_2 = input_value_on_hires_input_scale_2;
    int16x8_t reluish_value_3 = input_value_on_hires_input_scale_3;
    reluish_value_0 = vqshlq_s16(
        reluish_value_0, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_1 = vqshlq_s16(
        reluish_value_1, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_2 = vqshlq_s16(
        reluish_value_2, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_3 = vqshlq_s16(
        reluish_value_3, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_0 = vqrdmulhq_n_s16(
        reluish_value_0, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_1 = vqrdmulhq_n_s16(
        reluish_value_1, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_2 = vqrdmulhq_n_s16(
        reluish_value_2, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_3 = vqrdmulhq_n_s16(
        reluish_value_3, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_0 = vqshlq_s16(reluish_value_0,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_1 = vqshlq_s16(reluish_value_1,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_2 = vqshlq_s16(reluish_value_2,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_3 = vqshlq_s16(reluish_value_3,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_0 =
        vrshlq_s16(reluish_value_0, negative_reluish_multiplier_exponent);
    reluish_value_1 =
        vrshlq_s16(reluish_value_1, negative_reluish_multiplier_exponent);
    reluish_value_2 =
        vrshlq_s16(reluish_value_2, negative_reluish_multiplier_exponent);
    reluish_value_3 =
        vrshlq_s16(reluish_value_3, negative_reluish_multiplier_exponent);
    reluish_value_0 = vrhaddq_s16(reluish_value_0, constant_32767);
    reluish_value_1 = vrhaddq_s16(reluish_value_1, constant_32767);
    reluish_value_2 = vrhaddq_s16(reluish_value_2, constant_32767);
    reluish_value_3 = vrhaddq_s16(reluish_value_3, constant_32767);
    const int16x8_t preshift_output_value_0 =
        vqdmulhq_s16(reluish_value_0, input_value_on_preshift_output_scale_0);
    const int16x8_t preshift_output_value_1 =
        vqdmulhq_s16(reluish_value_1, input_value_on_preshift_output_scale_1);
    const int16x8_t preshift_output_value_2 =
        vqdmulhq_s16(reluish_value_2, input_value_on_preshift_output_scale_2);
    const int16x8_t preshift_output_value_3 =
        vqdmulhq_s16(reluish_value_3, input_value_on_preshift_output_scale_3);
    int16x8_t output_value_0 =
        vrshlq_s16(preshift_output_value_0, output_multiplier_exponent);
    int16x8_t output_value_1 =
        vrshlq_s16(preshift_output_value_1, output_multiplier_exponent);
    int16x8_t output_value_2 =
        vrshlq_s16(preshift_output_value_2, output_multiplier_exponent);
    int16x8_t output_value_3 =
        vrshlq_s16(preshift_output_value_3, output_multiplier_exponent);
    output_value_0 = vaddq_s16(output_value_0, output_zero_point);
    output_value_1 = vaddq_s16(output_value_1, output_zero_point);
    output_value_2 = vaddq_s16(output_value_2, output_zero_point);
    output_value_3 = vaddq_s16(output_value_3, output_zero_point);
    SaturateAndStore(output_value_0, output_data + i);
    SaturateAndStore(output_value_1, output_data + i + 8);
    SaturateAndStore(output_value_2, output_data + i + 16);
    SaturateAndStore(output_value_3, output_data + i + 24);
  }
  // NEON version of reference_ops::HardSwish. Read that first.
  for (; i <= flat_size - 8; i += 8) {
    using cpu_backend_gemm::detail::Load8AndSubtractZeroPoint;
    const int16x8_t input_value =
        Load8AndSubtractZeroPoint(input_data + i, params.input_zero_point);
    const int16x8_t input_value_on_hires_input_scale =
        vshlq_n_s16(input_value, 7);
    const int16x8_t input_value_on_preshift_output_scale =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale,
                        params.output_multiplier_fixedpoint_int16);
    int16x8_t reluish_value = input_value_on_hires_input_scale;
    reluish_value = vqshlq_s16(reluish_value,
                               positive_reluish_multiplier_exponent_minus_one);
    reluish_value = vqrdmulhq_n_s16(reluish_value,
                                    params.reluish_multiplier_fixedpoint_int16);
    reluish_value = vqshlq_s16(reluish_value,
                               positive_reluish_multiplier_exponent_last_bit);
    reluish_value =
        vrshlq_s16(reluish_value, negative_reluish_multiplier_exponent);
    reluish_value = vrhaddq_s16(reluish_value, constant_32767);
    const int16x8_t preshift_output_value =
        vqdmulhq_s16(reluish_value, input_value_on_preshift_output_scale);
    int16x8_t output_value =
        vrshlq_s16(preshift_output_value, output_multiplier_exponent);
    output_value = vaddq_s16(output_value, output_zero_point);
    SaturateAndStore(output_value, output_data + i);
  }
#endif
  // TODO(b/137208495): revisit when unit tests cover reference code.
  // Fall back to reference_ops::HardSwish. In general we have preferred
  // to duplicate such scalar code rather than call reference code to handle
  // leftovers, thinking that code duplication was not a big concern.
  // However, most of our unit tests happen to test only optimized code,
  // and the quantized HardSwish implementation is nontrivial enough that
  // I really want test coverage for the reference code.
  if (i < flat_size) {
    const RuntimeShape leftover_shape{flat_size - i};
    reference_ops::HardSwish(params, leftover_shape, input_data + i,
                             leftover_shape, output_data + i);
  }
}

template <typename T>
inline void IntegerExponentPow(const ArithmeticParams& params,
                               const RuntimeShape& unextended_base_shape,
                               const T* base_data, const int exponent,
                               const RuntimeShape& unextended_output_shape,
                               T* output_data) {
  TFLITE_DCHECK_GE(exponent, 1);
  if (exponent == 1) {
    // copy data over.
    std::memcpy(output_data, base_data,
                unextended_base_shape.FlatSize() * sizeof(T));
  } else {
    IntegerExponentPow(params, unextended_base_shape, base_data, exponent / 2,
                       unextended_output_shape, output_data);
    Mul(params, unextended_base_shape, output_data, unextended_base_shape,
        output_data, unextended_output_shape, output_data);
    if (exponent % 2 == 1) {
      Mul(params, unextended_base_shape, base_data, unextended_base_shape,
          output_data, unextended_output_shape, output_data);
    }
  }
}

template <typename T>
inline void BroadcastPow4D(const RuntimeShape& unextended_input1_shape,
                           const T* input1_data,
                           const RuntimeShape& unextended_input2_shape,
                           const T* input2_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  ruy::profiler::ScopeLabel label("PowBroadcast");

  if (unextended_input2_shape.FlatSize() == 1) {
    static const float epsilon = 1e-5;
    const T exponent = input2_data[0];
    const int int_exponent = static_cast<int>(std::round(exponent));
    if ((std::abs(input2_data[0] - int_exponent) < epsilon) &&
        (int_exponent >= 1)) {
      ArithmeticParams params;
      if (std::is_same<T, float>::value) {
        params.float_activation_max = std::numeric_limits<float>::max();
        params.float_activation_min = std::numeric_limits<float>::lowest();
      } else if (std::is_same<T, int>::value) {
        params.quantized_activation_max = std::numeric_limits<int>::max();
        params.quantized_activation_min = std::numeric_limits<int>::lowest();
      }
      IntegerExponentPow(params, unextended_input1_shape, input1_data,
                         int_exponent, unextended_output_shape, output_data);
      return;
    }
  }
  reference_ops::BroadcastPow4DSlow(unextended_input1_shape, input1_data,
                                    unextended_input2_shape, input2_data,
                                    unextended_output_shape, output_data);
}

#ifdef USE_NEON

inline void ScaleWithNewZeroPoint(const int32x4_t input,
                                  const float32x4_t scale_dup,
                                  const float32x4_t zero_times_scale_dup,
                                  float32x4_t* output) {
#ifdef __ARM_FEATURE_FMA
  *output = vfmaq_f32(zero_times_scale_dup, vcvtq_f32_s32(input), scale_dup);
#else
  *output = vaddq_f32(vmulq_f32(vcvtq_f32_s32(input), scale_dup),
                      zero_times_scale_dup);
#endif
}

#endif  // USE_NEON

inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const uint8_t* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Dequantize/Uint8");
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup =
      vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8) {
    const uint8x8_t input_u8 = vld1_u8(input_data + i);
    const uint16x8_t input_u16 = vmovl_u8(input_u8);
    const int16x8_t input_s16 = vreinterpretq_s16_u16(input_u16);
    const int16x4_t input_s16_low = vget_low_s16(input_s16);
    const int16x4_t input_s16_high = vget_high_s16(input_s16);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup,
                          &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup,
                          &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif  // NEON
  for (; i < flat_size; ++i) {
    const int32 val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Dequantize/Int8");
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup =
      vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8) {
    const int8x8_t input_s8 = vld1_s8(input_data + i);
    const int16x8_t input_s16 = vmovl_s8(input_s8);
    const int16x4_t input_s16_low = vget_low_s16(input_s16);
    const int16x4_t input_s16_high = vget_high_s16(input_s16);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup,
                          &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup,
                          &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif  // NEON
  for (; i < flat_size; ++i) {
    const int32 val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const int16_t* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Dequantize/Int16");
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup =
      vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8) {
    const int16x4_t input_s16_low = vld1_s16(input_data + i);
    const int16x4_t input_s16_high = vld1_s16(input_data + i + 4);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup,
                          &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup,
                          &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif  // NEON
  for (; i < flat_size; ++i) {
    const int32 val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void Dequantize(const RuntimeShape& input_shape,
                       const Eigen::half* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  reference_ops::Dequantize(input_shape, input_data, output_shape, output_data);
}

template <typename T>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape, T* output_data) {
  reference_ops::AffineQuantize(op_params, input_shape, input_data,
                                output_shape, output_data);
}

template <>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape,
                           int8_t* output_data) {
  ruy::profiler::ScopeLabel label("Quantize/Int8");
  const int32 zero_point = op_params.zero_point;
  const double scale = static_cast<double>(op_params.scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<int8_t>::min();
  static constexpr int32 max_val = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scale);
  const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
  const int32x4_t min_val_dup = vdupq_n_s32(min_val);
  const int32x4_t max_val_dup = vdupq_n_s32(max_val);

  for (; i <= flat_size - 8; i += 8) {
    const float* src_data_ptr = input_data + i;
    float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
    float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

    input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
    input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

    int32x4_t casted_val_0 = RoundToNearest(input_val_0);
    int32x4_t casted_val_1 = RoundToNearest(input_val_1);

    casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
    casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

    // Clamp the values to fit the target type's range.
    casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
    casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
    casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
    casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

    const int16x4_t narrowed_val_0 = vmovn_s32(casted_val_0);
    const int16x4_t narrowed_val_1 = vmovn_s32(casted_val_1);
    const int16x8_t combined_val = vcombine_s16(narrowed_val_0, narrowed_val_1);
    const int8x8_t combined_val_narrowed = vmovn_s16(combined_val);
    vst1_s8(output_data + i, combined_val_narrowed);
  }
#endif  // NEON

  for (; i < flat_size; ++i) {
    const float val = input_data[i];
    const int32 unclamped =
        static_cast<int32>(TfLiteRound(val / scale)) + zero_point;
    const int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

template <>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape,
                           uint8_t* output_data) {
  ruy::profiler::ScopeLabel label("Quantize/Uint8");
  const int32 zero_point = op_params.zero_point;
  const double scale = static_cast<double>(op_params.scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<uint8_t>::min();
  static constexpr int32 max_val = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scale);
  const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
  const int32x4_t min_val_dup = vdupq_n_s32(min_val);
  const int32x4_t max_val_dup = vdupq_n_s32(max_val);

  for (; i <= flat_size - 8; i += 8) {
    const float* src_data_ptr = input_data + i;
    float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
    float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

    input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
    input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

    int32x4_t casted_val_0 = RoundToNearest(input_val_0);
    int32x4_t casted_val_1 = RoundToNearest(input_val_1);

    casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
    casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

    // Clamp the values to fit the target type's range.
    casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
    casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
    casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
    casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

    const uint16x4_t narrowed_val_0 = vqmovun_s32(casted_val_0);
    const uint16x4_t narrowed_val_1 = vqmovun_s32(casted_val_1);
    const uint16x8_t combined_val =
        vcombine_u16(narrowed_val_0, narrowed_val_1);
    const uint8x8_t combined_val_narrowed = vmovn_u16(combined_val);
    vst1_u8(output_data + i, combined_val_narrowed);
  }
#endif  // NEON

  for (; i < flat_size; ++i) {
    const float val = input_data[i];
    const int32 unclamped =
        static_cast<int32>(TfLiteRound(val / scale)) + zero_point;
    const int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

template <>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape,
                           int16_t* output_data) {
  ruy::profiler::ScopeLabel label("Quantize/Int16");
  const int32 zero_point = op_params.zero_point;
  const double scale = static_cast<double>(op_params.scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<int16_t>::min();
  static constexpr int32 max_val = std::numeric_limits<int16_t>::max();

  int i = 0;
#ifdef USE_NEON
  const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scale);
  const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
  const int32x4_t min_val_dup = vdupq_n_s32(min_val);
  const int32x4_t max_val_dup = vdupq_n_s32(max_val);

  for (; i <= flat_size - 8; i += 8) {
    const float* src_data_ptr = input_data + i;
    float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
    float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

    input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
    input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

    int32x4_t casted_val_0 = RoundToNearest(input_val_0);
    int32x4_t casted_val_1 = RoundToNearest(input_val_1);

    casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
    casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

    // Clamp the values to fit the target type's range.
    casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
    casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
    casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
    casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

    const int16x4_t narrowed_val_0 = vmovn_s32(casted_val_0);
    const int16x4_t narrowed_val_1 = vmovn_s32(casted_val_1);
    vst1_s16(output_data + i, narrowed_val_0);
    vst1_s16(output_data + i + 4, narrowed_val_1);
  }
#endif  // NEON

  for (; i < flat_size; ++i) {
    const float val = input_data[i];
    const int32 unclamped =
        static_cast<int32>(TfLiteRound(val / scale)) + zero_point;
    const int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON

inline int16x8x4_t SaturatingRounding(
    int16x8_t input_val_0, int16x8_t input_val_1, int16x8_t input_val_2,
    int16x8_t input_val_3, int input_left_shift, int input_multiplier) {
  // This performs what is expressed in the scalar code as
  // const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
  //      static_cast<int16>(input_val_centered * (1 << input_left_shift)),
  //      static_cast<int16>(input_multiplier));
  const int16x8_t left_shift_dup = vdupq_n_s16(input_left_shift);
  const int16x8_t input_val_shifted_0 = vshlq_s16(input_val_0, left_shift_dup);
  const int16x8_t input_val_shifted_1 = vshlq_s16(input_val_1, left_shift_dup);
  const int16x8_t input_val_shifted_2 = vshlq_s16(input_val_2, left_shift_dup);
  const int16x8_t input_val_shifted_3 = vshlq_s16(input_val_3, left_shift_dup);
  int16x8x4_t result;
  result.val[0] = vqrdmulhq_n_s16(input_val_shifted_0, input_multiplier);
  result.val[1] = vqrdmulhq_n_s16(input_val_shifted_1, input_multiplier);
  result.val[2] = vqrdmulhq_n_s16(input_val_shifted_2, input_multiplier);
  result.val[3] = vqrdmulhq_n_s16(input_val_shifted_3, input_multiplier);
  return result;
}

// 4-bit fixed point is enough for tanh since tanh(16) is almost same with one,
// considering 7 digits under zero.
inline int16x8x4_t FixedPoint4Logistic(int16x8x4_t input_val) {
  // Invoke gemmlowp::logistic on FixedPoint wrapping int16x8_t
  using FixedPoint4 = gemmlowp::FixedPoint<int16x8_t, 4>;
  using FixedPoint0 = gemmlowp::FixedPoint<int16x8_t, 0>;
  const FixedPoint4 input_val_f4_0 = FixedPoint4::FromRaw(input_val.val[0]);
  const FixedPoint4 input_val_f4_1 = FixedPoint4::FromRaw(input_val.val[1]);
  const FixedPoint4 input_val_f4_2 = FixedPoint4::FromRaw(input_val.val[2]);
  const FixedPoint4 input_val_f4_3 = FixedPoint4::FromRaw(input_val.val[3]);

  // TODO(b/134622898) Implement a low accuracy version of logistic. In this
  // method, gemmlowp::tanh spends about 80% of the execution times. The
  // current implementation is rougly 12-bit accurate in the 16-bit fixed
  // point case. Until reaching to error bounds, there are rooms for
  // improvements.
  const FixedPoint0 output_val_f0_0 = gemmlowp::logistic(input_val_f4_0);
  const FixedPoint0 output_val_f0_1 = gemmlowp::logistic(input_val_f4_1);
  const FixedPoint0 output_val_f0_2 = gemmlowp::logistic(input_val_f4_2);
  const FixedPoint0 output_val_f0_3 = gemmlowp::logistic(input_val_f4_3);

  // Divide by 2^7 as in the scalar code
  int16x8x4_t result;
  result.val[0] = vrshrq_n_s16(output_val_f0_0.raw(), 7);
  result.val[1] = vrshrq_n_s16(output_val_f0_1.raw(), 7);
  result.val[2] = vrshrq_n_s16(output_val_f0_2.raw(), 7);
  result.val[3] = vrshrq_n_s16(output_val_f0_3.raw(), 7);
  return result;
}

// 4-bit fixed point is enough for tanh since tanh(16) is almost same with one,
// considering 11 digits under zero at least.
inline int16x8x4_t FixedPoint4Tanh(int16x8x4_t input_val) {
  // Invoke gemmlowp::logistic on FixedPoint wrapping int16x8_t
  using FixedPoint4 = gemmlowp::FixedPoint<int16x8_t, 4>;
  using FixedPoint0 = gemmlowp::FixedPoint<int16x8_t, 0>;
  const FixedPoint4 input_val_f4_0 = FixedPoint4::FromRaw(input_val.val[0]);
  const FixedPoint4 input_val_f4_1 = FixedPoint4::FromRaw(input_val.val[1]);
  const FixedPoint4 input_val_f4_2 = FixedPoint4::FromRaw(input_val.val[2]);
  const FixedPoint4 input_val_f4_3 = FixedPoint4::FromRaw(input_val.val[3]);

  // TODO(b/134622898) Implement a low accuracy version of logistic. In this
  // method, gemmlowp::tanh spends about 80% of the execution times. The
  // current implementation is rougly 12-bit accurate in the 16-bit fixed
  // point case. Until reaching to error bounds, there are rooms for
  // improvements.
  const FixedPoint0 output_val_f0_0 = gemmlowp::tanh(input_val_f4_0);
  const FixedPoint0 output_val_f0_1 = gemmlowp::tanh(input_val_f4_1);
  const FixedPoint0 output_val_f0_2 = gemmlowp::tanh(input_val_f4_2);
  const FixedPoint0 output_val_f0_3 = gemmlowp::tanh(input_val_f4_3);

  // Divide by 2^7 as in the scalar code
  int16x8x4_t result;
  result.val[0] = vrshrq_n_s16(output_val_f0_0.raw(), 8);
  result.val[1] = vrshrq_n_s16(output_val_f0_1.raw(), 8);
  result.val[2] = vrshrq_n_s16(output_val_f0_2.raw(), 8);
  result.val[3] = vrshrq_n_s16(output_val_f0_3.raw(), 8);
  return result;
}

inline uint8x16x2_t CalculateUnsignedClampingWithRangeBitMasks(
    int16x8x2_t input_val, int16x8_t range_radius_dup,
    int16x8_t neg_range_radius_dup) {
  const uint16x8_t mask_rightclamp_0 =
      vcgtq_s16(input_val.val[0], range_radius_dup);
  const uint16x8_t mask_rightclamp_1 =
      vcgtq_s16(input_val.val[1], range_radius_dup);

  const uint16x8_t mask_leftclamp_0 =
      vcgeq_s16(input_val.val[0], neg_range_radius_dup);
  const uint16x8_t mask_leftclamp_1 =
      vcgeq_s16(input_val.val[1], neg_range_radius_dup);

  uint8x16x2_t result;
  result.val[0] = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                              vshrn_n_u16(mask_leftclamp_1, 8));
  result.val[1] = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                              vshrn_n_u16(mask_rightclamp_1, 8));
  return result;
}

inline uint8x16x2_t CalculateSignedClampingWithRangeBitMasks(
    int16x8x2_t input_val, int16x8_t range_radius_dup,
    int16x8_t neg_range_radius_dup) {
  const uint16x8_t mask_rightclamp_0 =
      vcgtq_s16(input_val.val[0], range_radius_dup);
  const uint16x8_t mask_rightclamp_1 =
      vcgtq_s16(input_val.val[1], range_radius_dup);

  const uint16x8_t mask_leftclamp_0 =
      vcltq_s16(input_val.val[0], neg_range_radius_dup);
  const uint16x8_t mask_leftclamp_1 =
      vcltq_s16(input_val.val[1], neg_range_radius_dup);

  uint8x16x2_t result;
  result.val[0] = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                              vshrn_n_u16(mask_leftclamp_1, 8));
  result.val[1] = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                              vshrn_n_u16(mask_rightclamp_1, 8));
  return result;
}

inline void ClampWithRangeAndStore(uint8_t* output_dst, uint8x16_t input_val,
                                   uint8x16x2_t masks_clamp) {
  // Store back to memory
  vst1q_u8(output_dst, vandq_u8(vorrq_u8(input_val, masks_clamp.val[1]),
                                masks_clamp.val[0]));
}

inline void ClampWithRangeAndStore(int8_t* output_dst, int8x16_t input_val,
                                   uint8x16x2_t masks_clamp) {
  static const int8x16_t max_dup = vdupq_n_s8(127);
  static const int8x16_t min_dup = vdupq_n_s8(-128);
  // Store back to memory
  vst1q_s8(output_dst,
           vbslq_s8(masks_clamp.val[1], max_dup,
                    vbslq_s8(masks_clamp.val[0], min_dup, input_val)));
}

#endif  // GEMMLOWP_NEON

inline void Tanh16bitPrecision(const TanhParams& params,
                               const RuntimeShape& input_shape,
                               const uint8* input_data,
                               const RuntimeShape& output_shape,
                               uint8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  ruy::profiler::ScopeLabel label("Tanh/Uint8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int16 input_multiplier = static_cast<int16>(params.input_multiplier);
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  int16_t output_zero_point = 128;

// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);
  const int16x8_t output_zero_point_s16 = vdupq_n_s16(output_zero_point);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Tanh(input_val_rescaled);

    // Add the output zero point
    output_val_s16.val[0] =
        vaddq_s16(output_val_s16.val[0], output_zero_point_s16);
    output_val_s16.val[1] =
        vaddq_s16(output_val_s16.val[1], output_zero_point_s16);
    output_val_s16.val[2] =
        vaddq_s16(output_val_s16.val[2], output_zero_point_s16);
    output_val_s16.val[3] =
        vaddq_s16(output_val_s16.val[3], output_zero_point_s16);

    // Cast output values to uint8, saturating
    uint8x16_t output_val_u8_0_1 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[0]), vqmovun_s16(output_val_s16.val[1]));
    uint8x16_t output_val_u8_2_3 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[2]), vqmovun_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_u8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_u8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 8);
      output_val_s16 += output_zero_point;
      if (output_val_s16 == 256) {
        output_val_s16 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s16, 0);
      TFLITE_DCHECK_LE(output_val_s16, 255);
      output_val = static_cast<uint8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Tanh16bitPrecision(const TanhParams& params,
                               const RuntimeShape& input_shape,
                               const int8* input_data,
                               const RuntimeShape& output_shape,
                               int8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  ruy::profiler::ScopeLabel label("Tanh/Int8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int16 input_multiplier = static_cast<int16>(params.input_multiplier);
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input int8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = -128;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 127;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Tanh(input_val_rescaled);

    // Cast output values to uint8, saturating
    int8x16_t output_val_s8_0_1 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[0]), vqmovn_s16(output_val_s16.val[1]));
    int8x16_t output_val_s8_2_3 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[2]), vqmovn_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_s8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_s8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const int8 input_val_s8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_s8) - input_zero_point;
    int8 output_val;
    if (input_val_centered <= -input_range_radius) {
      output_val = -128;
    } else if (input_val_centered >= input_range_radius) {
      output_val = 127;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 8);
      if (output_val_s16 == 128) {
        output_val_s16 = 127;
      }
      TFLITE_DCHECK_GE(output_val_s16, -128);
      TFLITE_DCHECK_LE(output_val_s16, 127);
      output_val = static_cast<int8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Logistic16bitPrecision(const LogisticParams& params,
                                   const RuntimeShape& input_shape,
                                   const uint8* input_data,
                                   const RuntimeShape& output_shape,
                                   uint8* output_data) {
  ruy::profiler::ScopeLabel label("Logistic/Uint8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Logistic(input_val_rescaled);

    // Cast output values to uint8, saturating
    uint8x16_t output_val_u8_0_1 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[0]), vqmovun_s16(output_val_s16.val[1]));
    uint8x16_t output_val_u8_2_3 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[2]), vqmovun_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_u8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_u8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 7);
      if (output_val_s16 == 256) {
        output_val_s16 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s16, 0);
      TFLITE_DCHECK_LE(output_val_s16, 255);
      output_val = static_cast<uint8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Logistic16bitPrecision(const LogisticParams& params,
                                   const RuntimeShape& input_shape,
                                   const int8* input_data,
                                   const RuntimeShape& output_shape,
                                   int8* output_data) {
  ruy::profiler::ScopeLabel label("Logistic/Int8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  const int16 output_zero_point = 128;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);
  const int16x8_t output_zero_point_dup = vdupq_n_s16(output_zero_point);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input int8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = -128;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 127;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Logistic(input_val_rescaled);

    // Substract output zero point.
    output_val_s16.val[0] =
        vsubq_s16(output_val_s16.val[0], output_zero_point_dup);
    output_val_s16.val[1] =
        vsubq_s16(output_val_s16.val[1], output_zero_point_dup);
    output_val_s16.val[2] =
        vsubq_s16(output_val_s16.val[2], output_zero_point_dup);
    output_val_s16.val[3] =
        vsubq_s16(output_val_s16.val[3], output_zero_point_dup);

    // Cast output values to int8, saturating
    int8x16_t output_val_s8_0_1 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[0]), vqmovn_s16(output_val_s16.val[1]));
    int8x16_t output_val_s8_2_3 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[2]), vqmovn_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_s8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_s8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const int8 input_val_s8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_s8) - input_zero_point;
    int8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = -128;
    } else if (input_val_centered > input_range_radius) {
      output_val = 127;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 7);
      output_val_s16 -= output_zero_point;
      if (output_val_s16 == 128) {
        output_val_s16 = 127;
      }
      TFLITE_DCHECK_GE(output_val_s16, -128);
      TFLITE_DCHECK_LE(output_val_s16, 127);
      output_val = static_cast<int8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

// Transpose2D only deals with typical 2D matrix transpose ops.
// Perform transpose by transposing 4x4 blocks of the input, proceeding from
// left to right (down the rows) of the input, and then from top to bottom.
template <typename T>
inline void Transpose2D(const RuntimeShape& input_shape, const T* input_data,
                        const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  const int d0 = input_shape.DimsData()[0];
  const int d1 = input_shape.DimsData()[1];
  const int kLines = 4;
  const int kSkipSize = (kLines - 1) * d1;

  const T* input = input_data;

  int i = 0;
  for (; i <= d0 - kLines; i += kLines) {
    T* output = output_data + i;

    const T* input_ptr = input;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);

    int j = 0;
    for (; j <= d1 - kLines; j += kLines) {
      input_ptr = input;
      const T a00 = input_ptr[0];
      const T a01 = input_ptr[1];
      const T a02 = input_ptr[2];
      const T a03 = input_ptr[3];
      input_ptr += d1;
      const T a10 = input_ptr[0];
      const T a11 = input_ptr[1];
      const T a12 = input_ptr[2];
      const T a13 = input_ptr[3];
      input_ptr += d1;
      const T a20 = input_ptr[0];
      const T a21 = input_ptr[1];
      const T a22 = input_ptr[2];
      const T a23 = input_ptr[3];
      input_ptr += d1;
      const T a30 = input_ptr[0];
      const T a31 = input_ptr[1];
      const T a32 = input_ptr[2];
      const T a33 = input_ptr[3];

      output[0] = a00;
      output[1] = a10;
      output[2] = a20;
      output[3] = a30;
      output += d0;

      output[0] = a01;
      output[1] = a11;
      output[2] = a21;
      output[3] = a31;
      output += d0;

      output[0] = a02;
      output[1] = a12;
      output[2] = a22;
      output[3] = a32;
      output += d0;

      output[0] = a03;
      output[1] = a13;
      output[2] = a23;
      output[3] = a33;
      output += d0;

      input += kLines;
    }
    if (j == d1) {
      input += kSkipSize;
    } else {
      for (int p = 0; p < kLines; ++p) {
        for (int q = 0; q < d1 - j; ++q) {
          *(output + q * d0 + p) = *(input + p * d1 + q);
        }
      }
      input += (d1 - j) + kSkipSize;
    }
  }
  for (; i < d0; ++i) {
    T* output = output_data + i;
    for (int j = 0; j < d1; ++j) {
      *output = *input;
      output += d0;
      ++input;
    }
  }
}

template <>
inline void Transpose2D(const RuntimeShape& input_shape,
                        const int32_t* input_data,
                        const RuntimeShape& output_shape,
                        int32_t* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  const int d0 = input_shape.DimsData()[0];
  const int d1 = input_shape.DimsData()[1];
#ifdef USE_NEON
  const int kLines = 4;
  const int kSkipSize = (kLines - 1) * d1;
#endif

  const int32_t* input = input_data;

  int i = 0;
#ifdef USE_NEON
  for (; i <= d0 - kLines; i += kLines) {
    int32_t* output = output_data + i;

    const int32_t* input_ptr = input;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);
    input_ptr += d1;
    optimized_ops_preload_l1_keep(input_ptr);

    int j = 0;
    for (; j <= d1 - kLines; j += kLines) {
      input_ptr = input;
      int32x4_t a0 = vld1q_s32(input);
      input_ptr += d1;
      int32x4_t a1 = vld1q_s32(input_ptr);
      input_ptr += d1;
      int32x4_t a2 = vld1q_s32(input_ptr);
      input_ptr += d1;
      int32x4_t a3 = vld1q_s32(input_ptr);

      int32x4x2_t tmp1 = vuzpq_s32(a0, a2);
      int32x4x2_t tmp2 = vuzpq_s32(a1, a3);
      int32x4x2_t tmp3 = vtrnq_s32(tmp1.val[0], tmp2.val[0]);
      int32x4x2_t tmp4 = vtrnq_s32(tmp1.val[1], tmp2.val[1]);

      vst1q_s32(output, tmp3.val[0]);
      output += d0;
      vst1q_s32(output, tmp4.val[0]);
      output += d0;
      vst1q_s32(output, tmp3.val[1]);
      output += d0;
      vst1q_s32(output, tmp4.val[1]);
      output += d0;
      input += kLines;
    }
    if (j == d1) {
      input += kSkipSize;
    } else {
      for (int p = 0; p < kLines; ++p) {
        for (int q = 0; q < d1 - j; ++q) {
          *(output + q * d0 + p) = *(input + p * d1 + q);
        }
      }
      input += (d1 - j) + kSkipSize;
    }
  }
#endif
  for (; i < d0; ++i) {
    int32_t* output = output_data + i;
    for (int j = 0; j < d1; ++j) {
      *output = *input;
      output += d0;
      ++input;
    }
  }
}

// TODO(b/173718660): see if we can reduce the number
// of lines of code in branching without affecting latency.
template <typename T>
inline void Transpose3D(const TransposeParams& params,
                        const RuntimeShape& input_shape, const T* input_data,
                        const RuntimeShape& output_shape, T* output_data) {
  int s1, s2, s3;
  s1 = input_shape.Dims(0);
  s2 = input_shape.Dims(1);
  s3 = input_shape.Dims(2);

  int p1, p2, p3;
  if (params.perm[0] == 2) {
    p1 = 1;
  } else if (params.perm[1] == 2) {
    p2 = 1;
  } else {
    p3 = 1;
  }

  if (params.perm[0] == 1) {
    p1 = s3;
  } else if (params.perm[1] == 1) {
    p2 = s3;
  } else {
    p3 = s3;
  }

  if (params.perm[0] == 0) {
    p1 = s2 * s3;
  } else if (params.perm[1] == 0) {
    p2 = s2 * s3;
  } else {
    p3 = s2 * s3;
  }

  int o_s[3];
  o_s[0] = input_shape.Dims(params.perm[0]);
  o_s[1] = input_shape.Dims(params.perm[1]);
  o_s[2] = input_shape.Dims(params.perm[2]);

  for (int i1 = 0; i1 < o_s[0]; ++i1) {
    for (int i2 = 0; i2 < o_s[1]; ++i2) {
      for (int i3 = 0; i3 < o_s[2]; ++i3) {
        const int i = i1 * p1 + i2 * p2 + i3 * p3;
        const int o = i1 * o_s[1] * o_s[2] + i2 * o_s[2] + i3;
        output_data[o] = input_data[i];
      }
    }
  }
}

template <typename T, int N>
void TransposeImpl(const TransposeParams& params,
                   const RuntimeShape& input_shape, const T* input_data,
                   const RuntimeShape& output_shape, T* output_data) {
  const int dims_cnt = input_shape.DimensionsCount();

  int dim0, dim1;
  if (transpose_utils::IsTranspose2DApplicable(params, input_shape, &dim0,
                                               &dim1)) {
    Transpose2D(RuntimeShape({dim0, dim1}), input_data,
                RuntimeShape({dim1, dim0}), output_data);
    return;
  }

  // TODO(b/141217325): notably Eigen is better suited for
  // larger inputs whereas Transpose3D is generally
  // better for smaller ones.
  //
  // E.g. on Nexus 5, Eigen is better for size 96^3 and up
  // and Transpose3D is better for 72^3 and down.
  //
  // 96^3 is not mobile-friendly for certain usecases
  // (e.g. model used in beam search for seq2seq) but is in others.
  // Consider tradeoffs.
  if (dims_cnt == 3) {
    Transpose3D(params, input_shape, input_data, output_shape, output_data);
    return;
  }

  // Reroute to the reference version if an optimized method for the given data
  // is not available.
  reference_ops::Transpose<T, N>(params, input_shape, input_data, output_shape,
                                 output_data);
}

template <typename T, int N = 5>
void Transpose(const TransposeParams& unshrinked_params,
               const RuntimeShape& unshrinked_input_shape, const T* input_data,
               const RuntimeShape& unshrinked_output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Transpose");

  const int output_size = unshrinked_output_shape.DimensionsCount();
  TFLITE_DCHECK_LE(unshrinked_input_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_size, N);
  TFLITE_DCHECK_EQ(output_size, unshrinked_params.perm_count);

  RuntimeShape shrinked_input_shape = RuntimeShape(unshrinked_input_shape);
  RuntimeShape shrinked_output_shape = RuntimeShape(unshrinked_output_shape);
  TransposeParams shrinked_params = unshrinked_params;

  // Reduce any dimensions that have one size. Lower transpose op usually
  // performs better since memory access patterns will be improved.
  transpose_utils::RemoveOneSizeDimensions(
      &shrinked_input_shape, &shrinked_output_shape, &shrinked_params);

  // Handle identity cases.
  // TODO(b/140779653): Add an optimization pass in the conversion process to
  // remove transpose op nodes where they do nothing like the below one.
  bool identical = true;
  for (int i = 0; i < shrinked_params.perm_count; ++i) {
    if (shrinked_params.perm[i] != i) {
      identical = false;
      break;
    }
  }
  if (identical) {
    memcpy(output_data, input_data,
           unshrinked_input_shape.FlatSize() * sizeof(T));
    return;
  }

  // Reduce dimensions by flattening.
  if (shrinked_params.perm[0] == 0 && output_size >= 3) {
    RuntimeShape non_flatten_input_shape;
    RuntimeShape non_flatten_output_shape;
    TransposeParams non_flatten_params;
    const int total_size = shrinked_input_shape.FlatSize();
    const int non_flatten_size = transpose_utils::Flatten(
        shrinked_input_shape, shrinked_output_shape, shrinked_params,
        &non_flatten_input_shape, &non_flatten_output_shape,
        &non_flatten_params);
    TFLITE_DCHECK_NE(non_flatten_params.perm[0], 0);

    for (int i = 0; i < total_size; i += non_flatten_size) {
      TransposeImpl<T, N>(non_flatten_params, non_flatten_input_shape,
                          input_data + i, non_flatten_output_shape,
                          output_data + i);
    }
    return;
  }

  // Call non-flattened case.
  TransposeImpl<T, N>(shrinked_params, shrinked_input_shape, input_data,
                      shrinked_output_shape, output_data);
}

// Assume input1 & input2 have the same scale & zero point.
inline void MaximumElementwise(int size, const ArithmeticParams& params,
                               const int8* input1_data, const int8* input2_data,
                               int8* output_data) {
  ruy::profiler::ScopeLabel label("MaximumElementwiseInt8/8bit");
  int i = 0;
#ifdef USE_NEON
  for (; i <= size - 16; i += 16) {
    const int8x16_t input1_val_original = vld1q_s8(input1_data + i);
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);
    const int8x16_t max_data =
        vmaxq_s8(input1_val_original, input2_val_original);
    vst1q_s8(output_data + i, max_data);
  }
#endif  // USE_NEON
  for (; i < size; ++i) {
    const int8 input1_val = input1_data[i];
    const int8 input2_val = input2_data[i];
    output_data[i] = std::max(input1_val, input2_val);
  }
}

inline void MaximumScalarBroadcast(int size, const ArithmeticParams& params,
                                   int8 input1_data, const int8* input2_data,
                                   int8* output_data) {
  ruy::profiler::ScopeLabel label("MaximumScalarBroadcastInt8/8bit");
  int i = 0;

#ifdef USE_NEON
  const int8x16_t input1_val_original = vdupq_n_s8(input1_data);
  for (; i <= size - 16; i += 16) {
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);
    const int8x16_t max_data =
        vmaxq_s8(input1_val_original, input2_val_original);
    vst1q_s8(output_data + i, max_data);
  }
#endif  // USE_NEON
  for (; i < size; ++i) {
    const int8 input2_val = input2_data[i];
    output_data[i] = std::max(input1_data, input2_val);
  }
}

// Assume input1 & input2 have the same scale & zero point.
inline void MinimumElementwise(int size, const ArithmeticParams& params,
                               const int8* input1_data, const int8* input2_data,
                               int8* output_data) {
  ruy::profiler::ScopeLabel label("MinimumElementwiseInt8/8bit");
  int i = 0;
#ifdef USE_NEON
  for (; i <= size - 16; i += 16) {
    const int8x16_t input1_val_original = vld1q_s8(input1_data + i);
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);
    const int8x16_t min_data =
        vminq_s8(input1_val_original, input2_val_original);
    vst1q_s8(output_data + i, min_data);
  }
#endif  // USE_NEON
  for (; i < size; ++i) {
    const int8 input1_val = input1_data[i];
    const int8 input2_val = input2_data[i];
    output_data[i] = std::min(input1_val, input2_val);
  }
}

inline void MinimumScalarBroadcast(int size, const ArithmeticParams& params,
                                   int8 input1_data, const int8* input2_data,
                                   int8* output_data) {
  ruy::profiler::ScopeLabel label("MinimumScalarBroadcastInt8/8bit");
  int i = 0;

#ifdef USE_NEON
  const int8x16_t input1_val_original = vdupq_n_s8(input1_data);
  for (; i <= size - 16; i += 16) {
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);
    const int8x16_t min_data =
        vminq_s8(input1_val_original, input2_val_original);
    vst1q_s8(output_data + i, min_data);
  }
#endif  // USE_NEON
  for (; i < size; ++i) {
    const int8 input2_val = input2_data[i];
    output_data[i] = std::min(input1_data, input2_val);
  }
}

template <typename Op>
inline void BroadcastMaximumDispatch(const ArithmeticParams& params,
                                     const RuntimeShape& input1_shape,
                                     const int8* input1_data,
                                     const RuntimeShape& input2_shape,
                                     const int8* input2_data,
                                     const RuntimeShape& output_shape,
                                     int8* output_data, Op op) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_ops::MaximumMinimumBroadcastSlow(
        input1_shape, input1_data, input2_shape, input2_data, output_shape,
        output_data, op);
  }

  BinaryBroadcastFiveFold(params, input1_shape, input1_data, input2_shape,
                          input2_data, output_shape, output_data,
                          MaximumElementwise, MaximumScalarBroadcast);
}

template <typename Op>
inline void BroadcastMinimumDispatch(const ArithmeticParams& params,
                                     const RuntimeShape& input1_shape,
                                     const int8* input1_data,
                                     const RuntimeShape& input2_shape,
                                     const int8* input2_data,
                                     const RuntimeShape& output_shape,
                                     int8* output_data, Op op) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_ops::MaximumMinimumBroadcastSlow(
        input1_shape, input1_data, input2_shape, input2_data, output_shape,
        output_data, op);
  }

  BinaryBroadcastFiveFold(params, input1_shape, input1_data, input2_shape,
                          input2_data, output_shape, output_data,
                          MinimumElementwise, MinimumScalarBroadcast);
}

template <typename T>
void CumsumImpl(const T* input_data, const RuntimeShape& shape, int axis,
                bool exclusive, bool reverse, T* output_data) {
  Eigen::array<Eigen::DenseIndex, 3> dims = {1, 1, 1};

  for (int i = 0; i < axis; ++i) {
    dims[0] *= shape.Dims(i);
  }
  dims[1] = shape.Dims(axis);
  for (int i = axis + 1; i < shape.DimensionsCount(); ++i) {
    dims[2] *= shape.Dims(i);
  }

  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 3, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
      ConstTensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<T, 3, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
      Tensor;
  ConstTensor input(input_data, dims);
  Tensor output(output_data, dims);

  if (reverse) {
    Eigen::array<bool, 3> reverse_idx = {false, true, false};
    output =
        input.reverse(reverse_idx).cumsum(1, exclusive).reverse(reverse_idx);
  } else {
    output = input.cumsum(1, exclusive);
  }
}

template <typename T>
void CumSum(const T* input_data, const RuntimeShape& shape, int axis,
            bool exclusive, bool reverse, T* output_data) {
  const int dim = shape.DimensionsCount();
  TFLITE_DCHECK_GE(dim, 1);
  CumsumImpl<T>(input_data, shape, axis, exclusive, reverse, output_data);
}

inline void PReluScalarBroadcast(int size, const ArithmeticParams& params,
                                 float alpha, const float* input_data,
                                 float* output_data) {
  ruy::profiler::ScopeLabel label("PreluScalarBroadcast/float");
  int i = 0;

#ifdef USE_NEON
  const float32x4_t zero_dup = vdupq_n_f32(0.0f);
  const float32x4_t alpha_dup = vdupq_n_f32(alpha);
  for (; i <= size - 16; i += 16) {
    const float32x4_t input1 = vld1q_f32(input_data + i);
    const float32x4_t input2 = vld1q_f32(input_data + i + 4);
    const float32x4_t input3 = vld1q_f32(input_data + i + 8);
    const float32x4_t input4 = vld1q_f32(input_data + i + 12);

    const float32x4_t temp1 = vmulq_f32(input1, alpha_dup);
    const float32x4_t temp2 = vmulq_f32(input2, alpha_dup);
    const float32x4_t temp3 = vmulq_f32(input3, alpha_dup);
    const float32x4_t temp4 = vmulq_f32(input4, alpha_dup);

    const uint32x4_t mask1 = vcgeq_f32(input1, zero_dup);
    const uint32x4_t mask2 = vcgeq_f32(input2, zero_dup);
    const uint32x4_t mask3 = vcgeq_f32(input3, zero_dup);
    const uint32x4_t mask4 = vcgeq_f32(input4, zero_dup);

    const float32x4_t result1 = vbslq_f32(mask1, input1, temp1);
    vst1q_f32(output_data + i, result1);
    const float32x4_t result2 = vbslq_f32(mask2, input2, temp2);
    vst1q_f32(output_data + i + 4, result2);
    const float32x4_t result3 = vbslq_f32(mask3, input3, temp3);
    vst1q_f32(output_data + i + 8, result3);
    const float32x4_t result4 = vbslq_f32(mask4, input4, temp4);
    vst1q_f32(output_data + i + 12, result4);
  }

  for (; i <= size - 4; i += 4) {
    const float32x4_t input = vld1q_f32(input_data + i);
    const float32x4_t temp = vmulq_f32(input, alpha_dup);
    const uint32x4_t mask = vcgeq_f32(input, zero_dup);
    const float32x4_t result = vbslq_f32(mask, input, temp);
    vst1q_f32(output_data + i, result);
  }
#endif  // USE_NEON
  for (; i < size; ++i) {
    const float input = input_data[i];
    output_data[i] = input >= 0.f ? input : input * alpha;
  }
}

inline void PReluElementWise(int flat_size, const ArithmeticParams& params,
                             const float* alpha_data, const float* input_data,
                             float* output_data) {
  ruy::profiler::ScopeLabel label("PreluElementWise/float");

  int i = 0;
#ifdef USE_NEON
  const float32x4_t zero_dup = vdupq_n_f32(0.0f);
  for (; i <= flat_size - 16; i += 16) {
    const float32x4_t input1 = vld1q_f32(input_data + i);
    const float32x4_t alpha1 = vld1q_f32(alpha_data + i);
    const float32x4_t input2 = vld1q_f32(input_data + i + 4);
    const float32x4_t alpha2 = vld1q_f32(alpha_data + i + 4);
    const float32x4_t input3 = vld1q_f32(input_data + i + 8);
    const float32x4_t alpha3 = vld1q_f32(alpha_data + i + 8);
    const float32x4_t input4 = vld1q_f32(input_data + i + 12);
    const float32x4_t alpha4 = vld1q_f32(alpha_data + i + 12);

    const float32x4_t temp1 = vmulq_f32(input1, alpha1);
    const float32x4_t temp2 = vmulq_f32(input2, alpha2);
    const float32x4_t temp3 = vmulq_f32(input3, alpha3);
    const float32x4_t temp4 = vmulq_f32(input4, alpha4);

    const uint32x4_t mask1 = vcgeq_f32(input1, zero_dup);
    const uint32x4_t mask2 = vcgeq_f32(input2, zero_dup);
    const uint32x4_t mask3 = vcgeq_f32(input3, zero_dup);
    const uint32x4_t mask4 = vcgeq_f32(input4, zero_dup);

    const float32x4_t result1 = vbslq_f32(mask1, input1, temp1);
    vst1q_f32(output_data + i, result1);
    const float32x4_t result2 = vbslq_f32(mask2, input2, temp2);
    vst1q_f32(output_data + i + 4, result2);
    const float32x4_t result3 = vbslq_f32(mask3, input3, temp3);
    vst1q_f32(output_data + i + 8, result3);
    const float32x4_t result4 = vbslq_f32(mask4, input4, temp4);
    vst1q_f32(output_data + i + 12, result4);
  }

  for (; i <= flat_size - 4; i += 4) {
    const float32x4_t input = vld1q_f32(input_data + i);
    const float32x4_t alpha = vld1q_f32(alpha_data + i);

    const float32x4_t temp = vmulq_f32(input, alpha);
    const uint32x4_t mask = vcgeq_f32(input, zero_dup);
    const float32x4_t result = vbslq_f32(mask, input, temp);
    vst1q_f32(output_data + i, result);
  }
#endif  // USE_NEON
  for (; i < flat_size; ++i) {
    const float input = input_data[i];
    const float alpha = alpha_data[i];
    output_data[i] = input >= 0.f ? input : input * alpha;
  }
}

inline void BroadcastPReluDispatch(
    const ArithmeticParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& alpha_shape,
    const float* alpha_data, const RuntimeShape& output_shape,
    float* output_data, float (*func)(float, float)) {
  if (params.broadcast_category == BroadcastableOpCategory::kGenericBroadcast) {
    return reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
        input_shape, input_data, alpha_shape, alpha_data, output_shape,
        output_data, func);
  }

  BinaryBroadcastFiveFold(params, input_shape, input_data, alpha_shape,
                          alpha_data, output_shape, output_data,
                          PReluElementWise, PReluScalarBroadcast);
}

// Returns the index with minimum value within `input_data`.
// If there is a tie, returns the smaller index.
template <typename T>
inline int ArgMinVector(const T* input_data, int size) {
  T min_value = input_data[0];
  int min_index = 0;
  for (int i = 1; i < size; ++i) {
    const T curr_value = input_data[i];
    if (curr_value < min_value) {
      min_value = curr_value;
      min_index = i;
    }
  }
  return min_index;
}

// Returns the index with maximum value within `input_data`.
// If there is a tie, returns the smaller index.
template <typename T>
inline int ArgMaxVector(const T* input_data, int size) {
  T max_value = input_data[0];
  int max_index = 0;
  for (int i = 1; i < size; ++i) {
    const T curr_value = input_data[i];
    if (curr_value > max_value) {
      max_value = curr_value;
      max_index = i;
    }
  }
  return max_index;
}

template <>
inline int ArgMinVector(const float* input_data, int size) {
  int32_t min_index = 0;
  float min_value = input_data[0];
  int32_t i = 1;
#ifdef USE_NEON
  if (size >= 4) {
    float32x4_t min_value_f32x4 = vld1q_f32(input_data);
    const int32_t index_init[4] = {0, 1, 2, 3};
    int32x4_t min_index_s32x4 = vld1q_s32(index_init);
    int32x4_t index_s32x4 = min_index_s32x4;
    int32x4_t inc = vdupq_n_s32(4);
    for (i = 4; i <= size - 4; i += 4) {
      // Increase indices by 4.
      index_s32x4 = vaddq_s32(index_s32x4, inc);
      float32x4_t v = vld1q_f32(&input_data[i]);
      uint32x4_t mask = vcltq_f32(v, min_value_f32x4);
      min_value_f32x4 = vminq_f32(min_value_f32x4, v);
      min_index_s32x4 = vbslq_s32(mask, index_s32x4, min_index_s32x4);
    }
    // Find min element within float32x4_t.
#ifdef __aarch64__
    min_value = vminvq_f32(min_value_f32x4);
#else
    float32x2_t min_value_f32x2 = vpmin_f32(vget_low_f32(min_value_f32x4),
                                            vget_high_f32(min_value_f32x4));
    min_value_f32x2 = vpmin_f32(min_value_f32x2, min_value_f32x2);
    min_value = vget_lane_f32(min_value_f32x2, 0);
#endif  // __aarch64__
    // Mask indices of non-min values with max int32_t.
    float32x4_t fill_min_value_f32x4 = vdupq_n_f32(min_value);
    uint32x4_t mask = vceqq_f32(min_value_f32x4, fill_min_value_f32x4);
    int32x4_t all_set = vdupq_n_s32(std::numeric_limits<int>::max());
    min_index_s32x4 = vbslq_s32(mask, min_index_s32x4, all_set);
    // Find min index of min values.
#ifdef __aarch64__
    min_index = vminvq_s32(min_index_s32x4);
#else
    int32x2_t min_index_s32x2 = vpmin_s32(vget_low_s32(min_index_s32x4),
                                          vget_high_s32(min_index_s32x4));
    min_index_s32x2 = vpmin_s32(min_index_s32x2, min_index_s32x2);
    min_index = vget_lane_s32(min_index_s32x2, 0);
#endif  // __aarch64__
  }
#endif  // USE_NEON
  // Leftover loop.
  for (; i < size; ++i) {
    const float curr_value = input_data[i];
    if (curr_value < min_value) {
      min_value = curr_value;
      min_index = i;
    }
  }
  return min_index;
}

template <>
inline int ArgMaxVector(const float* input_data, int size) {
  int32_t max_index = 0;
  float max_value = input_data[0];
  int32_t i = 1;
#ifdef USE_NEON
  if (size >= 4) {
    float32x4_t max_value_f32x4 = vld1q_f32(input_data);
    const int32_t index_init[4] = {0, 1, 2, 3};
    int32x4_t max_index_s32x4 = vld1q_s32(index_init);
    int32x4_t index_s32x4 = max_index_s32x4;
    int32x4_t inc = vdupq_n_s32(4);
    for (i = 4; i <= size - 4; i += 4) {
      // Increase indices by 4.
      index_s32x4 = vaddq_s32(index_s32x4, inc);
      float32x4_t v = vld1q_f32(&input_data[i]);
      uint32x4_t mask = vcgtq_f32(v, max_value_f32x4);
      max_value_f32x4 = vmaxq_f32(max_value_f32x4, v);
      max_index_s32x4 = vbslq_s32(mask, index_s32x4, max_index_s32x4);
    }
    // Find max element within float32x4_t.
#ifdef __aarch64__
    max_value = vmaxvq_f32(max_value_f32x4);
#else
    float32x2_t max_value_f32x2 = vpmax_f32(vget_low_f32(max_value_f32x4),
                                            vget_high_f32(max_value_f32x4));
    max_value_f32x2 = vpmax_f32(max_value_f32x2, max_value_f32x2);
    max_value = vget_lane_f32(max_value_f32x2, 0);
#endif  // __aarch64__
    // Mask indices of non-max values with max int32_t.
    float32x4_t fill_max_value_f32x4 = vdupq_n_f32(max_value);
    uint32x4_t mask = vceqq_f32(max_value_f32x4, fill_max_value_f32x4);
    int32x4_t all_set = vdupq_n_s32(std::numeric_limits<int>::max());
    max_index_s32x4 = vbslq_s32(mask, max_index_s32x4, all_set);
    // Find min index of max values.
#ifdef __aarch64__
    max_index = vminvq_s32(max_index_s32x4);
#else
    int32x2_t max_index_s32x2 = vpmin_s32(vget_low_s32(max_index_s32x4),
                                          vget_high_s32(max_index_s32x4));
    max_index_s32x2 = vpmin_s32(max_index_s32x2, max_index_s32x2);
    max_index = vget_lane_s32(max_index_s32x2, 0);
#endif  // __aarch64__
  }
#endif  // USE_NEON
  // Leftover loop.
  for (; i < size; ++i) {
    const float curr_value = input_data[i];
    if (curr_value > max_value) {
      max_value = curr_value;
      max_index = i;
    }
  }
  return max_index;
}

template <>
inline int ArgMaxVector(const int8_t* input_data, int size) {
  int32_t max_index = 0;
  int8_t max_value = input_data[0];
  int32_t i = 0;
#ifdef USE_NEON
  constexpr int VECTOR_SIZE = 16;
  if (size >= VECTOR_SIZE) {
    int8x16_t max_value_s8x16;
    for (; i <= size - VECTOR_SIZE; i += VECTOR_SIZE) {
      max_value_s8x16 = vld1q_s8(input_data + i);
      int8_t max_from_vec;
#ifdef __aarch64__
      max_from_vec = vmaxvq_s8(max_value_s8x16);
#else   // 32 bit
      int8x8_t max_val_s8x8 =
          vpmax_s8(vget_low_s8(max_value_s8x16), vget_high_s8(max_value_s8x16));
      max_val_s8x8 = vpmax_s8(max_val_s8x8, max_val_s8x8);
      max_val_s8x8 = vpmax_s8(max_val_s8x8, max_val_s8x8);
      max_val_s8x8 = vpmax_s8(max_val_s8x8, max_val_s8x8);
      max_from_vec = vget_lane_s8(max_val_s8x8, 0);
#endif  // __aarch64__
      if (max_from_vec > max_value) {
        max_value = max_from_vec;
        max_index = i;
      }
    }
  }
  for (int start_idx = max_index; start_idx < max_index + VECTOR_SIZE;
       start_idx++) {
    if (input_data[start_idx] == max_value) {
      max_index = start_idx;
      break;
    }
  }

#endif  // USE_NEON
  // Leftover loop.
  for (; i < size; ++i) {
    const int8_t curr_value = input_data[i];
    if (curr_value > max_value) {
      max_value = curr_value;
      max_index = i;
    }
  }

  return max_index;
}

template <>
inline int ArgMaxVector(const uint8_t* input_data, int size) {
  int32_t max_index = 0;
  uint8_t max_value = input_data[0];
  int32_t i = 0;
#ifdef USE_NEON
  constexpr int VECTOR_SIZE = 16;
  if (size >= VECTOR_SIZE) {
    uint8x16_t max_value_u8x16;
    for (; i <= size - VECTOR_SIZE; i += VECTOR_SIZE) {
      max_value_u8x16 = vld1q_u8(input_data + i);
      uint8_t max_from_vec;
#ifdef __aarch64__
      max_from_vec = vmaxvq_u8(max_value_u8x16);
#else   // 32 bit
      uint8x8_t max_val_u8x8 =
          vpmax_u8(vget_low_u8(max_value_u8x16), vget_high_u8(max_value_u8x16));
      max_val_u8x8 = vpmax_u8(max_val_u8x8, max_val_u8x8);
      max_val_u8x8 = vpmax_u8(max_val_u8x8, max_val_u8x8);
      max_val_u8x8 = vpmax_u8(max_val_u8x8, max_val_u8x8);
      max_from_vec = vget_lane_u8(max_val_u8x8, 0);
#endif  // __aarch64__
      if (max_from_vec > max_value) {
        max_value = max_from_vec;
        max_index = i;
      }
    }
  }
  for (int start_idx = max_index; start_idx < max_index + VECTOR_SIZE;
       start_idx++) {
    if (input_data[start_idx] == max_value) {
      max_index = start_idx;
      break;
    }
  }

#endif  // USE_NEON
  // Leftover loop.
  for (; i < size; ++i) {
    const uint8_t curr_value = input_data[i];
    if (curr_value > max_value) {
      max_value = curr_value;
      max_index = i;
    }
  }

  return max_index;
}

// Specializes ArgMinMax function with axis=dims-1.
// In this case, ArgMinMax reduction is applied on contiguous memory.
template <typename T1, typename T2, bool is_arg_max>
inline void ArgMinMaxLastAxis(const RuntimeShape& input_shape,
                              const T1* input_data,
                              const RuntimeShape& output_shape,
                              T2* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_EQ(input_shape.Dims(0), output_shape.Dims(0));

  int outer_size = input_shape.Dims(0);
  int axis_size = input_shape.Dims(1);
  for (int outer = 0; outer < outer_size; ++outer) {
    if (is_arg_max) {
      output_data[outer] = static_cast<T2>(
          ArgMaxVector<T1>(input_data + outer * axis_size, axis_size));
    } else {
      output_data[outer] = static_cast<T2>(
          ArgMinVector<T1>(input_data + outer * axis_size, axis_size));
    }
  }
}

template <typename T1, typename T2, typename T3>
inline void ArgMinMax(const RuntimeShape& input1_shape, const T1* input1_data,
                      const T3* input2_data, const RuntimeShape& output_shape,
                      T2* output_data, const bool is_arg_max) {
  ruy::profiler::ScopeLabel label("ArgMinMax");

  TFLITE_DCHECK_GT(input1_shape.DimensionsCount(), 0);
  TFLITE_DCHECK_EQ(input1_shape.DimensionsCount() - 1,
                   output_shape.DimensionsCount());
  int axis = input2_data[0];
  if (axis < 0) {
    axis += input1_shape.DimensionsCount();
  }
  const int axis_size = input1_shape.Dims(axis);

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    TFLITE_DCHECK_EQ(input1_shape.Dims(i), output_shape.Dims(i));
    outer_size *= input1_shape.Dims(i);
  }

  int inner_size = 1;
  const int dims_count = input1_shape.DimensionsCount();
  for (int i = axis + 1; i < dims_count; ++i) {
    TFLITE_DCHECK_EQ(input1_shape.Dims(i), output_shape.Dims(i - 1));
    inner_size *= input1_shape.Dims(i);
  }

  // Call specialized function when axis=dims-1. So far, only float32 is
  // optimized so reroute to specialized function only when T1 is float32.
  if (inner_size == 1 &&
      (std::is_same<T1, float>::value || std::is_same<T1, int8_t>::value ||
       std::is_same<T1, uint8_t>::value)) {
    if (is_arg_max) {
      ArgMinMaxLastAxis<T1, T2, /*is_arg_max=*/true>(
          {outer_size, axis_size}, input1_data, {outer_size}, output_data);
    } else {
      ArgMinMaxLastAxis<T1, T2, /*is_arg_max=*/false>(
          {outer_size, axis_size}, input1_data, {outer_size}, output_data);
    }
    return;
  }

  reference_ops::ArgMinMax(input1_shape, input1_data, input2_data, output_shape,
                           output_data, is_arg_max);
}

template <typename T1, typename T2, typename T3>
void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
            const T3* input2_data, const RuntimeShape& output_shape,
            T2* output_data) {
  ArgMinMax(input1_shape, input1_data, input2_data, output_shape, output_data,
            /*is_arg_max=*/true);
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
// For backward compatibility, reference_ops has ArgMax function.
template <typename T1, typename T2, typename T3>
inline void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
                   const RuntimeShape& input2_shape, const T3* input2_data,
                   const RuntimeShape& output_shape, T2* output_data) {
  // Drop shape of second input: not needed.
  ArgMax(input1_shape, input1_data, input2_data, output_shape, output_data);
}

inline void Conv3D(const Conv3DParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& filter_shape,
                   const float* filter_data, const RuntimeShape& bias_shape,
                   const float* bias_data, const RuntimeShape& output_shape,
                   float* output_data, const RuntimeShape& im2col_shape,
                   float* im2col_data,
                   const RuntimeShape& transposed_filter_shape,
                   float* transposed_filter_data,
                   CpuBackendContext* cpu_backend_context) {
  const int stride_depth = params.stride_depth;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  const int dilation_depth_factor = params.dilation_depth;
  const int dilation_height_factor = params.dilation_height;
  const int dilation_width_factor = params.dilation_width;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);

  ruy::profiler::ScopeLabel label("Conv3D");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_depth = filter_shape.Dims(0);
  const bool need_dilated_im2col = dilation_width_factor != 1 ||
                                   dilation_height_factor != 1 ||
                                   dilation_depth_factor != 1;
  const bool need_im2col = stride_depth != 1 || stride_height != 1 ||
                           stride_width != 1 || filter_depth != 1 ||
                           filter_height != 1 || filter_width != 1;

  if (need_dilated_im2col) {
    DilatedIm2col3D(params, filter_depth, filter_height, filter_width,
                    float_zero_byte, input_shape, input_data, im2col_shape,
                    im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col3D(params, filter_depth, filter_height, filter_width, float_zero_byte,
             input_shape, input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  // Transpose the filter tensor.
  TransposeParams transpose_params;
  transpose_params.perm_count = 5;
  transpose_params.perm[0] = 4;
  transpose_params.perm[1] = 0;
  transpose_params.perm[2] = 1;
  transpose_params.perm[3] = 2;
  transpose_params.perm[4] = 3;
  Transpose<float, 5>(transpose_params, filter_shape, filter_data,
                      transposed_filter_shape, transposed_filter_data);

  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(4);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = n;
  lhs_params.cols = k;
  cpu_backend_gemm::MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = k;
  rhs_params.cols = m;
  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;
  cpu_backend_gemm::GemmParams<float, float> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  cpu_backend_gemm::Gemm(lhs_params, transposed_filter_data, rhs_params,
                         gemm_input_data, dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

// Returns in 'im_data' (assumed to be zero-initialized) image patch in storage
// order (planes, height, width, channel), constructed from patches in
// 'col_data', which is required to be in storage order (out_planes * out_height
// * out_width, filter_planes, filter_height, filter_width, in_channel).
//
// This function is copied from tensorflow/core/kernels/conv_grad_ops_3d.cc
// authored by Eugene Zhulenev(ezhulenev).
template <typename T>
void Col2im(const T* col_data, const int channel, const int planes,
            const int height, const int width, const int filter_p,
            const int filter_h, const int filter_w, const int pad_pt,
            const int pad_t, const int pad_l, const int pad_pb, const int pad_b,
            const int pad_r, const int stride_p, const int stride_h,
            const int stride_w, T* im_data) {
  const int planes_col = (planes + pad_pt + pad_pb - filter_p) / stride_p + 1;
  const int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  const int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int p_pad = -pad_pt;
  for (int p = 0; p < planes_col; ++p) {
    int h_pad = -pad_t;
    for (int h = 0; h < height_col; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < width_col; ++w) {
        T* im_patch_data =
            im_data +
            (p_pad * height * width + h_pad * width + w_pad) * channel;
        for (int ip = p_pad; ip < p_pad + filter_p; ++ip) {
          for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
            for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
              if (ip >= 0 && ip < planes && ih >= 0 && ih < height && iw >= 0 &&
                  iw < width) {
                for (int i = 0; i < channel; ++i) {
                  im_patch_data[i] += col_data[i];
                }
              }
              im_patch_data += channel;
              col_data += channel;
            }
            // Jump over remaining number of channel.
            im_patch_data += channel * (width - filter_w);
          }
          // Jump over remaining number of (channel * width).
          im_patch_data += (channel * width) * (height - filter_h);
        }
        w_pad += stride_w;
      }
      h_pad += stride_h;
    }
    p_pad += stride_p;
  }
}

template <typename T>
void BiasAdd3D(T* im_data, const T* bias_data, const RuntimeShape& input_shape,
               float float_activation_min, float float_activation_max) {
  if (bias_data) {
    const int outer_size = input_shape.Dims(0) * input_shape.Dims(1) *
                           input_shape.Dims(2) * input_shape.Dims(3);
    const int num_channels = input_shape.Dims(4);
    for (int n = 0; n < outer_size; ++n) {
      for (int c = 0; c < num_channels; ++c) {
        im_data[c] = ActivationFunctionWithMinMax(im_data[c] + bias_data[c],
                                                  float_activation_min,
                                                  float_activation_max);
      }
      im_data += num_channels;
    }
  } else {
    const int flat_size = input_shape.FlatSize();
    for (int i = 0; i < flat_size; ++i) {
      im_data[i] = ActivationFunctionWithMinMax(
          im_data[i], float_activation_min, float_activation_max);
    }
  }
}

inline void Conv3DTranspose(
    const Conv3DTransposeParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& bias_shape,
    const float* bias_data, const RuntimeShape& output_shape,
    float* const output_data, const RuntimeShape& col2im_shape,
    float* col2im_data, CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("Conv3DTranspose/float");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 5);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 5);
  TFLITE_DCHECK(col2im_data);

  const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_channel = MatchingDim(input_shape, 4, filter_shape, 4);
  const int output_channel = MatchingDim(output_shape, 4, filter_shape, 3);
  const int input_spatial_size =
      input_shape.Dims(1) * input_shape.Dims(2) * input_shape.Dims(3);
  const int output_spatial_size =
      output_shape.Dims(1) * output_shape.Dims(2) * output_shape.Dims(3);

  const int output_spatial_dim_1 = output_shape.Dims(1);
  const int output_spatial_dim_2 = output_shape.Dims(2);
  const int output_spatial_dim_3 = output_shape.Dims(3);
  const int input_offset = input_spatial_size * input_channel;
  const int output_offset = output_spatial_size * output_channel;

  const int filter_spatial_dim_1 = filter_shape.Dims(0);
  const int filter_spatial_dim_2 = filter_shape.Dims(1);
  const int filter_spatial_dim_3 = filter_shape.Dims(2);

  const int spatial_dim_1_padding_before = params.padding_values.depth;
  const int spatial_dim_1_padding_after =
      params.padding_values.height + params.padding_values.depth_offset;
  const int spatial_dim_2_padding_before = params.padding_values.height;
  const int spatial_dim_2_padding_after =
      params.padding_values.height + params.padding_values.height_offset;
  const int spatial_dim_3_padding_before = params.padding_values.width;
  const int spatial_dim_3_padding_after =
      params.padding_values.width + params.padding_values.width_offset;
  const int spatial_dim_1_stride = params.stride_depth;
  const int spatial_dim_2_stride = params.stride_height;
  const int spatial_dim_3_stride = params.stride_width;
  const int filter_total_size = filter_spatial_dim_1 * filter_spatial_dim_2 *
                                filter_spatial_dim_3 * output_channel;

  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = filter_total_size;
  lhs_params.cols = input_channel;
  float* output_data_p = output_data;
  std::fill_n(output_data, output_offset * batch_size, 0.0f);
  for (int i = 0; i < batch_size; ++i) {
    cpu_backend_gemm::MatrixParams<float> rhs_params;
    rhs_params.order = cpu_backend_gemm::Order::kColMajor;
    rhs_params.rows = input_channel;
    rhs_params.cols = input_spatial_size;
    cpu_backend_gemm::MatrixParams<float> dst_params;
    dst_params.order = cpu_backend_gemm::Order::kColMajor;
    dst_params.rows = filter_total_size;
    dst_params.cols = input_spatial_size;
    cpu_backend_gemm::GemmParams<float, float> gemm_params;
    cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params,
                           input_data + input_offset * i, dst_params,
                           col2im_data, gemm_params, cpu_backend_context);

    Col2im(col2im_data, output_channel, output_spatial_dim_1,
           output_spatial_dim_2, output_spatial_dim_3, filter_spatial_dim_1,
           filter_spatial_dim_2, filter_spatial_dim_3,
           spatial_dim_1_padding_before, spatial_dim_2_padding_before,
           spatial_dim_3_padding_before, spatial_dim_1_padding_after,
           spatial_dim_2_padding_after, spatial_dim_3_padding_after,
           spatial_dim_1_stride, spatial_dim_2_stride, spatial_dim_3_stride,
           output_data_p);
    output_data_p += output_offset;
  }
  output_data_p = output_data;
  BiasAdd3D(output_data_p, bias_data, output_shape, params.float_activation_min,
            params.float_activation_max);
}

// Worker for summing up within a single interval. Interval is identified by
// index from [start, end).
template <typename T>
struct AddNWorkerTask : cpu_backend_threadpool::Task {
  AddNWorkerTask(const T* const* input_data, T* scratch_buffer, int start,
                 int end, int num_elems, int split)
      : input_data(input_data),
        scratch_buffer(scratch_buffer),
        start(start),
        end(end),
        num_elems(num_elems),
        split(split) {}
  void Run() override {
    RuntimeShape shape(1);
    shape.SetDim(0, num_elems);
    ArithmeticParams params;
    T output_activation_min = std::numeric_limits<T>::lowest(),
      output_activation_max = std::numeric_limits<T>::max();
    SetActivationParams(output_activation_min, output_activation_max, &params);
    T* start_p = scratch_buffer + split * num_elems;
    memcpy(start_p, input_data[start], sizeof(T) * num_elems);
    for (int i = start + 1; i < end; i++) {
      Add(params, shape, start_p, shape, input_data[i], shape, start_p);
    }
  }

  const T* const* input_data;
  T* scratch_buffer;
  int start;
  int end;
  int num_elems;
  int split;
};

// T is expected to be either float or int.
template <typename T>
inline void AddN(const RuntimeShape& input_shape, const size_t num_inputs,
                 const T* const* input_data, T* output_data, T* scratch_buffer,
                 CpuBackendContext* cpu_backend_context) {
  // All inputs and output should have the same shape, this is checked during
  // Prepare stage.
  const size_t num_elems = input_shape.FlatSize();
  const int thread_count =
      std::min(std::max(1, static_cast<int>(num_inputs) / 2),
               cpu_backend_context->max_num_threads());
  memset(scratch_buffer, 0, sizeof(T) * num_elems * thread_count);

  std::vector<AddNWorkerTask<T>> tasks;
  tasks.reserve(thread_count);
  int start = 0;
  for (int i = 0; i < thread_count; ++i) {
    int end = start + (num_inputs - start) / (thread_count - i);
    tasks.emplace_back(AddNWorkerTask<T>(input_data, scratch_buffer, start, end,
                                         num_elems, i));
    start = end;
  }
  // Run all tasks on the thread pool.
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
  RuntimeShape shape(1);
  shape.SetDim(0, num_elems);
  ArithmeticParams params;
  T output_activation_min = std::numeric_limits<T>::lowest(),
    output_activation_max = std::numeric_limits<T>::max();
  SetActivationParams(output_activation_min, output_activation_max, &params);
  memcpy(output_data, scratch_buffer, sizeof(T) * num_elems);
  for (int i = 1; i < tasks.size(); i++) {
    Add(params, shape, output_data, shape, scratch_buffer + i * num_elems,
        shape, output_data);
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#if defined OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#undef OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#pragma GCC diagnostic pop
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_
