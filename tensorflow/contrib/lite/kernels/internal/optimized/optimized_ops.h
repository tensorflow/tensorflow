/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_OPS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_OPS_H_

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"
#include "tensorflow/contrib/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Unoptimized reference ops:
using reference_ops::BroadcastGreater;
using reference_ops::BroadcastGreaterEqual;
using reference_ops::BroadcastLess;
using reference_ops::BroadcastLessEqual;
using reference_ops::Greater;
using reference_ops::GreaterEqual;
using reference_ops::Less;
using reference_ops::LessEqual;
using reference_ops::RankOneSelect;
using reference_ops::Select;

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

template <typename Scalar, int N>
VectorMap<Scalar> MapAsVector(Scalar* data, const Dims<N>& dims) {
  const int size = RequiredBufferSizeForDims(dims);
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

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsRows(Scalar* data,
                                                const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsCols(Scalar* data,
                                               const Dims<N>& dims) {
  const int cols = dims.sizes[N - 1];
  int rows = 1;
  for (int d = 0; d < N - 1; d++) {
    rows *= dims.sizes[d];
  }
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar>
using ArrayMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Array<typename std::remove_const<Scalar>::type,
                                  Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar, int N>
ArrayMap<Scalar> MapAsArrayWithFirstDimAsRows(Scalar* data,
                                              const Dims<N>& dims) {
  const int rows = dims.sizes[0];
  int cols = 1;
  for (int d = 1; d < N; d++) {
    cols *= dims.sizes[d];
  }
  return ArrayMap<Scalar>(data, rows, cols);
}

// TODO(b/62193649): this function is only needed as long
// as we have the --variable_batch hack.
template <typename Scalar, int N>
MatrixMap<Scalar> MapAsMatrixWithGivenNumberOfRows(Scalar* data,
                                                   const Dims<N>& dims,
                                                   int rows) {
  int cols = 1;
  bool matched_rows = false;
  for (int d = 0; d < N; d++) {
    cols *= dims.sizes[d];
    if (cols == rows) {
      matched_rows = true;
      cols = 1;
    }
  }
  TFLITE_DCHECK(matched_rows);
  return MatrixMap<Scalar>(data, rows, cols);
}

// DO NOT USE THIS STRUCT FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING ELEMENT-WISE
// BROADCASTING.
//
// NdArrayDesc<N> describes the shape and memory layout of an N-dimensional
// rectangular array of numbers.
//
// NdArrayDesc<N> is basically identical to Dims<N> defined in types.h.
// However, as Dims<N> is to be deprecated, this class exists as an adaptor
// to enable simple unoptimized implementations of element-wise broadcasting
// operations.
template <int N>
struct NdArrayDesc {
  // The "extent" of each dimension. Indices along dimension d must be in the
  // half-open interval [0, extents[d]).
  int extents[N];

  // The number of *elements* (not bytes) between consecutive indices of each
  // dimension.
  int strides[N];
};

// DO NOT USE THIS FUNCTION FOR NEW FUNCTIONALITY BEYOND IMPLEMENTING
// ELEMENT-WISE BROADCASTING.
//
// Same as Offset(), except takes as NdArrayDesc<N> instead of Dims<N>.
inline int SubscriptToIndex(const NdArrayDesc<4>& desc, int i0, int i1, int i2,
                            int i3) {
  TFLITE_DCHECK(i0 >= 0 && i0 < desc.extents[0]);
  TFLITE_DCHECK(i1 >= 0 && i1 < desc.extents[1]);
  TFLITE_DCHECK(i2 >= 0 && i2 < desc.extents[2]);
  TFLITE_DCHECK(i3 >= 0 && i3 < desc.extents[3]);
  return i0 * desc.strides[0] + i1 * desc.strides[1] + i2 * desc.strides[2] +
         i3 * desc.strides[3];
}

// Given the dimensions of the operands for an element-wise binary broadcast,
// adjusts them so that they can be directly iterated over with simple loops.
// Returns the adjusted dims as instances of NdArrayDesc in 'desc0_out' and
// 'desc1_out'. 'desc0_out' and 'desc1_out' cannot be nullptr.
//
// This function assumes that the two input shapes are compatible up to
// broadcasting and the shorter one has already been prepended with 1s to be the
// same length. E.g., if shape0 is (1, 16, 16, 64) and shape1 is (1, 64),
// shape1 must already have been prepended to be (1, 1, 1, 64). Recall that
// Dims<N> refer to shapes in reverse order. In this case, input0_dims will be
// (64, 16, 16, 1) and input1_dims will be (64, 1, 1, 1).
//
// When two shapes are compatible up to broadcasting, for each dimension d,
// the input extents are either equal, or one of them is 1.
//
// This function performs the following for each dimension d:
// - If the extents are equal, then do nothing since the loop that walks over
//   both of the input arrays is correct.
// - Otherwise, one (and only one) of the extents must be 1. Say extent0 is 1
//   and extent1 is e1. Then set extent0 to e1 and stride0 *to 0*. This allows
//   array0 to be referenced *at any index* in dimension d and still access the
//   same slice.
template <int N>
inline void NdArrayDescsForElementwiseBroadcast(const Dims<N>& input0_dims,
                                                const Dims<N>& input1_dims,
                                                NdArrayDesc<N>* desc0_out,
                                                NdArrayDesc<N>* desc1_out) {
  TFLITE_DCHECK(desc0_out != nullptr);
  TFLITE_DCHECK(desc1_out != nullptr);

  // Copy dims to desc.
  for (int i = 0; i < N; ++i) {
    desc0_out->extents[i] = input0_dims.sizes[i];
    desc0_out->strides[i] = input0_dims.strides[i];
    desc1_out->extents[i] = input1_dims.sizes[i];
    desc1_out->strides[i] = input1_dims.strides[i];
  }

  // Walk over each dimension. If the extents are equal do nothing.
  // Otherwise, set the desc with extent 1 to have extent equal to the other and
  // stride 0.
  for (int i = 0; i < N; ++i) {
    const int extent0 = ArraySize(input0_dims, i);
    const int extent1 = ArraySize(input1_dims, i);
    if (extent0 != extent1) {
      if (extent0 == 1) {
        desc0_out->strides[i] = 0;
        desc0_out->extents[i] = extent1;
      } else {
        TFLITE_DCHECK_EQ(extent1, 1);
        desc1_out->strides[i] = 0;
        desc1_out->extents[i] = extent0;
      }
    }
  }
}

inline bool AreSameDims(const Dims<4>& dims1, const Dims<4>& dims2) {
  for (int i = 0; i < 4; i++) {
    if (dims1.sizes[i] != dims2.sizes[i]) {
      return false;
    }
  }
  return true;
}

inline void AddBiasAndEvalActivationFunction(const float* bias_data,
                                             const Dims<4>& bias_dims,
                                             float* array_data,
                                             const Dims<4>& array_dims,
                                             float output_activation_min,
                                             float output_activation_max) {
#ifdef USE_NEON
  gemmlowp::ScopedProfilingLabel label("AddBiasAndEvalActivationFunction");
  const int bias_size = bias_dims.sizes[3] * bias_dims.strides[3];
  const int array_size = array_dims.sizes[3] * array_dims.strides[3];
  TFLITE_DCHECK_EQ((array_size % bias_size), 0);
  float* array_ptr = array_data;
  float* array_end_ptr = array_ptr + array_size;
  const auto activation_min = vdupq_n_f32(output_activation_min);
  const auto activation_max = vdupq_n_f32(output_activation_max);
  for (; array_ptr != array_end_ptr; array_ptr += bias_size) {
    int i = 0;
    for (; i <= bias_size - 16; i += 16) {
      auto b0 = vld1q_f32(bias_data + i);
      auto b1 = vld1q_f32(bias_data + i + 4);
      auto b2 = vld1q_f32(bias_data + i + 8);
      auto b3 = vld1q_f32(bias_data + i + 12);
      auto a0 = vld1q_f32(array_ptr + i);
      auto a1 = vld1q_f32(array_ptr + i + 4);
      auto a2 = vld1q_f32(array_ptr + i + 8);
      auto a3 = vld1q_f32(array_ptr + i + 12);
      auto x0 = vaddq_f32(a0, b0);
      auto x1 = vaddq_f32(a1, b1);
      auto x2 = vaddq_f32(a2, b2);
      auto x3 = vaddq_f32(a3, b3);
      x0 = vmaxq_f32(activation_min, x0);
      x1 = vmaxq_f32(activation_min, x1);
      x2 = vmaxq_f32(activation_min, x2);
      x3 = vmaxq_f32(activation_min, x3);
      x0 = vminq_f32(activation_max, x0);
      x1 = vminq_f32(activation_max, x1);
      x2 = vminq_f32(activation_max, x2);
      x3 = vminq_f32(activation_max, x3);
      vst1q_f32(array_ptr + i, x0);
      vst1q_f32(array_ptr + i + 4, x1);
      vst1q_f32(array_ptr + i + 8, x2);
      vst1q_f32(array_ptr + i + 12, x3);
    }
    for (; i <= bias_size - 4; i += 4) {
      auto b = vld1q_f32(bias_data + i);
      auto a = vld1q_f32(array_ptr + i);
      auto x = vaddq_f32(a, b);
      x = vmaxq_f32(activation_min, x);
      x = vminq_f32(activation_max, x);
      vst1q_f32(array_ptr + i, x);
    }
    for (; i < bias_size; i++) {
      array_ptr[i] = ActivationFunctionWithMinMax(array_ptr[i] + bias_data[i],
                                                  output_activation_min,
                                                  output_activation_max);
    }
  }
#else  // not NEON
  gemmlowp::ScopedProfilingLabel label("AddBiasAndEvalActivationFunction");
  const int bias_size = bias_dims.sizes[3] * bias_dims.strides[3];
  const int array_size = array_dims.sizes[3] * array_dims.strides[3];
  TFLITE_DCHECK_EQ((array_size % bias_size), 0);
  for (int array_offset = 0; array_offset < array_size;
       array_offset += bias_size) {
    for (int i = 0; i < bias_size; i++) {
      array_data[array_offset + i] = ActivationFunctionWithMinMax(
          array_data[array_offset + i] + bias_data[i], output_activation_min,
          output_activation_max);
    }
  }
#endif
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AddBiasAndEvalActivationFunction(const float* bias_data,
                                      const Dims<4>& bias_dims,
                                      float* array_data,
                                      const Dims<4>& array_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  AddBiasAndEvalActivationFunction(bias_data, bias_dims, array_data, array_dims,
                                   output_activation_min,
                                   output_activation_max);
}

template <typename Lhs, typename Rhs, typename Result>
void Gemm(const Eigen::MatrixBase<Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs,
          Eigen::MatrixBase<Result>* result) {
  if (rhs.cols() == 1) {
    gemmlowp::ScopedProfilingLabel label("GEMV");
    result->col(0).noalias() = lhs * rhs.col(0);
  } else {
    gemmlowp::ScopedProfilingLabel label("GEMM");
    result->noalias() = lhs * rhs;
  }
}

inline void optimized_ops_preload_l1_stream(const uint8* ptr) {
#ifdef GEMMLOWP_ARM_64
  asm volatile("prfm pldl1strm, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#else
  gemmlowp::Prefetch(ptr);
#endif
}

inline void optimized_ops_preload_l1_keep(const uint8* ptr) {
#ifdef GEMMLOWP_ARM_64
  asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#else
  gemmlowp::Prefetch(ptr);
#endif
}

#ifdef GEMMLOWP_NEON
// In the common case of batch size 1, a fully-connected node degenerates
// to a matrix*vector product. LSTM cells contain a fully-connected node;
// when quantized, this becomes a special type of GEMV operation where
// the output is 16bit-quantized, thus needs its own special path.
inline void GEMVForLstmCell(const uint8* input_data, const Dims<4>& input_dims,
                            const uint8* weights_data,
                            const Dims<4>& weights_dims,
                            uint8 weights_zero_point, const int32* bias_data,
                            const Dims<4>& bias_dims, int32 accum_multiplier,
                            int accum_shift, int16* output_data,
                            const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("GEMVForLstmCell");
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(weights_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(bias_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  TFLITE_DCHECK_EQ(ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                       ArraySize(output_dims, 3),
                   1);
  const int input_size = input_dims.strides[3];
  const int output_size = MatchingArraySize(weights_dims, 1, output_dims, 0);
  // This special fast path for quantized LSTM cells does not try to support
  // odd sizes that we haven't encountered in any LSTM cell, that would
  // require special code (that would go untested until any LSTM cell
  // exercises it). We just guard our assumptions about size evenness with
  // the following assertions.
  TFLITE_DCHECK(!(output_size % 4));
  TFLITE_DCHECK(!(input_size % 8));
  const int32* bias_ptr = bias_data;
  int16* output_ptr = output_data;
  for (int out = 0; out < output_size; out += 4) {
    int32x4_t acc_0 = vdupq_n_s32(0);
    int32x4_t acc_1 = vdupq_n_s32(0);
    int32x4_t acc_2 = vdupq_n_s32(0);
    int32x4_t acc_3 = vdupq_n_s32(0);
    const int16x8_t input_offset_vec = vdupq_n_s16(-128);
    const int16x8_t weights_offset_vec = vdupq_n_s16(-weights_zero_point);
    int in = 0;
    // Handle 16 levels of depth at a time.
    for (; in <= input_size - 16; in += 16) {
      const uint8x16_t input_val_u8 = vld1q_u8(input_data + in);
      const uint8* weights_ptr = weights_data + in + out * input_size;
      uint8x16_t weights_val_u8_0 = vld1q_u8(weights_ptr + 0 * input_size);
      uint8x16_t weights_val_u8_1 = vld1q_u8(weights_ptr + 1 * input_size);
      uint8x16_t weights_val_u8_2 = vld1q_u8(weights_ptr + 2 * input_size);
      uint8x16_t weights_val_u8_3 = vld1q_u8(weights_ptr + 3 * input_size);
      int16x8_t input_val_0, input_val_1;
      const uint8x8_t low = vget_low_u8(input_val_u8);
      const uint8x8_t high = vget_high_u8(input_val_u8);
      input_val_0 = vreinterpretq_s16_u16(vmovl_u8(low));
      input_val_1 = vreinterpretq_s16_u16(vmovl_u8(high));
      input_val_0 = vaddq_s16(input_val_0, input_offset_vec);
      input_val_1 = vaddq_s16(input_val_1, input_offset_vec);
      int16x8_t weights_val_0_0, weights_val_1_0, weights_val_2_0,
          weights_val_3_0;
      int16x8_t weights_val_0_1, weights_val_1_1, weights_val_2_1,
          weights_val_3_1;
      weights_val_0_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_0))),
          weights_offset_vec);
      weights_val_0_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_0))),
          weights_offset_vec);
      weights_val_1_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_1))),
          weights_offset_vec);
      weights_val_1_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_1))),
          weights_offset_vec);
      weights_val_2_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_2))),
          weights_offset_vec);
      weights_val_2_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_2))),
          weights_offset_vec);
      weights_val_3_0 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(weights_val_u8_3))),
          weights_offset_vec);
      weights_val_3_1 = vaddq_s16(
          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(weights_val_u8_3))),
          weights_offset_vec);
      acc_0 = vmlal_s16(acc_0, vget_low_s16(weights_val_0_0),
                        vget_low_s16(input_val_0));
      acc_1 = vmlal_s16(acc_1, vget_low_s16(weights_val_1_0),
                        vget_low_s16(input_val_0));
      acc_2 = vmlal_s16(acc_2, vget_low_s16(weights_val_2_0),
                        vget_low_s16(input_val_0));
      acc_3 = vmlal_s16(acc_3, vget_low_s16(weights_val_3_0),
                        vget_low_s16(input_val_0));
      acc_0 = vmlal_s16(acc_0, vget_high_s16(weights_val_0_0),
                        vget_high_s16(input_val_0));
      acc_1 = vmlal_s16(acc_1, vget_high_s16(weights_val_1_0),
                        vget_high_s16(input_val_0));
      acc_2 = vmlal_s16(acc_2, vget_high_s16(weights_val_2_0),
                        vget_high_s16(input_val_0));
      acc_3 = vmlal_s16(acc_3, vget_high_s16(weights_val_3_0),
                        vget_high_s16(input_val_0));
      acc_0 = vmlal_s16(acc_0, vget_low_s16(weights_val_0_1),
                        vget_low_s16(input_val_1));
      acc_1 = vmlal_s16(acc_1, vget_low_s16(weights_val_1_1),
                        vget_low_s16(input_val_1));
      acc_2 = vmlal_s16(acc_2, vget_low_s16(weights_val_2_1),
                        vget_low_s16(input_val_1));
      acc_3 = vmlal_s16(acc_3, vget_low_s16(weights_val_3_1),
                        vget_low_s16(input_val_1));
      acc_0 = vmlal_s16(acc_0, vget_high_s16(weights_val_0_1),
                        vget_high_s16(input_val_1));
      acc_1 = vmlal_s16(acc_1, vget_high_s16(weights_val_1_1),
                        vget_high_s16(input_val_1));
      acc_2 = vmlal_s16(acc_2, vget_high_s16(weights_val_2_1),
                        vget_high_s16(input_val_1));
      acc_3 = vmlal_s16(acc_3, vget_high_s16(weights_val_3_1),
                        vget_high_s16(input_val_1));
    }
    // Handle 8 levels of depth at a time.
    for (; in < input_size; in += 8) {
      const uint8x8_t input_val_u8 = vld1_u8(input_data + in);
      const uint8* weights_ptr = weights_data + in + out * input_size;
      uint8x8_t weights_val_u8_0 = vld1_u8(weights_ptr + 0 * input_size);
      uint8x8_t weights_val_u8_1 = vld1_u8(weights_ptr + 1 * input_size);
      uint8x8_t weights_val_u8_2 = vld1_u8(weights_ptr + 2 * input_size);
      uint8x8_t weights_val_u8_3 = vld1_u8(weights_ptr + 3 * input_size);
      int16x8_t input_val;
      input_val = vreinterpretq_s16_u16(vmovl_u8(input_val_u8));
      input_val = vaddq_s16(input_val, input_offset_vec);
      int16x8_t weights_val_0, weights_val_1, weights_val_2, weights_val_3;
      weights_val_0 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_0)),
                    weights_offset_vec);
      weights_val_1 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_1)),
                    weights_offset_vec);
      weights_val_2 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_2)),
                    weights_offset_vec);
      weights_val_3 =
          vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(weights_val_u8_3)),
                    weights_offset_vec);
      acc_0 = vmlal_s16(acc_0, vget_low_s16(weights_val_0),
                        vget_low_s16(input_val));
      acc_1 = vmlal_s16(acc_1, vget_low_s16(weights_val_1),
                        vget_low_s16(input_val));
      acc_2 = vmlal_s16(acc_2, vget_low_s16(weights_val_2),
                        vget_low_s16(input_val));
      acc_3 = vmlal_s16(acc_3, vget_low_s16(weights_val_3),
                        vget_low_s16(input_val));
      acc_0 = vmlal_s16(acc_0, vget_high_s16(weights_val_0),
                        vget_high_s16(input_val));
      acc_1 = vmlal_s16(acc_1, vget_high_s16(weights_val_1),
                        vget_high_s16(input_val));
      acc_2 = vmlal_s16(acc_2, vget_high_s16(weights_val_2),
                        vget_high_s16(input_val));
      acc_3 = vmlal_s16(acc_3, vget_high_s16(weights_val_3),
                        vget_high_s16(input_val));
    }
    // Horizontally reduce accumulators
    int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,
        pairwise_reduced_acc_2, pairwise_reduced_acc_3;
    pairwise_reduced_acc_0 =
        vpadd_s32(vget_low_s32(acc_0), vget_high_s32(acc_0));
    pairwise_reduced_acc_1 =
        vpadd_s32(vget_low_s32(acc_1), vget_high_s32(acc_1));
    pairwise_reduced_acc_2 =
        vpadd_s32(vget_low_s32(acc_2), vget_high_s32(acc_2));
    pairwise_reduced_acc_3 =
        vpadd_s32(vget_low_s32(acc_3), vget_high_s32(acc_3));
    const int32x2_t reduced_lo =
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
    const int32x2_t reduced_hi =
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
    // Add bias values.
    int32x4_t bias_vec = vld1q_s32(bias_ptr);
    bias_ptr += 4;
    reduced = vaddq_s32(reduced, bias_vec);
    int left_shift = accum_shift > 0 ? accum_shift : 0;
    int right_shift = accum_shift > 0 ? 0 : -accum_shift;
    reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));
    // Multiply by the fixed-point multiplier.
    reduced = vqrdmulhq_n_s32(reduced, accum_multiplier);
    // Rounding-shift-right.
    using gemmlowp::RoundingDivideByPOT;
    reduced = RoundingDivideByPOT(reduced, right_shift);
    // Narrow values down to 16 bit signed.
    const int16x4_t res16 = vqmovn_s32(reduced);
    vst1_s16(output_ptr, res16);
    output_ptr += 4;
  }
}
#endif

#ifdef GEMMLOWP_NEON
inline void GEMVForLstmCellWithSymmetricRange(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 accum_multiplier,
    int accum_shift, int16* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("GEMVForLstmCellWithSymmetricRange");
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(weights_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(bias_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  TFLITE_DCHECK_EQ(ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                       ArraySize(output_dims, 3),
                   1);
  const int input_size = input_dims.strides[3];
  const int output_size = MatchingArraySize(weights_dims, 1, output_dims, 0);
  // This special fast path for quantized LSTM cells does not try to support
  // odd sizes that we haven't encountered in any LSTM cell, that would
  // require special code (that would go untested until any LSTM cell
  // exercises it). We just guard our assumptions about size evenness with
  // the following assertions.
  TFLITE_DCHECK(!(output_size % 4));
  TFLITE_DCHECK(!(input_size % 64));
  const int32* bias_ptr = bias_data;
  int16* output_ptr = output_data;
  const uint8x16_t signbit = vdupq_n_u8(0x80);
  for (int in = 0; in < input_size; in += 32) {
    optimized_ops_preload_l1_keep(input_data + in);
  }
  const int left_shift = accum_shift > 0 ? accum_shift : 0;
  const int right_shift = accum_shift > 0 ? 0 : -accum_shift;
  for (int out = 0; out < output_size; out += 4) {
    // Load the bias values
    int32x4_t bias_vec = vld1q_s32(bias_ptr);
    bias_ptr += 4;

    // Clear accumulators. We use 2 accumulator registers per row,
    // for 4 rows. row_accumRN is the N-th accumulator for row R.
    int32x4_t row_accum00 = vdupq_n_s32(0);
    int32x4_t row_accum01 = vdupq_n_s32(0);
    int32x4_t row_accum10 = vdupq_n_s32(0);
    int32x4_t row_accum11 = vdupq_n_s32(0);
    int32x4_t row_accum20 = vdupq_n_s32(0);
    int32x4_t row_accum21 = vdupq_n_s32(0);
    int32x4_t row_accum30 = vdupq_n_s32(0);
    int32x4_t row_accum31 = vdupq_n_s32(0);

    // kReadAhead parametrizes how far ahead we prefetch weights into L1 cache.
    const int kReadAhead = 512;
    // Prefetch the first weights values.
    for (int k = 0; k < kReadAhead; k += 64) {
      optimized_ops_preload_l1_stream(weights_data + (out + 0) * input_size +
                                      k);
      optimized_ops_preload_l1_stream(weights_data + (out + 1) * input_size +
                                      k);
      optimized_ops_preload_l1_stream(weights_data + (out + 2) * input_size +
                                      k);
      optimized_ops_preload_l1_stream(weights_data + (out + 3) * input_size +
                                      k);
    }
    // Loop along the rows, handling 64 bytes per iteration because that's
    // cache line size on most current ARM-architecture CPUs.
    for (int in = 0; in < input_size; in += 64) {
      // Prefetch some future weights values.
      optimized_ops_preload_l1_stream(weights_data + (out + 0) * input_size +
                                      in + kReadAhead);
      optimized_ops_preload_l1_stream(weights_data + (out + 1) * input_size +
                                      in + kReadAhead);
      optimized_ops_preload_l1_stream(weights_data + (out + 2) * input_size +
                                      in + kReadAhead);
      optimized_ops_preload_l1_stream(weights_data + (out + 3) * input_size +
                                      in + kReadAhead);

      // We will use 2 local 16-bit accumulators per row, for 2 rows.
      // See below (*) for the rationale of processing only 2 rows at a time.
      // local_accumRN is the N-th local accumulator for row R.
      int16x8_t local_accum00;
      int16x8_t local_accum01;
      int16x8_t local_accum10;
      int16x8_t local_accum11;

      // Load 64 bytes of input activations values. Convert to signed int8
      // by flipping the sign bit (i.e. subtracting 128, the required
      // zero_point value).
      int8x16_t input0 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 0)));
      int8x16_t input1 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 1)));
      int8x16_t input2 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 2)));
      int8x16_t input3 = vreinterpretq_s8_u8(
          veorq_u8(signbit, vld1q_u8(input_data + in + 16 * 3)));

      // Beginning of the core accumulation. Notice how while we have 4
      // rows to process, this code is taking care of only 2 rows at a time,
      // thus being divided into two parts looking similar ("Rows 0 and 1" and
      // "Rows 2 and 3").
      //
      // (*) The rationale for handling only 2 rows at a time is to avoid
      // cache aliasing issues on 4-way set-associative L1-cache CPUs, such
      // as Cortex-A53. With sufficiently large, power-of-two matrix dimensions,
      // we may find ourselves in a situation where rows alias each other in
      // the L1 cache, and moreover may also mutually alias with the input
      // activations. If we try to load 4 rows at a time, together with the
      // input activations, that may be 5 mutually-aliasing vectors, resulting
      // in constant mutual eviction from L1 cache. Handling 2 rows at a time
      // here largely mitigates these issues, and seems at least to be very
      // effective on Cortex-A53:
      //                          Before       After
      // big (Cortex-A73)         2.85 ms      2.85 ms
      // little (Cortex-A53)      11.0 ms      5.16 ms

      // Rows 0 and 1:
      // Load 64 bytes of weights values from each row. Convert to signed int8
      // by flipping the sign bit (i.e. subtracting 128, the required
      // zero_point value).
      int8x16_t weights00 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 0)));
      int8x16_t weights01 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 1)));
      int8x16_t weights02 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 2)));
      int8x16_t weights03 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 0) * input_size + in + 16 * 3)));
      int8x16_t weights10 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 0)));
      int8x16_t weights11 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 1)));
      int8x16_t weights12 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 2)));
      int8x16_t weights13 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 1) * input_size + in + 16 * 3)));
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights00), vget_low_s8(input0));
      local_accum01 = vmull_s8(vget_low_s8(weights01), vget_low_s8(input1));
      local_accum10 = vmull_s8(vget_low_s8(weights10), vget_low_s8(input0));
      local_accum11 = vmull_s8(vget_low_s8(weights11), vget_low_s8(input1));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights00),
                               vget_high_s8(input0));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights01),
                               vget_high_s8(input1));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights10),
                               vget_high_s8(input0));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights11),
                               vget_high_s8(input1));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum00 = vpadalq_s16(row_accum00, local_accum00);
      row_accum01 = vpadalq_s16(row_accum01, local_accum01);
      row_accum10 = vpadalq_s16(row_accum10, local_accum10);
      row_accum11 = vpadalq_s16(row_accum11, local_accum11);
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights02), vget_low_s8(input2));
      local_accum01 = vmull_s8(vget_low_s8(weights03), vget_low_s8(input3));
      local_accum10 = vmull_s8(vget_low_s8(weights12), vget_low_s8(input2));
      local_accum11 = vmull_s8(vget_low_s8(weights13), vget_low_s8(input3));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights02),
                               vget_high_s8(input2));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights03),
                               vget_high_s8(input3));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights12),
                               vget_high_s8(input2));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights13),
                               vget_high_s8(input3));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum00 = vpadalq_s16(row_accum00, local_accum00);
      row_accum01 = vpadalq_s16(row_accum01, local_accum01);
      row_accum10 = vpadalq_s16(row_accum10, local_accum10);
      row_accum11 = vpadalq_s16(row_accum11, local_accum11);

      // Rows 2 and 3:
      // Load 64 bytes of weights values from each row. Convert to signed int8
      // by flipping the sign bit (i.e. subtracting 128, the required
      // zero_point value).
      weights00 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 0)));
      weights01 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 1)));
      weights02 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 2)));
      weights03 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 2) * input_size + in + 16 * 3)));
      weights10 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 0)));
      weights11 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 1)));
      weights12 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 2)));
      weights13 = vreinterpretq_s8_u8(veorq_u8(
          signbit,
          vld1q_u8(weights_data + (out + 3) * input_size + in + 16 * 3)));
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights00), vget_low_s8(input0));
      local_accum01 = vmull_s8(vget_low_s8(weights01), vget_low_s8(input1));
      local_accum10 = vmull_s8(vget_low_s8(weights10), vget_low_s8(input0));
      local_accum11 = vmull_s8(vget_low_s8(weights11), vget_low_s8(input1));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights00),
                               vget_high_s8(input0));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights01),
                               vget_high_s8(input1));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights10),
                               vget_high_s8(input0));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights11),
                               vget_high_s8(input1));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum20 = vpadalq_s16(row_accum20, local_accum00);
      row_accum21 = vpadalq_s16(row_accum21, local_accum01);
      row_accum30 = vpadalq_s16(row_accum30, local_accum10);
      row_accum31 = vpadalq_s16(row_accum31, local_accum11);
      // Multiply-accumulate into local 16-bit accumulators.
      // We can accumulate two products without overflow because weights are
      // required to never be -128, so each product is at most 127^2 in absolute
      // value.
      local_accum00 = vmull_s8(vget_low_s8(weights02), vget_low_s8(input2));
      local_accum01 = vmull_s8(vget_low_s8(weights03), vget_low_s8(input3));
      local_accum10 = vmull_s8(vget_low_s8(weights12), vget_low_s8(input2));
      local_accum11 = vmull_s8(vget_low_s8(weights13), vget_low_s8(input3));
      local_accum00 = vmlal_s8(local_accum00, vget_high_s8(weights02),
                               vget_high_s8(input2));
      local_accum01 = vmlal_s8(local_accum01, vget_high_s8(weights03),
                               vget_high_s8(input3));
      local_accum10 = vmlal_s8(local_accum10, vget_high_s8(weights12),
                               vget_high_s8(input2));
      local_accum11 = vmlal_s8(local_accum11, vget_high_s8(weights13),
                               vget_high_s8(input3));
      // Pairwise add and accumulate into 32-bit accumulators
      row_accum20 = vpadalq_s16(row_accum20, local_accum00);
      row_accum21 = vpadalq_s16(row_accum21, local_accum01);
      row_accum30 = vpadalq_s16(row_accum30, local_accum10);
      row_accum31 = vpadalq_s16(row_accum31, local_accum11);
    }

    row_accum00 = vaddq_s32(row_accum00, row_accum01);
    row_accum10 = vaddq_s32(row_accum10, row_accum11);
    row_accum20 = vaddq_s32(row_accum20, row_accum21);
    row_accum30 = vaddq_s32(row_accum30, row_accum31);
    // Horizontally reduce accumulators
    int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,
        pairwise_reduced_acc_2, pairwise_reduced_acc_3;
    pairwise_reduced_acc_0 =
        vpadd_s32(vget_low_s32(row_accum00), vget_high_s32(row_accum00));
    pairwise_reduced_acc_1 =
        vpadd_s32(vget_low_s32(row_accum10), vget_high_s32(row_accum10));
    pairwise_reduced_acc_2 =
        vpadd_s32(vget_low_s32(row_accum20), vget_high_s32(row_accum20));
    pairwise_reduced_acc_3 =
        vpadd_s32(vget_low_s32(row_accum30), vget_high_s32(row_accum30));
    const int32x2_t reduced_lo =
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
    const int32x2_t reduced_hi =
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
    // Add bias values.
    reduced = vaddq_s32(reduced, bias_vec);
    reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));
    // Multiply by the fixed-point multiplier.
    reduced = vqrdmulhq_n_s32(reduced, accum_multiplier);
    // Rounding-shift-right.
    using gemmlowp::RoundingDivideByPOT;
    reduced = RoundingDivideByPOT(reduced, right_shift);
    // Narrow values down to 16 bit signed.
    const int16x4_t res16 = vqmovn_s32(reduced);
    vst1_s16(output_ptr, res16);
    output_ptr += 4;
  }
}
#endif

inline void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                           const float* weights_data,
                           const Dims<4>& weights_dims, const float* bias_data,
                           const Dims<4>& bias_dims,
                           float output_activation_min,
                           float output_activation_max, float* output_data,
                           const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("FullyConnected");
  // TODO(b/62193649): this convoluted shape computation (determining
  // input_rows from the weights_dims, then MapAsMatrixWithGivenNumberOfRows)
  // is because the current --variable_batch hack consists in overwriting the
  // 3rd dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  // When that is fixed, this should become:
  // const auto input_matrix_map =
  //     MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  const int input_rows = ArraySize(weights_dims, 0);
  const auto input_matrix_map =
      MapAsMatrixWithGivenNumberOfRows(input_data, input_dims, input_rows);
  const auto filter_matrix_map =
      MapAsMatrixWithFirstDimAsRows(weights_data, weights_dims);
  auto output_matrix_map =
      MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), input_matrix_map, &output_matrix_map);
  AddBiasAndEvalActivationFunction(bias_data, bias_dims, output_data,
                                   output_dims, output_activation_min,
                                   output_activation_max);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const float* input_data, const Dims<4>& input_dims,
                    const float* weights_data, const Dims<4>& weights_dims,
                    const float* bias_data, const Dims<4>& bias_dims,
                    float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  FullyConnected(input_data, input_dims, weights_data, weights_dims, bias_data,
                 bias_dims, output_activation_min, output_activation_max,
                 output_data, output_dims);
}

#ifdef USE_NEON
inline void FullyConnectedAsGEMV(
    const uint8* input_data, const Dims<4>& input_dims, int32 input_offset,
    const uint8* filter_data, const Dims<4>& filter_dims, int32 filter_offset,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, uint8* output_data,
    const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("FullyConnectedAsGEMV/8bit");
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(filter_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(bias_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  TFLITE_DCHECK_EQ(ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                       ArraySize(output_dims, 3),
                   1);
  const int input_size = input_dims.strides[3];
  const int output_size = MatchingArraySize(filter_dims, 1, output_dims, 0);
  static constexpr int kPeel = 4;
  for (int k = 0; k < input_size; k += 64) {
    optimized_ops_preload_l1_stream(input_data + k);
  }
  for (int k = 0; k < kPeel * input_size; k += 64) {
    optimized_ops_preload_l1_stream(filter_data + k);
  }
  TFLITE_DCHECK(!(output_size % kPeel));
  const int32* bias_ptr = bias_data;
  uint8* output_ptr = output_data;
  for (int out = 0; out < output_size; out += kPeel) {
    int32x4_t acc[kPeel];
    for (int k = 0; k < kPeel; k++) {
      acc[k] = vdupq_n_s32(0);
    }
    const int16x8_t input_offset_vec = vdupq_n_s16(input_offset);
    const int16x8_t filter_offset_vec = vdupq_n_s16(filter_offset);
    int in = 0;
    for (; in <= input_size - 16; in += 16) {
      const uint8x16_t input_val_u8 = vld1q_u8(input_data + in);
      uint8x16_t filter_val_u8[kPeel];
      for (int k = 0; k < kPeel; k++) {
        const uint8* filter_ptr = filter_data + in + (out + k) * input_size;
        filter_val_u8[k] = vld1q_u8(filter_ptr);
        optimized_ops_preload_l1_stream(filter_ptr + 64);
      }
      int16x8_t input_val[2];
      const uint8x8_t low = vget_low_u8(input_val_u8);
      const uint8x8_t high = vget_high_u8(input_val_u8);
      input_val[0] = vreinterpretq_s16_u16(vmovl_u8(low));
      input_val[1] = vreinterpretq_s16_u16(vmovl_u8(high));
      input_val[0] = vaddq_s16(input_val[0], input_offset_vec);
      input_val[1] = vaddq_s16(input_val[1], input_offset_vec);
      int16x8_t filter_val[kPeel][2];
      for (int k = 0; k < kPeel; k++) {
        const uint8x8_t low = vget_low_u8(filter_val_u8[k]);
        const uint8x8_t high = vget_high_u8(filter_val_u8[k]);
        filter_val[k][0] = vreinterpretq_s16_u16(vmovl_u8(low));
        filter_val[k][1] = vreinterpretq_s16_u16(vmovl_u8(high));
        filter_val[k][0] = vaddq_s16(filter_val[k][0], filter_offset_vec);
        filter_val[k][1] = vaddq_s16(filter_val[k][1], filter_offset_vec);
      }
      for (int p = 0; p < 2; p++) {
        for (int k = 0; k < kPeel; k++) {
          acc[k] = vmlal_s16(acc[k], vget_low_s16(filter_val[k][p]),
                             vget_low_s16(input_val[p]));
        }
        for (int k = 0; k < kPeel; k++) {
          acc[k] = vmlal_s16(acc[k], vget_high_s16(filter_val[k][p]),
                             vget_high_s16(input_val[p]));
        }
      }
    }
    for (; in <= input_size - 8; in += 8) {
      const uint8x8_t input_val_u8 = vld1_u8(input_data + in);
      uint8x8_t filter_val_u8[kPeel];
      for (int k = 0; k < kPeel; k++) {
        const uint8* filter_ptr = filter_data + in + (out + k) * input_size;
        filter_val_u8[k] = vld1_u8(filter_ptr);
      }
      int16x8_t input_val;
      input_val = vreinterpretq_s16_u16(vmovl_u8(input_val_u8));
      input_val = vaddq_s16(input_val, input_offset_vec);
      int16x8_t filter_val[kPeel];
      for (int k = 0; k < kPeel; k++) {
        filter_val[k] = vreinterpretq_s16_u16(vmovl_u8(filter_val_u8[k]));
        filter_val[k] = vaddq_s16(filter_val[k], filter_offset_vec);
      }
      for (int k = 0; k < kPeel; k++) {
        acc[k] = vmlal_s16(acc[k], vget_low_s16(filter_val[k]),
                           vget_low_s16(input_val));
      }
      for (int k = 0; k < kPeel; k++) {
        acc[k] = vmlal_s16(acc[k], vget_high_s16(filter_val[k]),
                           vget_high_s16(input_val));
      }
    }
    if (in < input_size) {
      int32 buf[4 * kPeel];
      for (int k = 0; k < 4; k++) {
        vst1q_s32(buf + 4 * k, acc[k]);
      }
      for (; in < input_size; in++) {
        int lane = (in + 8 - input_size) % 4;
        const int32 input_val = input_data[in] + input_offset;
        for (int k = 0; k < kPeel; k++) {
          int32 filter_val =
              filter_data[in + (out + k) * input_size] + filter_offset;
          buf[lane + 4 * k] += filter_val * input_val;
        }
      }
      for (int k = 0; k < 4; k++) {
        acc[k] = vld1q_s32(buf + 4 * k);
      }
    }

    // Horizontally reduce accumulators
    int32x2_t pairwise_reduced_acc[kPeel];
    for (int k = 0; k < kPeel; k++) {
      pairwise_reduced_acc[k] =
          vpadd_s32(vget_low_s32(acc[k]), vget_high_s32(acc[k]));
    }
    static_assert(kPeel == 4, "the code below currently assumes kPeel = 4");
    const int32x2_t reduced_lo =
        vpadd_s32(pairwise_reduced_acc[0], pairwise_reduced_acc[1]);
    const int32x2_t reduced_hi =
        vpadd_s32(pairwise_reduced_acc[2], pairwise_reduced_acc[3]);
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
    // Add bias values.
    int32x4_t bias_vec = vld1q_s32(bias_ptr);
    bias_ptr += 4;
    reduced = vaddq_s32(reduced, bias_vec);
    // Multiply by the fixed-point multiplier.
    reduced = vqrdmulhq_n_s32(reduced, output_multiplier);
    // Rounding-shift-right.
    using gemmlowp::RoundingDivideByPOT;
    reduced = RoundingDivideByPOT(reduced, output_shift);
    // Add the output offset.
    const int32x4_t output_offset_vec = vdupq_n_s32(output_offset);
    reduced = vaddq_s32(reduced, output_offset_vec);
    // Narrow values down to 16 bit signed.
    const int16x4_t res16 = vqmovn_s32(reduced);
    // Narrow values down to 8 bit unsigned, saturating.
    uint8x8_t res8 = vqmovun_s16(vcombine_s16(res16, res16));
    // Apply the clamping from the activation function
    res8 = vmax_u8(res8, vdup_n_u8(output_activation_min));
    res8 = vmin_u8(res8, vdup_n_u8(output_activation_max));
    // Store results to destination. Assumes 32bit alignment.
    vst1_lane_u32(reinterpret_cast<uint32*>(output_ptr),
                  vreinterpret_u32_u8(res8), 0);
    output_ptr += kPeel;
  }
}
#endif  // USE_NEON

struct GemmlowpOutputPipeline {
  typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
      ColVectorMap;
  typedef std::tuple<
      gemmlowp::OutputStageBiasAddition<ColVectorMap>,
      gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
      gemmlowp::OutputStageClamp, gemmlowp::OutputStageSaturatingCastToUint8>
      Pipeline;
  static Pipeline Make(const int32* bias_data, int output_rows,
                       int32 output_offset, int32 output_multiplier,
                       int output_shift, int32 output_activation_min,
                       int32 output_activation_max) {
    ColVectorMap bias_vector(bias_data, output_rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
        quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_offset;
    quantize_down_stage.result_fixedpoint_multiplier = output_multiplier;
    quantize_down_stage.result_shift = output_shift;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = output_activation_min;
    clamp_stage.max = output_activation_max;
    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage,
                           clamp_stage, saturating_cast_stage);
  }
};

inline void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                           int32 input_offset, const uint8* filter_data,
                           const Dims<4>& filter_dims, int32 filter_offset,
                           const int32* bias_data, const Dims<4>& bias_dims,
                           int32 output_offset, int32 output_multiplier,
                           int output_shift, int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data,
                           const Dims<4>& output_dims,
                           gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("FullyConnected/8bit");
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
#ifdef USE_NEON
  const int output_size = MatchingArraySize(filter_dims, 1, output_dims, 0);
  if (batches == 1 && !(output_size % 4)) {
    return FullyConnectedAsGEMV(
        input_data, input_dims, input_offset, filter_data, filter_dims,
        filter_offset, bias_data, bias_dims, output_offset, output_multiplier,
        output_shift, output_activation_min, output_activation_max, output_data,
        output_dims);
  }
#endif  // USE_NEON
  const int filter_rows = filter_dims.sizes[1];
  const int filter_cols = filter_dims.sizes[0];
  TFLITE_DCHECK_EQ(filter_dims.sizes[2], 1);
  TFLITE_DCHECK_EQ(filter_dims.sizes[3], 1);
  const int output_rows = output_dims.sizes[0];
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[0], output_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[1], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[2], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[3], 1);

  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, output_rows, filter_cols, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, filter_cols, batches, filter_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, batches, output_rows);
  const auto& output_pipeline = GemmlowpOutputPipeline::Make(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix, filter_offset,
      input_offset, output_pipeline);
}

inline void FullyConnected(
    const uint8* input_data, const Dims<4>& input_dims, int32 input_offset,
    const uint8* filter_data, const Dims<4>& filter_dims, int32 filter_offset,
    const int32* bias_data_int32, const Dims<4>& bias_dims, int32 output_offset,
    int32 output_multiplier, int output_shift, int32 output_activation_min,
    int32 output_activation_max, int16* output_data, const Dims<4>& output_dims,
    gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("FullyConnected/Uint8Int16");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.
  (void)gemm_context;  // only used in properly optimized code.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(output_offset, 0);

  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 1, output_dims, 0);
  const int accum_depth = ArraySize(filter_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(filter_dims));

  // Implementation of the fully connected node suited to the inside of an LSTM
  // cell. The operands are 8-bit integers, the accumulators are internally
  // 32bit integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
#ifdef GEMMLOWP_NEON
  if (batches == 1 && input_offset == -128 && output_activation_min == -32768 &&
      output_activation_max == 32767) {
    if (filter_offset == -128 && !(output_depth % 4) && !(accum_depth % 64)) {
      GEMVForLstmCellWithSymmetricRange(input_data, input_dims, filter_data,
                                        filter_dims, bias_data_int32, bias_dims,
                                        output_multiplier, -output_shift,
                                        output_data, output_dims);
      return;
    }
    if (!(output_depth % 4) && !(accum_depth % 8)) {
      GEMVForLstmCell(input_data, input_dims, filter_data, filter_dims,
                      filter_offset, bias_data_int32, bias_dims,
                      output_multiplier, -output_shift, output_data,
                      output_dims);
      return;
    }
  }
#endif
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> weights_matrix(
      filter_data, output_depth, accum_depth);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, accum_depth, batches);
  gemmlowp::MatrixMap<int16, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_depth, batches);
  typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
      ColVectorMap;
  ColVectorMap bias_vector(bias_data_int32, output_depth);
  gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
  bias_addition_stage.bias_vector = bias_vector;
  gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent scale_stage;
  scale_stage.result_offset_after_shift = 0;
  scale_stage.result_fixedpoint_multiplier = output_multiplier;
  // Note that this shift is negated wrt ordinary FC.
  scale_stage.result_exponent = -output_shift;
  gemmlowp::OutputStageClamp clamp_stage;
  clamp_stage.min = output_activation_min;
  clamp_stage.max = output_activation_max;
  gemmlowp::OutputStageSaturatingCastToInt16 saturating_cast_int16_stage;
  auto output_pipeline =
      std::make_tuple(bias_addition_stage, scale_stage, clamp_stage,
                      saturating_cast_int16_stage);
  gemmlowp::GemmWithOutputPipeline<uint8, int16,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, weights_matrix, input_matrix, &output_matrix, filter_offset,
      input_offset, output_pipeline);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void FullyConnected(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_offset, const uint8* filter_data,
                    const Dims<4>& filter_dims, int32 filter_offset,
                    const int32* bias_data, const Dims<4>& bias_dims,
                    int32 output_offset, int32 output_multiplier,
                    int output_shift, int32 output_activation_min,
                    int32 output_activation_max, uint8* output_data,
                    const Dims<4>& output_dims,
                    gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  FullyConnected(input_data, input_dims, input_offset, filter_data, filter_dims,
                 filter_offset, bias_data, bias_dims, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data, output_dims, gemm_context);
}

// Internal function doing the actual arithmetic work for
// ExperimentalShuffledFullyConnected.
// May be called either directly by it (single-threaded case) or may be used
// as the 'task' for worker threads to run (multi-threaded case, see
// ExperimentalShuffledFullyConnectedWorkerTask below).
inline void ExperimentalShuffledFullyConnectedWorkerImpl(
    const uint8* shuffled_input_workspace_data,
    const int8* shuffled_weights_data, int batches, int output_depth,
    int output_stride, int accum_depth, const int32* bias_data,
    int32 output_multiplier, int output_shift, int16* output_data) {
#if defined USE_NEON
  const int8* shuffled_weights_ptr = shuffled_weights_data;
  if (batches == 1) {
    const int right_shift = output_shift > 0 ? output_shift : 0;
    const int left_shift = output_shift > 0 ? 0 : -output_shift;
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
    const int right_shift = output_shift > 0 ? output_shift : 0;
    const int left_shift = output_shift > 0 ? 0 : -output_shift;
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
        acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                            -output_shift);
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
                                              -output_shift);
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

// Wraps ExperimentalShuffledFullyConnectedWorkerImpl into a Task class
// to allow using gemmlowp's threadpool.
struct ExperimentalShuffledFullyConnectedWorkerTask : gemmlowp::Task {
  ExperimentalShuffledFullyConnectedWorkerTask(
      const uint8* input_data, const int8* shuffled_weights_data, int batches,
      int output_depth, int output_stride, int accum_depth,
      const int32* bias_data, int32 output_multiplier, int output_shift,
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
    ExperimentalShuffledFullyConnectedWorkerImpl(
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

inline void ExperimentalShuffledFullyConnected(
    const uint8* input_data, const Dims<4>& input_dims,
    const uint8* shuffled_weights_data, const Dims<4>& weights_dims,
    const int32* bias_data, const Dims<4>& bias_dims, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    int16* output_data, const Dims<4>& output_dims,
    uint8* shuffled_input_workspace_data, gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label(
      "ExperimentalShuffledFullyConnected/8bit");
  (void)gemm_context;  // only used in optimized code.
  TFLITE_DCHECK_EQ(output_activation_min, -32768);
  TFLITE_DCHECK_EQ(output_activation_max, 32767);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int batches = ArraySize(output_dims, 1) * ArraySize(output_dims, 2) *
                      ArraySize(output_dims, 3);
  const int output_depth = MatchingArraySize(weights_dims, 1, output_dims, 0);
  const int accum_depth = ArraySize(weights_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(weights_dims));
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
  const int thread_count = gemmlowp::HowManyThreads<kKernelRows>(
      gemm_context->max_num_threads(), output_depth, batches, accum_depth);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    ExperimentalShuffledFullyConnectedWorkerImpl(
        shuffled_input_workspace_data, int8_shuffled_weights_data, batches,
        output_depth, output_depth, accum_depth, bias_data, output_multiplier,
        output_shift, output_data);
    return;
  }

  // Multi-threaded case: use the gemmlowp context's threadpool.
  TFLITE_DCHECK_GT(thread_count, 1);
  std::vector<gemmlowp::Task*> tasks(thread_count);
  const int kRowsPerWorker =
      gemmlowp::RoundUp<kKernelRows>(output_depth / thread_count);
  int row_start = 0;
  for (int i = 0; i < thread_count; i++) {
    int row_end = std::min(output_depth, row_start + kRowsPerWorker);
    tasks[i] = new ExperimentalShuffledFullyConnectedWorkerTask(
        shuffled_input_workspace_data,
        int8_shuffled_weights_data + row_start * accum_depth, batches,
        row_end - row_start, output_depth, accum_depth, bias_data + row_start,
        output_multiplier, output_shift, output_data + row_start);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_depth);
  gemm_context->workers_pool()->Execute(tasks);
}

template <typename T>
inline void ExtractPatchIntoBufferColumn(
    const Dims<4>& input_dims, int w, int h, int b, int kheight, int kwidth,
    int stride_width, int stride_height, int pad_width, int pad_height,
    int in_width, int in_height, int in_depth, int single_buffer_length,
    int buffer_id, const T* in_data, T* conv_buffer_data, uint8 byte_zero) {
  gemmlowp::ScopedProfilingLabel label("ExtractPatchIntoBufferColumn");
  // This chunk of code reshapes all the inputs corresponding to
  // output (b, h, w) to a column vector in conv_buffer(:, buffer_id).
  const int kwidth_times_indepth = kwidth * in_depth;
  const int inwidth_times_indepth = in_width * in_depth;
  const int ih_ungated_start = h * stride_height - pad_height;
  const int ih_ungated_end = (ih_ungated_start + kheight);
  const int ih_end = std::min(ih_ungated_end, in_height);
  const int iw_ungated_start = w * stride_width - pad_width;
  const int iw_ungated_end = (iw_ungated_start + kwidth);
  const int iw_end = std::min(iw_ungated_end, in_width);
  // If the patch is off the edge of the input image, skip writing those rows
  // and columns from the patch into the output array.
  const int h_offset = std::max(0, -ih_ungated_start);
  const int w_offset = std::max(0, -iw_ungated_start);
  const int ih_start = std::max(0, ih_ungated_start);
  const int iw_start = std::max(0, iw_ungated_start);
  const int single_row_num =
      std::min(kwidth - w_offset, in_width - iw_start) * in_depth;
  const int output_row_offset = (buffer_id * single_buffer_length);
  int out_offset =
      output_row_offset + (h_offset * kwidth + w_offset) * in_depth;
  int in_offset = Offset(input_dims, 0, iw_start, ih_start, b);

  // Express all of the calculations as padding around the input patch.
  const int top_padding = h_offset;
  const int bottom_padding = (ih_ungated_end - ih_end);
  const int left_padding = w_offset;
  const int right_padding = (iw_ungated_end - iw_end);
  assert(single_row_num ==
         ((kwidth - (left_padding + right_padding)) * in_depth));

  // Write out zeroes to the elements representing the top rows of the input
  // patch that are off the edge of the input image.
  if (top_padding > 0) {
    const int top_row_elements = (top_padding * kwidth * in_depth);
    memset(conv_buffer_data + output_row_offset, byte_zero,
           (top_row_elements * sizeof(T)));
  }

  // If the patch is on the interior of the input image horizontally, just copy
  // over the rows sequentially, otherwise add zero padding at the start or end.
  if ((left_padding == 0) && (right_padding == 0)) {
    for (int ih = ih_start; ih < ih_end; ++ih) {
      memcpy(conv_buffer_data + out_offset, in_data + in_offset,
             single_row_num * sizeof(T));
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  } else {
    for (int ih = ih_start; ih < ih_end; ++ih) {
      if (left_padding > 0) {
        const int left_start = (out_offset - (left_padding * in_depth));
        memset(conv_buffer_data + left_start, byte_zero,
               (left_padding * in_depth * sizeof(T)));
      }
      memcpy(conv_buffer_data + out_offset, in_data + in_offset,
             single_row_num * sizeof(T));
      if (right_padding > 0) {
        const int right_start = (out_offset + single_row_num);
        memset(conv_buffer_data + right_start, byte_zero,
               (right_padding * in_depth * sizeof(T)));
      }
      out_offset += kwidth_times_indepth;
      in_offset += inwidth_times_indepth;
    }
  }

  // If the bottom of the patch falls off the input image, pad the values
  // representing those input rows with zeroes.
  if (bottom_padding > 0) {
    const int bottom_row_elements = (bottom_padding * kwidth * in_depth);
    const int bottom_start =
        output_row_offset +
        ((top_padding + (ih_end - ih_start)) * kwidth * in_depth);
    memset(conv_buffer_data + bottom_start, byte_zero,
           (bottom_row_elements * sizeof(T)));
  }
}

template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride_width,
            int stride_height, int pad_width, int pad_height, int kheight,
            int kwidth, uint8 byte_zero, T* output_data,
            const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Im2col");
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = ArraySize(input_dims, 0);
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);
  const int output_depth = ArraySize(output_dims, 0);
  const int output_width = ArraySize(output_dims, 1);
  const int output_height = ArraySize(output_dims, 2);

  int buffer_id = 0;
  // Loop over the output nodes.
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        ExtractPatchIntoBufferColumn(
            input_dims, w, h, b, kheight, kwidth, stride_width, stride_height,
            pad_width, pad_height, input_width, input_height, input_depth,
            output_depth, buffer_id, input_data, output_data, byte_zero);
        ++buffer_id;
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int kheight, int kwidth,
            uint8 byte_zero, T* output_data, const Dims<4>& output_dims) {
  Im2col(input_data, input_dims, stride, stride, pad_width, pad_height, kheight,
         kwidth, byte_zero, output_data, output_dims);
}

inline void DilatedConv(const float* input_data, const Dims<4>& input_dims,
                        const float* filter_data, const Dims<4>& filter_dims,
                        const float* bias_data, const Dims<4>& bias_dims,
                        int stride_width, int stride_height,
                        int dilation_width_factor, int dilation_height_factor,
                        int pad_width, int pad_height,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims, float* im2col_data,
                        const Dims<4>& im2col_dims) {
  gemmlowp::ScopedProfilingLabel label("DilatedConv");
  // This is a copy of the reference Conv implementation. We do not currently
  // have an optimized path for dilation.
  (void)im2col_data;  // only used in optimized code.
  (void)im2col_dims;  // only used in optimized code.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  if (bias_data) {
    TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
  }
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_data[Offset(input_dims, in_channel,
                                                        in_x, in_y, batch)];
                  float filter_value =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          }
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int dilation_width_factor,
                 int dilation_height_factor, int pad_width, int pad_height,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  if ((dilation_width_factor != 1) || (dilation_height_factor != 1)) {
    return DilatedConv(input_data, input_dims, filter_data, filter_dims,
                       bias_data, bias_dims, stride_width, stride_height,
                       dilation_width_factor, dilation_height_factor, pad_width,
                       pad_height, output_activation_min, output_activation_max,
                       output_data, output_dims, im2col_data, im2col_dims);
  }

  (void)im2col_data;
  (void)im2col_dims;
  gemmlowp::ScopedProfilingLabel label("Conv");

  const float* gemm_input_data = nullptr;
  const Dims<4>* gemm_input_dims = nullptr;
  const int filter_width = ArraySize(filter_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col(input_data, input_dims, stride_width, stride_height, pad_width,
           pad_height, filter_height, filter_width, 0, im2col_data,
           im2col_dims);
    gemm_input_data = im2col_data;
    gemm_input_dims = &im2col_dims;
  } else {
    // TODO(aselle): We need to make sure to not send im2col if it is not
    // needed.
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_dims = &input_dims;
  }

  const auto im2col_matrix_map =
      MapAsMatrixWithFirstDimAsRows(gemm_input_data, *gemm_input_dims);
  const auto filter_matrix_map =
      MapAsMatrixWithLastDimAsCols(filter_data, filter_dims);
  auto output_matrix_map =
      MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), im2col_matrix_map, &output_matrix_map);

  AddBiasAndEvalActivationFunction(bias_data, bias_dims, output_data,
                                   output_dims, output_activation_min,
                                   output_activation_max);
}

template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int dilation_width_factor,
          int dilation_height_factor, int pad_width, int pad_height,
          float* output_data, const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, dilation_width_factor,
       dilation_height_factor, pad_width, pad_height, output_activation_min,
       output_activation_max, output_data, output_dims, im2col_data,
       im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride_width,
          int stride_height, int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  Conv(input_data, input_dims, filter_data, filter_dims, bias_data, bias_dims,
       stride_width, stride_height, 1, 1, pad_width, pad_height,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const float* input_data, const Dims<4>& input_dims,
          const float* filter_data, const Dims<4>& filter_dims,
          const float* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, float* output_data,
          const Dims<4>& output_dims, float* im2col_data,
          const Dims<4>& im2col_dims) {
  Conv<Ac>(input_data, input_dims, filter_data, filter_dims, bias_data,
           bias_dims, stride, stride, 1, 1, pad_width, pad_height, output_data,
           output_dims, im2col_data, im2col_dims);
}

inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("Conv/8bit");

  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(filter_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));

  const uint8* gemm_input_data = nullptr;
  const Dims<4>* gemm_input_dims = nullptr;
  const int filter_width = ArraySize(filter_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    Im2col(input_data, input_dims, stride_width, stride_height, pad_width,
           pad_height, filter_height, filter_width, input_zero_point,
           im2col_data, im2col_dims);
    gemm_input_data = im2col_data;
    gemm_input_dims = &im2col_dims;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_dims = &input_dims;
  }

  const int gemm_input_rows = gemm_input_dims->sizes[0];
  const int gemm_input_cols = gemm_input_dims->sizes[1] *
                              gemm_input_dims->sizes[2] *
                              gemm_input_dims->sizes[3];
  const int filter_rows = filter_dims.sizes[3];
  const int filter_cols =
      filter_dims.sizes[0] * filter_dims.sizes[1] * filter_dims.sizes[2];
  const int output_rows = output_dims.sizes[0];
  const int output_cols =
      output_dims.sizes[1] * output_dims.sizes[2] * output_dims.sizes[3];
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, gemm_input_cols);
  TFLITE_DCHECK_EQ(filter_cols, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[0], output_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[1], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[2], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[3], 1);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, filter_rows, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      gemm_input_data, gemm_input_rows, gemm_input_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, output_cols);
  const auto& output_pipeline = GemmlowpOutputPipeline::Make(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix, filter_offset,
      input_offset, output_pipeline);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void Conv(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_offset, const uint8* filter_data,
                 const Dims<4>& filter_dims, int32 filter_offset,
                 const int32* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int32 output_offset, int32 output_multiplier,
                 int output_shift, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims, uint8* im2col_data,
                 const Dims<4>& im2col_dims,
                 gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride_width, stride_height,
       pad_width, pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemm_context);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Conv(const uint8* input_data, const Dims<4>& input_dims,
          int32 input_offset, const uint8* filter_data,
          const Dims<4>& filter_dims, int32 filter_offset,
          const int32* bias_data, const Dims<4>& bias_dims, int stride,
          int pad_width, int pad_height, int32 output_offset,
          int32 output_multiplier, int output_shift,
          int32 output_activation_min, int32 output_activation_max,
          uint8* output_data, const Dims<4>& output_dims, uint8* im2col_data,
          const Dims<4>& im2col_dims, gemmlowp::GemmContext* gemm_context) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  Conv(input_data, input_dims, input_offset, filter_data, filter_dims,
       filter_offset, bias_data, bias_dims, stride, stride, pad_width,
       pad_height, output_offset, output_multiplier, output_shift,
       output_activation_min, output_activation_max, output_data, output_dims,
       im2col_data, im2col_dims, gemm_context);
}

template <typename T>
inline void DepthToSpace(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("DepthToSpace");

  const int input_depth = ArraySize(input_dims, 0);
  const int input_width = ArraySize(input_dims, 1);
  const int input_height = ArraySize(input_dims, 2);

  const int output_depth = ArraySize(output_dims, 0);
  const int batch_size = ArraySize(output_dims, 3);

  // Number of continuous values that we can copy in one interation.
  const int stride = block_size * output_depth;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const T* input_ptr = input_data + Offset(input_dims, 0, 0, in_h, batch);
      for (int offset_h = 0; offset_h < block_size; ++offset_h) {
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

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void Im2col(const T* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int kheight, int kwidth,
            uint8 byte_zero, T* output_data, const Dims<4>& output_dims) {
  Im2col(input_data, input_dims, stride, stride, pad_width, pad_height, kheight,
         kwidth, byte_zero, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void ConvAsGemm(const float* input_data, const Dims<4>& input_dims,
                const float* filter_data, const Dims<4>& filter_dims,
                const float* bias_data, const Dims<4>& bias_dims,
                float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("ConvAsGemm");

  const auto input_matrix_map =
      MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  const auto filter_matrix_map =
      MapAsMatrixWithLastDimAsCols(filter_data, filter_dims);
  auto output_matrix_map =
      MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  Gemm(filter_matrix_map.transpose(), input_matrix_map, &output_matrix_map);

  AddBiasAndEvalActivationFunction<Ac>(bias_data, bias_dims, output_data,
                                       output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void ConvAsGemm(const uint8* input_data, const Dims<4>& input_dims,
                int32 input_offset, const uint8* filter_data,
                const Dims<4>& filter_dims, int32 filter_offset,
                const int32* bias_data, const Dims<4>& bias_dims,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims,
                gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label("ConvAsGemm/8bit");
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  const int input_rows = input_dims.sizes[0];
  const int input_cols =
      input_dims.sizes[1] * input_dims.sizes[2] * input_dims.sizes[3];
  const int filter_rows = filter_dims.sizes[3];
  const int filter_cols =
      filter_dims.sizes[0] * filter_dims.sizes[1] * filter_dims.sizes[2];
  const int output_rows = output_dims.sizes[0];
  const int output_cols =
      output_dims.sizes[1] * output_dims.sizes[2] * output_dims.sizes[3];
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, input_cols);
  TFLITE_DCHECK_EQ(filter_cols, input_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[0], output_rows);
  TFLITE_DCHECK_EQ(bias_dims.sizes[1], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[2], 1);
  TFLITE_DCHECK_EQ(bias_dims.sizes[3], 1);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor> filter_matrix(
      filter_data, output_rows, filter_cols, filter_cols);
  gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
      input_data, filter_cols, output_cols, filter_cols);
  gemmlowp::MatrixMap<uint8, gemmlowp::MapOrder::ColMajor> output_matrix(
      output_data, output_rows, output_cols, output_rows);
  const auto& output_pipeline = GemmlowpOutputPipeline::Make(
      bias_data, output_rows, output_offset, output_multiplier, output_shift,
      output_activation_min, output_activation_max);
  gemmlowp::GemmWithOutputPipeline<uint8, uint8,
                                   gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
      gemm_context, filter_matrix, input_matrix, &output_matrix, filter_offset,
      input_offset, output_pipeline);
}

template <typename T>
inline void SpaceToDepth(const T* input_data, const Dims<4>& input_dims,
                         int block_size, T* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("SpaceToDepth");

  const int output_depth = ArraySize(output_dims, 0);
  const int output_width = ArraySize(output_dims, 1);
  const int output_height = ArraySize(output_dims, 2);

  const int input_depth = ArraySize(input_dims, 0);
  const int batch_size = ArraySize(input_dims, 3);

  // Number of continuous values that we can copy in one interation.
  const int stride = block_size * input_depth;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      T* output_ptr = output_data + Offset(output_dims, 0, 0, out_h, batch);
      for (int offset_h = 0; offset_h < block_size; ++offset_h) {
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

template <FusedActivationFunctionType Ac>
void NonGlobalBatchNormalization(
    const float* input_data, const Dims<4>& input_dims, const float* mean_data,
    const Dims<4>& mean_dims, const float* multiplier_data,
    const Dims<4>& multiplier_dims, const float* offset_data,
    const Dims<4>& offset_dims, float* output_data,
    const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("NonGlobalBatchNormalization");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input_dims, 2, mean_dims, 2, multiplier_dims, 2,
                        offset_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input_dims, 1, mean_dims, 1, multiplier_dims, 1,
                        offset_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input_dims, 0, mean_dims, 0, multiplier_dims, 0,
                        offset_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] = ActivationFunction<Ac>(
              (input_data[Offset(input_dims, c, x, y, b)] -
               mean_data[Offset(mean_dims, c, x, y, 0)]) *
                  multiplier_data[Offset(multiplier_dims, c, x, y, 0)] +
              offset_data[Offset(offset_dims, c, x, y, 0)]);
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac>
void GlobalBatchNormalization(const float* input_data,
                              const Dims<4>& input_dims, const float* mean_data,
                              const Dims<4>& mean_dims,
                              const float* multiplier_data,
                              const Dims<4>& multiplier_dims,
                              const float* offset_data,
                              const Dims<4>& offset_dims, float* output_data,
                              const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("GlobalBatchNormalization");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input_dims, 0, mean_dims, 0, multiplier_dims, 0,
                        offset_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] = ActivationFunction<Ac>(
              (input_data[Offset(input_dims, c, x, y, b)] -
               mean_data[Offset(mean_dims, c, 0, 0, 0)]) *
                  multiplier_data[Offset(multiplier_dims, c, 0, 0, 0)] +
              offset_data[Offset(offset_dims, c, 0, 0, 0)]);
        }
      }
    }
  }
}

inline void Relu(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Relu (not fused)");

  const auto input = MapAsVector(input_data, input_dims);
  auto output = MapAsVector(output_data, output_dims);
  output = input.cwiseMax(0.0f);
}

inline void Relu1(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Relu1 (not fused)");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          float val = input_data[Offset(input_dims, c, x, y, b)];
          const float upper = 1;
          const float lower = -1;
          float clamped = val > upper ? upper : val < lower ? lower : val;
          output_data[Offset(output_dims, c, x, y, b)] = clamped;
        }
      }
    }
  }
}

inline void Relu6(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Relu6 (not fused)");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          float val = input_data[Offset(input_dims, c, x, y, b)];
          const float upper = 6;
          const float lower = 0;
          float clamped = val > upper ? upper : val < lower ? lower : val;
          output_data[Offset(output_dims, c, x, y, b)] = clamped;
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac>
void L2Normalization(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("L2Normalization");
  static_assert(Ac == FusedActivationFunctionType::kNone, "");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float squared_l2_norm = 0;
        for (int c = 0; c < depth; ++c) {
          float val = input_data[Offset(input_dims, c, x, y, b)];
          squared_l2_norm += val * val;
        }
        float inverse_l2_norm = 1.0f / std::sqrt(squared_l2_norm);
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              input_data[Offset(input_dims, c, x, y, b)] * inverse_l2_norm;
        }
      }
    }
  }
}

inline void GetInvSqrtQuantizedMultiplier(int32 input, int32* output_inv_sqrt,
                                          int* output_shift) {
  *output_shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++*output_shift;
  }
  TFLITE_DCHECK_GT(input, 0);
  const unsigned max_left_shift_bits = __builtin_clz(input) - 1;
  const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
  const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
  *output_shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  TFLITE_DCHECK_GE(input, (1 << 27));
  TFLITE_DCHECK_LT(input, (1 << 29));
  using gemmlowp::FixedPoint;
  using gemmlowp::Rescale;
  using gemmlowp::SaturatingRoundingMultiplyByPOT;
  // Using 3 integer bits gives us enough room for the internal arithmetic in
  // this Newton-Raphson iteration.
  using F3 = FixedPoint<int32, 3>;
  using F0 = FixedPoint<int32, 0>;
  const F3 fixedpoint_input = F3::FromRaw(input >> 1);
  const F3 fixedpoint_half_input =
      SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
  const F3 fixedpoint_half_three =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
  // Newton-Raphson iteration
  // Naive unoptimized starting guess: x = 1
  F3 x = F3::One();
  // Naive unoptimized number of iterations: 5
  for (int i = 0; i < 5; i++) {
    const F3 x3 = Rescale<3>(x * x * x);
    x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
  }
  const F0 fixedpoint_half_sqrt_2 =
      GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
  x = x * fixedpoint_half_sqrt_2;
  *output_inv_sqrt = x.raw();
  if (*output_shift < 0) {
    *output_inv_sqrt <<= -*output_shift;
    *output_shift = 0;
  }
}

inline void L2Normalization(const uint8* input_data, const Dims<4>& input_dims,
                            int32 input_zero_point, uint8* output_data,
                            const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("L2Normalization/8bit");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  TFLITE_DCHECK_EQ(batches, 1);
  TFLITE_DCHECK_EQ(height, 1);
  TFLITE_DCHECK_EQ(width, 1);
  int32 square_l2_norm = 0;
  for (int i = 0; i < depth; i++) {
    int32 diff = input_data[i] - input_zero_point;
    square_l2_norm += diff * diff;
  }
  int32 inv_l2norm_multiplier;
  int inv_l2norm_shift;
  GetInvSqrtQuantizedMultiplier(square_l2_norm, &inv_l2norm_multiplier,
                                &inv_l2norm_shift);

  for (int i = 0; i < depth; i++) {
    int32 diff = input_data[i] - input_zero_point;
    int32 rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOne(
        128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
    int32 unclamped_output_val = 128 + rescaled_diff;
    int32 output_val = std::min(255, std::max(0, unclamped_output_val));
    output_data[i] = static_cast<uint8>(output_val);
  }
}

inline void Add(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Add");
  /* const int batches = */ MatchingArraySize(input1_dims, 3, input2_dims, 3,
                                              output_dims, 3);
  /* const int height = */ MatchingArraySize(input1_dims, 2, input2_dims, 2,
                                             output_dims, 2);
  /* const int width = */ MatchingArraySize(input1_dims, 1, input2_dims, 1,
                                            output_dims, 1);
  /* const int depth = */ MatchingArraySize(input1_dims, 0, input2_dims, 0,
                                            output_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input1_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(input2_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));

  int i = 0;
  const int size = input1_dims.sizes[3] * input1_dims.strides[3];
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
    output_data[i] = ActivationFunctionWithMinMax(x, output_activation_min,
                                                  output_activation_max);
  }
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, int left_shift, const uint8* input1_data,
                           int32 input1_offset, int32 input1_multiplier,
                           int input1_shift, const uint8* input2_data,
                           int32 input2_offset, int32 input2_multiplier,
                           int input2_shift, int32 output_offset,
                           int32 output_multiplier, int output_shift,
                           int32 output_activation_min,
                           int32 output_activation_max, uint8* output_data) {
  int i = 0;
  TFLITE_DCHECK_GT(input1_offset, -256);
  TFLITE_DCHECK_GT(input2_offset, -256);
  TFLITE_DCHECK_LT(input1_offset, 256);
  TFLITE_DCHECK_LT(input2_offset, 256);
#ifdef USE_NEON
  const auto output_activation_min_vector = vdup_n_u8(output_activation_min);
  const auto output_activation_max_vector = vdup_n_u8(output_activation_max);
  for (; i <= size - 8; i += 8) {
    const auto input1_val_original = vld1_u8(input1_data + i);
    const auto input2_val_original = vld1_u8(input2_data + i);
    const auto input1_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const auto input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const auto input1_val =
        vaddq_s16(input1_val_s16, vdupq_n_s16(input1_offset));
    const auto input2_val =
        vaddq_s16(input2_val_s16, vdupq_n_s16(input2_offset));
    const auto input1_val_high = vget_high_s16(input1_val);
    const auto input1_val_low = vget_low_s16(input1_val);
    const auto input2_val_high = vget_high_s16(input2_val);
    const auto input2_val_low = vget_low_s16(input2_val);
    auto x11 = vmovl_s16(input1_val_low);
    auto x12 = vmovl_s16(input1_val_high);
    auto x21 = vmovl_s16(input2_val_low);
    auto x22 = vmovl_s16(input2_val_high);
    const auto left_shift_dup = vdupq_n_s32(left_shift);
    x11 = vshlq_s32(x11, left_shift_dup);
    x12 = vshlq_s32(x12, left_shift_dup);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x11 = vqrdmulhq_n_s32(x11, input1_multiplier);
    x12 = vqrdmulhq_n_s32(x12, input1_multiplier);
    x21 = vqrdmulhq_n_s32(x21, input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, input2_multiplier);
    const auto input1_shift_dup = vdupq_n_s32(-input1_shift);
    const auto input2_shift_dup = vdupq_n_s32(-input2_shift);
    x11 = vshlq_s32(x11, input1_shift_dup);
    x12 = vshlq_s32(x12, input1_shift_dup);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    auto s1 = vaddq_s32(x11, x21);
    auto s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    s1 = RoundingDivideByPOT(s1, output_shift);
    s2 = RoundingDivideByPOT(s2, output_shift);
    const auto s1_narrowed = vmovn_s32(s1);
    const auto s2_narrowed = vmovn_s32(s2);
    const auto s = vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed),
                             vdupq_n_s16(output_offset));
    const auto clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(s)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = input1_offset + input1_data[i];
    const int32 input2_val = input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << left_shift);
    const int32 shifted_input2_val = input2_val * (1 << left_shift);
    const int32 scaled_input1_val = MultiplyByQuantizedMultiplierSmallerThanOne(
        shifted_input1_val, input1_multiplier, input1_shift);
    const int32 scaled_input2_val = MultiplyByQuantizedMultiplierSmallerThanOne(
        shifted_input2_val, input2_multiplier, input2_shift);
    const int32 raw_sum = scaled_input1_val + scaled_input2_val;
    const int32 raw_output = MultiplyByQuantizedMultiplierSmallerThanOne(
                                 raw_sum, output_multiplier, output_shift) +
                             output_offset;
    const int32 clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Add(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Add(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

template <FusedActivationFunctionType Ac>
inline void Add(int left_shift, const uint8* input1_data,
                const Dims<4>& input1_dims, int32 input1_offset,
                int32 input1_multiplier, int input1_shift,
                const uint8* input2_data, const Dims<4>& input2_dims,
                int32 input2_offset, int32 input2_multiplier, int input2_shift,
                int32 output_offset, int32 output_multiplier, int output_shift,
                int32 output_activation_min, int32 output_activation_max,
                uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  gemmlowp::ScopedProfilingLabel label("Add/8bit");
  const int flat_size = MatchingFlatSize(input1_dims, input2_dims, output_dims);
  TFLITE_DCHECK(IsPackedWithoutStrides(input1_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(input2_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));

  TFLITE_DCHECK_GT(input1_offset, -256);
  TFLITE_DCHECK_GT(input2_offset, -256);
  TFLITE_DCHECK_LT(input1_offset, 256);
  TFLITE_DCHECK_LT(input2_offset, 256);
  AddElementwise(flat_size, left_shift, input1_data, input1_offset,
                 input1_multiplier, input1_shift, input2_data, input2_offset,
                 input2_multiplier, input2_shift, output_offset,
                 output_multiplier, output_shift, output_activation_min,
                 output_activation_max, output_data);
}

template <FusedActivationFunctionType Ac>
inline void Add(const int16* input1_data, const Dims<4>& input1_dims,
                int input1_shift, const int16* input2_data,
                const Dims<4>& input2_dims, int input2_shift,
                int16 output_activation_min, int16 output_activation_max,
                int16* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Add/Int16");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, -32768);
    TFLITE_DCHECK_EQ(output_activation_max, 32767);
  }

  const int flat_size = RequiredBufferSizeForDims(output_dims);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input1_dims), flat_size);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input2_dims), flat_size);

  TFLITE_DCHECK(input1_shift == 0 || input2_shift == 0);
  TFLITE_DCHECK_GE(input1_shift, 0);
  TFLITE_DCHECK_GE(input2_shift, 0);
  const int16* not_shift_input = input1_shift == 0 ? input1_data : input2_data;
  const int16* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_shift = input1_shift == 0 ? input2_shift : input1_shift;

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
    F0 scaled_input =
        F0::FromRaw(gemmlowp::RoundingDivideByPOT(shift_input[i], input_shift));
    F0 result = gemmlowp::SaturatingAdd(scaled_input, input_ready_scaled);
    const int16 raw_output = result.raw();
    const int16 clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = clamped_output;
  }
}

template <FusedActivationFunctionType Ac>
void Add(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Add/int32");
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto input2_map = MapAsVector(input2_data, input2_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  if (AreSameDims(input1_dims, input2_dims)) {
    output_map.array() = input1_map.array() + input2_map.array();
  } else if (RequiredBufferSizeForDims(input2_dims) == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() + scalar;
  } else if (RequiredBufferSizeForDims(input1_dims) == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar + input2_map.array();
  } else {
    // Should not come here.
    TFLITE_DCHECK(false);
  }
}

// TODO(jiawen): We can implement BroadcastAdd on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastAdd is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAdd");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] +
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastAdd(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAddGeneric/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 shifted_input1_val = input1_val * (1 << left_shift);
          const int32 shifted_input2_val = input2_val * (1 << left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  shifted_input1_val, input1_multiplier, input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  shifted_input2_val, input2_multiplier, input2_shift);
          const int32 raw_sum = scaled_input1_val + scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  raw_sum, output_multiplier, output_shift) +
              output_offset;
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, raw_output));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAddFivefold/8bit");

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  uint8* output_data_ptr = output_data;
  const uint8* input1_data_ptr = input1_data;
  const uint8* input2_data_reset = input2_data;
  for (int i4 = 0; i4 < y4; ++i4) {
    const uint8* input2_data_ptr;
    for (int i3 = 0; i3 < y3; ++i3) {
      input2_data_ptr = input2_data_reset;
      for (int i2 = 0; i2 < y2; ++i2) {
        for (int i1 = 0; i1 < y1; ++i1) {
          AddElementwise(
              y0, left_shift, input1_data_ptr, input1_offset, input1_multiplier,
              input1_shift, input2_data_ptr, input2_offset, input2_multiplier,
              input2_shift, output_offset, output_multiplier, output_shift,
              output_activation_min, output_activation_max, output_data_ptr);
          input2_data_ptr += y0;
          output_data_ptr += y0;
        }
        input1_data_ptr += y0;
      }
    }
    input2_data_reset = input2_data_ptr;
  }
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAdd(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  BroadcastAdd(left_shift, input1_data, input1_dims, input1_offset,
               input1_multiplier, input1_shift, input2_data, input2_dims,
               input2_offset, input2_multiplier, input2_shift, output_offset,
               output_multiplier, output_shift, output_activation_min,
               output_activation_max, output_data, output_dims);
}

template <FusedActivationFunctionType Ac>
inline void BroadcastAddFivefold(
    int y0, int y1, int y2, int y3, int y4, int left_shift,
    const uint8* input1_data, const Dims<4>& input1_dims, int32 input1_offset,
    int32 input1_multiplier, int input1_shift, const uint8* input2_data,
    const Dims<4>& input2_dims, int32 input2_offset, int32 input2_multiplier,
    int input2_shift, int32 output_offset, int32 output_multiplier,
    int output_shift, int32 output_activation_min, int32 output_activation_max,
    uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  BroadcastAddFivefold(y0, y1, y2, y3, y4, left_shift, input1_data, input1_dims,
                       input1_offset, input1_multiplier, input1_shift,
                       input2_data, input2_dims, input2_offset,
                       input2_multiplier, input2_shift, output_offset,
                       output_multiplier, output_shift, output_activation_min,
                       output_activation_max, output_data, output_dims);
}

inline void Mul(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mul");
  /* const int batches = */ MatchingArraySize(input1_dims, 3, input2_dims, 3,
                                              output_dims, 3);
  /* const int height = */ MatchingArraySize(input1_dims, 2, input2_dims, 2,
                                             output_dims, 2);
  /* const int width = */ MatchingArraySize(input1_dims, 1, input2_dims, 1,
                                            output_dims, 1);
  /* const int depth = */ MatchingArraySize(input1_dims, 0, input2_dims, 0,
                                            output_dims, 0);
  TFLITE_DCHECK(IsPackedWithoutStrides(input1_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(input2_dims));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));

  int i = 0;
  const int size = input1_dims.sizes[3] * input1_dims.strides[3];
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

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void Mul(const float* input1_data, const Dims<4>& input1_dims,
         const float* input2_data, const Dims<4>& input2_dims,
         float* output_data, const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  Mul(input1_data, input1_dims, input2_data, input2_dims, output_activation_min,
      output_activation_max, output_data, output_dims);
}

template <FusedActivationFunctionType Ac>
void Mul(const int32* input1_data, const Dims<4>& input1_dims,
         const int32* input2_data, const Dims<4>& input2_dims,
         int32* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mul/int32");
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);

  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto input2_map = MapAsVector(input2_data, input2_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  if (AreSameDims(input1_dims, input2_dims)) {
    output_map.array() = input1_map.array() * input2_map.array();
  } else if (RequiredBufferSizeForDims(input2_dims) == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() * scalar;
  } else if (RequiredBufferSizeForDims(input1_dims) == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar * input2_map.array();
  } else {
    // Should not come here.
    TFLITE_DCHECK(false);
  }
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int16* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.

  const int flat_size = RequiredBufferSizeForDims(output_dims);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input1_dims), flat_size);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input2_dims), flat_size);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    output_data[i] = unclamped_result.raw();
  }
}

inline void Mul(const int16* input1_data, const Dims<4>& input1_dims,
                const int16* input2_data, const Dims<4>& input2_dims,
                int32 output_offset, int32 output_activation_min,
                int32 output_activation_max, uint8* output_data,
                const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16Uint8");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size = RequiredBufferSizeForDims(output_dims);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input1_dims), flat_size);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input2_dims), flat_size);

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

// TODO(jiawen): We can implement BroadcastMul on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastMul is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastMul");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] *
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac, typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T* output_data, const Dims<4>& output_dims) {
  T output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  BroadcastMul(input1_data, input1_dims, input2_data, input2_dims,
               output_activation_min, output_activation_max, output_data,
               output_dims);
}

inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastMul/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 unclamped_result =
              output_offset +
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  input1_val * input2_val, output_multiplier, output_shift);
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, unclamped_result));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
inline void BroadcastMul(const uint8* input1_data, const Dims<4>& input1_dims,
                         int32 input1_offset, const uint8* input2_data,
                         const Dims<4>& input2_dims, int32 input2_offset,
                         int32 output_offset, int32 output_multiplier,
                         int output_shift, int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  BroadcastMul(input1_data, input1_dims, input1_offset, input2_data,
               input2_dims, input2_offset, output_offset, output_multiplier,
               output_shift, output_activation_min, output_activation_max,
               output_data, output_dims);
}

// TODO(aselle): This is not actually optimized yet.
inline void Div(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  const int batches =
      MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[Offset(input1_dims, c, x, y, b)] /
                      input2_data[Offset(input2_dims, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastDiv is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T>
void BroadcastDiv(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastDiv");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] /
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// TODO(aselle): This is not actually optimized yet.
inline void Sub(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float output_activation_min, float output_activation_max,
                float* output_data, const Dims<4>& output_dims) {
  const int batches =
      MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[Offset(input1_dims, c, x, y, b)] -
                      input2_data[Offset(input2_dims, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// TODO(jiawen): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastSub is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T>
void BroadcastSub(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
                  T output_activation_min, T output_activation_max,
                  T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastSub");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, c, x, y, b)] -
                      input2_data[SubscriptToIndex(desc2, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

inline void BroadcastSub(int left_shift, const uint8* input1_data,
                         const Dims<4>& input1_dims, int32 input1_offset,
                         int32 input1_multiplier, int input1_shift,
                         const uint8* input2_data, const Dims<4>& input2_dims,
                         int32 input2_offset, int32 input2_multiplier,
                         int input2_shift, int32 output_offset,
                         int32 output_multiplier, int output_shift,
                         int32 output_activation_min,
                         int32 output_activation_max, uint8* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BroadcastSub/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          const int32 input1_val =
              input1_offset + input1_data[SubscriptToIndex(desc1, c, x, y, b)];
          const int32 input2_val =
              input2_offset + input2_data[SubscriptToIndex(desc2, c, x, y, b)];
          const int32 shifted_input1_val = input1_val * (1 << left_shift);
          const int32 shifted_input2_val = input2_val * (1 << left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  shifted_input1_val, input1_multiplier, input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  shifted_input2_val, input2_multiplier, input2_shift);
          const int32 raw_sub = scaled_input1_val - scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOne(
                  raw_sub, output_multiplier, output_shift) +
              output_offset;
          const int32 clamped_output =
              std::min(output_activation_max,
                       std::max(output_activation_min, raw_output));
          output_data[Offset(output_dims, c, x, y, b)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

template <FusedActivationFunctionType Ac, typename Scalar>
void Concatenation(int concat_dim, const Scalar* const* input_data,
                   const Dims<4>* const* input_dims, int inputs_count,
                   Scalar* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Concatenation");
  int concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    for (int j = 0; j < 4; j++) {
      if (j != concat_dim) {
        MatchingArraySize(*input_dims[i], j, output_dims, j);
      }
    }
    concat_size += ArraySize(*input_dims[i], concat_dim);
  }
  TFLITE_DCHECK_EQ(concat_size, ArraySize(output_dims, concat_dim));
  TFLITE_DCHECK(IsPackedWithoutStrides(output_dims));
  // for now we dont have a model with a Concatenation
  // with fused activation function.
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);
  int outer_size = 1;
  for (int i = concat_dim + 1; i < 4; i++) {
    outer_size *= output_dims.sizes[i];
  }
  Scalar* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size =
          input_dims[i]->sizes[concat_dim] * input_dims[i]->strides[concat_dim];
      memcpy(output_ptr, input_data[i] + k * copy_size,
             copy_size * sizeof(Scalar));
      output_ptr += copy_size;
    }
  }
}

// TODO(prabhumk): This is the same as the reference implementation.
// TODO(prabhumk): The quantized implementation of concatentation isn't fully
// quantized as it takes scale as a floating point value. This should be fixed
// when optimizng this routine further.
inline void Concatenation(int concat_dim, const uint8* const* input_data,
                          const Dims<4>* const* input_dims,
                          const int32* input_zeropoint,
                          const float* input_scale, int inputs_count,
                          uint8* output_data, const Dims<4>& output_dims,
                          const int32 output_zeropoint,
                          const float output_scale) {
  // The arguments input_zeropoint and input_scale are expected to be an array
  // that have the quantization parameters for all the inputs to the concat
  // operator.
  gemmlowp::ScopedProfilingLabel label("Concatenation");
  TFLITE_DCHECK_GT(inputs_count, 1);
  int concat_size = 0;
  for (int i = 0; i < inputs_count; i++) {
    for (int j = 0; j < 4; j++) {
      if (j != concat_dim) {
        MatchingArraySize(*input_dims[i], j, output_dims, j);
      }
    }
    concat_size += ArraySize(*input_dims[i], concat_dim);
  }
  TFLITE_DCHECK_EQ(concat_size, ArraySize(output_dims, concat_dim));
  int outer_size = 1;
  for (int i = concat_dim + 1; i < 4; i++) {
    outer_size *= output_dims.sizes[i];
  }
  const float inverse_output_scale = 1.f / output_scale;
  uint8* output_ptr = output_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      const int copy_size =
          input_dims[i]->sizes[concat_dim] * input_dims[i]->strides[concat_dim];
      const uint8* input_ptr = input_data[i] + k * copy_size;
      if (input_zeropoint[i] == output_zeropoint &&
          input_scale[i] == output_scale) {
        memcpy(output_ptr, input_ptr, copy_size);
      } else {
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
        for (int j = 0; j < copy_size; ++j) {
          const int32_t value =
              static_cast<int32_t>(round(input_ptr[j] * scale + bias)) +
              output_zeropoint;
          output_ptr[j] =
              static_cast<uint8_t>(std::max(std::min(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

template <FusedActivationFunctionType Ac, typename Scalar>
void DepthConcatenation(const Scalar* const* input_data,
                        const Dims<4>* const* input_dims, int inputs_count,
                        Scalar* output_data, const Dims<4>& output_dims) {
  Concatenation<Ac, Scalar>(0, input_data, input_dims, inputs_count,
                            output_data, output_dims);
}

inline void LstmCell(const float* input_data, const Dims<4>& input_dims,
                     const float* prev_activ_data,
                     const Dims<4>& prev_activ_dims, const float* weights_data,
                     const Dims<4>& weights_dims, const float* bias_data,
                     const Dims<4>& bias_dims, const float* prev_state_data,
                     const Dims<4>& prev_state_dims, float* output_state_data,
                     const Dims<4>& output_state_dims, float* output_activ_data,
                     const Dims<4>& output_activ_dims, float* concat_temp_data,
                     const Dims<4>& concat_temp_dims, float* activ_temp_data,
                     const Dims<4>& activ_temp_dims) {
  gemmlowp::ScopedProfilingLabel label("LstmCell");
  MatchingArraySize(  // batches
      input_dims, 3, prev_activ_dims, 3, prev_state_dims, 3, output_state_dims,
      3, output_activ_dims, 3);
  MatchingArraySize(  // height
      input_dims, 2, prev_activ_dims, 2, prev_state_dims, 2, output_state_dims,
      2, output_activ_dims, 2);
  MatchingArraySize(  // width
      input_dims, 1, prev_activ_dims, 1, prev_state_dims, 1, output_state_dims,
      1, output_activ_dims, 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 2), 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 3), 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int prev_activ_depth = ArraySize(prev_activ_dims, 0);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 0), total_input_depth);
  TFLITE_CHECK_EQ(MatchingArraySize(bias_dims, 1, bias_dims, 2, bias_dims, 3),
                  1);
  const int intern_activ_depth =
      MatchingArraySize(weights_dims, 1, bias_dims, 0);
  TFLITE_CHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingArraySize(prev_state_dims, 0, prev_activ_dims, 0,
                        output_state_dims, 0, output_activ_dims, 0);
  TFLITE_CHECK_EQ(output_depth, intern_activ_depth / 4);

  // Concatenate prev_activ and input data together
  std::vector<float const*> concat_input_arrays_data;
  std::vector<Dims<4> const*> concat_input_arrays_dims;
  concat_input_arrays_data.push_back(input_data);
  concat_input_arrays_data.push_back(prev_activ_data);
  concat_input_arrays_dims.push_back(&input_dims);
  concat_input_arrays_dims.push_back(&prev_activ_dims);
  Concatenation<FusedActivationFunctionType::kNone, float>(
      0, &(concat_input_arrays_data[0]), &(concat_input_arrays_dims[0]),
      concat_input_arrays_data.size(), concat_temp_data, concat_temp_dims);

  // Fully connected
  FullyConnected<FusedActivationFunctionType::kNone>(
      concat_temp_data, concat_temp_dims, weights_data, weights_dims, bias_data,
      bias_dims, activ_temp_data, activ_temp_dims);

  // Map raw arrays to Eigen arrays so we can use Eigen's optimized array
  // operations.
  ArrayMap<float> activ_temp_map =
      MapAsArrayWithFirstDimAsRows(activ_temp_data, activ_temp_dims);
  auto input_gate_sm = activ_temp_map.block(0 * output_depth, 0, output_depth,
                                            activ_temp_map.cols());
  auto new_input_sm = activ_temp_map.block(1 * output_depth, 0, output_depth,
                                           activ_temp_map.cols());
  auto forget_gate_sm = activ_temp_map.block(2 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  auto output_gate_sm = activ_temp_map.block(3 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  ArrayMap<const float> prev_state_map =
      MapAsArrayWithFirstDimAsRows(prev_state_data, prev_state_dims);
  ArrayMap<float> output_state_map =
      MapAsArrayWithFirstDimAsRows(output_state_data, output_state_dims);
  ArrayMap<float> output_activ_map =
      MapAsArrayWithFirstDimAsRows(output_activ_data, output_activ_dims);

  // Combined memory state and final output calculation
  gemmlowp::ScopedProfilingLabel label2("MemoryStateAndFinalOutput");
  output_state_map =
      input_gate_sm.unaryExpr(Eigen::internal::scalar_sigmoid_op<float>()) *
          new_input_sm.tanh() +
      forget_gate_sm.unaryExpr(Eigen::internal::scalar_sigmoid_op<float>()) *
          prev_state_map;
  output_activ_map =
      output_gate_sm.unaryExpr(Eigen::internal::scalar_sigmoid_op<float>()) *
      output_state_map.tanh();
}

// Quantized LSTM cell. Currently just a copy of the reference impl in
// reference_ops.h. See the big function comment there, not replicating it
// here.
template <int StateIntegerBits>
void LstmCell(const uint8* input_data_uint8, const Dims<4>& input_dims,
              const uint8* prev_activ_data_uint8,
              const Dims<4>& prev_activ_dims, const uint8* weights_data_uint8,
              const Dims<4>& weights_dims, const int32* bias_data_int32,
              const Dims<4>& bias_dims, const int16* prev_state_data_int16,
              const Dims<4>& prev_state_dims, int16* output_state_data_int16,
              const Dims<4>& output_state_dims, uint8* output_activ_data_uint8,
              const Dims<4>& output_activ_dims, uint8* concat_temp_data_uint8,
              const Dims<4>& concat_temp_dims, int16* activ_temp_data_int16,
              const Dims<4>& activ_temp_dims, int32 weights_zero_point,
              int32 accum_multiplier, int accum_shift,
              gemmlowp::GemmContext* gemm_context) {
  gemmlowp::ScopedProfilingLabel label(
      "LstmCell/quantized (8bit external, 16bit internal)");
  // Gather dimensions information, and perform consistency checks.
  const int batches =
      MatchingArraySize(input_dims, 3, prev_activ_dims, 3, prev_state_dims, 3,
                        output_state_dims, 3, output_activ_dims, 3);
  const int height =
      MatchingArraySize(input_dims, 2, prev_activ_dims, 2, prev_state_dims, 2,
                        output_state_dims, 2, output_activ_dims, 2);
  const int width =
      MatchingArraySize(input_dims, 1, prev_activ_dims, 1, prev_state_dims, 1,
                        output_state_dims, 1, output_activ_dims, 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 2), 1);
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 3), 1);
  const int input_depth = ArraySize(input_dims, 0);
  const int prev_activ_depth = ArraySize(prev_activ_dims, 0);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_CHECK_EQ(ArraySize(weights_dims, 0), total_input_depth);
  TFLITE_CHECK_EQ(MatchingArraySize(bias_dims, 1, bias_dims, 2, bias_dims, 3),
                  1);
  const int intern_activ_depth =
      MatchingArraySize(weights_dims, 1, bias_dims, 0);
  TFLITE_CHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingArraySize(prev_state_dims, 0, prev_activ_dims, 0,
                        output_state_dims, 0, output_activ_dims, 0);
  TFLITE_CHECK_EQ(output_depth, intern_activ_depth / 4);
  const int fc_batches = ArraySize(activ_temp_dims, 1) *
                         ArraySize(activ_temp_dims, 2) *
                         ArraySize(activ_temp_dims, 3);
  const int fc_output_depth =
      MatchingArraySize(weights_dims, 1, activ_temp_dims, 0);
  const int fc_accum_depth = ArraySize(weights_dims, 0);
  TFLITE_CHECK_EQ(fc_output_depth, 4 * output_depth);

  // Depth-concatenate prev_activ and input data together.
  uint8 const* concat_input_arrays_data[2] = {input_data_uint8,
                                              prev_activ_data_uint8};
  Dims<4> const* concat_input_arrays_dims[2] = {&input_dims, &prev_activ_dims};
  Concatenation<FusedActivationFunctionType::kNone, uint8>(
      0, concat_input_arrays_data, concat_input_arrays_dims, 2,
      concat_temp_data_uint8, concat_temp_dims);

  // Implementation of the fully connected node inside the LSTM cell.
  // The operands are 8-bit integers, the accumulators are internally 32bit
  // integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
  bool gemm_already_performed = false;
#ifdef GEMMLOWP_NEON
  if (fc_batches == 1 && !(fc_output_depth % 4) && !(fc_accum_depth % 8)) {
    GEMVForLstmCell(concat_temp_data_uint8, concat_temp_dims,
                    weights_data_uint8, weights_dims, weights_zero_point,
                    bias_data_int32, bias_dims, accum_multiplier, accum_shift,
                    activ_temp_data_int16, activ_temp_dims);
    gemm_already_performed = true;
  }
#endif
  if (!gemm_already_performed) {
    gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::RowMajor>
        weights_matrix(weights_data_uint8, fc_output_depth, fc_accum_depth);
    gemmlowp::MatrixMap<const uint8, gemmlowp::MapOrder::ColMajor> input_matrix(
        concat_temp_data_uint8, fc_accum_depth, fc_batches);
    gemmlowp::MatrixMap<int16, gemmlowp::MapOrder::ColMajor> output_matrix(
        activ_temp_data_int16, fc_output_depth, fc_batches);
    typedef gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>
        ColVectorMap;
    ColVectorMap bias_vector(bias_data_int32, fc_output_depth);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent scale_stage;
    scale_stage.result_offset_after_shift = 0;
    scale_stage.result_fixedpoint_multiplier = accum_multiplier;
    scale_stage.result_exponent = accum_shift;
    gemmlowp::OutputStageSaturatingCastToInt16 saturating_cast_int16_stage;
    auto output_pipeline = std::make_tuple(bias_addition_stage, scale_stage,
                                           saturating_cast_int16_stage);
    gemmlowp::GemmWithOutputPipeline<
        uint8, int16, gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
        gemm_context, weights_matrix, input_matrix, &output_matrix,
        -weights_zero_point, -128, output_pipeline);
  }

  // Rest of the LSTM cell: tanh and logistic math functions, and some adds
  // and muls, all done in 16-bit fixed-point.
  const int outer_size = batches * width * height;
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

template <FusedActivationFunctionType Ac, typename Scalar>
void TensorFlowSplit(const Scalar* input_data, const Dims<4>& input_dims,
                     int outputs_count, Scalar* const* output_data,
                     const Dims<4>* const* output_dims) {
  gemmlowp::ScopedProfilingLabel label("TensorFlowSplit");
  TFLITE_DCHECK_GE(outputs_count, 1);
  for (int i = 0; i < outputs_count; i++) {
    /* batches = */ MatchingArraySize(*output_dims[i], 3, input_dims, 3);
    /* height = */ MatchingArraySize(*output_dims[i], 2, input_dims, 2);
    /* width = */ MatchingArraySize(*output_dims[i], 1, input_dims, 1);
  }
  const int batches = MatchingArraySize(*output_dims[0], 3, input_dims, 3);
  const int height = MatchingArraySize(*output_dims[0], 2, input_dims, 2);
  const int width = MatchingArraySize(*output_dims[0], 1, input_dims, 1);
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  // for now we dont have a model with a TensorFlowSplit
  // with fused activation function.
  TFLITE_DCHECK(Ac == FusedActivationFunctionType::kNone);
  const int whb = width * height * batches;
  const Scalar* input_ptr = input_data;
  for (int k = 0; k < whb; k++) {
    for (int i = 0; i < outputs_count; ++i) {
      memcpy(output_data[i] + k * output_dims[i]->sizes[0], input_ptr,
             output_dims[i]->sizes[0] * sizeof(Scalar));
      input_ptr += output_dims[i]->sizes[0];
    }
  }
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline void AveragePool(const float* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int kwidth, int kheight,
                        float output_activation_min,
                        float output_activation_max, float* output_data,
                        const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("AveragePool");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  // TODO(benoitjacob) make this a proper reference impl without Eigen!
  const auto in_mat = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto out_mat = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);
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
        int hpad = h + pad_height;
        int wpad = w + pad_width;
        int h_start =
            (hpad < kheight) ? 0 : (hpad - kheight) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < kwidth) ? 0 : (wpad - kwidth) / stride_width + 1;
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

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  output_data[Offset(output_dims, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int kwidth, int kheight, float* output_data,
                 const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);

  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, kwidth, kheight, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const float* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, float* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_data, output_dims);
}

inline void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                        int stride_width, int stride_height, int pad_width,
                        int pad_height, int filter_width, int filter_height,
                        int32 output_activation_min,
                        int32 output_activation_max, uint8* output_data,
                        const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("AveragePool/8bit");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end =
            std::min(filter_width, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end =
            std::min(filter_height, input_height - in_y_origin);
        const int filter_count =
            (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
        // 1280 required by Inception v3
        static constexpr int kAccBufferMaxSize = 2048;
        TFLITE_DCHECK_LE(depth, kAccBufferMaxSize);
        uint16 acc[kAccBufferMaxSize];
        memset(acc, 0, depth * sizeof(acc[0]));
        const uint8* input_ptr =
            input_data + input_dims.strides[1] * in_x_origin +
            input_dims.strides[2] * in_y_origin + input_dims.strides[3] * batch;
        for (int fy = filter_y_start; fy < filter_y_end; fy++) {
          const uint8* input_row_ptr = input_ptr + fy * input_dims.strides[2] +
                                       filter_x_start * input_dims.strides[1];
          for (int fx = filter_x_start; fx < filter_x_end; fx++) {
            int channel = 0;
#ifdef USE_NEON
            for (; channel <= depth - 16; channel += 16) {
              uint16x8_t acc_reg[2];
              for (int i = 0; i < 2; i++) {
                acc_reg[i] = vld1q_u16(acc + channel + 8 * i);
              }
              uint8x16_t input_reg = vld1q_u8(input_row_ptr);
              input_row_ptr += 16;
              acc_reg[0] = vaddw_u8(acc_reg[0], vget_low_u8(input_reg));
              acc_reg[1] = vaddw_u8(acc_reg[1], vget_high_u8(input_reg));
              for (int i = 0; i < 2; i++) {
                vst1q_u16(acc + channel + 8 * i, acc_reg[i]);
              }
            }
            for (; channel <= depth - 8; channel += 8) {
              uint16x8_t acc_reg = vld1q_u16(acc + channel);
              uint8x8_t input_reg = vld1_u8(input_row_ptr);
              input_row_ptr += 8;
              acc_reg = vaddw_u8(acc_reg, input_reg);
              vst1q_u16(acc + channel, acc_reg);
            }
#endif
            for (; channel < depth; ++channel) {
              acc[channel] += *input_row_ptr++;
            }
          }
        }
        uint8* output_ptr =
            output_data + Offset(output_dims, 0, out_x, out_y, batch);
        int channel = 0;
#ifdef USE_NEON
#define AVGPOOL_DIVIDING_BY(FILTER_COUNT)                              \
  if (filter_count == FILTER_COUNT) {                                  \
    for (; channel <= depth - 8; channel += 8) {                       \
      uint16 buf[8];                                                   \
      for (int i = 0; i < 8; i++) {                                    \
        buf[i] = (acc[channel + i] + FILTER_COUNT / 2) / FILTER_COUNT; \
      }                                                                \
      uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));                     \
      buf8 = vmin_u8(buf8, vdup_n_u8(output_activation_max));          \
      buf8 = vmax_u8(buf8, vdup_n_u8(output_activation_min));          \
      vst1_u8(output_ptr + channel, buf8);                             \
    }                                                                  \
  }
        AVGPOOL_DIVIDING_BY(9)
        AVGPOOL_DIVIDING_BY(15)
#undef AVGPOOL_DIVIDING_BY
        for (; channel <= depth - 8; channel += 8) {
          uint16 buf[8];
          for (int i = 0; i < 8; i++) {
            buf[i] = (acc[channel + i] + filter_count / 2) / filter_count;
          }
          uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));
          buf8 = vmin_u8(buf8, vdup_n_u8(output_activation_max));
          buf8 = vmax_u8(buf8, vdup_n_u8(output_activation_min));
          vst1_u8(output_ptr + channel, buf8);
        }
#endif
        for (; channel < depth; ++channel) {
          uint16 a = (acc[channel] + filter_count / 2) / filter_count;
          a = std::max<uint16>(a, output_activation_min);
          a = std::min<uint16>(a, output_activation_max);
          output_ptr[channel] = static_cast<uint8>(a);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, int filter_width, int filter_height,
                 int32 output_activation_min, int32 output_activation_max,
                 uint8* output_data, const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  AveragePool(input_data, input_dims, stride_width, stride_height, pad_width,
              pad_height, filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void AveragePool(const uint8* input_data, const Dims<4>& input_dims, int stride,
                 int pad_width, int pad_height, int filter_width,
                 int filter_height, int32 output_activation_min,
                 int32 output_activation_max, uint8* output_data,
                 const Dims<4>& output_dims) {
  AveragePool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
                  filter_width, filter_height, output_activation_min,
                  output_activation_max, output_data, output_dims);
}

inline void MaxPool(const float* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int kwidth, int kheight,
                    float output_activation_min, float output_activation_max,
                    float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("MaxPool");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  const auto in_mat = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto out_mat = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);
  // Prefill the output to minimum representable float value
  out_mat.setConstant(std::numeric_limits<float>::lowest());
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + pad_height;
        int wpad = w + pad_width;
        int h_start =
            (hpad < kheight) ? 0 : (hpad - kheight) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < kwidth) ? 0 : (wpad - kwidth) / stride_width + 1;
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

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      for (int x = 0; x < output_width; ++x) {
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              ActivationFunctionWithMinMax(
                  output_data[Offset(output_dims, c, x, y, b)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int kwidth, int kheight, float* output_data,
             const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, kwidth, kheight, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const float* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             float* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_data, output_dims);
}

inline void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
                    int stride_width, int stride_height, int pad_width,
                    int pad_height, int filter_width, int filter_height,
                    int32 output_activation_min, int32 output_activation_max,
                    uint8* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("MaxPool/8bit");
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end =
            std::min(filter_width, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end =
            std::min(filter_height, input_height - in_y_origin);
        // 2048 required by Inception v3
        static constexpr int kAccBufferMaxSize = 2048;
        TFLITE_DCHECK_LE(depth, kAccBufferMaxSize);
        uint8 acc[kAccBufferMaxSize];
        memset(acc, 0, depth * sizeof(acc[0]));
        const uint8* input_ptr =
            input_data + input_dims.strides[1] * in_x_origin +
            input_dims.strides[2] * in_y_origin + input_dims.strides[3] * batch;
        for (int fy = filter_y_start; fy < filter_y_end; fy++) {
          const uint8* input_row_ptr = input_ptr + fy * input_dims.strides[2] +
                                       filter_x_start * input_dims.strides[1];
          for (int fx = filter_x_start; fx < filter_x_end; fx++) {
            int channel = 0;
#ifdef USE_NEON
            for (; channel <= depth - 16; channel += 16) {
              uint8x16_t acc_reg = vld1q_u8(acc + channel);
              uint8x16_t input_reg = vld1q_u8(input_row_ptr);
              input_row_ptr += 16;
              acc_reg = vmaxq_u8(acc_reg, input_reg);
              vst1q_u8(acc + channel, acc_reg);
            }

            for (; channel <= depth - 8; channel += 8) {
              uint8x8_t acc_reg = vld1_u8(acc + channel);
              uint8x8_t input_reg = vld1_u8(input_row_ptr);
              input_row_ptr += 8;
              acc_reg = vmax_u8(acc_reg, input_reg);
              vst1_u8(acc + channel, acc_reg);
            }
#endif
            for (; channel < depth; ++channel) {
              acc[channel] = std::max(acc[channel], *input_row_ptr++);
            }
          }
        }
        uint8* output_ptr =
            output_data + Offset(output_dims, 0, out_x, out_y, batch);
        int channel = 0;
#ifdef USE_NEON
        for (; channel <= depth - 16; channel += 16) {
          uint8x16_t a = vld1q_u8(acc + channel);
          a = vminq_u8(a, vdupq_n_u8(output_activation_max));
          a = vmaxq_u8(a, vdupq_n_u8(output_activation_min));
          vst1q_u8(output_ptr + channel, a);
        }
        for (; channel <= depth - 8; channel += 8) {
          uint8x8_t a = vld1_u8(acc + channel);
          a = vmin_u8(a, vdup_n_u8(output_activation_max));
          a = vmax_u8(a, vdup_n_u8(output_activation_min));
          vst1_u8(output_ptr + channel, a);
        }
#endif
        for (; channel < depth; ++channel) {
          uint8 a = acc[channel];
          a = std::max<uint8>(a, output_activation_min);
          a = std::min<uint8>(a, output_activation_max);
          output_ptr[channel] = static_cast<uint8>(a);
        }
      }
    }
  }
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims,
             int stride_width, int stride_height, int pad_width, int pad_height,
             int filter_width, int filter_height, int32 output_activation_min,
             int32 output_activation_max, uint8* output_data,
             const Dims<4>& output_dims) {
  static_assert(Ac == FusedActivationFunctionType::kNone ||
                    Ac == FusedActivationFunctionType::kRelu ||
                    Ac == FusedActivationFunctionType::kRelu6 ||
                    Ac == FusedActivationFunctionType::kRelu1,
                "");
  if (Ac == FusedActivationFunctionType::kNone) {
    TFLITE_DCHECK_EQ(output_activation_min, 0);
    TFLITE_DCHECK_EQ(output_activation_max, 255);
  }
  MaxPool(input_data, input_dims, stride_width, stride_height, pad_width,
          pad_height, filter_width, filter_height, output_activation_min,
          output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void MaxPool(const uint8* input_data, const Dims<4>& input_dims, int stride,
             int pad_width, int pad_height, int filter_width, int filter_height,
             int32 output_activation_min, int32 output_activation_max,
             uint8* output_data, const Dims<4>& output_dims) {
  MaxPool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
              filter_width, filter_height, output_activation_min,
              output_activation_max, output_data, output_dims);
}

inline void L2Pool(const float* input_data, const Dims<4>& input_dims,
                   int stride_width, int stride_height, int pad_width,
                   int pad_height, int filter_width, int filter_height,
                   float output_activation_min, float output_activation_max,
                   float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("L2Pool");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  // Actually carry out L2 Pool. Code is written in forward mode: we go through
  // the input values once, and write to all the pooled regions that it maps to.
  const auto in_mat = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto out_mat = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);
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
        const int hpad = h + pad_height;
        const int wpad = w + pad_width;
        const int h_start = (hpad < filter_height)
                                ? 0
                                : (hpad - filter_height) / stride_height + 1;
        const int h_end = std::min(hpad / stride_height + 1, output_height);
        const int w_start = (wpad < filter_width)
                                ? 0
                                : (wpad - filter_width) / stride_width + 1;
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
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims,
            int stride_width, int stride_height, int pad_width, int pad_height,
            int filter_width, int filter_height, float* output_data,
            const Dims<4>& output_dims) {
  float output_activation_min, output_activation_max;
  GetActivationMinMax(Ac, &output_activation_min, &output_activation_max);
  L2Pool(input_data, input_dims, stride_width, stride_height, pad_width,
         pad_height, filter_width, filter_height, output_activation_min,
         output_activation_max, output_data, output_dims);
}

// legacy, for compatibility with old checked-in code
template <FusedActivationFunctionType Ac>
void L2Pool(const float* input_data, const Dims<4>& input_dims, int stride,
            int pad_width, int pad_height, int filter_width, int filter_height,
            float* output_data, const Dims<4>& output_dims) {
  L2Pool<Ac>(input_data, input_dims, stride, stride, pad_width, pad_height,
             filter_width, filter_height, output_data, output_dims);
}

inline void LocalResponseNormalization(const float* input_data,
                                       const Dims<4>& input_dims, int range,
                                       float bias, float alpha, float beta,
                                       float* output_data,
                                       const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("LocalResponseNormalization");
  /* const int batches = */ MatchingArraySize(input_dims, 3, output_dims, 3);
  /* const int height = */ MatchingArraySize(input_dims, 2, output_dims, 2);
  /* const int width = */ MatchingArraySize(input_dims, 1, output_dims, 1);
  /* const int depth = */ MatchingArraySize(input_dims, 0, output_dims, 0);

  const auto data_in = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto data_out = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);

  // Carry out local response normalization, vector by vector.
  // Since the data are stored column major, making row-wise operation
  // probably not memory efficient anyway, we do an explicit for loop over
  // the columns.
  const int double_range = range * 2;
  Eigen::VectorXf padded_square(data_in.rows() + double_range);
  padded_square.setZero();
  for (int r = 0; r < data_in.cols(); ++r) {
    // Do local response normalization for data_in(:, r)
    // first, compute the square and store them in buffer for repeated use
    padded_square.block(range, 0, data_in.rows(), 1) =
        data_in.col(r).cwiseProduct(data_in.col(r)) * alpha;
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
  if (beta == 1) {
    data_out.array() = data_in.array() * data_out.array().inverse();
  } else if (beta == 0.5) {
    data_out.array() = data_in.array() * data_out.array().sqrt().inverse();
  } else {
    data_out.array() = data_in.array() * data_out.array().pow(-beta);
  }
}

inline void Softmax(const float* input_data, const Dims<4>& input_dims,
                    float beta, float* output_data,
                    const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Softmax");
  /* const int batches = */ MatchingArraySize(input_dims, 3, output_dims, 3);
  /* const int height = */ MatchingArraySize(input_dims, 2, output_dims, 2);
  /* const int width = */ MatchingArraySize(input_dims, 1, output_dims, 1);
  /* const int depth = */ MatchingArraySize(input_dims, 0, output_dims, 0);

  const auto in_mat = MapAsMatrixWithFirstDimAsRows(input_data, input_dims);
  auto out_mat = MapAsMatrixWithFirstDimAsRows(output_data, output_dims);
  // Compute the exponential first, removing the max coefficient for numerical
  // stability.
  out_mat = (in_mat.rowwise() - in_mat.colwise().maxCoeff()).array() * beta;
  // We are separating out the exp function so that exp can be vectorized.
  out_mat = out_mat.array().exp();
  // Normalize to get the activations.
  Eigen::Array<float, 1, Eigen::Dynamic> scale =
      out_mat.array().colwise().sum().inverse();
  out_mat.array().rowwise() *= scale;
}

inline void Softmax(const uint8* input_data, const Dims<4>& input_dims,
                    int32 input_beta_multiplier, int32 input_beta_left_shift,
                    int diff_min, uint8* output_data,
                    const Dims<4>& output_dims) {
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  gemmlowp::ScopedProfilingLabel label("Softmax/8bit");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  const int outer_size = batches * height * width;

  for (int b = 0; b < outer_size; ++b) {
    const uint8* input_data_ptr = input_data + b * depth;
    uint8* output_data_ptr = output_data + b * depth;

    // Determine the largest entry in the current row
    uint8 max_in_row = 0;
    {
      int c = 0;
#ifdef USE_NEON
      uint8x16_t max16_0 = vdupq_n_u8(0);
      uint8x16_t max16_1 = vdupq_n_u8(0);
      for (; c <= depth - 32; c += 32) {
        max16_0 = vmaxq_u8(max16_0, vld1q_u8(input_data_ptr + c + 0));
        max16_1 = vmaxq_u8(max16_1, vld1q_u8(input_data_ptr + c + 16));
      }
      uint8x16_t max16 = vmaxq_u8(max16_0, max16_1);
      if (c <= depth - 16) {
        max16 = vmaxq_u8(max16, vld1q_u8(input_data_ptr + c));
        c += 16;
      }
      uint8x8_t max8 = vmax_u8(vget_low_u8(max16), vget_high_u8(max16));
      if (c <= depth - 8) {
        max8 = vmax_u8(max8, vld1_u8(input_data_ptr + c));
        c += 8;
      }
      uint8x8_t max4 = vmax_u8(max8, vext_u8(max8, max8, 4));
      uint8x8_t max2 = vmax_u8(max4, vext_u8(max4, max4, 2));
      uint8x8_t max1 = vpmax_u8(max2, max2);
      max_in_row = vget_lane_u8(max1, 0);
#endif
      for (; c < depth; ++c) {
        max_in_row = std::max(max_in_row, input_data_ptr[c]);
      }
    }

#ifdef USE_NEON
    using FixedPointAccumInt32x4 =
        gemmlowp::FixedPoint<int32x4_t, kAccumulationIntegerBits>;
    using FixedPointScaledDiffInt32x4 =
        gemmlowp::FixedPoint<int32x4_t, kScaledDiffIntegerBits>;
    using FixedPoint0Int32x4 = gemmlowp::FixedPoint<int32x4_t, 0>;
    FixedPoint0Int32x4 input_beta_multiplier_f0 =
        FixedPoint0Int32x4::FromScalarRaw(input_beta_multiplier);
    int16x8_t max_in_row_s16 = vdupq_n_s16(max_in_row);
#endif

    // Compute the sum of exponentials of the differences of entries in the
    // current row from the largest entry in the current row.
    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    {
      int c = 0;
#ifdef USE_NEON
      int32x4_t diff_min_s32 = vdupq_n_s32(diff_min);
      FixedPointAccumInt32x4 sum_of_exps_0 = FixedPointAccumInt32x4::Zero();
      FixedPointAccumInt32x4 sum_of_exps_1 = FixedPointAccumInt32x4::Zero();
      FixedPointAccumInt32x4 zeros = FixedPointAccumInt32x4::Zero();
      for (; c <= depth - 8; c += 8) {
        uint16x8_t input_u16 = vmovl_u8(vld1_u8(input_data_ptr + c));
        int16x8_t input_diff_s16 =
            vsubq_s16(vreinterpretq_s16_u16(input_u16), max_in_row_s16);
        int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
        int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
        int32x4_t mask_0 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_diff_s32_0, diff_min_s32);
        int32x4_t mask_1 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_diff_s32_1, diff_min_s32);
        FixedPointScaledDiffInt32x4 scaled_diff_0 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_0, input_beta_left_shift));
        FixedPointScaledDiffInt32x4 scaled_diff_1 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_1, input_beta_left_shift));
        FixedPointAccumInt32x4 exps_0 =
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(scaled_diff_0));
        FixedPointAccumInt32x4 exps_1 =
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(scaled_diff_1));
        FixedPointAccumInt32x4 masked_exps_0 =
            SelectUsingMask(mask_0, exps_0, zeros);
        FixedPointAccumInt32x4 masked_exps_1 =
            SelectUsingMask(mask_1, exps_1, zeros);
        sum_of_exps_0 = sum_of_exps_0 + masked_exps_0;
        sum_of_exps_1 = sum_of_exps_1 + masked_exps_1;
      }
      int32x4_t sum_of_exps_reduced_4 = (sum_of_exps_0 + sum_of_exps_1).raw();
      int32x2_t sum_of_exps_reduced_2 =
          vadd_s32(vget_low_s32(sum_of_exps_reduced_4),
                   vget_high_s32(sum_of_exps_reduced_4));
      int32x2_t sum_of_exps_reduced_1 =
          vpadd_s32(sum_of_exps_reduced_2, sum_of_exps_reduced_2);
      sum_of_exps =
          FixedPointAccum::FromRaw(vget_lane_s32(sum_of_exps_reduced_1, 0));
#endif
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);
          sum_of_exps =
              sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                exp_on_negative_values(scaled_diff_f8));
        }
      }
    }

    // Compute the fixed-point multiplier and shift that we need to apply to
    // perform a division by the above-computed sum-of-exponentials.
    int32 fixed_sum_of_exps = sum_of_exps.raw();
    int headroom_plus_one =
        __builtin_clz(static_cast<uint32>(fixed_sum_of_exps));
    // This is the number of bits to the left of the binary point above 1.0.
    // Consider fixed_sum_of_exps=1.25.  In that case shifted_scale=0.8 and
    // no later adjustment will be needed.
    int num_bits_over_unit = kAccumulationIntegerBits - headroom_plus_one;
    int32 shifted_sum_minus_one = static_cast<int32>(
        (static_cast<uint32>(fixed_sum_of_exps) << headroom_plus_one) -
        (static_cast<uint32>(1) << 31));
    FixedPoint0 shifted_scale = gemmlowp::one_over_one_plus_x_for_x_in_0_1(
        FixedPoint0::FromRaw(shifted_sum_minus_one));

    // Compute the quotients of exponentials of differences of entries in the
    // current row from the largest entry, over the previously-computed sum of
    // exponentials.
    {
      int c = 0;
#ifdef USE_NEON
      int16x8_t diff_min_s16 = vdupq_n_s16(diff_min);
      for (; c <= depth - 8; c += 8) {
        uint16x8_t input_u16 = vmovl_u8(vld1_u8(input_data_ptr + c));
        int16x8_t input_diff_s16 =
            vsubq_s16(vreinterpretq_s16_u16(input_u16), max_in_row_s16);
        int32x4_t input_diff_s32_0 = vmovl_s16(vget_low_s16(input_diff_s16));
        int32x4_t input_diff_s32_1 = vmovl_s16(vget_high_s16(input_diff_s16));
        uint8x8_t mask = vmovn_u16(vcgeq_s16(input_diff_s16, diff_min_s16));
        FixedPointScaledDiffInt32x4 scaled_diff_0 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_0, input_beta_left_shift));
        FixedPointScaledDiffInt32x4 scaled_diff_1 =
            input_beta_multiplier_f0 *
            FixedPointScaledDiffInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_diff_s32_1, input_beta_left_shift));
        FixedPoint0Int32x4 exp_0 = exp_on_negative_values(scaled_diff_0);
        FixedPoint0Int32x4 exp_1 = exp_on_negative_values(scaled_diff_1);
        int32x4_t output_s32_0 = gemmlowp::RoundingDivideByPOT(
            vqrdmulhq_n_s32(exp_0.raw(), shifted_scale.raw()),
            num_bits_over_unit + 31 - 8);
        int32x4_t output_s32_1 = gemmlowp::RoundingDivideByPOT(
            vqrdmulhq_n_s32(exp_1.raw(), shifted_scale.raw()),
            num_bits_over_unit + 31 - 8);
        int16x8_t output_s16 =
            vcombine_s16(vqmovn_s32(output_s32_0), vqmovn_s32(output_s32_1));
        uint8x8_t output_u8 = vqmovun_s16(output_s16);
        uint8x8_t masked_output = vbsl_u8(mask, output_u8, vdup_n_u8(0));
        vst1_u8(output_data_ptr + c, masked_output);
      }
#endif
      for (; c < depth; ++c) {
        int32 input_diff = static_cast<int32>(input_data_ptr[c]) - max_in_row;
        if (input_diff >= diff_min) {
          const int32 input_diff_rescaled =
              MultiplyByQuantizedMultiplierGreaterThanOne(
                  input_diff, input_beta_multiplier, input_beta_left_shift);
          const FixedPointScaledDiff scaled_diff_f8 =
              FixedPointScaledDiff::FromRaw(input_diff_rescaled);

          FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
          int32 unsat_output = gemmlowp::RoundingDivideByPOT(
              (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);

          output_data_ptr[c] = std::max(std::min(unsat_output, 255), 0);

        } else {
          output_data_ptr[c] = 0;
        }
      }
    }
  }
}

// TODO(myenik): This is the same as the reference implementation, not actually
// optimized yet.
inline void LogSoftmax(const float* input_data, const Dims<4>& input_dims,
                       float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("LogSoftmax");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // Find max element value which we'll use to ensure numerical stability
        // taking advantage of the following equality:
        // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
        float max = std::numeric_limits<float>::lowest();
        for (int c = 0; c < depth; ++c) {
          max = std::max(max, input_data[Offset(input_dims, c, x, y, b)]);
        }

        // Compute sum.
        float sum = 0.f;
        for (int c = 0; c < depth; ++c) {
          sum += std::exp(input_data[Offset(input_dims, c, x, y, b)] - max);
        }

        // Compute result.
        const float log_sum = std::log(sum);
        for (int c = 0; c < depth; ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              input_data[Offset(input_dims, c, x, y, b)] - max - log_sum;
        }
      }
    }
  }
}

// Currently just a copy of the reference code.
inline void LogSoftmax(const uint8* input_data, const Dims<4>& input_dims,
                       int32 input_multiplier, int32 input_left_shift,
                       int32 reverse_scaling_divisor,
                       int32 reverse_scaling_right_shift, int diff_min,
                       uint8* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("LogSoftmax/Uint8");
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static constexpr int kScaledDiffIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  const int outer_size = MatchingFlatSizeSkipDim(input_dims, 0, output_dims);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  for (int i = 0; i < outer_size; ++i) {
    uint8 max_in_row = 0;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    // TODO(b/77858996): Implement fixed-point log().
    // Not a fully-quantized implementation: floating-point log().
    const float float_log_sum_of_exps =
        std::log(static_cast<float>(sum_of_exps.raw()) /
                 (1 << (31 - kAccumulationIntegerBits)));
    const int32 fixed_log_sum_of_exps = static_cast<int32>(TfLiteRound(
        float_log_sum_of_exps * (1 << (31 - kScaledDiffIntegerBits))));

    // rescaled_diff_min is smallest representable in
    // Q(kScaledDiffIntegerBits).(31-kScaledDiffIntegerBits) plus the
    // log-sub-exps that will be subtracted in the loop.
    //
    // The thresholds diff_min, etc are negative.
    const int rescaled_diff_min =
        fixed_log_sum_of_exps + std::numeric_limits<int32>::lowest();
    const int adjusted_diff_min =
        std::max(diff_min - 1,  // Note use of > below instead of >= above.
                 MultiplyByQuantizedMultiplierSmallerThanOne(
                     rescaled_diff_min, reverse_scaling_divisor,
                     reverse_scaling_right_shift));

    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff > adjusted_diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        int32 unsat_output =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_rescaled - fixed_log_sum_of_exps),
                31 - kScaledDiffIntegerBits - kOutputIntegerBits) +
            255;

        output_data[i * depth + c] = static_cast<uint8>(
            std::max(std::min(unsat_output, static_cast<int32>(255)), 0));
      } else {
        // Set output to smallest value.
        output_data[i * depth + c] = 0;
      }
    }
  }
}

inline void Logistic(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Logistic");
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() =
      input_map.array().unaryExpr(Eigen::internal::scalar_sigmoid_op<float>());
}

inline void Logistic(const uint8* input_data, const Dims<4>& input_dims,
                     int32 input_zero_point, int32 input_range_radius,
                     int32 input_multiplier, int input_left_shift,
                     uint8* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Logistic/Uint8");
  /* batches */ MatchingArraySize(input_dims, 3, output_dims, 3);
  /* height */ MatchingArraySize(input_dims, 2, output_dims, 2);
  /* width */ MatchingArraySize(input_dims, 1, output_dims, 1);
  /* depth */ MatchingArraySize(input_dims, 0, output_dims, 0);
  const int size = RequiredBufferSizeForDims(input_dims);

  int c = 0;
#ifdef USE_NEON
  // Handle 16 values at a time
  for (; c <= size - 16; c += 16) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    uint8x16_t input_val_u8 = vld1q_u8(input_data + c);
    int16x8_t input_val_centered_0 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));
    int16x8_t input_val_centered_1 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint16x8_t mask_rightclamp_0 =
        vcgtq_s16(input_val_centered_0, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_rightclamp_1 =
        vcgtq_s16(input_val_centered_1, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_leftclamp_0 =
        vcgeq_s16(input_val_centered_0, vdupq_n_s16(-input_range_radius));
    uint16x8_t mask_leftclamp_1 =
        vcgeq_s16(input_val_centered_1, vdupq_n_s16(-input_range_radius));
    uint8x16_t mask_rightclamp = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                                             vshrn_n_u16(mask_rightclamp_1, 8));
    uint8x16_t mask_leftclamp = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                                            vshrn_n_u16(mask_leftclamp_1, 8));

    // This performs what is expressed in the scalar code as
    // const int32 input_val_rescaled =
    //     MultiplyByQuantizedMultiplierGreaterThanOne(
    //         input_val_centered, input_multiplier, input_left_shift);
    int32x4_t input_val_rescaled_0 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_1 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_2 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_3 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    input_val_rescaled_0 =
        vqrdmulhq_n_s32(input_val_rescaled_0, input_multiplier);
    input_val_rescaled_1 =
        vqrdmulhq_n_s32(input_val_rescaled_1, input_multiplier);
    input_val_rescaled_2 =
        vqrdmulhq_n_s32(input_val_rescaled_2, input_multiplier);
    input_val_rescaled_3 =
        vqrdmulhq_n_s32(input_val_rescaled_3, input_multiplier);

    // Invoke gemmlowp::logistic on FixedPoint wrapping int32x4_t
    using FixedPoint4 = gemmlowp::FixedPoint<int32x4_t, 4>;
    using FixedPoint0 = gemmlowp::FixedPoint<int32x4_t, 0>;
    const FixedPoint4 input_val_f4_0 =
        FixedPoint4::FromRaw(input_val_rescaled_0);
    const FixedPoint4 input_val_f4_1 =
        FixedPoint4::FromRaw(input_val_rescaled_1);
    const FixedPoint4 input_val_f4_2 =
        FixedPoint4::FromRaw(input_val_rescaled_2);
    const FixedPoint4 input_val_f4_3 =
        FixedPoint4::FromRaw(input_val_rescaled_3);
    const FixedPoint0 output_val_f0_0 = gemmlowp::logistic(input_val_f4_0);
    const FixedPoint0 output_val_f0_1 = gemmlowp::logistic(input_val_f4_1);
    const FixedPoint0 output_val_f0_2 = gemmlowp::logistic(input_val_f4_2);
    const FixedPoint0 output_val_f0_3 = gemmlowp::logistic(input_val_f4_3);

    // Divide by 2^23 as in the scalar code
    using gemmlowp::RoundingDivideByPOT;
    int32x4_t output_val_s32_0 = RoundingDivideByPOT(output_val_f0_0.raw(), 23);
    int32x4_t output_val_s32_1 = RoundingDivideByPOT(output_val_f0_1.raw(), 23);
    int32x4_t output_val_s32_2 = RoundingDivideByPOT(output_val_f0_2.raw(), 23);
    int32x4_t output_val_s32_3 = RoundingDivideByPOT(output_val_f0_3.raw(), 23);

    // Cast output values to uint8, saturating
    int16x8_t output_val_s16_0 = vcombine_s16(vqmovn_s32(output_val_s32_0),
                                              vqmovn_s32(output_val_s32_1));
    int16x8_t output_val_s16_1 = vcombine_s16(vqmovn_s32(output_val_s32_2),
                                              vqmovn_s32(output_val_s32_3));
    uint8x16_t output_val_u8 = vcombine_u8(vqmovun_s16(output_val_s16_0),
                                           vqmovun_s16(output_val_s16_1));

    // Perform the bit-masking with the bit masks computed at the beginning,
    // see the comment there.
    output_val_u8 = vorrq_u8(output_val_u8, mask_rightclamp);
    output_val_u8 = vandq_u8(output_val_u8, mask_leftclamp);

    // Store back to memory
    vst1q_u8(output_data + c, output_val_u8);
  }
#endif
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 23);
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[c] = output_val;
  }
}

inline void Logistic(const int16* input_data, const Dims<4>& input_dims,
                     int16* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Logistic/Int16");
  const int flat_size = RequiredBufferSizeForDims(output_dims);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input_dims), flat_size);

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

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Tanh");
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() = input_map.array().tanh();
}

inline void Tanh(const uint8* input_data, const Dims<4>& input_dims,
                 int32 input_zero_point, int32 input_range_radius,
                 int32 input_multiplier, int input_left_shift,
                 uint8* output_data, const Dims<4>& output_dims) {
  // Note that this is almost the exact same code as in Logistic().
  gemmlowp::ScopedProfilingLabel label("Tanh");
  /* batches */ MatchingArraySize(input_dims, 3, output_dims, 3);
  /* height */ MatchingArraySize(input_dims, 2, output_dims, 2);
  /* width */ MatchingArraySize(input_dims, 1, output_dims, 1);
  /* depth */ MatchingArraySize(input_dims, 0, output_dims, 0);
  const int size = RequiredBufferSizeForDims(input_dims);

  int c = 0;
  int32_t output_zero_point = 128;
#ifdef USE_NEON
  // Handle 16 values at a time
  for (; c <= size - 16; c += 16) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    uint8x16_t input_val_u8 = vld1q_u8(input_data + c);
    int16x8_t input_val_centered_0 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));
    int16x8_t input_val_centered_1 =
        vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_val_u8))),
                  vdupq_n_s16(input_zero_point));

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint16x8_t mask_rightclamp_0 =
        vcgtq_s16(input_val_centered_0, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_rightclamp_1 =
        vcgtq_s16(input_val_centered_1, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_leftclamp_0 =
        vcgeq_s16(input_val_centered_0, vdupq_n_s16(-input_range_radius));
    uint16x8_t mask_leftclamp_1 =
        vcgeq_s16(input_val_centered_1, vdupq_n_s16(-input_range_radius));
    uint8x16_t mask_rightclamp = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                                             vshrn_n_u16(mask_rightclamp_1, 8));
    uint8x16_t mask_leftclamp = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                                            vshrn_n_u16(mask_leftclamp_1, 8));

    // This performs what is expressed in the scalar code as
    // const int32 input_val_rescaled =
    //     MultiplyByQuantizedMultiplierGreaterThanOne(
    //         input_val_centered, input_multiplier, input_left_shift);
    int32x4_t input_val_rescaled_0 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_1 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_0)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_2 =
        vshlq_s32(vmovl_s16(vget_low_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_3 =
        vshlq_s32(vmovl_s16(vget_high_s16(input_val_centered_1)),
                  vdupq_n_s32(input_left_shift));
    input_val_rescaled_0 =
        vqrdmulhq_n_s32(input_val_rescaled_0, input_multiplier);
    input_val_rescaled_1 =
        vqrdmulhq_n_s32(input_val_rescaled_1, input_multiplier);
    input_val_rescaled_2 =
        vqrdmulhq_n_s32(input_val_rescaled_2, input_multiplier);
    input_val_rescaled_3 =
        vqrdmulhq_n_s32(input_val_rescaled_3, input_multiplier);

    // Invoke gemmlowp::tanh on FixedPoint wrapping int32x4_t
    using FixedPoint4 = gemmlowp::FixedPoint<int32x4_t, 4>;
    using FixedPoint0 = gemmlowp::FixedPoint<int32x4_t, 0>;
    const FixedPoint4 input_val_f4_0 =
        FixedPoint4::FromRaw(input_val_rescaled_0);
    const FixedPoint4 input_val_f4_1 =
        FixedPoint4::FromRaw(input_val_rescaled_1);
    const FixedPoint4 input_val_f4_2 =
        FixedPoint4::FromRaw(input_val_rescaled_2);
    const FixedPoint4 input_val_f4_3 =
        FixedPoint4::FromRaw(input_val_rescaled_3);
    const FixedPoint0 output_val_f0_0 = gemmlowp::tanh(input_val_f4_0);
    const FixedPoint0 output_val_f0_1 = gemmlowp::tanh(input_val_f4_1);
    const FixedPoint0 output_val_f0_2 = gemmlowp::tanh(input_val_f4_2);
    const FixedPoint0 output_val_f0_3 = gemmlowp::tanh(input_val_f4_3);

    // Divide by 2^24 as in the scalar code
    using gemmlowp::RoundingDivideByPOT;
    int32x4_t output_val_s32_0 = RoundingDivideByPOT(output_val_f0_0.raw(), 24);
    int32x4_t output_val_s32_1 = RoundingDivideByPOT(output_val_f0_1.raw(), 24);
    int32x4_t output_val_s32_2 = RoundingDivideByPOT(output_val_f0_2.raw(), 24);
    int32x4_t output_val_s32_3 = RoundingDivideByPOT(output_val_f0_3.raw(), 24);

    // Add the output zero point
    int32x4_t output_zero_point_s32 = vdupq_n_s32(output_zero_point);
    output_val_s32_0 = vaddq_s32(output_val_s32_0, output_zero_point_s32);
    output_val_s32_1 = vaddq_s32(output_val_s32_1, output_zero_point_s32);
    output_val_s32_2 = vaddq_s32(output_val_s32_2, output_zero_point_s32);
    output_val_s32_3 = vaddq_s32(output_val_s32_3, output_zero_point_s32);

    // Cast output values to uint8, saturating
    int16x8_t output_val_s16_0 = vcombine_s16(vqmovn_s32(output_val_s32_0),
                                              vqmovn_s32(output_val_s32_1));
    int16x8_t output_val_s16_1 = vcombine_s16(vqmovn_s32(output_val_s32_2),
                                              vqmovn_s32(output_val_s32_3));
    uint8x16_t output_val_u8 = vcombine_u8(vqmovun_s16(output_val_s16_0),
                                           vqmovun_s16(output_val_s16_1));

    // Perform the bit-masking with the bit masks computed at the beginning,
    // see the comment there.
    output_val_u8 = vorrq_u8(output_val_u8, mask_rightclamp);
    output_val_u8 = vandq_u8(output_val_u8, mask_leftclamp);

    // Store back to memory
    vst1q_u8(output_data + c, output_val_u8);
  }
#endif
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int32 input_val_centered =
        static_cast<int32>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      const int32 input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int32 output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 24);
      output_val_s32 += output_zero_point;
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s32, 0);
      TFLITE_DCHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8>(output_val_s32);
    }
    output_data[c] = output_val;
  }
}

inline void Tanh(const int16* input_data, const Dims<4>& input_dims,
                 int input_left_shift, int16* output_data,
                 const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Tanh/Int16");
  // Support for shifts is limited until we have a parameterized version of
  // SaturatingRoundingMultiplyByPOT().
  TFLITE_DCHECK_GE(input_left_shift, 0);
  TFLITE_DCHECK_LE(input_left_shift, 1);

  const int flat_size = RequiredBufferSizeForDims(output_dims);
  TFLITE_DCHECK_EQ(RequiredBufferSizeForDims(input_dims), flat_size);

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

inline void Dequantize(const uint8* input_data, const Dims<4>& input_dims,
                       int32 zero_point, double scale, float* output_data,
                       const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Dequantize");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          int32 val = input_data[Offset(input_dims, c, x, y, b)];
          float result = static_cast<float>(scale * (val - zero_point));
          output_data[Offset(output_dims, c, x, y, b)] = result;
        }
      }
    }
  }
}

inline void FakeQuant(const float* input_data, const Dims<4>& input_dims,
                      float rmin, float rmax, int num_bits, float* output_data,
                      const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("FakeQuant");

  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_DCHECK_LE(rmin, 0.0f);
  TFLITE_DCHECK_GE(rmax, 0.0f);
  TFLITE_DCHECK_LT(rmin, rmax);

  // Code matches tensorflow's FakeQuantWithMinMaxArgsFunctor.
  int quant_min = 0;
  int quant_max = (1 << num_bits) - 1;
  float nudged_min, nudged_max, nudged_scale;
  NudgeQuantizationRange(rmin, rmax, quant_min, quant_max, &nudged_min,
                         &nudged_max, &nudged_scale);
  const float inv_nudged_scale = 1.0f / nudged_scale;

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          const float src_val = input_data[Offset(input_dims, c, x, y, b)];
          const float clamped =
              std::min(nudged_max, std::max(nudged_min, src_val));
          const float clamped_shifted = clamped - nudged_min;
          const float dst_val =
              TfLiteRound(clamped_shifted * inv_nudged_scale) * nudged_scale +
              nudged_min;
          output_data[Offset(output_dims, c, x, y, b)] = dst_val;
        }
      }
    }
  }
}

template <typename SrcT, typename DstT>
inline void Cast(const SrcT* input_data, const Dims<4>& input_dims,
                 DstT* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Cast");
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() = input_map.array().template cast<DstT>();
}

inline void Floor(const float* input_data, const Dims<4>& input_dims,
                  float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Floor");
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() = Eigen::floor(input_map.array());
}

template <typename T>
inline void Gather(const T* input_data, const Dims<4>& input_dims,
                   int input_rank, const int32* coords_data,
                   const Dims<4>& coords_dims, T* output_data,
                   const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Gather");

  TFLITE_DCHECK(coords_dims.sizes[0] == output_dims.sizes[input_rank - 1]);
  int stride = input_dims.strides[input_rank - 1];
  T* out = output_data;

  for (int i = 0; i < coords_dims.sizes[0]; i++) {
    TFLITE_DCHECK_GE(coords_data[i], 0);
    TFLITE_DCHECK_LT(coords_data[i], input_dims.sizes[input_rank - 1]);
    const T* in = input_data + coords_data[i] * stride;
    memcpy(out, in, sizeof(T) * stride);
    out += stride;
  }
}

#ifdef USE_NEON
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  int ic = 0;
  // Handle 32 input channels at a time.
  for (; ic <= depth - 32; ic += 32) {
    float32x4x2_t input[4];
    for (int i = 0; i < 4; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[4];
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 4; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 32;
  }
  // Handle 16 input channels at a time.
  for (; ic <= depth - 16; ic += 16) {
    float32x4x2_t input[2];
    for (int i = 0; i < 2; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[2];
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 2; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 16;
  }
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    float32x4x2_t input;
    input.val[0] = vld1q_f32(input_ptr);
    input.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t acc;
    acc.val[0] = vld1q_f32(output_ptr);
    acc.val[1] = vld1q_f32(output_ptr + 4);
    acc.val[0] = vmlaq_n_f32(acc.val[0], input.val[0], scale);
    acc.val[1] = vmlaq_n_f32(acc.val[1], input.val[1], scale);

    vst1q_f32(output_ptr, acc.val[0]);
    vst1q_f32(output_ptr + 4, acc.val[1]);

    input_ptr += 8;
    output_ptr += 8;
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    float32x4_t input = vld1q_f32(input_ptr);
    float32x4_t acc = vld1q_f32(output_ptr);

    acc = vmlaq_n_f32(acc, input, scale);
    vst1q_f32(output_ptr, acc);

    input_ptr += 4;
    output_ptr += 4;
  }
  // Handle 1 input channel at a time.
  for (; ic < depth; ic++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#else
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  for (int32 i = 0; i < depth; i++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#endif

inline void ResizeBilinearKernel2x2(int32 x0, int32 x1, int32 y0, int32 y1,
                                    int32 x, int32 y, int32 depth, int32 batch,
                                    const float* input_data,
                                    const Dims<4>& input_dims,
                                    float* output_data,
                                    const Dims<4>& output_dims) {
  const int32 input_width = ArraySize(input_dims, 1);
  const int32 output_width = ArraySize(output_dims, 1);

  const int32 input_x_offset = (x1 - x0) * depth;
  const int32 input_y_offset = (y1 - y0) * depth * input_width;
  const int32 output_x_offset = depth;
  const int32 output_y_offset = depth * output_width;

#ifdef USE_NEON
  TFLITE_DCHECK(IsPackedWithoutStrides(input_dims));
  TFLITE_DCHECK(x1 >= x0);
  TFLITE_DCHECK(y1 >= y0);

  int ic = 0;
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    const float* input_ptr = nullptr;

    float32x4x2_t x0y0;
    input_ptr = &input_data[Offset(input_dims, ic, x0, y0, batch)];
    x0y0.val[0] = vld1q_f32(input_ptr);
    x0y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y0;
    input_ptr += input_x_offset;
    x1y0.val[0] = vld1q_f32(input_ptr);
    x1y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x0y1;
    input_ptr += -input_x_offset + input_y_offset;
    x0y1.val[0] = vld1q_f32(input_ptr);
    x0y1.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y1;
    input_ptr += input_x_offset;
    x1y1.val[0] = vld1q_f32(input_ptr);
    x1y1.val[1] = vld1q_f32(input_ptr + 4);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_dims, ic, x, y, batch)];
    vst1q_f32(output_ptr, x0y0.val[0]);
    vst1q_f32(output_ptr + 4, x0y0.val[1]);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4x2_t tr;
    tr.val[0] = vaddq_f32(x0y0.val[0], x1y0.val[0]);
    tr.val[1] = vaddq_f32(x0y0.val[1], x1y0.val[1]);
    tr.val[0] = vmulq_n_f32(tr.val[0], 0.5f);
    tr.val[1] = vmulq_n_f32(tr.val[1], 0.5f);

    vst1q_f32(output_ptr, tr.val[0]);
    vst1q_f32(output_ptr + 4, tr.val[1]);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4x2_t bl;
    bl.val[0] = vaddq_f32(x0y0.val[0], x0y1.val[0]);
    bl.val[1] = vaddq_f32(x0y0.val[1], x0y1.val[1]);
    bl.val[0] = vmulq_n_f32(bl.val[0], 0.5f);
    bl.val[1] = vmulq_n_f32(bl.val[1], 0.5f);
    vst1q_f32(output_ptr, bl.val[0]);
    vst1q_f32(output_ptr + 4, bl.val[1]);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4x2_t br;
    br.val[0] = vaddq_f32(x1y0.val[0], x1y1.val[0]);
    br.val[1] = vaddq_f32(x1y0.val[1], x1y1.val[1]);
    br.val[0] = vmlaq_n_f32(bl.val[0], br.val[0], 0.5f);
    br.val[1] = vmlaq_n_f32(bl.val[1], br.val[1], 0.5f);
    br.val[0] = vmulq_n_f32(br.val[0], 0.5f);
    br.val[1] = vmulq_n_f32(br.val[1], 0.5f);
    vst1q_f32(output_ptr, br.val[0]);
    vst1q_f32(output_ptr + 4, br.val[1]);
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    const float* input_ptr = &input_data[Offset(input_dims, ic, x0, y0, batch)];
    float32x4_t x0y0 = vld1q_f32(input_ptr);
    float32x4_t x1y0 = vld1q_f32(input_ptr + input_x_offset);
    float32x4_t x0y1 = vld1q_f32(input_ptr + input_y_offset);
    float32x4_t x1y1 = vld1q_f32(input_ptr + input_x_offset + input_y_offset);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_dims, ic, x, y, batch)];
    vst1q_f32(output_ptr, x0y0);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4_t tr = vaddq_f32(x0y0, x1y0);
    tr = vmulq_n_f32(tr, 0.5f);
    vst1q_f32(output_ptr, tr);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4_t bl = vaddq_f32(x0y0, x0y1);
    bl = vmulq_n_f32(bl, 0.5f);
    vst1q_f32(output_ptr, bl);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4_t br = vaddq_f32(x1y0, x1y1);
    br = vmlaq_n_f32(bl, br, 0.5f);
    br = vmulq_n_f32(br, 0.5f);
    vst1q_f32(output_ptr, br);
  }
  // Handle one input channel at a time.
  for (; ic < depth; ic++) {
    const int32 input_offset = Offset(input_dims, ic, x0, y0, batch);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_dims, ic, x, y, batch);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#else
  for (int ch = 0; ch < depth; ch++) {
    const int32 input_offset = Offset(input_dims, ch, x0, y0, batch);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_dims, ch, x, y, batch);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#endif
}

inline void ResizeBilinear2x2(const float* input_data,
                              const Dims<4>& input_dims, float* output_data,
                              const Dims<4>& output_dims, int32 batches,
                              int32 input_height, int32 input_width,
                              int32 depth, int32 output_height,
                              int32 output_width) {
  for (int b = 0; b < batches; b++) {
    for (int y0 = 0, y = 0; y <= output_height - 2; y += 2, y0++) {
      for (int x0 = 0, x = 0; x <= output_width - 2; x += 2, x0++) {
        int32 x1 = std::min(x0 + 1, input_width - 1);
        int32 y1 = std::min(y0 + 1, input_height - 1);
        ResizeBilinearKernel2x2(x0, x1, y0, y1, x, y, depth, b, input_data,
                                input_dims, output_data, output_dims);
      }
    }
  }
}

inline void ResizeBilinearGeneric(const float* input_data,
                                  const Dims<4>& input_dims, float* output_data,
                                  const Dims<4>& output_dims, int32 batches,
                                  int32 input_height, int32 input_width,
                                  int32 depth, int32 output_height,
                                  int32 output_width, float height_scale,
                                  float width_scale) {
  memset(output_data, 0,
         batches * output_height * output_width * depth * sizeof(float));

  int32 output_offset = 0;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = y * height_scale;
      int32 y0 = static_cast<int32>(std::floor(input_y));
      int32 y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = x * width_scale;
        int32 x0 = static_cast<int32>(input_x);
        int32 x1 = std::min(x0 + 1, input_width - 1);
        float* output_ptr = &output_data[output_offset];

        // Run kernel on the 4 corners of the bilinear resize algorithm.
        int32 input_offset = Offset(input_dims, 0, x0, y0, b);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        const float* input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_dims, 0, x1, y0, b);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_dims, 0, x0, y1, b);
        scale = (input_y - y0) * (1 - (input_x - x0));
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_dims, 0, x1, y1, b);
        scale = (input_y - y0) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        output_offset += depth;
      }
    }
  }
}

inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims, bool align_corners) {
  gemmlowp::ScopedProfilingLabel label("ResizeBilinear");
  int32 batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  int32 input_height = ArraySize(input_dims, 2);
  int32 input_width = ArraySize(input_dims, 1);
  int32 depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 3), 1);
  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 2), 1);
  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 1), 1);
  TFLITE_DCHECK_EQ(ArraySize(output_size_dims, 0), 2);
  int32 output_height = output_size_data[Offset(output_size_dims, 0, 0, 0, 0)];
  int32 output_width = output_size_data[Offset(output_size_dims, 1, 0, 0, 0)];

  // Specialize for 2x2 upsample.
  if (!align_corners && output_height == 2 * input_height &&
      output_width == 2 * input_width) {
    ResizeBilinear2x2(input_data, input_dims, output_data, output_dims, batches,
                      input_height, input_width, depth, output_height,
                      output_width);
  } else {
    float height_scale = static_cast<float>(input_height) / output_height;
    float width_scale = static_cast<float>(input_width) / output_width;
    if (align_corners && output_height > 1) {
      height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
    }
    if (align_corners && output_width > 1) {
      width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
    }

    ResizeBilinearGeneric(input_data, input_dims, output_data, output_dims,
                          batches, input_height, input_width, depth,
                          output_height, output_width, height_scale,
                          width_scale);
  }
}

// legacy, for compatibility with old checked-in code
inline void ResizeBilinear(const float* input_data, const Dims<4>& input_dims,
                           const int32* output_size_data,
                           const Dims<4>& output_size_dims, float* output_data,
                           const Dims<4>& output_dims) {
  ResizeBilinear(input_data, input_dims, output_size_data, output_size_dims,
                 output_data, output_dims, /*align_corners=*/false);
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims) {
  // Unoptimized - Straight copy from reference ops.
  gemmlowp::ScopedProfilingLabel label("SpaceToBatchND");

  const int output_batch_size = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int input_batch_size = ArraySize(input_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int depth = ArraySize(input_dims, 0);
  const int block_shape_height = block_shape_data[0];
  const int block_shape_width = block_shape_data[1];
  const int padding_top = paddings_data[0];
  const int padding_left = paddings_data[2];

  for (int out_b = 0; out_b < output_batch_size; ++out_b) {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        T* out = output_data + Offset(output_dims, 0, out_w, out_h, out_b);
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >=
                padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          memset(out, 0, depth * sizeof(T));
        } else {
          const T* in =
              input_data +
              Offset(input_dims, 0,
                     (out_w * block_shape_width + shift_w) - padding_left,
                     (out_h * block_shape_height + shift_h) - padding_top,
                     input_batch);
          memcpy(out, in, depth * sizeof(T));
        }
      }
    }
  }
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
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* crops_data, const Dims<4>& crops_dims,
                           T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BatchToSpaceND");

  const int output_batch_size = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int input_batch_size = ArraySize(input_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int depth = ArraySize(input_dims, 0);
  const int block_shape_width = block_shape_data[1];
  const int block_shape_height = block_shape_data[0];
  const int crops_top = crops_data[0];
  const int crops_left = crops_data[2];

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
        T* out = output_data + Offset(output_dims, 0, out_w, out_h, out_batch);
        const T* in = input_data + Offset(input_dims, 0, in_w, in_h, in_batch);
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

template <typename T>
inline void PadV2(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& left_paddings,
                  const std::vector<int>& right_paddings, T* output_data,
                  const Dims<4>& output_dims, const T pad_value) {
  gemmlowp::ScopedProfilingLabel label("Pad");
  TFLITE_DCHECK_EQ(left_paddings.size(), 4);
  TFLITE_DCHECK_EQ(right_paddings.size(), 4);

  const int output_batch = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int output_depth = ArraySize(output_dims, 0);

  const int left_b_padding = left_paddings[3];
  const int left_h_padding = left_paddings[2];
  const int left_w_padding = left_paddings[1];
  const int left_d_padding = left_paddings[0];

  const int right_b_padding = right_paddings[3];
  const int right_h_padding = right_paddings[2];
  const int right_w_padding = right_paddings[1];
  const int right_d_padding = right_paddings[0];

  const int input_depth = ArraySize(input_dims, 0);

  if (left_b_padding != 0) {
    TypedMemset<T>(
        output_data, pad_value,
        left_b_padding * output_height * output_width * output_depth);
  }
  for (int out_b = left_b_padding; out_b < output_batch - right_b_padding;
       ++out_b) {
    if (left_h_padding != 0) {
      TypedMemset<T>(output_data + Offset(output_dims, 0, 0, 0, out_b),
                     pad_value, left_h_padding * output_width * output_depth);
    }
    for (int out_h = left_h_padding; out_h < output_height - right_h_padding;
         ++out_h) {
      if (left_w_padding != 0) {
        TypedMemset<T>(output_data + Offset(output_dims, 0, 0, out_h, out_b),
                       pad_value, left_w_padding * output_depth);
      }
      for (int out_w = left_w_padding; out_w < output_width - right_w_padding;
           ++out_w) {
        if (left_d_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(output_dims, 0, out_w, out_h, out_b),
              pad_value, left_d_padding);
        }

        T* out = output_data +
                 Offset(output_dims, left_d_padding, out_w, out_h, out_b);
        const T* in =
            input_data + Offset(input_dims, 0, out_w - left_w_padding,
                                out_h - left_h_padding, out_b - left_b_padding);
        memcpy(out, in, input_depth * sizeof(T));

        if (right_d_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(output_dims, output_depth - right_d_padding,
                                   out_w, out_h, out_b),
              pad_value, right_d_padding);
        }
      }
      if (right_w_padding != 0) {
        TypedMemset<T>(
            output_data + Offset(output_dims, 0, output_width - right_w_padding,
                                 out_h, out_b),
            pad_value, right_w_padding * output_depth);
      }
    }
    if (right_h_padding != 0) {
      TypedMemset<T>(
          output_data +
              Offset(output_dims, 0, 0, output_height - right_h_padding, out_b),
          pad_value, right_h_padding * output_width * output_depth);
    }
  }
  if (right_b_padding != 0) {
    TypedMemset<T>(
        output_data +
            Offset(output_dims, 0, 0, 0, output_batch - right_b_padding),
        pad_value,
        right_b_padding * output_height * output_width * output_depth);
  }
}

// Legacy Pad() method that casts an int32_t to T before padding.
template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims, const int32_t pad_value) {
  const T converted_pad_value = static_cast<T>(pad_value);
  PadV2<T>(input_data, input_dims, left_paddings, right_paddings, output_data,
           output_dims, converted_pad_value);
}

template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  Pad(input_data, input_dims, left_paddings, right_paddings, output_data,
      output_dims, 0);
}

// UNOPTIMIZED COPY of StridedSlice from reference_ops.h.
template <typename T>
inline void StridedSlice(const T* input_data, const Dims<4>& input_dims,
                         int begin_mask, int end_mask,
                         const std::vector<int>& start_indices,
                         const std::vector<int>& stop_indices,
                         const std::vector<int>& strides, T* output_data,
                         const Dims<4>& output_dims) {
  TFLITE_DCHECK_EQ(start_indices.size(), 4);
  TFLITE_DCHECK_EQ(stop_indices.size(), 4);
  TFLITE_DCHECK_EQ(strides.size(), 4);
  const int start_b = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 3);
  const int stop_b = strided_slice::StopForAxis(end_mask, stop_indices, strides,
                                                input_dims.sizes, 3);
  const int start_h = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 2);
  const int stop_h = strided_slice::StopForAxis(end_mask, stop_indices, strides,
                                                input_dims.sizes, 2);
  const int start_w = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 1);
  const int stop_w = strided_slice::StopForAxis(end_mask, stop_indices, strides,
                                                input_dims.sizes, 1);
  const int start_d = strided_slice::StartForAxis(begin_mask, start_indices,
                                                  strides, input_dims.sizes, 0);
  const int stop_d = strided_slice::StopForAxis(end_mask, stop_indices, strides,
                                                input_dims.sizes, 0);

  T* out_ptr = output_data;
  for (int in_b = start_b;
       !strided_slice::LoopCondition(in_b, stop_b, strides[3]);
       in_b += strides[3]) {
    for (int in_h = start_h;
         !strided_slice::LoopCondition(in_h, stop_h, strides[2]);
         in_h += strides[2]) {
      for (int in_w = start_w;
           !strided_slice::LoopCondition(in_w, stop_w, strides[1]);
           in_w += strides[1]) {
        for (int in_d = start_d;
             !strided_slice::LoopCondition(in_d, stop_d, strides[0]);
             in_d += strides[0]) {
          *out_ptr++ = input_data[Offset(input_dims, in_d, in_w, in_h, in_b)];
        }
      }
    }
  }
}

template <typename T>
inline void Slice(const T* input_data, const Dims<4>& input_dims,
                  const std::vector<int>& begin, const std::vector<int>& size,
                  T* output_data, const Dims<4>& output_dims) {
  // TODO(dkalenichenko): This op only supports 4D tensors.
  TFLITE_DCHECK_EQ(begin.size(), 4);
  TFLITE_DCHECK_EQ(size.size(), 4);
  const int start_b = begin[3];
  const int stop_b =
      size[3] == -1 ? input_dims.sizes[3] - start_b : start_b + size[3];
  const int start_h = begin[2];
  const int stop_h =
      size[2] == -1 ? input_dims.sizes[2] - start_h : start_h + size[2];
  const int start_w = begin[1];
  const int stop_w =
      size[1] == -1 ? input_dims.sizes[1] - start_w : start_w + size[1];
  const int start_d = begin[0];
  const int stop_d =
      size[0] == -1 ? input_dims.sizes[0] - start_d : start_d + size[0];

  T* out_ptr = output_data;
  for (int in_b = start_b; in_b < stop_b; ++in_b) {
    for (int in_h = start_h; in_h < stop_h; ++in_h) {
      for (int in_w = start_w; in_w < stop_w; ++in_w) {
        const int len = stop_d - start_d;
        memcpy(out_ptr,
               input_data + Offset(input_dims, start_d, in_w, in_h, in_b),
               len * sizeof(T));
        out_ptr += len;
      }
    }
  }
}

template <typename T>
inline void Mean(const T* input_data, const Dims<4>& input_dims,
                 const std::vector<int>& reduction_indices, T* output_data,
                 const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Mean");
  const int output_batch = ArraySize(output_dims, 3);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  const int output_depth = ArraySize(output_dims, 0);

  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);

  // The current implementation only supports simultaneous reduction over
  // width and height.
  TFLITE_DCHECK_EQ(reduction_indices.size(), 2);
  TFLITE_DCHECK((reduction_indices[0] == 1 && reduction_indices[1] == 2) ||
                (reduction_indices[0] == 2 && reduction_indices[1] == 1));
  TFLITE_DCHECK_EQ(output_height, 1);
  TFLITE_DCHECK_EQ(output_width, 1);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_d = 0; out_d < output_depth; ++out_d) {
      float value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          value += input_data[Offset(input_dims, out_d, in_w, in_h, out_b)];
        }
      }
      output_data[Offset(output_dims, out_d, 0, 0, out_b)] =
          value / (input_width * input_height);
    }
  }
}

template <typename T>
void GenericBroadcastSub(const T* input1_data, const Dims<4>& input1_dims,
                         const T* input2_data, const Dims<4>& input2_dims,
                         T* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("GenericBroadcastSub");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_dims, input2_dims, &desc1, &desc2);

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
  for (int b = 0; b < ArraySize(output_dims, 3); ++b) {
    for (int y = 0; y < ArraySize(output_dims, 2); ++y) {
      for (int x = 0; x < ArraySize(output_dims, 1); ++x) {
        for (int c = 0; c < ArraySize(output_dims, 0); ++c) {
          output_data[Offset(output_dims, c, x, y, b)] =
              input1_data[SubscriptToIndex(desc1, c, x, y, b)] -
              input2_data[SubscriptToIndex(desc2, c, x, y, b)];
        }
      }
    }
  }
}

template <typename T>
void Sub(const T* input1_data, const Dims<4>& input1_dims, const T* input2_data,
         const Dims<4>& input2_dims, T* output_data,
         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Sub");

  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto input2_map = MapAsVector(input2_data, input2_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  if (AreSameDims(input1_dims, input2_dims)) {
    output_map.array() = input1_map.array() - input2_map.array();
  } else if (RequiredBufferSizeForDims(input1_dims) == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar - input2_map.array();
  } else if (RequiredBufferSizeForDims(input2_dims) == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() - scalar;
  } else {
    GenericBroadcastSub(input1_data, input1_dims, input2_data, input2_dims,
                        output_data, output_dims);
  }
}

template <typename T>
void TensorFlowMinimum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("TensorFlowMinimum");
  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  auto min_value = input2_data[0];
  output_map.array() = input1_map.array().min(min_value);
}

template <typename T>
void TensorFlowMaximum(const T* input1_data, const Dims<4>& input1_dims,
                       const T* input2_data, T* output_data,
                       const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("TensorFlowMaximum");
  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  auto max_value = input2_data[0];
  output_map.array() = input1_map.array().max(max_value);
}

template <typename T1, typename T2, typename T3>
void ArgMax(const T3* axis, const T1* input_data, const Dims<4>& input_dims,
            T2* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("ArgMax");

  // The current ArgMax implemention can only determine the index of the maximum
  // value in the last dimension. So the axis argument is ignored.
  TFLITE_DCHECK_EQ(axis[0], 3);

  // For ArgMax, the number of output dimensions = (number of input dimensions -
  // 1). For the sake of simplicity, the output dimensions are equal to the
  // input dimensions here. We enforce the constraint that the last dimension
  // must always be 1.
  TFLITE_DCHECK_EQ(ArraySize(output_dims, 0), 1);
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = ArraySize(input_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        auto max_value = input_data[Offset(input_dims, 0, x, y, b)];
        int max_index = 0;
        for (int d = 1; d < depth; ++d) {
          const auto& curr_value = input_data[Offset(input_dims, d, x, y, b)];
          if (curr_value > max_value) {
            max_value = curr_value;
            max_index = d;
          }
        }
        output_data[Offset(output_dims, 0, x, y, b)] = max_index;
      }
    }
  }
}

template <typename T>
void Transpose(const T* input, const Dims<4>& input_dims, T* output,
               const Dims<4>& output_dims, const int* permuted_axes) {
  int out_sizes[4];
  // Compute the inverse permutation array so we can do an output centered
  // transpose. Also, check to make sure output_dims is matching input_dims.
  for (int k = 0; k < 4; k++) {
    out_sizes[k] =
        MatchingArraySize(input_dims, permuted_axes[k], output_dims, k);
  }

  // Naive transpose loop (iterate on output index and compute input index).
  int o[4];  // loop index (on output).
  int i[4];
  for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
    i[permuted_axes[3]] = o[3];
    for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
      i[permuted_axes[2]] = o[2];
      for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
        i[permuted_axes[1]] = o[1];
        for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
          i[permuted_axes[0]] = o[0];
          output[Offset(output_dims, o)] = input[Offset(input_dims, i)];
        }
      }
    }
  }
}

inline void TransposeConv(const float* input_data, const Dims<4>& input_dims,
                          const float* filter_data, const Dims<4>& filter_dims,
                          int stride_width, int stride_height, int pad_width,
                          int pad_height, float* output_data,
                          const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("TransposeConv");
  // THIS FUNCTION IS A COPY FROM reference_ops.h.
  // To optimize, start by using the conv code with transposed weights for the
  // case of stride_height = stride_width = 1.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 3);
  const int output_depth = MatchingArraySize(filter_dims, 0, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  // Although transpose convolution simplifies to convolution with transposed
  // weights for strides of 1, non-unitary striding complicates matters. To
  // keep this reference implementation as clear as possible, we use a "scatter"
  // access pattern, where we loop through all the input elements, computing
  // their influence on the output, rather than looping through the output
  // elements in the typical "gather" access pattern of a conv. We therefore
  // must initialize the output array to zero.
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              0.0f;
        }
      }
    }
  }

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < input_depth;
                   ++out_channel) {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  float input_value = input_data[Offset(input_dims, in_channel,
                                                        in_x, in_y, batch)];
                  float filter_value =
                      filter_data[Offset(filter_dims, out_channel, filter_x,
                                         filter_y, in_channel)];
                  output_data[Offset(output_dims, out_channel, out_x, out_y,
                                     batch)] += input_value * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#if defined OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#undef OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#pragma GCC diagnostic pop
#endif

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_OPS_H_
