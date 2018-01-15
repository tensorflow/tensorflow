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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_OPS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_OPS_H_

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
#include "tensorflow/contrib/lite/kernels/internal/round.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

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

inline void preload_l1_stream(const uint8* ptr) {
#ifdef GEMMLOWP_ARM_64
  asm volatile("prfm pldl1strm, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#else
  gemmlowp::Prefetch(ptr);
#endif
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
    preload_l1_stream(input_data + k);
  }
  for (int k = 0; k < kPeel * input_size; k += 64) {
    preload_l1_stream(filter_data + k);
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
        preload_l1_stream(filter_ptr + 64);
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

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, float output_activation_min,
                 float output_activation_max, float* output_data,
                 const Dims<4>& output_dims, float* im2col_data,
                 const Dims<4>& im2col_dims) {
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
       stride_width, stride_height, pad_width, pad_height,
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
           bias_dims, stride, stride, pad_width, pad_height, output_data,
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
  TFLITE_DCHECK_GT(input1_offset, -256);
  TFLITE_DCHECK_GT(input2_offset, -256);
  TFLITE_DCHECK_LT(input1_offset, 256);
  TFLITE_DCHECK_LT(input2_offset, 256);
#ifdef USE_NEON
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
    vst1_u8(output_data + i, vqmovun_s16(s));
  }
#endif  // NEON

  for (; i < size; i++) {
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
template <FusedActivationFunctionType Ac, typename T>
void BroadcastAdd(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
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
          output_data[Offset(output_dims, c, x, y, b)] = ActivationFunction<Ac>(
              input1_data[SubscriptToIndex(desc1, c, x, y, b)] +
              input2_data[SubscriptToIndex(desc2, c, x, y, b)]);
        }
      }
    }
  }
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
  gemmlowp::ScopedProfilingLabel label("BroadcastAdd/8bit");

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

// TODO(jiawen): We can implement BroadcastMul on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastMul is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <FusedActivationFunctionType Ac, typename T>
void BroadcastMul(const T* input1_data, const Dims<4>& input1_dims,
                  const T* input2_data, const Dims<4>& input2_dims,
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
          output_data[Offset(output_dims, c, x, y, b)] = ActivationFunction<Ac>(
              input1_data[SubscriptToIndex(desc1, c, x, y, b)] *
              input2_data[SubscriptToIndex(desc2, c, x, y, b)]);
        }
      }
    }
  }
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

  gemmlowp::ScopedProfilingLabel label("Softmax");
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int x = 0; x < width; ++x) {
      for (int y = 0; y < height; ++y) {
        uint8 max_in_row = 0;
        for (int c = 0; c < depth; ++c) {
          max_in_row =
              std::max(max_in_row, input_data[Offset(input_dims, c, x, y, b)]);
        }

        FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
        for (int c = 0; c < depth; ++c) {
          int32 input_diff =
              static_cast<int32>(input_data[Offset(input_dims, c, x, y, b)]) -
              max_in_row;
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

        int32 fixed_sum_of_exps = sum_of_exps.raw();
        // TODO(starka): Use a NEON intrinsic like vclzq_u32 instead.
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

        for (int c = 0; c < depth; ++c) {
          int32 input_diff =
              static_cast<int32>(input_data[Offset(input_dims, c, x, y, b)]) -
              max_in_row;
          if (input_diff >= diff_min) {
            const int32 input_diff_rescaled =
                MultiplyByQuantizedMultiplierGreaterThanOne(
                    input_diff, input_beta_multiplier, input_beta_left_shift);
            const FixedPointScaledDiff scaled_diff_f8 =
                FixedPointScaledDiff::FromRaw(input_diff_rescaled);

            FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
            int32 unsat_output = gemmlowp::RoundingDivideByPOT(
                (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);

            output_data[Offset(output_dims, c, x, y, b)] =
                std::max(std::min(unsat_output, 255), 0);

          } else {
            output_data[Offset(output_dims, c, x, y, b)] = 0;
          }
        }
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
  gemmlowp::ScopedProfilingLabel label("Logistic");
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

inline void Tanh(const float* input_data, const Dims<4>& input_dims,
                 float* output_data, const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Tanh");
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() = input_map.array().tanh();
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
                      float rmin, float rmax, float* output_data,
                      const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("FakeQuant");

  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_DCHECK_LE(rmin, 0.);
  TFLITE_DCHECK_GE(rmax, 0.);

  // Determine quantization parameters: zero_point, scale.
  using Integer = uint8;
  const Integer qmin = std::numeric_limits<Integer>::min();
  const Integer qmax = std::numeric_limits<Integer>::max();
  const float qmin_float = qmin;
  const float qmax_float = qmax;
  int32 zero_point = 0;
  float scale = 0.f;
  // If rmin==rmax, both must be zero per the above assertion,
  // so we are done.
  if (rmin != rmax) {
    // First determine the scale.
    scale = (rmax - rmin) / (qmax_float - qmin_float);

    // Zero-point computation.
    // First the initial floating-point computation. The zero-point can be
    // determined from solving an affine equation for any known pair
    // (real value, corresponding quantized value).
    // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    // The arithmetic error on the zero point computed from either pair
    // will be roughly machine_epsilon * (sum of absolute values of terms)
    // so we want to use the variant that adds the smaller terms.
    const float zero_point_from_min = qmin_float - rmin / scale;
    const float zero_point_from_max = qmax_float - rmax / scale;
    const float zero_point_from_min_error =
        std::abs(qmin_float) + std::abs(rmin / scale);
    const float zero_point_from_max_error =
        std::abs(qmax_float) + std::abs(rmax / scale);

    const float zero_point_float =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;

    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the requirement
    // to be able to represent the real value "0" exactly as a quantized value,
    // which is required in multiple places, for example in Im2col with SAME
    // padding).
    if (zero_point_float < qmin_float) {
      zero_point = qmin;
    } else if (zero_point_float > qmax_float) {
      zero_point = qmax;
    } else {
      zero_point = static_cast<int32>(TfLiteRound(zero_point_float));
    }
    // The zero point should always be in the range of quantized value,
    // [qmin, qmax].
    TFLITE_DCHECK_GE(zero_point, qmin);
    TFLITE_DCHECK_LE(zero_point, qmax);
  }

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          const float src_val = input_data[Offset(input_dims, c, x, y, b)];
          const float unclamped_quantized_val =
              TfLiteRound(zero_point + src_val / scale);
          const float quantized_val = std::min(
              qmax_float, std::max(qmin_float, unclamped_quantized_val));
          const float dst_val = scale * (quantized_val - zero_point);
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
                           const Dims<4>& output_dims) {
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
  if (output_height == 2 * input_height && output_width == 2 * input_width) {
    ResizeBilinear2x2(input_data, input_dims, output_data, output_dims, batches,
                      input_height, input_width, depth, output_height,
                      output_width);
  } else {
    float height_scale = static_cast<float>(input_height) / output_height;
    float width_scale = static_cast<float>(input_width) / output_width;

    ResizeBilinearGeneric(input_data, input_dims, output_data, output_dims,
                          batches, input_height, input_width, depth,
                          output_height, output_width, height_scale,
                          width_scale);
  }
}

template <typename T>
inline void SpaceToBatchND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims,
                           const int32* paddings_data,
                           const Dims<4>& paddings_dims, T* output_data,
                           const Dims<4>& output_dims) {
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

template <typename T>
inline void BatchToSpaceND(const T* input_data, const Dims<4>& input_dims,
                           const int32* block_shape_data,
                           const Dims<4>& block_shape_dims, T* output_data,
                           const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("BatchToSpaceND");

  const int output_batch_size = ArraySize(output_dims, 3);
  const int input_batch_size = ArraySize(input_dims, 3);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int depth = ArraySize(input_dims, 0);
  const int block_shape_width = block_shape_data[1];
  const int block_shape_height = block_shape_data[0];

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      for (int in_w = 0; in_w < input_width; ++in_w) {
        int out_batch = in_batch % output_batch_size;
        int out_w = in_w * block_shape_width +
                    (in_batch / output_batch_size) % block_shape_width;
        int out_h = in_h * block_shape_height +
                    (in_batch / output_batch_size) / block_shape_width;
        T* out = output_data + Offset(output_dims, 0, out_w, out_h, out_batch);
        const T* in = input_data + Offset(input_dims, 0, in_w, in_h, in_batch);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

template <typename T>
inline void Pad(const T* input_data, const Dims<4>& input_dims,
                const std::vector<int>& left_paddings,
                const std::vector<int>& right_paddings, T* output_data,
                const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("Pad");
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
    memset(output_data, 0,
           left_b_padding * output_height * output_width * output_depth *
               sizeof(T));
  }
  for (int out_b = left_b_padding; out_b < output_batch - right_b_padding;
       ++out_b) {
    if (left_h_padding != 0) {
      memset(output_data + Offset(output_dims, 0, 0, 0, out_b), 0,
             left_h_padding * output_width * output_depth * sizeof(T));
    }
    for (int out_h = left_h_padding; out_h < output_height - right_h_padding;
         ++out_h) {
      if (left_w_padding != 0) {
        memset(output_data + Offset(output_dims, 0, 0, out_h, out_b), 0,
               left_w_padding * output_depth * sizeof(T));
      }
      for (int out_w = left_w_padding; out_w < output_width - right_w_padding;
           ++out_w) {
        if (left_d_padding != 0) {
          memset(output_data + Offset(output_dims, 0, out_w, out_h, out_b), 0,
                 left_d_padding * sizeof(T));
        }

        T* out = output_data +
                 Offset(output_dims, left_d_padding, out_w, out_h, out_b);
        const T* in =
            input_data + Offset(input_dims, 0, out_w - left_w_padding,
                                out_h - left_h_padding, out_b - left_b_padding);
        memcpy(out, in, input_depth * sizeof(T));

        if (right_d_padding != 0) {
          memset(
              output_data + Offset(output_dims, output_depth - right_d_padding,
                                   out_w, out_h, out_b),
              0, right_d_padding * sizeof(T));
        }
      }
      if (right_w_padding != 0) {
        memset(
            output_data + Offset(output_dims, 0, output_width - right_w_padding,
                                 out_h, out_b),
            0, right_w_padding * output_depth * sizeof(T));
      }
    }
    if (right_h_padding != 0) {
      memset(output_data + Offset(output_dims, 0, 0,
                                  output_height - right_h_padding, out_b),
             0, right_h_padding * output_width * output_depth * sizeof(T));
    }
  }
  if (right_b_padding != 0) {
    memset(output_data +
               Offset(output_dims, 0, 0, 0, output_batch - right_b_padding),
           0,
           right_b_padding * output_height * output_width * output_depth *
               sizeof(T));
  }
}

template <typename T>
inline void StridedSlice(const T* input_data, const Dims<4>& input_dims,
                         int begin_mask, int end_mask,
                         const std::vector<int>& starts,
                         const std::vector<int>& stops,
                         const std::vector<int>& strides, T* output_data,
                         const Dims<4>& output_dims) {
  gemmlowp::ScopedProfilingLabel label("StridedSlice");
  const int start_b = (begin_mask & 8) ? 0 : starts[3];
  const int stop_b = (end_mask & 8) ? input_dims.sizes[3] : stops[3];
  const int start_h = (begin_mask & 4) ? 0 : starts[2];
  const int stop_h = (end_mask & 4) ? input_dims.sizes[2] : stops[2];
  const int start_w = (begin_mask & 2) ? 0 : starts[1];
  const int stop_w = (end_mask & 2) ? input_dims.sizes[1] : stops[1];
  const int start_d = (begin_mask & 1) ? 0 : starts[0];
  const int stop_d = (end_mask & 1) ? input_dims.sizes[0] : stops[0];

  T* out_ptr = output_data;
  if (strides[0] == 0) {
    for (int in_b = start_b; in_b < stop_b; in_b += strides[3]) {
      for (int in_h = start_h; in_h < stop_h; in_h += strides[2]) {
        for (int in_w = start_w; in_w < stop_w; in_w += strides[1]) {
          const int len = stop_d - start_d;
          memcpy(out_ptr,
                 input_data + Offset(input_dims, start_d, in_w, in_h, in_b),
                 len * sizeof(T));
          out_ptr += len;
        }
      }
    }
  } else {
    for (int in_b = start_b; in_b < stop_b; in_b += strides[3]) {
      for (int in_h = start_h; in_h < stop_h; in_h += strides[2]) {
        for (int in_w = start_w; in_w < stop_w; in_w += strides[1]) {
          for (int in_d = start_d; in_d < stop_d; in_d += strides[0]) {
            *out_ptr++ = input_data[Offset(input_dims, in_d, in_w, in_h, in_b)];
          }
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
      size[2] == -1 ? input_dims.sizes[2] - start_b : start_b + size[2];
  const int start_w = begin[1];
  const int stop_w =
      size[1] == -1 ? input_dims.sizes[1] - start_b : start_b + size[1];
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

}  // namespace optimized_ops
}  // namespace tflite

#if defined OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#undef OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#pragma GCC diagnostic pop
#endif

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_OPS_H_
