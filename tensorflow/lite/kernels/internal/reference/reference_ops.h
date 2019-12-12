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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_

#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>

#include "third_party/eigen3/Eigen/Core"
#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/arg_min_max.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/ceil.h"
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/floor.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/neg.h"
#include "tensorflow/lite/kernels/internal/reference/pad.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/reference/round.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void DepthToSpace(const tflite::DepthToSpaceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_batch = input_shape.Dims(0);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch = output_shape.Dims(0);

  const int32 block_size = op_params.block_size;

  TFLITE_DCHECK_EQ(input_width * block_size, output_width);
  TFLITE_DCHECK_EQ(input_height * block_size, output_height);
  TFLITE_DCHECK_EQ(input_depth, output_depth * block_size * block_size);
  TFLITE_DCHECK_EQ(input_batch, output_batch);

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        for (int out_d = 0; out_d < output_depth; ++out_d) {
          const int in_d =
              out_d + ((out_h % block_size) * block_size + out_w % block_size) *
                          output_depth;

          const int in_w = out_w / block_size;
          const int in_h = out_h / block_size;
          const int in_b = out_b;

          const int input_index = Offset(input_shape, in_b, in_h, in_w, in_d);
          const int output_index =
              Offset(output_shape, out_b, out_h, out_w, out_d);

          output_data[output_index] = input_data[input_index];
        }
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
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_batch = input_shape.Dims(0);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch = output_shape.Dims(0);

  const int32 block_size = op_params.block_size;

  TFLITE_DCHECK_EQ(input_width, output_width * block_size);
  TFLITE_DCHECK_EQ(input_height, output_height * block_size);
  TFLITE_DCHECK_EQ(input_depth * block_size * block_size, output_depth);
  TFLITE_DCHECK_EQ(input_batch, output_batch);

  for (int in_b = 0; in_b < input_batch; ++in_b) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      for (int in_w = 0; in_w < input_width; ++in_w) {
        for (int in_d = 0; in_d < input_depth; ++in_d) {
          const int out_d =
              in_d + ((in_h % block_size) * block_size + in_w % block_size) *
                         input_depth;
          const int out_w = in_w / block_size;
          const int out_h = in_h / block_size;
          const int out_b = in_b;

          const int input_index = Offset(input_shape, in_b, in_h, in_w, in_d);
          const int output_index =
              Offset(output_shape, out_b, out_h, out_w, out_d);

          output_data[output_index] = input_data[input_index];
        }
      }
    }
  }
}

inline void Elu(const RuntimeShape& input_shape, const float* input_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    output_data[i] = val < 0.0 ? std::exp(val) - 1 : val;
  }
}

template <typename T>
inline void Relu(const RuntimeShape& input_shape, const T* input_data,
                 const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T lower = 0;
    const T clamped = val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename T>
inline void Relu1(const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Relu1 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T upper = 1;
    const T lower = -1;
    const T clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

inline void Relu6(const RuntimeShape& input_shape, const float* input_data,
                  const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("Relu6 (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    const float upper = 6;
    const float lower = 0;
    const float clamped = val > upper ? upper : val < lower ? lower : val;
    output_data[i] = clamped;
  }
}

template <typename T>
inline void ReluX(const tflite::ReluParams& params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Quantized ReluX (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const int32 val = static_cast<int32_t>(input_data[i]);
    int32 clamped = params.output_offset +
                    MultiplyByQuantizedMultiplier(val - params.input_offset,
                                                  params.output_multiplier,
                                                  params.output_shift);
    clamped = std::max(params.quantized_activation_min, clamped);
    clamped = std::min(params.quantized_activation_max, clamped);
    output_data[i] = static_cast<T>(clamped);
  }
}

template <typename T>
inline void ReluX(const tflite::ActivationParams& params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Quantized ReluX (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const T max_value = params.quantized_activation_max;
  const T min_value = params.quantized_activation_min;
  for (int i = 0; i < flat_size; ++i) {
    const T val = input_data[i];
    const T clamped =
        val > max_value ? max_value : val < min_value ? min_value : val;
    output_data[i] = clamped;
  }
}

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("LeakyRelu (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    output_data[i] = val > 0 ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params, T q_alpha,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  ruy::profiler::ScopeLabel label("LeakyRelu (not fused)");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32 quantized_min = std::numeric_limits<T>::min();
  static const int32 quantized_max = std::numeric_limits<T>::max();
  static const int32 alpha_value = q_alpha - params.alpha_offset;
  for (int i = 0; i < flat_size; ++i) {
    const int32 input_value = input_data[i] - params.input_offset;
    if (input_value >= 0) {
      output_data[i] = input_data[i];
    } else {
      const int32 unclamped_output =
          params.output_offset + MultiplyByQuantizedMultiplierSmallerThanOneExp(
                                     input_value * alpha_value,
                                     params.output_multiplier,
                                     params.output_shift);
      const T clamped_output =
          std::min(quantized_max, std::max(quantized_min, unclamped_output));
      output_data[i] = static_cast<uint8>(clamped_output);
    }
  }
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const float* input_data,
                            const RuntimeShape& output_shape,
                            float* output_data, float epsilon = 1e-6) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i) {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c) {
      const float val = input_data[depth * i + c];
      squared_l2_norm += val * val;
    }
    float l2_norm = std::sqrt(squared_l2_norm);
    l2_norm = std::max(l2_norm, epsilon);
    for (int c = 0; c < depth; ++c) {
      output_data[depth * i + c] = input_data[depth * i + c] / l2_norm;
    }
  }
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const uint8* input_data,
                            const RuntimeShape& output_shape,
                            uint8* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int32 input_zero_point = op_params.input_zero_point;
  for (int i = 0; i < outer_size; ++i) {
    int32 square_l2_norm = 0;
    for (int c = 0; c < depth; c++) {
      int32 diff = input_data[depth * i + c] - input_zero_point;
      square_l2_norm += diff * diff;
    }
    int32 inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(square_l2_norm, kReverseShift,
                                     &inv_l2norm_multiplier, &inv_l2norm_shift);
    for (int c = 0; c < depth; c++) {
      int32 diff = input_data[depth * i + c] - input_zero_point;
      int32 rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
      int32 unclamped_output_val = 128 + rescaled_diff;
      int32 output_val = std::min(255, std::max(0, unclamped_output_val));
      output_data[depth * i + c] = static_cast<uint8>(output_val);
    }
  }
}

// T is expected to be either float or int.
template <typename T>
inline void AddN(const RuntimeShape& input_shape, const size_t num_inputs,
                 T* const* input_data, T* output_data) {
  // All inputs and output should have the same shape, this is checked during
  // Prepare stage.
  const size_t size = input_shape.FlatSize();
  for (int i = 0; i < size; ++i) {
    T x = 0;
    for (int j = 0; j < num_inputs; ++j) {
      x += input_data[j][i];
    }
    output_data[i] = x;
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
inline void BroadcastMulFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input2_offset = unswitched_params.input1_offset;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const uint8* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const uint8* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise Mul of
  // sections of the arrays.
  uint8* output_data_ptr = output_data;
  const uint8* input1_data_ptr = input1_data;
  const uint8* input2_data_reset = input2_data;
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  for (int i0 = 0; i0 < y0; ++i0) {
    const uint8* input2_data_ptr;
    for (int i1 = 0; i1 < y1; ++i1) {
      input2_data_ptr = input2_data_reset;
      for (int i2 = 0; i2 < y2; ++i2) {
        for (int i3 = 0; i3 < y3; ++i3) {
          MulElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                         output_data_ptr);
          input2_data_ptr += y4;
          output_data_ptr += y4;
        }
        input1_data_ptr += y4;
      }
    }
    input2_data_reset = input2_data_ptr;
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16");

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
  int32 output_offset = params.output_offset;
  int32 output_activation_min = params.quantized_activation_min;
  int32 output_activation_max = params.quantized_activation_max;
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

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastDiv4DSlow(const ArithmeticParams& params,
                        const RuntimeShape& unextended_input1_shape,
                        const T* input1_data,
                        const RuntimeShape& unextended_input2_shape,
                        const T* input2_data,
                        const RuntimeShape& unextended_output_shape,
                        T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          output_data[Offset(output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] /
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

template <typename T>
inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] / input2_data[i], output_activation_min,
        output_activation_max);
  }
}

// Element-wise div that can often be used for inner loop of broadcast Div as
// well as the non-broadcast Div.
inline void DivElementwise(int size, const ArithmeticParams& params,
                           const uint8* input1_data, const uint8* input2_data,
                           uint8* output_data) {
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);

  for (int i = 0; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    TFLITE_DCHECK_NE(input2_val, 0);
    int recip_shift;
    const int32 input2_inv =
        (input2_val > 0) ? GetReciprocal(input2_val, 31, &recip_shift)
                         : -GetReciprocal(-input2_val, 31, &recip_shift);
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
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8* input1_data,
                const RuntimeShape& input2_shape, const uint8* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("Div/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastDiv4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& unextended_input1_shape,
                               const uint8* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const uint8* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               uint8* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          const int32 input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          const int32 input2_val =
              params.input2_offset +
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          TFLITE_DCHECK_NE(input2_val, 0);
          int recip_shift;
          const int32 input2_inv =
              (input2_val > 0) ? GetReciprocal(input2_val, 31, &recip_shift)
                               : -GetReciprocal(-input2_val, 31, &recip_shift);
          const int headroom = CountLeadingSignBits(input1_val);
          const int32 unscaled_quotient =
              MultiplyByQuantizedMultiplierGreaterThanOne(input1_val,
                                                          input2_inv, headroom);
          const int total_shift = params.output_shift - recip_shift - headroom;
          const int32 unclamped_result =
              params.output_offset +
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  unscaled_quotient, params.output_multiplier, total_shift);
          const int32 clamped_output = std::min(
              params.quantized_activation_max,
              std::max(params.quantized_activation_min, unclamped_result));
          output_data[Offset(output_shape, b, y, x, c)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const float* input1_data,
                            const RuntimeShape& input2_shape,
                            const float* input2_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const int32* input1_data,
                            const RuntimeShape& input2_shape,
                            const int32* input2_data,
                            const RuntimeShape& output_shape,
                            int32* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.quantized_activation_min,
        params.quantized_activation_max);
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
inline void BroadcastSub4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const float* input1_data,
                               const RuntimeShape& input2_shape,
                               const float* input2_data,
                               const RuntimeShape& output_shape,
                               float* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub4DSlow/float");
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

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
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  params.float_activation_min, params.float_activation_max);
        }
      }
    }
  }
}

inline void BroadcastSub4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const uint8* input1_data,
                               const RuntimeShape& input2_shape,
                               const uint8* input2_data,
                               const RuntimeShape& output_shape,
                               uint8* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub4DSlow/uint8");
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

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
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int32 input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          const int32 input2_val =
              params.input2_offset +
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          const int32 shifted_input1_val =
              input1_val * (1 << params.left_shift);
          const int32 shifted_input2_val =
              input2_val * (1 << params.left_shift);
          const int32 scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, params.input1_multiplier,
                  params.input1_shift);
          const int32 scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, params.input2_multiplier,
                  params.input2_shift);
          const int32 raw_sub = scaled_input1_val - scaled_input2_val;
          const int32 raw_output =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  raw_sub, params.output_multiplier, params.output_shift) +
              params.output_offset;
          const int32 clamped_output =
              std::min(params.quantized_activation_max,
                       std::max(params.quantized_activation_min, raw_output));
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

inline void BroadcastSub4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const int32* input1_data,
                               const RuntimeShape& input2_shape,
                               const int32* input2_data,
                               const RuntimeShape& output_shape,
                               int32* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub4DSlow/int32");
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

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
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  params.quantized_activation_min,
                  params.quantized_activation_max);
        }
      }
    }
  }
}

template <typename T>
void BroadcastSub4DSlow(const ArithmeticParams& params,
                        const RuntimeShape& input1_shape, const T* input1_data,
                        const RuntimeShape& input2_shape, const T* input2_data,
                        const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub4DSlow/templated");
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

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
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  params.quantized_activation_min,
                  params.quantized_activation_max);
        }
      }
    }
  }
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

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
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
        }
      }
    }
  }
}

inline void SubWithActivation(const ArithmeticParams& params,
                              const RuntimeShape& input1_shape,
                              const int32* input1_data,
                              const RuntimeShape& input2_shape,
                              const int32* input2_data,
                              const RuntimeShape& output_shape,
                              int32* output_data) {
  ruy::profiler::ScopeLabel label("SubWithActivation");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.quantized_activation_min,
        params.quantized_activation_max);
  }
}

inline void SubWithActivation(const ArithmeticParams& params,
                              const RuntimeShape& input1_shape,
                              const float* input1_data,
                              const RuntimeShape& input2_shape,
                              const float* input2_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

inline void Sub16(const ArithmeticParams& params,
                  const RuntimeShape& input1_shape, const int16_t* input1_data,
                  const RuntimeShape& input2_shape, const int16_t* input2_data,
                  const RuntimeShape& output_shape, int16_t* output_data) {
  ruy::profiler::ScopeLabel label("Sub/Int16");
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

  if (input1_shift == 0) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    for (int i = 0; i < flat_size; ++i) {
      F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
      F0 scaled_input = F0::FromRaw(
          gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
      F0 result = SaturatingSub(input_ready_scaled, scaled_input);
      const int16 raw_output = result.raw();
      const int16 clamped_output = std::min(
          output_activation_max, std::max(output_activation_min, raw_output));
      output_data[i] = clamped_output;
    }
  } else {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    for (int i = 0; i < flat_size; ++i) {
      F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
      F0 scaled_input = F0::FromRaw(
          gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
      F0 result = SaturatingSub(scaled_input, input_ready_scaled);
      const int16 raw_output = result.raw();
      const int16 clamped_output = std::min(
          output_activation_max, std::max(output_activation_min, raw_output));
      output_data[i] = clamped_output;
    }
  }
}

template <typename Scalar>
void Pack(const PackParams& params, const RuntimeShape* const* input_shapes,
          const Scalar* const* input_data, const RuntimeShape& output_shape,
          Scalar* output_data) {
  ruy::profiler::ScopeLabel label("Pack");
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  int inputs_count = params.inputs_count;

  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = params.axis + 1; i < dimensions; i++) {
    copy_size *= output_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ((**input_shapes).FlatSize(), copy_size * outer_size);

  for (int i = 0; i < inputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      const Scalar* input_ptr = input_data[i] + copy_size * k;
      int loc = k * inputs_count * copy_size + i * copy_size;
      memcpy(output_data + loc, input_ptr, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar>
void Unpack(const UnpackParams& params, const RuntimeShape& input_shape,
            const Scalar* input_data, const RuntimeShape& output_shape,
            Scalar* const* output_datas) {
  ruy::profiler::ScopeLabel label("Unpack");
  const int dimensions = input_shape.DimensionsCount();
  const int outputs_count = params.num_split;

  int outer_size = 1;
  int axis = params.axis;
  if (axis < 0) {
    axis += dimensions;
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, dimensions);
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= input_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ(output_shape.FlatSize(), copy_size * outer_size);

  for (int i = 0; i < outputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      Scalar* output_ptr = output_datas[i] + copy_size * k;
      int loc = k * outputs_count * copy_size + i * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar>
void PackWithScaling(const PackParams& params,
                     const RuntimeShape* const* input_shapes,
                     const uint8* const* input_data,
                     const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("PackWithScaling");
  const int dimensions = output_shape.DimensionsCount();
  int axis = params.axis;
  const int32* input_zeropoint = params.input_zeropoint;
  const float* input_scale = params.input_scale;
  int inputs_count = params.inputs_count;
  const int32 output_zeropoint = params.output_zeropoint;
  const float output_scale = params.output_scale;

  int outer_size = 1;
  for (int i = 0; i < axis; i++) {
    outer_size *= output_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; i++) {
    copy_size *= output_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ((**input_shapes).FlatSize(), copy_size * outer_size);

  Scalar* output_ptr = output_data;
  const float inverse_output_scale = 1.f / output_scale;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < inputs_count; ++i) {
      if (input_zeropoint[i] == output_zeropoint &&
          input_scale[i] == output_scale) {
        memcpy(output_ptr, input_data[i] + k * copy_size,
               copy_size * sizeof(Scalar));
      } else {
        assert(false);
        const float scale = input_scale[i] * inverse_output_scale;
        const float bias = -input_zeropoint[i] * scale;
        auto input_ptr = input_data[i];
        for (int j = 0; j < copy_size; ++j) {
          const int32_t value =
              static_cast<int32_t>(std::round(input_ptr[j] * scale + bias)) +
              output_zeropoint;
          output_ptr[j] =
              static_cast<uint8_t>(std::max(std::min(255, value), 0));
        }
      }
      output_ptr += copy_size;
    }
  }
}

template <typename Scalar>
void DepthConcatenation(const ConcatenationParams& params,
                        const RuntimeShape* const* input_shapes,
                        const Scalar* const* input_data,
                        const RuntimeShape& output_shape, Scalar* output_data) {
  ruy::profiler::ScopeLabel label("DepthConcatenation");
  auto params_copy = params;
  params_copy.axis = 3;
  Concatenation(params_copy, input_shapes, input_data, output_shape,
                output_data);
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
    const RuntimeShape& unextended_activ_temp_shape, float* activ_temp_data) {
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
  const int batches =
      MatchingDim(input_shape, 0, prev_activ_shape, 0, prev_state_shape, 0,
                  output_state_shape, 0, output_activ_shape, 0);
  const int height =
      MatchingDim(input_shape, 1, prev_activ_shape, 1, prev_state_shape, 1,
                  output_state_shape, 1, output_activ_shape, 1);
  const int width =
      MatchingDim(input_shape, 2, prev_activ_shape, 2, prev_state_shape, 2,
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
  FullyConnected(fc_params, concat_temp_shape, concat_temp_data, weights_shape,
                 weights_data, bias_shape, bias_data, activ_temp_shape,
                 activ_temp_data);

  // Memory state update (the LSTM "guts")
  for (int b = 0; b < batches; ++b) {
    for (int w = 0; w < width; ++w) {
      for (int h = 0; h < height; ++h) {
        for (int c = 0; c < output_depth; ++c) {
          const float input_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(activ_temp_shape, b, h, w,
                                                      0 * output_depth + c)]));
          const float new_input = std::tanh(activ_temp_data[Offset(
              activ_temp_shape, b, h, w, 1 * output_depth + c)]);
          const float forget_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(activ_temp_shape, b, h, w,
                                                      2 * output_depth + c)]));
          const float output_gate =
              1.f /
              (1.f + std::exp(-activ_temp_data[Offset(activ_temp_shape, b, h, w,
                                                      3 * output_depth + c)]));
          const float new_state =
              input_gate * new_input +
              forget_gate *
                  prev_state_data[Offset(prev_state_shape, b, h, w, c)];
          output_state_data[Offset(output_state_shape, b, h, w, c)] = new_state;
          output_activ_data[Offset(output_activ_shape, b, h, w, c)] =
              output_gate * std::tanh(new_state);
        }
      }
    }
  }
}

// Quantized LSTM cell implementation.
// The quantization of the input, output arrays is as follows:
//  - The input activations are quantized as uint8 on the interval
//    [-1, 127/128].
//    The rationale for that is that is the natural interval for output
//    activations (see next point) and these need to be concatenated together.
//    We could accommodate different ranges by re-scaling, but we empirically
//    found that setting the input activations range to be [-1, 127/128] in the
//    first place, removing the need for re-scaling, greatly improves accuracy.
//  - The output activations are quantized as uint8 on the interval
//    [-1, 127/128].
//    The rationale for that is that the definition of a LSTM cell makes them
//    intrinsically constrained in [-1, 1]; tweaking that to [-1, 127/128]
//    makes for simpler, more accurate fixed-point arithmetic.
//  - The output-at-previous-timestep state array is obviously quantized as
//    the output activations.
//  - The internal LSTM memory (not the output-at-previous-timestep, the other
//    internal state array) is int16-quantized and may use any power-of-two,
//    symmetric range i.e. [-2^N, 2^N * 32767/32768] for any N, which we call
//    StateIntegerBits below, see the below discussion of that template
//    parameter ("The StateIntegerBits template parameter").
//  - The output of the internal fully-connected node is int16-quantized
//    on the interval [-8, 8 * 32767/32768], the rationale for which is
//    explained just below ("Why [-8, 8] for fully-connected output?").
//
//
// === The StateIntegerBits template parameter ===
//
// The StateIntegerBits template parameter controls the fixed-point format used
// to represent the internal memory of the LSTM cell (not the
// output-at-previous-timestep, the other internal state array). It's currently
// a template parameter so that the model can control that. The most typical
// value for StateIntegerBits is 4. Other plausible values are anywhere between
// 3 and 5. We might eventually standardize on a single supported value, e.g. 4,
// and drop that template parameter. The reason why it can't be a runtime
// parameter is that this controls the fixed-point format used, i.e. we need to
// generate actually different code based on it. In particular, we generate code
// for a fixed-point tanh() implementation for that format, which internally
// uses a fixed-point exp() implementation, which internally uses a
// barrel-shifter with a number of steps that depends on StateIntegerBits.
// Another consequence of that is that a higher value of StateIntegerBits
// results in a more expensive implementation (more barrel shifter steps
// needed).
//
//
// === Why [-8, 8] for fully-connected output? ===
//
// This array is only fed to Logistic and Tanh functions, for which
// the quantized implementation will want to use fixed-point arithmetic,
// requiring a power-of-two representation interval. Thus, we should right
// away quantize this array to a power-of-two interval; otherwise,
// implementation will need to rescale that, losing any benefit that a tighter
// representation interval might otherwise yield, while introducing some
// numerical error and computational overhead.
//
// Now, Logistic and Tanh
// are nearly constant (nearly equal to their horizontal asymptotes)
// outside of a small bounded interval around 0:
//
//   Logistic(4) = 1 - 1.8e-2     Tanh(4) = 1 - 6.7e-4
//   Logistic(8) = 1 - 3.4e-4     Tanh(8) = 1 - 2.3e-7
//   Logistic(16) = 1 - 1.1e-7    Tanh(16) = 1 - 2.5e-14
//
// From this, we see that clamping to [-4, 4] would be too inaccurate
// (the error of 1.8e-2 on Logistic would be felt even in 8bit precision)
// while clamping to [-16, 16] would make no difference even in float32.
// However, for a fixed-point implementation in 16-bit integers, using 5
// integer bits to represent the [-16, 16] range would leave only 11
// fractional bits, giving an increment of 2^-11 = 4.9e-4 between consecutive
// representable values. Notice that is higher than the
// worst-case clamping error with clamping to [-8, 8]: 3.4e-4 for Logistic.
// Using [-8, 8] thus seems like the better compromise overall, enjoying
// an increment of 2.4e-4 between representable values and a worst-case
// clamping error of 3.4e-4, both better than the increment of 4.9e-4 with
// [-16, 16].
//
// Moreover, all other things being equal, it is nice to choose the narrower
// representation range, as that makes the implementation of fixed-point
// math functions a little cheaper (each integer bit requires an additional
// barrel-shifter atep in the implementation of exp(-x)). That is further
// reason to prefer [-8, 8] over [-16, 16]. The choice of [-16, 16] would make
// sense for 32-bit float or 32-bit fixed-point quantization, but we are
// aiming for 16-bit fixed-point quantization of these internal nodes here.
//
template <int StateIntegerBits>
inline void LstmCell(const LstmCellParams& params,
                     const RuntimeShape& unextended_input_shape,
                     const uint8* input_data_uint8,
                     const RuntimeShape& unextended_prev_activ_shape,
                     const uint8* prev_activ_data_uint8,
                     const RuntimeShape& weights_shape,
                     const uint8* weights_data_uint8,
                     const RuntimeShape& unextended_bias_shape,
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
                     int16* activ_temp_data_int16, void* gemmlowp_context) {
  (void)gemmlowp_context;  // only used in optimized code.
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
  for (int b = 0; b < fc_batches; ++b) {
    for (int out_c = 0; out_c < fc_output_depth; ++out_c) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum = bias_data_int32[out_c];
      // Accumulation loop.
      for (int d = 0; d < fc_accum_depth; ++d) {
        int16 input_val = concat_temp_data_uint8[b * fc_accum_depth + d] - 128;
        int16 weights_val =
            weights_data_uint8[out_c * fc_accum_depth + d] - weights_zero_point;
        accum += input_val * weights_val;
      }
      // Down-scale the final int32 accumulator to the scale used by our
      // (16-bit, using 3 integer bits) fixed-point format. The quantized
      // multiplier and shift here have been pre-computed offline
      // (e.g. by toco).
      accum =
          MultiplyByQuantizedMultiplier(accum, accum_multiplier, accum_shift);
      // Saturate, cast to int16, and store to the temporary activations array.
      accum = std::max(-32768, std::min(32767, accum));
      activ_temp_data_int16[out_c + fc_output_depth * b] = accum;
    }
  }

  // Rest of the LSTM cell: tanh and logistic math functions, and some adds
  // and muls, all done in 16-bit fixed-point.
  for (int b = 0; b < outer_size; ++b) {
    for (int c = 0; c < output_depth; ++c) {
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
      F3 input_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 0 * output_depth + c]);
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 1 * output_depth + c]);
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 2 * output_depth + c]);
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(
          activ_temp_data_int16[b * fc_output_depth + 3 * output_depth + c]);
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(prev_state_data_int16[b * output_depth + c]);
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
      output_state_data_int16[b * output_depth + c] = new_state.raw();
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16 rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int16 clamped_output_activ =
          std::max<int16>(-128, std::min<int16>(127, rescaled_output_activ));
      output_activ_data_uint8[b * output_depth + c] =
          128 + clamped_output_activ;
    }
  }
}

template <typename Scalar>
void Split(const SplitParams& params, const RuntimeShape& input_shape,
           const Scalar* input_data, const RuntimeShape* const* output_shapes,
           Scalar* const* output_data) {
  ruy::profiler::ScopeLabel label("Split");
  const int split_dimensions = input_shape.DimensionsCount();
  int axis = params.axis < 0 ? params.axis + split_dimensions : params.axis;
  int outputs_count = params.num_split;
  TFLITE_DCHECK_LT(axis, split_dimensions);

  int64_t split_size = 0;
  for (int i = 0; i < outputs_count; i++) {
    TFLITE_DCHECK_EQ(output_shapes[i]->DimensionsCount(), split_dimensions);
    for (int j = 0; j < split_dimensions; j++) {
      if (j != axis) {
        MatchingDim(*output_shapes[i], j, input_shape, j);
      }
    }
    split_size += output_shapes[i]->Dims(axis);
  }
  TFLITE_DCHECK_EQ(split_size, input_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  // For all output arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_shape.Dims(i);
  }

  const Scalar* input_ptr = input_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < outputs_count; ++i) {
      const int copy_size = output_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_data[i] + k * copy_size, input_ptr,
             copy_size * sizeof(Scalar));
      input_ptr += copy_size;
    }
  }
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline void LocalResponseNormalization(
    const tflite::LocalResponseNormalizationParams& op_params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    for (int c = 0; c < depth; ++c) {
      const int begin_input_c = std::max(0, c - op_params.range);
      const int end_input_c = std::min(depth, c + op_params.range);
      float accum = 0.f;
      for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
        const float input_val = input_data[i * depth + input_c];
        accum += input_val * input_val;
      }
      const float multiplier =
          std::pow(op_params.bias + op_params.alpha * accum, -op_params.beta);
      output_data[i * depth + c] = input_data[i * depth + c] * multiplier;
    }
  }
}

inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const float* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp(input_data[i * depth + c] - max);
    }

    // Compute result.
    const float log_sum = std::log(sum);
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = input_data[i * depth + c] - max - log_sum;
    }
  }
}

inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const uint8* input_data,
                       const RuntimeShape& output_shape, uint8* output_data) {
  ruy::profiler::ScopeLabel label("LogSoftmax/8bit");
  const int32 input_multiplier = params.input_multiplier;
  const int32 input_left_shift = params.input_left_shift;
  const int32 reverse_scaling_divisor = params.reverse_scaling_divisor;
  const int32 reverse_scaling_right_shift = params.reverse_scaling_right_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large
  // as -32 before multiplying by input_beta_multiplier, and therefore as
  // large as -16 afterwards.  Note that exp(-8) is definitely not
  // insignificant to accumulation, but exp(-16) definitely is.
  static constexpr int kScaledDiffIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

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

    const int32 fixed_log_sum_of_exps =
        log_x_for_x_greater_than_or_equal_to_1<kScaledDiffIntegerBits>(
            sum_of_exps)
            .raw();

    // rescaled_diff_min is smallest representable in
    // Q(kScaledDiffIntegerBits).(31-kScaledDiffIntegerBits) plus the
    // log-sub-exps that will be subtracted in the loop.
    //
    // The thresholds diff_min, etc are negative.
    const int rescaled_diff_min =
        fixed_log_sum_of_exps + std::numeric_limits<int32>::lowest();
    const int adjusted_diff_min =
        std::max(diff_min - 1,  // Note use of > below instead of >= above.
                 MultiplyByQuantizedMultiplierSmallerThanOneExp(
                     rescaled_diff_min, reverse_scaling_divisor,
                     -reverse_scaling_right_shift));

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

inline void Tanh(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    float val = input_data[i];
    float result = std::tanh(val);
    output_data[i] = result;
  }
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
  const int input_left_shift = params.input_left_shift;
  // Support for shifts is limited until we have a parameterized version of
  // SaturatingRoundingMultiplyByPOT().
  TFLITE_DCHECK_GE(input_left_shift, 0);
  TFLITE_DCHECK_LE(input_left_shift, 1);

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // F0 uses 0 integer bits, range [-1, 1].
  // This is the return type of math functions such as tanh, logistic,
  // whose range is in [-1, 1].
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
  using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

  if (input_left_shift == 0) {
    for (int i = 0; i < flat_size; i++) {
      F3 input = F3::FromRaw(input_data[i]);
      F0 output = gemmlowp::tanh(input);
      output_data[i] = output.raw();
    }
  } else {
    for (int i = 0; i < flat_size; i++) {
      F3 input = F3::FromRaw(
          gemmlowp::SaturatingRoundingMultiplyByPOT<1>(input_data[i]));
      F0 output = gemmlowp::tanh(input);
      output_data[i] = output.raw();
    }
  }
}

inline void Dequantize(const RuntimeShape& input_shape,
                       const Eigen::half* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = Eigen::half_impl::half_to_float(input_data[i]);
  }
}

inline void FakeQuant(const tflite::FakeQuantParams& op_params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  ruy::profiler::ScopeLabel label("FakeQuant");
  float rmin = op_params.minmax.min;
  float rmax = op_params.minmax.max;
  int num_bits = op_params.num_bits;
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
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  FakeQuantizeArray(nudged_scale, nudged_min, nudged_max, input_data,
                    output_data, flat_size);
}

template <typename SrcT, typename DstT>
inline void Cast(const RuntimeShape& input_shape, const SrcT* input_data,
                 const RuntimeShape& output_shape, DstT* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    int offset = i;
    output_data[offset] = static_cast<DstT>(input_data[offset]);
  }
}

template <typename T>
T FloorMod(T input1, T input2) {
  struct FloatMod {
    float operator()(const float lhs, const float rhs) const {
      return std::fmod(lhs, rhs);
    }
  };
  using ModFunc = typename std::conditional<std::is_integral<T>::value,
                                            std::modulus<T>, FloatMod>::type;
  ModFunc mod_func;
  T trunc_mod = mod_func(input1, input2);
  return (trunc_mod != 0) && ((input2 < 0) != (trunc_mod < 0))
             ? (trunc_mod + input2)
             : trunc_mod;
}

template <typename T, typename CoordsT = int32>
inline void Gather(const tflite::GatherParams& op_params,
                   const RuntimeShape& input_shape, const T* input_data,
                   const RuntimeShape& coords_shape, const CoordsT* coords_data,
                   const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("Gather");
  int axis = op_params.axis;
  if (axis < 0) {
    axis += input_shape.DimensionsCount();
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, input_shape.DimensionsCount());
  const int axis_size = input_shape.Dims(axis);
  const int coords_count = coords_shape.FlatSize();

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int inner_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i) {
    inner_size *= input_shape.Dims(i);
  }

  for (int outer = 0; outer < outer_size; ++outer) {
    for (int i = 0; i < coords_count; ++i) {
      TFLITE_DCHECK_GE(coords_data[i], 0);
      TFLITE_DCHECK_LT(coords_data[i], axis_size);
      std::memcpy(
          output_data + (outer * coords_count + i) * inner_size,
          input_data + (outer * axis_size + coords_data[i]) * inner_size,
          sizeof(T) * inner_size);
    }
  }
}

template <typename ParamsT, typename IndicesT = int32>
inline void GatherNd(const RuntimeShape& params_shape,
                     const ParamsT* params_data,
                     const RuntimeShape& indices_shape,
                     const IndicesT* indices_data,
                     const RuntimeShape& output_shape, ParamsT* output_data) {
  ruy::profiler::ScopeLabel label("GatherNd");

  int n_slices = 1;
  int slice_size = 1;
  const int indices_dims = indices_shape.DimensionsCount();
  const int indices_nd = indices_shape.Dims(indices_dims - 1);
  const int params_dims = params_shape.DimensionsCount();
  for (int i = 0; i < indices_dims - 1; ++i) {
    n_slices *= indices_shape.Dims(i);
  }
  for (int i = indices_nd; i < params_dims; ++i) {
    slice_size *= params_shape.Dims(i);
  }

  int remain_flat_size = params_shape.FlatSize();
  std::vector<int> dims_to_count(indices_nd, 0);
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / params_shape.Dims(i);
    remain_flat_size = dims_to_count[i];
  }

  for (int i = 0; i < n_slices; ++i) {
    int from_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      from_pos += indices_data[i * indices_nd + j] * dims_to_count[j];
    }
    std::memcpy(output_data + i * slice_size, params_data + from_pos,
                sizeof(ParamsT) * slice_size);
  }
}

template <typename IndicesT, typename UpdatesT>
inline void ScatterNd(const RuntimeShape& indices_shape,
                      const IndicesT* indices_data,
                      const RuntimeShape& updates_shape,
                      const UpdatesT* updates_data,
                      const RuntimeShape& output_shape, UpdatesT* output_data) {
  ruy::profiler::ScopeLabel label("ScatterNd");

  int n_slices = 1;
  int slice_size = 1;
  const int outer_dims = indices_shape.DimensionsCount() - 1;
  const int indices_nd = indices_shape.Dims(outer_dims);
  const int updates_dims = updates_shape.DimensionsCount();
  for (int i = 0; i < outer_dims; ++i) {
    n_slices *= indices_shape.Dims(i);
  }
  for (int i = outer_dims; i < updates_dims; ++i) {
    slice_size *= updates_shape.Dims(i);
  }

  int output_flat_size = output_shape.FlatSize();
  int remain_flat_size = output_flat_size;
  std::vector<int> dims_to_count(indices_nd, 0);
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / output_shape.Dims(i);
    remain_flat_size = dims_to_count[i];
  }

  memset(output_data, 0, sizeof(UpdatesT) * output_flat_size);
  for (int i = 0; i < n_slices; ++i) {
    int to_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      IndicesT idx = indices_data[i * indices_nd + j];
      TFLITE_DCHECK(0 <= idx && idx < output_shape.Dims(j));
      to_pos += idx * dims_to_count[j];
    }
    for (int j = 0; j < slice_size; j++) {
      output_data[to_pos + j] += updates_data[i * slice_size + j];
    }
  }
}

inline void ComputeInterpolationValues(const float value, const float scale,
                                       const bool half_pixel_centers,
                                       int32 input_size, float* scaled_value,
                                       int32* lower_bound, int32* upper_bound) {
  if (half_pixel_centers) {
    *scaled_value = (value + 0.5f) * scale - 0.5f;
  } else {
    *scaled_value = value * scale;
  }
  float scaled_value_floor = std::floor(*scaled_value);
  *lower_bound =
      std::max(static_cast<int32>(scaled_value_floor), static_cast<int32>(0));
  *upper_bound =
      std::min(static_cast<int32>(std::ceil(*scaled_value)), input_size - 1);
}

template <typename T>
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const T* input_data,
                           const RuntimeShape& unextended_output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  // If half_pixel_centers is True, align_corners must be False.
  TFLITE_DCHECK(!op_params.half_pixel_centers || !op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_size_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_size_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.Dims(0), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(1), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(2), 1);
  TFLITE_DCHECK_EQ(output_size_shape.Dims(3), 2);
  int32 output_height = output_size_data[Offset(output_size_shape, 0, 0, 0, 0)];
  int32 output_width = output_size_data[Offset(output_size_shape, 0, 0, 0, 1)];

  float height_scale = static_cast<float>(input_height) / output_height;
  float width_scale = static_cast<float>(input_width) / output_width;
  if (op_params.align_corners && output_height > 1) {
    height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
  }
  if (op_params.align_corners && output_width > 1) {
    width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
  }

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y;
      int32 y0, y1;
      ComputeInterpolationValues(y, height_scale, op_params.half_pixel_centers,
                                 input_height, &input_y, &y0, &y1);
      for (int x = 0; x < output_width; ++x) {
        float input_x;
        int32 x0, x1;
        ComputeInterpolationValues(x, width_scale, op_params.half_pixel_centers,
                                   input_width, &input_x, &x0, &x1);
        for (int c = 0; c < depth; ++c) {
          T interpolation =
              static_cast<T>(input_data[Offset(input_shape, b, y0, x0, c)] *
                                 (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y1, x0, c)] *
                                 (input_y - y0) * (1 - (input_x - x0)) +
                             input_data[Offset(input_shape, b, y0, x1, c)] *
                                 (1 - (input_y - y0)) * (input_x - x0) +
                             input_data[Offset(input_shape, b, y1, x1, c)] *
                                 (input_y - y0) * (input_x - x0));
          output_data[Offset(output_shape, b, y, x, c)] = interpolation;
        }
      }
    }
  }
}

template <typename T>
inline void SpaceToBatchND(
    const SpaceToBatchParams& params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const int32* block_shape_data,
    const RuntimeShape& unextended_input3_shape, const int32* paddings_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("SpaceToBatchND");
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input1_shape =
      RuntimeShape::ExtendedShape(4, unextended_input1_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int block_shape_height = block_shape_data[0];
  const int block_shape_width = block_shape_data[1];
  const int padding_top = paddings_data[0];
  const int padding_left = paddings_data[2];

  // For uint8 quantized, the correct padding "zero value" is the output offset.
  const int32_t pad_value = params.output_offset;

  for (int out_b = 0; out_b < output_batch_size; ++out_b) {
    int input_batch = out_b % input_batch_size;
    int shift_w = (out_b / input_batch_size) % block_shape_width;
    int shift_h = (out_b / input_batch_size) / block_shape_width;
    for (int out_h = 0; out_h < output_height; ++out_h) {
      for (int out_w = 0; out_w < output_width; ++out_w) {
        T* out = output_data + Offset(output_shape, out_b, out_h, out_w, 0);
        if (out_h * block_shape_height + shift_h < padding_top ||
            out_h * block_shape_height + shift_h >=
                padding_top + input_height ||
            out_w * block_shape_width + shift_w < padding_left ||
            out_w * block_shape_width + shift_w >= padding_left + input_width) {
          // This may not execute correctly when pad_value != 0 and T != uint8.
          memset(out, pad_value, depth * sizeof(T));
        } else {
          const T* in =
              input1_data +
              Offset(input1_shape, input_batch,
                     (out_h * block_shape_height + shift_h) - padding_top,
                     (out_w * block_shape_width + shift_w) - padding_left, 0);
          memcpy(out, in, depth * sizeof(T));
        }
      }
    }
  }
}

template <typename T>
inline void BatchToSpaceND(
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const int32* block_shape_data,
    const RuntimeShape& unextended_input3_shape, const int32* crops_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BatchToSpaceND");
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input1_shape =
      RuntimeShape::ExtendedShape(4, unextended_input1_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int block_shape_width = block_shape_data[1];
  const int block_shape_height = block_shape_data[0];
  const int crops_top = crops_data[0];
  const int crops_left = crops_data[2];

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const int out_h = in_h * block_shape_height +
                        spatial_offset / block_shape_width - crops_top;
      if (out_h < 0 || out_h >= output_height) {
        continue;
      }
      for (int in_w = 0; in_w < input_width; ++in_w) {
        const int out_w = in_w * block_shape_width +
                          spatial_offset % block_shape_width - crops_left;

        if (out_w < 0 || out_w >= output_width) {
          continue;
        }
        T* out = output_data + Offset(output_shape, out_batch, out_h, out_w, 0);
        const T* in =
            input1_data + Offset(input1_shape, in_batch, in_h, in_w, 0);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape,
                  const RuntimeShape& output_shape,
                  SequentialTensorWriter<T>* writer) {
  const RuntimeShape ext_shape = RuntimeShape::ExtendedShape(4, input_shape);
  // TODO(dkalenichenko): This op only supports 4D tensors or smaller.
  TFLITE_DCHECK_LE(op_params.begin_count, 4);
  TFLITE_DCHECK_LE(op_params.size_count, 4);
  const int begin_count = op_params.begin_count;
  const int size_count = op_params.size_count;
  // We front-pad the begin and size vectors.
  const int start_b = 4 - begin_count > 0 ? 0 : op_params.begin[0];
  const int stop_b = (4 - size_count > 0 || op_params.size[0] == -1)
                         ? ext_shape.Dims(0)
                         : start_b + op_params.size[0];
  const int start_h = begin_count < 3 ? 0 : op_params.begin[begin_count - 3];
  const int stop_h = (size_count < 3 || op_params.size[size_count - 3] == -1)
                         ? ext_shape.Dims(1)
                         : start_h + op_params.size[size_count - 3];
  const int start_w = begin_count < 2 ? 0 : op_params.begin[begin_count - 2];
  const int stop_w = (size_count < 2 || op_params.size[size_count - 2] == -1)
                         ? ext_shape.Dims(2)
                         : start_w + op_params.size[size_count - 2];
  const int start_d = begin_count < 1 ? 0 : op_params.begin[begin_count - 1];
  const int stop_d = (size_count < 1 || op_params.size[size_count - 1] == -1)
                         ? ext_shape.Dims(3)
                         : start_d + op_params.size[size_count - 1];

  for (int in_b = start_b; in_b < stop_b; ++in_b) {
    for (int in_h = start_h; in_h < stop_h; ++in_h) {
      for (int in_w = start_w; in_w < stop_w; ++in_w) {
        for (int in_d = start_d; in_d < stop_d; ++in_d) {
          writer->Write(Offset(ext_shape, in_b, in_h, in_w, in_d));
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

template <typename T>
inline void Exp(const T* input_data, const size_t num_elements,
                T* output_data) {
  ruy::profiler::ScopeLabel label("Exp");
  for (size_t idx = 0; idx < num_elements; ++idx) {
    output_data[idx] = std::exp(input_data[idx]);
  }
}

template <typename T>
void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto min_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] > min_value ? min_value : input1_data[i];
  }
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
  const int flat_size = MatchingFlatSize(input1_shape, output_shape);

  auto max_value = input2_data[0];
  for (int i = 0; i < flat_size; i++) {
    output_data[i] = input1_data[i] < max_value ? max_value : input1_data[i];
  }
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

template <typename T1, typename T2, typename T3>
void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
            const T3* input2_data, const RuntimeShape& output_shape,
            T2* output_data) {
  ArgMinMax(input1_shape, input1_data, input2_data, output_shape, output_data,
            std::greater<T1>());
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T1, typename T2, typename T3>
inline void ArgMax(const RuntimeShape& input1_shape, const T1* input1_data,
                   const RuntimeShape& input2_shape, const T3* input2_data,
                   const RuntimeShape& output_shape, T2* output_data) {
  // Drop shape of second input: not needed.
  ArgMax(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void TransposeImpl(const TransposeParams& params,
                   const RuntimeShape& unextended_input_shape,
                   const T* input_data,
                   const RuntimeShape& unextended_output_shape,
                   T* output_data) {
  const int unextended_output_size = unextended_output_shape.DimensionsCount();
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_size, 4);
  TFLITE_DCHECK_EQ(unextended_output_size, params.perm_count);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int input_ext_size = 4 - unextended_input_shape.DimensionsCount();
  const int output_ext_size = 4 - unextended_output_size;

  // The perm data is extended to match the output, each index incremented by
  // the amount of front padding of the input shape.
  int extended_perm[4];
  for (int i = 0; i < output_ext_size; ++i) {
    extended_perm[i] = i;
  }
  for (int i = 0; i < unextended_output_size; ++i) {
    extended_perm[i + output_ext_size] = params.perm[i] + input_ext_size;
  }

  int out_sizes[4];
  // Compute the inverse permutation array so we can do an output centered
  // transpose. Also, check to make sure output_dims is matching input_dims.
  for (int k = 0; k < 4; k++) {
    out_sizes[k] = MatchingDim(input_shape, extended_perm[k], output_shape, k);
  }

  // Naive transpose loop (iterate on output index and compute input index).
  int o[4];  // loop index (on output).
  int i[4];
  for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
    i[extended_perm[3]] = o[3];
    for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
      i[extended_perm[2]] = o[2];
      for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
        i[extended_perm[1]] = o[1];
        for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
          i[extended_perm[0]] = o[0];
          output_data[Offset(output_shape, o)] =
              input_data[Offset(input_shape, i)];
        }
      }
    }
  }
}

template <typename T>
void Transpose(const TransposeParams& params,
               const RuntimeShape& unextended_input_shape, const T* input_data,
               const RuntimeShape& unextended_output_shape, T* output_data) {
  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (sizeof(T)) {
    case 1:
      TransposeImpl<int8_t>(params, unextended_input_shape,
                            reinterpret_cast<const int8_t*>(input_data),
                            unextended_output_shape,
                            reinterpret_cast<int8_t*>(output_data));
      break;
    case 2:
      TransposeImpl<int16_t>(params, unextended_input_shape,
                             reinterpret_cast<const int16_t*>(input_data),
                             unextended_output_shape,
                             reinterpret_cast<int16_t*>(output_data));
      break;

    case 4:
      TransposeImpl<int32_t>(params, unextended_input_shape,
                             reinterpret_cast<const int32_t*>(input_data),
                             unextended_output_shape,
                             reinterpret_cast<int32_t*>(output_data));
      break;
    case 8:
      TransposeImpl<int64_t>(params, unextended_input_shape,
                             reinterpret_cast<const int64_t*>(input_data),
                             unextended_output_shape,
                             reinterpret_cast<int64_t*>(output_data));
      break;
  }
}

inline void TransposeConv(
    const ConvParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& filter_shape,
    const float* filter_data, const RuntimeShape& output_shape,
    float* output_data, const RuntimeShape& im2col_shape, float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // Although transpose convolution simplifies to convolution with transposed
  // weights for strides of 1, non-unitary striding complicates matters. To
  // keep this reference implementation as clear as possible, we use a
  // "scatter" access pattern, where we loop through all the input elements,
  // computing their influence on the output, rather than looping through the
  // output elements in the typical "gather" access pattern of a conv. We
  // therefore must initialize the output array to zero.
  const int num_elements = output_shape.FlatSize();
  for (int i = 0; i < num_elements; i++) {
    output_data[i] = 0.0f;
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
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  float input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  float filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  output_data[Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] +=
                      input_value * filter_value;
                }
              }
            }
          }
        }
      }
    }
  }
}

inline void TransposeConv(const ConvParams& params,
                          const RuntimeShape& input_shape,
                          const uint8* input_data,
                          const RuntimeShape& filter_shape,
                          const uint8* filter_data,
                          const RuntimeShape& output_shape, uint8* output_data,
                          const RuntimeShape& im2col_shape, uint8* im2col_data,
                          int32* scratch_buffer) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int num_elements = output_shape.FlatSize();
  // We need to initialize scratch_buffer to all 0s, as we apply the same
  // 'scatter' based trick as in float version.
  memset(scratch_buffer, 0, num_elements * sizeof(int32));

  // Loop through input elements one at a time.
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          // Loop through the output elements it will influence.
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int out_channel = 0; out_channel < output_depth;
                   ++out_channel) {
                // Compute output element location.
                const int out_x = out_x_origin + filter_x;
                const int out_y = out_y_origin + filter_y;
                // We cannot accumulate out of bounds.
                if ((out_x >= 0) && (out_x < output_width) && (out_y >= 0) &&
                    (out_y < output_height)) {
                  uint8 input_value = input_data[Offset(
                      input_shape, batch, in_y, in_x, in_channel)];
                  uint8 filter_value =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  scratch_buffer[Offset(output_shape, batch, out_y, out_x,
                                        out_channel)] +=
                      (input_value + input_offset) *
                      (filter_value + filter_offset);
                }
              }
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < num_elements; ++i) {
    int32 acc = scratch_buffer[i];
    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
    acc += output_offset;
    // Clamp the output before converting back to uint8.
    acc = std::max(acc, output_activation_min);
    acc = std::min(acc, output_activation_max);
    output_data[i] = static_cast<uint8>(acc);
  }
}

template <typename D, typename T>
void Select(const RuntimeShape& input_condition_shape,
            const D* input_condition_data, const RuntimeShape& input_x_shape,
            const T* input_x_data, const RuntimeShape& input_y_shape,
            const T* input_y_data, const RuntimeShape& output_shape,
            T* output_data) {
  const int64_t flatsize = MatchingFlatSize(
      input_condition_shape, input_x_shape, input_y_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] =
        input_condition_data[i] ? input_x_data[i] : input_y_data[i];
  }
}

template <typename D, typename T>
void RankOneSelect(const RuntimeShape& input_condition_shape,
                   const D* input_condition_data,
                   const RuntimeShape& input_x_shape, const T* input_x_data,
                   const RuntimeShape& input_y_shape, const T* input_y_data,
                   const RuntimeShape& output_shape, T* output_data) {
  const int64_t outer_size = input_condition_shape.FlatSize();
  TFLITE_DCHECK_EQ(
      MatchingDim(input_x_shape, 0, input_y_shape, 0, output_shape, 0),
      outer_size);
  const int64_t inner_size =
      MatchingFlatSizeSkipDim(input_x_shape, 0, input_y_shape, output_shape);

  int64_t offset = 0;
  for (int64_t i = 0; i < outer_size; i++) {
    const T* input_data = input_condition_data[i] ? input_x_data : input_y_data;
    memcpy(output_data + offset, input_data + offset, inner_size * sizeof(T));
    offset += inner_size;
  }
}

template <typename D, typename T>
void BroadcastSelect4DSlow(const RuntimeShape& input_condition_shape,
                           const D* input_condition_data,
                           const RuntimeShape& input_x_shape,
                           const T* input_x_data,
                           const RuntimeShape& input_y_shape,
                           const T* input_y_data,
                           const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_LE(input_condition_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(input_x_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(input_y_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), 4);

  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  NdArrayDesc<4> desc_condition;
  NdArrayDesc<4> desc_x;
  NdArrayDesc<4> desc_y;
  NdArrayDescsForElementwiseBroadcast(input_condition_shape, input_x_shape,
                                      input_y_shape, &desc_condition, &desc_x,
                                      &desc_y);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int condition_index =
              SubscriptToIndex(desc_condition, b, y, x, c);
          const int x_index = SubscriptToIndex(desc_x, b, y, x, c);
          const int y_index = SubscriptToIndex(desc_y, b, y, x, c);
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              input_condition_data[condition_index] ? input_x_data[x_index]
                                                    : input_y_data[y_index];
        }
      }
    }
  }
}

template <typename D, typename T>
void SelectTrueCoords(const RuntimeShape& input_condition_shape,
                      const D* input_condition_data, T* output_data) {
  const size_t size = input_condition_shape.FlatSize();
  const size_t cond_rank = input_condition_shape.DimensionsCount();

  std::vector<int> dims_to_count(cond_rank, 0);
  int cur_flat_size = size;
  for (int i = 0; i < cond_rank; ++i) {
    dims_to_count[i] = cur_flat_size / input_condition_shape.Dims(i);
    cur_flat_size = dims_to_count[i];
  }

  int output_index = 0;
  for (int i = 0; i < size; ++i) {
    if (input_condition_data[i]) {
      // Insert the coordinate of the current item (row major) into output.
      int flat_index = i;
      for (int j = 0; j < cond_rank; ++j) {
        int coord_j = flat_index / dims_to_count[j];
        output_data[output_index * cond_rank + j] = coord_j;
        flat_index %= dims_to_count[j];
      }
      output_index++;
    }
  }
}

// For easy implementation, the indices is always a vector of size-4 vectors.
template <typename T, typename TI>
inline void SparseToDense(const std::vector<std::vector<TI>>& indices,
                          const T* values, T default_value,
                          bool value_is_scalar,
                          const RuntimeShape& unextended_output_shape,
                          T* output_data) {
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int value_count = indices.size();

  // First fill the output_data with default value.
  const int num_elements = output_shape.FlatSize();
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = default_value;
  }

  // Special handle for value is scalar case to avoid checking the boolean
  // condition within the loop every time.
  if (value_is_scalar) {
    for (int i = 0; i < value_count; ++i) {
      const std::vector<TI>& index = indices[i];
      TFLITE_DCHECK_EQ(index.size(), 4);
      const T value = *values;  // just use the first value.
      output_data[Offset(output_shape, index[0], index[1], index[2],
                         index[3])] = value;
    }
    return;
  }

  // Go through the values and indices to fill the sparse values.
  for (int i = 0; i < value_count; ++i) {
    const std::vector<TI>& index = indices[i];
    TFLITE_DCHECK_EQ(index.size(), 4);
    const T value = values[i];
    output_data[Offset(output_shape, index[0], index[1], index[2], index[3])] =
        value;
  }
}

template <typename T>
inline void Pow(const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = std::pow(input1_data[i], input2_data[i]);
  }
}

template <typename T>
inline void BroadcastPow4DSlow(const RuntimeShape& unextended_input1_shape,
                               const T* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const T* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          auto out_idx = Offset(output_shape, b, y, x, c);
          auto in1_idx = SubscriptToIndex(desc1, b, y, x, c);
          auto in2_idx = SubscriptToIndex(desc2, b, y, x, c);
          auto in1_val = input1_data[in1_idx];
          auto in2_val = input2_data[in2_idx];
          output_data[out_idx] = std::pow(in1_val, in2_val);
        }
      }
    }
  }
}

template <typename T>
inline void ResizeNearestNeighbor(
    const tflite::ResizeNearestNeighborParams& op_params,
    const RuntimeShape& unextended_input_shape, const T* input_data,
    const RuntimeShape& output_size_shape, const int32* output_size_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  // Align corners = true is not supported.
  TFLITE_DCHECK(!op_params.align_corners);
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

  // We use float to ensure agreement with the Tensorflow implementation.
  const float height_scale = static_cast<float>(input_height) / output_height;
  const float width_scale = static_cast<float>(input_width) / output_width;

  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const T* input_ptr = input_data;
  T* output_ptr = output_data;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32 in_y = std::min(static_cast<int32>(std::floor(y * height_scale)),
                            input_height - 1);
      const T* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        int32 in_x = std::min(static_cast<int32>(std::floor(x * width_scale)),
                              input_width - 1);
        const T* x_input_ptr = y_input_ptr + in_x * col_offset;
        memcpy(output_ptr, x_input_ptr, depth * sizeof(T));
        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}

template <typename T>
void Fill(const RuntimeShape& value_shape, const T* value_data,
          const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(value_shape.DimensionsCount(), 0);
  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = *value_data;
  }
}

template <typename Scalar>
void Reverse(int axis, const RuntimeShape& input_shape,
             const Scalar* input_data, const RuntimeShape& output_shape,
             Scalar* output_data) {
  ruy::profiler::ScopeLabel label("Reverse");

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int copy_size = 1;
  for (int i = axis + 1; i < input_shape.DimensionsCount(); ++i) {
    copy_size *= input_shape.Dims(i);
  }

  const int dims_at_axis = input_shape.Dims(axis);
  for (int i = 0; i < outer_size; ++i) {
    for (int j = 0; j < dims_at_axis; ++j) {
      const int start_pos = (i * dims_at_axis + j) * copy_size;
      Scalar* output_ptr = output_data + start_pos;
      int loc = (i * dims_at_axis + dims_at_axis - j - 1) * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}

template <typename Scalar, typename TS>
void ReverseSequence(const TS* seq_lengths, const int seq_dim,
                     const int batch_dim, const RuntimeShape& input_shape,
                     const Scalar* input_data, const RuntimeShape& output_shape,
                     Scalar* output_data) {
  ruy::profiler::ScopeLabel label("ReverseSequence");

  int outer_size = 1;
  int outer_dim = std::min(batch_dim, seq_dim);
  int medium_dim = std::max(batch_dim, seq_dim);
  for (int i = 0; i < outer_dim; ++i) {
    outer_size *= input_shape.Dims(i);
  }

  int medium_size = 1;
  for (int i = outer_dim + 1; i < medium_dim; ++i) {
    medium_size *= input_shape.Dims(i);
  }

  int copy_size = 1;
  for (int i = medium_dim + 1; i < input_shape.DimensionsCount(); ++i) {
    copy_size *= input_shape.Dims(i);
  }

  const int dims_at_outer_dim = input_shape.Dims(outer_dim);
  const int dims_at_medium_dim = input_shape.Dims(medium_dim);

  Scalar* output_ptr;
  if (batch_dim > seq_dim) {
    for (int i = 0; i < outer_size; ++i) {
      for (int j = 0; j < dims_at_outer_dim; ++j) {
        const int in_pos_base = (i * dims_at_outer_dim + j) * medium_size;
        for (int p = 0; p < medium_size; ++p) {
          for (int q = 0; q < dims_at_medium_dim; ++q) {
            const int in_pos =
                ((in_pos_base + p) * dims_at_medium_dim + q) * copy_size;
            const Scalar* in_ptr = input_data + in_pos;
            int sl = seq_lengths[q] - 1;
            if (j > sl) {
              output_ptr = output_data + in_pos;
            } else {
              const int out_pos_base =
                  (i * dims_at_outer_dim + sl - j) * medium_size;
              const int out_pos =
                  ((out_pos_base + p) * dims_at_medium_dim + q) * copy_size;
              output_ptr = output_data + out_pos;
            }
            memcpy(output_ptr, in_ptr, copy_size * sizeof(Scalar));
          }
        }
      }
    }
  } else if (batch_dim < seq_dim) {
    for (int i = 0; i < outer_size; ++i) {
      for (int j = 0; j < dims_at_outer_dim; ++j) {
        const int in_pos_base = (i * dims_at_outer_dim + j) * medium_size;
        int sl = seq_lengths[j] - 1;
        const int out_pos_base = (i * dims_at_outer_dim + j) * medium_size;
        for (int p = 0; p < medium_size; ++p) {
          for (int q = 0; q < dims_at_medium_dim; ++q) {
            const int in_pos =
                ((in_pos_base + p) * dims_at_medium_dim + q) * copy_size;
            const Scalar* in_ptr = input_data + in_pos;
            if (q > sl) {
              output_ptr = output_data + in_pos;
            } else {
              const int out_pos =
                  ((out_pos_base + p) * dims_at_medium_dim + sl - q) *
                  copy_size;
              output_ptr = output_data + out_pos;
            }
            memcpy(output_ptr, in_ptr, copy_size * sizeof(Scalar));
          }
        }
      }
    }
  }
}

template <typename T>
inline void HardSwish(const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Float");
  auto matching_size = MatchingFlatSize(input_shape, output_shape);
  const T* in_end = input_data + matching_size;
  for (; input_data < in_end; input_data++, output_data++) {
    const float in = *input_data;
    *output_data =
        in * std::min(static_cast<T>(6), std::max(static_cast<T>(0), in + 3)) /
        6;
  }
}

inline int16_t SaturatingLeftShift(int16_t value, int amount) {
  int32_t result = static_cast<int32_t>(value) * (1 << amount);
  result = std::min<int32_t>(result, std::numeric_limits<int16_t>::max());
  result = std::max<int32_t>(result, std::numeric_limits<int16_t>::min());
  return result;
}

// Similar to ARM instruction SQDMULH.
// Similar to gemmlowp::SaturatingRoundingDoublingHighMul except
// rounding to zero instead of to nearest (SQRDMULH).
inline std::int16_t SaturatingDoublingHighMul(std::int16_t a, std::int16_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int16_t>::min();
  std::int32_t a_32(a);
  std::int32_t b_32(b);
  std::int32_t ab_32 = a_32 * b_32;
  std::int16_t ab_x2_high16 = static_cast<std::int16_t>((ab_32) / (1 << 15));
  return overflow ? std::numeric_limits<std::int16_t>::max() : ab_x2_high16;
}

template <typename T>
inline void HardSwish(const HardSwishParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Quantized");

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int16_t input_value = input_data[i] - params.input_zero_point;
    // Left-shift as much as we can without overflow/saturation to put
    // significant bits in the high bits of our 16-bit fixedpoint values, so
    // that fixed-point approximate computations below are as accurate as
    // possible.
    const int16_t input_value_on_hires_input_scale = input_value << 7;
    // Compute the input value on essentially the output scale, just not
    // right-shifted yet. This is the value that we'll use in the (x >= +3)
    // case, and that in the general case we'll multiply against the "relu-ish"
    // fixed-point multiplier in [0, 1].
    const int16_t input_value_on_preshift_output_scale =
        gemmlowp::SaturatingRoundingDoublingHighMul(
            input_value_on_hires_input_scale,
            params.output_multiplier_fixedpoint_int16);
    // Now compute the "relu-ish multiplier". In the (-3 <= x <= +3) case, that
    // is just an affine rescaling of x from [-3, 3] to [0, 1]. In the general
    // case, it is just that plus saturation at the boundaries of [-3, 3].
    // First, we rescale from [-3, 3] to [-1, 1], saturating.
    // That is done by rescaling the input value with a fixed-point multiplier
    // (reluish_multiplier_fixedpoint) and bit-shift such that we represent
    // that input value on the scale where the real value 3.0f is represented
    // by the quantized value 32768.  (+32768 is actually not representable as
    // int16, so this saturates at +32767, and that is seen empirically to be
    // a negligible contribution to numerical error/bias).
    //
    // This code is careful to correctly implement any magnitude of multiplier,
    // involving either a right shift or a left shift, with correct saturation
    // behavior in the left-shift case. This forces this code to be more
    // complicated, but is necessary for real applications: a partially
    // trained quantized MobileNet v3-small model that motivated this code
    // exhibits some large [min, max] range boundaries, of the order of
    // magnitude of 10 or 100 depending on layers.
    //
    // The next few lines are basically just an ordinary
    // MultiplyByQuantizedMultiplier, except that we are more careful here
    // about the fine details of saturation when left-shifting, because here
    // overflow in left-shift is a common case, not an anomaly as
    // MultiplyByQuantizedMultiplier assumes.
    int16_t reluish_value = input_value_on_hires_input_scale;
    // Shift left, saturating, as much as we can while ensuring that this
    // saturation will not contribute to the result. That is, left shift amount
    // reduced by 1.
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(
          reluish_value, params.reluish_multiplier_exponent - 1);
    }
    // Apply the fixed-point multiplier, dividing the value by a divisor
    // ranging in [1, 2].
    reluish_value = gemmlowp::SaturatingRoundingDoublingHighMul(
        reluish_value, params.reluish_multiplier_fixedpoint_int16);
    // Apply the last bit of left-shift. Thus, in the left-shifting case, if
    // any saturation affects the result, it is happening here --- any
    // saturation having occurred above is overwritten here, not affecting the
    // result.
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(reluish_value, 1);
    }
    // Shift right, in the right-shifting case.
    if (params.reluish_multiplier_exponent < 0) {
      reluish_value = gemmlowp::RoundingDivideByPOT(
          reluish_value, -params.reluish_multiplier_exponent);
    }
    // At this point we have rescaled the value into a 16bit fixedpoint
    // reluish_value in [-1, 1].
    // We now convert that to a 16bit fixedpoint value in [0, 1].
    reluish_value = (reluish_value + (1 << 15)) >> 1;
    // Use of SaturatingDoublingHighMul here is important to cancel the biases
    // from the above SaturatingRoundingDoublingHighMul.
    //
    // On a partially trained MobileNet-v3-small,
    //
    //                                       | bias on    |  ImageNet
    //                                       | quantized  |  Top-1
    // Operation used here                   | values     |  accuracy (50k)
    // --------------------------------------+------------+-----------
    // SaturatingDoublingHighMul             | -0.0024    |  58.920
    // SaturatingRoundingDoublingHighMul     | -0.0067    |  58.064
    //
    // In activations_test, this is covered by this testcase:
    //     QuantizedActivationsOpTest.HardSwishBias
    //
    const int16_t preshift_output_value = SaturatingDoublingHighMul(
        reluish_value, input_value_on_preshift_output_scale);
    // We were so far operating on the pre-shift output scale. Now we finally
    // apply that output shift, arriving at the final output scale.
    int16_t output_value = gemmlowp::RoundingDivideByPOT(
        preshift_output_value, -params.output_multiplier_exponent);
    output_value += params.output_zero_point;
    output_value =
        std::min<int16_t>(output_value, std::numeric_limits<T>::max());
    output_value =
        std::max<int16_t>(output_value, std::numeric_limits<T>::min());
    output_data[i] = output_value;
  }
}

template <typename T>
inline void SegmentSum(const RuntimeShape& input_shape, const T* input_data,
                       const RuntimeShape& segment_ids_shape,
                       const int32_t* segment_ids_data,
                       const RuntimeShape& output_shape, T* output_data) {
  const int segment_flat_size =
      MatchingFlatSizeSkipDim(input_shape, 0, output_shape);

  memset(output_data, 0, sizeof(T) * output_shape.FlatSize());

  for (int i = 0; i < input_shape.Dims(0); i++) {
    int output_index = segment_ids_data[i];
    for (int j = 0; j < segment_flat_size; ++j) {
      output_data[output_index * segment_flat_size + j] +=
          input_data[i * segment_flat_size + j];
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_REFERENCE_OPS_H_
