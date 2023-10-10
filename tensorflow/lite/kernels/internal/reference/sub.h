/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <class T>
struct SubImpl {
  template <class F>
  static void BroadcastInput1(const ArithmeticParams& params,
                              const T* input1_data, const T* input2_data,
                              T* output_data, size_t size, F binary_func) {
    for (size_t c = 0; c < size; ++c) {
      output_data[c] = binary_func(input1_data[0], input2_data[c], params);
    }
  }

  template <class F>
  static void BroadcastInput2(const ArithmeticParams& params,
                              const T* input1_data, const T* input2_data,
                              T* output_data, size_t size, F binary_func) {
    for (size_t c = 0; c < size; ++c) {
      output_data[c] = binary_func(input1_data[c], input2_data[0], params);
    }
  }

  template <class F>
  static void ElementWise(const ArithmeticParams& params, const T* input1_data,
                          const T* input2_data, T* output_data, size_t size,
                          F binary_func) {
    for (size_t c = 0; c < size; ++c) {
      output_data[c] = binary_func(input1_data[c], input2_data[c], params);
    }
  }
};

template <>
struct SubImpl<int32_t> {
  template <class F>
  static void BroadcastInput1(const ArithmeticParams& params,
                              const int32_t* input1_data,
                              const int32_t* input2_data, int32_t* output_data,
                              size_t size, F binary_func) {
    size_t c = 0;
    int32_t activation_min, activation_max;
    GetActivationParams(params, &activation_min, &activation_max);
#ifdef USE_NEON
    const int32x4_t vmax = vdupq_n_s32(activation_max);
    const int32x4_t vmin = vdupq_n_s32(activation_min);
    const int32x4_t va = vdupq_n_s32(input1_data[0]);
    for (; c + 4 <= size; c += 4) {
      const int32x4_t vb = vld1q_s32(&input2_data[c]);
      int32x4_t vres = vsubq_s32(va, vb);
      vres = vmaxq_s32(vmin, vres);
      vres = vminq_s32(vmax, vres);
      vst1q_s32(&output_data[c], vres);
    }
#endif
    for (; c < size; ++c) {
      output_data[c] = binary_func(input1_data[0], input2_data[c], params);
    }
  }

  template <class F>
  static void BroadcastInput2(const ArithmeticParams& params,
                              const int32_t* input1_data,
                              const int32_t* input2_data, int32_t* output_data,
                              size_t size, F binary_func) {
    size_t c = 0;
    int32_t activation_min, activation_max;
    GetActivationParams(params, &activation_min, &activation_max);
#ifdef USE_NEON
    const int32x4_t vmax = vdupq_n_s32(activation_max);
    const int32x4_t vmin = vdupq_n_s32(activation_min);
    const int32x4_t vb = vdupq_n_s32(input2_data[0]);
    for (; c + 4 <= size; c += 4) {
      const int32x4_t va = vld1q_s32(&input1_data[c]);
      int32x4_t vres = vsubq_s32(va, vb);
      vres = vmaxq_s32(vmin, vres);
      vres = vminq_s32(vmax, vres);
      vst1q_s32(&output_data[c], vres);
    }
#endif
    for (; c < size; ++c) {
      output_data[c] = binary_func(input1_data[c], input2_data[0], params);
    }
  }

  template <class F>
  static void ElementWise(const ArithmeticParams& params,
                          const int32_t* input1_data,
                          const int32_t* input2_data, int32_t* output_data,
                          size_t size, F binary_func) {
    size_t c = 0;
    int32_t activation_min, activation_max;
    GetActivationParams(params, &activation_min, &activation_max);
#ifdef USE_NEON
    int32x4_t vmax = vdupq_n_s32(activation_max);
    int32x4_t vmin = vdupq_n_s32(activation_min);
    for (; c + 4 <= size; c += 4) {
      const int32x4_t va = vld1q_s32(&input1_data[c]);
      const int32x4_t vb = vld1q_s32(&input2_data[c]);
      int32x4_t vres = vsubq_s32(va, vb);
      vres = vmaxq_s32(vmin, vres);
      vres = vminq_s32(vmax, vres);
      vst1q_s32(&output_data[c], vres);
    }
#endif
    for (; c < size; ++c) {
      output_data[c] = binary_func(input1_data[c], input2_data[c], params);
    }
  }
};

template <typename T, typename F>
inline void BroadcastSubRecursiveDimensions(
    int dimension, const ArithmeticParams& params, const T* input1_data,
    const T* input2_data, T* output_data, size_t* input1_offset_p,
    size_t* input2_offset_p, size_t* output_offset,
    size_t* compressed_input1_stride, size_t* compressed_input2_stride,
    size_t* compressed_output_shape, F binary_func) {
  if (dimension > 0) {
    for (size_t c = 0; c < compressed_output_shape[dimension]; ++c) {
      size_t input1_offset_c = *input1_offset_p;
      size_t input2_offset_c = *input2_offset_p;
      BroadcastSubRecursiveDimensions(
          dimension - 1, params, input1_data, input2_data, output_data,
          &input1_offset_c, &input2_offset_c, output_offset,
          compressed_input1_stride, compressed_input2_stride,
          compressed_output_shape, binary_func);
      *input1_offset_p += compressed_input1_stride[dimension];
      *input2_offset_p += compressed_input2_stride[dimension];
    }
  } else {
    TFLITE_DCHECK(dimension == 0);
    bool input1_is_broadcast = compressed_input1_stride[dimension] == 0;
    bool input2_is_broadcast = compressed_input2_stride[dimension] == 0;
    TFLITE_DCHECK(!(input1_is_broadcast && input2_is_broadcast));
    const T* input1_data_ptr = input1_data + *input1_offset_p;
    const T* input2_data_ptr = input2_data + *input2_offset_p;
    T* output_data_ptr = output_data + *output_offset;
    if (input1_is_broadcast) {
      // input1 is broadcast.
      SubImpl<T>::BroadcastInput1(
          params, input1_data_ptr, input2_data_ptr, output_data_ptr,
          compressed_output_shape[dimension], binary_func);
      *input2_offset_p += compressed_output_shape[dimension];
    } else if (input2_is_broadcast) {
      // input2 is broadcast.
      SubImpl<T>::BroadcastInput2(
          params, input1_data_ptr, input2_data_ptr, output_data_ptr,
          compressed_output_shape[dimension], binary_func);
      *input1_offset_p += compressed_output_shape[dimension];
    } else {
      // Add element-wise.
      SubImpl<T>::ElementWise(params, input1_data_ptr, input2_data_ptr,
                              output_data_ptr,
                              compressed_output_shape[dimension], binary_func);
      *input1_offset_p += compressed_output_shape[dimension];
      *input2_offset_p += compressed_output_shape[dimension];
    }
    *output_offset += compressed_output_shape[dimension];
  }
}

// TODO: b/296510380 - we may be able to factor out this to common.h for all
// binary arithmetic ops (add, sub, mul).
template <typename T, typename F>
inline void BroadcastSubCommon(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const T* input1_data,
                               const RuntimeShape& input2_shape,
                               const T* input2_data,
                               const RuntimeShape& output_shape, T* output_data,
                               F binary_func) {
  constexpr int kMaxBroadcastDim = 6;
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), kMaxBroadcastDim);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), kMaxBroadcastDim);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), kMaxBroadcastDim);

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

  size_t compressed_input1_stride[kMaxBroadcastDim];
  size_t compressed_input2_stride[kMaxBroadcastDim];
  size_t compressed_output_shape[kMaxBroadcastDim];
  bool broadcastable_shape = ReduceDimensionsForBroadcast<kMaxBroadcastDim>(
      input1_shape, input2_shape, compressed_input1_stride,
      compressed_input2_stride, compressed_output_shape);
  // Skip broadcasting for degenerate shapes.
  if (!broadcastable_shape) {
    return;
  }

  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastSubRecursiveDimensions(
      kMaxBroadcastDim - 1, params, input1_data, input2_data, output_data,
      &input1_offset, &input2_offset, &output_offset, compressed_input1_stride,
      compressed_input2_stride, compressed_output_shape, binary_func);
}

// TODO(b/151345304): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastSubSlow(const ArithmeticParams& params,
                      const RuntimeShape& input1_shape, const T* input1_data,
                      const RuntimeShape& input2_shape, const T* input2_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/T");
  BroadcastSubCommon<T>(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data,
      [](T input1_val, T input2_val, const ArithmeticParams& params) {
        T activation_min, activation_max;
        GetActivationParams(params, &activation_min, &activation_max);
        return ActivationFunctionWithMinMax(input1_val - input2_val,
                                            activation_min, activation_max);
      });
}

inline void BroadcastSub16POTSlow(const ArithmeticParams& params,
                                  const RuntimeShape& input1_shape,
                                  const int16_t* input1_data,
                                  const RuntimeShape& input2_shape,
                                  const int16_t* input2_data,
                                  const RuntimeShape& output_shape,
                                  int16_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub16POTSlow/int16_t");
  BroadcastSubCommon<int16_t>(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data,
      [](int16_t input1_val, int16_t input2_val,
         const ArithmeticParams& params) {
        const int32_t scaled_input1_val =
            gemmlowp::RoundingDivideByPOT(input1_val, -params.input1_shift);
        const int32_t scaled_input2_val =
            gemmlowp::RoundingDivideByPOT(input2_val, -params.input2_shift);
        const int32_t raw_output = scaled_input1_val - scaled_input2_val;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        return static_cast<int16_t>(clamped_output);
      });
}

template <typename T>
void BroadcastQuantSubSlow(const ArithmeticParams& params,
                           const RuntimeShape& input1_shape,
                           const T* input1_data,
                           const RuntimeShape& input2_shape,
                           const T* input2_data,
                           const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastQuantSubSlow/T");
  BroadcastSubCommon<T>(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data,
      [](T input1_val, T input2_val, const ArithmeticParams& params) {
        const int32_t shifted_input1_val =
            (params.input1_offset + input1_val) * (1 << params.left_shift);
        const int32_t shifted_input2_val =
            (params.input2_offset + input2_val) * (1 << params.left_shift);
        const int32_t scaled_input1_val =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                shifted_input1_val, params.input1_multiplier,
                params.input1_shift);
        const int32_t scaled_input2_val =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                shifted_input2_val, params.input2_multiplier,
                params.input2_shift);
        const int32_t raw_sub = scaled_input1_val - scaled_input2_val;
        const int32_t raw_output =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                raw_sub, params.output_multiplier, params.output_shift) +
            params.output_offset;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        return static_cast<T>(clamped_output);
      });
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
template <typename T>
inline void SubElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
  for (int i = 0; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sub = scaled_input1_val - scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sub, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GE(params.input1_offset, -128);
  TFLITE_DCHECK_GE(params.input2_offset, -128);
  // offset = -quantization_params.zero_point in PrepareGeneralSubOp().
  // So it's maximum can be 128 not 127.
  TFLITE_DCHECK_LE(params.input1_offset, 128);
  TFLITE_DCHECK_LE(params.input2_offset, 128);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int16_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_EQ(params.input1_offset, 0);
  TFLITE_DCHECK_EQ(params.input2_offset, 0);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  BroadcastSubCommon<T>(
      params, input1_shape, input1_data, input2_shape, input2_data,
      output_shape, output_data,
      [](T input1_val, T input2_val, const ArithmeticParams& params) {
        return input1_val - input2_val;
      });
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                int32_t* activation_min,
                                int32_t* activation_max) {
  *activation_min = params.quantized_activation_min;
  *activation_max = params.quantized_activation_max;
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                float* activation_min, float* activation_max) {
  *activation_min = params.float_activation_min;
  *activation_max = params.float_activation_max;
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                int64_t* activation_min,
                                int64_t* activation_max) {
  *activation_min = params.int64_activation_min;
  *activation_max = params.int64_activation_max;
}

template <typename T>
inline void SubWithActivation(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("SubWithActivation");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  T activation_min, activation_max;
  SetActivationMinMax(params, &activation_min, &activation_max);

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], activation_min, activation_max);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
