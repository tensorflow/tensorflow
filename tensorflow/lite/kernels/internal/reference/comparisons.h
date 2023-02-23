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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_COMPARISONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_COMPARISONS_H_

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline bool EqualFn(T lhs, T rhs) {
  return lhs == rhs;
}

template <typename T>
inline bool NotEqualFn(T lhs, T rhs) {
  return lhs != rhs;
}

template <typename T>
inline bool GreaterFn(T lhs, T rhs) {
  return lhs > rhs;
}
template <typename T>
inline bool GreaterEqualFn(T lhs, T rhs) {
  return lhs >= rhs;
}
template <typename T>
inline bool LessFn(T lhs, T rhs) {
  return lhs < rhs;
}
template <typename T>
inline bool LessEqualFn(T lhs, T rhs) {
  return lhs <= rhs;
}

template <typename T>
using ComparisonFn = bool (*)(T, T);

template <typename T, ComparisonFn<T> F>
inline void ComparisonImpl(
    const ComparisonParams& op_params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, bool* output_data) {
  const int64_t flatsize =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    output_data[i] = F(input1_data[i], input2_data[i]);
  }
}

template <ComparisonFn<float> F>
inline void Comparison(const ComparisonParams& op_params,
                       const RuntimeShape& input1_shape,
                       const float* input1_data,
                       const RuntimeShape& input2_shape,
                       const float* input2_data,
                       const RuntimeShape& output_shape, bool* output_data) {
  ComparisonImpl<float, F>(op_params, input1_shape, input1_data, input2_shape,
                           input2_data, output_shape, output_data);
}

template <typename T, ComparisonFn<int32_t> F>
inline void ComparisonWithScaling(
    const ComparisonParams& op_params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, bool* output_data) {
  int left_shift = op_params.left_shift;
  int32_t input1_offset = op_params.input1_offset;
  int32_t input1_multiplier = op_params.input1_multiplier;
  int input1_shift = op_params.input1_shift;
  int32_t input2_offset = op_params.input2_offset;
  int32_t input2_multiplier = op_params.input2_multiplier;
  int input2_shift = op_params.input2_shift;

  const int64_t flatsize =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    const int32_t input1_val = input1_offset + input1_data[i];
    const int32_t input2_val = input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, input1_multiplier, input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, input2_multiplier, input2_shift);
    output_data[i] = F(scaled_input1_val, scaled_input2_val);
  }
}

struct BroadcastComparison4DSlowCommon {
  const RuntimeShape output_shape;
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
};

inline BroadcastComparison4DSlowCommon BroadcastComparison4DSlowPreprocess(
    const RuntimeShape& unextended_input1_shape,
    const RuntimeShape& unextended_input2_shape,
    const RuntimeShape& unextended_output_shape) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  return {RuntimeShape::ExtendedShape(4, unextended_output_shape), desc1,
          desc2};
}

template <typename T, ComparisonFn<T> F>
inline void BroadcastComparison4DSlowImpl(
    const ComparisonParams& op_params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const T* input2_data,
    const RuntimeShape& unextended_output_shape, bool* output_data) {
  const BroadcastComparison4DSlowCommon dims =
      BroadcastComparison4DSlowPreprocess(unextended_input1_shape,
                                          unextended_input2_shape,
                                          unextended_output_shape);

  for (int b = 0; b < dims.output_shape.Dims(0); ++b) {
    for (int y = 0; y < dims.output_shape.Dims(1); ++y) {
      for (int x = 0; x < dims.output_shape.Dims(2); ++x) {
        for (int c = 0; c < dims.output_shape.Dims(3); ++c) {
          output_data[Offset(dims.output_shape, b, y, x, c)] =
              F(input1_data[SubscriptToIndex(dims.desc1, b, y, x, c)],
                input2_data[SubscriptToIndex(dims.desc2, b, y, x, c)]);
        }
      }
    }
  }
}

template <ComparisonFn<float> F>
inline void BroadcastComparison4DSlow(const ComparisonParams& op_params,
                                      const RuntimeShape& input1_shape,
                                      const float* input1_data,
                                      const RuntimeShape& input2_shape,
                                      const float* input2_data,
                                      const RuntimeShape& output_shape,
                                      bool* output_data) {
  BroadcastComparison4DSlowImpl<float, F>(op_params, input1_shape, input1_data,
                                          input2_shape, input2_data,
                                          output_shape, output_data);
}

template <typename T, ComparisonFn<int32_t> F>
inline void BroadcastComparison4DSlowWithScaling(
    const ComparisonParams& op_params,
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const T* input2_data,
    const RuntimeShape& unextended_output_shape, bool* output_data) {
  const BroadcastComparison4DSlowCommon dims =
      BroadcastComparison4DSlowPreprocess(unextended_input1_shape,
                                          unextended_input2_shape,
                                          unextended_output_shape);

  int left_shift = op_params.left_shift;
  int32_t input1_offset = op_params.input1_offset;
  int32_t input1_multiplier = op_params.input1_multiplier;
  int input1_shift = op_params.input1_shift;
  int32_t input2_offset = op_params.input2_offset;
  int32_t input2_multiplier = op_params.input2_multiplier;
  int input2_shift = op_params.input2_shift;

  for (int b = 0; b < dims.output_shape.Dims(0); ++b) {
    for (int y = 0; y < dims.output_shape.Dims(1); ++y) {
      for (int x = 0; x < dims.output_shape.Dims(2); ++x) {
        for (int c = 0; c < dims.output_shape.Dims(3); ++c) {
          const int32_t input1_val =
              input1_offset +
              input1_data[SubscriptToIndex(dims.desc1, b, y, x, c)];
          const int32_t input2_val =
              input2_offset +
              input2_data[SubscriptToIndex(dims.desc2, b, y, x, c)];
          const int32_t shifted_input1_val = input1_val * (1 << left_shift);
          const int32_t shifted_input2_val = input2_val * (1 << left_shift);
          const int32_t scaled_input1_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input1_val, input1_multiplier, input1_shift);
          const int32_t scaled_input2_val =
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  shifted_input2_val, input2_multiplier, input2_shift);
          output_data[Offset(dims.output_shape, b, y, x, c)] =
              F(scaled_input1_val, scaled_input2_val);
        }
      }
    }
  }
}

#define TFLITE_COMPARISON_OP(name)                                             \
  inline void name(const ComparisonParams& op_params,                          \
                   const RuntimeShape& input1_shape, const float* input1_data, \
                   const RuntimeShape& input2_shape, const float* input2_data, \
                   const RuntimeShape& output_shape, bool* output_data) {      \
    Comparison<name##Fn>(op_params, input1_shape, input1_data, input2_shape,   \
                         input2_data, output_shape, output_data);              \
  }                                                                            \
  template <typename T>                                                        \
  inline void name##NoScaling(                                                 \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    ComparisonImpl<T, name##Fn>(op_params, input1_shape, input1_data,          \
                                input2_shape, input2_data, output_shape,       \
                                output_data);                                  \
  }                                                                            \
  template <typename T>                                                        \
  inline void name##WithScaling(                                               \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    ComparisonWithScaling<T, name##Fn>(op_params, input1_shape, input1_data,   \
                                       input2_shape, input2_data,              \
                                       output_shape, output_data);             \
  }                                                                            \
  template <typename T>                                                        \
  inline void Broadcast4DSlow##name##NoScaling(                                \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlowImpl<T, name##Fn>(                                \
        op_params, input1_shape, input1_data, input2_shape, input2_data,       \
        output_shape, output_data);                                            \
  }                                                                            \
  inline void Broadcast4DSlow##name(                                           \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const float* input1_data, const RuntimeShape& input2_shape,              \
      const float* input2_data, const RuntimeShape& output_shape,              \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlow<name##Fn>(op_params, input1_shape, input1_data,  \
                                        input2_shape, input2_data,             \
                                        output_shape, output_data);            \
  }                                                                            \
  template <typename T>                                                        \
  inline void Broadcast4DSlow##name##WithScaling(                              \
      const ComparisonParams& op_params, const RuntimeShape& input1_shape,     \
      const T* input1_data, const RuntimeShape& input2_shape,                  \
      const T* input2_data, const RuntimeShape& output_shape,                  \
      bool* output_data) {                                                     \
    BroadcastComparison4DSlowWithScaling<T, name##Fn>(                         \
        op_params, input1_shape, input1_data, input2_shape, input2_data,       \
        output_shape, output_data);                                            \
  }
TFLITE_COMPARISON_OP(Equal);
TFLITE_COMPARISON_OP(NotEqual);
TFLITE_COMPARISON_OP(Greater);
TFLITE_COMPARISON_OP(GreaterEqual);
TFLITE_COMPARISON_OP(Less);
TFLITE_COMPARISON_OP(LessEqual);
#undef TFLITE_COMPARISON_OP

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_COMPARISONS_H_
