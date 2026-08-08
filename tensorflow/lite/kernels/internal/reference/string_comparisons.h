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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRING_COMPARISONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRING_COMPARISONS_H_

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_loop.h"
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

namespace reference_ops {

inline bool StringRefEqualFn(const StringRef& lhs, const StringRef& rhs) {
  if (lhs.len != rhs.len) return false;
  for (int i = 0; i < lhs.len; ++i) {
    if (lhs.str[i] != rhs.str[i]) return false;
  }
  return true;
}

inline bool StringRefNotEqualFn(const StringRef& lhs, const StringRef& rhs) {
  return !StringRefEqualFn(lhs, rhs);
}

inline void ComparisonStringImpl(bool (*F)(const StringRef&, const StringRef&),
                                 const RuntimeShape& input1_shape,
                                 const TfLiteTensor* input1,
                                 const RuntimeShape& input2_shape,
                                 const TfLiteTensor* input2,
                                 const RuntimeShape& output_shape,
                                 bool* output_data) {
  const int64_t flatsize =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int64_t i = 0; i < flatsize; ++i) {
    const auto lhs = GetString(input1, i);
    const auto rhs = GetString(input2, i);
    output_data[i] = F(lhs, rhs);
  }
}

struct TfLiteTensorStringPointer {
  const TfLiteTensor* tensor;
  int index;

  TfLiteTensorStringPointer operator+(int offset) const {
    return {tensor, index + offset};
  }
  StringRef operator*() const { return GetString(tensor, index); }
  StringRef operator[](int offset) const {
    return GetString(tensor, index + offset);
  }
};

inline void BroadcastComparison4DSlowStringImpl(
    bool (*F)(const StringRef&, const StringRef&),
    const RuntimeShape& unextended_input1_shape, const TfLiteTensor* input1,
    const RuntimeShape& unextended_input2_shape, const TfLiteTensor* input2,
    const RuntimeShape& unextended_output_shape, bool* output_data) {
  auto op = [F](const StringRef& a, const StringRef& b) { return F(a, b); };
  BroadcastBinaryOpSimple(
      unextended_input1_shape, TfLiteTensorStringPointer{input1, 0},
      unextended_input2_shape, TfLiteTensorStringPointer{input2, 0},
      unextended_output_shape, output_data, op);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_STRING_COMPARISONS_H_
