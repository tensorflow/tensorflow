/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REDUCE_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REDUCE_COMMON_H_

#include <stddef.h>

#include <type_traits>

#include "absl/types/span.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace reduce_utils {

inline bool CheckedElementCount(absl::Span<const int> dims, size_t& count) {
  return ::tflite::CheckedNumElements(dims, count) == kTfLiteOk;
}

template <typename Count>
inline bool CheckedReducedElementCount(absl::Span<const int> dims,
                                       absl::Span<const int> axis,
                                       Count& count) {
  static_assert(std::is_integral_v<Count>);
  ::tflite::CheckedInt<Count> product(1);
  for (const int axis_idx : axis) {
    if (axis_idx < 0 ||
        static_cast<size_t>(axis_idx) >= static_cast<size_t>(dims.size())) {
      return false;
    }
    const int dim = dims[axis_idx];
    if (dim < 0) {
      return false;
    }
    product *= dim;
  }
  if (product.Overflow()) {
    return false;
  }
  count = product.Value();
  return true;
}

}  // namespace reduce_utils
namespace ops {
namespace builtin {
namespace reduce {

enum ReduceType {
  kSum,
  kProd,
  kMax,
  kMin,
  kAny,
  kAll,
};

}  // namespace reduce
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REDUCE_COMMON_H_
