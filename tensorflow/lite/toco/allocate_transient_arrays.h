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
#ifndef TENSORFLOW_LITE_TOCO_ALLOCATE_TRANSIENT_ARRAYS_H_
#define TENSORFLOW_LITE_TOCO_ALLOCATE_TRANSIENT_ARRAYS_H_

#include "tensorflow/lite/toco/model.h"

namespace toco {

// We align the allocated sizes to the next multiple of a cache line,
// to get simple performance characteristics without side effects of
// accesses to one buffer on accesses to another buffer.
// That also takes care of data type alignment for any reasonable type
// (no reasonable data type should have alignment greater than a cache line).
// Here we make CPU-centric assumptions, in particular, we assume 64-byte cache
// lines. Getting this wrong by a factor of 2x (if this ever changes) wouldn't
// be terrible.
// Embedded architectures may use a different value for alignment.
constexpr std::size_t kDefaultTransientDataAlignment = 64;

// Rounds up dividend to a value divisible by divisor.
inline std::size_t RoundUpToNextMultipleOf(std::size_t dividend,
                                           std::size_t divisor) {
  return ((dividend + divisor - 1) / divisor) * divisor;
}

void AllocateTransientArrays(Model* model,
                             std::size_t transient_data_alignment);

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_ALLOCATE_TRANSIENT_ARRAYS_H_
