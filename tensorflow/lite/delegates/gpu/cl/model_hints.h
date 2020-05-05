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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_MODEL_HINTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_MODEL_HINTS_H_

#include <cstdint>

namespace tflite {
namespace gpu {
namespace cl {

struct ModelHints {
  using ModelHint = uint64_t;

  // By default we want the fastest inference
  static constexpr ModelHint kFastestInference = 0x00000000;
  // Can improve compilation time, but inference can be slower
  static constexpr ModelHint kReduceKernelsCount = 0x00000001;
  // Can improve tuning time, but inference can be slower
  static constexpr ModelHint kFastTuning = 0x00000002;

  void Add(ModelHint hint) {
    if (hint == kFastestInference) {
      hints = kFastestInference;
    } else {
      hints |= hint;
    }
  }

  bool Check(ModelHint hint) const { return hints & hint; }

  uint64_t hints = kFastestInference;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_MODEL_HINTS_H_
