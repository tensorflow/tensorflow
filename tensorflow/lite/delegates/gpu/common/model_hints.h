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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_HINTS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_HINTS_H_

#include <cstdint>

namespace tflite {
namespace gpu {

struct ModelHints {
  using ModelHint = uint64_t;

  // By default we want the fastest inference.
  static constexpr ModelHint kFastestInference = 0x00000000;
  // Can improve compilation time, but inference can be slower.
  static constexpr ModelHint kReduceKernelsCount = 0x00000001;
  // Can improve tuning time, but inference can be slower.
  static constexpr ModelHint kFastTuning = 0x00000001 << 1;

  // Can improve performance and memory consumption, but slow down
  // initialization and create more unique kernels.
  static constexpr ModelHint kAllowSpecialKernels = 0x00000001 << 2;

  // By default we apply Winograd optimized kernels and it improves performance.
  // But it also can increase memory usage and decrease precision.
  // This hint can disable Winograd optimizations.
  static constexpr ModelHint kNoWinogradOptimizations = 0x00000001 << 3;

  // By default, the same weights can be duplicated among different nodes of
  // convolution. With this hint we will try to reuse common object, when it
  // possible.
  // Can decrease constant memory usage(if model has the same weights).
  static constexpr ModelHint kReuseConvWeights = 0x00000001 << 4;

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

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_HINTS_H_
