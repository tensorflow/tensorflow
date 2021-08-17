/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_PROFILING_INFO_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_PROFILING_INFO_H_

#include <string>
#include <vector>

#include "absl/time/time.h"

namespace tflite {
namespace gpu {

struct ProfilingInfo {
  struct DispatchInfo {
    std::string label;
    absl::Duration duration;
  };

  std::vector<DispatchInfo> dispatches;

  absl::Duration GetTotalTime() const;

  // Returns report (string of lines delimited by \n)
  // This method uses GPU counters and measure GPU time only.
  // Report has next structure:
  // Per kernel timing(K kernels):
  //   conv2d 3.2ms
  //   ...
  // --------------------
  // Accumulated time per operation type:
  //   conv2d - 14.5ms
  //   ....
  // --------------------
  // Ideal total time: 23.4ms // Total time for all kernels
  std::string GetDetailedReport() const;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_PROFILING_INFO_H_
