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

#ifndef TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_NVTX_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_NVTX_UTILS_H_

#include <stack>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {

/***
 * We have no intention to use NVTX in tensorflow right now, we use this class
 * to track NVTX instrumentation inside NVIDIA libraries (such as TensorRT).
 * This bears a lot of resemblance to ScopedAnnotation for now.  In the future,
 * we will use TraceMe to keep track trace context within a thread.
 */
class NVTXRangeTracker {
 public:
  static void EnterRange(const std::string& range) {
    auto& range_stack = GetRangeStack();
    range_stack.push(range);
  }
  static void ExitRange() {
    auto& range_stack = GetRangeStack();
    if (!range_stack.empty()) range_stack.pop();
  }
  static const absl::string_view CurrentRange() {
    auto& range_stack = GetRangeStack();
    if (!range_stack.empty()) return range_stack.top();
    return "";
  }

 private:
  static std::stack<std::string>& GetRangeStack();

  TF_DISALLOW_COPY_AND_ASSIGN(NVTXRangeTracker);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_BACKENDS_GPU_NVTX_UTILS_H_
