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

#ifndef TENSORFLOW_COMPILER_XLA_BACKENDS_PROFILER_GPU_NVTX_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_BACKENDS_PROFILER_GPU_NVTX_UTILS_H_

#include <stack>

#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/default/logging.h"
#include "nvtx3/nvToolsExt.h"

namespace xla {
namespace profiler {

/***
 * We use this class to track NVTX instrumentation inside NVIDIA
 * libraries (such as TensorRT).  This bears a lot of resemblance to
 * ScopedAnnotation for now.  In the future, we will use TraceMe to
 * keep track trace context within a thread.
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

namespace nvtx {

// A helper function that return the domains to use if NVTX profiling
// is enabled.
inline std::optional<nvtxDomainHandle_t> GetNVTXDomain() {
  static nvtxDomainHandle_t domain;
  static bool is_enabled = [] {
    bool _is_enabled = false;
    // Force NVTX marker if a tool triggered the profiler.
    domain = nvtxDomainCreateA("TSL");
    if (domain) {
      _is_enabled = true;
    }
    VLOG(1) << "Is NVTX marker enabled? " << _is_enabled;
    return _is_enabled;
  }();
  if (is_enabled)
    return domain;
  return {};
}

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
inline bool RangesEnabled() {
  return GetNVTXDomain().has_value();
}

// Note: The memory backing msg must persist until the result of this function
// has been consumed by an NVTX API.
void MakeAttributes(const char* msg, nvtxEventAttributes_t* result);
}  // namespace nvtx
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_BACKENDS_PROFILER_GPU_NVTX_UTILS_H_
