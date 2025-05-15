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
#ifndef XLA_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
#define XLA_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_

#include <atomic>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace profiler {

// Backend for ScopedAnnotation.
class AnnotationStack {
 public:
  // Appends name to the annotations for the current thread, separated by "::".
  // The choice of separator "::" is based on characters not used by TensorFlow
  // for its TensorOps.
  static void PushAnnotation(absl::string_view name);

  // Resizes the annotation stack for the current thread.
  static void PopAnnotation();

  // Returns the annotation stack for the current thread.
  static const string& Get();

  // Returns the range id sequence for the stack for the current thread.
  static absl::Span<const int64_t> GetScopeRangeIds();

  // Enables or disables the annotation stack.
  static void Enable(bool enable);

  // Returns whether the annotation stack is enabled.
  static bool IsEnabled() {
    return generation_.load(std::memory_order_acquire) & 1;
  }

 private:
  AnnotationStack() = default;

  // Enabled if odd, disabled if even. The value is incremented for every call
  // to Enable() which changes the enabled state.
  static std::atomic<int> generation_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
