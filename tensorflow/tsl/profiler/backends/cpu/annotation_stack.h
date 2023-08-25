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
#ifndef TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
#define TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_

#include <stddef.h>

#include <atomic>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/types.h"

namespace tsl {
namespace profiler {
namespace internal {

// Whether annotations are enabled.
// Static atomic so Annotation::IsEnabled can be fast and non-blocking.
TF_EXPORT extern std::atomic<int> g_annotation_enabled;

}  // namespace internal

// Backend for ScopedAnnotation.
class AnnotationStack {
 public:
  // Appends name to the annotation for the current thread and returns the
  // original length of the annotation.
  // Append name to the current annotation, separated by "::".
  // The choice of separator "::" is based on characters not used by
  // TensorFlow for its TensorOps.
  static size_t PushAnnotation(absl::string_view name) {
    string* annotation_stack = ThreadAnnotationStack();
    size_t old_length = annotation_stack->size();
    if (old_length != 0) {
      absl::StrAppend(annotation_stack, "::", name);
    } else {
      *annotation_stack = string(name);
    }
    return old_length;
  }

  static size_t PushAnnotation(string&& name) {
    string* annotation_stack = ThreadAnnotationStack();
    size_t old_length = annotation_stack->size();
    if (old_length != 0) {
      absl::StrAppend(annotation_stack, "::", name);
    } else {
      *annotation_stack = std::move(name);
    }
    return old_length;
  }

  // Returns the annotation stack for the current thread.
  static const string& Get() { return *ThreadAnnotationStack(); }

  // Resizes the annotation stack for the current thread to its old length.
  static void PopAnnotation(size_t old_length) {
    ThreadAnnotationStack()->resize(old_length);
  }

  static void Enable(bool enable) {
    internal::g_annotation_enabled.store(enable, std::memory_order_release);
  }

  static bool IsEnabled() {
    return internal::g_annotation_enabled.load(std::memory_order_acquire);
  }

 private:
  AnnotationStack() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(AnnotationStack);

  // Returns a reference to the annotation for the current thread.
  static string* ThreadAnnotationStack();
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_BACKENDS_CPU_ANNOTATION_STACK_H_
