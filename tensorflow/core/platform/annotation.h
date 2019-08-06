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
#ifndef TENSORFLOW_CORE_PLATFORM_ANNOTATION_H_
#define TENSORFLOW_CORE_PLATFORM_ANNOTATION_H_

#include <stddef.h>

#include <atomic>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// Backend for ScopedAnnotation.
class Annotation {
 public:
  // Appends name to the annotation for the current thread and returns the
  // original length of the annotation.
  // Append name to the current annotation, separated by "::".
  // The choice of separator "::" is based on characters not used by
  // TensorFlow for its TensorOps.
  static size_t PushAnnotation(absl::string_view name) {
    std::string* annotation = ThreadAnnotation();
    size_t old_length = annotation->size();
    if (old_length != 0) {
      absl::StrAppend(annotation, "::", name);
    } else {
      *annotation = std::string(name);
    }
    return old_length;
  }

  static size_t PushAnnotation(std::string&& name) {
    std::string* annotation = ThreadAnnotation();
    size_t old_length = annotation->size();
    if (old_length != 0) {
      absl::StrAppend(annotation, "::", name);
    } else {
      *annotation = std::move(name);
    }
    return old_length;
  }

  // Returns the annotation for the current thread.
  static const std::string& CurrentAnnotation() { return *ThreadAnnotation(); }

  // Resizes the annotation for the current thread to its old length.
  static void PopAnnotation(size_t old_length) {
    ThreadAnnotation()->resize(old_length);
  }

 private:
  Annotation(const Annotation&) = delete;  // Unconstructible.

  // Returns a reference to the annotation for the current thread.
  static std::string* ThreadAnnotation() {
    static thread_local std::string annotation;
    return &annotation;
  }
};

namespace tracing {
// Adds an annotation to all activities for the duration of the instance
// lifetime through the currently registered TraceCollector.
//
// Usage: {
//          ScopedAnnotation annotation("my kernels");
//          Kernel1<<<x,y>>>;
//          LaunchKernel2(); // Launches a CUDA kernel.
//        }
// This will add 'my kernels' to both kernels in the profiler UI
class ScopedAnnotation {
 public:
  explicit ScopedAnnotation(absl::string_view name) {
    if (TF_PREDICT_FALSE(IsEnabled())) {
      old_length_ = Annotation::PushAnnotation(name);
    }
  }

  explicit ScopedAnnotation(const char* name)
      : ScopedAnnotation(absl::string_view(name)) {}

  explicit ScopedAnnotation(const std::string& name) {
    if (TF_PREDICT_FALSE(IsEnabled())) {
      old_length_ = Annotation::PushAnnotation(name);
    }
  }

  explicit ScopedAnnotation(std::string&& name) {
    if (TF_PREDICT_FALSE(IsEnabled())) {
      old_length_ = Annotation::PushAnnotation(std::move(name));
    }
  }

  template <typename NameGeneratorT>
  explicit ScopedAnnotation(NameGeneratorT name_generator) {
    if (TF_PREDICT_FALSE(IsEnabled())) {
      old_length_ = Annotation::PushAnnotation(name_generator());
    }
  }

  // Pops the name passed in the constructor from the current annotation.
  ~ScopedAnnotation() {
    // TODO(b/137971921): without this memory fence, two presubmit tests will
    // fail probably due to compiler in that presubmit config.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (TF_PREDICT_FALSE(old_length_ != kInvalidLength)) {
      Annotation::PopAnnotation(old_length_);
    }
  }

  static void Enable(bool enable);
  static const bool IsEnabled();

 private:
  // signals that annotation is disabled at the constructor.
  static constexpr size_t kInvalidLength = static_cast<size_t>(-1);
  size_t old_length_ = kInvalidLength;
};

}  // namespace tracing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ANNOTATION_H_
