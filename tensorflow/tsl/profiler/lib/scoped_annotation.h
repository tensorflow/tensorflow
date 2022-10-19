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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_H_
#define TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_H_

#include <stddef.h>

#include <atomic>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/types.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/tsl/profiler/backends/cpu/annotation_stack.h"
#endif

namespace tsl {
namespace profiler {

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
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(name);
    }
#endif
  }

  explicit ScopedAnnotation(const char* name)
      : ScopedAnnotation(absl::string_view(name)) {}

  explicit ScopedAnnotation(const string& name) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(name);
    }
#endif
  }

  explicit ScopedAnnotation(string&& name) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(std::move(name));
    }
#endif
  }

  template <typename NameGeneratorT>
  explicit ScopedAnnotation(NameGeneratorT name_generator) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      old_length_ = AnnotationStack::PushAnnotation(name_generator());
    }
#endif
  }

  // Pops the name passed in the constructor from the current annotation.
  ~ScopedAnnotation() {
    // TODO(b/137971921): without this memory fence, two presubmit tests will
    // fail probably due to compiler in that presubmit config.
    std::atomic_thread_fence(std::memory_order_acquire);
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(old_length_ != kInvalidLength)) {
      AnnotationStack::PopAnnotation(old_length_);
    }
#endif
  }

  static bool IsEnabled() {
#if !defined(IS_MOBILE_PLATFORM)
    return AnnotationStack::IsEnabled();
#else
    return false;
#endif
  }

 private:
  // signals that annotation is disabled at the constructor.
  static constexpr size_t kInvalidLength = static_cast<size_t>(-1);

  TF_DISALLOW_COPY_AND_ASSIGN(ScopedAnnotation);

  size_t old_length_ = kInvalidLength;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_H_
