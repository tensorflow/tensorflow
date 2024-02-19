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

#include "tsl/platform/macros.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tsl/profiler/backends/cpu/annotation_stack.h"
#endif

#if GOOGLE_CUDA
#include "tsl/profiler/lib/nvtx_utils.h"
#endif

namespace tsl {
namespace profiler {

// Adds an annotation to all activities through the currently registered
// TraceCollector until PopAnnotation() is called.
template <typename T>
inline void PushAnnotation(const T& generator) {
#if GOOGLE_CUDA
  if (auto domain = GetNVTXDomain(); TF_PREDICT_FALSE(domain.has_value())) {
    return RangePush(*domain, generator());
  }
#endif

#if !defined(IS_MOBILE_PLATFORM)
  if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
    AnnotationStack::PushAnnotation(static_cast<std::string_view>(generator()));
  }
#endif
}

inline void PushAnnotation(const char* name) {
  PushAnnotation([&] { return name; });
}
inline void PushAnnotation(const std::string& name) {
  PushAnnotation([&] { return name; });
}

inline void PopAnnotation() {
  // TODO(b/137971921): without this memory fence, two presubmit tests will
  // fail probably due to compiler in that presubmit config.
  std::atomic_thread_fence(std::memory_order_acquire);

#if GOOGLE_CUDA
  if (auto domain = GetNVTXDomain(); TF_PREDICT_FALSE(domain.has_value())) {
    ::nvtxDomainRangePop(*domain);
    return;
  }
#endif

#if !defined(IS_MOBILE_PLATFORM)
  if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
    AnnotationStack::PopAnnotation();
  }
#endif
}

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
  template <typename T>
  explicit ScopedAnnotation(T&& annotation) {
    PushAnnotation(std::forward<T>(annotation));
  }

  // Pops the name passed in the constructor from the current annotation.
  ~ScopedAnnotation() { PopAnnotation(); }

  static bool IsEnabled() {
#if !defined(IS_MOBILE_PLATFORM)
    return AnnotationStack::IsEnabled();
#else
    return false;
#endif
  }

 private:
  ScopedAnnotation(const ScopedAnnotation&) = delete;
  ScopedAnnotation& operator=(const ScopedAnnotation&) = delete;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_H_
