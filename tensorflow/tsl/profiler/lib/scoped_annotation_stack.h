/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_STACK_H_
#define TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_STACK_H_

#include <stddef.h>

#include <atomic>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/string_view.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tensorflow/tsl/profiler/lib/nvtx_utils.h"
#endif

namespace tsl {
namespace profiler {

// ScopedAnnotation for clients that can't use RAII for managing the lifetime
// of annotations. It provides an API similar to the `TraceMe::ActivityStart`
// and `TraceMe::ActivityEnd`.
//
// Usage:
//   int64_t id = ScopedAnnotationStack::ActivityStart("foo");
//   foo();
//   ScopedAnnotationStack::ActivityEnd(id);
//
// Prefer a regular `ScopedAnnotation`. The name of this class is a misnomer,
// because it doesn't do any automatic destruction at the scope end, it's just
// for the sake of consistency.
class ScopedAnnotationStack {
  static constexpr size_t kInvalidActivity = static_cast<size_t>(-1);

 public:
  static int64_t ActivityStart(std::string name) {
#if !defined(IS_MOBILE_PLATFORM)
#if GOOGLE_CUDA
    std::optional<nvtxDomainHandle_t> domain =
        tsl::profiler::nvtx::GetNVTXDomain();
    if (TF_PREDICT_FALSE(domain.has_value())) {
      nvtxEventAttributes_t attrs;
      std::string name_str(name);
      tsl::profiler::nvtx::MakeAttributes(name_str.c_str(), &attrs);
      ::nvtxDomainRangePushEx(domain.value(), &attrs);
    } else  // NOLINT
#endif
        if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      return AnnotationStack::PushAnnotation(std::move(name));
    }
#endif
    return kInvalidActivity;
  }

  static int64_t ActivityStart(std::string_view name) {
    return ActivityStart(std::string(name));
  }

  static int64_t ActivityStart(const char* name) {
    return ActivityStart(std::string_view(name));
  }

  template <typename NameGeneratorT>
  static int64_t ActivityStart(NameGeneratorT name_generator) {
#if !defined(IS_MOBILE_PLATFORM)
#if GOOGLE_CUDA
    std::optional<nvtxDomainHandle_t> domain =
        tsl::profiler::nvtx::GetNVTXDomain();
    if (TF_PREDICT_FALSE(domain.has_value())) {
      auto name = name_generator();
      nvtxEventAttributes_t attrs;
      std::string name_str(name);
      tsl::profiler::nvtx::MakeAttributes(name_str.c_str(), &attrs);
      ::nvtxDomainRangePushEx(domain.value(), &attrs);
    } else  // NOLINT
#endif
        if (TF_PREDICT_FALSE(AnnotationStack::IsEnabled())) {
      return AnnotationStack::PushAnnotation(name_generator());
    }
#endif
    return kInvalidActivity;
  }

  static void ActivityEnd(int64_t activity_id) {
#if !defined(IS_MOBILE_PLATFORM)
#if GOOGLE_CUDA
    std::optional<nvtxDomainHandle_t> domain =
        tsl::profiler::nvtx::GetNVTXDomain();
    if (TF_PREDICT_FALSE(domain.has_value())) {
      ::nvtxDomainRangePop(domain.value());
    } else  // NOLINT
#endif
        if (TF_PREDICT_FALSE(activity_id != kInvalidActivity)) {
      AnnotationStack::PopAnnotation(activity_id);
    }
#endif
  }
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_SCOPED_ANNOTATION_STACK_H_
