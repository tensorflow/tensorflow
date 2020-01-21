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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_ANNOTATED_TRACEME_H_
#define TENSORFLOW_CORE_PROFILER_LIB_ANNOTATED_TRACEME_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace profiler {

// Combination of TraceMe and ScopedAnnotation which share the same label.
// Optimization are done to ensure the label generation are done once.
class AnnotatedTraceMe {
 public:
  template <typename NameGeneratorT>
  explicit AnnotatedTraceMe(NameGeneratorT name_generator, int level = 1) {
    DCHECK_GE(level, 1);
#if !defined(IS_MOBILE_PLATFORM)
    bool annotation_enabled = AnnotationStack::IsEnabled();
    bool traceme_enabled = TraceMeRecorder::Active(level);
    if (TF_PREDICT_FALSE(annotation_enabled || traceme_enabled)) {
      std::string label = name_generator();
      if (annotation_enabled) {
        scoped_annotation_.emplace(absl::string_view(label));
      }
      if (TF_PREDICT_TRUE(traceme_enabled)) {
        trace_me_.emplace(std::move(label), level);
      }
    }
#endif
  }

 private:
  absl::optional<TraceMe> trace_me_;
  absl::optional<ScopedAnnotation> scoped_annotation_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_ANNOTATED_TRACEME_H_
