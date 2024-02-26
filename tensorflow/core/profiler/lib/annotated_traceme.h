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

#include <optional>
#include <string>
#include <utility>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace profiler {

// Combination of TraceMe and ScopedAnnotation which share the same label.
// Optimization are done to ensure the label generation are done once.
class AnnotatedTraceMe {
 public:
  template <typename NameGeneratorT>
  explicit AnnotatedTraceMe(NameGeneratorT&& name_generator, int level = 1) {
    DCHECK_GE(level, 1);
    bool annotation_enabled = ScopedAnnotation::IsEnabled();
    bool traceme_enabled = TraceMe::Active(level);
    if (TF_PREDICT_TRUE(!annotation_enabled && !traceme_enabled)) {
      return;
    }
    std::string name = name_generator();
    if (annotation_enabled) {
      scoped_annotation_.emplace(name);
    }
    if (TF_PREDICT_TRUE(traceme_enabled)) {
      trace_me_.emplace([&name] { return std::move(name); }, level);
    }
  }

 private:
  std::optional<TraceMe> trace_me_;
  std::optional<ScopedAnnotation> scoped_annotation_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_ANNOTATED_TRACEME_H_
