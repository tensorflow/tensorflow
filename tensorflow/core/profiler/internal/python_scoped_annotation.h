/* Copyright 2019 The TensorFlow Authors All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_PYTHON_SCOPED_ANNOTATION_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_PYTHON_SCOPED_ANNOTATION_H_

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/annotation.h"

namespace tensorflow {
namespace profiler {

// DO NOT USE THIS CLASS DIRECTLY IN C++ CODE.
// This class is only used to implement ScopedAnnotation
// as a python context manager.
class PythonScopedAnnotation {
 public:
  explicit PythonScopedAnnotation(const std::string& name) : name_(name) {}

  void Enter() { current_.emplace(std::move(name_)); }
  void Exit() { current_.reset(); }

  static bool IsEnabled() { return tracing::ScopedAnnotation::IsEnabled(); }

 private:
  std::string name_;
  absl::optional<tracing::ScopedAnnotation> current_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_PYTHON_SCOPED_ANNOTATION_H_
