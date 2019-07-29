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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_PYTHON_TRACEME_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_PYTHON_TRACEME_H_

#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace profiler {

// DO NOT USE THIS CLASS DIRECTLY IN C++ CODE.
// This class is only used to implement TraceMe as a python context manager.
class PythonTraceMe {
 public:
  explicit PythonTraceMe(const std::string& name) : activity_name_(name) {}
  void Enter() { current_.emplace(std::move(activity_name_)); }
  void Exit() { current_.reset(); }

 private:
  std::string activity_name_;
  absl::optional<TraceMe> current_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_PYTHON_TRACEME_H_
