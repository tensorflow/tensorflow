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

#ifndef TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
#define TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_

#include <string>

#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Language agnostic stack trace class. It only saves an id, and language
// clients are responsible for managing the actual stack trace objects.
class AbstractStackTrace {
 public:
  AbstractStackTrace(int id, std::vector<StackFrame> (*to_stack_frames)(int))
      : id_(id), to_stack_frames_(to_stack_frames) {}

  // Returns stack trace as a vector of `StackFrame`s.
  std::vector<StackFrame> ToStackFrames() const {
    return to_stack_frames_(id_);
  }

 private:
  int id_;
  std::vector<StackFrame> (*to_stack_frames_)(int);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
