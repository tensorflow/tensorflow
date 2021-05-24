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
#include <vector>

#include "absl/strings/match.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/stack_frame.h"

namespace tensorflow {

// Maps filename/line_no combination into a stack frame.
using StackTraceMap =
    std::function<absl::optional<StackFrame>(std::pair<const char*, int>)>;

// Returns "true" on filenames which should be skipped.
using StackTraceFilter = std::function<bool(const char*)>;

using ToStackFramesFunctor = std::vector<StackFrame>(int, const StackTraceMap&,
                                                     const StackTraceFilter&,
                                                     bool, int);

// Returns whether the given frame is internal to TF.
inline bool IsInternalFrameForFilename(absl::string_view file_name) {
  // Use a simple heuristic for now.
  // TODO(cheshire): Build a more sophisticated mechanism, rely on @tf.export.
  return (absl::StrContains(file_name, "tensorflow/python") ||
          absl::StrContains(file_name, "tensorflow\\python")) &&
         !absl::StrContains(file_name, "keras") &&
         !absl::StrContains(file_name, "test.py");
}

// Language agnostic stack trace class. It only saves an id, and language
// clients are responsible for managing the actual stack trace objects.
class ManagedStackTrace {
 public:
  ManagedStackTrace(int id, ToStackFramesFunctor* to_stack_frames)
      : id_(id), to_stack_frames_(to_stack_frames) {}

  // Returns stack trace as a vector of `StackFrame`s.
  std::vector<StackFrame> ToStackFrames(const StackTraceMap& mapper = {},
                                        const StackTraceFilter& filtered = {},
                                        bool reverse_traversal = false,
                                        int limit = -1) const {
    return to_stack_frames_(id_, mapper, filtered, reverse_traversal, limit);
  }

 private:
  int id_;
  ToStackFramesFunctor* to_stack_frames_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
