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

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/match.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/stack_frame.h"

namespace tensorflow {

// Returns "true" on filenames which should be skipped.
using StackTraceFilter = std::function<bool(const char*)>;

using SourceLoc = std::pair<std::string, int>;

// Using absl::Hash breaks NVCC under Windows :P
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    std::size_t h1 = std::hash<T1>()(pair.first);
    std::size_t h2 = std::hash<T2>()(pair.second);
    return h1 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
  }
};

// Maps filename/line_no combination into a stack frame.
using SourceMap = std::unordered_map<SourceLoc, StackFrame, PairHash>;

using ToStackFramesFunctor = std::vector<StackFrame>(int, const SourceMap&,
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
  std::vector<StackFrame> ToStackFrames(const SourceMap& source_map,
                                        const StackTraceFilter& filtered,
                                        bool reverse_traversal = false,
                                        int limit = -1) const {
    return to_stack_frames_(id_, source_map, filtered, reverse_traversal,
                            limit);
  }

 private:
  int id_;
  ToStackFramesFunctor* to_stack_frames_;
};

// Generates a message with a definition location based on a provided stack
// trace, or an empty one if the stack trace is empty.
inline std::string DefinitionLocationMsg(
    const absl::optional<ManagedStackTrace>& stack_trace) {
  if (stack_trace.has_value()) {
    std::vector<StackFrame> stack_frames =
        stack_trace->ToStackFrames({}, IsInternalFrameForFilename,
                                   /*reverse_traversal=*/true,
                                   /*limit=*/1);
    if (!stack_frames.empty()) {
      const StackFrame& last_frame = stack_frames[0];
      return absl::StrCat(" (defined @ ", last_frame.file_name, ":",
                          last_frame.line_number, ")");
    }
  }
  return "";
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
