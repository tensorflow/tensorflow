/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_SCOPED_COMMAND_BUFFER_ANNOTATION_H_
#define XLA_STREAM_EXECUTOR_GPU_SCOPED_COMMAND_BUFFER_ANNOTATION_H_

#include "absl/strings/string_view.h"

namespace stream_executor {

// A thread-local scope used to track the current HLO instruction being
// recorded into a command buffer.
//
// Example usage:
//   void RecordThunk(Thunk* thunk) {
//     // Push the thunk's annotation onto the thread-local stack.
//     ScopedCommandBufferAnnotation annotation(thunk->profile_annotation());
//
//     // Any command buffer node creation calls nested within this scope
//     // (even in lower-level libraries) can retrieve this annotation.
//     RecordCommands();
//   }
class ScopedCommandBufferAnnotation {
 public:
  // Pushes the given `annotation` onto the thread-local annotation stack.
  explicit ScopedCommandBufferAnnotation(absl::string_view annotation);

  // Pops the annotation from the thread-local annotation stack.
  ~ScopedCommandBufferAnnotation();

  // Returns the annotation at the top of the thread-local stack,
  // or empty string if the stack is empty.
  static absl::string_view GetCurrentAnnotation();

 private:
  ScopedCommandBufferAnnotation(const ScopedCommandBufferAnnotation&) = delete;
  ScopedCommandBufferAnnotation& operator=(
      const ScopedCommandBufferAnnotation&) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_SCOPED_COMMAND_BUFFER_ANNOTATION_H_
