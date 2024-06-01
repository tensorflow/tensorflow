/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_WHILE_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_WHILE_THUNK_H_

#include "absl/status/status.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"

namespace xla::cpu {

// While loop written as two thunk sequences:
//
// while (condition_thunk.Execute(...) && condition_buffer) {
//   body_thunk.Execute(...);
// }
//
// Condition buffer must be a i1 (bool) buffer that holds a loop predicate.
class WhileThunk final : public Thunk {
 public:
  WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
             ThunkSequence cond_sequence, ThunkSequence body_sequence);

  absl::Status Execute(const ExecuteParams& params) final;

 private:
  BufferAllocation::Slice cond_buffer_;
  ThunkSequence cond_sequence_;
  ThunkSequence body_sequence_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_WHILE_THUNK_H_
