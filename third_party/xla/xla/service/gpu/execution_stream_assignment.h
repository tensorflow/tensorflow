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

#ifndef XLA_SERVICE_GPU_EXECUTION_STREAM_ASSIGNMENT_H_
#define XLA_SERVICE_GPU_EXECUTION_STREAM_ASSIGNMENT_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::gpu {

// `ExecutionStreamAssignment` maps async scope-start `HloInstructions` to
// `ExecutionStreamIds`. Only operations that start a new execution scope (e.g.
// async-start, all-reduce-start, send, recv, copy-start) are assigned an
// `ExecutionStreamId`. Operations inside an execution scope inherit their
// stream from the parent `AsyncStartThunk` at run time (structured
// concurrency).
class ExecutionStreamAssignment {
 public:
  struct Options {
    // Number of additional compute execution streams for round-robin
    // assignment of async compute operations.
    int number_of_compute_execution_streams = 4;
    // Number of additional communication execution streams for round-robin
    // assignment of async communication operations.
    int number_of_communication_execution_streams = 1;
  };

  // The `HloModule` must be flat. In other words, there must be a one-to-one
  // mapping between callsites and computations. One way to guarantee this is to
  // pass the module through the `FlattenCallGraph` pass.
  //
  // The ExecutionStreamAssignment does not take ownership of the `HloModule`.
  ExecutionStreamAssignment(const HloModule* module, const Options& options);

  explicit ExecutionStreamAssignment(const HloModule* module)
      : ExecutionStreamAssignment(module, Options{}) {}

  // Returns the `ExecutionStreamId` for the given instruction, which must be
  // an async scope-start operation. Returns an error if the instruction is not
  // a scope-start or is not reachable from the module's entrypoint.
  absl::StatusOr<ExecutionStreamId> GetExecutionStreamId(
      const HloInstruction* instruction) const;

 private:
  absl::flat_hash_map<const HloInstruction*, ExecutionStreamId>
      async_start_instructions_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_EXECUTION_STREAM_ASSIGNMENT_H_
