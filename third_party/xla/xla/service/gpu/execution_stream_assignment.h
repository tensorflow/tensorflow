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
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/runtime/thunk.h"

namespace xla::gpu {

struct ExecutionStreamAssignmentOptions {
  // The `ExecutionStreamAssignment` will round-robin across this many
  // `ExecutionStreams`.
  int number_of_execution_streams = 4;
};

// `ExecutionStreamAssignments` represent a mapping from `HloInstructions` to
// `ExecutionStreamIds`. Asynchronous calls (`async-start`, `async-update`, and
// `async-done`) result in the target computations being assigned new
// `ExecutionStreamIds` to support concurrent execution.
class ExecutionStreamAssignment {
 public:
  // The `HloModule` must be flat. In other words, there must be a one-to-one
  // mapping between callsites and computations. One way to guarantee this is to
  // pass the module through the `FlattenCallGraph` pass.
  //
  // The ExecutionStreamAssignment does not take ownership of the `HloModule`.
  explicit ExecutionStreamAssignment(
      const HloModule* module, ExecutionStreamAssignmentOptions options = {});

  // Returns the `ExecutionStreamId` for the given instruction, which *must* be
  // synchronous. Returns an error if the instruction is either not reachable
  // from the module's entrypoint, or is only reachable through embedded calls.
  absl::StatusOr<ExecutionStreamId> GetSyncExecutionStreamId(
      const HloInstruction* instruction) const;

  // Returns the source and destination `ExecutionStreamIds` for the given
  // instruction, which *must* be asynchronous. Returns an error if the
  // instruction is either not reachable from the module's entrypoint, or is
  // only reachable through embedded calls.
  struct AsyncExecutionStreamIds {
    // The `ExecutionStreamId` for the calling instruction (e.g. the computation
    // that invokes `async-start`).
    ExecutionStreamId source_stream_id;
    // The `ExecutionStreamId` for the callee computation (e.g. the callee of an
    // `async-start` instruction).
    ExecutionStreamId destination_stream_id;
  };
  absl::StatusOr<AsyncExecutionStreamIds> GetAsyncExecutionStreamIds(
      const HloInstruction* instruction) const;

 private:
  // Maps from `HloInstructions` to `ExecutionStreamIds` for synchronous and
  // asynchronous instructions, respectively. All instructions reachable through
  // non-embedded calls must be present.
  absl::flat_hash_map<HloInstruction*, ExecutionStreamId> sync_instructions_;
  absl::flat_hash_map<HloInstruction*, AsyncExecutionStreamIds>
      async_instructions_;
};

inline bool operator==(
    const ExecutionStreamAssignment::AsyncExecutionStreamIds& first,
    const ExecutionStreamAssignment::AsyncExecutionStreamIds& second) {
  return first.source_stream_id == second.source_stream_id &&
         first.destination_stream_id == second.destination_stream_id;
}

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_EXECUTION_STREAM_ASSIGNMENT_H_
