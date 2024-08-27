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

#include "xla/service/gpu/execution_stream_assignment.h"

#include <deque>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/runtime/thunk.h"

namespace xla::gpu {

ExecutionStreamAssignment::ExecutionStreamAssignment(
    const HloModule* module, ExecutionStreamAssignmentOptions options) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  // We'll walk the `CallGraph` starting from the entrypoint. The instructions
  // on the entrypoint computation will be assigned `ExecutionStreamId(0)`, and
  // each invocation of `async-start` will result in the target computation
  // being assigned a new `ExecutionStreamId`.
  ExecutionStreamId next_stream_id = ExecutionStreamId(1);

  // Each `Pending` item represents an `HloComputation` that needs to be
  // processed. We start with the entrypoint and add callees as we discover
  // them.
  struct Pending {
    Pending(HloComputation* node, ExecutionStreamId stream_id)
        : node(node), stream_id(stream_id) {}
    HloComputation* node;
    ExecutionStreamId stream_id;
  };
  std::deque<Pending> queue;
  queue.emplace_back(module->entry_computation(), ExecutionStreamId(0));

  // Enqueues called computations of a given `callsite` unless the callees are
  // only invoked in an embedded context, in which case children nodes will all
  // be executed in a single kernel.
  auto enqueue_called_computations = [&](const CallSite& callsite,
                                         ExecutionStreamId stream) {
    if (GetInstructionCallContext(callsite.instruction()->opcode()) ==
        CallContext::kEmbedded) {
      return;
    }
    for (HloComputation* computation : callsite.called_computations()) {
      queue.emplace_back(computation, stream);
    }
  };

  // Assigns source and destination streams to an instruction and records it in
  // async_instructions_.
  auto assign_async_execution_streams =
      [&](HloInstruction* instruction, ExecutionStreamId source_stream_id) {
        AsyncExecutionStreamIds streams;
        streams.source_stream_id = source_stream_id;
        streams.destination_stream_id = next_stream_id;

        CHECK(async_instructions_.try_emplace(instruction, streams).second);

        next_stream_id++;
        if (next_stream_id.value() > options.number_of_execution_streams) {
          next_stream_id = ExecutionStreamId(1);
        }
      };

  while (!queue.empty()) {
    Pending pending = queue.front();
    queue.pop_front();

    // First, we'll assign the current `ExecutionStreamId` to all synchronous
    // instructions. Asynchronous instructions will be handled afterwards.
    for (HloInstruction* instruction : pending.node->instructions()) {
      if (instruction->IsAsynchronous()) continue;
      if (instruction->opcode() == HloOpcode::kCopyStart) {
        // CopyStart is morally an async instruction, let us treat it
        // as an async instruction.
        assign_async_execution_streams(instruction, pending.stream_id);
      } else {
        CHECK(sync_instructions_.try_emplace(instruction, pending.stream_id)
                  .second);
      }
    }

    // Next, we'll process all callsites in the current computation.
    for (const CallSite& callsite :
         call_graph->GetNode(pending.node).callsites()) {
      if (callsite.instruction()->IsAsynchronous()) {
        // Asynchronous calls will result in a new `ExecutionStreamId` being
        // dispensed for the called computations.
        CHECK_EQ(callsite.instruction()->opcode(), HloOpcode::kAsyncStart);
        enqueue_called_computations(callsite, next_stream_id);
        assign_async_execution_streams(callsite.instruction(),
                                       pending.stream_id);
      } else {
        // Synchronous calls will result in the called computations being
        // invoked using the same `ExecutionStreamId`.
        enqueue_called_computations(callsite, pending.stream_id);
      }
    }

    // And finally, we need to assign `ExecutionStreamIds` to all asynchronous
    // instructions that are were not handled by the iteration over callsites
    // above. These are the `async-updates` and `async-dones`. Both of these
    // should share the `ExecutionStreamId` with the originating `async-starts`.
    for (HloInstruction* instruction : pending.node->instructions()) {
      if (!instruction->IsAsynchronous()) continue;
      if (instruction->opcode() == HloOpcode::kAsyncStart) {
        CHECK(async_instructions_.find(instruction) !=
              async_instructions_.end());
      } else {
        HloInstruction* async_start =
            Cast<HloAsyncInstruction>(instruction)->async_chain_start();
        AsyncExecutionStreamIds async_start_streams =
            async_instructions_.at(async_start);
        CHECK(async_instructions_.try_emplace(instruction, async_start_streams)
                  .second);
      }
    }
  }
}

namespace {
absl::Status StreamNotFoundError(const HloInstruction* instruction) {
  return absl::NotFoundError(absl::StrCat(
      "No ExecutionStreamId found for ", instruction->ToString(),
      "; this may happen if the Computation is not reachable from the module's "
      "entrypoint, or if it's only reachable through a embedded calls."));
}
}  // namespace

absl::StatusOr<ExecutionStreamId>
ExecutionStreamAssignment::GetSyncExecutionStreamId(
    const HloInstruction* instruction) const {
  CHECK(!instruction->IsAsynchronous());
  auto stream = sync_instructions_.find(instruction);
  if (stream == sync_instructions_.end()) {
    return StreamNotFoundError(instruction);
  }
  return stream->second;
}

absl::StatusOr<ExecutionStreamAssignment::AsyncExecutionStreamIds>
ExecutionStreamAssignment::GetAsyncExecutionStreamIds(
    const HloInstruction* instruction) const {
  CHECK(instruction->IsAsynchronous() ||
        instruction->opcode() == HloOpcode::kCopyStart);
  auto streams = async_instructions_.find(instruction);
  if (streams == async_instructions_.end()) {
    return StreamNotFoundError(instruction);
  }
  return streams->second;
}

}  // namespace xla::gpu
