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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/side_effect_util.h"

namespace xla::gpu {
namespace {

// There are two kinds of async execution scopes: compute and collective. We
// need just two as our goal is to effectively overlap computation with
// communication.
enum class ExecutionScopeKind { kCompute, kCollective };

template <typename Sink>
void AbslStringify(Sink sink, ExecutionScopeKind kind) {
  switch (kind) {
    case ExecutionScopeKind::kCompute:
      sink.Append("compute");
      break;
    case ExecutionScopeKind::kCollective:
      sink.Append("collective");
      break;
  }
}

// A helper class to generate the next execution stream id. We assign ids
// using round-robin assignment for two execution scope kinds.
class ExecutionStreams {
 public:
  explicit ExecutionStreams(const ExecutionStreamAssignment::Options& opts)
      : opts_(opts),
        compute_id_(1),
        collective_id_(1 + opts_.number_of_compute_execution_streams) {}

  ExecutionStreamId Next(ExecutionScopeKind kind) {
    switch (kind) {
      case ExecutionScopeKind::kCompute: {
        ExecutionStreamId id(compute_id_++);
        if (compute_id_ > opts_.number_of_compute_execution_streams) {
          compute_id_ = 1;
        }
        return id;
      }
      case ExecutionScopeKind::kCollective: {
        ExecutionStreamId id(collective_id_++);
        if (collective_id_ > (opts_.number_of_compute_execution_streams +
                              opts_.number_of_collective_execution_streams)) {
          collective_id_ = 1 + opts_.number_of_compute_execution_streams;
        }
        return id;
      }
    }
  }

 private:
  ExecutionStreamAssignment::Options opts_;
  size_t compute_id_;
  size_t collective_id_;
};

// Returns true if async instruction wraps a collective operation.
bool IsWrappedCollective(const HloAsyncInstruction* async) {
  switch (async->async_wrapped_opcode()) {
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReduceScatter:
      return true;
    default:
      return false;
  }
}

// Returns a computation that corresponds to HLO operation execution scope.
const HloComputation* ExecutionScopeComputation(const HloInstruction* hlo) {
  if (auto* async = DynCast<HloAsyncInstruction>(hlo)) {
    return async->async_wrapped_computation();
  };
  return nullptr;
}

// Returns an execution scope if operations starts it. Null optional otherwise.
std::optional<ExecutionScopeKind> IsExecutionScopeStart(
    const HloInstruction* hlo) {
  // Async operation that starts a new execution scope.
  if (auto* start = DynCast<HloAsyncStartInstruction>(hlo)) {
    return IsWrappedCollective(start) ? ExecutionScopeKind::kCollective
                                      : ExecutionScopeKind::kCompute;
  }

  // Async-collective operations not yet migrated to async wrappers.
  if (HloPredicateIsOp<HloOpcode::kAllGatherStart, HloOpcode::kAllReduceStart,
                       HloOpcode::kCollectivePermuteStart>(hlo)) {
    return ExecutionScopeKind::kCollective;
  }

  // Send/Recv operations are the only ones that can be partially pipelined and
  // require a special handling. If this send/recv is not a canonical start,
  // it must be an execution scope use operation.
  if (HloPredicateIsOp<HloOpcode::kRecv, HloOpcode::kSend>(hlo)) {
    return hlo == FindCanonicalSendRecvStartOp(hlo)
               ? std::make_optional(ExecutionScopeKind::kCollective)
               : std::nullopt;
  }

  // A special case of asynchronous compute operation.
  if (HloPredicateIsOp<HloOpcode::kCopyStart>(hlo)) {
    return ExecutionScopeKind::kCompute;
  }

  return std::nullopt;
}

// Returns an execution scope if operations uses it. Null optional otherwise.
std::optional<ExecutionScopeKind> IsExecutionScopeUse(
    const HloInstruction* hlo) {
  if (HloPredicateIsOp<HloOpcode::kAsyncUpdate>(hlo)) {
    auto* update = Cast<HloAsyncInstruction>(hlo);
    return IsWrappedCollective(update) ? ExecutionScopeKind::kCollective
                                       : ExecutionScopeKind::kCompute;
  }

  // A special case for partially pipelined send/recv operations. If this is
  // a non-canonical send/recv inside a loop, we treat it as scope use.
  if (HloPredicateIsOp<HloOpcode::kRecv, HloOpcode::kSend>(hlo)) {
    return hlo != FindCanonicalSendRecvStartOp(hlo)
               ? std::make_optional(ExecutionScopeKind::kCollective)
               : std::nullopt;
  }

  return std::nullopt;
}

// Returns an execution scope if operations ends it. Null optional otherwise.
std::optional<ExecutionScopeKind> IsExecutionScopeEnd(
    const HloInstruction* hlo) {
  // Async operation that ends the execution scope.
  if (HloPredicateIsOp<HloOpcode::kAsyncDone>(hlo)) {
    auto* done = Cast<HloAsyncInstruction>(hlo);
    return IsWrappedCollective(done) ? ExecutionScopeKind::kCollective
                                     : ExecutionScopeKind::kCompute;
  }

  // Async-collective operations not yet migrated to async wrappers.
  if (HloPredicateIsOp<HloOpcode::kAllGatherDone, HloOpcode::kAllReduceDone,
                       HloOpcode::kCollectivePermuteDone>(hlo)) {
    return ExecutionScopeKind::kCollective;
  }

  // We always treat send/recv done operations as execution scope end. Strictly
  // speaking pipelined send/recv can be an execution scope use, but today we
  // don't care about that as it doesn't impact stream assignment.
  if (HloPredicateIsOp<HloOpcode::kRecvDone, HloOpcode::kSendDone>(hlo)) {
    return ExecutionScopeKind::kCollective;
  }

  // A special case of asynchronous compute operation.
  if (HloPredicateIsOp<HloOpcode::kCopyDone>(hlo)) {
    return ExecutionScopeKind::kCompute;
  }

  return std::nullopt;
}

// Find HLO operation that starts an execution scope.
const HloInstruction* FindExecutionScopeStart(const HloInstruction* hlo) {
  DCHECK(IsExecutionScopeUse(hlo) || IsExecutionScopeEnd(hlo));

  if (auto* async = DynCast<HloAsyncInstruction>(hlo)) {
    return async->async_chain_start();
  }

  // A special-case for partially pipelined send/recv operations.
  if (HloPredicateIsOp<HloOpcode::kRecv, HloOpcode::kSend, HloOpcode::kRecvDone,
                       HloOpcode::kSendDone>(hlo)) {
    return FindCanonicalSendRecvStartOp(hlo);
  }

  DCHECK((
      HloPredicateIsOp<HloOpcode::kAllGatherDone, HloOpcode::kAllReduceDone,
                       HloOpcode::kCollectivePermuteDone, HloOpcode::kCopyDone>(
          hlo)))
      << "Unsupported async operation: " << hlo->name();

  return hlo->operand(0);
}

// Check if instruction has explicit stream assignment via the attributes.
std::optional<ExecutionStreamId> FindAssignedStreamId(
    const HloInstruction* instr) {
  auto& attrs = instr->frontend_attributes().map();
  if (auto it = attrs.find(kXlaStreamAnnotationAttr); it != attrs.end()) {
    int32_t assigned_stream_id;
    CHECK(absl::SimpleAtoi(it->second, &assigned_stream_id));  // Crash OK
    return ExecutionStreamId(assigned_stream_id);
  }
  return std::nullopt;
}

// Each `PendingComputation` item represents an `HloComputation` that needs to
// be processed. We start with the ENTRY computation and follow the call graph.
struct PendingComputation {
  const HloComputation* computation;
  ExecutionStreamId stream_id;
};

}  // namespace

ExecutionStreamAssignment::ExecutionStreamAssignment(const HloModule* module,
                                                     const Options& options) {
  VLOG(1) << absl::StreamFormat(
      "Assign execution streams to module %s: #compute_streams=%d "
      "#collective_streams=%d",
      module->name(), options.number_of_compute_execution_streams,
      options.number_of_collective_execution_streams);
  ExecutionStreams execution_streams(options);

  std::deque<PendingComputation> queue;
  queue.push_back({module->entry_computation(), ExecutionStreamId(0)});

  while (!queue.empty()) {
    PendingComputation pending = queue.front();
    queue.pop_front();

    VLOG(2) << "Assign execution streams to computation: "
            << pending.computation->name();
    ExecutionStreamId parent_stream_id = pending.stream_id;

    // Process instructions in the post order, because we traverse the use-def
    // chain to find execution stream assignment and must process `start`
    // operations before `done`.
    std::vector<HloInstruction*> instructions =
        pending.computation->MakeInstructionPostOrder();

    for (const HloInstruction* hlo : instructions) {
      // If operation starts an async execution scope, assign a pair of
      // execution stream ids to it, and maybe enqueue nested computation.
      if (std::optional<ExecutionScopeKind> kind = IsExecutionScopeStart(hlo)) {
        // Try to find explicitly assigned stream id, otherwise generate a new
        // execution stream id for the new execution scope.
        std::optional<ExecutionStreamId> async_stream_id =
            FindAssignedStreamId(hlo);
        if (!async_stream_id.has_value()) {
          async_stream_id = execution_streams.Next(*kind);
        }

        VLOG(3) << absl::StreamFormat(
            "Start new %v execution scope: instr=%s parent=%v async=%v", *kind,
            hlo->name(), parent_stream_id, *async_stream_id);

        auto [_, emplaced] = async_instructions_.emplace(
            hlo, AsyncExecutionStreamIds{parent_stream_id, *async_stream_id});
        DCHECK(emplaced) << "Found duplicate execution stream assignment: "
                         << hlo->name();

        if (auto* computation = ExecutionScopeComputation(hlo)) {
          queue.push_back(PendingComputation{computation, *async_stream_id});
        }
      }

      // For operations that use execution scope copy execution stream
      // assignment from the start operation.
      if (std::optional<ExecutionScopeKind> kind = IsExecutionScopeUse(hlo)) {
        if (auto* start = FindExecutionScopeStart(hlo)) {
          CHECK(async_instructions_.contains(start))  // Crash OK
              << "Execution scope use operation '" << hlo->name()
              << "' does't have stream assignment for the start operation '"
              << start->name() << "'";

          AsyncExecutionStreamIds assn = async_instructions_.at(start);
          VLOG(3) << absl::StreamFormat(
              "Use %v execution scope: instr=%s parent=%v async=%v", *kind,
              hlo->name(), assn.parent_stream_id, assn.async_stream_id);
          auto [_, emplaced] = async_instructions_.emplace(hlo, assn);
          DCHECK(emplaced) << "Found duplicate execution stream assignment: "
                           << hlo->name();
        }
      }

      // For operations that end execution scope copy execution stream
      // assignment from the start operation.
      if (std::optional<ExecutionScopeKind> kind = IsExecutionScopeEnd(hlo)) {
        if (auto* start = FindExecutionScopeStart(hlo)) {
          CHECK(async_instructions_.contains(start))  // Crash OK
              << "Execution scope end operation '" << hlo->name()
              << "' does't have stream assignment for the start operation '"
              << start->name() << "'";

          AsyncExecutionStreamIds assn = async_instructions_.at(start);
          VLOG(3) << absl::StreamFormat(
              "End %v execution scope: instr=%s parent=%v async=%v", *kind,
              hlo->name(), assn.parent_stream_id, assn.async_stream_id);
          auto [_, emplaced] = async_instructions_.emplace(hlo, assn);
          DCHECK(emplaced) << "Found duplicate execution stream assignment: "
                           << hlo->name();
        }
      }

      // Check and skip HLO operations that modify execution scopes.
      if (IsExecutionScopeStart(hlo) || IsExecutionScopeUse(hlo) ||
          IsExecutionScopeEnd(hlo)) {
        DCHECK(async_instructions_.contains(hlo))
            << "Async instruction was not assigned an execution stream";
        continue;
      }

      // For operations that do not modify execution scopes assign a single
      // execution stream id assigned to the whole computation.
      auto [_, emplaced] = sync_instructions_.emplace(hlo, parent_stream_id);
      DCHECK(emplaced) << "Found duplicate execution stream assignment: "
                       << hlo->name();

      // For control flow operations we keep processing called computation with
      // the same stream id.
      if (HloPredicateIsOp<HloOpcode::kCall, HloOpcode::kConditional,
                           HloOpcode::kWhile>(hlo)) {
        for (auto* computation : hlo->called_computations()) {
          queue.push_back(PendingComputation{computation, parent_stream_id});
        }
      }

      // We don't need to process any other called computation because:
      // 1. For custom calls the semantics of called computation is defined
      //    by the custom call implementation.
      // 2. For reductions and other HLO operations with called computations XLA
      //    will generate a single kernel and we don't need to assign execution
      //    streams to embedded computations.
    }
  }

  VLOG(1) << absl::StreamFormat(
      "Assigned execution streams to module %s: #sync_instructions=%d "
      "#async_instructions=%d",
      module->name(), sync_instructions_.size(), async_instructions_.size());
}

static absl::Status StreamNotFoundError(const HloInstruction* instruction) {
  return absl::NotFoundError(absl::StrCat(
      "No ExecutionStreamId found for ", instruction->ToString(),
      "; this may happen if the Computation is not reachable from the module's "
      "entrypoint, or if it's only reachable through a embedded calls."));
}

absl::StatusOr<ExecutionStreamId>
ExecutionStreamAssignment::GetSyncExecutionStreamId(
    const HloInstruction* instruction) const {
  auto stream = sync_instructions_.find(instruction);
  if (stream == sync_instructions_.end()) {
    return StreamNotFoundError(instruction);
  }
  return stream->second;
}

absl::StatusOr<ExecutionStreamAssignment::AsyncExecutionStreamIds>
ExecutionStreamAssignment::GetAsyncExecutionStreamIds(
    const HloInstruction* instruction) const {
  auto streams = async_instructions_.find(instruction);
  if (streams == async_instructions_.end()) {
    return StreamNotFoundError(instruction);
  }
  return streams->second;
}

}  // namespace xla::gpu
