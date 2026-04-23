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

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/side_effect_util.h"

namespace xla::gpu {
namespace {

// There are two kinds of async execution scopes: compute and collective. We
// need just two as our goal is to effectively overlap computation with
// communication.
enum class ExecutionScopeKind { kCompute, kCommunication };

template <typename Sink>
void AbslStringify(Sink sink, ExecutionScopeKind kind) {
  switch (kind) {
    case ExecutionScopeKind::kCompute:
      sink.Append("compute");
      break;
    case ExecutionScopeKind::kCommunication:
      sink.Append("communication");
      break;
  }
}

// Maps pipelined P2P ops to CommunicationStreamId(1) and (2), running them
// on separate streams to avoid cyclic deadlocks.
ExecutionStreamId GetP2PStreamId(const HloInstruction* instruction) {
  const auto& fe_map = instruction->frontend_attributes().map();
  auto it = fe_map.find(kSendRecvPipelineAttr);
  if (it != fe_map.end() && it->second == "1") {
    return ExecutionStreamId(CommunicationStreamId(2));
  }
  return ExecutionStreamId(CommunicationStreamId(1));
}

// A helper class to generate the next execution stream id using round-robin
// assignment for two execution scope kinds. Returns correctly typed
// ComputationStreamId or CommunicationStreamId wrapped in ExecutionStreamId.
class ExecutionStreams {
 public:
  explicit ExecutionStreams(const ExecutionStreamAssignment::Options& opts)
      : opts_(opts), compute_id_(0), collective_id_(0) {}

  ExecutionStreamId Next(ExecutionScopeKind kind) {
    switch (kind) {
      case ExecutionScopeKind::kCompute: {
        ComputationStreamId stream_id{compute_id_};
        compute_id_ =
            (compute_id_ + 1) % opts_.number_of_compute_execution_streams;
        return stream_id;
      }
      case ExecutionScopeKind::kCommunication: {
        CommunicationStreamId stream_id{collective_id_};
        collective_id_ = (collective_id_ + 1) %
                         opts_.number_of_communication_execution_streams;
        return stream_id;
      }
    }
  }

 private:
  ExecutionStreamAssignment::Options opts_;
  uint64_t compute_id_;
  uint64_t collective_id_;
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

// Returns an execution scope kind if operations starts it.
std::optional<ExecutionScopeKind> IsExecutionScopeStart(
    const HloInstruction* hlo) {
  // Async operation that starts a new execution scope.
  if (auto* start = DynCast<HloAsyncStartInstruction>(hlo)) {
    return IsWrappedCollective(start) ? ExecutionScopeKind::kCommunication
                                      : ExecutionScopeKind::kCompute;
  }

  // Async-collective operations not yet migrated to async wrappers.
  if (HloPredicateIsOp<HloOpcode::kAllGatherStart, HloOpcode::kAllReduceStart,
                       HloOpcode::kCollectivePermuteStart>(hlo)) {
    return ExecutionScopeKind::kCommunication;
  }

  // Send/Recv operations: only the canonical start gets a scope.
  if (HloPredicateIsOp<HloOpcode::kRecv, HloOpcode::kSend>(hlo)) {
    return hlo == FindCanonicalSendRecvStartOp(hlo)
               ? std::make_optional(ExecutionScopeKind::kCommunication)
               : std::nullopt;
  }

  // A special case of asynchronous compute operation.
  if (HloPredicateIsOp<HloOpcode::kCopyStart>(hlo)) {
    return ExecutionScopeKind::kCompute;
  }

  return std::nullopt;
}

// Check if instruction has explicit stream assignment via the attributes.
std::optional<ExecutionStreamId> FindAssignedStreamId(
    const HloInstruction* instr, ExecutionScopeKind kind) {
  auto& attrs = instr->frontend_attributes().map();
  if (auto it = attrs.find(kXlaStreamAnnotationAttr); it != attrs.end()) {
    int32_t assigned_stream_id;
    CHECK(absl::SimpleAtoi(it->second, &assigned_stream_id));  // Crash OK
    switch (kind) {
      case ExecutionScopeKind::kCompute:
        return ExecutionStreamId(ComputationStreamId(assigned_stream_id));
      case ExecutionScopeKind::kCommunication:
        return ExecutionStreamId(CommunicationStreamId(assigned_stream_id));
    }
  }
  return std::nullopt;
}

}  // namespace

ExecutionStreamAssignment::ExecutionStreamAssignment(const HloModule* module,
                                                     const Options& options) {
  VLOG(1) << absl::StreamFormat(
      "Assign execution streams to module %s: #compute_streams=%d "
      "#communication_streams=%d",
      module->name(), options.number_of_compute_execution_streams,
      options.number_of_communication_execution_streams);
  ExecutionStreams execution_streams(options);

  std::deque<const HloComputation*> queue;
  queue.push_back(module->entry_computation());

  while (!queue.empty()) {
    const HloComputation* computation = queue.front();
    queue.pop_front();

    VLOG(2) << "Assign execution streams to computation: "
            << computation->name();

    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();

    for (const HloInstruction* hlo : instructions) {
      // Only assign execution stream IDs to scope-start operations.
      if (std::optional<ExecutionScopeKind> kind = IsExecutionScopeStart(hlo)) {
        // Try to find explicitly assigned stream id, or use dedicated P2P
        // stream for pipelined send/recv, otherwise generate a new execution
        // stream id for the new execution scope.
        std::optional<ExecutionStreamId> stream_id =
            FindAssignedStreamId(hlo, *kind);
        if (!stream_id.has_value() && IsPipelinedP2P(hlo) &&
            options.number_of_communication_execution_streams > 1) {
          stream_id = GetP2PStreamId(hlo);
        }
        if (!stream_id.has_value()) {
          stream_id = execution_streams.Next(*kind);
        }

        VLOG(3) << absl::StreamFormat(
            "Start new %v execution scope: instr=%s stream=%v", *kind,
            hlo->name(), *stream_id);

        auto [_, emplaced] = async_start_instructions_.emplace(hlo, *stream_id);
        DCHECK(emplaced) << "Found duplicate execution stream assignment: "
                         << hlo->name();
      }

      // For control flow operations keep processing called computations.
      if (HloPredicateIsOp<HloOpcode::kCall, HloOpcode::kConditional,
                           HloOpcode::kWhile>(hlo)) {
        for (auto* called : hlo->called_computations()) {
          queue.push_back(called);
        }
      }
    }
  }

  VLOG(1) << absl::StreamFormat(
      "Assigned execution streams to module %s: #async_start_instructions=%d",
      module->name(), async_start_instructions_.size());
}

absl::StatusOr<ExecutionStreamId>
ExecutionStreamAssignment::GetExecutionStreamId(
    const HloInstruction* instruction) const {
  auto it = async_start_instructions_.find(instruction);
  if (it == async_start_instructions_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "No ExecutionStreamId found for ", instruction->ToString(),
        "; this instruction is either not a scope-start operation, not "
        "reachable from the module's entrypoint, or only reachable through "
        "embedded calls."));
  }
  return it->second;
}

}  // namespace xla::gpu
