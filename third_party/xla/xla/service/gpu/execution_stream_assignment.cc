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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/transforms/collectives/collective_domain.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
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

constexpr CommunicationStreamId kPipelinedP2PStreamId0(1);
constexpr CommunicationStreamId kPipelinedP2PStreamId1(2);

// Maps pipelined P2P ops to CommunicationStreamId(1) and (2), running them
// on separate streams to avoid cyclic deadlocks.
ExecutionStreamId GetP2PStreamId(const HloInstruction* instruction) {
  const auto& fe_map = instruction->frontend_attributes().map();
  auto it = fe_map.find(kSendRecvPipelineAttr);
  if (it != fe_map.end() && it->second == "1") {
    return ExecutionStreamId(kPipelinedP2PStreamId1);
  }
  return ExecutionStreamId(kPipelinedP2PStreamId0);
}

// A helper class to generate the next execution stream id using round-robin
// assignment for two execution scope kinds. Returns correctly typed
// ComputationStreamId or CommunicationStreamId wrapped in ExecutionStreamId.
class ExecutionStreams {
 public:
  explicit ExecutionStreams(const ExecutionStreamAssignment::Options& opts)
      : opts_(opts),
        compute_id_(0),
        next_collective_domain_stream_id_(std::max<uint64_t>(
            kPipelinedP2PStreamId1.value() + 1,
            opts.number_of_communication_execution_streams)) {
    collective_stream_pools_.emplace(kUnspecifiedCollectiveDomain,
                                     CollectiveStreamPool{/*base=*/0});
  }

  ExecutionStreamId Next(ExecutionScopeKind kind) {
    switch (kind) {
      case ExecutionScopeKind::kCompute: {
        ComputationStreamId stream_id{compute_id_};
        compute_id_ =
            (compute_id_ + 1) % opts_.number_of_compute_execution_streams;
        return stream_id;
      }
      case ExecutionScopeKind::kCommunication: {
        return NextCollective(kUnspecifiedCollectiveDomain);
      }
    }
  }

  ExecutionStreamId NextCollective(CollectiveCommunicationDomain domain) {
    auto [it, inserted] = collective_stream_pools_.try_emplace(domain);
    CollectiveStreamPool& pool = it->second;
    if (inserted) {
      pool.base = next_collective_domain_stream_id_;
      next_collective_domain_stream_id_ +=
          opts_.number_of_communication_execution_streams;
    }
    CommunicationStreamId stream_id(pool.base + pool.ordinal);
    pool.ordinal =
        (pool.ordinal + 1) % opts_.number_of_communication_execution_streams;
    return stream_id;
  }

 private:
  struct CollectiveStreamPool {
    uint64_t base = 0;
    uint64_t ordinal = 0;
  };

  ExecutionStreamAssignment::Options opts_;
  uint64_t compute_id_;
  uint64_t next_collective_domain_stream_id_;
  absl::flat_hash_map<CollectiveCommunicationDomain, CollectiveStreamPool>
      collective_stream_pools_;
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
    return IsWrappedCollective(start) || IsCustomCollectiveOp(start) ||
                   start->frontend_attributes().map().contains(
                       kCollectiveGroupMarkerAttr)
               ? ExecutionScopeKind::kCommunication
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
  if (auto it = attrs.find(kXlaStreamAnnotationAttr);
      it != attrs.end() &&
      !absl::EqualsIgnoreCase(it->second, kXlaCollectiveStreamAnnotation)) {
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

std::optional<CollectiveCommunicationDomain> FindCollectiveDomain(
    const HloInstruction* instruction, ExecutionScopeKind kind) {
  if (kind != ExecutionScopeKind::kCommunication ||
      !SupportsCollectiveCommunicationDomain(*instruction)) {
    return std::nullopt;
  }

  absl::StatusOr<CollectiveCommunicationDomain> domain =
      GetCollectiveCommunicationDomain(*instruction);
  CHECK_OK(domain.status());
  if (*domain == kUnspecifiedCollectiveDomain) {
    return std::nullopt;
  }
  return *domain;
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
        // Prefer an explicitly assigned stream id, then a collective-domain
        // stream or a dedicated P2P stream for pipelined send/recv. Otherwise,
        // generate a new stream id for the execution scope.
        std::optional<ExecutionStreamId> stream_id =
            FindAssignedStreamId(hlo, *kind);
        if (!stream_id.has_value()) {
          std::optional<CollectiveCommunicationDomain> collective_domain =
              FindCollectiveDomain(hlo, *kind);
          if (collective_domain.has_value()) {
            stream_id = execution_streams.NextCollective(*collective_domain);
          }
        }
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
