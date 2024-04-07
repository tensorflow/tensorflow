/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/collective_permute_decomposer.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

using SourceTargetPair = std::pair<int64_t, int64_t>;
using SourceTargetPairs = std::vector<SourceTargetPair>;

// Returns true if the (source, target) relationship has a cycle.
//
bool HasCycles(const SourceTargetPairs& pairs) {
  // Build a direct graph to check for cycles in (source, target) relationship.
  tensorflow::GraphCycles graph;

  // Map replica numbers to graph node ids.
  absl::flat_hash_map<int64_t, int32_t> replica_to_node_id;
  auto get_node_id = [&](int64_t replica) {
    auto it_and_inserted = replica_to_node_id.emplace(replica, -1);
    auto it = it_and_inserted.first;
    auto inserted = it_and_inserted.second;
    if (inserted) {
      // First time to see the replica, create a node for it.
      it->second = graph.NewNode();
    }
    return it->second;
  };

  for (auto pair : pairs) {
    auto source = get_node_id(pair.first);
    auto target = get_node_id(pair.second);
    VLOG(3) << "See source " << source << " -> target " << target;
    if (!graph.InsertEdge(source, target)) {
      VLOG(3) << "Detected cycles";
      return true;
    }
  }
  return false;
}

// Returns true if the CollectivePermute instruction should be transformed
// to Send/Recv. We currently limit the transformation to CollectivePermute
// operations without any cycle in their (source, target) relationship,
// with only one input and without any context data.
bool ShouldDecompose(const HloCollectivePermuteInstruction& collective_permute,
                     int64_t threshold_in_bytes) {
  // TODO(b/316043789): enable the transformation for the no channel_id case.
  if (!collective_permute.channel_id().has_value()) {
    return false;
  }

  const Shape& result_shape = collective_permute.shape();
  // Skip the transformation if result is not an array, such as containing
  // context data.
  if (!result_shape.IsArray()) {
    return false;
  }

  if (ShapeUtil::ByteSizeOf(result_shape) < threshold_in_bytes) {
    return false;
  }
  return !HasCycles(collective_permute.source_target_pairs());
}

// Returns true for a pipelineable collective-permute. As a simple heuristic,
// currently only pipeline a collective-permute with a loop input as its send
// data.
bool MayPipeline(const HloCollectivePermuteInstruction& collective_permute) {
  const HloInstruction* data = collective_permute.operand(0);
  return (data->opcode() == HloOpcode::kGetTupleElement &&
          data->operand(0)->opcode() == HloOpcode::kParameter);
}

// Decomposes a collective-permute and adds frontend attributes to record
// pipeline decision. The present of the frontend attribute means that the
// collective-permute will be pipelined and the value of the attribute
// represents the runtime stream to execute the instruction. Without the
// frontend attribute, the collective-permute will not be pipelined.
Status DecomposeCollectivePermute(
    HloCollectivePermuteInstruction* collective_permute,
    HloComputation* computation, const std::string& pipeline_decision) {
  // We currently only decompose collective-permute with a channel_id.
  int64_t channel_id = collective_permute->channel_id().value();
  HloInstruction* data = collective_permute->mutable_operand(0);
  const Shape& data_shape = data->shape();
  const OpMetadata& metadata = collective_permute->metadata();

  const xla::FrontendAttributes& old_attributes =
      collective_permute->frontend_attributes();
  xla::FrontendAttributes attributes;
  std::string source_target_pairs_string =
      "{" +
      absl::StrJoin(collective_permute->source_target_pairs(), ",",
                    absl::PairFormatter(
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, "{", value);
                        },
                        ",",
                        [](std::string* out, int64_t value) {
                          absl::StrAppend(out, value, "}");
                        })) +
      "}";
  attributes.mutable_map()->insert(old_attributes.map().begin(),
                                   old_attributes.map().end());
  (*attributes.mutable_map())[kSendRecvSourceTargetPairsAttr] =
      source_target_pairs_string;

  HloInstruction* after_all =
      computation->AddInstruction(HloInstruction::CreateToken());
  HloInstruction* recv = computation->AddInstruction(
      HloInstruction::CreateRecv(data_shape, after_all, channel_id));
  recv->add_frontend_attributes(attributes);
  recv->set_metadata(metadata);

  HloInstruction* send = computation->AddInstruction(
      HloInstruction::CreateSend(data, after_all, channel_id));
  send->add_frontend_attributes(attributes);
  send->set_metadata(metadata);

  HloInstruction* recv_done =
      computation->AddInstruction(HloInstruction::CreateRecvDone(recv));
  HloInstruction* send_done =
      computation->AddInstruction(HloInstruction::CreateSendDone(send));

  HloInstruction* recv_data = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(recv_done, 0));
  TF_RETURN_IF_ERROR(collective_permute->ReplaceAllUsesWith(recv_data));
  TF_RETURN_IF_ERROR(
      computation->RemoveInstructionAndUnusedOperands(collective_permute));

  if (!pipeline_decision.empty()) {
    xla::FrontendAttributes attributes;
    (*attributes.mutable_map())[kSendRecvPipelineAttr] = pipeline_decision;
    send->add_frontend_attributes(attributes);
    send_done->add_frontend_attributes(attributes);
    recv->add_frontend_attributes(attributes);
    recv_done->add_frontend_attributes(attributes);
  }

  return OkStatus();
}

// Returns true if the (source, target) pairs form a forward cycle with all
// participants in the cycle, such as {{0,1},{1,2},{2,3},{3,0}}. We assume that
// the (source, target) pairs are ordered via increasing source IDs, as they are
// currently generated by SPMD partitioning.
//
bool IsForwardCycle(const SourceTargetPair& backedge,
                    const SourceTargetPairs& others) {
  int64_t num_pairs = others.size() + 1;
  if (backedge.first != num_pairs - 1 || backedge.second != 0) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.first != i || pair.second != i + 1) {
      return false;
    }
  }
  return true;
}

// Returns true if the (source, target) pairs form a backward cycle with all
// participants in the cycle, such as {{0,3},{1,0},{2,1},{3,2}}. We assume that
// the (source, target) pairs are ordered via increasing source IDs, as they are
// currently generated by SPMD partitioning.
//
bool IsBackwardCycle(const SourceTargetPair& backedge,
                     const SourceTargetPairs& others) {
  int64_t num_pairs = others.size() + 1;
  if (backedge.first != 0 || backedge.second != num_pairs - 1) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.first != i + 1 || pair.second != i) {
      return false;
    }
  }
  return true;
}

// Checks whether the two collective-permutes for a forward cycle or a backward
// cycle for pipelining. If the two collective-permutes form a cycle, returns
// a pair of the collective-permutes with the one for the backward edge of the
// cycle as the first entry in the pair.
std::optional<std::pair<HloCollectivePermuteInstruction*,
                        HloCollectivePermuteInstruction*>>
CheckCyclePatterns(HloCollectivePermuteInstruction* cp0,
                   HloCollectivePermuteInstruction* cp1) {
  const SourceTargetPairs& cp0_pairs = cp0->source_target_pairs();
  const SourceTargetPairs& cp1_pairs = cp1->source_target_pairs();
  if (cp0_pairs.size() == 1) {
    if (IsForwardCycle(cp0_pairs.front(), cp1_pairs) ||
        IsBackwardCycle(cp0_pairs.front(), cp1_pairs)) {
      // cp0 represents the backedge for the cycle.
      return std::make_pair(cp0, cp1);
    }
  }
  if (cp1_pairs.size() == 1) {
    if (IsForwardCycle(cp1_pairs.front(), cp0_pairs) ||
        IsBackwardCycle(cp1_pairs.front(), cp0_pairs)) {
      // cp1 represents the forward edge for the cycle.
      return std::make_pair(cp1, cp0);
    }
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> CollectivePermuteDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> all_computations =
      module->MakeComputationPostOrder(execution_threads);
  absl::flat_hash_set<HloComputation*> while_bodies;
  // Process the computation from callers to callees and collect while-body
  // along the way. When we process a computation, we know whether it is a
  // while-body computation or not.
  for (auto iter = all_computations.rbegin(); iter != all_computations.rend();
       ++iter) {
    HloComputation* computation = *iter;
    bool may_pipeline = while_bodies.contains(computation);
    // Record the collective-permute to be decomposed as well as at most two
    // collective-permute for which the decomposed Send-Recv chains will be
    // pipelined.
    //
    // Currently, we simply choose the first pipelineable collect-permute we
    // encounter, along with another pipelineable collective-permute that forms
    // and cycle with the first collective-permute. We consider a
    // collective-permute pipelineable if the send-data is a loop parameter.
    // When two collective-permutes that form a cycle are selected,
    // cp0_to_pipeline records the collective-permute for the backedge of the
    // cycle.
    std::vector<HloCollectivePermuteInstruction*> cps_to_decompose;
    HloCollectivePermuteInstruction* cp0_to_pipeline = nullptr;
    HloCollectivePermuteInstruction* cp1_to_pipeline = nullptr;
    for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() == HloOpcode::kWhile) {
        // Collect while-body computations.
        while_bodies.insert(hlo->while_body());
        continue;
      }
      if (hlo->opcode() != HloOpcode::kCollectivePermute) {
        continue;
      }

      HloCollectivePermuteInstruction* cp =
          Cast<HloCollectivePermuteInstruction>(hlo);
      if (!ShouldDecompose(*cp, threshold_in_bytes_)) {
        continue;
      }
      // Record collective-permute to be decomposed.
      cps_to_decompose.push_back(cp);

      if (!while_bodies.contains(computation) || !may_pipeline) {
        continue;
      }
      if (cp0_to_pipeline != nullptr && cp1_to_pipeline != nullptr) {
        // Already find a pair of collective-permute that forms a cycle to
        // pipeline.
        continue;
      }
      if (!MayPipeline(*cp)) {
        continue;
      }
      if (cp0_to_pipeline == nullptr) {
        // Record the first pipelineable collective-permute.
        cp0_to_pipeline = cp;
        continue;
      }
      auto optional_pair = CheckCyclePatterns(cp0_to_pipeline, cp);
      if (optional_pair.has_value()) {
        // Add another pipelineable collective-permute that forms a cycle with
        // the first pipelineable collect-permute.

        // Collective-permute for the backward edge.
        cp0_to_pipeline = optional_pair.value().first;
        // Collective-permute for the forward edges.
        cp1_to_pipeline = optional_pair.value().second;
      }
    }

    // Decompose the collective-permute, may add frontend attribute to record
    // pipeline decision.
    for (HloCollectivePermuteInstruction* cp : cps_to_decompose) {
      std::string pipeline_decision;
      if (cp0_to_pipeline == cp) {
        pipeline_decision = "0";
      } else if (cp1_to_pipeline == cp) {
        pipeline_decision = "1";
      }
      TF_RETURN_IF_ERROR(
          DecomposeCollectivePermute(cp, computation, pipeline_decision));
    }
    if (!cps_to_decompose.empty()) {
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
