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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_conflict_analysis.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_permute_cycle.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/source_target_pairs.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

// Returns true if the CollectivePermute instruction should be transformed
// to Send/Recv. We currently limit the transformation to CollectivePermute
// operations without any cycle in their (source, target) relationship,
// with only one input and without any context data.
static bool ShouldDecompose(
    const HloCollectivePermuteInstruction& collective_permute,
    int64_t threshold_in_bytes, const CallGraph& call_graph,
    DebugOptions::PipelineParallelismOptLevel pipeline_parallelism_opt_level) {
  const Shape& result_shape = collective_permute.shape();

  // Skip the transformation if result is not an array, such as containing
  // context data.
  if (!result_shape.IsArray()) {
    return false;
  }

  // Respect threshold to limit this pass.
  if (pipeline_parallelism_opt_level ==
          DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE &&
      ShapeUtil::ByteSizeOf(result_shape) < threshold_in_bytes) {
    return false;
  }

  // Do not decompose cycles as this leads to deadlocks in NCCL.
  if (collective_permute_cycle::HasCycles(
          SourceTargetPairs(collective_permute.source_target_pairs()))) {
    return false;
  }

  // Only decompose collective permutes that may be subject to pipelining.
  if (pipeline_parallelism_opt_level !=
      DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE) {
    if (!Match(collective_permute.operand(0),
               match::GetTupleElement(match::Parameter()))) {
      return false;
    }
  }
  auto callers = call_graph.GetComputationCallers(collective_permute.parent());
  if (callers.size() != 1 || callers.front()->opcode() != HloOpcode::kWhile) {
    return false;
  }

  return true;
}

// Returns true for a pipelineable collective-permute. As a simple heuristic,
// currently only pipeline a collective-permute with a loop input as its send
// data.
static bool MayPipeline(
    const HloCollectivePermuteInstruction& collective_permute) {
  return Match(collective_permute.operand(0),
               match::GetTupleElement(match::Parameter()));
}

namespace {

// Contains source-target pairs from the permute operation and send and recv
// instructions it was decomposed to.
struct DecomposedCp {
  HloInstruction* send;
  HloInstruction* recv;
  HloInstruction* send_done;
  HloInstruction* recv_done;
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
};

}  // namespace

static xla::FrontendAttributes ExtractFrontendAttributes(
    const HloCollectivePermuteInstruction& cp) {
  const xla::FrontendAttributes& old_attributes = cp.frontend_attributes();
  xla::FrontendAttributes attributes;
  attributes.mutable_map()->insert(old_attributes.map().begin(),
                                   old_attributes.map().end());
  (*attributes.mutable_map())[kSendRecvSourceTargetPairsAttr] =
      SourceTargetPairs(cp.source_target_pairs()).ToString();
  return attributes;
}

// Decomposes a collective-permute into send, send-done, recv, recv-done.
// Adds frontend attributes to record pipeline decision. The present of the
// frontend attribute means that the collective-permute will be pipelined and
// the value of the attribute represents the runtime stream to execute the
// instruction. Without the frontend attribute, the collective-permute will not
// be pipelined.
static absl::StatusOr<DecomposedCp> DecomposeCollectivePermute(
    HloCollectivePermuteInstruction* cp, HloComputation* computation,
    const std::string& pipeline_decision,
    DebugOptions::PipelineParallelismOptLevel pipeline_parallelism_opt_level) {
  absl::string_view cp_name = cp->name();
  std::optional<int64_t> channel_id = cp->channel_id();
  HloInstruction* data = cp->mutable_operand(0);
  const Shape& shape = data->shape();
  const OpMetadata& metadata = cp->metadata();
  const xla::FrontendAttributes attributes = ExtractFrontendAttributes(*cp);

  HloInstruction* after_all = computation->AddInstruction(
      HloInstruction::CreateToken(), absl::StrCat(cp_name, "-after-all"));
  HloInstruction* recv = computation->AddInstruction(
      HloInstruction::CreateRecv(shape, after_all, channel_id,
                                 /*is_host_transfer=*/false),
      absl::StrCat(cp_name, "-recv"));
  recv->set_frontend_attributes(attributes);
  recv->set_metadata(metadata);

  HloInstruction* send = computation->AddInstruction(
      HloInstruction::CreateSend(data, after_all, channel_id,
                                 /*is_host_transfer=*/false),
      absl::StrCat(cp_name, "-send"));
  send->set_frontend_attributes(attributes);
  send->set_metadata(metadata);

  HloInstruction* recv_done = computation->AddInstruction(
      HloInstruction::CreateRecvDone(recv, channel_id,
                                     /*is_host_transfer=*/false),
      absl::StrCat(cp_name, "-recv-done"));
  HloInstruction* send_done = computation->AddInstruction(
      HloInstruction::CreateSendDone(send, channel_id,
                                     /*is_host_transfer=*/false),
      absl::StrCat(cp_name, "-send-done"));

  HloInstruction* recv_data = computation->AddInstruction(
      HloInstruction::CreateGetTupleElement(recv_done, 0),
      absl::StrCat(cp_name, "-recv-data"));

  TF_RETURN_IF_ERROR(cp->ReplaceAllUsesWith(recv_data));
  TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(cp));

  // We choose to run recv before send as an invariant, which helps avoid
  // deadlocks. At the same time, running recv before send allows for pipelining
  // recv into prior loop iterations, which is especially beneficial for
  // pipeline parallelism.
  switch (pipeline_parallelism_opt_level) {
    case DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE:
      TF_RETURN_IF_ERROR(recv->AddControlDependencyTo(send));
      TF_RETURN_IF_ERROR(send->AddControlDependencyTo(recv_done));
      break;
    case DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE:
      TF_RETURN_IF_ERROR(recv_done->AddControlDependencyTo(send));
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported pipeline parallelism opt level: ",
                       pipeline_parallelism_opt_level));
  }

  if (!pipeline_decision.empty()) {
    send->set_frontend_attribute(kSendRecvPipelineAttr, pipeline_decision);
    send_done->set_frontend_attribute(kSendRecvPipelineAttr, pipeline_decision);
    recv->set_frontend_attribute(kSendRecvPipelineAttr, pipeline_decision);
    recv_done->set_frontend_attribute(kSendRecvPipelineAttr, pipeline_decision);
  }
  return DecomposedCp{send, recv, send_done, recv_done,
                      cp->source_target_pairs()};
}

// Checks whether the two collective-permutes for a forward cycle or a backward
// cycle for pipelining. If the two collective-permutes form a cycle, returns
// a pair of the collective-permutes with the one for the backward edge of the
// cycle as the first entry in the pair.
static std::optional<std::pair<HloCollectivePermuteInstruction*,
                               HloCollectivePermuteInstruction*>>
CheckCyclePatterns(HloCollectivePermuteInstruction* cp0,
                   HloCollectivePermuteInstruction* cp1) {
  const SourceTargetPairs cp0_pairs(cp0->source_target_pairs());
  const SourceTargetPairs cp1_pairs(cp1->source_target_pairs());
  if (collective_permute_cycle::IsForwardCycle(cp0_pairs, cp1_pairs) ||
      collective_permute_cycle::IsBackwardCycle(cp0_pairs, cp1_pairs)) {
    // cp0 represents the backedge for the cycle.
    return std::make_pair(cp0, cp1);
  }
  if (collective_permute_cycle::IsForwardCycle(cp1_pairs, cp0_pairs) ||
      collective_permute_cycle::IsBackwardCycle(cp1_pairs, cp0_pairs)) {
    // cp1 represents the forward edge for the cycle.
    return std::make_pair(cp1, cp0);
  }
  return std::nullopt;
}

static std::vector<HloInstruction*> FindAllConflictingCollectives(
    HloComputation* computation,
    const std::vector<HloCollectivePermuteInstruction*>& cps) {
  std::vector<HloInstruction*> seed_collectives;
  seed_collectives.reserve(cps.size());
  for (HloCollectivePermuteInstruction* cp : cps) {
    seed_collectives.push_back(static_cast<HloInstruction*>(cp));
  }
  return FindAllConflictingCollectives(computation, seed_collectives);
}

static void AddCollectiveStreamAnnotationP2P(
    std::vector<HloInstruction*>& instructions) {
  xla::FrontendAttributes attributes;
  (*attributes.mutable_map())[kCollectiveStreamAttrName] = kCollectiveStreamP2P;
  for (HloInstruction* instr : instructions) {
    instr->add_frontend_attributes(attributes);
  }
}

static void AddCollectiveStreamAnnotationP2P(
    std::vector<DecomposedCp>& decomposed) {
  std::vector<HloInstruction*> instructions;
  for (DecomposedCp& cp : decomposed) {
    instructions.push_back(cp.send);
    instructions.push_back(cp.recv);
  }
  AddCollectiveStreamAnnotationP2P(instructions);
}

// Inserts control dependencies to enforce send/recv chain order.
// The order protects from a potential deadlock when every device tries to
// execute recv with no devices executing send - if there are no constraints,
// the scheduler is free to schedule all recv ops first.
// deco_post_order is expected to be post order within a computation.
// TODO b/388072780 add second hueristic to enforce back edge before the forward
// edge for max performance.
static absl::Status EnforceOrderOfSendRecvChain(
    std::vector<DecomposedCp>& deco_post_order) {
  for (size_t i = 1; i < deco_post_order.size(); ++i) {
    DecomposedCp& cur = deco_post_order[i];
    DecomposedCp& prev = deco_post_order[i - 1];
    TF_RETURN_IF_ERROR(prev.send->AddControlDependencyTo(cur.recv));
    TF_RETURN_IF_ERROR(prev.send_done->AddControlDependencyTo(cur.recv_done));
  }
  return absl::OkStatus();
}

static absl::Status EnforceOrderOfSendRecvChainRelativeToConflictingCollectives(
    std::vector<DecomposedCp>& deco_post_order,
    std::vector<HloInstruction*> conflicting_collectives) {
  // Find last collective in send/recv chain.
  if (deco_post_order.empty()) return absl::OkStatus();
  HloInstruction* last_in_chain = deco_post_order.back().send_done;

  // Add control dependencies from chain to all conflicting collectives.
  for (HloInstruction* instr : conflicting_collectives) {
    TF_RETURN_IF_ERROR(last_in_chain->AddControlDependencyTo(instr));
  }

  return absl::OkStatus();
}

// TODO(b/399486412) clean up collective permute decomposer and only return one
// CP op
void RemoveAllButOne(std::vector<HloCollectivePermuteInstruction*>& cps) {
  // This is a heuristic that chooses the collective permute with the highest
  // number of source-target pairs.
  int cp_index = 0;
  int max_num_pairs = 0;
  for (int i = 0; i < cps.size(); ++i) {
    if (cps[i]->source_target_pairs().size() > max_num_pairs) {
      max_num_pairs = cps[i]->source_target_pairs().size();
      cp_index = i;
    }
  }
  cps = {cps[cp_index]};
}

absl::StatusOr<bool> CollectivePermuteDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

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
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        // Collect while-body computations.
        while_bodies.insert(instr->while_body());
        continue;
      }

      if (instr->opcode() != HloOpcode::kCollectivePermute) {
        continue;
      }

      HloCollectivePermuteInstruction* cp =
          Cast<HloCollectivePermuteInstruction>(instr);
      if (!ShouldDecompose(*cp, threshold_in_bytes_, *call_graph,
                           pipeline_parallelism_opt_level_)) {
        continue;
      }
      // Record collective-permute to be decomposed.
      cps_to_decompose.push_back(cp);

      if (!while_bodies.contains(computation) || !may_pipeline) {
        continue;
      }
      if (cp0_to_pipeline != nullptr && cp1_to_pipeline != nullptr) {
        // Already found a pair of collective-permute that forms a cycle to
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
    }  // for MakeInstructionPostOrder

    if (cps_to_decompose.size() > 1 &&
        pipeline_parallelism_opt_level_ !=
            DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE) {
      RemoveAllButOne(cps_to_decompose);
    }

    // Find all collectives conflicting with the collective permutes that we
    // want to decompose. We need this information to achieve two things:
    // 1. We want to run these in parallel with non-conflicting collectives,
    // e.g. those used on inner sharding strategies. The annotation allows us to
    // later execute them on a separate stream.
    // 2. We want to add control dependencies to these conflicting collectives
    // so that they cannot move in between the decomposed send/recv, which would
    // lead to deadlocks.
    std::vector<HloInstruction*> conflicing_collectives =
        FindAllConflictingCollectives(computation, cps_to_decompose);

    // cps to decompose were collected post order, similarly we will collect
    // the decomposed send/recv pairs.
    std::vector<DecomposedCp> deco_post_order;
    deco_post_order.reserve(cps_to_decompose.size());
    // Decompose the collective-permute, may add frontend attribute to record
    // pipeline decision.
    for (HloCollectivePermuteInstruction* cp : cps_to_decompose) {
      std::string pipeline_decision;
      if (cp0_to_pipeline == cp) {
        pipeline_decision = "0";
      } else if (cp1_to_pipeline == cp) {
        pipeline_decision = "1";
      }
      TF_ASSIGN_OR_RETURN(
          DecomposedCp decomposed_ops,
          DecomposeCollectivePermute(cp, computation, pipeline_decision,
                                     pipeline_parallelism_opt_level_));
      deco_post_order.push_back(decomposed_ops);
    }

    // Move all decomposed and conflicting collectives to a separate stream for
    // p2p communication. This will allow for overlap of pipeline parallelism
    // with other inner sharding strategies. We can remove this when XLA:GPU
    // supports multi-stream collectives more generally.
    if (pipeline_parallelism_opt_level_ !=
        DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE) {
      AddCollectiveStreamAnnotationP2P(conflicing_collectives);
      AddCollectiveStreamAnnotationP2P(deco_post_order);
    }

    // Enforce order of send/recv pairs at the beginning of the loop body. Also
    // enforce all other conflicting collectives to follow the send/recv chain
    // so that these cannot be scheduled in between the send/recv, which would
    // also lead to deadlocks.
    TF_RETURN_IF_ERROR(EnforceOrderOfSendRecvChain(deco_post_order));
    TF_RETURN_IF_ERROR(
        EnforceOrderOfSendRecvChainRelativeToConflictingCollectives(
            deco_post_order, conflicing_collectives));

    if (!cps_to_decompose.empty()) {
      changed = true;
    }
  }  // for reverse MakeComputationPostOrder
  return changed;
}
}  // namespace xla
