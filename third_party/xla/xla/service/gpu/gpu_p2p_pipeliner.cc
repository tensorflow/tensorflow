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

#include "xla/service/gpu/gpu_p2p_pipeliner.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/collective_conflict_analysis.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

using ::xla::match::GetTupleElement;
using ::xla::match::Op;
using ::xla::match::Parameter;
using ::xla::match::Recv;
using ::xla::match::Send;

// Rather than pipelining the send/recv and *-done instructions, we only
// pipeline send/recv instructions. This allows spanning async send/recv across
// the loop boundary.
bool PipelineOnlySendRecvStart(
    const HloInstruction* instruction,
    absl::flat_hash_map<const HloInstruction*, bool>& cache) {
  auto it = cache.find(instruction);
  if (it != cache.end()) {
    return it->second;
  }

  // Only pipeline send/recv instructions that operate on a loop parameter.
  if (!Match(instruction, Recv()) &&
      !Match(instruction, Send(GetTupleElement(Parameter()), Op()))) {
    cache[instruction] = false;
    return false;
  }

  // Only pipeline them if all control predecessors are also pipelined.
  for (HloInstruction* other_instruction :
       instruction->control_predecessors()) {
    if (!PipelineOnlySendRecvStart(other_instruction, cache)) {
      cache[instruction] = false;
      return false;
    }
  }

  cache.insert({instruction, true});
  return true;
}

// Fully pipeline recv and recv-done instructions w/o any control dependencies.
// Determines if a given instruction matches `recv-done(recv())` to configure
// the pipeliner to pipeline these two instructions together.
bool FullyPipelineRecv(const HloInstruction* recv_done_candidate) {
  if (recv_done_candidate->opcode() != HloOpcode::kRecvDone ||
      !recv_done_candidate->control_predecessors().empty()) {
    return false;
  }
  const HloInstruction* recv_candidate = recv_done_candidate->operand(0);
  return recv_candidate->opcode() == HloOpcode::kRecv &&
         recv_candidate->control_predecessors().empty();
}

bool ShouldPipeline(const HloInstruction* instruction) {
  if (!HloPredicateIsOp<HloOpcode::kRecvDone, HloOpcode::kSendDone>(
          instruction)) {
    return false;
  }
  // Not annotated for pipelining.
  auto it =
      instruction->frontend_attributes().map().find(kSendRecvPipelineAttr);
  if (it == instruction->frontend_attributes().map().end()) {
    return false;
  }

  // Allow RecvDone to have a Send as a control predecessor. This control
  // predecessor will be dropped by the pipeliner, which is what we needed
  // when we rotate the RecvDone to the beginning of the while-body.
  auto allowed_predecessor = [&]() {
    return instruction->opcode() == HloOpcode::kRecvDone &&
           instruction->control_predecessors().size() == 1 &&
           instruction->control_predecessors()[0]->opcode() == HloOpcode::kSend;
  };
  if (!instruction->control_successors().empty() ||
      (!instruction->control_predecessors().empty() &&
       !allowed_predecessor())) {
    return false;
  }

  // Checks that the SendDone or RecvDone is used for non-trivial computation.
  // This avoids repeatedly pipelining a loop.
  bool is_pipelined =
      (instruction->user_count() == 1 && instruction->parent() != nullptr &&
       instruction->users()[0] == instruction->parent()->root_instruction());
  return !is_pipelined;
}

bool ShouldAllowLoopVariantParameterInChain(const HloInstruction* instr) {
  // Allow any loop parameter needed for pipelining the Send/Recv instructions
  // that have been decided to pipeline.
  CHECK(instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->operand(0)->opcode() == HloOpcode::kParameter);
  return true;
}

absl::Status PostprocessP2PImpl(
    HloInstruction* instr,
    std::function<std::string(std::vector<ReplicaGroup>&)> transformer) {
  // The input instruction is a Done instruction.
  if (!HloPredicateIsOp<HloOpcode::kRecvDone, HloOpcode::kSendDone>(instr)) {
    return Internal("Expected SendDone/RecvDone as the pipelined collective");
  }
  instr = instr->mutable_operand(0);
  if (!HloPredicateIsOp<HloOpcode::kRecv, HloOpcode::kSend>(instr)) {
    return Internal("Expected Send/Recv as the SendDone/RecvDone operand");
  }
  auto validation_it =
      instr->frontend_attributes().map().find(kSendRecvValidationAttr);
  if (validation_it == instr->frontend_attributes().map().end() ||
      validation_it->second == "invalid") {
    return absl::OkStatus();
  }
  auto statusor_bounds = ParseReplicaGroupsOnly(validation_it->second);
  if (!statusor_bounds.ok()) {
    return statusor_bounds.status();
  }
  std::string validation_attr = transformer(statusor_bounds.value());
  xla::FrontendAttributes attributes = instr->frontend_attributes();
  (*attributes.mutable_map())[kSendRecvValidationAttr] = validation_attr;
  instr->set_frontend_attributes(attributes);
  return absl::OkStatus();
}

// Modifies the loop iteration frontend attribute for the peeled off Send and
// Recv for the first iteration of a loop.
absl::Status PostprocessPeeledP2P(HloInstruction* instr,
                                  HloInstruction* new_while_instr) {
  // We only use this to post-process the peeled send/recv before the new loop
  // was created.
  CHECK(new_while_instr == nullptr);

  auto transform_bounds = [&](std::vector<ReplicaGroup>& replica_groups) {
    std::vector<std::pair<int64_t, int64_t>> bounds;
    bounds.reserve(replica_groups.size());
    bool all_invalid = true;
    for (const auto& replica_group : replica_groups) {
      // The peeled off instruction is for executing the first iteration of
      // the loop.
      int64_t lower_bound = replica_group.replica_ids(0);
      int64_t upper_bound = replica_group.replica_ids(1);
      if (lower_bound <= 0 && upper_bound >= 0) {
        all_invalid = false;
        bounds.push_back({0, 0});
      } else {
        bounds.push_back({1, 0});
      }
    }
    std::string validation_attr;
    if (all_invalid) {
      // An optimized way to represent that all source-target pairs are
      // communicating invalid data, to avoid the overhead related to the use
      // of execution counters.
      validation_attr = "invalid";
    } else {
      validation_attr = "{" +
                        absl::StrJoin(bounds, ",",
                                      absl::PairFormatter(
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, "{", value);
                                          },
                                          ",",
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, value, "}");
                                          })) +
                        "}";
    }
    return validation_attr;
  };
  return PostprocessP2PImpl(instr, transform_bounds);
};

// Modifies the loop iteration frontend attribute for the rotated Send and Recv
// for the remaining iterations in a loop.
absl::Status PostprocessRotatedP2P(HloInstruction* instr,
                                   HloInstruction* new_while_instr) {
  // We only use this to post-process the peeled send/recv before the new loop
  // was created.
  CHECK(new_while_instr == nullptr);

  auto transform_bounds = [&](std::vector<ReplicaGroup>& replica_groups) {
    std::vector<std::pair<int64_t, int64_t>> bounds;
    bounds.reserve(replica_groups.size());
    bool all_invalid = true;
    for (const auto& replica_group : replica_groups) {
      int64_t lower_bound = replica_group.replica_ids(0);
      int64_t upper_bound = replica_group.replica_ids(1);
      if (lower_bound <= upper_bound) {
        if (lower_bound >= 1) {
          --lower_bound;
        }
        if (upper_bound >= 1) {
          --upper_bound;
        }
        if (lower_bound <= upper_bound) {
          all_invalid = false;
          bounds.push_back({lower_bound, upper_bound});
        } else {
          bounds.push_back({1, 0});
        }
      } else {
        bounds.push_back({lower_bound, upper_bound});
      }
    }

    std::string validation_attr;
    if (all_invalid) {
      // An optimized way to represent that all source-target pairs are
      // communicating invalid data, to avoid the overhead related to the use
      // of execution counters.
      validation_attr = "invalid";
    } else {
      validation_attr = "{" +
                        absl::StrJoin(bounds, ",",
                                      absl::PairFormatter(
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, "{", value);
                                          },
                                          ",",
                                          [](std::string* out, int64_t value) {
                                            absl::StrAppend(out, value, "}");
                                          })) +
                        "}";
    }
    return validation_attr;
  };

  return PostprocessP2PImpl(instr, transform_bounds);
}

struct PeeledHloInstructionInfo {
  HloInstruction* instr;
  // The new while out of which this instruction was peeled. Can be nullptr if
  // the new loop was not yet created.
  HloInstruction* while_instr;
};

}  // anonymous namespace

// Finds the start instruction for send/recv where send/recv-done are chosen to
// be pipelined.
static HloInstruction* GetSendRecvStartInstruction(HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kRecv ||
      instr->opcode() == HloOpcode::kSend) {
    return instr;
  }
  if (instr->opcode() == HloOpcode::kRecvDone ||
      instr->opcode() == HloOpcode::kSendDone) {
    return instr->mutable_operand(0);
  }
  return nullptr;
}

static std::vector<HloInstruction*> GetSendRecvStartInstructions(
    const std::vector<HloInstruction*>& instructions) {
  std::vector<HloInstruction*> start_instructions;
  for (HloInstruction* instr : instructions) {
    HloInstruction* start_instr = GetSendRecvStartInstruction(instr);
    if (start_instr != nullptr) start_instructions.push_back(start_instr);
  }
  return start_instructions;
}

static absl::Nullable<HloInstruction*> GetSendRecvDoneInstructions(
    absl::Nonnull<HloInstruction*> rotated_instr) {
  auto it = absl::c_find_if(rotated_instr->users(), [](HloInstruction* user) {
    return user->opcode() == HloOpcode::kRecvDone ||
           user->opcode() == HloOpcode::kSendDone;
  });
  return it != rotated_instr->users().end() ? *it : nullptr;
}

// Post-process rotated send/recv ops to add control dependencies with
// conflicting collectives.
static absl::Status PostProcessRotatedSendRecvOps(
    const std::vector<HloInstruction*>& rotated) {
  // Find the start instructions for send/recv.
  std::vector<HloInstruction*> rotated_send_recvs =
      GetSendRecvStartInstructions(rotated);

  VLOG(5) << "Post-processing rotated send/recv ops:";
  if (VLOG_IS_ON(5)) {
    for (HloInstruction* instr : rotated_send_recvs) {
      VLOG(5) << " - " << instr->ToShortString();
    }
  }

  // Convert to set for faster lookup.
  absl::flat_hash_set<HloInstruction*> rotated_send_recvs_set(
      rotated_send_recvs.begin(), rotated_send_recvs.end());

  // Add control dependencies from conflicting collectives to rotated send/recv
  // ops.
  for (HloInstruction* rotated_instr : rotated_send_recvs) {
    VLOG(5) << "Working on " << rotated_instr->ToShortString();
    CHECK(rotated_instr->opcode() == HloOpcode::kRecv ||
          rotated_instr->opcode() == HloOpcode::kSend);
    HloComputation* parent = rotated_instr->parent();
    int64_t num_conflicting_collectives = 0;
    for (HloInstruction* conflicting_collective :
         FindAllConflictingCollectives(parent, {rotated_instr})) {
      if (rotated_send_recvs_set.contains(conflicting_collective)) continue;
      num_conflicting_collectives++;
      HloInstruction* new_control_dependency =
          GetSendRecvDoneInstructions(rotated_instr);
      CHECK_NE(new_control_dependency, nullptr);
      TF_RETURN_IF_ERROR(conflicting_collective->AddControlDependencyTo(
          new_control_dependency));
      VLOG(5) << "Adding control dependency from "
              << conflicting_collective->ToShortString() << " to "
              << rotated_instr->ToShortString() << "\n";
    }
    VLOG(5) << "Conflicting collectives: " << num_conflicting_collectives;
  }

  return absl::OkStatus();
}

// For a peeled send/recv instruction, find the corresponding send/recv-done
// instruction after the while loop.
// TODO(frgossen): Simplify this to support only directly consuming
// send/recv-done ops when we no longer support partial pipelining.
static HloInstruction* FindSendRecvDoneInstruction(HloInstruction* instr) {
  CHECK(instr->opcode() == HloOpcode::kRecv ||
        instr->opcode() == HloOpcode::kSend);
  CHECK_EQ(instr->user_count(), 1);

  HloInstruction* candidate = instr->users().front();
  if (candidate->opcode() == HloOpcode::kTuple) {
    HloInstruction* tuple_op = candidate;
    int64_t i = tuple_op->operand_index(instr);
    CHECK_EQ(tuple_op->user_count(), 1);
    HloInstruction* while_op = tuple_op->users().front();
    CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
    for (HloInstruction* user : while_op->users()) {
      HloGetTupleElementInstruction* gte_op =
          DynCast<HloGetTupleElementInstruction>(user);
      if (gte_op == nullptr || gte_op->tuple_index() != i) continue;
      CHECK_EQ(gte_op->user_count(), 1);
      candidate = gte_op->users().front();
      break;
    }
  }
  CHECK(candidate->opcode() == HloOpcode::kRecvDone ||
        candidate->opcode() == HloOpcode::kSendDone);
  return candidate;
}

static absl::Status AddControlDependencies(
    std::vector<HloInstruction*>& from_instructions, HloInstruction* to_instr) {
  for (HloInstruction* from_instr : from_instructions) {
    VLOG(5) << "Adding control dependency from " << from_instr->ToShortString()
            << " to " << to_instr->ToShortString();
    TF_RETURN_IF_ERROR(from_instr->AddControlDependencyTo(to_instr));
  }
  return absl::OkStatus();
}

static absl::Status AddControlDependencies(
    HloInstruction* from_instr,
    absl::flat_hash_set<HloInstruction*>& to_instructions) {
  for (HloInstruction* to_instr : to_instructions) {
    VLOG(5) << "Adding control dependency from " << from_instr->ToShortString()
            << " to " << to_instr->ToShortString();
    TF_RETURN_IF_ERROR(from_instr->AddControlDependencyTo(to_instr));
  }
  return absl::OkStatus();
}

static std::vector<PeeledHloInstructionInfo> GetSendRecvStartInstructions(
    const std::vector<PeeledHloInstructionInfo>& peeled) {
  std::vector<PeeledHloInstructionInfo> peeled_send_recvs;
  for (const auto [instr, while_instr] : peeled) {
    HloInstruction* send_recv_start = GetSendRecvStartInstruction(instr);
    if (send_recv_start == nullptr) continue;
    peeled_send_recvs.push_back({send_recv_start, while_instr});
  }
  return peeled_send_recvs;
}

static absl::Status PostProcessPeeledSendRecvOps(
    const std::vector<PeeledHloInstructionInfo>& peeled) {
  // Find the start instructions for send/recv.
  std::vector<PeeledHloInstructionInfo> peeled_send_recvs =
      GetSendRecvStartInstructions(peeled);

  VLOG(5) << "Post-processing peeled send/recv ops:";
  if (VLOG_IS_ON(5)) {
    for (const auto [instr, while_instr] : peeled_send_recvs) {
      VLOG(5) << " - " << instr->ToShortString();
    }
  }

  // Convert to set for faster lookup.
  absl::flat_hash_set<HloInstruction*> peeled_send_recvs_set;
  for (const PeeledHloInstructionInfo& info : peeled_send_recvs) {
    peeled_send_recvs_set.insert(info.instr);
  }

  // Add control dependencies between conflicting collectives and peeled
  // send/recv ops.
  for (PeeledHloInstructionInfo peeled : peeled_send_recvs) {
    VLOG(5) << "Working on " << peeled.instr->ToShortString();
    CHECK(peeled.instr->opcode() == HloOpcode::kRecv ||
          peeled.instr->opcode() == HloOpcode::kSend);

    // Find all conflicting collectives that were not peeled out of the loop.
    absl::flat_hash_set<HloInstruction*> unpeeled_conflicting_collectives;
    for (HloInstruction* instr : FindAllConflictingCollectives(peeled.instr)) {
      if (peeled_send_recvs_set.contains(instr)) continue;
      unpeeled_conflicting_collectives.insert(instr);
    }
    VLOG(5) << "Conflicting collectives: "
            << unpeeled_conflicting_collectives.size();

    // We separate unpeeled conflicting collectives into two categories: those
    // dominating the while loop (while loop has a data dependency on them), and
    // those that don't.
    std::vector<HloInstruction*> dominating_unpeeled_conflicting_collectives;
    for (HloInstruction* instr :
         peeled.while_instr->parent()->MakeInstructionPostOrderFrom(
             *peeled.while_instr)) {
      if (unpeeled_conflicting_collectives.contains(instr)) {
        dominating_unpeeled_conflicting_collectives.push_back(instr);
        unpeeled_conflicting_collectives.erase(instr);
      }
    }

    // Add control dependencies from dominating conflicting collectives to the
    // peeled send/recv instruction. This guarantees that the conflicting
    // collectives cannot slip in between the peeled send/recv instructions
    // where it could cause a deadlock.
    VLOG(5) << "Adding control dependencies FROM dominating conflicting";
    TF_RETURN_IF_ERROR(AddControlDependencies(
        dominating_unpeeled_conflicting_collectives, peeled.instr));

    // Add control dependencies from the final peeleled send/recv-done
    // instruction to the conflicting collectives that are dominated by the
    // while loop. This guarantees that the conflicting collectives cannot slip
    // in between the peeled send/recv instructions where it could cause a
    // deadlock.
    VLOG(5) << "Adding control dependencies TO dominating conflicting";
    HloInstruction* done_op = FindSendRecvDoneInstruction(peeled.instr);
    CHECK_NE(done_op, nullptr);
    TF_RETURN_IF_ERROR(
        AddControlDependencies(done_op, unpeeled_conflicting_collectives));
  }

  return absl::OkStatus();
}

static absl::Status PostProcessPeeledSendRecvOps(
    const std::vector<HloInstruction*>& peeled) {
  std::vector<PeeledHloInstructionInfo> peeled_info;

  // For each peeled non-trailing send/recv instruction, find the corresponding
  // while loop. The while loop is the immediate user of the recv -> recv-done
  // chain.
  for (HloInstruction* instr : peeled) {
    HloInstruction* instr_start = GetSendRecvStartInstruction(instr);
    CHECK_NE(instr_start, nullptr);

    // Find the while loop.
    HloInstruction* peeled_instr = instr_start;
    CHECK_EQ(peeled_instr->user_count(), 1);
    HloInstruction* closest_to_while_op = peeled_instr->users().front();
    if (closest_to_while_op->opcode() == HloOpcode::kRecvDone ||
        closest_to_while_op->opcode() == HloOpcode::kSendDone) {
      CHECK_EQ(closest_to_while_op->user_count(), 1);
      closest_to_while_op = closest_to_while_op->users().front();
    }
    HloInstruction* tuple_op = closest_to_while_op;
    CHECK_EQ(tuple_op->opcode(), HloOpcode::kTuple);
    CHECK_EQ(tuple_op->user_count(), 1);
    HloInstruction* while_op = tuple_op->users().front();
    CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);

    peeled_info.push_back({instr_start, while_op});
  }

  // Post-process peeled send/recv ops with the now known corresponding while
  // loop.
  return PostProcessPeeledSendRecvOps(peeled_info);
}

absl::StatusOr<bool> GpuP2PPipeliner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloPredicate should_process = ShouldPipeline;
  CollectivePipeliner::HloPostprocessor postprocess_backward_peeled_op =
      PostprocessPeeledP2P;
  CollectivePipeliner::HloPostprocessor postprocess_backward_rotated_op =
      PostprocessRotatedP2P;
  CollectivePipeliner::HloPostprocessor
      postprocess_backward_peeled_trailing_op = std::nullopt;

  // If partial send/recv pipelining is enabled, collect send/recv instructions
  // for post-processing.
  std::vector<HloInstruction*> peeled_send_recvs;
  std::vector<HloInstruction*> rotated_send_recvs;
  std::vector<PeeledHloInstructionInfo> peeled_trailing_send_recvs;

  if (enable_partial_send_recv_pipelining_) {
    should_process = FullyPipelineRecv;
    postprocess_backward_peeled_op = [&](HloInstruction* it,
                                         HloInstruction* new_while_instr) {
      // When post-processing non-trailing peeled send/recv, the new while loop
      // was not yet created.
      CHECK_EQ(new_while_instr, nullptr);
      peeled_send_recvs.push_back(it);
      return absl::OkStatus();
    };
    postprocess_backward_rotated_op = [&](HloInstruction* it,
                                          HloInstruction* new_while_instr) {
      // When post-processing non-trailing peeled send/recv, the new while loop
      // was not yet created.
      CHECK_EQ(new_while_instr, nullptr);
      rotated_send_recvs.push_back(it);
      return absl::OkStatus();
    };
    postprocess_backward_peeled_trailing_op =
        [&](HloInstruction* it, HloInstruction* new_while_instr) {
          // When post-processing trailing peeled send/recv, we need the new
          // while loop.
          CHECK_NE(new_while_instr, nullptr);
          peeled_trailing_send_recvs.push_back({it, new_while_instr});
          return absl::OkStatus();
        };
  }

  // Run pipeliner.
  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      // Pipeline everything annotated for pipelining.
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      CollectivePipeliner::PipeliningDirection::kBackward,
      /*should_process=*/should_process,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateTrue,
      /*should_allow_loop_variant_parameter_in_chain=*/
      ShouldAllowLoopVariantParameterInChain,
      /*should_allow_control_dependencies=*/true,
      /*postprocess_backward_peeled_op=*/postprocess_backward_peeled_op,
      /*postprocess_backward_rotated_op=*/postprocess_backward_rotated_op,
      /*postprocess_backward_peeled_trailing_op=*/
      postprocess_backward_peeled_trailing_op};
  TF_ASSIGN_OR_RETURN(
      bool changed, CollectivePipeliner(config).Run(module, execution_threads));

  VLOG(5) << "After pipelining, before post-processing:";
  XLA_VLOG_LINES(5, module->ToString());

  // Post-process rotated and peeled send/recv ops to add control dependencies
  // with conflicting collectives.
  if (enable_partial_send_recv_pipelining_) {
    TF_RETURN_IF_ERROR(PostProcessRotatedSendRecvOps(rotated_send_recvs));
    TF_RETURN_IF_ERROR(
        PostProcessPeeledSendRecvOps(peeled_trailing_send_recvs));
    TF_RETURN_IF_ERROR(PostProcessPeeledSendRecvOps(peeled_send_recvs));
  }

  VLOG(5) << "After post-processing:";
  XLA_VLOG_LINES(5, module->ToString());

  return changed;
}

}  // namespace gpu
}  // namespace xla
