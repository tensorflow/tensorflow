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
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/status.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

bool ShouldPipeline(const HloInstruction* instr) {
  if (!HloPredicateIsOp<HloOpcode::kRecvDone, HloOpcode::kSendDone>(instr)) {
    return false;
  }

  // Not annotated for pipelining.
  auto it = instr->frontend_attributes().map().find(kSendRecvPipelineAttr);
  if (it == instr->frontend_attributes().map().end()) {
    return false;
  }

  // Checks that the SendDone or RecvDone is used for non-trivial computation.
  // This avoids repeatedly pipelining a loop.
  bool is_pipelined =
      (instr->user_count() == 1 && instr->parent() != nullptr &&
       instr->users()[0] == instr->parent()->root_instruction());
  return !is_pipelined;
}

bool ShouldAllowLoopVariantParameterInChain(const HloInstruction* instr) {
  // Allow any loop parameter needed for pipelining the Send/Recv instructions
  // that have been decided to pipeline.
  CHECK(instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->operand(0)->opcode() == HloOpcode::kParameter);
  return true;
}

Status PostprocessP2PImpl(
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
    return OkStatus();
  }
  auto statusor_bounds = ParseReplicaGroupsOnly(validation_it->second);
  if (!statusor_bounds.ok()) {
    return statusor_bounds.status();
  }
  std::string validation_attr = transformer(statusor_bounds.value());
  xla::FrontendAttributes attributes = instr->frontend_attributes();
  (*attributes.mutable_map())[kSendRecvValidationAttr] = validation_attr;
  instr->set_frontend_attributes(attributes);
  return OkStatus();
}

// Modifies the loop iteration frontend attribute for the peeled off Send and
// Recv for the first iteration of a loop.
Status PostprocessPeeledP2P(HloInstruction* instr) {
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
Status PostprocessRotatedP2P(HloInstruction* instr) {
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

}  // anonymous namespace

void AddP2PPipeliner(HloPassPipeline& pipeline) {
  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      // Pipeline everything annotated for pipelining.
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      CollectivePipeliner::PipeliningDirection::kBackward, ShouldPipeline,
      /*acceptable_formatting=*/HloPredicateTrue,
      /*reuse_pipelined_op_buffer=*/HloPredicateTrue,
      ShouldAllowLoopVariantParameterInChain, PostprocessPeeledP2P,
      PostprocessRotatedP2P};
  pipeline.AddPass<CollectivePipeliner>(config);
}

}  // namespace gpu
}  // namespace xla
