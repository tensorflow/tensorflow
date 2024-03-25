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
#include "xla/service/gpu/loop_double_buffer_transformer.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/status.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

void SetChannelIdForNewCollective(HloInstruction* new_instr,
                                  const HloModule* module) {
  // This is to track mappings of old->new channel id for async collectives
  // wrapped in the form of HloAsyncInstruction, the start and done need to
  // have the same unique channel id.
  absl::flat_hash_map<int64_t, int64_t> old_to_new_channel_id_map;
  absl::flat_hash_map<int64_t, HloComputation*> channel_id_comp_map;
  if (new_instr->IsAsynchronous() && hlo_query::IsCollectiveCommunicationOp(
                                         new_instr->async_wrapped_opcode())) {
    HloInstruction* wrapped_instr =
        DynCast<HloAsyncInstruction>(new_instr)->async_wrapped_instruction();
    int64_t old_channel_id = *wrapped_instr->channel_id();
    int64_t new_channel_id = old_to_new_channel_id_map[old_channel_id];
    if (old_to_new_channel_id_map.find(old_channel_id) ==
        old_to_new_channel_id_map.end()) {
      new_channel_id = hlo_query::NextChannelId(*module);
      VLOG(2) << "Generated new channel id " << new_channel_id;
      old_to_new_channel_id_map[old_channel_id] = new_channel_id;
    }

    VLOG(2) << "Setting channel id to " << new_channel_id;

    wrapped_instr->set_channel_id(new_channel_id);
    if (channel_id_comp_map.find(new_channel_id) == channel_id_comp_map.end()) {
      channel_id_comp_map[new_channel_id] =
          new_instr->async_wrapped_computation();
    } else {
      channel_id_comp_map[new_channel_id]->AddAsyncStart(new_instr);
    }
  } else if (hlo_query::IsCollectiveCommunicationOp(new_instr->opcode()) ||
             hlo_query::IsAsyncCollectiveStartOp(new_instr)) {
    new_instr->set_channel_id(hlo_query::NextChannelId(*module));
  }
}

absl::Status PeelInstructionsForOddTripCount(HloModule* module,
                                             HloInstruction* while_instr) {
  std::string suffix = "peeled_double_buffer";
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
  HloComputation* while_body = while_instr->while_body();
  HloInstruction* input_parameter = while_body->parameter_instruction(0);
  HloInstruction* input_tuple = while_instr->mutable_operand(0);

  auto old_loop_roots = while_body->root_instruction()->mutable_operands();
  HloComputation* parent_comp = while_instr->parent();
  old_to_new_map[input_parameter] = input_tuple;

  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      continue;
    }
    VLOG(2) << "Peeling instruction " << old_instr->ToString();
    std::vector<HloInstruction*> new_operands(old_instr->operand_count());
    for (int64_t i = 0; i < old_instr->operand_count(); i++) {
      new_operands[i] = old_to_new_map[old_instr->mutable_operand(i)];
    }
    HloInstruction* new_instr =
        parent_comp->AddInstruction(old_instr->CloneWithNewOperands(
            old_instr->shape(), new_operands, suffix));

    SetChannelIdForNewCollective(new_instr, module);
    old_to_new_map[old_instr] = new_instr;
    VLOG(2) << "Added instruction " << new_instr->ToString()
            << " to parent computation.";
  }

  std::vector<HloInstruction*> new_roots;
  for (HloInstruction* instr : old_loop_roots) {
    new_roots.push_back(old_to_new_map[instr]);
  }
  TF_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(
      0, old_to_new_map[while_body->root_instruction()]));
  VLOG(2) << "Replaced with new input tuple "
          << while_instr->operand(0)->ToString();

  // Handle existing control dependencies.
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      HloInstruction* new_instr = old_to_new_map[old_instr];
      VLOG(2) << "Processing control predecessors for peeled instruction "
              << new_instr->ToString();
      std::vector<HloInstruction*> new_control_pred(
          old_instr->control_predecessors().size());
      for (HloInstruction* pred : old_instr->control_predecessors()) {
        new_control_pred.push_back(old_to_new_map[pred]);
      }

      TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
      for (HloInstruction* new_pred : new_control_pred) {
        TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
        VLOG(2) << "Adding " << new_pred->ToString()
                << " to control dependency of peeled instruction: "
                << new_instr->ToString();
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> LoopDoubleBufferTransformer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  VLOG(2) << "Processing " << while_instrs.size() << " while loops.";

  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                        while_instr->backend_config<WhileLoopBackendConfig>());
    if (!config.has_known_trip_count()) {
      VLOG(2) << while_instr->ToString()
              << " doesn't have exact trip count, skipping double buffering "
                 "for now";
      continue;
    }
    int64_t exact_trip_count = config.known_trip_count().n();
    VLOG(2) << "Processing while loop " << while_instr->ToString()
            << " with trip count: " << exact_trip_count;

    HloComputation* while_body = while_instr->while_body();

    VLOG(2) << "Processing root " << while_body->root_instruction()->ToString();

    auto old_loop_roots = while_body->root_instruction()->mutable_operands();
    HloInstruction* input_parameter = while_body->parameter_instruction(0);
    VLOG(2) << "Processing input parameter " << input_parameter->ToString();
    absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
    absl::flat_hash_set<HloInstruction*> skip_control_dep_injection;

    if (exact_trip_count % 2) {
      VLOG(2) << "Found loops with odd trip count, 1 iteration will be peeled "
                 "outside of the main body.";
      TF_RETURN_IF_ERROR(PeelInstructionsForOddTripCount(module, while_instr));
      exact_trip_count -= 1;
    }
    std::string suffix = "double_buffer_clone";
    old_to_new_map[input_parameter] = while_body->root_instruction();
    for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
      if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
        continue;
      }
      VLOG(2) << "Cloning instruction " << old_instr->ToString();
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* old_operand : old_instr->mutable_operands()) {
        new_operands.push_back(old_to_new_map[old_operand]);
      }
      HloInstruction* new_instr =
          while_body->AddInstruction(old_instr->CloneWithNewOperands(
              old_instr->shape(), new_operands, suffix));

      // If an elementwise instruction with constant operand is present, we
      // won't inject control dependency at the end to allow more constant
      // folding opportunities.
      if (old_instr->IsElementwiseBinary() && old_instr->HasConstantOperand()) {
        skip_control_dep_injection.insert(old_instr);
      }
      SetChannelIdForNewCollective(new_instr, module);
      old_to_new_map[old_instr] = new_instr;
      VLOG(2) << "Added instruction " << new_instr->ToString();
    }

    while_body->set_root_instruction(
        old_to_new_map[while_body->root_instruction()]);
    VLOG(2) << "Replaced with new root "
            << while_body->root_instruction()->ToString();

    // Handle existing control dependencies.
    for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
      if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
        HloInstruction* new_instr = old_to_new_map[old_instr];
        VLOG(2) << "Processing control predecessors for "
                << new_instr->ToString();
        std::vector<HloInstruction*> new_control_pred(
            old_instr->control_predecessors().size());
        for (HloInstruction* pred : old_instr->control_predecessors()) {
          new_control_pred.push_back(old_to_new_map[pred]);
        }

        TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
        for (HloInstruction* new_pred : new_control_pred) {
          TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
          VLOG(2) << "Adding " << new_pred->ToString()
                  << " to control dependency of " << new_instr->ToString();
        }
      }
    }
    for (HloInstruction* input_consumer : input_parameter->users()) {
      for (HloInstruction* old_input : input_consumer->users()) {
        if (old_to_new_map.find(old_input) != old_to_new_map.end()) {
          HloInstruction* new_input = old_to_new_map[old_input];
          if (skip_control_dep_injection.find(old_input) ==
                  skip_control_dep_injection.end() &&
              !IsCollective(old_input)) {
            for (HloInstruction* old_root : old_loop_roots) {
              TF_RETURN_IF_ERROR(old_root->AddControlDependencyTo(new_input));
            }
          }
        }
      }
    }
    WhileLoopBackendConfig new_config;
    new_config.mutable_known_trip_count()->set_n((exact_trip_count / 2));
    TF_RETURN_IF_ERROR(while_instr->set_backend_config(new_config));
    changed = true;
  }

  VLOG(2) << "LoopDoubleBufferTransformer output: " << module->ToString();

  // Run necessary cleanup to ensure LoopDoubleBufferTransformer behaves
  // correctly.
  if (changed) {
    // The call graph will not be flat if one of the loops that was unrolled
    // contains any kind of call to another computation---since the call will
    // be duplicated, thereby adding a second callsite for that computation.
    TF_RETURN_IF_ERROR(
        FlattenCallGraph().Run(module, execution_threads).status());
  }

  return changed;
}

}  // end namespace gpu
}  // end namespace xla
