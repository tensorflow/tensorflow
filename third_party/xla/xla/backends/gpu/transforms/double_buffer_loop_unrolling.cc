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
#include "xla/backends/gpu/transforms/double_buffer_loop_unrolling.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using StaticLoopIteration = DoubleBufferLoopUnrolling::StaticLoopIteration;
using DynamicLoopIteration = DoubleBufferLoopUnrolling::DynamicLoopIteration;
using LoopIteration = DoubleBufferLoopUnrolling::LoopIteration;

DynamicSliceConfig DoubleBufferLoopUnrolling::MakeConfigForLoopIteration(
    const DynamicSliceConfig& config,
    const DoubleBufferLoopUnrolling::LoopIteration& loop_iteration) {
  if (!config.has_loop_index()) {
    return config;
  }
  DynamicSliceConfig new_config = config;
  if (const auto* dyn = std::get_if<DynamicLoopIteration>(&loop_iteration)) {
    new_config.set_byte_offset(config.byte_offset() +
                               dyn->start_iteration * config.byte_stride());
    new_config.set_byte_stride(config.byte_stride() * dyn->iteration_stride);
    return new_config;
  }
  const auto& stat = std::get<StaticLoopIteration>(loop_iteration);
  new_config.set_byte_offset(config.byte_offset() +
                             stat.iteration * config.byte_stride());
  new_config.set_byte_stride(0);
  new_config.clear_loop_index();
  return new_config;
}

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
    }
  } else if (hlo_query::IsCollectiveCommunicationOp(new_instr->opcode()) ||
             hlo_query::IsAsyncCollectiveStartOp(new_instr)) {
    new_instr->set_channel_id(hlo_query::NextChannelId(*module));
  }
}

using Interval = std::pair<int64_t, int64_t>;

bool IsDynamicSliceOp(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kDynamicSlice ||
         instr->opcode() == HloOpcode::kDynamicUpdateSlice;
}

absl::Status AdjustDynamicSliceConfig(HloInstruction* instr,
                                      const LoopIteration& loop_iteration) {
  if (!IsDynamicSliceOp(instr) || !instr->has_backend_config()) {
    return absl::OkStatus();
  }
  return instr->MutateBackendConfig<GpuBackendConfig>(
      [&](GpuBackendConfig* gbc) -> absl::Status {
        if (!gbc->has_dynamic_slice_config()) {
          return absl::OkStatus();
        }
        DynamicSliceConfig* config = gbc->mutable_dynamic_slice_config();
        // loop_index > 0 means the offset depends on a loop outside the one
        // being unrolled here, so its config is unaffected by this
        // transformation.
        if (!config->has_loop_index() || config->loop_index() != 0) {
          return absl::OkStatus();
        }
        *config = DoubleBufferLoopUnrolling::MakeConfigForLoopIteration(
            *config, loop_iteration);
        return absl::OkStatus();
      });
}

// Adjusts the DS/DUS DynamicSliceConfig on `instr` and, if it is a fusion,
// recurses into its fused computation. Fusion computations are cloned with the
// fusion instruction, so each clone has its own private fused computation and
// no cross-clone deduplication is needed.
absl::Status AdjustDynamicSliceConfigsForInstruction(
    HloInstruction* instr, const LoopIteration& loop_iteration) {
  RETURN_IF_ERROR(AdjustDynamicSliceConfig(instr, loop_iteration));
  if (instr->opcode() == HloOpcode::kFusion) {
    for (HloComputation* called_computation : instr->called_computations()) {
      for (HloInstruction* nested : called_computation->instructions()) {
        RETURN_IF_ERROR(
            AdjustDynamicSliceConfigsForInstruction(nested, loop_iteration));
      }
    }
  }
  return absl::OkStatus();
}

// This function fixes the `_xla_send_recv_validation` attribute for peeled
// instructions. When the loop trip count is odd, the peeled instructions are
// moved before the loop. The collectives in these instructions correspond to
// the first iteration of the original loop. We have to run this peeled
// collective for all those devices that had the 0-th iteration as a valid
// iteration.

// This function fixes the `_xla_send_recv_validation` attribute for the two new
// collectives inside the loop. The calculation of the new valid iterations
// depends on whether the loop was peeled or not.
//
// If the loop was not peeled, then
//  - iteration 0 of the new loop corresponds to iteration 0,1 of the old loop.
//  - iteration 1 of the new loop corresponds to iteration 2,3 of the old loop.
//  - and so on...
//
// If the loop was peeled, then the first iteration runs before the loop. So,
//  - iteration 0 of the new loop corresponds to iteration 1,2 of the old loop.
//  - iteration 1 of the new loop corresponds to iteration 3,4 of the old loop.
//  - and so on...
//
// Consider the case when the loop was peeled, and the original attribute for
// some device was {4,7}. Consider that the two new collectives are
// `collective.1` and `collective.2` (they execute in this order inside the new
// loop). In the old loop, iterations 4,5,6,7 were valid. In the new
// loop,
//  - collective.2 in iteration 1 of new loop runs 4th iteration of old loop.
//  - collective.1 in iteration 2 of new loop runs 5th iteration of old loop.
//  - collective.2 in iteration 2 of new loop runs 6th iteration of old loop.
//  - collective.1 in iteration 3 of new loop runs 7th iteration of old loop.
// So, the updated attribute for that device are {1,2} for `collective.2` and
// {2,3} for `collective.1`.
//
// In a similar fashion we can generalize the computation of new values based on
// the values of the old attribute as done in the logic below.

// Handle control predecessors/successors for every old-new instruction pair.
// For every new instruction, we find the relevant predecessor/successor
// relationships of the old instruction and we reconstruct them by looking up
// new (already created) predecessors/successors.
//
// When rewiring dependencies from output of the original body, to the input of
// the cloned body we skip collectives, and ops in `skip_control_dep_injection`.
absl::Status HandleControlDependencies(
    const HloComputation* while_body,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>& old_to_new_map,
    HloInstruction::InstructionVector* old_loop_roots,
    HloInstruction* input_parameter,
    const absl::flat_hash_set<HloInstruction*>& skip_control_dep_injection) {
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      HloInstruction* new_instr = old_to_new_map.at(old_instr);
      VLOG(2) << "Processing control predecessors for "
              << new_instr->ToString();
      std::vector<HloInstruction*> new_control_pred;
      new_control_pred.reserve(old_instr->control_predecessors().size());
      for (HloInstruction* pred : old_instr->control_predecessors()) {
        if (!old_to_new_map.contains(pred)) {
          continue;
        }
        new_control_pred.push_back(old_to_new_map.at(pred));
      }

      RETURN_IF_ERROR(new_instr->DropAllControlDeps());
      for (HloInstruction* new_pred : new_control_pred) {
        RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
        VLOG(2) << "Adding " << new_pred->ToString()
                << " to control dependency of " << new_instr->ToString();
      }
    }
  }
  for (HloInstruction* input_consumer : input_parameter->users()) {
    for (HloInstruction* old_input : input_consumer->users()) {
      if (old_to_new_map.find(old_input) != old_to_new_map.end()) {
        HloInstruction* new_input = old_to_new_map.at(old_input);
        if (skip_control_dep_injection.find(old_input) ==
                skip_control_dep_injection.end() &&
            !IsCollective(old_input)) {
          for (HloInstruction* old_root : *old_loop_roots) {
            RETURN_IF_ERROR(old_root->AddControlDependencyTo(new_input));
          }
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> FullyUnroll(HloInstruction* while_instr,
                                 HloModule* module) {
  HloComputation* while_body = while_instr->while_body();
  bool changed = false;
  VLOG(2) << "Processing root " << while_body->root_instruction()->ToString();

  auto loop_roots = while_body->root_instruction()->mutable_operands();
  HloInstruction* input_parameter = while_body->parameter_instruction(0);
  VLOG(2) << "Processing input parameter " << input_parameter->ToString();

  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
  absl::flat_hash_set<HloInstruction*> skip_control_dep_injection;
  std::string clone_suffix = "full_unroll_clone";

  ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                   while_instr->backend_config<WhileLoopBackendConfig>());
  std::vector<HloInstruction*> ops_to_clone;
  ops_to_clone.reserve(while_body->MakeInstructionPostOrder().size());

  // Pre-loop prep.
  HloInstruction* old_input_parameter = input_parameter;
  HloInstruction* new_input_parameter = while_body->root_instruction();
  absl::flat_hash_set<HloInstruction*> seen_ops;
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (seen_ops.contains(old_instr)) {
      continue;
    }
    ops_to_clone.push_back(old_instr);
    seen_ops.insert(old_instr);
  }

  // Adjustment is deferred to the end because each unrolled iteration's
  // clones become the source operands for the next iteration's clones. If we
  // adjusted clones in place, `CloneWithNewOperands` in iteration i+1 would
  // copy an already-flattened DynamicSliceConfig and miscompute its offset.
  std::vector<std::pair<HloInstruction*, int64_t>> static_iteration_for;
  static_iteration_for.reserve(config.known_trip_count().n() *
                               ops_to_clone.size());
  for (HloInstruction* instr : ops_to_clone) {
    static_iteration_for.emplace_back(instr, 0);
  }

  for (int64_t i = 1; i < config.known_trip_count().n(); ++i) {
    std::vector<HloInstruction*> new_ops_to_clone;
    old_to_new_map[old_input_parameter] = new_input_parameter;
    for (HloInstruction* old_instr : ops_to_clone) {
      if (old_to_new_map.contains(old_instr)) {
        continue;
      }
      VLOG(2) << "Cloning instruction " << old_instr->ToString();
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* old_operand : old_instr->mutable_operands()) {
        new_operands.push_back(old_to_new_map[old_operand]);
      }
      HloInstruction* new_instr =
          while_body->AddInstruction(old_instr->CloneWithNewOperands(
              old_instr->shape(), new_operands, clone_suffix));

      // If an elementwise instruction with constant operand is present, we
      // won't inject control dependency at the end to allow more constant
      // folding opportunities.
      if (old_instr->IsElementwiseBinary() && old_instr->HasConstantOperand()) {
        skip_control_dep_injection.insert(old_instr);
      }
      SetChannelIdForNewCollective(new_instr, module);
      old_to_new_map[old_instr] = new_instr;
      new_ops_to_clone.push_back(new_instr);
      static_iteration_for.emplace_back(new_instr, i);
      VLOG(2) << "Added instruction " << new_instr->ToString();
    }

    while_body->set_root_instruction(
        old_to_new_map[while_body->root_instruction()]);
    VLOG(2) << "Replaced with new root "
            << while_body->root_instruction()->ToString();

    RETURN_IF_ERROR(HandleControlDependencies(while_body, old_to_new_map,
                                              &loop_roots, old_input_parameter,
                                              skip_control_dep_injection));

    // Inductive step update, clean/update necessary buffers to prepare them for
    // the next unrolling iteration.
    old_to_new_map.clear();
    skip_control_dep_injection.clear();
    loop_roots = while_body->root_instruction()->mutable_operands();
    old_input_parameter = new_input_parameter;
    new_input_parameter = while_body->root_instruction();
    ops_to_clone = std::move(new_ops_to_clone);
    changed = true;
  }

  for (const auto& [instr, iteration] : static_iteration_for) {
    RETURN_IF_ERROR(AdjustDynamicSliceConfigsForInstruction(
        instr, StaticLoopIteration{iteration}));
  }

  WhileLoopBackendConfig old_config;
  ASSIGN_OR_RETURN(old_config,
                   while_instr->backend_config<WhileLoopBackendConfig>());

  WhileLoopBackendConfig new_config = old_config;
  new_config.mutable_known_trip_count()->set_n(1);

  RETURN_IF_ERROR(while_instr->set_backend_config(new_config));

  return changed;
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

  std::vector<HloInstruction*> old_instructions =
      while_body->MakeInstructionPostOrder();
  for (HloInstruction* old_instr : old_instructions) {
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
    RETURN_IF_ERROR(AdjustDynamicSliceConfigsForInstruction(
        new_instr, StaticLoopIteration{0}));
    old_to_new_map[old_instr] = new_instr;
    VLOG(2) << "Added instruction " << new_instr->ToString()
            << " to parent computation.";
  }

  std::vector<HloInstruction*> new_roots;
  for (HloInstruction* instr : old_loop_roots) {
    new_roots.push_back(old_to_new_map[instr]);
  }
  RETURN_IF_ERROR(while_instr->ReplaceOperandWith(
      0, old_to_new_map[while_body->root_instruction()]));
  VLOG(2) << "Replaced with new input tuple "
          << while_instr->operand(0)->ToString();

  // Handle existing control dependencies.
  for (HloInstruction* old_instr : old_instructions) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      HloInstruction* new_instr = old_to_new_map[old_instr];
      VLOG(2) << "Processing control predecessors for peeled instruction "
              << new_instr->ToString();
      std::vector<HloInstruction*> new_control_pred;
      new_control_pred.reserve(old_instr->control_predecessors().size());
      for (HloInstruction* pred : old_instr->control_predecessors()) {
        new_control_pred.push_back(old_to_new_map[pred]);
      }

      RETURN_IF_ERROR(new_instr->DropAllControlDeps());
      for (HloInstruction* new_pred : new_control_pred) {
        RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
        VLOG(2) << "Adding " << new_pred->ToString()
                << " to control dependency of peeled instruction: "
                << new_instr->ToString();
      }
    }
  }
  return absl::OkStatus();
}

// TODO(olechwierowicz): Extract common logic of this and `FullyUnroll` to
// a separate function.
absl::StatusOr<bool> DoubleBufferingUnroll(HloInstruction* while_instr,
                                           HloModule* module) {
  ASSIGN_OR_RETURN(auto config,
                   while_instr->backend_config<WhileLoopBackendConfig>());

  CHECK(config.has_known_trip_count())
      << "Only loops with known trip count are supported.";
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

  bool peel_one_iteration = exact_trip_count % 2;
  if (peel_one_iteration) {
    VLOG(2) << "Found loops with odd trip count, 1 iteration will be peeled "
               "outside of the main body.";
    RETURN_IF_ERROR(PeelInstructionsForOddTripCount(module, while_instr));
    exact_trip_count -= 1;
  }

  constexpr int64_t kUnrollFactor = 2;
  constexpr int64_t kFirstUnrollCopy = 0;
  constexpr int64_t kSecondUnrollCopy = 1;
  std::string suffix = "double_buffer_clone";
  old_to_new_map[input_parameter] = while_body->root_instruction();
  std::vector<HloInstruction*> old_instructions =
      while_body->MakeInstructionPostOrder();
  for (HloInstruction* old_instr : old_instructions) {
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
    RETURN_IF_ERROR(AdjustDynamicSliceConfigsForInstruction(
        new_instr, DynamicLoopIteration{/*start_iteration=*/peel_one_iteration +
                                            kSecondUnrollCopy,
                                        /*iteration_stride=*/kUnrollFactor}));
    old_to_new_map[old_instr] = new_instr;
    VLOG(2) << "Added instruction " << new_instr->ToString();
  }

  // Originals stay in the unrolled loop and execute the first copy of every
  // pair, with stride doubled.
  for (HloInstruction* old_instr : old_instructions) {
    RETURN_IF_ERROR(AdjustDynamicSliceConfigsForInstruction(
        old_instr, DynamicLoopIteration{/*start_iteration=*/peel_one_iteration +
                                            kFirstUnrollCopy,
                                        /*iteration_stride=*/kUnrollFactor}));
  }

  while_body->set_root_instruction(
      old_to_new_map[while_body->root_instruction()]);
  VLOG(2) << "Replaced with new root "
          << while_body->root_instruction()->ToString();

  // Handle existing control dependencies.
  RETURN_IF_ERROR(HandleControlDependencies(while_body, old_to_new_map,
                                            &old_loop_roots, input_parameter,
                                            skip_control_dep_injection));

  WhileLoopBackendConfig new_config = config;
  new_config.mutable_known_trip_count()->set_n(exact_trip_count / 2);

  // Update the init/step metadata if it was present before.
  if (config.has_known_init_step()) {
    int64_t step = config.known_init_step().step();
    new_config.mutable_known_init_step()->set_step(step * 2);
    new_config.mutable_known_init_step()->set_init(
        config.known_init_step().init() + (peel_one_iteration ? step : 0));
  }

  for (auto& dv : *new_config.mutable_dynamic_variables()) {
    if (!dv.has_init() || !dv.has_step()) {
      continue;
    }
    int64_t dv_step = dv.step();
    dv.set_step(dv_step * 2);
    if (peel_one_iteration) {
      dv.set_init(dv.init() + dv_step);
    }
  }

  RETURN_IF_ERROR(while_instr->set_backend_config(new_config));
  return true;  // changed
}

// Function performs double buffering unrolling strategy iff there is any
// collective operation within a body computation.
absl::StatusOr<bool> AutoUnroll(HloInstruction* while_instr,
                                HloModule* module) {
  CHECK_EQ(while_instr->opcode(), HloOpcode::kWhile);

  bool any_collective_present = absl::c_any_of(
      while_instr->while_body()->MakeInstructionPostOrder(),
      [](HloInstruction* instr) {
        return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
      });
  if (any_collective_present) {
    return DoubleBufferingUnroll(while_instr, module);
  }
  return false;  // IR not changed.
}

}  // namespace

absl::StatusOr<bool> DoubleBufferLoopUnrolling::RunImpl(
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
    ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                     while_instr->backend_config<WhileLoopBackendConfig>());
    if (!config.has_known_trip_count()) {
      VLOG(2) << while_instr->ToString()
              << " doesn't have exact trip count, skipping loop unrolling.";
      continue;
    }

    if (config.known_trip_count().n() == 1) {
      VLOG(2) << while_instr->ToString()
              << " has an iteration count of one, skipping unrolling.";
      continue;
    }
    absl::string_view manual_unroll_attr_val = "";
    auto& attrs = while_instr->frontend_attributes().map();
    if (auto it = attrs.find(kXlaLoopUnrollAttr); it != attrs.end()) {
      manual_unroll_attr_val = it->second;
      if (!(absl::EqualsIgnoreCase(manual_unroll_attr_val,
                                   kManualUnrollDoubleBuffer) ||
            absl::EqualsIgnoreCase(manual_unroll_attr_val,
                                   kManualUnrollFull))) {
        LOG(FATAL) << "Invalid value for " << kXlaLoopUnrollAttr
                   << " attribute: " << manual_unroll_attr_val
                   << ". Expected values are: " << kManualUnrollDoubleBuffer
                   << ", " << kManualUnrollFull;
      }
    }
    if (unroll_strategy_ == UnrollStrategy::kFullUnroll) {
      ASSIGN_OR_RETURN(changed, FullyUnroll(while_instr, module));
    } else if (unroll_strategy_ == UnrollStrategy::kDoubleBuffer) {
      ASSIGN_OR_RETURN(changed, DoubleBufferingUnroll(while_instr, module));
    } else if (unroll_strategy_ == UnrollStrategy::kAuto) {
      ASSIGN_OR_RETURN(changed, AutoUnroll(while_instr, module));
    } else if (unroll_strategy_ == UnrollStrategy::kManual) {
      if (absl::EqualsIgnoreCase(manual_unroll_attr_val, kManualUnrollFull)) {
        ASSIGN_OR_RETURN(changed, FullyUnroll(while_instr, module));
      }
      if (absl::EqualsIgnoreCase(manual_unroll_attr_val,
                                 kManualUnrollDoubleBuffer)) {
        ASSIGN_OR_RETURN(changed, DoubleBufferingUnroll(while_instr, module));
      }
    } else {
      LOG(FATAL) << absl::StrCat("Unhandled unrolling strategy: ",
                                 unroll_strategy_);
    }
  }

  VLOG(2) << "LoopDoubleBufferTransformer output: " << module->ToString();

  // Run necessary cleanup to ensure LoopDoubleBufferTransformer behaves
  // correctly.
  if (changed) {
    // The call graph will not be flat if one of the loops that was unrolled
    // contains any kind of call to another computation---since the call will
    // be duplicated, thereby adding a second callsite for that computation.
    RETURN_IF_ERROR(FlattenCallGraph().Run(module, execution_threads).status());
  }

  return changed;
}

}  // end namespace gpu
}  // end namespace xla
