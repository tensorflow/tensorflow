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

#include "xla/service/while_loop_pipeline_unroller.h"

#include <cstdint>
#include <numeric>
#include <stack>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/while_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
/*static*/
int64_t WhileLoopPipelineUnroller::ComputeWhileLoopPipelineDepth(
    const HloInstruction& while_instruction) {
  CHECK_EQ(while_instruction.opcode(), HloOpcode::kWhile);
  const HloComputation* while_body = while_instruction.while_body();

  // Look for pattern param -> gte -> root, where indices in the param and
  // root tuples are mismatching.
  absl::flat_hash_map<int64_t, int64_t> loop_permutations;
  HloInstruction* while_param = while_body->parameter_instruction(0);
  HloInstruction* while_root = while_body->root_instruction();
  CHECK_EQ(while_root->opcode(), HloOpcode::kTuple)
      << "While Instruction has not been canonicalized to have a tuple shape";
  for (int64_t output_index = 0; output_index < while_root->operand_count();
       ++output_index) {
    const HloInstruction* operand = while_root->operand(output_index);
    if (operand->opcode() == HloOpcode::kGetTupleElement &&
        operand->operand(0) == while_param) {
      int64_t input_index = operand->tuple_index();
      if (input_index != output_index) {
        // Don't try to analyze loops with complicated permutation patterns.
        // TODO(vsytch): analyze loops with complicated permutation patterns.
        if (!loop_permutations.contains(input_index)) {
          loop_permutations.emplace(input_index, output_index);
        }
      }
    }
  }

  // Find all indices at which the pipelined chains start from.
  std::vector<int64_t> start_indices;
  absl::flat_hash_set<int64_t> output_indices;
  for (auto&& [_, output_index] : loop_permutations) {
    output_indices.insert(output_index);
  }
  for (auto&& [input_index, _] : loop_permutations) {
    if (!output_indices.contains(input_index)) {
      start_indices.push_back(input_index);
    }
  }

  // Find all pipelining chains.
  std::vector<std::vector<int64_t>> pipelined_chains;
  for (int64_t start_index : start_indices) {
    std::stack<std::pair<int64_t, std::vector<int64_t>>> stack;
    stack.push({start_index, {start_index}});
    while (!stack.empty()) {
      auto [current_index, current_chain] = stack.top();
      stack.pop();
      if (!loop_permutations.contains(current_index)) {
        pipelined_chains.push_back(std::move(current_chain));
      } else {
        int64_t next_index = loop_permutations[current_index];
        current_chain.push_back(next_index);
        stack.emplace(next_index, std::move(current_chain));
      }
    }
  }

  // Compute the pipeline depth of the loop body.
  // https://en.wikipedia.org/wiki/Permutation#Order_of_a_permutation
  int64_t pipeline_depth = 1;
  for (auto&& pipelined_chain : pipelined_chains) {
    pipeline_depth =
        std::lcm<int64_t>(pipelined_chain.size() + 1, pipeline_depth);
  }

  return pipeline_depth;
}

absl::StatusOr<bool> WhileLoopPipelineUnroller::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<std::pair<HloInstruction*, int64_t>> while_instructions;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        int64_t pipeline_depth = ComputeWhileLoopPipelineDepth(*instruction);
        if (pipeline_depth > 1) {
          // The pipeline depth is our unroll factor.
          while_instructions.emplace_back(instruction, pipeline_depth);
        }
      }
    }
  }

  std::vector<HloInstruction*> original_roots;
  for (auto&& [while_instruction, unroll_factor] : while_instructions) {
    HloComputation* body = while_instruction->while_body();
    HloComputation* condition = while_instruction->while_condition();

    // Generate the unrolled loop body. This will call the original body
    // unroll_factor times.
    HloComputation::Builder b(
        absl::StrFormat("%s.unrolled_%dx", body->name(), unroll_factor));
    HloInstruction* input_tuple =
        b.AddInstruction(HloInstruction::CreateParameter(
            0, while_instruction->shape(), "input_tuple"));
    HloComputation* unrolled_body = module->AddEmbeddedComputation(b.Build());
    for (int64_t step = 0; step < unroll_factor; ++step) {
      HloComputation* loop_step = module->AddEmbeddedComputation(body->Clone(
          absl::StrFormat("unrolled_%dx_step_%d", unroll_factor, step)));
      input_tuple = unrolled_body->AddInstruction(HloInstruction::CreateCall(
          while_instruction->shape(), {input_tuple}, loop_step));
      original_roots.push_back(input_tuple);
    }
    // The final original root is now the root of the unrolled loop.
    HloInstruction* unrolled_root = original_roots.back();
    original_roots.pop_back();
    unrolled_body->set_root_instruction(unrolled_root);

    // We need the unrolled loop and the remainder (original) loop to execute
    // a combined number of steps equal to the unroll factor. Since the unrolled
    // loop on each iteration executes unroll_factor steps, we split the
    // work by having the unrolled loop execute num_steps // unroll_factor
    // times, and then the remainder loop will execute num_steps % unroll_factor
    // times. This can be guaranteed by using the original condition for the
    // unrolled loop, but reducing its trip count by (unroll_factor - 1),
    // accounting for the original body execution.
    HloComputation* unrolled_condition = module->AddEmbeddedComputation(
        condition->Clone(absl::StrFormat("unrolled_%dx", unroll_factor)));
    // We don't set the unrolled body right away, as it is non-trivial for
    // IncrementWhileLoopTripCount to find the trip count variable inside the
    // unrolled version.
    HloInstruction* unrolled_while_instruction =
        while_instruction->parent()->AddInstruction(HloInstruction::CreateWhile(
            while_instruction->shape(), unrolled_condition, body,
            while_instruction->mutable_operand(0)));
    TF_RETURN_IF_ERROR(WhileUtil::IncrementWhileLoopTripCount(
        *unrolled_while_instruction, -(unroll_factor - 1)));
    unrolled_while_instruction->set_while_body(unrolled_body);

    TF_RETURN_IF_ERROR(
        while_instruction->ReplaceOperandWith(0, unrolled_while_instruction));
  }

  const bool changed = !while_instructions.empty();
  if (changed) {
    // When we cloned the loop body for each unrolled step, we didn't
    // recursively clone all the nested computations. FCG will take care of this
    // for us.
    FlattenCallGraph fcg;
    TF_RETURN_IF_ERROR(fcg.Run(module).status());
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }

  return changed;
}
}  // namespace xla
