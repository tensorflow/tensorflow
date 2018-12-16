/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>
#include <stack>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

void HloMatcherPattern::Verify() {
  if (outputs.size() == 0) {
    LOG(FATAL) << "Pattern " << type
               << " has no outputs, at least one required.";
  }
  if (outputs.size() > 1) {
    LOG(FATAL) << "Pattern " << type
               << " has multiple outputs, currently not supported.";
  }
}

HloMatcher::HloMatcher(const std::vector<HloMatcherPattern>& patterns,
                       struct CompilerAnnotations& annotations,
                       bool root_computation_only,
                       unsigned look_through_max_depth)
    : patterns_(std::move(patterns)),
      annotations_(annotations),
      root_computation_only_(root_computation_only),
      look_through_max_depth_(look_through_max_depth) {}

// A set of sets of ops which are all associative together
static std::set<std::set<HloOpcode>> associative_ops_sets = {
    {HloOpcode::kMultiply},
    {HloOpcode::kAdd},
};

StatusOr<Trace> HloMatcher::FindNextMatchingOp(HloInstruction* user,
                                               HloInstruction* inst,
                                               const HloOpcode desiredOpcode) {
  for (const auto ops_set : associative_ops_sets) {
    // user needs to be an associative op
    if (!ops_set.count(user->opcode())) {
      continue;
    }

    // Non recursive depth first DAG traversal to try and find an inst with
    // right opcode using associativity
    std::stack<Trace> to_visit;
    // The list of instructions visited while searching for each pattern
    std::set<HloInstruction*> visited = {user};

    // Traverse from inst
    Trace start_trace = {{user, user->operand_index(inst)}};
    to_visit.push(start_trace);
    while (!to_visit.empty()) {
      // Get value of the stack
      auto current = to_visit.top();
      to_visit.pop();

      HloInstruction* current_inst =
          current.back().inst->mutable_operand(current.back().op_idx);
      visited.insert(current_inst);
      // Check if the current instruction matches
      if (current_inst->opcode() == desiredOpcode) {
        current.push_back({current_inst, -1});
        return current;
      }

      // Check the current instruction is associative and matches the shape,
      // if not then we can't look through it
      if (!(ops_set.count(current_inst->opcode()) &&
            ShapeUtil::Equal(current_inst->shape(), inst->shape()))) {
        continue;
      }

      // Add operands to visit without going past the maximum search depth
      if (current.size() - 1 < look_through_max_depth_) {
        for (int64 i = 0; i < current_inst->operand_count(); i++) {
          auto* operand = current_inst->mutable_operand(i);
          // Only add the operand if:
          // * we have never seen it before
          // * it has one user
          // * it has the same shape
          if (visited.count(operand) == 0 && operand->user_count() == 1 &&
              ShapeUtil::Equal(operand->shape(), inst->shape())) {
            // We need to know which operand will be replaced at the root
            // instruction - we only need to know this on depth 0, otherwise
            // keep it the same
            auto next_trace = current;
            next_trace.push_back({current_inst, i});
            to_visit.push(next_trace);
          }
        }
      }
    }
  }
  return tensorflow::errors::Internal("no_match");
}

bool HloMatcher::MatchPattern(HloInstruction* root,
                              const HloMatcherPattern& pattern,
                              HloMatcherMatched& match) {
  match.instructions[0] = root;

  // Construct a mapping from a pattern node to all other pattern nodes which
  // use it
  std::vector<std::set<std::pair<unsigned int, unsigned int>>> node_mapping(
      pattern.pattern.size());

  // Create lookup for input indexes to parameter number
  std::map<NodeId, int64> input_id_to_param_num;
  for (int64 i = 0; i < pattern.inputs.size(); i++) {
    input_id_to_param_num[pattern.inputs[i]] = i;
  }
  const auto is_input = [&input_id_to_param_num](const NodeId pid) {
    return input_id_to_param_num.count(pid);
  };

  for (unsigned int node_num = 0; node_num < pattern.pattern.size();
       node_num++) {
    for (unsigned int op_idx = 0;
         op_idx < pattern.pattern[node_num].operands.size(); op_idx++) {
      node_mapping[pattern.pattern[node_num].operands[op_idx]].insert(
          {node_num, op_idx});
    }

    if (node_num) {
      match.instructions[node_num] = nullptr;
    }
  }

  for (unsigned int node_num = 0; node_num < pattern.pattern.size();
       node_num++) {
    HloInstruction* inst = match.instructions[node_num];
    if (inst == nullptr) {
      return false;
    }

    const HloMatcherNode& node(pattern.pattern[node_num]);

    if (node.opcode != HloOpcode::kParameter) {
      if (node.opcode != inst->opcode()) {
        // Try to find an op using associativity, unless this is the first node
        // or search depth is 0 or this inst is used more than once
        if (node_num == 0 || look_through_max_depth_ == 0 ||
            inst->user_count() != 1) {
          return false;
        }
        unsigned int user_node_num = node_mapping[node_num].begin()->first;
        auto* user = match.instructions[user_node_num];
        auto status_or = FindNextMatchingOp(user, inst, node.opcode);
        // Check whether we managed to find a match
        if (!status_or.ok()) {
          return false;
        }
        Trace found = status_or.ValueOrDie();
        match.instructions[node_num] = found.back().inst;
        inst = found.back().inst;
        match.replacement_traces.push_back(found);
      }
    }

    if (node.node_condition && !(*node.node_condition)(inst)) {
      return false;
    }

    if (!is_input(node_num)) {
      if ((node.operands.size() > 0) &&
          (inst->operand_count() != node.operands.size())) {
        return false;
      }

      for (unsigned int i = 0; i < node.operands.size(); i++) {
        HloInstruction* operand = inst->mutable_operand(i);
        int n = node.operands[i];

        if (n >= match.instructions.size()) {
          LOG(FATAL) << "Invalid matcher reference " << n;
        }

        if (match.instructions[n] != nullptr) {
          // Instructions can only match once
          if (match.instructions[n] != operand) {
            return false;
          }
        } else {
          // Each instruction can match only one entry in the pattern
          for (auto* op : match.instructions) {
            if (op == operand) {
              return false;
            }
          }

          match.instructions[n] = operand;
        }
      }
    }
  }

  ReplacedInstructions replaced;
  for (unsigned int node_num = 0; node_num < pattern.pattern.size();
       node_num++) {
    const HloMatcherNode& node(pattern.pattern[node_num]);

    if (!is_input(node_num)) {
      // If we are including the instruction, then for each operand:
      // * if it's a parameter, insert the index
      // * else insert -1
      auto* inst = match.instructions[node_num];
      std::transform(
          node.operands.begin(), node.operands.end(),
          std::back_inserter(match.inst_parameters[inst]),
          [&](const unsigned int& op_idx) {
            return is_input(op_idx) ? input_id_to_param_num.at(op_idx) : -1;
          });
      replaced.push_back(inst);
    }
  }
  match.instructions = std::move(replaced);

  return true;
}

void HloMatcher::AddMatch(unsigned pattern, const HloMatcherMatched& match) {
  matches_[pattern].push_back(match);
  for (unsigned i = 0; i < match.instructions.size(); i++) {
    match_map_.insert(
        std::make_pair(match.instructions[i], &matches_[pattern].back()));
  }
  // We also need to add all the instructions in the associativity traces as
  // these are modified/values change
  for (auto trace : match.replacement_traces) {
    for (auto pair : trace) {
      match_map_.insert(std::make_pair(pair.inst, &matches_[pattern].back()));
    }
  }
}

void HloMatcher::MatchPatternStart(HloComputation* computation,
                                   HloInstruction* root) {
  // Non recursive depth first DAG traversal to match the patterns
  std::stack<HloInstruction*> to_visit;
  // The list of instructions visited while searching for each pattern
  std::set<HloInstruction*> visited;

  // Traverse from root
  to_visit.push(root);
  while (!to_visit.empty()) {
    HloInstruction* instruction = to_visit.top();
    to_visit.pop();
    visited.insert(instruction);

    for (unsigned i = 0; i < patterns_.size(); i++) {
      if (instruction->opcode() == patterns_[i].pattern[0].opcode) {
        // Try matching the whole pattern
        HloMatcherMatched match;
        match.ok = true;
        match.computation = computation;
        match.instructions.resize(patterns_[i].pattern.size());
        if (MatchPattern(instruction, patterns_[i], match)) {
          AddMatch(i, match);
        }
      }
    }

    for (HloInstruction* operand : instruction->operands()) {
      if (visited.count(operand) == 0) {
        to_visit.push(operand);
      }
    }
  }
}

StatusOr<bool> HloMatcher::Run(HloModule* module) {
  matches_.clear();
  matches_.resize(patterns_.size());
  match_map_.clear();

  if (root_computation_only_) {
    HloComputation* comp = module->entry_computation();
    MatchPatternStart(comp, comp->root_instruction());

  } else {
    // Copy list of computations as we will be introducing new ones
    std::vector<HloComputation*> comps(module->computations().begin(),
                                       module->computations().end());

    for (auto* comp : comps) {
      if (!comp->IsFusionComputation() && !IsPopOpsCall(comp)) {
        MatchPatternStart(comp, comp->root_instruction());
      }
    }
  }

  unsigned int replacement_count = ReplaceNodes();

  matches_.clear();
  match_map_.clear();

  return replacement_count != 0;
}

std::set<HloInstruction*> HloMatcher::ReorderGraph(
    const HloMatcherMatched& matched) {
  std::set<HloInstruction*> modified_instructions;
  for (auto trace : matched.replacement_traces) {
    auto root = trace[0];
    auto target_user = trace.rbegin()[1];  // second to last element
    auto target = trace.back();

    root.inst->ReplaceAllUsesWith(target_user.inst);
    target_user.inst->ReplaceOperandWith(target_user.op_idx, root.inst);
    root.inst->ReplaceOperandWith(root.op_idx, target.inst);
    std::transform(
        trace.begin(), trace.end(),
        std::inserter(modified_instructions, modified_instructions.begin()),
        [](InstructionIndex const& x) { return x.inst; });
  }
  return modified_instructions;
}

OutlinedInfo HloMatcher::OutlineExpressionFromComputation(
    const HloMatcherMatched& matched,
    const std::string& outlined_computation_name, const char metadata_index,
    std::vector<HloInstruction*> forced_parameters) {
  HloModule* module = matched.computation->parent();
  // First we need to update the graph with any instructions that will be
  // reordered
  auto modified_instructions = ReorderGraph(matched);

  auto& instructions_to_outline = matched.instructions;
  HloInstruction* root = instructions_to_outline[0];

  std::vector<HloInstruction*> to_outline = instructions_to_outline;
  std::reverse(to_outline.begin(), to_outline.end());

  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their new counterparts
  std::unordered_map<HloInstruction*, HloInstruction*> outlined;

  std::vector<HloInstruction*> arguments;

  for (HloInstruction* instruction_to_outline : to_outline) {
    if (outlined.find(instruction_to_outline) == outlined.end()) {
      auto* new_inst = builder.AddInstruction(instruction_to_outline->Clone());
      outlined[instruction_to_outline] = new_inst;

      for (int64 operand = 0; operand < new_inst->operand_count(); ++operand) {
        HloInstruction* old_operand = new_inst->mutable_operand(operand);
        HloInstruction** operand_slot = &(outlined[old_operand]);
        if (*operand_slot == nullptr) {
          auto op_indecies_it =
              matched.inst_parameters.find(instruction_to_outline);
          if (op_indecies_it == matched.inst_parameters.end()) {
            continue;
          }
          auto parameter_num = op_indecies_it->second[operand];
          if (parameter_num != -1) {
            if (arguments.size() <= parameter_num) {
              arguments.resize(parameter_num + 1);
            }
            arguments[parameter_num] = old_operand;
            *operand_slot =
                builder.AddInstruction(HloInstruction::CreateParameter(
                    parameter_num, old_operand->shape(),
                    StrCat("arg_", parameter_num)));
          }
        }

        if (*operand_slot != nullptr) {
          TF_CHECK_OK(new_inst->ReplaceOperandWith(operand, *operand_slot));
        } else {
          LOG(FATAL) << "Replacement not found for  " << new_inst->name() << ":"
                     << operand << " in outline " << outlined_computation_name;
        }
      }
    }
  }

  for (int i = 0; i < arguments.size(); i++) {
    if (arguments[i] == nullptr) {
      LOG(FATAL) << "Argument " << i << " not found for outline "
                 << outlined_computation_name;
    }
  }

  // Add forced parameters as arguments - DCE doe not remove unused parameters.
  // This allows us to link and easily maintain outputs of a fwd pass to the bwd
  // pass
  for (unsigned i = 0; i < forced_parameters.size(); i++) {
    HloInstruction* inst = forced_parameters[i];
    const unsigned parameter_num = arguments.size() + i;
    builder.AddInstruction(HloInstruction::CreateParameter(
        parameter_num, inst->shape(), StrCat("arg_", parameter_num)));
    arguments.push_back(inst);
  }

  // Creates a call to the nested computation.
  HloComputation* nested_computation =
      module->AddEmbeddedComputation(builder.Build(FindOrDie(outlined, root)));

  HloInstruction* call = matched.computation->AddInstruction(
      HloInstruction::CreateCall(root->shape(), arguments, nested_computation));

  auto* old = instructions_to_outline[metadata_index];
  annotations_.fusion_map[nested_computation] = outlined.at(old);

  call->set_metadata(old->metadata());
  if (old->has_sharding()) {
    call->set_sharding(old->sharding());
  }

  TF_CHECK_OK(root->ReplaceAllUsesWith(call));

  OutlinedInfo outlined_info = {call, {}};
  // Add the removed instructions
  for (auto inst : instructions_to_outline) {
    if (inst->user_count() == 0) {
      TF_CHECK_OK(matched.computation->RemoveInstruction(inst));
      outlined_info.removed_or_modified_instructions.push_back(inst);
    }
  }

  // Add the modified instructions
  outlined_info.removed_or_modified_instructions.insert(
      outlined_info.removed_or_modified_instructions.end(),
      modified_instructions.begin(), modified_instructions.end());

  return outlined_info;
}

unsigned HloMatcher::MarkReplacedInstructions(
    const OutlinedInfo& outlined_info) {
  unsigned int replacement_count = 0;
  for (auto i : outlined_info.removed_or_modified_instructions) {
    auto range = match_map_.equal_range(i);
    for (auto m = range.first; m != range.second; ++m) {
      m->second->ok = false;
    }

    replacement_count++;
  }
  return replacement_count;
}

}  // namespace poplarplugin
}  // namespace xla
