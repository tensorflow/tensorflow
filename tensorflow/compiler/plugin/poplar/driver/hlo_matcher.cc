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

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

using ::tensorflow::strings::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

void CopyConvolutionData(HloInstruction* inst, const HloInstruction* old) {
  inst->set_window(old->window());
  inst->set_convolution_dimension_numbers(old->convolution_dimension_numbers());
}

void CopyMetadataToInstruction(HloInstruction* inst,
                               const HloInstruction* old) {
  // Copy instruction data
  // Note that some data is protected by opcode and can't be copied
  switch (old->opcode()) {
    case HloOpcode::kCall:
      if (IsPopOpsConvolution(old)) {
        CopyConvolutionData(inst, old);
      }
      break;
    case HloOpcode::kConvolution:
      CopyConvolutionData(inst, old);
      break;
    default:
      break;
  }

  inst->set_metadata(old->metadata());
}
}  // namespace

HloMatcher::HloMatcher(const std::vector<HloMatcherPattern>& patterns,
                       bool root_computation_only)
    : root_computation_only_(root_computation_only),
      patterns_(std::move(patterns)) {
  matches_.resize(patterns.size());
}

bool HloMatcher::MatchPattern(HloInstruction* root,
                              const HloMatcherPattern& pattern,
                              HloMatcherMatched& match) {
  match.instructions[0] = root;

  for (unsigned int node_num = 1; node_num < pattern.size(); node_num++) {
    match.instructions[node_num] = nullptr;
  }

  for (unsigned int node_num = 0; node_num < pattern.size(); node_num++) {
    HloInstruction* inst = match.instructions[node_num];
    if (inst == nullptr) {
      return false;
    }

    const HloMatcherNode& node(pattern[node_num]);

    if (node.opcode != HloOpcode::kParameter) {
      if (node.opcode != inst->opcode()) {
        return false;
      }
    }

    if (node.verification_fn && !node.verification_fn(inst)) {
      return false;
    }

    if (node.include_in_replacement) {
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
  for (unsigned int node_num = 0; node_num < pattern.size(); node_num++) {
    const HloMatcherNode& node(pattern[node_num]);

    if (node.include_in_replacement) {
      replaced.push_back(match.instructions[node_num]);
    } else {
      if (match.parameters.size() <= node.parameter_index) {
        match.parameters.resize(node.parameter_index + 1);
      }
      HloInstruction* user;
      for (auto* u : match.instructions[node_num]->users()) {
        for (auto* r : replaced) {
          if (r == u) {
            user = u;
            break;
          }
        }
      }
      if (!user) {
        LOG(FATAL) << "User instruction cannot be found";
      }
      int64 index = user->operand_index(match.instructions[node_num]);
      match.parameters[node.parameter_index] = std::make_pair(user, index);
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
}

// TODO - make this non-recursive
void HloMatcher::MatchPatternStart(HloComputation* computation,
                                   HloInstruction* instruction) {
  visited_.insert(instruction);

  for (unsigned i = 0; i < patterns_.size(); i++) {
    if (instruction->opcode() == patterns_[i][0].opcode) {
      // Try matching the whole pattern
      HloMatcherMatched match;
      match.ok = true;
      match.computation = computation;
      match.instructions.resize(patterns_[i].size());

      if (MatchPattern(instruction, patterns_[i], match)) {
        AddMatch(i, match);
      }
    }
  }

  for (HloInstruction* operand : instruction->operands()) {
    if (visited_.count(operand) == 0) {
      MatchPatternStart(computation, operand);
    }
  }
}

StatusOr<bool> HloMatcher::Run(HloModule* module) {
  if (root_computation_only_) {
    HloComputation* comp = module->entry_computation();
    visited_.clear();
    MatchPatternStart(comp, comp->root_instruction());

  } else {
    // Copy list of computations as we will be introducing new ones
    std::vector<HloComputation*> comps(module->computations().begin(),
                                       module->computations().end());

    for (auto* comp : comps) {
      if (!comp->IsFusionComputation()) {
        visited_.clear();
        MatchPatternStart(comp, comp->root_instruction());
      }
    }
  }

  unsigned int replacement_count = 0;
  for (int pattern = 0; pattern < matches_.size(); pattern++) {
    for (HloMatcherMatched& match : matches_[pattern]) {
      if (match.ok) {
        const ReplacedInstructions& replaced = ReplaceNodes(pattern, match);
        for (auto i : replaced) {
          auto range = match_map_.equal_range(i);
          for (auto m = range.first; m != range.second; ++m) {
            m->second->ok = false;
          }

          replacement_count++;
        }
      }
    }
  }

  patterns_.clear();
  visited_.clear();
  matches_.clear();
  match_map_.clear();

  return replacement_count != 0;
}

ReplacedInstructions HloMatcher::OutlineExpressionFromComputation(
    const HloMatcherMatched& matched,
    const std::string& outlined_computation_name, const char metadata_index) {
  auto& instructions_to_outline = matched.instructions;
  HloModule* module = matched.computation->parent();
  HloInstruction* root = instructions_to_outline[0];

  std::vector<HloInstruction*> to_outline = instructions_to_outline;
  std::reverse(to_outline.begin(), to_outline.end());

  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their new counterparts
  std::unordered_map<HloInstruction*, HloInstruction*> outlined;

  std::vector<HloInstruction*> arguments(matched.parameters.size());

  for (HloInstruction* instruction_to_outline : to_outline) {
    if (outlined.find(instruction_to_outline) == outlined.end()) {
      auto* new_inst = builder.AddInstruction(instruction_to_outline->Clone());
      outlined[instruction_to_outline] = new_inst;

      for (int64 operand = 0; operand < new_inst->operand_count(); ++operand) {
        HloInstruction* old_operand = new_inst->mutable_operand(operand);
        HloInstruction** operand_slot = &(outlined[old_operand]);
        if (*operand_slot == nullptr) {
          int parameter_num = -1;
          for (auto* old_user : old_operand->users()) {
            for (int i = 0; i < matched.parameters.size(); i++) {
              auto& param = matched.parameters[i];
              if (param.first == old_user &&
                  old_user->operand_count() > param.second &&
                  old_user->operand(param.second) == old_operand) {
                parameter_num = i;
                break;
              }
            }
            if (parameter_num != -1) {
              break;
            }
          }

          if (parameter_num != -1) {
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

  // Creates a call to the nested computation.
  HloComputation* nested_computation =
      module->AddEmbeddedComputation(builder.Build(FindOrDie(outlined, root)));

  HloInstruction* call = matched.computation->AddInstruction(
      HloInstruction::CreateCall(root->shape(), arguments, nested_computation));

  CopyMetadataToInstruction(call, instructions_to_outline[metadata_index]);
  TF_CHECK_OK(root->ReplaceAllUsesWith(call));

  ReplacedInstructions replaced;
  for (auto inst : instructions_to_outline) {
    if (inst->user_count() == 0) {
      TF_CHECK_OK(matched.computation->RemoveInstruction(inst));
      replaced.push_back(inst);
    }
  }

  return replaced;
}

}  // namespace poplarplugin
}  // namespace xla
