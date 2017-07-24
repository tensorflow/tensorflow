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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

HloMatcher::HloMatcher(const std::vector<HloMatcherPattern>& patterns,
                       bool root_computation_only)
        : root_computation_only_(root_computation_only)
        , patterns_(std::move(patterns)) {
  matches_.resize(patterns.size());
}

bool HloMatcher::MatchPattern(HloInstruction* root,
                              const HloMatcherPattern& pattern,
                              HloMatcherMatched& match) {

  match.instructions[0] = root;

  for (unsigned int node_num=0; node_num < pattern.size(); node_num++) {
    const HloInstruction* inst = match.instructions[node_num];
    if (inst == nullptr) {
      return false;
    }

    const HloMatcherNode& node(pattern[node_num]);

    if (node.opcode != inst->opcode()) {
      return false;
    }

    if (node.verification_fn && !node.verification_fn(inst)) {
      return false;
    }

    if (inst->operand_count() != node.operands.size()) {
      return false;
    }

    for (unsigned int i=0; i<node.operands.size(); i++) {
      int n = node.operands[i];
      if (n == -1) continue;
      if (n <= node_num) continue;

      match.instructions[n] = inst->operand(i);
    }
  }

  return true;
}

void HloMatcher::AddMatch(unsigned pattern, const HloMatcherMatched& match) {
  matches_[pattern].push_back(match);
  for (unsigned i=0; i<match.instructions.size(); i++) {
    match_map_.insert(std::make_pair(match.instructions[i],
                                     &matches_[pattern].back()));
  }
}

// TODO - make this non-recursive
void HloMatcher::MatchPatternStart(HloInstruction* inst) {
  visited_.insert(inst);

  for (unsigned i=0; i<patterns_.size(); i++) {
    if (inst->opcode() == patterns_[i][0].opcode) {
      // Try matching the whole pattern
      HloMatcherMatched match;
      match.ok = true;
      match.instructions.resize(patterns_[i].size());

      if (MatchPattern(inst, patterns_[i], match)) {
        VLOG(3) << "Matched pattern " << i << " against " << inst->ToString();
        AddMatch(i, match);
      }
    }
  }

  for (HloInstruction* operand : inst->operands()) {
    if (visited_.count(operand) == 0) {
      MatchPatternStart(operand);
    }
  }
}



StatusOr<bool> HloMatcher::Run(HloModule *module) {

  if (root_computation_only_) {
    HloComputation* comp = module->entry_computation();
    visited_.clear();
    MatchPatternStart(comp->root_instruction());

  } else {
    // loop over computations to get list of replacements
    for (const auto& comp : module->computations()) {
      visited_.clear();
      MatchPatternStart(comp->root_instruction());
    }
  }

  unsigned int replacement_count = 0;
  for (unsigned int pattern=0; pattern<matches_.size(); pattern++) {
    for (HloMatcherMatched& match :  matches_[pattern]) {
      if (match.ok) {
        if (ReplaceNodes(pattern, match)) {
          for (auto i : match.instructions) {
            auto range = match_map_.equal_range(i);
            for (auto m = range.first; m != range.second; ++m) {
              m->second->ok = false;
            }
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

}
}
