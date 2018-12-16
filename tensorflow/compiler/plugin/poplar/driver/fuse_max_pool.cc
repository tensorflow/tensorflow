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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_max_pool.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 */

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
    // Max Pool fwd
  HloMatcherPattern(
    PatternType("max_pool"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kReduceWindow, NodeOperands({2, 1}), Is2DMaxPool},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstantNegativeInfinity},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

    // Max Pool bwd
  HloMatcherPattern(
    PatternType("max_pool_grad"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSelectAndScatter, NodeOperands({2, 3, 1}), Is2DMaxPoolGrad},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstantOne},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
     })
  ),
};
// clang-format on

FuseMaxPool::FuseMaxPool(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false) {}

unsigned FuseMaxPool::ReplaceNodes() {
  const unsigned max_pool_fwd_pattern_index = 0;
  const unsigned max_pool_bwd_pattern_index = 1;
  unsigned int replacement_count = 0;

  std::map<const HloInstruction*, HloInstruction*> input_to_fwd_max_pool_map;
  // First handle all the fwd Max Pools
  for (HloMatcherMatched& match : matches_[max_pool_fwd_pattern_index]) {
    if (match.ok) {
      const auto& pattern = patterns[max_pool_fwd_pattern_index];
      std::string name = op_prefix_ + pattern.type;
      const HloInstruction* input = match.instructions[0]->operand(0);
      const OutlinedInfo outlined_info =
          OutlineExpressionFromComputation(match, name, pattern.meta_target);
      input_to_fwd_max_pool_map[input] =
          outlined_info.call_to_outlined_computation;
      replacement_count += MarkReplacedInstructions(outlined_info);
    }
  }

  // For each bwd Max Pool, try and find the fwd Max Pool (same input)
  for (HloMatcherMatched& match : matches_[max_pool_bwd_pattern_index]) {
    if (match.ok) {
      const HloInstruction* input = match.instructions[0]->operand(0);
      auto it = input_to_fwd_max_pool_map.find(input);
      if (it != input_to_fwd_max_pool_map.end()) {
        // Found a match, we can outline now, but do need to add the output
        // tensor as a parameter
        const auto& pattern = patterns[max_pool_bwd_pattern_index];
        std::string name = op_prefix_ + pattern.type;
        const OutlinedInfo outlined_info = OutlineExpressionFromComputation(
            match, name, pattern.meta_target, {it->second});
        replacement_count += MarkReplacedInstructions(outlined_info);
      }
    }
  }

  return replacement_count;
}

}  // namespace poplarplugin
}  // namespace xla
