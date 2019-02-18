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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_max_pool.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
     })
  ),
};
// clang-format on

FuseMaxPool::FuseMaxPool(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

bool FuseMaxPool::HandleMatch(HloMatcherMatched& match,
                              const absl::optional<int64> sharding_device) {
  const unsigned max_pool_fwd_pattern_index = 0;
  const unsigned max_pool_bwd_pattern_index = 1;
  auto& pattern = patterns_[match.pattern_idx];

  if (match.pattern_idx == max_pool_fwd_pattern_index) {
    std::string name = op_prefix_ + pattern.GetType();
    const HloInstruction* input =
        match.instruction_mapping[pattern.GetOutputs()[0]]->operand(0);
    HloInstruction* call_to_outlined_computation =
        OutlineExpressionFromComputation(match, name, sharding_device);
    input_to_fwd_max_pool_map_[input] = call_to_outlined_computation;
  } else {
    CHECK_EQ(match.pattern_idx, max_pool_bwd_pattern_index);
    const HloInstruction* input =
        match.instruction_mapping[pattern.GetOutputs()[0]]->operand(0);

    // Find a matching fwd max pool.
    auto it = input_to_fwd_max_pool_map_.find(input);
    if (it == input_to_fwd_max_pool_map_.end()) {
      return false;
    }
    // Found a match, we can outline now, but do need to add the output
    // tensor as a parameter
    std::string name = op_prefix_ + pattern.GetType();
    OutlineExpressionFromComputation(match, name, sharding_device,
                                     {it->second});
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
