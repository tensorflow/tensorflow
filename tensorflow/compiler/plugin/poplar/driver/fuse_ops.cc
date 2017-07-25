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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static bool IsTruncatedNormalWhile(HloInstruction* inst) {
  return inst->while_condition()->name().substr(0, 16) == "truncated_normal";
}

static bool IsConstantZero(HloInstruction* inst) {
  return inst->literal().IsAllFloat(0.0);
}

static bool IsConstantHalf(HloInstruction* inst) {
  return inst->literal().IsAllFloat(0.5);
}

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 */

static const std::vector<HloMatcherPattern> patterns = {
  // dynamic update slice with constant coordinate
  {{HloOpcode::kDynamicUpdateSlice, nullptr, {-1, -1, 1}},
   {HloOpcode::kConstant, nullptr, {}}},

  // dynamic slice with constant coordinate
  {{HloOpcode::kDynamicSlice, nullptr, {-1, 1}},
   {HloOpcode::kConstant, nullptr, {}}},

  // Truncated normal
  {{HloOpcode::kWhile, IsTruncatedNormalWhile, {1}},
   {HloOpcode::kRng, nullptr, {}}},

  // Relu
  {{HloOpcode::kMaximum, nullptr, {-1, 1}},
   {HloOpcode::kConstant, IsConstantZero, {}}},

  // Sigmoid
  {{HloOpcode::kAdd, nullptr, {4, 1}},
   {HloOpcode::kMultiply, nullptr, {4, 2}},
   {HloOpcode::kTanh, nullptr, {3}},
   {HloOpcode::kMultiply, nullptr, {4, -1}},
   {HloOpcode::kConstant, IsConstantHalf, {}}}
};

FuseOps::FuseOps() : HloMatcher(patterns, false) {}

ReplacedInstructions FuseOps::ReplaceNodes(unsigned int pattern,
                                           const HloMatcherMatched& match) {
  auto* comp = match.computation;
  HloInstruction* fused = comp->CreateFusionInstruction(
          match.instructions,
          HloInstruction::FusionKind::kCustom);

  fused->set_fusion_custom_tag(pattern);

  ReplacedInstructions replaced;

  std::set<HloInstruction*> remaining;
  for (auto& i : comp->instructions()) {
    remaining.insert(i.get());
  }

  for (auto inst : match.instructions) {
    if (remaining.count(inst) == 0) {
      replaced.push_back(inst);
    }
  }

  return replaced;
}

}
}
