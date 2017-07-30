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

static bool IsRandomBernoulli(HloInstruction* inst) {
  return inst->random_distribution() == RandomDistribution::RNG_BERNOULLI;
}

static bool IsRandomNormal(HloInstruction* inst) {
  return inst->random_distribution() == RandomDistribution::RNG_NORMAL;
}

static bool IsRandomUniform(HloInstruction* inst) {
  return inst->random_distribution() == RandomDistribution::RNG_UNIFORM;
}

static bool IsConstantZero(HloInstruction* inst) {
  return inst->literal().IsAll(0);
}

static bool IsConstantHalf(HloInstruction* inst) {
  return inst->literal().IsAllFloat(0.5);
}

static bool IsPoplarConvolution(HloInstruction* inst) {
  return inst->to_apply()->name().substr(0, 15) == "pop_convolution";
}

static bool IsExternalPadding(HloInstruction* inst) {
  const PaddingConfig& cfg(inst->padding_config());
  for (auto& d : cfg.dimensions()) {
    if (d.interior_padding() > 0) return false;
  }
  return true;
}

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 */

static const std::vector<HloMatcherPattern> patterns = {
  // dynamic update slice with constant coordinate
  {{HloOpcode::kDynamicUpdateSlice, true, nullptr, {-1, -1, 1}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // dynamic slice with constant coordinate
  {{HloOpcode::kDynamicSlice, true, nullptr, {-1, 1}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Relu
  {{HloOpcode::kMaximum, true, nullptr, {-1, 1}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // Sigmoid
  {{HloOpcode::kAdd, true, nullptr, {4, 1}},
   {HloOpcode::kMultiply, true, nullptr, {4, 2}},
   {HloOpcode::kTanh, true, nullptr, {3}},
   {HloOpcode::kMultiply, true, nullptr, {4, -1}},
   {HloOpcode::kConstant, true, IsConstantHalf, {}}},

  // BiasAdd on convolution (explicit broadcast)
  {{HloOpcode::kAdd, true, nullptr, {1, 2}},
   {HloOpcode::kCall, false, IsPoplarConvolution, {-1, -1}},
   {HloOpcode::kBroadcast, true, nullptr, {-1}}},

  // BiasAdd on convolution (implicit broadcast)
  {{HloOpcode::kAdd, true, nullptr, {1, -1}},
   {HloOpcode::kCall, false, IsPoplarConvolution, {-1, -1}}},

  // External padding with constant zero
  {{HloOpcode::kPad, true, IsExternalPadding, {-1, 1}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // Random truncated normal with post scale and add
  {{HloOpcode::kAdd, true, nullptr, {2, 1}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kMultiply, true, nullptr, {4, 3}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kWhile, true, IsTruncatedNormalWhile, {5}},
   {HloOpcode::kRng, true, nullptr, {6, 7}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Random truncated normal without post scale and add
  {{HloOpcode::kWhile, true, IsTruncatedNormalWhile, {1}},
   {HloOpcode::kRng, true, nullptr, {2, 3}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Random normal with post scale and add
  {{HloOpcode::kAdd, true, nullptr, {2, 1}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kMultiply, true, nullptr, {4, 3}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kRng, true, IsRandomNormal, {5, 6}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Random uniform with post scale and add
  {{HloOpcode::kAdd, true, nullptr, {2, 1}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kMultiply, true, nullptr, {4, 3}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kRng, true, IsRandomUniform, {5, 6}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Random 2-constant without post scale and add
  {{HloOpcode::kRng, true, IsRandomNormal, {1, 2}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Random 2-constant without post scale and add
  {{HloOpcode::kRng, true, IsRandomUniform, {1, 2}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

        // Random bernoulli
  {{HloOpcode::kRng, true, IsRandomBernoulli, {1}},
   {HloOpcode::kConstant, true, nullptr, {}}},

};

FuseOps::FuseOps() : HloMatcher(patterns, false) {}

ReplacedInstructions FuseOps::ReplaceNodes(int pattern,
                                           const HloMatcherMatched& match) {
  HloInstruction::FusionKind type =
          static_cast<HloInstruction::FusionKind>(FUSED_BASE + pattern);

  auto* comp = match.computation;
  comp->CreateFusionInstruction(match.instructions, type);

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
