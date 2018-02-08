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

#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static const char* names[] = {
  "pop_backprop_conv",
  "pop_depth_conv",
  "pop_convolution",
};

static bool
IsPoplibsFusion(const HloInstruction* inst, const std::string& type) {
  const HloComputation* comp = inst->to_apply();
  if (comp->name().substr(0, 8) == "_pop_op_") {
    auto end = comp->name().find('.');
    std::string name = comp->name().substr(8, end - 8);
    return name == type;
  }
  return false;
}

static bool IsPoplibsBackpropInputConv(const HloInstruction* inst) {
  return IsPoplibsFusion(inst, "conv_with_reverse");
}

static bool IsPoplibsDepthwiseConv(const HloInstruction* inst) {
  return IsPoplibsFusion(inst, "depthwise_conv");
}

static const std::vector<HloMatcherPattern> patterns = {

  // Backprop input convolution
  {{HloOpcode::kCall, true, IsPoplibsBackpropInputConv, {-1, -2}}},

  // Depthwise convolution (forward pass)
  {{HloOpcode::kCall, true, IsPoplibsDepthwiseConv, {-1, -2}}},

  // Stand-alone convolution
  {{HloOpcode::kConvolution, true, nullptr, {-1, -2}}},

};

Outliner::Outliner() : HloMatcher(patterns, true) {}

ReplacedInstructions Outliner::ReplaceNodes(int pattern,
                                            const HloMatcherMatched& match) {
  return OutlineExpressionFromComputation(match, names[pattern]);
}

}
}
