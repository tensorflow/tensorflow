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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static const FusedGraphInfo fuse_info[] = {
    {"pop_backprop_conv", 0}, {"pop_depth_conv", 0}, {"pop_convolution", 0},
};

static const std::vector<HloMatcherPattern> patterns = {

    // Backprop input convolution
    {{HloOpcode::kCall, true, 0, IsFusedReverseInputConv, {1, 2}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Depthwise convolution (forward pass)
    {{HloOpcode::kCall, true, 0, IsFusedDepthwiseConv, {1, 2}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Stand-alone convolution
    {{HloOpcode::kConvolution, true, 0, nullptr, {1, 2}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

};

Outliner::Outliner(struct CompilerAnnotations& annotations) :
    HloMatcher(patterns, annotations, true) {}

ReplacedInstructions Outliner::ReplaceNodes(int pattern,
                                            const HloMatcherMatched& match) {
  auto& fuse = fuse_info[pattern];
  return OutlineExpressionFromComputation(match, fuse.name, fuse.op_index);
}

}  // namespace poplarplugin
}  // namespace xla
