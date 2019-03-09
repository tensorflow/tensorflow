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

#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  // ------ Convolutions ------
  // DepthwiseConv2DBackpropFilter
  HloMatcherPattern(
    PatternType("depthwise_filter"),
    PatternMetaTarget(12),
    PatternInputs({13, 14}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kReshape, NodeOperands({1})},
      {HloOpcode::kReduce, NodeOperands({3, 2})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kSelect, NodeOperands({6, 12, 4})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kEq, NodeOperands({8, 7})},
      {HloOpcode::kIota, NodeOperands({})},
      {HloOpcode::kDivide, NodeOperands({11, 9})},
      {HloOpcode::kBroadcast, NodeOperands({10})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kIota, NodeOperands({})},
      {HloOpcode::kConvolution, NodeOperands({13, 14}), IsOpWithWindowNoBaseDilation},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })
  ),

  // DepthwiseConv2DBackpropInput
  HloMatcherPattern(
    PatternType("conv_with_reverse"),
    PatternMetaTarget(0),
    PatternInputs({16, 17}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvolution, NodeOperands({16, 1}), IsOpWithWindowNoStride},
      {HloOpcode::kReverse, NodeOperands({2}), IsConvFilterTranspose},
      {HloOpcode::kSelect, NodeOperands({8, 3, 6})},
      {HloOpcode::kAdd, NodeOperands({4, 6})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kReshape, NodeOperands({17})},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kEq, NodeOperands({11, 9})},
      {HloOpcode::kBroadcast, NodeOperands({10})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kBroadcast, NodeOperands({12})},
      {HloOpcode::kDivide, NodeOperands({15, 13})},
      {HloOpcode::kBroadcast, NodeOperands({14})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })
  ),

  // Conv{2,3}DBackpropInput
  HloMatcherPattern(
    PatternType("conv_with_reverse"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvolution, NodeOperands({2, 1}), IsOpWithWindowNoStride},
      {HloOpcode::kReverse, NodeOperands({3}), IsConvFilterTranspose},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })
  ),

  // DepthwiseConv2D
  HloMatcherPattern(
    PatternType("depthwise_conv"),
    PatternMetaTarget(0),
    PatternInputs({15, 16}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvolution, NodeOperands({15, 1})},
      {HloOpcode::kSelect, NodeOperands({7, 2, 5})},
      {HloOpcode::kAdd, NodeOperands({3, 5})},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kReshape, NodeOperands({16})},
      {HloOpcode::kBroadcast, NodeOperands({6})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kEq, NodeOperands({10, 8})},
      {HloOpcode::kBroadcast, NodeOperands({9})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kBroadcast, NodeOperands({11})},
      {HloOpcode::kDivide, NodeOperands({14, 12})},
      {HloOpcode::kBroadcast, NodeOperands({13})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })
  ),
};
// clang-format on

FuseOpsEarly::FuseOpsEarly(struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_") {}

}  // namespace poplarplugin
}  // namespace xla
