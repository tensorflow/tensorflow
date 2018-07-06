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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static FusedGraphInfo fuse_info[] = {
    {"depthwise_filter", 14},
    {"conv_with_reverse", 0},
    {"conv_with_reverse", 0},
    {"depthwise_conv", 0},
};

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

static const std::vector<HloMatcherPattern> patterns = {
    // ------ Convolutions ------
    // DepthwiseConv2DBackpropFilter
    {{HloOpcode::kReshape, true, 0, nullptr, {1}},
     {HloOpcode::kReduce, true, 0, nullptr, {3, 2}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {6, 14, 4}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {5}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kEq, true, 0, nullptr, {9, 7}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {8}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {10}},
     {HloOpcode::kDivide, true, 0, nullptr, {13, 11}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {12}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConvolution, true, 0, IsOpWithWindowNoBaseDilation, {15, 16}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},
    // DepthwiseConv2DBackpropInput
    {{HloOpcode::kConvolution, true, 0, IsOpWithWindowNoStride, {16, 1}},
     {HloOpcode::kReverse, true, 0, IsConvFilterTranspose, {2}},
     {HloOpcode::kSelect, true, 0, nullptr, {8, 3, 6}},
     {HloOpcode::kAdd, true, 0, nullptr, {4, 6}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {5}},
     {HloOpcode::kReshape, true, 0, nullptr, {17}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {7}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kEq, true, 0, nullptr, {11, 9}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {10}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {12}},
     {HloOpcode::kDivide, true, 0, nullptr, {15, 13}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {14}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},
    // Conv{2,3}DBackpropInput
    {{HloOpcode::kConvolution, true, 0, IsOpWithWindowNoStride, {2, 1}},
     {HloOpcode::kReverse, true, 0, IsConvFilterTranspose, {3}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},
    // DepthwiseConv2D
    {{HloOpcode::kConvolution, true, 0, nullptr, {15, 1}},
     {HloOpcode::kSelect, true, 0, nullptr, {7, 2, 5}},
     {HloOpcode::kAdd, true, 0, nullptr, {3, 5}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {4}},
     {HloOpcode::kReshape, true, 0, nullptr, {16}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {6}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kEq, true, 0, nullptr, {10, 8}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {9}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {11}},
     {HloOpcode::kDivide, true, 0, nullptr, {14, 12}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {13}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},
};

FuseOpsEarly::FuseOpsEarly(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false) {}

ReplacedInstructions FuseOpsEarly::ReplaceNodes(
    int pattern, const HloMatcherMatched& match) {
  std::string name("_pop_op_");
  name += fuse_info[pattern].name;

  char index = fuse_info[pattern].op_index;

  return OutlineExpressionFromComputation(match, name, index);
}

}  // namespace poplarplugin
}  // namespace xla
