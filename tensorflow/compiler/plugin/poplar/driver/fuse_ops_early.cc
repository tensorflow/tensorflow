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

  // ------ Random Ops ------
  // Truncated normal - mean 0 std 1, non scalar shape
  HloMatcherPattern(
    PatternType("trunc_norm"),
    PatternMetaTarget(1),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMultiply, NodeOperands({99, 1}), IsTruncatedNormal},
      {HloOpcode::kMultiply, NodeOperands({2, 82})},
      {HloOpcode::kAdd, NodeOperands({68, 3})},
      {HloOpcode::kMultiply, NodeOperands({4, 18})},
      {HloOpcode::kAdd, NodeOperands({63, 5})},
      {HloOpcode::kMultiply, NodeOperands({6, 18})},
      {HloOpcode::kAdd, NodeOperands({58, 7})},
      {HloOpcode::kMultiply, NodeOperands({8, 18})},
      {HloOpcode::kAdd, NodeOperands({53, 9})},
      {HloOpcode::kMultiply, NodeOperands({10, 18})},
      {HloOpcode::kAdd, NodeOperands({48, 11})},
      {HloOpcode::kMultiply, NodeOperands({12, 18})},
      {HloOpcode::kAdd, NodeOperands({43, 13})},
      {HloOpcode::kMultiply, NodeOperands({14, 18})},
      {HloOpcode::kAdd, NodeOperands({38, 15})},
      {HloOpcode::kMultiply, NodeOperands({16, 18})},
      {HloOpcode::kAdd, NodeOperands({33, 17})},
      {HloOpcode::kMultiply, NodeOperands({28, 18})},
      {HloOpcode::kSelect, NodeOperands({73, 25, 19})},
      {HloOpcode::kSubtract, NodeOperands({22, 20})},
      {HloOpcode::kBroadcast, NodeOperands({21})},
      {HloOpcode::kConstant, NodeOperands({})},  // 3
      {HloOpcode::kPower, NodeOperands({76, 23})},
      {HloOpcode::kBroadcast, NodeOperands({24})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.5
      {HloOpcode::kSubtract, NodeOperands({76, 26})},
      {HloOpcode::kBroadcast, NodeOperands({27})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2.5
      {HloOpcode::kSelect, NodeOperands({73, 31, 29})},
      {HloOpcode::kBroadcast, NodeOperands({30})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.000200214257
      {HloOpcode::kBroadcast, NodeOperands({32})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2.81022636e-08
      {HloOpcode::kSelect, NodeOperands({73, 36, 34})},
      {HloOpcode::kBroadcast, NodeOperands({35})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.000100950558
      {HloOpcode::kBroadcast, NodeOperands({37})},
      {HloOpcode::kConstant, NodeOperands({})},  // 3.43273939e-07
      {HloOpcode::kSelect, NodeOperands({73, 41, 39})},
      {HloOpcode::kBroadcast, NodeOperands({40})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00134934322
      {HloOpcode::kBroadcast, NodeOperands({42})},
      {HloOpcode::kConstant, NodeOperands({})},  // -3.5233877e-06
      {HloOpcode::kSelect, NodeOperands({73, 46, 44})},
      {HloOpcode::kBroadcast, NodeOperands({45})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.00367342844
      {HloOpcode::kBroadcast, NodeOperands({47})},
      {HloOpcode::kConstant, NodeOperands({})},  // -4.39150654e-06
      {HloOpcode::kSelect, NodeOperands({73, 51, 49})},
      {HloOpcode::kBroadcast, NodeOperands({50})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00573950773
      {HloOpcode::kBroadcast, NodeOperands({52})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00021858087
      {HloOpcode::kSelect, NodeOperands({73, 56, 54})},
      {HloOpcode::kBroadcast, NodeOperands({55})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.0076224613
      {HloOpcode::kBroadcast, NodeOperands({57})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.00125372503
      {HloOpcode::kSelect, NodeOperands({73, 61, 59})},
      {HloOpcode::kBroadcast, NodeOperands({60})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00943887047
      {HloOpcode::kBroadcast, NodeOperands({62})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.00417768164
      {HloOpcode::kSelect, NodeOperands({73, 66, 64})},
      {HloOpcode::kBroadcast, NodeOperands({65})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1.00167406
      {HloOpcode::kBroadcast, NodeOperands({67})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.246640727
      {HloOpcode::kSelect, NodeOperands({73, 71, 69})},
      {HloOpcode::kBroadcast, NodeOperands({70})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2.83297682
      {HloOpcode::kBroadcast, NodeOperands({72})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1.50140941
      {HloOpcode::kLt, NodeOperands({76, 74})},
      {HloOpcode::kBroadcast, NodeOperands({75})},
      {HloOpcode::kConstant, NodeOperands({})},  // 5
      {HloOpcode::kNegate, NodeOperands({77})},
      {HloOpcode::kLog, NodeOperands({78})},
      {HloOpcode::kMultiply, NodeOperands({81, 79})},
      {HloOpcode::kAdd, NodeOperands({80, 82})},
      {HloOpcode::kBroadcast, NodeOperands({98})},
      {HloOpcode::kSubtract, NodeOperands({97, 82})},
      {HloOpcode::kSubtract, NodeOperands({85, 83})},
      {HloOpcode::kBroadcast, NodeOperands({84})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1
      {HloOpcode::kMultiply, NodeOperands({95, 86})},
      {HloOpcode::kAdd, NodeOperands({93, 87})},
      {HloOpcode::kMultiply, NodeOperands({91, 88})},
      {HloOpcode::kRng, NodeOperands({90, 89})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1
      {HloOpcode::kConstant, NodeOperands({})},  // 1.17549435e-38
      {HloOpcode::kBroadcast, NodeOperands({92})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.954499722
      {HloOpcode::kBroadcast, NodeOperands({94})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.0227501318
      {HloOpcode::kBroadcast, NodeOperands({96})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2
      {HloOpcode::kBroadcast, NodeOperands({98})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},  // 1
      {HloOpcode::kBroadcast, NodeOperands({100})},
      {HloOpcode::kConstant, NodeOperands({})}  // 1.41421354
    })
  ),

  // Truncated normal - mean 0 std 1, scalar shape
  HloMatcherPattern(
    PatternType("trunc_norm"),
    PatternMetaTarget(1),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMultiply, NodeOperands({89, 1})},
      {HloOpcode::kMultiply, NodeOperands({2, 77})},
      {HloOpcode::kAdd, NodeOperands({65, 3})},
      {HloOpcode::kMultiply, NodeOperands({4, 18})},
      {HloOpcode::kAdd, NodeOperands({60, 5})},
      {HloOpcode::kMultiply, NodeOperands({6, 18})},
      {HloOpcode::kAdd, NodeOperands({55, 7})},
      {HloOpcode::kMultiply, NodeOperands({8, 18})},
      {HloOpcode::kAdd, NodeOperands({50, 9})},
      {HloOpcode::kMultiply, NodeOperands({10, 18})},
      {HloOpcode::kAdd, NodeOperands({45, 11})},
      {HloOpcode::kMultiply, NodeOperands({12, 18})},
      {HloOpcode::kAdd, NodeOperands({40, 13})},
      {HloOpcode::kMultiply, NodeOperands({14, 18})},
      {HloOpcode::kAdd, NodeOperands({35, 15})},
      {HloOpcode::kMultiply, NodeOperands({16, 18})},
      {HloOpcode::kAdd, NodeOperands({30, 17})},
      {HloOpcode::kMultiply, NodeOperands({25, 18})},
      {HloOpcode::kSelect, NodeOperands({70, 23, 19})},
      {HloOpcode::kSubtract, NodeOperands({21, 20})},
      {HloOpcode::kConstant, NodeOperands({})},  // 3
      {HloOpcode::kPower, NodeOperands({72, 22})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.5
      {HloOpcode::kSubtract, NodeOperands({72, 24})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2.5
      {HloOpcode::kSelect, NodeOperands({70, 28, 26})},
      {HloOpcode::kBroadcast, NodeOperands({27})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.000200214257
      {HloOpcode::kBroadcast, NodeOperands({29})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2.81022636e-08
      {HloOpcode::kSelect, NodeOperands({70, 33, 31})},
      {HloOpcode::kBroadcast, NodeOperands({32})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.000100950558
      {HloOpcode::kBroadcast, NodeOperands({34})},
      {HloOpcode::kConstant, NodeOperands({})},  // 3.43273939e-07
      {HloOpcode::kSelect, NodeOperands({70, 38, 36})},
      {HloOpcode::kBroadcast, NodeOperands({37})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00134934322
      {HloOpcode::kBroadcast, NodeOperands({39})},
      {HloOpcode::kConstant, NodeOperands({})},  // -3.5233877e-06
      {HloOpcode::kSelect, NodeOperands({70, 43, 41})},
      {HloOpcode::kBroadcast, NodeOperands({42})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.00367342844
      {HloOpcode::kBroadcast, NodeOperands({44})},
      {HloOpcode::kConstant, NodeOperands({})},  // -4.39150654e-06
      {HloOpcode::kSelect, NodeOperands({70, 48, 46})},
      {HloOpcode::kBroadcast, NodeOperands({47})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00573950773
      {HloOpcode::kBroadcast, NodeOperands({49})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00021858087
      {HloOpcode::kSelect, NodeOperands({70, 53, 51})},
      {HloOpcode::kBroadcast, NodeOperands({52})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.0076224613
      {HloOpcode::kBroadcast, NodeOperands({54})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.00125372503
      {HloOpcode::kSelect, NodeOperands({70, 58, 56})},
      {HloOpcode::kBroadcast, NodeOperands({57})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.00943887047
      {HloOpcode::kBroadcast, NodeOperands({59})},
      {HloOpcode::kConstant, NodeOperands({})},  // -0.00417768164
      {HloOpcode::kSelect, NodeOperands({70, 63, 61})},
      {HloOpcode::kBroadcast, NodeOperands({62})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1.00167406
      {HloOpcode::kBroadcast, NodeOperands({64})},
      {HloOpcode::kConstant, NodeOperands({})},  // 0.246640727
      {HloOpcode::kSelect, NodeOperands({70, 68, 66})},
      {HloOpcode::kBroadcast, NodeOperands({67})},
      {HloOpcode::kConstant, NodeOperands({})},  // 2.83297682
      {HloOpcode::kBroadcast, NodeOperands({69})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1.50140941
      {HloOpcode::kLt, NodeOperands({72, 71})},
      {HloOpcode::kConstant, NodeOperands({})},  // 5
      {HloOpcode::kNegate, NodeOperands({73})},
      {HloOpcode::kLog, NodeOperands({74})},
      {HloOpcode::kMultiply, NodeOperands({76, 75})},
      {HloOpcode::kAdd, NodeOperands({88, 77})},
      {HloOpcode::kSubtract, NodeOperands({88, 77})},
      {HloOpcode::kSubtract, NodeOperands({79, 78})},
      {HloOpcode::kConstant, NodeOperands({})},  // 1
      {HloOpcode::kMultiply, NodeOperands({87, 80})},
      {HloOpcode::kAdd, NodeOperands({86, 81})},
      {HloOpcode::kMultiply, NodeOperands({85, 82})},
      {HloOpcode::kRng, NodeOperands({84, 83})},
      {HloOpcode::kConstant, NodeOperands({})},    // 1
      {HloOpcode::kConstant, NodeOperands({})},    // 1.17549435e-38
      {HloOpcode::kConstant, NodeOperands({})},    // 0.954499722
      {HloOpcode::kConstant, NodeOperands({})},    // 0.0227501318
      {HloOpcode::kConstant, NodeOperands({})},    // 2
      {HloOpcode::kConstant, NodeOperands({})},    // 1
      {HloOpcode::kConstant, NodeOperands({})}  // 1.41421354
    })
  ),

  // Average pool (valid) - broadcast
  HloMatcherPattern(
    PatternType("avg_pool"),
    PatternMetaTarget(1),
    PatternInputs({5}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDivide, NodeOperands({1, 3}), IsAveragePool},
      {HloOpcode::kReduceWindow, NodeOperands({5, 2}), Is2DReductionWindow},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})}
    })
  ),
};
// clang-format on

FuseOpsEarly::FuseOpsEarly(struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_") {}

}  // namespace poplarplugin
}  // namespace xla
