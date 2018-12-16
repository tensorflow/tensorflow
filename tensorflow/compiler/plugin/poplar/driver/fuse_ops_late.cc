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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/inplace_util.h"
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
  // dynamic update slice with constant coordinate
  HloMatcherPattern(
    PatternType("const_slice_update"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDynamicUpdateSlice, NodeOperands({2, 3, 1})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // dynamic update slice with wide constant coordinate
  HloMatcherPattern(
    PatternType("const_slice_update"),
    PatternMetaTarget(0),
    PatternInputs({3, 4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDynamicUpdateSlice, NodeOperands({3, 4, 1})},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // dynamic slice with constant coordinate
  HloMatcherPattern(
    PatternType("const_slice"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDynamicSlice, NodeOperands({2, 1})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // dynamic slice with wide constant coordinate
  HloMatcherPattern(
    PatternType("const_slice"),
    PatternMetaTarget(0),
    PatternInputs({3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDynamicSlice, NodeOperands({3, 1})},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Relu
  HloMatcherPattern(
    PatternType("relu"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMaximum, NodeOperands({2, 1}), IsFloatType},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Relu with broadcast
  HloMatcherPattern(
    PatternType("relu"),
    PatternMetaTarget(0),
    PatternInputs({3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMaximum, NodeOperands({3, 1}), IsFloatType},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Sigmoid
  HloMatcherPattern(
    PatternType("sigmoid"),
    PatternMetaTarget(0),
    PatternInputs({5}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({4, 1}), IsFloatType},
      {HloOpcode::kMultiply, NodeOperands({4, 2})},
      {HloOpcode::kTanh, NodeOperands({3})},
      {HloOpcode::kMultiply, NodeOperands({4, 5})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantHalf},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Sigmoid with broadcast
  HloMatcherPattern(
    PatternType("sigmoid"),
    PatternMetaTarget(0),
    PatternInputs({6}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({1, 4}), IsFloatType},
      {HloOpcode::kMultiply, NodeOperands({2, 4})},
      {HloOpcode::kTanh, NodeOperands({3})},
      {HloOpcode::kMultiply, NodeOperands({6, 4})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantHalf},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // ReluGrad
  HloMatcherPattern(
    PatternType("relugrad"),
    PatternMetaTarget(0),
    PatternInputs({4, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSelect, NodeOperands({1, 3, 2}), IsFloatType},
      {HloOpcode::kGt, NodeOperands({4, 2}), IsTfReluGradOp},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // ReluGrad with broadcast
  HloMatcherPattern(
    PatternType("relugrad"),
    PatternMetaTarget(0),
    PatternInputs({5, 4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSelect, NodeOperands({1, 4, 2}), IsFloatType},
      {HloOpcode::kGt, NodeOperands({5, 2}), IsTfReluGradOp},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // SigmoidGrad
  HloMatcherPattern(
    PatternType("sigmoidgrad"),
    PatternMetaTarget(0),
    PatternInputs({5, 4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMultiply, NodeOperands({1, 2}), IsFloatType},
      {HloOpcode::kMultiply, NodeOperands({4, 5})},
      {HloOpcode::kSubtract, NodeOperands({3, 5})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // SigmoidGrad with broadcast
  HloMatcherPattern(
    PatternType("sigmoidgrad"),
    PatternMetaTarget(0),
    PatternInputs({6, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kMultiply, NodeOperands({1, 2}), IsFloatType},
      {HloOpcode::kMultiply, NodeOperands({5, 6})},
      {HloOpcode::kSubtract, NodeOperands({3, 6})},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // BiasAdd on convolution (w/ broadcast)
  HloMatcherPattern(
    PatternType("conv_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kCall, NodeOperands({}), IsPopOpsConvolution},
      {HloOpcode::kParameter, NodeOperands({}), Is1DVector}
    })
  ),

  // BiasAdd on convolution (w/ broadcast)
  HloMatcherPattern(
    PatternType("conv_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConvolution, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({}), Is1DVector}
    })
  ),

  // BiasAdd on convolution (w/ reshape)
  HloMatcherPattern(
    PatternType("conv_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kReshape, NodeOperands({3}), IsExpandingReshape},
      {HloOpcode::kConvolution, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({}), Is1DVector}
    })
  ),

  // BiasAdd on a MatMul (w/ broadcast)
  HloMatcherPattern(
    PatternType("matmul_biasadd"),
    PatternMetaTarget(0),
    PatternInputs({2, 3}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1}), IsBiasAdd},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kDot, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({}), Is1DVector}
    })
  ),

  // External padding with constant zero
  HloMatcherPattern(
    PatternType("zero_pad"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kPad, NodeOperands({2, 1}), IsExternalPadding},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Random normal with post scale and add
  HloMatcherPattern(
    PatternType("norm_scale_add"),
    PatternMetaTarget(4),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kMultiply, NodeOperands({4, 3})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kRng, NodeOperands({5, 6}), IsRandomNormal},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})}
    })
  ),

  // Random normal with broadcasted post scale and add
  HloMatcherPattern(
    PatternType("norm_scale_add"),
    PatternMetaTarget(6),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({3, 1})},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kMultiply, NodeOperands({6, 4})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kRng, NodeOperands({7, 8}), IsRandomNormal},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})}
    })
  ),

  // Random uniform with post scale and add
  HloMatcherPattern(
    PatternType("uniform_scale_add"),
    PatternMetaTarget(4),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({2, 1})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kMultiply, NodeOperands({4, 3})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kRng, NodeOperands({5, 6}), IsRandomUniform},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})}
    })
  ),

  // Random uniform with broadcasted post scale and add
  HloMatcherPattern(
    PatternType("uniform_scale_add"),
    PatternMetaTarget(6),
    PatternInputs({}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({3, 1})},
      {HloOpcode::kBroadcast, NodeOperands({2})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kMultiply, NodeOperands({6, 4})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kRng, NodeOperands({7, 8}), IsRandomUniform},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kConstant, NodeOperands({})}
    })
  ),

  // Average pool (valid)
  HloMatcherPattern(
    PatternType("avg_pool"),
    PatternMetaTarget(1),
    PatternInputs({4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDivide, NodeOperands({1, 3}), IsAveragePool},
      {HloOpcode::kReduceWindow, NodeOperands({4, 2}), Is2DReductionWindow},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Average pool (same)
  HloMatcherPattern(
    PatternType("avg_pool"),
    PatternMetaTarget(1),
    PatternInputs({7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDivide, NodeOperands({1, 2}), IsAveragePool},
      {HloOpcode::kReduceWindow, NodeOperands({7, 6}), Is2DReductionWindow},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kReduceWindow, NodeOperands({4, 6})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Average pool (same) - broadcast converted to reshape
  HloMatcherPattern(
    PatternType("avg_pool"),
    PatternMetaTarget(1),
    PatternInputs({7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kDivide, NodeOperands({1, 2}), IsAveragePool},
      {HloOpcode::kReduceWindow, NodeOperands({7, 6}), Is2DReductionWindow},
      {HloOpcode::kReshape, NodeOperands({3})},
      {HloOpcode::kReduceWindow, NodeOperands({4, 6})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantOne},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Bias reduction and application
  HloMatcherPattern(
    PatternType("bias_apply"),
    PatternMetaTarget(0),
    PatternInputs({1, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSubtract, NodeOperands({1, 2}), IsOutputFeed},
      {HloOpcode::kParameter, NodeOperands({}), IsTrueParameter},
      {HloOpcode::kMultiply, NodeOperands({5, 3})},
      {HloOpcode::kBroadcast, NodeOperands({4})},
      {HloOpcode::kConstant, NodeOperands({})},
      {HloOpcode::kReduce, NodeOperands({7, 6}), IsBiasReduce},
      {HloOpcode::kConstant, NodeOperands({}), IsConstantZero},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Convolution followed by scaled add to - A = A + B * c
  HloMatcherPattern(
    PatternType("conv_scaled_inplace"),
    PatternMetaTarget(4),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({5, 1})},
      {HloOpcode::kMultiply, NodeOperands({4, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
      {HloOpcode::kConvolution, NodeOperands({6, 7})},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Convolution followed by scaled subtract from - A = A - B * c
  HloMatcherPattern(
    PatternType("conv_scaled_inplace"),
    PatternMetaTarget(4),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSubtract, NodeOperands({5, 1})},
      {HloOpcode::kMultiply, NodeOperands({4, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
      {HloOpcode::kConvolution, NodeOperands({6, 7})},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Scaled add to - A = A + B * c
  HloMatcherPattern(
    PatternType("scaled_inplace"),
    PatternMetaTarget(0),
    PatternInputs({4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kAdd, NodeOperands({4, 1})},
      {HloOpcode::kMultiply, NodeOperands({5, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Scaled subtract from - A = A - B * c
  HloMatcherPattern(
    PatternType("scaled_inplace"),
    PatternMetaTarget(0),
    PatternInputs({4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kSubtract, NodeOperands({4, 1})},
      {HloOpcode::kMultiply, NodeOperands({5, 2})},
      {HloOpcode::kBroadcast, NodeOperands({3})},
      {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),

  // Reduce window with a window size of 1x1, stride 1 and identity reduction
  // function (param 1 is returned)
  HloMatcherPattern(
    PatternType("padding_reduce_window"),
    PatternMetaTarget(0),
    PatternInputs({1, 2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kReduceWindow, NodeOperands({1, 2}), IsPaddingReduceWindow},
      {HloOpcode::kParameter, NodeOperands({})},
      {HloOpcode::kParameter, NodeOperands({})}
    })
  ),
};
// clang-format on

FuseOpsLate::FuseOpsLate(struct CompilerAnnotations& annotations)
    : SingleHloMatcher(annotations, patterns, "_pop_op_") {}

}  // namespace poplarplugin
}  // namespace xla
