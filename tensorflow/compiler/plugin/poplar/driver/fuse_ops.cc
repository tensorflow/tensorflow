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
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static FusedGraphInfo fuse_info[] = {
  {"const_slice_update", 0},
  {"const_slice", 0},
  {"relu", 0},
  {"relu", 0},
  {"sigmoid", 0},
  {"sigmoid", 0},
  {"relugrad", 0},
  {"relugrad", 0},
  {"sigmoidgrad", 0},
  {"sigmoidgrad", 0},
  {"biasadd_broadcast", 0},
  {"biasadd_broadcast", 0},
  {"biasadd", 0},
  {"biasadd", 0},
  {"zero_pad", 0},
  {"trunc_norm_scale_add", 5},
  {"trunc_norm", 1},
  {"norm_scale_add", 4},
  {"uniform_scale_add", 4},
  {"norm", 0},
  {"uniform", 0},
  {"avgpool", 1},
  {"avgpool", 1},
  {"avgpool", 1},
  {"depthwise_conv", 0},
  {"depthwise_conv", 0},
  {"conv_with_reverse", 0},
  {"bias_apply", 0},
  {"wide_const", 1},
};

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * The parameters of the post-fused call are in the reverse order that negative
 * entries appear in the list.  An op marked include_in_replacement=false
 * counts as negative on other instructions on which it appears.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

static const std::vector<HloMatcherPattern> patterns = {
  // dynamic update slice with constant coordinate
  {{HloOpcode::kDynamicUpdateSlice, true, nullptr, {-1, -2, 1}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // dynamic slice with constant coordinate
  {{HloOpcode::kDynamicSlice, true, nullptr, {-1, 1}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Relu
  {{HloOpcode::kMaximum, true, IsFloatType, {-1, 1}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // Relu with broadcast
  {{HloOpcode::kMaximum, true, IsFloatType, {1, -1}},
   {HloOpcode::kBroadcast, true, nullptr, {2}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // Sigmoid
  {{HloOpcode::kAdd, true, IsFloatType, {4, 1}},
   {HloOpcode::kMultiply, true, nullptr, {4, 2}},
   {HloOpcode::kTanh, true, nullptr, {3}},
   {HloOpcode::kMultiply, true, nullptr, {4, -1}},
   {HloOpcode::kConstant, true, IsConstantHalf, {}}},

  // Sigmoid with broadcast
  {{HloOpcode::kAdd, true, IsFloatType, {4, 1}},
   {HloOpcode::kMultiply, true, nullptr, {5, 2}},
   {HloOpcode::kTanh, true, nullptr, {3}},
   {HloOpcode::kMultiply, true, nullptr, {6, -1}},
   {HloOpcode::kBroadcast, true, nullptr, {7}},
   {HloOpcode::kBroadcast, true, nullptr, {7}},
   {HloOpcode::kBroadcast, true, nullptr, {7}},
   {HloOpcode::kConstant, true, IsConstantHalf, {}}},

  // ReluGrad
  {{HloOpcode::kSelect, true, IsFloatType, {1, -1, 2}},
   {HloOpcode::kGt, true, IsTfReluGradOp, {-2, 2}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // ReluGrad with broadcast
  {{HloOpcode::kSelect, true, IsFloatType, {1, -1, 2}},
   {HloOpcode::kGt, true, IsTfReluGradOp, {-2, 2}},
   {HloOpcode::kBroadcast, true, nullptr, {3}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // SigmoidGrad
  {{HloOpcode::kMultiply, true, IsFloatType, {1, 2}},
   {HloOpcode::kMultiply, true, nullptr, {-1, -2}},
   {HloOpcode::kSubtract, true, nullptr, {3, -2}},
   {HloOpcode::kConstant, true, IsConstantOne, {}}},

    // SigmoidGrad with broadcast
  {{HloOpcode::kMultiply, true, IsFloatType, {1, 2}},
   {HloOpcode::kMultiply, true, nullptr, {-1, -2}},
   {HloOpcode::kSubtract, true, nullptr, {3, -2}},
   {HloOpcode::kBroadcast, true, nullptr, {4}},
   {HloOpcode::kConstant, true, IsConstantOne, {}}},

  // BiasAdd on convolution (explicit broadcast)
  {{HloOpcode::kAdd, true, nullptr, {2, 1}},
   {HloOpcode::kBroadcast, true, nullptr, {-1}},
   {HloOpcode::kCall, false, IsPoplarConvolution, {-2, -3}}},

  // BiasAdd on convolution (explicit broadcast)
  {{HloOpcode::kAdd, true, nullptr, {2, 1}},
   {HloOpcode::kBroadcast, true, nullptr, {-1}},
   {HloOpcode::kConvolution, false, nullptr, {-2, -3}}},

  // BiasAdd on convolution (implicit broadcast)
  {{HloOpcode::kAdd, true, nullptr, {1, -1}},
   {HloOpcode::kCall, false, IsPoplarConvolution, {-2, -3}}},

  // BiasAdd on convolution (implicit broadcast)
  {{HloOpcode::kAdd, true, nullptr, {1, -1}},
   {HloOpcode::kConvolution, false, nullptr, {-2, -3}}},

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

  // Average pool (valid)
  {{HloOpcode::kDivide, true, IsAveragePool, {1, 3}},
   {HloOpcode::kReduceWindow, true, Is2DReductionWindow, {-1, 2}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Average pool (same)
  {{HloOpcode::kDivide, true, IsAveragePool, {1, 3}},
   {HloOpcode::kReduceWindow, true, Is2DReductionWindow, {-1, 2}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kBroadcast, true, nullptr, {4}},
   {HloOpcode::kReduceWindow, true, nullptr, {5, 7}},
   {HloOpcode::kBroadcast, true, nullptr, {6}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Average pool (same) - broadcast converted to reshape
  {{HloOpcode::kDivide, true, IsAveragePool, {1, 3}},
   {HloOpcode::kReduceWindow, true, Is2DReductionWindow, {-1, 2}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kReshape, true, nullptr, {4}},
   {HloOpcode::kReduceWindow, true, nullptr, {5, 7}},
   {HloOpcode::kBroadcast, true, nullptr, {6}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Depthwise convolution (forward pass)
  {{HloOpcode::kConvolution, true, nullptr, {-1, 1}},
   {HloOpcode::kSelect, true, nullptr, {6, 4, 2}},
   {HloOpcode::kBroadcast, true, nullptr, {3}},
   {HloOpcode::kConstant, true, IsConstantZero, {}},
   {HloOpcode::kBroadcast, true, nullptr, {5}},
   {HloOpcode::kReshape, true, nullptr, {-2}},
   {HloOpcode::kEq, true, nullptr, {9, 7}},
   {HloOpcode::kBroadcast, true, nullptr, {8}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kBroadcast, true, nullptr, {10}},
   {HloOpcode::kDivide, true, nullptr, {13, 11}},
   {HloOpcode::kBroadcast, true, nullptr, {12}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Depthwise convolution (forward pass, multiplier=1)
  {{HloOpcode::kConvolution, true, nullptr, {-1, 1}},
   {HloOpcode::kSelect, true, nullptr, {6, 4, 2}},
   {HloOpcode::kBroadcast, true, nullptr, {3}},
   {HloOpcode::kConstant, true, IsConstantZero, {}},
   {HloOpcode::kBroadcast, true, nullptr, {5}},
   {HloOpcode::kReshape, true, nullptr, {-2}},
   {HloOpcode::kEq, true, nullptr, {7, 8}},
   {HloOpcode::kBroadcast, true, nullptr, {9}},
   {HloOpcode::kBroadcast, true, nullptr, {9}},
   {HloOpcode::kConstant, true, nullptr, {}}},

  // Backprop input convolution
  {{HloOpcode::kConvolution, true, nullptr, {-1, 1}},
   {HloOpcode::kReverse, true, IsConvFilterSpatialReverse, {-2}}},

  // Bias reduction and application
  {{HloOpcode::kSubtract, true, IsOutputFeed, {1, 2}},
   {HloOpcode::kParameter, false, nullptr, {}},
   {HloOpcode::kMultiply, true, nullptr, {3, 5}},
   {HloOpcode::kBroadcast, true, nullptr, {4}},
   {HloOpcode::kConstant, true, nullptr, {}},
   {HloOpcode::kReduce, true, IsBiasReduce, {-1, 6}},
   {HloOpcode::kConstant, true, IsConstantZero, {}}},

  // Broadcast scalar constant (must be low priority)
  {{HloOpcode::kBroadcast, true, nullptr, {1}},
   {HloOpcode::kConstant, true, IsScalarConstant, {}}},
};

FuseOps::FuseOps() : HloMatcher(patterns, false) {}

ReplacedInstructions FuseOps::ReplaceNodes(int pattern,
                                           const HloMatcherMatched& match) {

  std::string name("_pop_op_");
  name += fuse_info[pattern].name;

  char index = fuse_info[pattern].op_index;

  return OutlineExpressionFromComputation(match, name, index);
}

}
}
