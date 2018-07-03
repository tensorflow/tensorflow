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
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

static FusedGraphInfo fuse_info[] = {
    {"const_slice_update", 0},
    {"const_slice_update", 0},
    {"const_slice", 0},
    {"const_slice", 0},
    {"relu", 0},
    {"relu", 0},
    {"sigmoid", 0},
    {"sigmoid", 0},
    {"relugrad", 0},
    {"relugrad", 0},
    {"sigmoidgrad", 0},
    {"sigmoidgrad", 0},
    {"biasadd", 0},
    {"biasadd", 0},
    {"zero_pad", 0},
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
    {"conv_with_reverse", 0},
    {"depthwise_filter", 13},
    {"bias_apply", 0},
    {"reduction_no_convert", 1},
    {"reduction_no_convert", 1},
    {"wide_const", 1},
};

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 * NOTE: Highest match priority is nearer the top of the list
 */

static const std::vector<HloMatcherPattern> patterns = {
    // dynamic update slice with constant coordinate
    {{HloOpcode::kDynamicUpdateSlice, true, 0, nullptr, {2, 3, 1}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // dynamic update slice with wide constant coordinate
    {{HloOpcode::kDynamicUpdateSlice, true, 0, nullptr, {3, 4, 1}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {2}},
     {HloOpcode::kConstant, true, 0, IsScalarConstant, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // dynamic slice with constant coordinate
    {{HloOpcode::kDynamicSlice, true, 0, nullptr, {2, 1}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // dynamic slice with wide constant coordinate
    {{HloOpcode::kDynamicSlice, true, 0, nullptr, {3, 1}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {2}},
     {HloOpcode::kConstant, true, 0, IsScalarConstant, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Relu
    {{HloOpcode::kMaximum, true, 0, IsFloatType, {2, 1}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Relu with broadcast
    {{HloOpcode::kMaximum, true, 0, IsFloatType, {1, 3}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {2}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Sigmoid
    {{HloOpcode::kAdd, true, 0, IsFloatType, {4, 1}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 2}},
     {HloOpcode::kTanh, true, 0, nullptr, {3}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 5}},
     {HloOpcode::kConstant, true, 0, IsConstantHalf, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Sigmoid with broadcast
    {{HloOpcode::kAdd, true, 0, IsFloatType, {4, 1}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 2}},
     {HloOpcode::kTanh, true, 0, nullptr, {3}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 6}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {5}},
     {HloOpcode::kConstant, true, 0, IsConstantHalf, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // ReluGrad
    {{HloOpcode::kSelect, true, 0, IsFloatType, {1, 3, 2}},
     {HloOpcode::kGt, true, 0, IsTfReluGradOp, {4, 2}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // ReluGrad with broadcast
    {{HloOpcode::kSelect, true, 0, IsFloatType, {1, 4, 2}},
     {HloOpcode::kGt, true, 0, IsTfReluGradOp, {5, 2}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {3}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // SigmoidGrad
    {{HloOpcode::kMultiply, true, 0, IsFloatType, {1, 2}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 5}},
     {HloOpcode::kSubtract, true, 0, nullptr, {3, 5}},
     {HloOpcode::kConstant, true, 0, IsConstantOne, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // SigmoidGrad with broadcast
    {{HloOpcode::kMultiply, true, 0, IsFloatType, {1, 2}},
     {HloOpcode::kMultiply, true, 0, nullptr, {5, 6}},
     {HloOpcode::kSubtract, true, 0, nullptr, {3, 6}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {4}},
     {HloOpcode::kConstant, true, 0, IsConstantOne, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // BiasAdd on convolution (w/ broadcast)
    {{HloOpcode::kAdd, true, 0, nullptr, {2, 1}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {3}},
     {HloOpcode::kCall, false, 0, IsPoplarConvolution, {}},
     {HloOpcode::kParameter, false, 1, Is1DVector, {}}},

    // BiasAdd on convolution (w/ broadcast)
    {{HloOpcode::kAdd, true, 0, nullptr, {2, 1}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {3}},
     {HloOpcode::kConvolution, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, Is1DVector, {}}},

    // External padding with constant zero
    {{HloOpcode::kPad, true, 0, IsExternalPadding, {2, 1}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Random truncated normal without post scale and add
    {{HloOpcode::kMultiply, true, 0, IsTruncatedNormal, {1, 2}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kMultiply, true, 0, nullptr, {3, 60}},
     {HloOpcode::kAdd, true, 0, nullptr, {50, 4}},
     {HloOpcode::kMultiply, true, 0, nullptr, {5, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {47, 6}},
     {HloOpcode::kMultiply, true, 0, nullptr, {7, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {44, 8}},
     {HloOpcode::kMultiply, true, 0, nullptr, {9, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {41, 10}},
     {HloOpcode::kMultiply, true, 0, nullptr, {11, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {38, 12}},
     {HloOpcode::kMultiply, true, 0, nullptr, {13, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {35, 14}},
     {HloOpcode::kMultiply, true, 0, nullptr, {15, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {32, 16}},
     {HloOpcode::kMultiply, true, 0, nullptr, {17, 19}},
     {HloOpcode::kAdd, true, 0, nullptr, {29, 18}},
     {HloOpcode::kMultiply, true, 0, nullptr, {26, 19}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 24, 20}},
     {HloOpcode::kAdd, true, 0, nullptr, {22, 21}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kPower, true, 0, nullptr, {55, 23}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kAdd, true, 0, nullptr, {55, 25}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 27, 28}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 30, 31}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 33, 34}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 36, 37}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 39, 40}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 42, 43}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 45, 46}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 48, 49}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kSelect, true, 0, nullptr, {53, 51, 52}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kLt, true, 0, nullptr, {55, 54}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kNegate, true, 0, nullptr, {56}},
     {HloOpcode::kLog, true, 0, nullptr, {57}},
     {HloOpcode::kMultiply, true, 0, nullptr, {59, 58}},
     {HloOpcode::kAdd, true, 0, nullptr, {60, 70}},
     {HloOpcode::kSubtract, true, 0, nullptr, {70, 60}},
     {HloOpcode::kAdd, true, 0, nullptr, {62, 61}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kMultiply, true, 0, nullptr, {64, 63}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kAdd, true, 0, nullptr, {66, 65}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kMultiply, true, 0, nullptr, {67, 68}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kRng, true, 0, nullptr, {69, 70}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, IsConstantOne, {}}},

    // Random normal with post scale and add
    {{HloOpcode::kAdd, true, 0, nullptr, {2, 1}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 3}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kRng, true, 0, IsRandomNormal, {5, 6}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}},

    // Random uniform with post scale and add
    {{HloOpcode::kAdd, true, 0, nullptr, {2, 1}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 3}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kRng, true, 0, IsRandomUniform, {5, 6}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}},

    // Random 2-constant without post scale and add
    {{HloOpcode::kRng, true, 0, IsRandomNormal, {1, 2}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}},

    // Random 2-constant without post scale and add
    {{HloOpcode::kRng, true, 0, IsRandomUniform, {1, 2}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}},

    // Average pool (valid)
    {{HloOpcode::kDivide, true, 0, IsAveragePool, {1, 3}},
     {HloOpcode::kReduceWindow, true, 0, Is2DReductionWindow, {4, 2}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Average pool (same)
    {{HloOpcode::kDivide, true, 0, IsAveragePool, {1, 3}},
     {HloOpcode::kReduceWindow, true, 0, Is2DReductionWindow, {8, 2}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {4}},
     {HloOpcode::kReduceWindow, true, 0, nullptr, {5, 7}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {6}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Average pool (same) - broadcast converted to reshape
    {{HloOpcode::kDivide, true, 0, IsAveragePool, {1, 3}},
     {HloOpcode::kReduceWindow, true, 0, Is2DReductionWindow, {8, 2}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kReshape, true, 0, nullptr, {4}},
     {HloOpcode::kReduceWindow, true, 0, nullptr, {5, 7}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {6}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}}},

    // Depthwise convolution (forward pass)
    {{HloOpcode::kConvolution, true, 0, nullptr, {14, 1}},
     {HloOpcode::kSelect, true, 0, nullptr, {6, 4, 2}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {3}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {5}},
     {HloOpcode::kReshape, true, 0, nullptr, {15}},
     {HloOpcode::kEq, true, 0, nullptr, {9, 7}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {8}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {10}},
     {HloOpcode::kDivide, true, 0, nullptr, {13, 11}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {12}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Depthwise convolution (forward pass, multiplier=1)
    {{HloOpcode::kConvolution, true, 0, nullptr, {10, 1}},
     {HloOpcode::kSelect, true, 0, nullptr, {6, 4, 2}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {3}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {5}},
     {HloOpcode::kReshape, true, 0, nullptr, {11}},
     {HloOpcode::kEq, true, 0, nullptr, {7, 8}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {9}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {9}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Depthwise convolution (backward pass, input)
    {{HloOpcode::kConvolution, true, 0, nullptr, {15, 1}},
     {HloOpcode::kReverse, true, 0, IsConvFilterTranspose, {2}},
     {HloOpcode::kSelect, true, 0, nullptr, {7, 5, 3}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {4}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {6}},
     {HloOpcode::kReshape, true, 0, nullptr, {16}},
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

    // Backprop input convolution
    {{HloOpcode::kConvolution, true, 0, nullptr, {2, 1}},
     {HloOpcode::kReverse, true, 0, IsConvFilterTranspose, {3}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Depthwise backprop filter convolution
    {{HloOpcode::kReshape, true, 0, nullptr, {1}},
     {HloOpcode::kReduce, true, 0, nullptr, {2, 4}},
     {HloOpcode::kSelect, true, 0, nullptr, {5, 13, 3}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {4}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kEq, true, 0, nullptr, {8, 6}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {7}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {9}},
     {HloOpcode::kDivide, true, 0, nullptr, {12, 10}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {11}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kConvolution, true, 0, nullptr, {14, 15}},
     {HloOpcode::kParameter, false, 0, nullptr, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Bias reduction and application
    {{HloOpcode::kSubtract, true, 0, IsOutputFeed, {1, 2}},
     {HloOpcode::kParameter, false, 0, IsTrueParameter, {}},
     {HloOpcode::kMultiply, true, 0, nullptr, {3, 5}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {4}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},
     {HloOpcode::kReduce, true, 0, IsBiasReduce, {7, 6}},
     {HloOpcode::kConstant, true, 0, IsConstantZero, {}},
     {HloOpcode::kParameter, false, 1, nullptr, {}}},

    // Remove convert to/from F32 before/after reduction, where initial value is
    // a constant
    {{HloOpcode::kConvert, true, 0, IsF32ToF16Convert, {1}},
     {HloOpcode::kReduce, true, 0, IsF32, {2, 3}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {4}},
     {HloOpcode::kConstant, true, 0, IsF32, {}},
     {HloOpcode::kParameter, false, 0, IsF16, {}}},

    // Remove convert to/from F32 before/after reduction, where initial value is
    // a convert from F16
    {{HloOpcode::kConvert, true, 0, IsF32ToF16Convert, {1}},
     {HloOpcode::kReduce, true, 0, IsF32, {2, 3}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {4}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {5}},
     {HloOpcode::kParameter, false, 0, IsF16, {}},
     {HloOpcode::kParameter, false, 1, IsF16, {}}},

    // Broadcast scalar constant (must be low priority)
    {{HloOpcode::kBroadcast, true, 0, nullptr, {1}},
     {HloOpcode::kConstant, true, 0, IsScalarConstant, {}}},
};

FuseOpsLate::FuseOpsLate(struct CompilerAnnotations& annotations) :
    HloMatcher(patterns, annotations, false) {}

ReplacedInstructions FuseOpsLate::ReplaceNodes(int pattern,
                                               const HloMatcherMatched& match) {
  std::string name("_pop_op_");
  name += fuse_info[pattern].name;

  char index = fuse_info[pattern].op_index;

  return OutlineExpressionFromComputation(match, name, index);
}

}  // namespace poplarplugin
}  // namespace xla
