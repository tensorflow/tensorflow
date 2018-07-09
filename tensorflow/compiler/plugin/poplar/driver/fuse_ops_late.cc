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
     {HloOpcode::kCall, false, 0, IsPopOpsConvolution, {}},
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

    // Random truncated normal mean 0 sd 1
    {{HloOpcode::kMultiply, true, 0, IsTruncatedNormal, {70,1}},
     {HloOpcode::kMultiply, true, 0, nullptr, {2,59}},
     {HloOpcode::kAdd, true, 0, nullptr, {49,3}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {46,5}},
     {HloOpcode::kMultiply, true, 0, nullptr, {6,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {43,7}},
     {HloOpcode::kMultiply, true, 0, nullptr, {8,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {40,9}},
     {HloOpcode::kMultiply, true, 0, nullptr, {10,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {37,11}},
     {HloOpcode::kMultiply, true, 0, nullptr, {12,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {34,13}},
     {HloOpcode::kMultiply, true, 0, nullptr, {14,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {31,15}},
     {HloOpcode::kMultiply, true, 0, nullptr, {16,18}},
     {HloOpcode::kAdd, true, 0, nullptr, {28,17}},
     {HloOpcode::kMultiply, true, 0, nullptr, {25,18}},
     {HloOpcode::kSelect, true, 0, nullptr, {52,23,19}},
     {HloOpcode::kAdd, true, 0, nullptr, {21,20}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -3
     {HloOpcode::kPower, true, 0, nullptr, {54,22}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.5
     {HloOpcode::kAdd, true, 0, nullptr, {54,24}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -2.5
     {HloOpcode::kSelect, true, 0, nullptr, {52,27,26}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -0.000200214257
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 2.81022636e-08
     {HloOpcode::kSelect, true, 0, nullptr, {52,30,29}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.000100950558
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 3.43273939e-07
     {HloOpcode::kSelect, true, 0, nullptr, {52,33,32}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.00134934322
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -3.5233877e-06
     {HloOpcode::kSelect, true, 0, nullptr, {52,36,35}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -0.00367342844
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -4.39150654e-06
     {HloOpcode::kSelect, true, 0, nullptr, {52,39,38}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.00573950773
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.00021858087
     {HloOpcode::kSelect, true, 0, nullptr, {52,42,41}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -0.0076224613
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -0.00125372503
     {HloOpcode::kSelect, true, 0, nullptr, {52,45,44}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.00943887047
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -0.00417768164
     {HloOpcode::kSelect, true, 0, nullptr, {52,48,47}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 1.00167406
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.246640727
     {HloOpcode::kSelect, true, 0, nullptr, {52,51,50}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 2.83297682
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 1.50140941
     {HloOpcode::kLt, true, 0, nullptr, {54,53}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 5
     {HloOpcode::kNegate, true, 0, nullptr, {55}},
     {HloOpcode::kLog, true, 0, nullptr, {56}},
     {HloOpcode::kMultiply, true, 0, nullptr, {58,57}},
     {HloOpcode::kAdd, true, 0, nullptr, {59,69}},
     {HloOpcode::kSubtract, true, 0, nullptr, {69,59}},
     {HloOpcode::kAdd, true, 0, nullptr, {61,60}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // -1
     {HloOpcode::kMultiply, true, 0, nullptr, {68,62}},
     {HloOpcode::kAdd, true, 0, nullptr, {64,63}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.0227501318
     {HloOpcode::kMultiply, true, 0, nullptr, {67,65}},
     {HloOpcode::kRng, true, 0, nullptr, {66,69}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 1.17549435e-38
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 0.954499722
     {HloOpcode::kConstant, true, 0, nullptr, {}}, // 2
     {HloOpcode::kConstant, true, 0, IsConstantOne, {}}, // 1
     {HloOpcode::kConstant, true, 0, nullptr, {}}}, // 1.41421354

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

FuseOpsLate::FuseOpsLate(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false) {}

ReplacedInstructions FuseOpsLate::ReplaceNodes(int pattern,
                                               const HloMatcherMatched& match) {
  std::string name("_pop_op_");
  name += fuse_info[pattern].name;

  char index = fuse_info[pattern].op_index;

  return OutlineExpressionFromComputation(match, name, index);
}

}  // namespace poplarplugin
}  // namespace xla
