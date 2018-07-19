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
    {"depthwise_filter", 14}, {"conv_with_reverse", 0},
    {"conv_with_reverse", 0}, {"depthwise_conv", 0},
    {"trunc_norm", 1},        {"trunc_norm", 1},
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
    // ------ Random Ops ------
    // Truncated normal - mean 0 std 1, non scalar shape
    {{HloOpcode::kMultiply, true, 0, IsTruncatedNormal, {99, 1}},
     {HloOpcode::kMultiply, true, 0, nullptr, {2, 82}},
     {HloOpcode::kAdd, true, 0, nullptr, {68, 3}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {63, 5}},
     {HloOpcode::kMultiply, true, 0, nullptr, {6, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {58, 7}},
     {HloOpcode::kMultiply, true, 0, nullptr, {8, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {53, 9}},
     {HloOpcode::kMultiply, true, 0, nullptr, {10, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {48, 11}},
     {HloOpcode::kMultiply, true, 0, nullptr, {12, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {43, 13}},
     {HloOpcode::kMultiply, true, 0, nullptr, {14, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {38, 15}},
     {HloOpcode::kMultiply, true, 0, nullptr, {16, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {33, 17}},
     {HloOpcode::kMultiply, true, 0, nullptr, {28, 18}},
     {HloOpcode::kSelect, true, 0, nullptr, {73, 25, 19}},
     {HloOpcode::kSubtract, true, 0, nullptr, {22, 20}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {21}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 3
     {HloOpcode::kPower, true, 0, nullptr, {76, 23}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {24}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.5
     {HloOpcode::kSubtract, true, 0, nullptr, {76, 26}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {27}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2.5
     {HloOpcode::kSelect, true, 0, nullptr, {73, 31, 29}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {30}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.000200214257
     {HloOpcode::kBroadcast, true, 0, nullptr, {32}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2.81022636e-08
     {HloOpcode::kSelect, true, 0, nullptr, {73, 36, 34}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {35}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.000100950558
     {HloOpcode::kBroadcast, true, 0, nullptr, {37}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 3.43273939e-07
     {HloOpcode::kSelect, true, 0, nullptr, {73, 41, 39}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {40}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00134934322
     {HloOpcode::kBroadcast, true, 0, nullptr, {42}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -3.5233877e-06
     {HloOpcode::kSelect, true, 0, nullptr, {73, 46, 44}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {45}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.00367342844
     {HloOpcode::kBroadcast, true, 0, nullptr, {47}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -4.39150654e-06
     {HloOpcode::kSelect, true, 0, nullptr, {73, 51, 49}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {50}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00573950773
     {HloOpcode::kBroadcast, true, 0, nullptr, {52}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00021858087
     {HloOpcode::kSelect, true, 0, nullptr, {73, 56, 54}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {55}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.0076224613
     {HloOpcode::kBroadcast, true, 0, nullptr, {57}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.00125372503
     {HloOpcode::kSelect, true, 0, nullptr, {73, 61, 59}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {60}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00943887047
     {HloOpcode::kBroadcast, true, 0, nullptr, {62}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.00417768164
     {HloOpcode::kSelect, true, 0, nullptr, {73, 66, 64}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {65}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1.00167406
     {HloOpcode::kBroadcast, true, 0, nullptr, {67}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.246640727
     {HloOpcode::kSelect, true, 0, nullptr, {73, 71, 69}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {70}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2.83297682
     {HloOpcode::kBroadcast, true, 0, nullptr, {72}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1.50140941
     {HloOpcode::kLt, true, 0, nullptr, {76, 74}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {75}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 5
     {HloOpcode::kNegate, true, 0, nullptr, {77}},
     {HloOpcode::kLog, true, 0, nullptr, {78}},
     {HloOpcode::kMultiply, true, 0, nullptr, {81, 79}},
     {HloOpcode::kAdd, true, 0, nullptr, {80, 82}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {98}},
     {HloOpcode::kSubtract, true, 0, nullptr, {97, 82}},
     {HloOpcode::kSubtract, true, 0, nullptr, {85, 83}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {84}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1
     {HloOpcode::kMultiply, true, 0, nullptr, {95, 86}},
     {HloOpcode::kAdd, true, 0, nullptr, {93, 87}},
     {HloOpcode::kMultiply, true, 0, nullptr, {91, 88}},
     {HloOpcode::kRng, true, 0, nullptr, {90, 89}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1.17549435e-38
     {HloOpcode::kBroadcast, true, 0, nullptr, {92}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.954499722
     {HloOpcode::kBroadcast, true, 0, nullptr, {94}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.0227501318
     {HloOpcode::kBroadcast, true, 0, nullptr, {96}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2
     {HloOpcode::kBroadcast, true, 0, nullptr, {98}},
     {HloOpcode::kConstant, true, 0, IsConstantOne, {}},  // 1
     {HloOpcode::kBroadcast, true, 0, nullptr, {100}},
     {HloOpcode::kConstant, true, 0, nullptr, {}}},  // 1.41421354
    // Truncated normal - mean 0 std 1, scalar shape
    {{HloOpcode::kMultiply, true, 0, nullptr, {89, 1}},
     {HloOpcode::kMultiply, true, 0, nullptr, {2, 77}},
     {HloOpcode::kAdd, true, 0, nullptr, {65, 3}},
     {HloOpcode::kMultiply, true, 0, nullptr, {4, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {60, 5}},
     {HloOpcode::kMultiply, true, 0, nullptr, {6, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {55, 7}},
     {HloOpcode::kMultiply, true, 0, nullptr, {8, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {50, 9}},
     {HloOpcode::kMultiply, true, 0, nullptr, {10, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {45, 11}},
     {HloOpcode::kMultiply, true, 0, nullptr, {12, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {40, 13}},
     {HloOpcode::kMultiply, true, 0, nullptr, {14, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {35, 15}},
     {HloOpcode::kMultiply, true, 0, nullptr, {16, 18}},
     {HloOpcode::kAdd, true, 0, nullptr, {30, 17}},
     {HloOpcode::kMultiply, true, 0, nullptr, {25, 18}},
     {HloOpcode::kSelect, true, 0, nullptr, {70, 23, 19}},
     {HloOpcode::kSubtract, true, 0, nullptr, {21, 20}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 3
     {HloOpcode::kPower, true, 0, nullptr, {72, 22}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.5
     {HloOpcode::kSubtract, true, 0, nullptr, {72, 24}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2.5
     {HloOpcode::kSelect, true, 0, nullptr, {70, 28, 26}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {27}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.000200214257
     {HloOpcode::kBroadcast, true, 0, nullptr, {29}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2.81022636e-08
     {HloOpcode::kSelect, true, 0, nullptr, {70, 33, 31}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {32}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.000100950558
     {HloOpcode::kBroadcast, true, 0, nullptr, {34}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 3.43273939e-07
     {HloOpcode::kSelect, true, 0, nullptr, {70, 38, 36}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {37}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00134934322
     {HloOpcode::kBroadcast, true, 0, nullptr, {39}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -3.5233877e-06
     {HloOpcode::kSelect, true, 0, nullptr, {70, 43, 41}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {42}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.00367342844
     {HloOpcode::kBroadcast, true, 0, nullptr, {44}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -4.39150654e-06
     {HloOpcode::kSelect, true, 0, nullptr, {70, 48, 46}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {47}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00573950773
     {HloOpcode::kBroadcast, true, 0, nullptr, {49}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00021858087
     {HloOpcode::kSelect, true, 0, nullptr, {70, 53, 51}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {52}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.0076224613
     {HloOpcode::kBroadcast, true, 0, nullptr, {54}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.00125372503
     {HloOpcode::kSelect, true, 0, nullptr, {70, 58, 56}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {57}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.00943887047
     {HloOpcode::kBroadcast, true, 0, nullptr, {59}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // -0.00417768164
     {HloOpcode::kSelect, true, 0, nullptr, {70, 63, 61}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {62}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1.00167406
     {HloOpcode::kBroadcast, true, 0, nullptr, {64}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 0.246640727
     {HloOpcode::kSelect, true, 0, nullptr, {70, 68, 66}},
     {HloOpcode::kBroadcast, true, 0, nullptr, {67}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 2.83297682
     {HloOpcode::kBroadcast, true, 0, nullptr, {69}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1.50140941
     {HloOpcode::kLt, true, 0, nullptr, {72, 71}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 5
     {HloOpcode::kNegate, true, 0, nullptr, {73}},
     {HloOpcode::kLog, true, 0, nullptr, {74}},
     {HloOpcode::kMultiply, true, 0, nullptr, {76, 75}},
     {HloOpcode::kAdd, true, 0, nullptr, {88, 77}},
     {HloOpcode::kSubtract, true, 0, nullptr, {88, 77}},
     {HloOpcode::kSubtract, true, 0, nullptr, {79, 78}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},  // 1
     {HloOpcode::kMultiply, true, 0, nullptr, {87, 80}},
     {HloOpcode::kAdd, true, 0, nullptr, {86, 81}},
     {HloOpcode::kMultiply, true, 0, nullptr, {85, 82}},
     {HloOpcode::kRng, true, 0, nullptr, {84, 83}},
     {HloOpcode::kConstant, true, 0, nullptr, {}},   // 1
     {HloOpcode::kConstant, true, 0, nullptr, {}},   // 1.17549435e-38
     {HloOpcode::kConstant, true, 0, nullptr, {}},   // 0.954499722
     {HloOpcode::kConstant, true, 0, nullptr, {}},   // 0.0227501318
     {HloOpcode::kConstant, true, 0, nullptr, {}},   // 2
     {HloOpcode::kConstant, true, 0, nullptr, {}},   // 1
     {HloOpcode::kConstant, true, 0, nullptr, {}}},  // 1.41421354
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
