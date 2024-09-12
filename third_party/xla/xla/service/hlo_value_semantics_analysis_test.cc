/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/hlo_value_semantics_analysis.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

const char kMnistHlo[] = R"(
HloModule MnistTrainingLoopWithInfeed.140, entry_computation_layout={(f32[784,128]{1,0:T(8,128)},f32[128]{0:T(256)},f32[128,32]{1,0:T(8,128)},f32[32]{0:T(256)},f32[32,10]{1,0:T(8,128)},f32[10]{0:T(256)})->(f32[784,128]{1,0:T(8,128)}, f32[128]{0:T(256)}, f32[128,32]{1,0:T(8,128)}, f32[32]{0:T(256)}, f32[32,10]{1,0:T(8,128)}, /*index=5*/f32[10]{0:T(256)})}

relu.9 {
  x.10 = f32[] parameter(0)
  constant.11 = f32[] constant(0)
  ROOT maximum.12 = f32[] maximum(x.10, constant.11)
}

max_F32.17 {
  lhs.18 = f32[] parameter(0)
  rhs.19 = f32[] parameter(1)
  ROOT maximum.20 = f32[] maximum(lhs.18, rhs.19)
}

add_F32.1 {
  lhs.22 = f32[] parameter(0)
  rhs.23 = f32[] parameter(1)
  ROOT add.24 = f32[] add(lhs.22, rhs.23)
}

relu_gradients.29 {
  activation.30 = f32[] parameter(0)
  constant.32 = f32[] constant(0)
  compare.33 = pred[] compare(activation.30, constant.32), direction=GT
  backprop.31 = f32[] parameter(1)
  ROOT select.34 = f32[] select(compare.33, backprop.31, constant.32)
}

body.49 {
  after-all.51 = token[] after-all()
  infeed.52 = ((f32[100,784]{1,0}, f32[100,10]{1,0}, pred[]), token[]) infeed(after-all.51)
  get.53 = (f32[100,784]{1,0}, f32[100,10]{1,0}, pred[]) get-tuple-element(infeed.52), index=0
  get.54 = f32[100,784]{1,0} get-tuple-element(get.53), index=0
  prev.50 = (f32[784,128]{1,0}, f32[128]{0}, f32[128,32]{1,0}, f32[32]{0}, f32[32,10]{1,0}, /*index=5*/f32[10]{0}, pred[]) parameter(0)
  get.57 = f32[784,128]{1,0} get-tuple-element(prev.50), index=0
  dot.63 = f32[100,128]{1,0} dot(get.54, get.57), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  get.58 = f32[128]{0} get-tuple-element(prev.50), index=1
  broadcast.64 = f32[100,128]{1,0} broadcast(get.58), dimensions={1}
  add.65 = f32[100,128]{1,0} add(dot.63, broadcast.64)
  map.66 = f32[100,128]{1,0} map(add.65), dimensions={0,1}, to_apply=relu.9
  get.59 = f32[128,32]{1,0} get-tuple-element(prev.50), index=2
  dot.67 = f32[100,32]{1,0} dot(map.66, get.59), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  get.60 = f32[32]{0} get-tuple-element(prev.50), index=3
  broadcast.68 = f32[100,32]{1,0} broadcast(get.60), dimensions={1}
  add.69 = f32[100,32]{1,0} add(dot.67, broadcast.68)
  map.70 = f32[100,32]{1,0} map(add.69), dimensions={0,1}, to_apply=relu.9
  get.61 = f32[32,10]{1,0} get-tuple-element(prev.50), index=4
  dot.71 = f32[100,10]{1,0} dot(map.70, get.61), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  get.62 = f32[10]{0} get-tuple-element(prev.50), index=5
  broadcast.72 = f32[100,10]{1,0} broadcast(get.62), dimensions={1}
  add.73 = f32[100,10]{1,0} add(dot.71, broadcast.72)
  constant.74 = f32[] constant(-inf)
  reduce.75 = f32[100]{0} reduce(add.73, constant.74), dimensions={1}, to_apply=max_F32.17
  broadcast.76 = f32[100,10]{1,0} broadcast(reduce.75), dimensions={0}
  subtract.77 = f32[100,10]{1,0} subtract(add.73, broadcast.76)
  exponential.78 = f32[100,10]{1,0} exponential(subtract.77)
  constant.79 = f32[] constant(0)
  reduce.80 = f32[100]{0} reduce(exponential.78, constant.79), dimensions={1}, to_apply=add_F32.1
  broadcast.81 = f32[100,10]{1,0} broadcast(reduce.80), dimensions={0}
  divide.82 = f32[100,10]{1,0} divide(exponential.78, broadcast.81)
  get.55 = f32[100,10]{1,0} get-tuple-element(get.53), index=1
  subtract.83 = f32[100,10]{1,0} subtract(divide.82, get.55)
  transpose.88 = f32[10,32]{0,1} transpose(get.61), dimensions={1,0}
  dot.89 = f32[100,32]{1,0} dot(subtract.83, transpose.88), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  map.90 = f32[100,32]{1,0} map(map.70, dot.89), dimensions={0,1}, to_apply=relu_gradients.29
  transpose.95 = f32[32,128]{0,1} transpose(get.59), dimensions={1,0}
  dot.96 = f32[100,128]{1,0} dot(map.90, transpose.95), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  map.97 = f32[100,128]{1,0} map(map.66, dot.96), dimensions={0,1}, to_apply=relu_gradients.29
  transpose.98 = f32[784,100]{0,1} transpose(get.54), dimensions={1,0}
  dot.99 = f32[784,128]{1,0} dot(transpose.98, map.97), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.104 = f32[] constant(0.01)
  broadcast.105 = f32[784,128]{1,0} broadcast(constant.104), dimensions={}
  multiply.106 = f32[784,128]{1,0} multiply(dot.99, broadcast.105)
  subtract.107 = f32[784,128]{1,0} subtract(get.57, multiply.106)
  reduce.101 = f32[128]{0} reduce(map.97, constant.79), dimensions={0}, to_apply=add_F32.1
  broadcast.109 = f32[128]{0} broadcast(constant.104), dimensions={}
  multiply.110 = f32[128]{0} multiply(reduce.101, broadcast.109)
  subtract.111 = f32[128]{0} subtract(get.58, multiply.110)
  transpose.91 = f32[128,100]{0,1} transpose(map.66), dimensions={1,0}
  dot.92 = f32[128,32]{1,0} dot(transpose.91, map.90), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast.113 = f32[128,32]{1,0} broadcast(constant.104), dimensions={}
  multiply.114 = f32[128,32]{1,0} multiply(dot.92, broadcast.113)
  subtract.115 = f32[128,32]{1,0} subtract(get.59, multiply.114)
  reduce.94 = f32[32]{0} reduce(map.90, constant.79), dimensions={0}, to_apply=add_F32.1
  broadcast.117 = f32[32]{0} broadcast(constant.104), dimensions={}
  multiply.118 = f32[32]{0} multiply(reduce.94, broadcast.117)
  subtract.119 = f32[32]{0} subtract(get.60, multiply.118)
  transpose.84 = f32[32,100]{0,1} transpose(map.70), dimensions={1,0}
  dot.85 = f32[32,10]{1,0} dot(transpose.84, subtract.83), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast.121 = f32[32,10]{1,0} broadcast(constant.104), dimensions={}
  multiply.122 = f32[32,10]{1,0} multiply(dot.85, broadcast.121)
  subtract.123 = f32[32,10]{1,0} subtract(get.61, multiply.122)
  reduce.87 = f32[10]{0} reduce(subtract.83, constant.79), dimensions={0}, to_apply=add_F32.1
  broadcast.125 = f32[10]{0} broadcast(constant.104), dimensions={}
  multiply.126 = f32[10]{0} multiply(reduce.87, broadcast.125)
  subtract.127 = f32[10]{0} subtract(get.62, multiply.126)
  get.56 = pred[] get-tuple-element(get.53), index=2
  ROOT tuple.128 = (f32[784,128]{1,0}, f32[128]{0}, f32[128,32]{1,0}, f32[32]{0}, f32[32,10]{1,0}, /*index=5*/f32[10]{0}, pred[]) tuple(subtract.107, subtract.111, subtract.115, subtract.119, subtract.123, subtract.127, get.56)
}

condition.129 {
  prev.130 = (f32[784,128]{1,0}, f32[128]{0}, f32[128,32]{1,0}, f32[32]{0}, f32[32,10]{1,0}, /*index=5*/f32[10]{0}, pred[]) parameter(0)
  ROOT get.131 = pred[] get-tuple-element(prev.130), index=6
}

ENTRY MnistTrainingLoopWithInfeed.140 {
  layer1_weights.1 = f32[784,128]{1,0} parameter(0)
  layer1_biases.2 = f32[128]{0} parameter(1)
  layer2_weights.3 = f32[128,32]{1,0} parameter(2)
  layer2_biases.4 = f32[32]{0} parameter(3)
  layer3_weights.5 = f32[32,10]{1,0} parameter(4)
  layer3_biases.6 = f32[10]{0} parameter(5)
  constant.7 = pred[] constant(true)
  tuple.8 = (f32[784,128]{1,0}, f32[128]{0}, f32[128,32]{1,0}, f32[32]{0}, f32[32,10]{1,0}, /*index=5*/f32[10]{0}, pred[]) tuple(layer1_weights.1, layer1_biases.2, layer2_weights.3, layer2_biases.4, layer3_weights.5, layer3_biases.6, constant.7)
  while.132 = (f32[784,128]{1,0}, f32[128]{0}, f32[128,32]{1,0}, f32[32]{0}, f32[32,10]{1,0}, /*index=5*/f32[10]{0}, pred[]) while(tuple.8), condition=condition.129, body=body.49
  get.133 = f32[784,128]{1,0} get-tuple-element(while.132), index=0
  get.134 = f32[128]{0} get-tuple-element(while.132), index=1
  get.135 = f32[128,32]{1,0} get-tuple-element(while.132), index=2
  get.136 = f32[32]{0} get-tuple-element(while.132), index=3
  get.137 = f32[32,10]{1,0} get-tuple-element(while.132), index=4
  get.138 = f32[10]{0} get-tuple-element(while.132), index=5
  ROOT tuple.139 = (f32[784,128]{1,0}, f32[128]{0}, f32[128,32]{1,0}, f32[32]{0}, f32[32,10]{1,0}, /*index=5*/f32[10]{0}) tuple(get.133, get.134, get.135, get.136, get.137, get.138)
}
)";

class HloValueSemanticsAnalysisTest : public HloTestBase {
 public:
  bool HasLabel(const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
                HloModule* module, absl::string_view instruction_name,
                const HloValueSemanticLabel& expected_label) {
    HloInstruction* instruction = FindInstruction(module, instruction_name);
    const HloValueSemantics* semantics =
        hlo_value_semantics_analysis.GetSemantics(instruction);
    LOG(INFO) << "instruction: " << instruction->ToString()
              << semantics->ToString();
    return semantics->label() == expected_label;
  }
  bool IsStatic(const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
                HloModule* module, absl::string_view instruction_name) {
    return HasLabel(hlo_value_semantics_analysis, module, instruction_name,
                    HloValueSemanticLabel::kStatic);
  }
  bool IsWeight(const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
                HloModule* module, absl::string_view instruction_name) {
    return HasLabel(hlo_value_semantics_analysis, module, instruction_name,
                    HloValueSemanticLabel::kWeight);
  }
  bool IsActivation(
      const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
      HloModule* module, absl::string_view instruction_name) {
    return HasLabel(hlo_value_semantics_analysis, module, instruction_name,
                    HloValueSemanticLabel::kActivation);
  }
  bool IsActivationGradient(
      const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
      HloModule* module, absl::string_view instruction_name) {
    return HasLabel(hlo_value_semantics_analysis, module, instruction_name,
                    HloValueSemanticLabel::kActivationGradient);
  }
  bool IsWeightGradient(
      const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
      HloModule* module, absl::string_view instruction_name) {
    return HasLabel(hlo_value_semantics_analysis, module, instruction_name,
                    HloValueSemanticLabel::kWeightGradient);
  }
  bool IsTupleOrToken(
      const HloValueSemanticsAnalysis& hlo_value_semantics_analysis,
      HloModule* module, absl::string_view instruction_name) {
    return HasLabel(hlo_value_semantics_analysis, module, instruction_name,
                    HloValueSemanticLabel::kTupleOrToken);
  }
};

TEST_F(HloValueSemanticsAnalysisTest, OneMatmul) {
  const std::string module_str = R"(
HloModule OneMatmul

region_0.39 {
  Arg_0.40 = f32[] parameter(0)
  Arg_1.41 = f32[] parameter(1)
  ROOT add.42 = f32[] add(Arg_0.40, Arg_1.41)
}

ENTRY entry {
  Arg_1.2 = f32[32,128]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  Arg_7.8 = f32[4,32]{1,0} parameter(1), sharding={devices=[2,1]0,1}
  copy = f32[4,32]{1,0} copy(Arg_7.8), sharding={devices=[2,1]0,1}
  dot.0 = f32[4,128]{1,0} dot(copy, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  constant.5 = f32[] constant(0), sharding={replicated}
  broadcast.2 = f32[4,128]{1,0} broadcast(constant.5), dimensions={}, sharding={devices=[2,1]0,1}
  maximum.33 = f32[4,128]{1,0} maximum(dot.0, broadcast.2), sharding={devices=[2,1]0,1}
  compare.34 = pred[4,128]{1,0} compare(dot.0, maximum.33), direction=EQ, sharding={devices=[2,1]0,1}
  constant.4 = f32[] constant(1), sharding={replicated}
  broadcast.1 = f32[4,128]{1,0} broadcast(constant.4), dimensions={}, sharding={devices=[2,1]0,1}
  select.35 = f32[4,128]{1,0} select(compare.34, broadcast.1, broadcast.2), sharding={devices=[2,1]0,1}
  dot.2 = f32[32,128]{0,1} dot(copy, select.35), lhs_contracting_dims={0}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  constant.11 = f32[] constant(-0.01), sharding={replicated}
  broadcast.12 = f32[32,128]{1,0} broadcast(constant.11), dimensions={}, sharding={devices=[2,1]0,1}
  multiply.52 = f32[32,128]{0,1} multiply(dot.2, broadcast.12), sharding={devices=[2,1]0,1}
  add.93 = f32[32,128]{1,0} add(Arg_1.2, multiply.52), sharding={devices=[2,1]0,1}
  reduce.43 = f32[] reduce(maximum.33, constant.5), dimensions={0,1}, to_apply=region_0.39, sharding={replicated}
  ROOT tuple.109 = (f32[32,128]{1,0}, f32[]) tuple(add.93, reduce.43), sharding={{devices=[2,1]0,1}, {replicated}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1,
                                                /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(*module));
  EXPECT_TRUE(IsWeight(*hlo_value_semantics_analysis, module.get(), "copy"));
  EXPECT_TRUE(IsWeight(*hlo_value_semantics_analysis, module.get(), "Arg_1.2"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.0"));
  EXPECT_TRUE(
      IsStatic(*hlo_value_semantics_analysis, module.get(), "select.35"));
  EXPECT_TRUE(IsWeight(*hlo_value_semantics_analysis, module.get(), "dot.2"));
}

TEST_F(HloValueSemanticsAnalysisTest, HandleConditional) {
  const std::string module_str = R"(
    HloModule Module

    branch0 {
      tparam = f32[4] parameter(0)
      tgte1 = f32[4] ceil(tparam)
      ROOT tuple = (f32[4], f32[4]) tuple(tparam, tgte1)
    }

    branch1 {
      fparam = f32[4] parameter(0)
      %async-start = ((f32[4]), f32[4], s32[]) abs-start(f32[4] fparam), async_execution_thread="parallel_thread"
      %async-done = f32[4] abs-done(((f32[4]), f32[4], s32[]) %async-start)
      ROOT tuple = (f32[4], f32[4]) tuple(fparam, %async-done)
    }

    ENTRY entry {
      p0 = f32[4] parameter(0)
      b0 = s32[] parameter(1)
      ROOT conditional = (f32[4], f32[4]) conditional(b0, p0, p0),
        branch_computations={branch0, branch1}
    }
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1,
                                                /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(*module));
  EXPECT_TRUE(IsTupleOrToken(*hlo_value_semantics_analysis, module.get(),
                             "conditional"));
}

TEST_F(HloValueSemanticsAnalysisTest, TwoMatmuls) {
  const std::string module_str = R"(
HloModule TwoMatmuls

region_0.44 {
  Arg_0.45 = f32[] parameter(0)
  Arg_1.46 = f32[] parameter(1)
  ROOT add.47 = f32[] add(Arg_0.45, Arg_1.46)
}

ENTRY entry {
  Arg_1.2 = f32[32,128]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  Arg_8.9 = f32[4,32]{1,0} parameter(2), sharding={devices=[2,1]0,1}
  copy = f32[4,32]{1,0} copy(Arg_8.9), sharding={devices=[2,1]0,1}
  dot.0 = f32[4,128]{1,0} dot(copy, Arg_1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  Arg_2.3 = f32[128,8]{1,0} parameter(1), sharding={devices=[1,2]0,1}
  dot.1 = f32[4,8]{1,0} dot(dot.0, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[1,2]0,1}
  constant.5 = f32[] constant(0), sharding={replicated}
  broadcast.1 = f32[4,8]{1,0} broadcast(constant.5), dimensions={}, sharding={devices=[1,2]0,1}
  maximum.38 = f32[4,8]{1,0} maximum(dot.1, broadcast.1), sharding={devices=[1,2]0,1}
  compare.39 = pred[4,8]{1,0} compare(dot.1, maximum.38), direction=EQ, sharding={devices=[1,2]0,1}
  constant.4 = f32[] constant(1), sharding={replicated}
  broadcast.0 = f32[4,8]{1,0} broadcast(constant.4), dimensions={}, sharding={devices=[1,2]0,1}
  select.40 = f32[4,8]{1,0} select(compare.39, broadcast.0, broadcast.1), sharding={devices=[1,2]0,1}
  dot.2 = f32[4,128]{1,0} dot(select.40, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,1]0,1}
  dot.5 = f32[32,128]{0,1} dot(copy, dot.2), lhs_contracting_dims={0}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  constant.12 = f32[] constant(-0.01), sharding={replicated}
  broadcast.13 = f32[32,128]{1,0} broadcast(constant.12), dimensions={}, sharding={devices=[2,1]0,1}
  multiply.68 = f32[32,128]{0,1} multiply(dot.5, broadcast.13), sharding={devices=[2,1]0,1}
  add.79 = f32[32,128]{1,0} add(Arg_1.2, multiply.68), sharding={devices=[2,1]0,1}
  dot.6 = f32[128,8]{0,1} dot(dot.0, select.40), lhs_contracting_dims={0}, rhs_contracting_dims={0}, sharding={devices=[1,2]0,1}
  broadcast.11 = f32[128,8]{1,0} broadcast(constant.12), dimensions={}, sharding={devices=[1,2]0,1}
  multiply.69 = f32[128,8]{0,1} multiply(dot.6, broadcast.11), sharding={devices=[1,2]0,1}
  add.80 = f32[128,8]{1,0} add(Arg_2.3, multiply.69), sharding={devices=[1,2]0,1}
  reduce.48 = f32[] reduce(maximum.38, constant.5), dimensions={0,1}, to_apply=region_0.44, sharding={replicated}
  ROOT tuple.95 = (f32[32,128]{1,0}, f32[128,8]{1,0}, f32[]) tuple(add.79, add.80, reduce.48), sharding={{devices=[2,1]0,1}, {devices=[1,2]0,1}, {replicated}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1,
                                                /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(*module));
  EXPECT_FALSE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "copy"));
  EXPECT_FALSE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "Arg_1.2"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.0"));
  EXPECT_FALSE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "Arg_2.3"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.1"));
  EXPECT_TRUE(
      IsStatic(*hlo_value_semantics_analysis, module.get(), "select.40"));
  EXPECT_TRUE(IsWeight(*hlo_value_semantics_analysis, module.get(), "dot.2"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.5"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.6"));
}

TEST_F(HloValueSemanticsAnalysisTest, RepeatWhile) {
  const std::string module_str = R"(
HloModule RepeatWhile

region_0.52 {
  arg_tuple.53 = (s32[], f32[4,32]{1,0}, f32[3,4,128]{2,1,0}, f32[3,4,32]{2,1,0}, f32[3,4,32]{2,1,0}, /*index=5*/f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}) parameter(0), sharding={{replicated}, {devices=[2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}}
  get-tuple-element.54 = s32[] get-tuple-element(arg_tuple.53), index=0, sharding={replicated}
  constant.61 = s32[] constant(1), sharding={replicated}
  add.105 = s32[] add(get-tuple-element.54, constant.61), sharding={replicated}
  get-tuple-element.55 = f32[4,32]{1,0} get-tuple-element(arg_tuple.53), index=1, sharding={devices=[2,1]0,1}
  get-tuple-element.59 = f32[3,32,128]{2,1,0} get-tuple-element(arg_tuple.53), index=5, sharding={devices=[1,2,1]0,1}
  constant.69 = s32[] constant(0), sharding={replicated}
  compare.70 = pred[] compare(get-tuple-element.54, constant.69), direction=LT, sharding={replicated}
  constant.68 = s32[] constant(3), sharding={replicated}
  add.71 = s32[] add(get-tuple-element.54, constant.68), sharding={replicated}
  select.72 = s32[] select(compare.70, add.71, get-tuple-element.54), sharding={replicated}
  dynamic-slice.73 = f32[1,32,128]{2,1,0} dynamic-slice(get-tuple-element.59, select.72, constant.69, constant.69), dynamic_slice_sizes={1,32,128}, sharding={devices=[1,2,1]0,1}
  reshape.74 = f32[32,128]{1,0} reshape(dynamic-slice.73), sharding={devices=[2,1]0,1}
  dot.0 = f32[4,128]{1,0} dot(get-tuple-element.55, reshape.74), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  get-tuple-element.60 = f32[3,128,32]{2,1,0} get-tuple-element(arg_tuple.53), index=6, sharding={devices=[1,1,2]0,1}
  dynamic-slice.78 = f32[1,128,32]{2,1,0} dynamic-slice(get-tuple-element.60, select.72, constant.69, constant.69), dynamic_slice_sizes={1,128,32}, sharding={devices=[1,1,2]0,1}
  reshape.79 = f32[128,32]{1,0} reshape(dynamic-slice.78), sharding={devices=[1,2]0,1}
  dot.1 = f32[4,32]{1,0} dot(dot.0, reshape.79), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  constant.43 = f32[] constant(0), sharding={replicated}
  broadcast.2 = f32[4,32]{1,0} broadcast(constant.43), dimensions={}, sharding={devices=[2,1]0,1}
  maximum.84 = f32[4,32]{1,0} maximum(dot.1, broadcast.2), sharding={devices=[2,1]0,1}
  get-tuple-element.56 = f32[3,4,128]{2,1,0} get-tuple-element(arg_tuple.53), index=2, sharding={devices=[1,2,1]0,1}
  reshape.90 = f32[1,4,128]{2,1,0} reshape(dot.0), sharding={devices=[1,2,1]0,1}
  dynamic-update-slice.94 = f32[3,4,128]{2,1,0} dynamic-update-slice(get-tuple-element.56, reshape.90, select.72, constant.69, constant.69), sharding={devices=[1,2,1]0,1}
  get-tuple-element.57 = f32[3,4,32]{2,1,0} get-tuple-element(arg_tuple.53), index=3, sharding={devices=[1,2,1]0,1}
  compare.85 = pred[4,32]{1,0} compare(dot.1, maximum.84), direction=EQ, sharding={devices=[2,1]0,1}
  constant.42 = f32[] constant(1), sharding={replicated}
  broadcast.1 = f32[4,32]{1,0} broadcast(constant.42), dimensions={}, sharding={devices=[2,1]0,1}
  select.86 = f32[4,32]{1,0} select(compare.85, broadcast.1, broadcast.2), sharding={devices=[2,1]0,1}
  reshape.95 = f32[1,4,32]{2,1,0} reshape(select.86), sharding={devices=[1,2,1]0,1}
  dynamic-update-slice.99 = f32[3,4,32]{2,1,0} dynamic-update-slice(get-tuple-element.57, reshape.95, select.72, constant.69, constant.69), sharding={devices=[1,2,1]0,1}
  get-tuple-element.58 = f32[3,4,32]{2,1,0} get-tuple-element(arg_tuple.53), index=4, sharding={devices=[1,2,1]0,1}
  reshape.100 = f32[1,4,32]{2,1,0} reshape(get-tuple-element.55), sharding={devices=[1,2,1]0,1}
  dynamic-update-slice.104 = f32[3,4,32]{2,1,0} dynamic-update-slice(get-tuple-element.58, reshape.100, select.72, constant.69, constant.69), sharding={devices=[1,2,1]0,1}
  ROOT tuple.106 = (s32[], f32[4,32]{1,0}, f32[3,4,128]{2,1,0}, f32[3,4,32]{2,1,0}, f32[3,4,32]{2,1,0}, /*index=5*/f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}) tuple(add.105, maximum.84, dynamic-update-slice.94, dynamic-update-slice.99, dynamic-update-slice.104, /*index=5*/get-tuple-element.59, get-tuple-element.60), sharding={{replicated}, {devices=[2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}}
}

region_1.107 {
  arg_tuple.108 = (s32[], f32[4,32]{1,0}, f32[3,4,128]{2,1,0}, f32[3,4,32]{2,1,0}, f32[3,4,32]{2,1,0}, /*index=5*/f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}) parameter(0), sharding={{replicated}, {devices=[2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}}
  get-tuple-element.109 = s32[] get-tuple-element(arg_tuple.108), index=0, sharding={replicated}
  constant.116 = s32[] constant(3)
  ROOT compare.117 = pred[] compare(get-tuple-element.109, constant.116), direction=LT
}

region_2.126 {
  Arg_0.127 = f32[] parameter(0)
  Arg_1.128 = f32[] parameter(1)
  ROOT add.129 = f32[] add(Arg_0.127, Arg_1.128)
}

wide.wide.region_3.156.clone.clone {
  wide_param.7 = (s32[], f32[4,32]{1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,128]{2,1,0}, /*index=5*/f32[3,4,32]{2,1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,32]{2,1,0}) parameter(0), sharding={{replicated}, {devices=[1,2]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}}
  get-tuple-element.185 = s32[] get-tuple-element(wide_param.7), index=0, sharding={replicated}
  constant.34 = s32[] constant(1), sharding={replicated}
  add.14 = s32[] add(get-tuple-element.185, constant.34), sharding={replicated}
  get-tuple-element.186 = f32[4,32]{1,0} get-tuple-element(wide_param.7), index=1, sharding={devices=[2,1]0,1}
  get-tuple-element.190 = f32[3,4,32]{2,1,0} get-tuple-element(wide_param.7), index=5, sharding={devices=[1,2,1]0,1}
  constant.35 = s32[] constant(3), sharding={replicated}
  subtract.3 = s32[] subtract(constant.35, get-tuple-element.185), sharding={replicated}
  constant.6..sunk.4 = s32[] constant(-1), sharding={replicated}
  add.15 = s32[] add(subtract.3, constant.6..sunk.4), sharding={replicated}
  constant.36 = s32[] constant(0), sharding={replicated}
  compare.7 = pred[] compare(add.15, constant.36), direction=LT, sharding={replicated}
  constant.26..sunk.1 = s32[] constant(2), sharding={replicated}
  add.16 = s32[] add(subtract.3, constant.26..sunk.1), sharding={replicated}
  select.4 = s32[] select(compare.7, add.16, add.15), sharding={replicated}
  dynamic-slice.15 = f32[1,4,32]{2,1,0} dynamic-slice(get-tuple-element.190, select.4, constant.36, constant.36), dynamic_slice_sizes={1,4,32}, sharding={devices=[1,2,1]0,1}
  reshape.21 = f32[4,32]{1,0} reshape(dynamic-slice.15), sharding={devices=[2,1]0,1}
  multiply.3 = f32[4,32]{1,0} multiply(get-tuple-element.186, reshape.21), sharding={devices=[2,1]0,1}
  get-tuple-element.192 = f32[3,128,32]{2,1,0} get-tuple-element(wide_param.7), index=7, sharding={devices=[1,1,2]0,1}
  dynamic-slice.16 = f32[1,128,32]{2,1,0} dynamic-slice(get-tuple-element.192, select.4, constant.36, constant.36), dynamic_slice_sizes={1,128,32}, sharding={devices=[1,1,2]0,1}
  reshape.22 = f32[128,32]{1,0} reshape(dynamic-slice.16), sharding={devices=[1,2]0,1}
  dot.20 = f32[4,128]{1,0} dot(multiply.3, reshape.22), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,1]0,1}
  get-tuple-element.191 = f32[3,32,128]{2,1,0} get-tuple-element(wide_param.7), index=6, sharding={devices=[1,2,1]0,1}
  dynamic-slice.17 = f32[1,32,128]{2,1,0} dynamic-slice(get-tuple-element.191, select.4, constant.36, constant.36), dynamic_slice_sizes={1,32,128}, sharding={devices=[1,2,1]0,1}
  reshape.23 = f32[32,128]{1,0} reshape(dynamic-slice.17), sharding={devices=[2,1]0,1}
  dot.21 = f32[4,32]{1,0} dot(dot.20, reshape.23), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[1,2]0,1}
  get-tuple-element.187 = f32[3,32,128]{2,1,0} get-tuple-element(wide_param.7), index=2, sharding={devices=[1,2,1]0,1}
  get-tuple-element.193 = f32[3,4,32]{2,1,0} get-tuple-element(wide_param.7), index=8, sharding={devices=[1,2,1]0,1}
  dynamic-slice.18 = f32[1,4,32]{2,1,0} dynamic-slice(get-tuple-element.193, select.4, constant.36, constant.36), dynamic_slice_sizes={1,4,32}, sharding={devices=[1,2,1]0,1}
  reshape.24 = f32[4,32]{1,0} reshape(dynamic-slice.18), sharding={devices=[2,1]0,1}
  dot.22 = f32[32,128]{0,1} dot(reshape.24, dot.20), lhs_contracting_dims={0}, rhs_contracting_dims={0}, sharding={devices=[2,1]0,1}
  reshape.25 = f32[1,32,128]{2,1,0} reshape(dot.22), sharding={devices=[1,2,1]0,1}
  dynamic-update-slice.6 = f32[3,32,128]{2,1,0} dynamic-update-slice(get-tuple-element.187, reshape.25, select.4, constant.36, constant.36), sharding={devices=[1,2,1]0,1}
  get-tuple-element.188 = f32[3,128,32]{2,1,0} get-tuple-element(wide_param.7), index=3, sharding={devices=[1,1,2]0,1}
  get-tuple-element.189 = f32[3,4,128]{2,1,0} get-tuple-element(wide_param.7), index=4, sharding={devices=[1,2,1]0,1}
  dynamic-slice.19 = f32[1,4,128]{2,1,0} dynamic-slice(get-tuple-element.189, select.4, constant.36, constant.36), dynamic_slice_sizes={1,4,128}, sharding={devices=[1,2,1]0,1}
  reshape.26 = f32[4,128]{1,0} reshape(dynamic-slice.19), sharding={devices=[2,1]0,1}
  dot.23 = f32[128,32]{0,1} dot(reshape.26, multiply.3), lhs_contracting_dims={0}, rhs_contracting_dims={0}, sharding={devices=[1,2]0,1}
  reshape.27 = f32[1,128,32]{2,1,0} reshape(dot.23), sharding={devices=[1,1,2]0,1}
  dynamic-update-slice.7 = f32[3,128,32]{2,1,0} dynamic-update-slice(get-tuple-element.188, reshape.27, select.4, constant.36, constant.36), sharding={devices=[1,1,2]0,1}
  ROOT tuple.19 = (s32[], f32[4,32]{1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,128]{2,1,0}, /*index=5*/f32[3,4,32]{2,1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,32]{2,1,0}) tuple(add.14, dot.21, dynamic-update-slice.6, dynamic-update-slice.7, get-tuple-element.189, /*index=5*/get-tuple-element.190, get-tuple-element.191, get-tuple-element.192, get-tuple-element.193), sharding={{replicated}, {devices=[1,2]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}}
}

wide.wide.region_4.218.clone.clone {
  wide_param.6 = (s32[], f32[4,32]{1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,128]{2,1,0}, /*index=5*/f32[3,4,32]{2,1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,32]{2,1,0}) parameter(0), sharding={{replicated}, {devices=[1,2]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}}
  get-tuple-element.184 = s32[] get-tuple-element(wide_param.6), index=0, sharding={replicated}
  constant.28 = s32[] constant(3)
  ROOT compare.6 = pred[] compare(get-tuple-element.184, constant.28), direction=LT
}

ENTRY entry {
  Arg_1.2 = f32[3,32,128]{2,1,0} parameter(0), sharding={devices=[1,2,1]0,1}
  constant.45 = s32[] constant(0), sharding={replicated}
  constant.23 = f32[] constant(1), sharding={replicated}
  broadcast.24 = f32[4,32]{1,0} broadcast(constant.23), dimensions={}, sharding={devices=[1,2]0,1}
  constant.21 = f32[] constant(0), sharding={replicated}
  broadcast.22 = f32[3,32,128]{2,1,0} broadcast(constant.21), dimensions={}, sharding={devices=[1,2,1]0,1}
  broadcast.20 = f32[3,128,32]{2,1,0} broadcast(constant.21), dimensions={}, sharding={devices=[1,1,2]0,1}
  Arg_8.9 = f32[4,32]{1,0} parameter(2), sharding={devices=[2,1]0,1}
  copy = f32[4,32]{1,0} copy(Arg_8.9), sharding={devices=[2,1]0,1}
  broadcast.28 = f32[3,4,128]{2,1,0} broadcast(constant.21), dimensions={}, sharding={devices=[1,2,1]0,1}
  broadcast.26 = f32[3,4,32]{2,1,0} broadcast(constant.21), dimensions={}, sharding={devices=[1,2,1]0,1}
  Arg_2.3 = f32[3,128,32]{2,1,0} parameter(1), sharding={devices=[1,1,2]0,1}
  tuple.42 = (s32[], f32[4,32]{1,0}, f32[3,4,128]{2,1,0}, f32[3,4,32]{2,1,0}, f32[3,4,32]{2,1,0}, /*index=5*/f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}) tuple(constant.45, copy, broadcast.28, broadcast.26, broadcast.26, /*index=5*/Arg_1.2, Arg_2.3), sharding={{replicated}, {devices=[2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}}
  while.118 = (s32[], f32[4,32]{1,0}, f32[3,4,128]{2,1,0}, f32[3,4,32]{2,1,0}, f32[3,4,32]{2,1,0}, /*index=5*/f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}) while(tuple.42), condition=region_1.107, body=region_0.52, sharding={{replicated}, {devices=[2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}}
  get-tuple-element.179 = f32[3,4,128]{2,1,0} get-tuple-element(while.118), index=2, sharding={devices=[1,2,1]0,1}
  get-tuple-element.180 = f32[3,4,32]{2,1,0} get-tuple-element(while.118), index=3, sharding={devices=[1,2,1]0,1}
  get-tuple-element.183 = f32[3,4,32]{2,1,0} get-tuple-element(while.118), index=4, sharding={devices=[1,2,1]0,1}
  tuple.18 = (s32[], f32[4,32]{1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,128]{2,1,0}, /*index=5*/f32[3,4,32]{2,1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,32]{2,1,0}) tuple(constant.45, broadcast.24, broadcast.22, broadcast.20, get-tuple-element.179, /*index=5*/get-tuple-element.180, Arg_1.2, Arg_2.3, get-tuple-element.183), sharding={{replicated}, {devices=[1,2]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}}
  while.3 = (s32[], f32[4,32]{1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,128]{2,1,0}, /*index=5*/f32[3,4,32]{2,1,0}, f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[3,4,32]{2,1,0}) while(tuple.18), condition=wide.wide.region_4.218.clone.clone, body=wide.wide.region_3.156.clone.clone, sharding={{replicated}, {devices=[1,2]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}, /*index=5*/{devices=[1,2,1]0,1}, {devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {devices=[1,2,1]0,1}}
  get-tuple-element.234 = f32[3,32,128]{2,1,0} get-tuple-element(while.3), index=2, sharding={devices=[1,2,1]0,1}
  constant.16 = f32[] constant(-0.01), sharding={replicated}
  broadcast.17 = f32[3,32,128]{2,1,0} broadcast(constant.16), dimensions={}, sharding={devices=[1,2,1]0,1}
  multiply.243 = f32[3,32,128]{2,1,0} multiply(get-tuple-element.234, broadcast.17), sharding={devices=[1,2,1]0,1}
  add.255 = f32[3,32,128]{2,1,0} add(Arg_1.2, multiply.243), sharding={devices=[1,2,1]0,1}
  get-tuple-element.235 = f32[3,128,32]{2,1,0} get-tuple-element(while.3), index=3, sharding={devices=[1,1,2]0,1}
  broadcast.15 = f32[3,128,32]{2,1,0} broadcast(constant.16), dimensions={}, sharding={devices=[1,1,2]0,1}
  multiply.244 = f32[3,128,32]{2,1,0} multiply(get-tuple-element.235, broadcast.15), sharding={devices=[1,1,2]0,1}
  add.256 = f32[3,128,32]{2,1,0} add(Arg_2.3, multiply.244), sharding={devices=[1,1,2]0,1}
  get-tuple-element.120 = f32[4,32]{1,0} get-tuple-element(while.118), index=1, sharding={devices=[2,1]0,1}
  reduce.130 = f32[] reduce(get-tuple-element.120, constant.21), dimensions={0,1}, to_apply=region_2.126, sharding={replicated}
  ROOT tuple.271 = (f32[3,32,128]{2,1,0}, f32[3,128,32]{2,1,0}, f32[]) tuple(add.255, add.256, reduce.130), sharding={{devices=[1,2,1]0,1}, {devices=[1,1,2]0,1}, {replicated}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, /*replica_count=*/1,
                                                /*num_partitions=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(*module));
  EXPECT_TRUE(IsWeight(*hlo_value_semantics_analysis, module.get(),
                       "get-tuple-element.55"));
  EXPECT_TRUE(
      IsWeight(*hlo_value_semantics_analysis, module.get(), "reshape.74"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.0"));
  EXPECT_TRUE(
      IsWeight(*hlo_value_semantics_analysis, module.get(), "reshape.79"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.1"));
  EXPECT_TRUE(
      IsWeight(*hlo_value_semantics_analysis, module.get(), "reshape.22"));
  EXPECT_TRUE(
      IsStatic(*hlo_value_semantics_analysis, module.get(), "reshape.95"));
  EXPECT_TRUE(IsStatic(*hlo_value_semantics_analysis, module.get(),
                       "dynamic-update-slice.99"));
  EXPECT_TRUE(IsStatic(*hlo_value_semantics_analysis, module.get(),
                       "get-tuple-element.180"));
  EXPECT_TRUE(IsStatic(*hlo_value_semantics_analysis, module.get(),
                       "get-tuple-element.190"));
  EXPECT_TRUE(
      IsStatic(*hlo_value_semantics_analysis, module.get(), "reshape.21"));
  EXPECT_TRUE(
      IsStatic(*hlo_value_semantics_analysis, module.get(), "multiply.3"));
  EXPECT_TRUE(IsWeight(*hlo_value_semantics_analysis, module.get(), "dot.20"));
  EXPECT_TRUE(
      IsWeight(*hlo_value_semantics_analysis, module.get(), "reshape.23"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.21"));
  EXPECT_TRUE(
      IsWeight(*hlo_value_semantics_analysis, module.get(), "reshape.24"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.22"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "reshape.26"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.23"));
}

TEST_F(HloValueSemanticsAnalysisTest, ConvWithClamp) {
  const std::string module_str = R"(
HloModule ConvWithClamp

ENTRY entry {
  constant.123 = bf16[]{:T(256)} constant(127)
  constant.127 = bf16[]{:T(256)} constant(-128)
  arg_0 = bf16[128,14,14,1024]{3,0,2,1:T(8,128)(2,1)} parameter(0)
  broadcast.819 = bf16[1,1,1024,512]{3,2,1,0:T(8,128)(2,1)} broadcast(constant.127), dimensions={}
  arg_1 = bf16[1,1,1024,512]{3,2,1,0:T(8,128)(2,1)} parameter(1)
  broadcast.818 = bf16[1,1,1024,512]{3,2,1,0:T(8,128)(2,1)} broadcast(constant.123), dimensions={}
  clamp.42 = bf16[1,1,1024,512]{3,2,1,0:T(8,128)(2,1)} clamp(broadcast.819, arg_1, broadcast.818)
  round-nearest-even.42 = bf16[1,1,1024,512]{3,2,1,0:T(8,128)(2,1)} round-nearest-even(clamp.42)
  convert.219 = s8[1,1,1024,512]{3,2,1,0:T(8,128)(4,1)} convert(round-nearest-even.42)
  ROOT convolution.43 = bf16[128,14,14,512]{3,0,2,1:T(8,128)(2,1)} convolution(arg_0, convert.219), window={size=1x1}, dim_labels=b01f_01io->b01f
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str,
                                                       /*replica_count=*/1,
                                                       /*num_partitions=*/1));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(*module));
  EXPECT_TRUE(
      IsWeight(*hlo_value_semantics_analysis, module.get(), "convert.219"));
}

TEST_F(HloValueSemanticsAnalysisTest, MnistTrainingLoop) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kMnistHlo,
                                                       /*replica_count=*/1,
                                                       /*num_partitions=*/1));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(*module));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.63"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.67"));
  EXPECT_TRUE(
      IsActivation(*hlo_value_semantics_analysis, module.get(), "dot.71"));
  EXPECT_TRUE(
      IsWeightGradient(*hlo_value_semantics_analysis, module.get(), "dot.85"));
  EXPECT_TRUE(IsActivationGradient(*hlo_value_semantics_analysis, module.get(),
                                   "dot.89"));
  EXPECT_TRUE(
      IsWeightGradient(*hlo_value_semantics_analysis, module.get(), "dot.92"));
  EXPECT_TRUE(IsActivationGradient(*hlo_value_semantics_analysis, module.get(),
                                   "dot.96"));
  EXPECT_TRUE(
      IsWeightGradient(*hlo_value_semantics_analysis, module.get(), "dot.99"));
}

class EinsumDepthAnalysisTest : public HloTestBase {
 public:
  int GetInstructionDepth(const EinsumDepthMap& depth_map,
                          HloComputation* computation, absl::string_view name) {
    HloInstruction* instruction = computation->GetInstructionWithName(name);
    auto depth_iter = depth_map.find(instruction);
    EXPECT_NE(depth_iter, depth_map.end());
    return depth_iter->second.element({});
  }
};

TEST_F(EinsumDepthAnalysisTest, MnistTrainingLoop) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kMnistHlo,
                                                       /*replica_count=*/1,
                                                       /*num_partitions=*/1));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<EinsumDepthAnalysis> einsum_depth_analysis,
      EinsumDepthAnalysis::Run(*module->entry_computation(),
                               SendRecvGroupMap(*module)));
  const EinsumDepthMap& einsum_depth_map =
      einsum_depth_analysis->GetEinsumDepthMap();
  HloComputation* computation = module->GetComputationWithName("body.49");
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.63"), 5);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.67"), 4);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.71"), 3);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.89"), 2);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.96"), 1);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.92"), 0);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.99"), 0);
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "dot.85"), 0);
}

TEST_F(EinsumDepthAnalysisTest, HandleConditional) {
  const char* const hlo_string = R"(
    HloModule Module

    branch0 {
      tparam = f32[4] parameter(0)
      ROOT tgte1 = f32[4] ceil(tparam)
    }

    branch1 {
      fparam = f32[4] parameter(0)
      %async-start = ((f32[4]), f32[4], s32[]) abs-start(f32[4] fparam), async_execution_thread="parallel_thread"
      ROOT %async-done = f32[4] abs-done(((f32[4]), f32[4], s32[]) %async-start)
    }

    branch2 {
      sparam = f32[4] parameter(0)
      ROOT sgte1 = f32[4] ceil(sparam)
    }

    ENTRY entry {
      p0 = f32[4] parameter(0)
      b0 = s32[] parameter(1)
      ROOT conditional = f32[4] conditional(b0, p0, p0, p0),
        branch_computations={branch0, branch1, branch2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<EinsumDepthAnalysis> einsum_depth_analysis,
      EinsumDepthAnalysis::Run(*module->entry_computation(),
                               SendRecvGroupMap(*module)));
  const EinsumDepthMap& einsum_depth_map =
      einsum_depth_analysis->GetEinsumDepthMap();
  HloComputation* computation = module->GetComputationWithName("entry");
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "conditional"),
            0);
}

TEST_F(EinsumDepthAnalysisTest, HandleAfterAll) {
  const char* const hlo_string = R"(
    ENTRY entry {
      after-all.1 = token[] after-all()
      parameter.1 = f32[] parameter(0)
      send.1 = (f32[], u32[], token[]) send(parameter.1, after-all.1), channel_id=1, is_host_transfer=true, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="rendezvous1"}
      send-done.1 = token[] send-done(send.1), channel_id=1, is_host_transfer=true, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="rendezvous1"}
      ROOT after-all.2 = token[] after-all(send-done.1), frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="rendezvous1"}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<EinsumDepthAnalysis> einsum_depth_analysis,
      EinsumDepthAnalysis::Run(*module->entry_computation(),
                               SendRecvGroupMap(*module)));
  const EinsumDepthMap& einsum_depth_map =
      einsum_depth_analysis->GetEinsumDepthMap();
  HloComputation* computation = module->GetComputationWithName("entry");
  EXPECT_EQ(GetInstructionDepth(einsum_depth_map, computation, "after-all.2"),
            0);
}

class EinsumHeightAnalysisTest : public HloTestBase {
 public:
  int GetInstructionHeight(const EinsumHeightMap& height_map,
                           HloComputation* computation,
                           absl::string_view name) {
    HloInstruction* instruction = computation->GetInstructionWithName(name);
    auto height_iter = height_map.find(instruction);
    EXPECT_NE(height_iter, height_map.end());
    return height_iter->second.element({});
  }
};

TEST_F(EinsumHeightAnalysisTest, MnistTrainingLoop) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kMnistHlo,
                                                       /*replica_count=*/1,
                                                       /*num_partitions=*/1));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<EinsumHeightAnalysis> einsum_height_analysis,
      EinsumHeightAnalysis::Run(*module->entry_computation(),
                                SendRecvGroupMap(*module)));
  const EinsumHeightMap& einsum_height_map =
      einsum_height_analysis->GetEinsumHeightMap();
  HloComputation* computation = module->GetComputationWithName("body.49");
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.63"), 1);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.67"), 2);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.71"), 3);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.89"), 4);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.96"), 5);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.92"), 5);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.99"), 6);
  EXPECT_EQ(GetInstructionHeight(einsum_height_map, computation, "dot.85"), 4);
}

TEST_F(HloValueSemanticsAnalysisTest,
       HandleIncompleteForeignThreadComputation) {
  constexpr std::string_view hlo = R"(
HloModule Module

ENTRY entry {
  foreign-call-start = ((), s32[], s32[]) custom-call-start(), custom_call_target="ThreadSpecificCustomCall", async_execution_thread="foreign_thread"
  ROOT foreign-call-done = s32[] custom-call-done(foreign-call-start)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloValueSemanticsAnalysis> hlo_value_semantics_analysis,
      HloValueSemanticsAnalysis::Run(
          *module,
          /*execution_threads=*/{HloInstruction::kMainExecutionThread}));
}

}  // namespace
}  // namespace xla
