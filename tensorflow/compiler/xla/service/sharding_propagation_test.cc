/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/sharding_propagation.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ShardingPropagationTest = HloTestBase;

TEST_F(ShardingPropagationTest, ElementwiseOperationForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%param0, %param1)
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "add"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, ElementwiseOperationBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%param0, %param1)
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%add),
    sharding={devices=[1,2,2,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "add"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, BroadcastForwardPassNoSharding) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[7,11]{1,0} parameter(0),
    sharding={devices=[2,2]0,1,2,3}
  %broadcast = f32[5,7,11,13]{3,2,1,0} broadcast(%param0), dimensions={1,2}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%broadcast)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_FALSE(changed);
}

// Regression Test for b/129569657.
TEST_F(ShardingPropagationTest, BroadcastForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[3,2048,2048]{2,1,0} parameter(0),
    sharding={devices=[1,2,2]0,1,2,3}
  %broadcast = f32[3,2048,2048,3]{3,2,1,0} broadcast(%param0), dimensions={0,1,2}
  ROOT %copy = f32[3,2048,2048,3]{3,2,1,0} copy(%broadcast)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "broadcast"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, BroadcastBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[13]{0} parameter(0)
  %broadcast = f32[5,7,11,13]{3,2,1,0} broadcast(%param0), dimensions={3}
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%broadcast),
    sharding={devices=[1,2,2,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "broadcast"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, BroadcastUser) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %broadcast {
  %param0 = f32[24,8]{0,1} parameter(0)
  %copy = f32[24,8]{0,1} copy(%param0)
  ROOT %broadcast = f32[4,24,6,8]{3,2,1,0} broadcast(%copy), dimensions={1,3},
    sharding={devices=[1,2,1,4]0,1,2,3,4,5,6,7}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "copy"),
              op::Sharding("{devices=[2,4]0,1,2,3,4,5,6,7}"));
}

TEST_F(ShardingPropagationTest, MaximalReduceForwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3}
  %init = f32[] parameter(1)
  %reduce = f32[5,7]{1,0} reduce(%param0, %init), dimensions={2,3}, to_apply=%add
  ROOT %copy = f32[5,7]{0,1} copy(%reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "reduce"),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, ShardedReduceForwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[1,2,2,1]0,1,2,3}
  %init = f32[] parameter(1)
  %reduce = f32[7,11]{1,0} reduce(%param0, %init), dimensions={0,3}, to_apply=%add
  ROOT %copy = f32[7,11]{0,1} copy(f32[7,11]{1,0} %reduce)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "reduce"),
              op::Sharding("{devices=[2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, ShardedTupleReduceForwardAndBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

%minmax_func {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.2 = pred[] compare(%lhs_value, %rhs_value), direction=GT
  %select.4 = f32[] select(%compare.2, %lhs_value, %rhs_value)
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %select.5 = s32[] select(%compare.2, %lhs_index, %rhs_index)
  ROOT %tuple.2 = (f32[], s32[]) tuple(%select.4, %select.5)
}

ENTRY %main {
  %param0 = f32[28,10] parameter(0)
  %param1 = s32[28,10] parameter(1), sharding={devices=[2,1]0,1}
  %copy_param0 = f32[28,10] copy(%param0)
  %init0 = f32[] parameter(2)
  %init1 = s32[] parameter(3)
  %reduce = (f32[28], s32[28]) reduce(%copy_param0, %param1, %init0, %init1),
    dimensions={1}, to_apply=%minmax_func
  %gte0 = f32[28] get-tuple-element(%reduce), index=0
  %gte1 = s32[28] get-tuple-element(%reduce), index=1
  %copy0 = f32[28] copy(%gte0)
  %copy1 = s32[28] copy(%gte1)
  ROOT %tuple = (f32[28], s32[28]) tuple(%copy0, %copy1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, ShardingPropagation(/*is_spmd=*/true).Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "reduce"),
              op::Sharding("{{devices=[2]0,1},{devices=[2]0,1}}"));
  EXPECT_THAT(FindInstruction(module.get(), "copy_param0"),
              op::Sharding("{devices=[2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, GetTupleElementForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %gte {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param0, %param0)
  %tuple.1 = (f32[5,7,11,13]{3,2,1,0},
              (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0})) tuple(
    %param0, %tuple),
    sharding={{devices=[1,2,2,1]0,1,2,3},
              {replicated},
              {devices=[1,2,2,1]0,1,2,3}}
  %gte = f32[5,7,11,13]{3,2,1,0} get-tuple-element(%tuple.1), index=0
  %gte.1 = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) get-tuple-element(
    %tuple.1), index=1
  %gte.2 = f32[5,7,11,13]{3,2,1,0} get-tuple-element(%gte.1), index=0
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%gte.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "gte"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  EXPECT_THAT(FindInstruction(module.get(), "gte.1"),
              op::Sharding("{{replicated},"
                           " {devices=[1,2,2,1]0,1,2,3}}"));
  EXPECT_THAT(FindInstruction(module.get(), "gte.2"),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, TupleForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tuple {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={replicated}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={devices=[1,2,2,1]0,1,2,3}
  %param2 = f32[5,7,11,13]{3,2,1,0} parameter(2)
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param1, %param2)
  %tuple.1 = (f32[5,7,11,13]{3,2,1,0},
              (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0})) tuple(
    %param0, %tuple)
  ROOT %copy = (f32[5,7,11,13]{3,2,1,0},
                (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0})) copy(
    %tuple.1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "tuple"),
              op::Sharding("{{devices=[1,2,2,1]0,1,2,3},"
                           " {replicated}}"));
  EXPECT_THAT(FindInstruction(module.get(), "tuple.1"),
              op::Sharding("{{replicated},"
                           " {devices=[1,2,2,1]0,1,2,3},"
                           " {replicated}}"));
}

TEST_F(ShardingPropagationTest, ForwardConvolutionForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={devices=[2,2,2,1]0,1,2,3,4,5,6,7}
  %rhs = f32[3,3,13,17]{3,2,1,0} parameter(1)
  %convolution = f32[5,7,11,17]{3,2,1,0} convolution(%lhs, %rhs),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  ROOT %copy = f32[5,7,11,17]{3,2,1,0} copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "convolution"),
              op::Sharding("{devices=[2,2,2,1]0,1,2,3,4,5,6,7}"));
}

TEST_F(ShardingPropagationTest, ForwardConvolutionLargeDilationForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %lhs = f32[8,64,2]{2,1,0} parameter(0),
    sharding={devices=[1,4,1]0,1,2,3}
  %rhs = f32[3,2,2]{2,1,0} parameter(1)
  %convolution = f32[8,32,2]{2,1,0} convolution(%lhs, %rhs),
    window={size=3 rhs_dilate=16}, dim_labels=b0f_0io->b0f
  ROOT %copy = f32[8,32,2]{2,1,0} copy(%convolution)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "convolution"),
              op::Sharding("{devices=[1,4,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, TransposeForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0),
    sharding={devices=[2,1,2]0,1,2,3}
  %transpose = f32[11,13,7]{2,1,0} transpose(%param), dimensions={1,2,0}
  ROOT %copy = f32[11,13,7]{2,1,0} copy(%transpose)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "transpose"),
              op::Sharding("{devices=[1,2,2]0,2,1,3}"));
}

TEST_F(ShardingPropagationTest, TransposeBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %transpose {
  %param = f32[7,11,13]{2,1,0} parameter(0)
  %copy = f32[7,11,13]{2,1,0} copy(%param)
  ROOT %transpose = f32[11,13,7]{2,1,0} transpose(%copy), dimensions={1,2,0},
    sharding={devices=[1,2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "copy"),
              op::Sharding("{devices=[2,1,2]0,2,1,3}"));
}

TEST_F(ShardingPropagationTest, ReshapeForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[1430,1]{1,0} parameter(0),
    sharding={devices=[2,1]0,1}
  %reshape = f32[10,11,13]{2,1,0} reshape(%param0)
  ROOT %copy = f32[10,11,13]{2,1,0} copy(%reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "reshape"),
              op::Sharding("{devices=[2,1,1]0,1}"));
}

TEST_F(ShardingPropagationTest, ReshapeBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %reshape {
  %param0 = f32[2002,1]{1,0} parameter(0)
  %copy = f32[2002,1]{1,0} copy(f32[2002,1]{1,0} %param0)
  ROOT %reshape = f32[14,11,13]{2,1,0} reshape(%copy),
    sharding={devices=[2,1,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "copy"),
              op::Sharding("{devices=[2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, PadForwardPass) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %pad {
  %input = f32[11,17]{1,0} parameter(0),
    sharding={devices=[2,2]0,1,2,3}
  %pad_value = f32[] parameter(1)
  %pad = f32[27,51]{1,0} pad(%input, %pad_value), padding=2_4_1x1_1_2
  ROOT %copy = f32[27,51]{1,0} copy(%pad)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "pad"),
              op::Sharding("{devices=[2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, ShardedPreferredOverReplicated) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %replicated {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={replicated}
  %copy = f32[5,7,11,13]{3,2,1,0} copy(%param0)
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={devices=[1,2,2,1]0,1,2,3}
  %copy.1 = f32[5,7,11,13]{3,2,1,0} copy(%param1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%copy, %copy.1)
  ROOT %copy.2 = f32[5,7,11,13]{3,2,1,0} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "copy"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  EXPECT_THAT(FindInstruction(module.get(), "copy.1"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
  EXPECT_THAT(FindInstruction(module.get(), "add"),
              op::Sharding("{devices=[1,2,2,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, DontShardTuplesIfAllInputIsMaximal) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tuple {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0),
    sharding={maximal device=0}
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1),
    sharding={maximal device=1}
  %tuple = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) tuple(
    %param0, %param1)
  ROOT %copy = (f32[5,7,11,13]{3,2,1,0}, f32[5,7,11,13]{3,2,1,0}) copy(%tuple)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "tuple"), op::NoSharding());
}

TEST_F(ShardingPropagationTest, ValidConvolution) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[13,17,19]{2,1,0} parameter(0),
    sharding={devices=[1,2,1]0,1}
  %rhs = f32[19,5,19]{2,1,0} parameter(1)
  %conv = f32[13,13,19]{2,1,0} convolution(%lhs, %rhs),
    window={size=5}, dim_labels=b0f_i0o->b0f
  ROOT %tuple = (f32[13,13,19]{2,1,0}) tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[1,2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, StridedSlice) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %slice {
  %param = f32[17,13]{1,0} parameter(0),
    sharding={devices=[2,1]0,1}
  %slice = f32[7,5]{1,0} slice(%param), slice={[1:15:2], [5:10:1]}
  ROOT %tuple = (f32[7,5]{1,0}) tuple(%slice)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "slice"),
              op::Sharding("{devices=[2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, ReduceWindowBackwardPass) {
  const char* const hlo_string = R"(
HloModule module
%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(%lhs, %rhs)
}
ENTRY %reduce_window {
  %param = f32[13,17]{1,0} parameter(0)
  %param.copy = f32[13,17]{1,0} copy(%param)
  %init = f32[] parameter(1)
  ROOT %reduce-window = f32[7,17]{1,0} reduce-window(%param.copy, %init),
    window={size=3x2 stride=2x1 pad=1_1x0_1}, to_apply=%add,
    sharding={devices=[2,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "param.copy"),
              op::Sharding("{devices=[2,1]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "reduce-window"),
              op::Sharding("{devices=[2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, ReplicatedConvolutionLhs) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[3,2,3]{2,1,0} parameter(0), sharding={replicated}
  %rhs = f32[2,2,1]{2,1,0} parameter(1)
  %conv = f32[3,2,3]{2,1,0} convolution(%lhs, %rhs),
    window={size=1}, dim_labels=bf0_oi0->bf0
  ROOT %tuple = f32[3,2,3]{2,1,0} tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "lhs"),
              op::Sharding("{replicated}"));
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, ConvolutionShardedFeature) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[3,2,3]{2,1,0} parameter(0),
    sharding={devices=[1,2,1]0,1}
  %rhs = f32[2,2,1]{2,1,0} parameter(1)
  %conv = f32[3,2,3]{2,1,0} convolution(%lhs, %rhs),
    window={size=1}, dim_labels=bf0_oi0->bf0
  ROOT %tuple = f32[3,2,3]{2,1,0} tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ShardingPropagationTest, ConvolutionDifferentDimensionNumbers) {
  const char* const hlo_string = R"(
HloModule module

ENTRY conv {
  %lhs = f32[8,16,512] parameter(0),
    sharding={devices=[1,2,1]0,1}
  %rhs = f32[8,2,512] parameter(1)
  %conv = f32[3,512,512] convolution(%lhs, %rhs),
    window={size=2 stride=5},
    dim_labels=f0b_i0o->0bf
  ROOT %tuple = f32[3,512,512] tuple(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[2,1,1]0,1}"));
}

TEST_F(ShardingPropagationTest, Concatenate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %concat {
  %param.0 = f32[5,7] parameter(0),
    sharding={devices=[2,1]0,1}
  %param.1 = f32[5,9] parameter(1),
    sharding={devices=[2,1]0,1}
  %concat = f32[5,16] concatenate(%param.0, %param.1),
    dimensions={1}
  ROOT %tuple = (f32[5,16]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "concat"),
              op::Sharding("{devices=[2,1]0,1}"));
}

TEST_F(ShardingPropagationTest, TupleBackwardPass) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %tuple {
  %param.0 = f32[1] parameter(0)
  %param.1 = f32[3] parameter(1)
  %copy.0 = f32[1] copy(%param.0)
  %copy.1 = f32[3] copy(param.1)
  ROOT %tuple = (f32[1], f32[3]) tuple(%copy.0, %copy.1),
    sharding={{replicated}, {devices=[2]0,1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "copy.0"),
              op::Sharding("{replicated}"));
  EXPECT_THAT(FindInstruction(module.get(), "copy.1"),
              op::Sharding("{devices=[2]0,1}"));
}

TEST_F(ShardingPropagationTest, AllReduce) {
  const char* const hlo_string = R"(
HloModule module

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %add_lhs = f32[] parameter(0)
  %add_rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %add_lhs, f32[] %add_rhs)
}

ENTRY %entry {
  %param.0 = f32[3] parameter(0)
  %param.1 = f32[3] parameter(1)

  %copy_f_t = f32[3] copy(%param.1), sharding={devices=[2]0,1}
  %crs_f.tiled = f32[3] all-reduce(%copy_f_t), to_apply=%add
  %crs_f.none = f32[3] all-reduce(%copy_f_t), to_apply=%add,
    channel_id=1

  %crs_b.replicated = f32[3] all-reduce(%param.0), to_apply=%add
  %copy_b_r = f32[3] copy(%crs_b.replicated), sharding={replicated}

  ROOT %tuple = (f32[3], f32[3], f32[3], f32[3]) tuple(
    %crs_f.tiled, crs_f.none, %copy_b_r)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "crs_f.tiled"),
              op::Sharding("{devices=[2]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "crs_f.none"), op::NoSharding());

  EXPECT_THAT(FindInstruction(module.get(), "crs_b.replicated"),
              op::Sharding("{replicated}"));
}

TEST_F(ShardingPropagationTest, While) {
  const char* const hlo_string = R"(
HloModule module

%cond {
  %vars.cond = (u32[], f32[10]{0}) parameter(0)
  %count.cond = u32[] get-tuple-element((u32[], f32[10]{0}) %vars.cond), index=0
  %limit = u32[] constant(10)
  ROOT %lt = pred[] compare(u32[] %count.cond, u32[] %limit), direction=LT
}

%body {
  %vars = (u32[], f32[10]{0}) parameter(0)
  %count = u32[] get-tuple-element(%vars), index=0
  %acc = f32[10]{0} get-tuple-element((u32[], f32[10]{0}) %vars), index=1

  %one = u32[] constant(1)
  %count.1 = u32[] add(u32[] %count, u32[] %one), sharding={replicated}
  %acc.1 = f32[10]{0} add(f32[10]{0} %acc, f32[10]{0} %acc)
  ROOT %tuple = (u32[], f32[10]{0}) tuple(u32[] %count.1, f32[10]{0} %acc.1)
}

ENTRY %entry {
  %p0 = f32[10]{0} parameter(0)
  %p0.copy = f32[10]{0} copy(f32[10]{0} %p0)
  %p1 = f32[10]{0} parameter(1)
  %zero = u32[] constant(0)
  %init = (u32[], f32[10]{0}) tuple(u32[] %zero, f32[10]{0} %p0.copy)
  %while = (u32[], f32[10]{0}) while((u32[], f32[10]{0}) %init),
    body=%body, condition=%cond
  %res = f32[10]{0} get-tuple-element((u32[], f32[10]{0}) %while), index=1
  %prev = f32[10]{0} get-tuple-element((u32[], f32[10]{0}) %init), index=1
  %res.1 = f32[10]{0} multiply(f32[10]{0} %res, %prev)
  ROOT %res_tuple = (f32[10]{0}) tuple(f32[10]{0} %res.1)
})";

  auto while_is_sharded = [this](HloModule* module,
                                 const HloSharding& sharding) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed, ShardingPropagation().Run(module));
    EXPECT_TRUE(changed);
    auto while_instr = FindInstruction(module, "while");
    EXPECT_NE(nullptr, while_instr);
    std::vector<const HloInstruction*> instructions{
        while_instr, while_instr->while_body()->root_instruction(),
        while_instr->while_body()->parameter_instruction(0),
        while_instr->while_condition()->parameter_instruction(0)};

    for (auto instr : instructions) {
      EXPECT_TRUE(instr->has_sharding());
      EXPECT_EQ(sharding, instr->sharding());
    }
  };
  {
    // Propagation of user-defined partial sharding of while-related instruction
    // (body root in this test).
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    auto body_root = FindInstruction(module.get(), "tuple");
    EXPECT_NE(nullptr, body_root);
    auto sharding =
        ParseSharding("{{replicated}, {devices=[2]0,1}}").ConsumeValueOrDie();
    body_root->set_sharding(sharding);
    while_is_sharded(module.get(), sharding);
  }
  {
    // Propagation from acc.1 to the rest of the loop.
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string));
    auto acc_1 = FindInstruction(module.get(), "acc.1");
    EXPECT_NE(nullptr, acc_1);
    acc_1->set_sharding(ParseSharding("{devices=[2]0,1}").ConsumeValueOrDie());

    while_is_sharded(
        module.get(),
        ParseSharding("{{replicated}, {devices=[2]0,1}}").ConsumeValueOrDie());
  }
}

TEST_F(ShardingPropagationTest, Dot) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %param.0 = f32[8,256,128] parameter(0)
  %param.1 = f32[8,128,512] parameter(1)
  %param.2 = f32[8,128] parameter(2)

  %p0_copy_0 = f32[8,256,128] copy(%param.0),
    sharding={devices=[1,4,1]0,1,2,3}
  %p1_copy_0 = f32[8,128,512] copy(%param.1),
    sharding={devices=[1,2,2]0,1,2,3}
  %p2_copy = f32[8,128] copy(%param.2)
  %dot_prop_rhs = f32[8,256,512] dot(%p0_copy_0, %p1_copy_0),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  %dot_prop_lhs = f32[8,512,256] dot(%p1_copy_0, %p0_copy_0),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_contracting_dims={2}
  %dot_mat_vec = f32[8,256] dot(%p0_copy_0, %p2_copy),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}

  %p0_copy_1 = f32[8,256,128] copy(%param.0)
  %p1_copy_1 = f32[8,128,512] copy(%param.1)
  %dot_back_prop_rhs = f32[8,256,512] dot(%p0_copy_1, %p1_copy_1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  %copy_back_prop_rhs = f32[8,256,512] copy(%dot_back_prop_rhs),
    sharding={devices=[1,2,2]0,1,2,3}

  ROOT %tuple = (f32[8,256,256], f32[8,256,256], f32[8,256])
    tuple(%dot_prop_lhs, %dot_prop_rhs, %dot_mat_vec, %copy_back_prop_rhs)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "dot_prop_rhs"),
              op::Sharding("{devices=[1,2,2]0,1,2,3}"));
  EXPECT_THAT(FindInstruction(module.get(), "dot_prop_lhs"),
              op::Sharding("{devices=[1,2,2]0,1,2,3}"));
  EXPECT_THAT(FindInstruction(module.get(), "dot_mat_vec"),
              op::Sharding("{devices=[1,4]0,1,2,3}"));

  EXPECT_THAT(FindInstruction(module.get(), "p0_copy_1"),
              op::Sharding("{replicated}"));
  EXPECT_THAT(FindInstruction(module.get(), "p1_copy_1"),
              op::Sharding("{devices=[1,2,2]0,1,2,3}"));
  EXPECT_THAT(FindInstruction(module.get(), "dot_back_prop_rhs"),
              op::Sharding("{devices=[1,2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, DotTiledBatchDim) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,256,512] parameter(0)
  %p1 = f32[8,512,128] parameter(1)

  %add = f32[8,256,512] add(%p0, %p0)
  %dot = f32[8,256,128] dot(%add, %p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  %res = f32[8,32768] reshape(%dot), sharding={devices=[2,2]0,1,2,3}

  ROOT %tuple = (f32[8,32768]) tuple(%res)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "add"),
              op::Sharding("{devices=[2,2,1]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, ConcatFromUserUnshardedDim) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,128] parameter(0)
  %p1 = f32[8,128] parameter(1)
  %c0 = f32[8,128] copy(%p0)
  %c1 = f32[8,128] copy(%p1)

  %concat = f32[16,128] concatenate(%c0, %c1),
    dimensions={0},
    sharding={devices=[1,2]0,1}
  ROOT %tuple = (f32[16,128]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "c0"),
              op::Sharding("{devices=[1,2]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "c1"),
              op::Sharding("{devices=[1,2]0,1}"));
}

TEST_F(ShardingPropagationTest, ConcatFromUserShardedDim) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,128] parameter(0)
  %p1 = f32[8,128] parameter(1)
  %c0 = f32[8,128] copy(%p0)
  %c1 = f32[8,128] copy(%p1)

  %concat = f32[16,128] concatenate(%c0, %c1),
    dimensions={0},
    sharding={devices=[3,1]0,1,2}
  ROOT %tuple = (f32[16,128]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "c0"),
              op::Sharding("{devices=[2,1]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "c1"),
              op::Sharding("{devices=[2,1]1,2}"));
}

TEST_F(ShardingPropagationTest, ConcatFromUserShardedDimMaximalOperand) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %conv {
  %p0 = f32[8,128] parameter(0)
  %p1 = f32[24,128] parameter(1)
  %c0 = f32[8,128] copy(%p0)
  %c1 = f32[24,128] copy(%p1)

  %concat = f32[32,128] concatenate(%c0, %c1),
    dimensions={0},
    sharding={devices=[4,1]0,1,2,3}
  ROOT %tuple = (f32[32,128]) tuple(%concat)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "c0"), op::NoSharding());
  EXPECT_THAT(FindInstruction(module.get(), "c1"),
              op::Sharding("{devices=[3,1]1,2,3}"));
}

TEST_F(ShardingPropagationTest, ReplicatedToSideEffecting) {
  const char* const hlo_string = R"(
HloModule module
ENTRY entry_computation {
  %const.0 = s32[] constant(0), sharding={replicated}
  %const.1 = s32[] constant(2147483647), sharding={replicated}
  %rng = s32[4]{0} rng(%const.0, %const.1),
    distribution=rng_uniform
  ROOT %root = (s32[4]{0}) tuple(%rng)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "rng"), op::NoSharding());
}

TEST_F(ShardingPropagationTest, PartReplicatedTupleUser) {
  const char* const hlo_string = R"(
HloModule module
ENTRY entry_computation {
  %param.0 = f32[5] parameter(0)
  %param.1 = f32[7] parameter(1)
  %param.2 = f32[9] parameter(2)
  %tuple.0 = (f32[5], f32[7]) tuple(%param.0, %param.1)
  ROOT %tuple.1 = ((f32[5], f32[7]), f32[9]) tuple(%tuple.0, %param.2),
    sharding={{maximal device=0}, {replicated}, {maximal device=1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "tuple.0"),
              op::Sharding("{{maximal device=0}, {replicated}}"));
}

TEST_F(ShardingPropagationTest, Conditional) {
  const char* const hlo_string = R"(
HloModule module

%true_comp {
  %tp = (f32[3,5]) parameter(0)
  %tgte = f32[3,5] get-tuple-element(%tp), index=0
  %ttr = f32[5,3] transpose(%tgte), dimensions={1,0}
  ROOT %tr = (f32[5,3]) tuple(%ttr)
}

%false_comp {
  %fp = (f32[5,3]) parameter(0)
  %fgte = f32[5,3] get-tuple-element(%fp), index=0
  ROOT %fr = (f32[5,3]) tuple(%fgte)
}

ENTRY entry {
  %cond = pred[] parameter(0)
  %true_param = (f32[3,5]) parameter(1), sharding={{devices=[1,2]0,1}}
  %false_param = (f32[5,3]) parameter(2), sharding={{devices=[1,3]0,1,2}}
  %conditional = (f32[5,3]) conditional(
      %cond, %true_param, %false_param),
    true_computation=%true_comp,
    false_computation=%false_comp
  ROOT %root = f32[5,3] get-tuple-element(%conditional), index=0
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "tp"),
              op::Sharding("{{devices=[1,2]0,1}}"));
  EXPECT_THAT(FindInstruction(module.get(), "tgte"),
              op::Sharding("{devices=[1,2]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "ttr"),
              op::Sharding("{devices=[2,1]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "tr"),
              op::Sharding("{{devices=[2,1]0,1}}"));
  EXPECT_THAT(FindInstruction(module.get(), "fp"),
              op::Sharding("{{devices=[1,3]0,1,2}}"));
  EXPECT_THAT(FindInstruction(module.get(), "fgte"),
              op::Sharding("{devices=[1,3]0,1,2}"));
  EXPECT_THAT(FindInstruction(module.get(), "fr"),
              op::Sharding("{{devices=[2,1]0,1}}"));
  EXPECT_THAT(FindInstruction(module.get(), "conditional"),
              op::Sharding("{{devices=[2,1]0,1}}"));
}

TEST_F(ShardingPropagationTest, TupleFromUser) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[13] parameter(0)
  %p1 = f32[15] parameter(1)
  %p2 = f32[17] parameter(2)
  %t0 = (f32[13], f32[15]) tuple(%p0, %p1)
  %t1 = ((f32[13], f32[15]), f32[17]) tuple(%t0, %p2)
  %gte.0 = (f32[13], f32[15]) get-tuple-element(%t1), index=0
  %gte.1 = f32[13] get-tuple-element(%gte.0), index=0
  %gte.2 = f32[15] get-tuple-element(%gte.0), index=1
  %gte.3 = f32[17] get-tuple-element(%t1), index=1
  ROOT %t2 = (f32[13], f32[15], f32[17]) tuple(%gte.1, %gte.2, %gte.3),
    sharding={{replicated}, {devices=[2]0,1}, {devices=[3]1,2,3}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "t0"),
              op::Sharding("{{replicated}, {devices=[2]0,1}}"));
  EXPECT_THAT(
      FindInstruction(module.get(), "t1"),
      op::Sharding("{{replicated}, {devices=[2]0,1}, {devices=[3]1,2,3}}"));
}

TEST_F(ShardingPropagationTest, DynamicSliceForwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0), sharding={devices=[1,1,2]0,1}
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %ds = f32[11,1,15] dynamic-slice(%c0, %i0, %p1, %i0),
    dynamic_slice_sizes={11,1,15}
  ROOT %root = (f32[11,1,15]) tuple(%ds)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "ds"),
              op::Sharding("{devices=[1,1,2]0,1}"));
}

TEST_F(ShardingPropagationTest, DynamicSliceBackwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = s32[] parameter(1)
  %i0 = s32[] constant(0)
  %ds = f32[11,1,15] dynamic-slice(%c0, %i0, %p1, %i0),
    dynamic_slice_sizes={11,1,15},
    sharding={devices=[1,1,2]0,1}
  ROOT %root = (f32[11,1,15]) tuple(%ds)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "ds"),
              op::Sharding("{devices=[1,1,2]0,1}"));
}

TEST_F(ShardingPropagationTest, DynamicUpdateSliceForwardPassBase) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0), sharding={devices=[1,1,2]0,1}
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1)
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %dus = f32[11,13,15] dynamic-update-slice(%c0, %c1, %i0, %p2, %i0)
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "dus"),
              op::Sharding("{devices=[1,1,2]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "c1"),
              op::Sharding("{devices=[1,1,2]0,1}"));
}

TEST_F(ShardingPropagationTest, DynamicUpdateSliceForwardPassUpdate) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1), sharding={devices=[1,1,2]0,1}
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %dus = f32[11,13,15] dynamic-update-slice(%c0, %c1, %i0, %p2, %i0)
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "dus"),
              op::Sharding("{devices=[1,1,2]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "c0"),
              op::Sharding("{devices=[1,1,2]0,1}"));
}

TEST_F(ShardingPropagationTest, DynamicUpdateSliceBackwardPass) {
  const char* hlo_string = R"(
HloModule module
ENTRY %entry {
  %p0 = f32[11,13,15] parameter(0)
  %c0 = f32[11,13,15] copy(%p0)
  %p1 = f32[11,1,15] parameter(1)
  %c1 = f32[11,1,15] copy(%p1)
  %p2 = s32[] parameter(2)
  %i0 = s32[] constant(0)
  %dus = f32[11,13,15] dynamic-update-slice(%c0, %c1, %i0, %p2, %i0),
    sharding={devices=[1,1,2]0,1}
  ROOT %root = (f32[11,13,15]) tuple(%dus)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "c0"),
              op::Sharding("{devices=[1,1,2]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "c1"),
              op::Sharding("{devices=[1,1,2]0,1}"));
}

TEST_F(ShardingPropagationTest, EinsumLHSBatchPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs), sharding={devices=[2,1,1]0,1}
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs)
  %conv = f32[32,24,39296] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf_0oi->0bf, window={size=32 stride=31 lhs_dilate=32}
  ROOT %copy = f32[32,24,39296] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "rhs.copy"),
              op::Sharding("{devices=[2,1,1]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[2,1,1]0,1}"));
}

TEST_F(ShardingPropagationTest, EinsumOutputBatchPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64] parameter(0)
  %lhs.copy = f32[32,24,64] copy(%lhs)
  %rhs = f32[32,39296,64] parameter(1)
  %rhs.copy = f32[32,39296,64] copy(%rhs)
  %conv = f32[32,24,39296] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf_0oi->0bf, window={size=32 stride=31 lhs_dilate=32},
    sharding={devices=[2,1,1]0,1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "lhs.copy"),
              op::Sharding("{devices=[2,1,1]0,1}"));
  EXPECT_THAT(FindInstruction(module.get(), "rhs.copy"),
              op::Sharding("{devices=[2,1,1]0,1}"));
}

TEST_F(ShardingPropagationTest, EinsumLHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs), sharding={devices=[1,2,1,2]0,1,2,3}
  %rhs = f32[32,39296,64,1] parameter(1)
  %rhs.copy = f32[32,39296,64,1] copy(%rhs)
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1, window={size=32x1 stride=31x1 lhs_dilate=32x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[1,2,1,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, EinsumOutputLHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,128] parameter(0)
  %lhs.copy = f32[32,24,64,128] copy(%lhs)
  %rhs = f32[32,39296,64,1] parameter(1)
  %rhs.copy = f32[32,39296,64,1] copy(%rhs)
  ROOT %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1, window={size=32x1 stride=31x1 lhs_dilate=32x1},
    sharding={devices=[1,2,1,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "lhs.copy"),
              op::Sharding("{devices=[1,2,1,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, EinsumRHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs)
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={devices=[1,2,1,2]0,1,2,3}
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[1,1,2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, EinsumOutputRHSNonContractingPartitioned) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs)
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs)
  ROOT %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1},
    sharding={devices=[1,1,2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "rhs.copy"),
              op::Sharding("{devices=[1,2,1,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, EinsumChooseLargerOperand) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs), sharding={devices=[1,4,1,1]0,1,2,3}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={devices=[1,2,1,2]0,1,2,3}
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[1,1,2,2]0,1,2,3}"));
}

TEST_F(ShardingPropagationTest, EinsumChooseBatchFirst) {
  const char* hlo_string = R"(
HloModule module

ENTRY entry {
  %lhs = f32[32,24,64,1] parameter(0)
  %lhs.copy = f32[32,24,64,1] copy(%lhs), sharding={devices=[1,2,1,1]0,1}
  %rhs = f32[32,39296,64,128] parameter(1)
  %rhs.copy = f32[32,39296,64,128] copy(%rhs), sharding={devices=[2,1,1,1]0,1}
  %conv = f32[32,24,39296,128] convolution(%lhs.copy, %rhs.copy),
    dim_labels=0bf1_0oi1->0bf1,
    window={size=32x128 stride=31x1 pad=0_0x127_127 lhs_dilate=32x1 rhs_reversal=0x1}
  ROOT %copy = f32[32,24,39296,128] copy(%conv)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          ShardingPropagation().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(FindInstruction(module.get(), "conv"),
              op::Sharding("{devices=[2,1,1,1]0,1}"));
}

}  // namespace
}  // namespace xla
