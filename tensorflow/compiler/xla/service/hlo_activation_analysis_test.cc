/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_activation_analysis.h"

#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloActivationAnalysisTest : public HloTestBase {};

TEST_F(HloActivationAnalysisTest, OneMatmul) {
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
  auto act_set = ComputeHloActivationAnalysis(module.get());
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "copy")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "Arg_1.2")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.0")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "select.35")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.2")));
}

TEST_F(HloActivationAnalysisTest, TwoMatmuls) {
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
  auto act_set = ComputeHloActivationAnalysis(module.get());
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "copy")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "Arg_1.2")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.0")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "Arg_2.3")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.1")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "select.40")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.2")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.5")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.6")));
}

TEST_F(HloActivationAnalysisTest, RepeatWhile) {
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
  auto act_set = ComputeHloActivationAnalysis(module.get());
  EXPECT_FALSE(
      act_set.count(FindInstruction(module.get(), "get-tuple-element.55")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "reshape.74")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.0")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "reshape.79")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.1")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "reshape.22")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "reshape.95")));
  EXPECT_TRUE(
      act_set.count(FindInstruction(module.get(), "dynamic-update-slice.99")));
  EXPECT_TRUE(
      act_set.count(FindInstruction(module.get(), "get-tuple-element.180")));
  EXPECT_TRUE(
      act_set.count(FindInstruction(module.get(), "get-tuple-element.190")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "reshape.21")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "multiply.3")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.20")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "reshape.23")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.21")));
  EXPECT_FALSE(act_set.count(FindInstruction(module.get(), "reshape.24")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.22")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "reshape.26")));
  EXPECT_TRUE(act_set.count(FindInstruction(module.get(), "dot.23")));
}

}  // namespace
}  // namespace xla
