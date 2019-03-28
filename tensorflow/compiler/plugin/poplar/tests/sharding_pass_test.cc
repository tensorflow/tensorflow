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

#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ShardingPassTest = HloTestBase;

TEST_F(ShardingPassTest, TestNoSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0)
  mul0 = f16[4] multiply(sin0, arg1)
  mul1 = f16[4] multiply(mul0, arg2)
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_FALSE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_FALSE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddShardingSimple) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(ShardingPassTest, UnsupportedSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  a0 = s32[] parameter(0)
  a1 = f16[4] parameter(1)
  ROOT %tuple = () tuple(), sharding={replicated}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_FALSE(shardingPass.Run(module).ValueOrDie());
}

TEST_F(ShardingPassTest, UnsupportedAndSupportedShardingMixed) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={replicated}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT add = f16[4] add(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(ShardingPassTest, TestAddShardingTuplesAfter) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=1}
  tuple1 = (f16[4], f16[4]) tuple(mul0, mul1), sharding={maximal device=1}
  gte0 = f16[4] get-tuple-element(tuple1), index=0
  gte1 = f16[4] get-tuple-element(tuple1), index=1
  ROOT tuple2 = (f16[4], f16[4]) tuple(gte0, gte1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  HloInstruction* inst;

  inst = comp->GetInstructionWithName("tuple1");
  EXPECT_TRUE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetAsShapeTree(inst->shape()).leaf_count(), 2);
  inst = comp->GetInstructionWithName("tuple2");
  EXPECT_TRUE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetAsShapeTree(inst->shape()).leaf_count(), 2);
  inst = comp->GetInstructionWithName("gte0");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
  inst = comp->GetInstructionWithName("gte1");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
}

TEST_F(ShardingPassTest, TestAddShardingTuplesBefore) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = (f16[4], f16[4]) parameter(0)
  gte0 = f16[4] get-tuple-element(arg0), index=0
  gte1 = f16[4] get-tuple-element(arg0), index=1
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  mul0 = f16[4] multiply(gte0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(gte1, arg2), sharding={maximal device=1}
  ROOT tuple1 = (f16[4], f16[4]) tuple(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  HloInstruction* inst;

  inst = comp->GetInstructionWithName("arg0");
  EXPECT_TRUE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetAsShapeTree(inst->shape()).leaf_count(), 2);
  EXPECT_EQ(inst->sharding()
                .GetAsShapeTree(inst->shape())
                .element({0})
                .GetUniqueDevice(),
            0);
  EXPECT_EQ(inst->sharding()
                .GetAsShapeTree(inst->shape())
                .element({1})
                .GetUniqueDevice(),
            1);
  inst = comp->GetInstructionWithName("gte0");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
  inst = comp->GetInstructionWithName("gte1");
  EXPECT_FALSE(inst->sharding().IsTuple());
  EXPECT_EQ(inst->sharding().GetUniqueDevice(), 1);
}

TEST_F(ShardingPassTest, TestAddToCallSiteNonTuple) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1)
  ROOT s4 = f16[4] add(s3, s2), sharding={maximal device=0}
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  cal1 = f16[4] call(arg0, arg1, arg2), to_apply=subcomp
  ROOT add = f16[4] add(cal1, arg0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    EXPECT_FALSE(inst->sharding().IsTuple());
    EXPECT_EQ(inst->sharding().GetUniqueDevice(), 0);
  }
}

TEST_F(ShardingPassTest, TestAddToCallSiteTupleOutputCall) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s2), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s4)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  cal1 = (f16[4], f16[4]) call(arg0, arg1, arg2), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddToCallSiteTupleOutputWhile) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = (f16[4], f16[4], f16[4]) parameter(0)
  s1 = f16[4] get-tuple-element(s0), index=0
  s2 = f16[4] get-tuple-element(s0), index=1
  s3 = f16[4] get-tuple-element(s0), index=2
  s4 = f16[4] add(s1, s2), sharding={maximal device=1}
  s5 = f16[4] add(s3, s2), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s4, s5)
}

main {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  tup1 = (f16[4], f16[4], f16[4]) tuple(arg0, arg1, arg2)
  cal1 = (f16[4], f16[4]) call(tup1), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddToCallSiteTupleOutputWhileFromTupleParameter) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = (f16[4], f16[4], f16[4]) parameter(0)
  s1 = f16[4] get-tuple-element(s0), index=0
  s2 = f16[4] get-tuple-element(s0), index=1
  s3 = f16[4] get-tuple-element(s0), index=2
  s4 = f16[4] add(s1, s2), sharding={maximal device=1}
  s5 = f16[4] add(s3, s2), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s4, s5)
}

main {
  arg0 = (f16[4], f16[4], f16[4]) parameter(0)
  cal1 = (f16[4], f16[4]) call(arg0), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationHasUnusedInput) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s4)
}

main {
  arg0 = (f16[4], f16[4], f16[4]) parameter(0)
  cal1 = (f16[4], f16[4]) call(arg0), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationPassesInputToOutput) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = f16[4] parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s2)
}

main {
  arg0 = (f16[4], f16[4], f16[4]) parameter(0)
  cal1 = (f16[4], f16[4]) call(arg0), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  g1 = f16[4] get-tuple-element(cal1), index=1
  ROOT add = f16[4] add(g0, g1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationPassesTupleInputToOutput) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  s0 = f16[4] parameter(0)
  s1 = f16[4] parameter(1)
  s2 = (f16[4], f16[4]) parameter(2)
  s3 = f16[4] add(s0, s1), sharding={maximal device=1}
  s4 = f16[4] add(s3, s1), sharding={maximal device=0}
  ROOT t = (f16[4], f16[4]) tuple(s3, s2)
}

main {
  arg0 = (f16[4], f16[4], (f16[4], f16[4])) parameter(0)
  cal1 = (f16[4], (f16[4], f16[4])) call(arg0), to_apply=subcomp
  g0 = f16[4] get-tuple-element(cal1), index=0
  ROOT add = f16[4] add(g0, g0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestComputationPassesTupleInputToOutputInWhileBody) {
  std::string hlo_string = R"(
HloModule top

cond {
  cp = (s32[], f16[4], f16[4]) parameter(0)
  gte = f16[4] get-tuple-element(cp), index=0
  cc = s32[] constant(10)
  ROOT lt = pred[] less-than(gte, cc)
}

body {
  bp = (s32[], f16[4], f16[4]) parameter(0)
  g0 = s32[] get-tuple-element(bp), index=0
  g1 = f16[4] get-tuple-element(bp), index=1
  g2 = f16[4] get-tuple-element(bp), index=2
  s3 = f16[4] add(g1, g1), sharding={maximal device=1}
  s4 = f16[4] add(g2, g2), sharding={maximal device=0}
  c = s32[] constant(1)
  add = s32[] add(g0, c), sharding={maximal device=0}
  ROOT t = (s32[], f16[4], f16[4]) tuple(add, s3, s4)
}

main {
  a0 = s32[] parameter(0)
  a1 = f16[4] parameter(1)
  a2 = f16[4] parameter(2)
  t = (s32[], f16[4], f16[4]) tuple(a0, a1, a2)
  w = (f16[4], (f16[4], f16[4])) while(t), condition=cond, body=body
  g = f16[4] get-tuple-element(w), index=1
  ROOT add = f16[4] add(g, g)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestSomeInputsPassThrough) {
  std::string hlo_string = R"(
HloModule top

main {
  bp = (s32[], f16[4], f16[4]) parameter(0)
  g0 = s32[] get-tuple-element(bp), index=0
  g1 = f16[4] get-tuple-element(bp), index=1
  g2 = f16[4] get-tuple-element(bp), index=2
  m0 = f16[4] multiply(g1, g2), sharding={maximal device=0}
  m1 = f16[4] multiply(g1, g2), sharding={maximal device=1}
  ROOT tp = (s32[], f16[4], f16[4]) tuple(g0, m0, m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestSomeGTEsMissingFromTuple) {
  std::string hlo_string = R"(
HloModule top

main {
  bp = (s32[], f16[4], f16[4]) parameter(0)
  g0 = s32[] get-tuple-element(bp), index=0
  g1 = f16[4] get-tuple-element(bp), index=1
  m0 = f16[4] multiply(g1, g1), sharding={maximal device=0}
  ROOT tp = (s32[], f16[4], f16[4]) tuple(g0, m0, m0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestBatchnormsHaveSingleSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f32[1,4,4,2] parameter(0)
  arg9 = f32[1,1,2,2] parameter(1)
  arg8 = f32[2] parameter(2)
  arg7 = f32[2] parameter(3)
  arg6 = f32[1,1,2,2] parameter(4)
  arg5 = f32[2] parameter(5)
  arg4 = f32[2] parameter(6)
  c1 = f32[1,4,4,2] convolution(arg0, arg9), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=0}
  bn1 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c1, arg8, arg7),
      epsilon=0.001, feature_index=3
  gte0 = f32[1,4,4,2] get-tuple-element(bn1), index=0
  c2 = f32[1,4,4,2] convolution(gte0, arg6), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=1}
  bn2 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c2, arg5, arg4),
      epsilon=0.001, feature_index=3
  gte1 = f32[1,4,4,2] get-tuple-element(bn2), index=0
  ROOT tuple = (f32[1,4,4,2]) tuple(gte1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  auto* bn1 = comp->GetInstructionWithName("bn1");
  EXPECT_FALSE(bn1->sharding().IsTuple());

  auto* bn2 = comp->GetInstructionWithName("bn2");
  EXPECT_FALSE(bn2->sharding().IsTuple());
}

TEST_F(ShardingPassTest, TestInfeedsDontTakeTokenSharding) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f32[1] parameter(0)
  tok1 = token[] after-all(), sharding={maximal device=0}
  inf1 = ((f32[1], f32[1]), token[]) infeed(tok1), sharding={maximal device=0}
  gte1 = (f32[1], f32[1]) get-tuple-element(inf1), index=0,
      sharding={maximal device=0}
  gte2 = f32[1] get-tuple-element(gte1), index=0, sharding={maximal device=0}
  gte3 = f32[1] get-tuple-element(gte1), index=1, sharding={maximal device=0}
  add1 = f32[1] add(arg0, gte2), sharding={maximal device=1}
  add2 = f32[1] add(add1, gte3), sharding={maximal device=1}

  ROOT tuple = (f32[1,4,4,2]) tuple(add2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  auto* inf1 = comp->GetInstructionWithName("inf1");
  ASSERT_TRUE(inf1->sharding().IsTuple());
  auto shardings = inf1->sharding().tuple_elements();
  ASSERT_TRUE(shardings[0].HasUniqueDevice());
  EXPECT_EQ(shardings[0].GetUniqueDevice(), 1);
  ASSERT_TRUE(shardings[1].HasUniqueDevice());
  EXPECT_EQ(shardings[1].GetUniqueDevice(), 1);
}

TEST_F(ShardingPassTest, TestGteOpsMatchTheirOperands) {
  std::string hlo_string = R"(
HloModule top

main {
  arg0 = f32[1,4,4,2] parameter(0)
  arg9 = f32[1,1,2,2] parameter(1)
  arg8 = f32[2] parameter(2)
  arg7 = f32[2] parameter(3)
  arg6 = f32[1,1,2,2] parameter(4)
  arg5 = f32[2] parameter(5)
  arg4 = f32[2] parameter(6)
  c1 = f32[1,4,4,2] convolution(arg0, arg9), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=0}
  bn1 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c1, arg8, arg7),
      epsilon=0.001, feature_index=3, sharding={maximal device=0}
  gte0 = f32[1,4,4,2] get-tuple-element(bn1), index=0
  c2 = f32[1,4,4,2] convolution(gte0, arg6), window={size=1x1},
      dim_labels=b01f_01io->b01f, sharding={maximal device=1}
  bn2 = (f32[1,4,4,2], f32[2], f32[2]) batch-norm-training(c2, arg5, arg4),
      epsilon=0.001, feature_index=3, sharding={maximal device=1}
  gte1 = f32[1,4,4,2] get-tuple-element(bn2), index=0
  ROOT tuple = (f32[1,4,4,2]) tuple(gte1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  ShardingPass shardingPass;
  ASSERT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = comp->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
  }

  // gte0 should match its operand bn1, not its user c2
  auto* gte0 = comp->GetInstructionWithName("gte0");
  ASSERT_TRUE(gte0->sharding().HasUniqueDevice());
  EXPECT_EQ(gte0->sharding().GetUniqueDevice(), 0);
}

// computations which are entrely empty

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
