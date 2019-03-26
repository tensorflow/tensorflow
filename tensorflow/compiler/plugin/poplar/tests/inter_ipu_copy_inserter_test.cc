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

#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using InterIpuCopyInserterTest = HloTestBase;

TEST_F(InterIpuCopyInserterTest, TestAddInterIpuCopySimple) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[4] parameter(0), sharding={maximal device=1}
  arg1 = f16[4] parameter(1), sharding={maximal device=0}
  arg2 = f16[4] parameter(2), sharding={maximal device=0}
  sin0 = f16[4] sine(arg0), sharding={maximal device=1}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4]) tuple(mul0, mul1),
      sharding={{maximal device=0}, {maximal device=0}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* root = module->entry_computation()->root_instruction();
  auto* mul0 = root->operand(0);
  auto* sin0 = mul0->operand(0);

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(module->entry_computation()->instruction_count(), 8);
  EXPECT_EQ(mul0->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(mul0->operand(0)->operand(0), sin0);
}

TEST_F(InterIpuCopyInserterTest, TestAddInterIpuCopyOneFeedsMany) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[4] parameter(0), sharding={maximal device=1}
  arg1 = f16[4] parameter(1), sharding={maximal device=0}
  arg2 = f16[4] parameter(2), sharding={maximal device=0}
  sin0 = f16[4] sine(arg0), sharding={maximal device=1}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(sin0, arg2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4]) tuple(mul0, mul1),
      sharding={{maximal device=0}, {maximal device=0}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* root = module->entry_computation()->root_instruction();
  auto* mul0 = root->operand(0);
  auto* mul1 = root->operand(1);
  auto* sin0 = mul0->operand(0);

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(module->entry_computation()->instruction_count(), 8);
  EXPECT_EQ(mul0->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(mul1->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(mul1->operand(0)->operand(0), sin0);
  EXPECT_EQ(mul1->operand(0)->operand(0), sin0);
}

TEST_F(InterIpuCopyInserterTest, CheckProgressNotMade) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = s32[] parameter(0), sharding={maximal device=0}
  a1 = f16[4] parameter(1), sharding={maximal device=0}
  ROOT %tuple = () tuple(), sharding={maximal device=0}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  InterIpuCopyInserter inserterPass;
  TF_PREDICT_FALSE(inserterPass.Run(module).ok());
}

TEST_F(InterIpuCopyInserterTest, CloneConstantToIpu) {
  // In this test we check that the constant inst has been cloned with the new
  // sharding information rather than copied.
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[] parameter(0), sharding={maximal device=0}
  const0 = f16[] constant(0.1), sharding={maximal device=0}
  ROOT mul = f16[4] multiply(arg0, const0), sharding={maximal device=1}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kConstant);

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(InterIpuCopyInserterTest, CloneWideConstantToIpu) {
  // In this test we check that the wide constant inst has been cloned with the
  // new sharding information rather than copied.
  std::string hlo_string = R"(
HloModule top

%_pop_op_wide_const () -> f16[4] {
  constant.clone = f16[] constant(0.1)
  ROOT %broadcast.clone = f16[4] broadcast(f16[] constant.clone), dimensions={}
}

%cluster_1  {
  arg0 = f16[4] parameter(0), sharding={maximal device=0}
  wide_const = f16[4] fusion(), kind=kCustom, calls=%_pop_op_wide_const, sharding={maximal device=0}
  ROOT mul = f16[4] multiply(arg0, wide_const), sharding={maximal device=1}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kFusion);

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(InterIpuCopyInserterTest, TestAddInterIpuCopyBeforeCall) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = f16[4] parameter(0), sharding={maximal device=0}
  p1 = f16[4] parameter(1), sharding={maximal device=0}
  p2 = f16[4] add(p0, p1), sharding={maximal device=1}
  ROOT a2 = f16[4] add(p2, p2), sharding={maximal device=0}
}

%cluster_1  {
  a0 = f16[4] parameter(0), sharding={maximal device=1}
  a1 = f16[4] parameter(1), sharding={maximal device=1}
  c0 = f16[4] call(a0, a1), to_apply=subcomp, sharding={maximal device=0}
  ROOT %tuple = (f16[4]) tuple(c0), sharding={{maximal device=0}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* call = comp->GetInstructionWithName("c0");
  auto* a0 = comp->GetInstructionWithName("a0");
  auto* a1 = comp->GetInstructionWithName("a1");

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(module->entry_computation()->instruction_count(), 6);
  EXPECT_EQ(call->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(call->operand(1)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(call->operand(0)->operand(0), a0);
  EXPECT_EQ(call->operand(1)->operand(0), a1);
}

TEST_F(InterIpuCopyInserterTest, TestAddInterIpuCopyBeforeWhile) {
  std::string hlo_string = R"(
HloModule top

cond {
  c_p0 = (s32[], f16[4], f16[4]) parameter(0)
  c_c0 = s32[] constant(4)
  c_e0 = s32[] get-tuple-element(c_p0), index=0
  ROOT c_eq = pred[] equal-to(c_e0, c_c0)
}

body {
  b_p0 = (s32[], f16[4], f16[4]) parameter(0),
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  b_e0 = s32[] get-tuple-element(b_p0), index=0, sharding={maximal device=1}
  b_e1 = f16[4] get-tuple-element(b_p0), index=1, sharding={maximal device=0}
  b_e2 = f16[4] get-tuple-element(b_p0), index=2, sharding={maximal device=0}
  b_s0 = f16[4] sine(b_e1), sharding={maximal device=0}
  b_s1 = f16[4] sine(b_e2), sharding={maximal device=0}
  b_c0 = s32[] constant(1), sharding={maximal device=0}
  b_a0 = s32[] add(b_e0, b_c0), sharding={maximal device=1}
  ROOT b_t0 = (s32[], f16[4], f16[4]) tuple(b_a0, b_s0, b_e2),
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
}

%cluster_1  {
  a0 = s32[] parameter(0), sharding={maximal device=0}
  a1 = f16[4] parameter(1), sharding={maximal device=0}
  t0 = (s32[], f16[4], f16[4]) tuple(a0, a1, a1),
      sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  w0 = (s32[], f16[4], f16[4]) while(t0), condition=cond, body=body,
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  e1 = f16[4] get-tuple-element(w0), index=1, sharding={maximal device=0}
  e2 = f16[4] get-tuple-element(w0), index=2, sharding={maximal device=0}
  s0 = f16[4] sine(e1), sharding={maximal device=0}
  s1 = f16[4] sine(e2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4]) tuple(s0, s1),
      sharding={{maximal device=0}, {maximal device=0}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* whl = comp->GetInstructionWithName("w0");

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(module->entry_computation()->instruction_count(), 10);
  EXPECT_EQ(whl->operand(0)->opcode(), HloOpcode::kCustomCall);
}

TEST_F(InterIpuCopyInserterTest, TestAddInterIpuCopyInsideCall) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = f16[4] parameter(0), sharding={maximal device=0}
  p1 = f16[4] parameter(1), sharding={maximal device=0}
  s0 = f16[4] sine(p0), sharding={maximal device=0}
  s1 = f16[4] sine(p1), sharding={maximal device=0}
  ROOT a = f16[4] add(s0, s1), sharding={maximal device=1}
}

%cluster_1  {
  a0 = f16[4] parameter(0), sharding={maximal device=1}
  a1 = f16[4] parameter(1), sharding={maximal device=1}
  c0 = f16[4] call(a0, a1), to_apply=subcomp, sharding={maximal device=1}
  ROOT %tuple = (f16[4]) tuple(c0), sharding={{maximal device=1}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* call = comp->GetInstructionWithName("c0");
  auto* add = call->to_apply()->GetInstructionWithName("a");

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(module->entry_computation()->instruction_count(), 6);
  EXPECT_EQ(call->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(call->operand(1)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(add->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(add->operand(1)->opcode(), HloOpcode::kCustomCall);
}

TEST_F(InterIpuCopyInserterTest, TestAddInterIpuCopyInsideWhile) {
  std::string hlo_string = R"(
HloModule top

cond {
  c_p0 = (s32[], f16[4], f16[4]) parameter(0)
  c_c0 = s32[] constant(4)
  c_e0 = s32[] get-tuple-element(c_p0), index=0
  ROOT c_eq = pred[] equal-to(c_e0, c_c0)
}

body {
  b_p0 = (s32[], f16[4], f16[4]) parameter(0),
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  b_e0 = s32[] get-tuple-element(b_p0), index=0, sharding={maximal device=1}
  b_e1 = f16[4] get-tuple-element(b_p0), index=1, sharding={maximal device=0}
  b_e2 = f16[4] get-tuple-element(b_p0), index=2, sharding={maximal device=0}
  b_s0 = f16[4] sine(b_e1), sharding={maximal device=0}
  b_s1 = f16[4] sine(b_e2), sharding={maximal device=0}
  b_c0 = s32[] constant(1), sharding={maximal device=0}
  b_a0 = s32[] add(b_e0, b_c0), sharding={maximal device=1}
  ROOT b_t0 = (s32[], f16[4], f16[4]) tuple(b_a0, b_s0, b_e2),
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
}

%cluster_1  {
  a0 = s32[] parameter(0), sharding={maximal device=0}
  a1 = f16[4] parameter(1), sharding={maximal device=0}
  t0 = (s32[], f16[4], f16[4]) tuple(a0, a1, a1),
      sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  w0 = (s32[], f16[4], f16[4]) while(t0), condition=cond, body=body,
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  e1 = f16[4] get-tuple-element(w0), index=1, sharding={maximal device=0}
  e2 = f16[4] get-tuple-element(w0), index=2, sharding={maximal device=0}
  s0 = f16[4] sine(e1), sharding={maximal device=0}
  s1 = f16[4] sine(e2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4]) tuple(s0, s1),
      sharding={{maximal device=0}, {maximal device=0}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* whl = comp->GetInstructionWithName("w0");
  auto* body = whl->while_body();

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(body->instruction_count(), 10);
}

TEST_F(InterIpuCopyInserterTest, TestNoIpuCopyOnTokens) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[4] parameter(0), sharding={maximal device=1}
  arg1 = f16[4] parameter(1), sharding={maximal device=0}
  arg2 = f16[4] parameter(2), sharding={maximal device=0}
  sin0 = f16[4] sine(arg0), sharding={maximal device=1}
  tok1 = token[] after-all(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4], token[]) tuple(mul0, mul1, tok1),
      sharding={{maximal device=0}, {maximal device=0},  {maximal device=1}}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  EXPECT_EQ(module->entry_computation()->instruction_count(), 9);
}

TEST_F(InterIpuCopyInserterTest, TestGteShardingMustMatchTupleSharding) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[4] parameter(0), sharding={maximal device=1}
  a1 = f16[4] parameter(1), sharding={maximal device=0}
  a2 = f16[4] parameter(2), sharding={maximal device=0}
  t0 = (s32[], f16[4], f16[4]) tuple(a0, a1, a2),
      sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  ROOT e1 = f16[4] get-tuple-element(t0), index=0, sharding={maximal device=0}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  InterIpuCopyInserter inserterPass;
  EXPECT_EQ(tensorflow::error::INTERNAL,
            inserterPass.Run(module).status().code());
}

// Test feeding the second operand of a call instruction

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
