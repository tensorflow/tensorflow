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

#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using FindAddUsersTest = HloTestBase;

TEST_F(FindAddUsersTest, FindSimpleUsers) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  a2 = f16[4] parameter(2)
  s0 = f16[4] sine(a0)
  m0 = f16[4] multiply(s0, a1)
  m1 = f16[4] multiply(m0, a2)
  ROOT %tuple = (f16[4], f16[4]) tuple(m0, m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* mul0 = comp->GetInstructionWithName("m0");
  auto* mul1 = comp->GetInstructionWithName("m1");
  auto* sin0 = comp->GetInstructionWithName("s0");
  auto* p0 = comp->GetInstructionWithName("a0");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(sin0) == 1);
  EXPECT_EQ(finder.PathFor(sin0).size(), 1);

  // kTuple instructions are not listed as users - so the root doesn't count
  finder.Find(mul0);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 1);
  EXPECT_TRUE(users1.count(mul1) == 1);
  EXPECT_EQ(finder.PathFor(mul1).size(), 1);
}

TEST_F(FindAddUsersTest, FindMultipleUsers) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  a2 = f16[4] parameter(2)
  s0 = f16[4] sine(a0)
  m0 = f16[4] multiply(s0, a0)
  m1 = f16[4] multiply(a1, a2)
  m2 = f16[4] multiply(a1, a2)
  m3 = f16[4] multiply(a1, a1)
  ROOT %tuple = (f16[4], f16[4], f16[4], f16[4]) tuple(m0, m1, m2, m3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* m0 = comp->GetInstructionWithName("m0");
  auto* m1 = comp->GetInstructionWithName("m1");
  auto* m2 = comp->GetInstructionWithName("m2");
  auto* m3 = comp->GetInstructionWithName("m3");
  auto* sin0 = comp->GetInstructionWithName("s0");
  auto* p0 = comp->GetInstructionWithName("a0");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* p2 = comp->GetInstructionWithName("a2");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 2);
  EXPECT_TRUE(users0.count(sin0) == 1);
  EXPECT_TRUE(users0.count(m0) == 1);
  EXPECT_EQ(finder.PathFor(sin0).size(), 1);
  EXPECT_EQ(finder.PathFor(m0).size(), 1);

  finder.Find(p1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 3);
  EXPECT_TRUE(users1.count(m1) == 1);
  EXPECT_TRUE(users1.count(m2) == 1);
  EXPECT_TRUE(users1.count(m3) == 1);
  EXPECT_EQ(finder.PathFor(m1).size(), 1);
  EXPECT_EQ(finder.PathFor(m2).size(), 1);
  EXPECT_EQ(finder.PathFor(m3).size(), 1);

  finder.Find(p2);
  auto users2 = finder.Users();
  ASSERT_EQ(users2.size(), 2);
  EXPECT_TRUE(users2.count(m1) == 1);
  EXPECT_TRUE(users2.count(m2) == 1);
  EXPECT_EQ(finder.PathFor(m1).size(), 1);
  EXPECT_EQ(finder.PathFor(m2).size(), 1);
}

TEST_F(FindAddUsersTest, FindMultipleFusionUsers) {
  std::string hlo_string = R"(
HloModule top

_pop_op_special {
  p0 = f16[4] parameter(0)
  ROOT ad0 = f16[4] negate(p0)
}

%cluster_1  {
  a0 = f16[4] parameter(0)
  s0 = f16[4] fusion(a0), kind=kCustom, calls=_pop_op_special
  s1 = f16[4] fusion(a0), kind=kCustom, calls=_pop_op_special
  ROOT %tuple = (f16[4], f16[4]) tuple(s0, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* s0 = comp->GetInstructionWithName("s0");
  auto* s1 = comp->GetInstructionWithName("s1");
  auto* p0 = comp->GetInstructionWithName("a0");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 2);
  EXPECT_TRUE(users0.count(s0) == 1);
  EXPECT_TRUE(users0.count(s1) == 1);
  EXPECT_EQ(finder.PathFor(s0).size(), 1);
  EXPECT_EQ(finder.PathFor(s1).size(), 1);
}

// Through a kTuple/kGetTupleElement pair
TEST_F(FindAddUsersTest, FindUsersThroughTuple) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  a2 = f16[4] parameter(2)
  s0 = f16[4] sine(a0)
  t0 = (f16[4], f16[4]) tuple(a1, a2)
  e0 = f16[4] get-tuple-element(t0), index=0
  m0 = f16[4] multiply(s0, e0)
  e1 = f16[4] get-tuple-element(t0), index=1
  m1 = f16[4] multiply(m0, e1)
  ROOT %tuple = (f16[4], f16[4]) tuple(m0, m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* m0 = comp->GetInstructionWithName("m0");
  auto* m1 = comp->GetInstructionWithName("m1");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* p2 = comp->GetInstructionWithName("a2");

  FindAllUsers finder;
  finder.Find(p1);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(m0) == 1);
  EXPECT_EQ(finder.PathFor(m0).size(), 3);

  finder.Find(p2);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 1);
  EXPECT_TRUE(users1.count(m1) == 1);
  EXPECT_EQ(finder.PathFor(m1).size(), 3);
}

// Into a kCall
TEST_F(FindAddUsersTest, FindIntoCall) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = f16[4] parameter(0)
  p1 = f16[4] parameter(1)
  ROOT a = f16[4] add(p0, p1)
}

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  c0 = f16[4] call(a0, a1), to_apply=subcomp
  ROOT %tuple = (f16[4]) tuple(c0)
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
  auto* add = call->to_apply()->GetInstructionWithName("a");

  FindAllUsers finder;
  finder.Find(a0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(add) == 1);
  EXPECT_EQ(finder.PathFor(add).size(), 2);
  EXPECT_EQ(finder.PathFor(add)[0], call);
  EXPECT_EQ(finder.PathFor(add)[1], add);

  finder.Find(a1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 1);
  EXPECT_TRUE(users1.count(add) == 1);
  EXPECT_EQ(finder.PathFor(add).size(), 2);
  EXPECT_EQ(finder.PathFor(add)[0], call);
  EXPECT_EQ(finder.PathFor(add)[1], add);
}

// Into a kCall with a tuple argument
TEST_F(FindAddUsersTest, FindIntoCallWithTupleArgument) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = (f16[4], f16[4]) parameter(0)
  e0 = f16[4] get-tuple-element(p0), index=0
  e1 = f16[4] get-tuple-element(p0), index=1
  ROOT a = f16[4] add(e0, e1)
}

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  c0 = f16[4] call(t0), to_apply=subcomp
  ROOT %tuple = (f16[4]) tuple(c0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* call = comp->GetInstructionWithName("c0");
  auto* p0 = comp->GetInstructionWithName("a0");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* add = call->to_apply()->GetInstructionWithName("a");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(add) == 1);

  finder.Find(p1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 1);
  EXPECT_TRUE(users1.count(add) == 1);
}

// Into a kFusion with a tuple argument
TEST_F(FindAddUsersTest, FindTargetsPoplibsCall) {
  std::string hlo_string = R"(
HloModule top

_pop_op_specialadd {
  p0 = f16[4] parameter(0)
  p1 = f16[4] parameter(1)
  ROOT ad0 = f16[4] add(p0, p1)
}

subcomp {
  p0 = (f16[4], f16[4]) parameter(0)
  e0 = f16[4] get-tuple-element(p0), index=0
  e1 = f16[4] get-tuple-element(p0), index=1
  ROOT ad1 = f16[4] fusion(e0, e1), kind=kCustom, calls=_pop_op_specialadd
}

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  c0 = f16[4] call(t0), to_apply=subcomp
  ROOT %tuple = (f16[4]) tuple(c0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* call = comp->GetInstructionWithName("c0");
  auto* p0 = comp->GetInstructionWithName("a0");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* popcall = call->to_apply()->GetInstructionWithName("ad1");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(popcall) == 1);

  finder.Find(p1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 1);
  EXPECT_TRUE(users1.count(popcall) == 1);
}

// Through a kCall without tuple return
TEST_F(FindAddUsersTest, FindIntoAndOutOfCall) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = (f16[4], f16[4]) parameter(0)
  ROOT e0 = f16[4] get-tuple-element(p0), index=0
}

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  c0 = f16[4] call(t0), to_apply=subcomp
  s0 = f16[4] sine(c0)
  ROOT %tuple = (f16[4]) tuple(s0)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* p0 = comp->GetInstructionWithName("a0");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* s0 = comp->GetInstructionWithName("s0");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(s0) == 1);

  finder.Find(p1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 0);
}

// Through a kCall without tuple return
TEST_F(FindAddUsersTest, FindIntoAndOutOfCallWithTupleReturn) {
  std::string hlo_string = R"(
HloModule top

subcomp {
  p0 = (f16[4], f16[4]) parameter(0)
  e0 = f16[4] get-tuple-element(p0), index=0
  e1 = f16[4] get-tuple-element(p0), index=1
  ROOT t0 = (f16[4], f16[4]) tuple(e0, e1)
}

%cluster_1  {
  a0 = f16[4] parameter(0)
  a1 = f16[4] parameter(1)
  t0 = (f16[4], f16[4]) tuple(a0, a1)
  c0 = (f16[4], f16[4]) call(t0), to_apply=subcomp
  e0 = f16[4] get-tuple-element(c0), index=0
  s0 = f16[4] sine(e0)
  e1 = f16[4] get-tuple-element(c0), index=1
  s1 = f16[4] sine(e1)
  ROOT %tuple = (f16[4], f16[4]) tuple(s0, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* p0 = comp->GetInstructionWithName("a0");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* s0 = comp->GetInstructionWithName("s0");
  auto* s1 = comp->GetInstructionWithName("s1");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(s0) == 1);

  finder.Find(p1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 1);
  EXPECT_TRUE(users1.count(s1) == 1);
}

// Into a kWhile
TEST_F(FindAddUsersTest, FindIntoWhile) {
  std::string hlo_string = R"(
HloModule top

cond {
  c_p0 = (s32[], f16[4], f16[4]) parameter(0)
  c_c0 = s32[] constant(4)
  c_e0 = s32[] get-tuple-element(c_p0), index=0
  ROOT eq = pred[] equal-to(c_e0, c_c0)
}

body {
  b_p0 = (s32[], f16[4], f16[4]) parameter(0)
  b_e0 = f16[4] get-tuple-element(b_p0), index=0
  b_e1 = f16[4] get-tuple-element(b_p0), index=1
  b_e2 = f16[4] get-tuple-element(b_p0), index=2
  b_s0 = f16[4] sine(b_e1)
  b_c0 = s32[] constant(1)
  b_a0 = s32[] add(b_e0, b_c0)
  ROOT b_t0 = (s32[], f16[4], f16[4]) tuple(b_a0, b_s0, b_e2)
}

%cluster_1  {
  a0 = s32[] parameter(0)
  a1 = f16[4] parameter(1)
  t0 = (s32[], f16[4], f16[4]) tuple(a0, a1, a1)
  w0 = (s32[], f16[4], f16[4]) while(t0), condition=cond, body=body
  e1 = f16[4] get-tuple-element(w0), index=1
  e2 = f16[4] get-tuple-element(w0), index=2
  s0 = f16[4] sine(e1)
  s1 = f16[4] sine(e2)
  ROOT %tuple = (f16[4], f16[4]) tuple(s0, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* p0 = comp->GetInstructionWithName("a0");
  auto* p1 = comp->GetInstructionWithName("a1");
  auto* s0 = comp->GetInstructionWithName("s0");
  auto* s1 = comp->GetInstructionWithName("s1");

  auto* w_comp = comp->GetInstructionWithName("w0")->while_body();
  auto* w_a0 = w_comp->GetInstructionWithName("b_a0");
  auto* w_s0 = w_comp->GetInstructionWithName("b_s0");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 1);
  EXPECT_TRUE(users0.count(w_a0) == 1);

  finder.Find(p1);
  auto users1 = finder.Users();
  ASSERT_EQ(users1.size(), 2);
  EXPECT_TRUE(users1.count(w_s0) == 1);
  EXPECT_TRUE(users1.count(s1) == 1);
}

// Into a Repeat instruction
TEST_F(FindAddUsersTest, FindIntoRepeat) {
  std::string hlo_string = R"(
HloModule top

body {
  body_tuple.1 = (s32[], f16[4], f16[4]) parameter(0)
  p.4 = s32[] get-tuple-element(body_tuple.1), index=0
  p.5 = f16[4] get-tuple-element(body_tuple.1), index=1
  p.6 = f16[4] get-tuple-element(body_tuple.1), index=2
  s0 = f16[4] sine(p.5)
  ROOT b_t0 = (s32[], f16[4], f16[4]) tuple(p.4, s0, p.6)
}

ENTRY in {
  constant.4 = s32[] constant(10)
  p0 = f16[4] parameter(0)
  in = (s32[], f16[4], f16[4]) tuple(s32[] constant.4, f16[4] p0, f16[4] p0)
  r0 = (s32[], f16[4], f16[4]) call((s32[], f16[4], f16[4]) in), to_apply=body, backend_config="{\"repeatConfig\":{\"isRepeatLoop\":true,\"repeatCount\":\"100\"}}"
  e1 = f16[4] get-tuple-element(r0), index=1
  e2 = f16[4] get-tuple-element(r0), index=2
  s0 = f16[4] sine(e1)
  s1 = f16[4] sine(e2)
  ROOT tuple = (f16[4], f16[4]) tuple(s0, s1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();
  auto* p0 = comp->GetInstructionWithName("p0");
  auto* s0 = comp->GetInstructionWithName("s0");
  auto* s1 = comp->GetInstructionWithName("s1");

  auto* r_comp = comp->GetInstructionWithName("r0")->to_apply();
  auto* r_s0 = r_comp->GetInstructionWithName("s0");

  FindAllUsers finder;
  finder.Find(p0);
  auto users0 = finder.Users();
  ASSERT_EQ(users0.size(), 2);
  EXPECT_TRUE(users0.count(r_s0) == 1);
  EXPECT_TRUE(users0.count(s1) == 1);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
