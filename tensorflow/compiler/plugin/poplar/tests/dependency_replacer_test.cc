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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dependency_replacer.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using DependencyReplacerTest = HloTestBase;

TEST_F(DependencyReplacerTest, TestReplaceOne) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  s0 = f16[] sine(a0)
  aa = token[] after-all(s0)
  m0 = f16[] multiply(s0, a1)
  d1 = f16[] add-dependency(a1, aa)
  m1 = f16[] multiply(d1, a2)
  ROOT %tuple = (f16[], f16[]) tuple(m0, m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  DependencyReplacer replacer(true);
  EXPECT_TRUE(replacer.Run(module).ValueOrDie());

  EXPECT_EQ(comp->instruction_count(), 7);

  auto* m1 = comp->GetInstructionWithName("m1");
  auto* s0 = comp->GetInstructionWithName("s0");
  ASSERT_EQ(m1->control_predecessors().size(), 1);
  EXPECT_EQ(m1->control_predecessors()[0], s0);
}

TEST_F(DependencyReplacerTest, TestMultipleDeps) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  s0 = f16[] sine(a0)
  aa = token[] after-all(s0)
  m0 = f16[] multiply(s0, a1)
  d1 = f16[] add-dependency(a1, aa)
  m1 = f16[] multiply(d1, a2)
  d2 = f16[] add-dependency(a1, aa)
  m2 = f16[] multiply(d2, a2)
  d3 = f16[] add-dependency(a1, aa)
  m3 = f16[] multiply(d3, a2)
  ROOT %tuple = (f16[], f16[], f16[], f16[]) tuple(m0, m1, m2, m3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  DependencyReplacer replacer(true);
  EXPECT_TRUE(replacer.Run(module).ValueOrDie());

  EXPECT_EQ(comp->instruction_count(), 9);

  auto* m1 = comp->GetInstructionWithName("m1");
  auto* m2 = comp->GetInstructionWithName("m2");
  auto* m3 = comp->GetInstructionWithName("m3");
  auto* s0 = comp->GetInstructionWithName("s0");
  ASSERT_EQ(m1->control_predecessors().size(), 1);
  EXPECT_EQ(m1->control_predecessors()[0], s0);
  ASSERT_EQ(m2->control_predecessors().size(), 1);
  EXPECT_EQ(m2->control_predecessors()[0], s0);
  ASSERT_EQ(m3->control_predecessors().size(), 1);
  EXPECT_EQ(m3->control_predecessors()[0], s0);
}

TEST_F(DependencyReplacerTest, TestMultipleDepsFromOneInst) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  s0 = f16[] sine(a0)
  aa = token[] after-all(s0)
  m0 = f16[] multiply(s0, a1)
  ab = token[] after-all(m0)
  m1 = f16[] multiply(s0, a2)
  ac = token[] after-all(m1)
  m2 = f16[] multiply(s0, a2)
  d1 = f16[] add-dependency(a1, aa)
  d2 = f16[] add-dependency(d1, ab)
  d3 = f16[] add-dependency(d2, ac)
  m3 = f16[] multiply(d3, a2)
  ROOT %tuple = (f16[], f16[], f16[], f16[]) tuple(m0, m1, m2, m3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  DependencyReplacer replacer(true);
  EXPECT_TRUE(replacer.Run(module).ValueOrDie());

  EXPECT_EQ(comp->instruction_count(), 9);

  auto* m1 = comp->GetInstructionWithName("m1");
  auto* m2 = comp->GetInstructionWithName("m2");
  auto* m3 = comp->GetInstructionWithName("m3");
  auto* s0 = comp->GetInstructionWithName("s0");
  ASSERT_EQ(m3->control_predecessors().size(), 3);
}

TEST_F(DependencyReplacerTest, TestDepFromEmptyAfterAll) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  s0 = f16[] sine(a0)
  aa = token[] after-all()
  m0 = f16[] multiply(s0, a1)
  d1 = f16[] add-dependency(a1, aa)
  m1 = f16[] multiply(d1, a2)
  ROOT %tuple = (f16[], f16[]) tuple(m0, m1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  DependencyReplacer replacer(true);
  EXPECT_TRUE(replacer.Run(module).ValueOrDie());

  EXPECT_EQ(comp->instruction_count(), 7);
}

TEST_F(DependencyReplacerTest, TestMultipleDepsFromDepTree) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  s0 = f16[] sine(a0)
  aa = token[] after-all(s0)
  m0 = f16[] multiply(s0, a1)
  ab = token[] after-all(m0)
  m1 = f16[] multiply(s0, a2)
  ac = token[] after-all(aa, ab)
  m2 = f16[] multiply(s0, a2)
  d3 = f16[] add-dependency(a1, ac)
  m3 = f16[] multiply(d3, a2)
  ROOT %tuple = (f16[], f16[], f16[], f16[]) tuple(m0, m1, m2, m3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  DependencyReplacer replacer(true);
  EXPECT_TRUE(replacer.Run(module).ValueOrDie());

  EXPECT_EQ(comp->instruction_count(), 9);

  auto* m3 = comp->GetInstructionWithName("m3");
  ASSERT_EQ(m3->control_predecessors().size(), 2);
}

TEST_F(DependencyReplacerTest, TestNoCtrlDeps) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  s0 = f16[] sine(a0)
  aa = token[] after-all(s0)
  m0 = f16[] multiply(s0, a1)
  ab = token[] after-all(m0)
  m1 = f16[] multiply(s0, a2)
  ac = token[] after-all(aa, ab)
  m2 = f16[] multiply(s0, a2)
  d3 = f16[] add-dependency(a1, ac)
  m3 = f16[] multiply(d3, a2)
  ROOT %tuple = (f16[], f16[], f16[], f16[]) tuple(m0, m1, m2, m3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  DependencyReplacer replacer(false);
  EXPECT_TRUE(replacer.Run(module).ValueOrDie());

  EXPECT_EQ(comp->instruction_count(), 9);

  auto* m3 = comp->GetInstructionWithName("m3");
  ASSERT_EQ(m3->control_predecessors().size(), 0);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
