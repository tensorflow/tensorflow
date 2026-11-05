/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/erase_xla_late_frontend_attributes.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace gpu {
namespace {

using EraseXlaLateFrontendAttributesTest = HloHardwareIndependentTestBase;

TEST_F(EraseXlaLateFrontendAttributesTest, ErasesXlaLateAttribute) {
  const char* const hlo_string = R"hlo(
HloModule TestModule

%callee (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %mul = f32[] multiply(%x, %x)
}

ENTRY %entry (a: f32[]) -> (f32[], f32[]) {
  %a = f32[] parameter(0)
  %call_late = f32[] call(%a), to_apply=%callee, frontend_attributes={inlineable="xla_late", other_attr="keep"}
  %call_false = f32[] call(%a), to_apply=%callee, frontend_attributes={inlineable="false"}
  ROOT %tuple = (f32[], f32[]) tuple(%call_late, %call_false)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  EraseXlaLateFrontendAttributes pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_late =
      module->entry_computation()->GetInstructionWithName("call_late");
  EXPECT_FALSE(call_late->frontend_attributes().map().contains("inlineable"));
  EXPECT_EQ(call_late->get_frontend_attribute("other_attr"), "keep");

  const HloInstruction* call_false =
      module->entry_computation()->GetInstructionWithName("call_false");
  EXPECT_EQ(call_false->get_frontend_attribute("inlineable"), "false");
}

TEST_F(EraseXlaLateFrontendAttributesTest,
       PreservesShardyManualComputationXlaLateAttribute) {
  const char* const hlo_string = R"hlo(
HloModule TestModule

%xla.sdy.manual_computation_body (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %mul = f32[] multiply(%x, %x)
}

%callee (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %add = f32[] add(%x, %x)
}

ENTRY %entry (a: f32[]) -> (f32[], f32[]) {
  %a = f32[] parameter(0)
  %shmap_call = f32[] call(%a), to_apply=%xla.sdy.manual_computation_body, frontend_attributes={inlineable="xla_late"}
  %regular_late_call = f32[] call(%a), to_apply=%callee, frontend_attributes={inlineable="xla_late"}
  ROOT %tuple = (f32[], f32[]) tuple(%shmap_call, %regular_late_call)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config().set_use_shardy_partitioner(true);

  EraseXlaLateFrontendAttributes pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* shmap_call =
      module->entry_computation()->GetInstructionWithName("shmap_call");
  EXPECT_EQ(shmap_call->get_frontend_attribute("inlineable"), "xla_late");

  const HloInstruction* regular_late_call =
      module->entry_computation()->GetInstructionWithName("regular_late_call");
  EXPECT_FALSE(
      regular_late_call->frontend_attributes().map().contains("inlineable"));
}

TEST_F(EraseXlaLateFrontendAttributesTest, NoOpWhenNoXlaLate) {
  const char* const hlo_string = R"hlo(
HloModule TestModule

%callee (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %mul = f32[] multiply(%x, %x)
}

ENTRY %entry (a: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  ROOT %call = f32[] call(%a), to_apply=%callee, frontend_attributes={inlineable="false"}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  EraseXlaLateFrontendAttributes pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
