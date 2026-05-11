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

#include "xla/hlo/transforms/debug_log_custom_call_processor.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

class DebugLogCustomCallProcessorTest : public HloHardwareIndependentTestBase {
};

TEST_F(DebugLogCustomCallProcessorTest, RetainsGuaranteedLog) {
  const char* hlo_string = R"hlo(
HloModule m
ENTRY %main.1 (x.1: s32[3]) -> () {
  %x.1 = s32[3]{0} parameter(0)
  ROOT %xla_debug_log.5 = () custom-call(%x.1),
    custom_call_target="xla.debug.Log",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config="{\"debug_attributes_config\": {\"callback_id\": 3, \"log_mode\": \"GUARANTEED\"}}"
}
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DebugLogCustomCallProcessor pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify that the custom call instruction is retained.
  auto instructions = module->entry_computation()->instructions();
  bool found = false;
  for (auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "xla.debug.Log") {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(DebugLogCustomCallProcessorTest, ProcessesDefaultMode) {
  const char* hlo_string = R"hlo(
HloModule m
ENTRY %main.1 (x.1: s32[3]) -> s32[3] {
  ROOT %x.1 = s32[3]{0} parameter(0), origin={{"x.1"}}
  %xla_debug_log.5 = () custom-call(%x.1),
    custom_call_target="xla.debug.Log",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config="{\"debug_attributes_config\": {\"callback_id\": 3, \"log_mode\": \"DEFAULT\"}}"
}
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DebugLogCustomCallProcessor pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  auto instructions = module->entry_computation()->instructions();
  bool found = false;
  for (auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "xla.debug.Log") {
      found = true;
      break;
    }
  }
  EXPECT_FALSE(found);
  EXPECT_FALSE(module->debug_attributes().empty());
}

TEST_F(DebugLogCustomCallProcessorTest, ProcessesFusionDebuggerMode) {
  const char* hlo_string = R"hlo(
HloModule m
ENTRY %main.1 (x.1: s32[3]) -> s32[3] {
  ROOT %x.1 = s32[3]{0} parameter(0), origin={{"x.1"}}
  %xla_debug_log.5 = () custom-call(%x.1),
    custom_call_target="xla.debug.Log",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config="{\"debug_attributes_config\": {\"callback_id\": 3, \"log_mode\": \"FUSION_DEBUGGER\"}}"
}
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DebugLogCustomCallProcessor pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  auto instructions = module->entry_computation()->instructions();
  bool found = false;
  for (auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kCustomCall &&
        instr->custom_call_target() == "xla.debug.Log") {
      found = true;
      break;
    }
  }
  EXPECT_FALSE(found);
  EXPECT_FALSE(module->debug_attributes().empty());
}

TEST_F(DebugLogCustomCallProcessorTest, ProcessesNoFusionMode) {
  const char* hlo_string = R"hlo(
HloModule m
ENTRY %main.1 (x.1: s32[3]) -> s32[3] {
  %x.1 = s32[3]{0} parameter(0)
  %xla_debug_log.5 = () custom-call(%x.1),
    custom_call_target="xla.debug.Log",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config="{\"debug_attributes_config\": {\"callback_id\": 3, \"log_mode\": \"NO_FUSION\"}}"
  ROOT %add = s32[3]{0} add(%x.1, %x.1)
}
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DebugLogCustomCallProcessor pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  auto instructions = module->entry_computation()->instructions();
  bool found_barrier = false;
  for (auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kOptimizationBarrier) {
      found_barrier = true;
      break;
    }
  }
  EXPECT_TRUE(found_barrier);
}

TEST_F(DebugLogCustomCallProcessorTest, ErrorsOnDuplicateHloId) {
  const char* hlo_string = R"hlo(
HloModule m
ENTRY %main.1 (x.1: s32[3]) -> s32[3] {
  ROOT %x.1 = s32[3]{0} parameter(0)
  %xla_debug_log.5 = () custom-call(%x.1),
    custom_call_target="xla.debug.Log",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config="{\"debug_attributes_config\": {\"callback_id\": 3, \"log_mode\": \"DEFAULT\"}}"
  %xla_debug_log.6 = () custom-call(%x.1),
    custom_call_target="xla.debug.Log",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_STATUS_RETURNING,
    backend_config="{\"debug_attributes_config\": {\"callback_id\": 3, \"log_mode\": \"DEFAULT\"}}"
}
  )hlo";
  auto module_status = ParseAndReturnVerifiedModule(hlo_string);
  ASSERT_TRUE(module_status.ok());
  auto module = std::move(module_status).value();
  DebugLogCustomCallProcessor pass;
  auto status = pass.Run(module.get());
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.status().message(),
                                "Duplicate callback_id found"));
}

}  // namespace
}  // namespace xla
