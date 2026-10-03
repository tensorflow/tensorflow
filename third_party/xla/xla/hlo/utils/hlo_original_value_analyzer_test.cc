/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_original_value_analyzer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class HloOriginalValueAnalyzerTest : public HloHardwareIndependentTestBase {};

TEST_F(HloOriginalValueAnalyzerTest, LoggingRequestedTracking) {
  const char* module_str = R"(
HloModule TestModule

ENTRY main {
  src0 = f32[2] parameter(0)
  src1 = f32[2] parameter(1)
  res0 = f32[2] copy(src0), origin={{"orig0res0"}}
  ROOT res1 = f32[2] copy(src1), origin={{"orig1res1"}}
}
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(module_str));

  OriginalArray oa0{"orig0res0"};
  OriginalArray oa1{"orig1res1"};
  OriginalArray oa_unmapped{"unmapped"};

  HloModule::DebugAttributes attr0;
  attr0.callback_id = 123;
  HloModule::DebugAttributes attr1;
  attr1.callback_id = 456;
  HloModule::DebugAttributes attr_unmapped;
  attr_unmapped.callback_id = 789;

  module->AddDebugAttributes(oa0, attr0);
  module->AddDebugAttributes(oa1, attr1);
  module->AddDebugAttributes(oa_unmapped, attr_unmapped);

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));

  HloOriginalValueAnalyzer analyzer(std::move(analysis));

  // Before marking, everything is not logged.
  auto not_logged = analyzer.GetRequestedButNotLoggedOriginalArrays();
  EXPECT_EQ(not_logged.size(), 3);

  // Mark res0
  EXPECT_TRUE(analyzer.MarkOptimizedTensorAndCheckWhetherLoggingRequested(
      TensorKey::Create("res0")));

  // Now res0 is logged.
  not_logged = analyzer.GetRequestedButNotLoggedOriginalArrays();
  EXPECT_EQ(not_logged.size(), 2);

  // Mark res1
  EXPECT_TRUE(analyzer.MarkOptimizedTensorAndCheckWhetherLoggingRequested(
      TensorKey::Create("res1")));

  // Now res1 is logged. Unmapped is never logged.
  not_logged = analyzer.GetRequestedButNotLoggedOriginalArrays();
  ASSERT_EQ(not_logged.size(), 1);
  EXPECT_EQ(not_logged.front().instruction_name, "unmapped");

  // Mark unknown tensor
  EXPECT_FALSE(analyzer.MarkOptimizedTensorAndCheckWhetherLoggingRequested(
      TensorKey::Create("unknown")));
}

TEST_F(HloOriginalValueAnalyzerTest, CallbackTriggered) {
  const char* module_str = R"(
HloModule TestModule

ENTRY main {
  src0 = f32[2] parameter(0)
  ROOT res0 = f32[2] copy(src0), origin={{"orig0res0"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));

  OriginalArray oa0{"orig0res0"};
  HloModule::DebugAttributes attr0;
  attr0.callback_id = 123;
  module->AddDebugAttributes(oa0, attr0);

  TF_ASSERT_OK_AND_ASSIGN(auto analysis,
                          HloOriginalValueAnalysis::Create(module.get()));

  std::vector<TensorKey> triggered_keys;
  auto callback = [&](const TensorKey& key) { triggered_keys.push_back(key); };

  HloOriginalValueAnalyzer analyzer(std::move(analysis), callback);

  TensorKey key = TensorKey::Create("res0");
  EXPECT_TRUE(analyzer.MarkOptimizedTensorAndCheckWhetherLoggingRequested(key));

  ASSERT_EQ(triggered_keys.size(), 1);
  EXPECT_EQ(triggered_keys[0], key);

  TensorKey unknown_key = TensorKey::Create("unknown");
  EXPECT_FALSE(
      analyzer.MarkOptimizedTensorAndCheckWhetherLoggingRequested(unknown_key));
  EXPECT_EQ(triggered_keys.size(), 1);
}

}  // namespace
}  // namespace xla
