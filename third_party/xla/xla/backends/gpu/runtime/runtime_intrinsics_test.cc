/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using RuntimeIntrinsicsTest = HloTestBase;

using ::testing::EndsWith;
using ::testing::HasSubstr;

TEST_F(RuntimeIntrinsicsTest, NopReturnTokenWorks) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  constant = u32[2]{0} constant({0, 1})
  ROOT nop_return_token = token[] custom-call(constant), custom_call_target="NopReturnToken", custom_call_has_side_effect=true, api_version=API_VERSION_STATUS_RETURNING
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  // The parameter of the NopReturnToken is not removed.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);
  // Can run.
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

TEST_F(RuntimeIntrinsicsTest, AssertionCustomCall) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  constant = pred[] constant(true)
  ROOT nop_return_token = token[] custom-call(constant), backend_config="{error_msg = \"1\"}", custom_call_target="__xla_gpu_assert", custom_call_has_side_effect=true, api_version=API_VERSION_TYPED_FFI
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  // The parameter of the NopReturnToken is not removed.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);
  // Can run.
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

TEST_F(RuntimeIntrinsicsTest, AssertionCustomCallFalse) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  constant = pred[] constant(false)
  ROOT nop_return_token = token[] custom-call(constant), backend_config="{error_msg = \"1\"}", custom_call_target="__xla_gpu_assert", custom_call_has_side_effect=true, api_version=API_VERSION_TYPED_FFI
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  // The parameter of the NopReturnToken is not removed.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);
  // Can run.
  EXPECT_FALSE(Run(std::move(module), /*run_hlo_passes=*/false));
}

TEST_F(RuntimeIntrinsicsTest, DebugPrintCustomCallFailsWhenFormatIsMissing) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  constant = f32[2]{0} constant({1, 2})
  ROOT print_token = token[] custom-call(constant),
    backend_config="{format = \"test format\"}",
    custom_call_target="__xla_gpu_debug_print",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_TYPED_FFI
})";

  ::testing::AssertionResult result = Run(kHloText, /*run_hlo_passes=*/false);
  EXPECT_FALSE(result);
  EXPECT_THAT(result.message(), HasSubstr("Missing formatter for argument 0"));
}

TEST_F(RuntimeIntrinsicsTest, DebugPrintCustomCallWithCorrectLogsAsInfo) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  constant = f32[2]{0} constant({1, 2})
  constant2 = f16[3]{0} constant({3, 4, 5})
  ROOT print_token = token[] custom-call(constant, constant2),
    backend_config="{format = \"test format $0 $1\"}",
    custom_call_target="__xla_gpu_debug_print",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_TYPED_FFI
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  // The parameters of the custom call are not removed.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log,
              Log(absl::LogSeverity::kInfo, EndsWith("runtime_intrinsics.cc"),
                  HasSubstr("test format f32[2] {1, 2} f16[3] {3, 4, 5}")));
  // Run the program once before starting capturing the locks. This works around
  // a deadlock caused by ScopedMockLog.
  std::unique_ptr<HloModule> module2 = module->Clone();
  EXPECT_TRUE(Run(std::move(module2), /*run_hlo_passes=*/false));
  mock_log.StartCapturingLogs();
  // Runs successfully and logs the expected info.
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  mock_log.StopCapturingLogs();
}

}  // namespace
}  // namespace gpu
}  // namespace xla
