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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::HasSubstr;

class OutputBufferDebugTest : public HloTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> GetModule() {
    constexpr absl::string_view kHlo = R"hlo(
HloModule Dynamic_Select_NaN

ENTRY main (p0: pred[]) -> f32[] {
  %p0 = pred[] parameter(0)
  %zero = f32[] constant(0.0)
  %nan = f32[] constant(nan)
  ROOT %result = f32[] select(%p0, %nan, %zero)
}
)hlo";
    ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(kHlo));
    // Disable fusion pipeline to prevent fusing the select with nan constant.
    // The tests need to distinguish reports from the intermediate thunk
    // (select) and the output.
    module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
        "fusion");
    return module;
  }

  void Run(std::unique_ptr<HloModule> module, bool emit_nan_output) {
    Literal cond = LiteralUtil::CreateR0<bool>(emit_nan_output);
    std::vector<const Literal*> args = {&cond};
    auto result = Execute(std::move(module), args);
    EXPECT_OK(result.status());
  }
};

TEST_F(OutputBufferDebugTest,
       IntermediateNanNotReportedWhenModuleOutputsChecked) {
  ASSERT_OK_AND_ASSIGN(auto module, GetModule());
  module->mutable_config().mutable_debug_options().set_xla_gpu_detect_nan(
      DebugOptions::DETECTION_MODE_WARNING);
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_thunk_buffer_debug_module_outputs(true);

  absl::ScopedMockLog log;
  // Module outputs checked, but no other filters set up => intermediate %nan
  // should not be reported
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("found NaN")))
      .Times(0);
  log.StartCapturingLogs();
  Run(std::move(module), /*emit_nan_output=*/false);
  log.StopCapturingLogs();
}

TEST_F(OutputBufferDebugTest, NanInOutputReportedWhenModuleOutputsChecked) {
  ASSERT_OK_AND_ASSIGN(auto module, GetModule());
  module->mutable_config().mutable_debug_options().set_xla_gpu_detect_nan(
      DebugOptions::DETECTION_MODE_WARNING);
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_thunk_buffer_debug_module_outputs(true);

  absl::ScopedMockLog log;
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  AllOf(HasSubstr("found NaN"), HasSubstr("Output Check"))));
  log.StartCapturingLogs();
  Run(std::move(module), /*emit_nan_output=*/true);
  log.StopCapturingLogs();
}

TEST_F(OutputBufferDebugTest,
       IntermediateNanReportedByDefaultWhenNanCheckEnabled) {
  ASSERT_OK_AND_ASSIGN(auto module, GetModule());
  module->mutable_config().mutable_debug_options().set_xla_gpu_detect_nan(
      DebugOptions::DETECTION_MODE_WARNING);

  absl::ScopedMockLog log;
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kError, _,
               AllOf(HasSubstr("found NaN"), Not(HasSubstr("Output Check")))));
  log.StartCapturingLogs();
  Run(std::move(module), /*emit_nan_output=*/false);
  log.StopCapturingLogs();
}

TEST_F(OutputBufferDebugTest,
       IntermediateNanReportedWhenFilterMatchesAndOutputsChecked) {
  ASSERT_OK_AND_ASSIGN(auto module, GetModule());
  module->mutable_config().mutable_debug_options().set_xla_gpu_detect_nan(
      DebugOptions::DETECTION_MODE_WARNING);
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_thunk_buffer_debug_module_outputs(true);
  module->mutable_config()
      .mutable_debug_options()
      .mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
      ->add_profile_annotation_regexes(".*");

  absl::ScopedMockLog log;
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kError, _,
               AllOf(HasSubstr("found NaN"), Not(HasSubstr("Output Check")))));
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _,
                  AllOf(HasSubstr("found NaN"), HasSubstr("Output Check"))));
  log.StartCapturingLogs();
  Run(std::move(module), /*emit_nan_output=*/true);
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace xla::gpu
