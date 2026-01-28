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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/io/record_reader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "tsl/platform/path.h"
#include "tsl/platform/tstring.h"

namespace xla {
namespace gpu {
namespace {

class RuntimeIntrinsicsTest : public HloPjRtTestBase {};

using ::testing::EndsWith;
using ::testing::HasSubstr;

absl::StatusOr<std::vector<std::pair<std::string, Literal>>>
ReadTFRecordIOLiteral(const std::string& dir) {
  auto* env = tsl::Env::Default();

  std::vector<std::string> files;
  TF_RETURN_IF_ERROR(env->GetChildren(dir, &files));

  std::vector<std::pair<std::string, Literal>> result;
  for (const std::string& path : files) {
    std::unique_ptr<tsl::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->NewRandomAccessFile(
        tsl::io::JoinPath(dir, path), &file));
    tsl::io::RecordReader reader(file.get());

    uint64_t offset = 0;
    tsl::tstring record;

    for (;;) {
      tsl::tstring metadata;
      absl::Status status = reader.ReadRecord(&offset, &metadata);
      if (absl::IsOutOfRange(status)) {
        break;
      }
      TF_RETURN_IF_ERROR(status);

      TF_RETURN_IF_ERROR(reader.ReadRecord(&offset, &record));
      TF_ASSIGN_OR_RETURN(Literal literal,
                          Literal::DeserializeFromString(record));
      result.emplace_back(metadata, std::move(literal));
    }
  }
  return result;
}

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

TEST_F(RuntimeIntrinsicsTest, AppendToFile) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());

  std::string hlo = absl::StrFormat(R"hlo(
HloModule m

ENTRY e {
  constant = f32[2]{0} constant({1, 2})
  ROOT token1 = token[] custom-call(constant),
    backend_config="{dir = \"%1$s\", metadata = \"op.1\"}",
    custom_call_target="__xla_gpu_append_to_file",
    custom_call_has_side_effect=true,
    api_version=API_VERSION_TYPED_FFI
})hlo",
                                    temp_dir.path());

  Literal expected = LiteralUtil::CreateR1<float>({1.0f, 2.0f});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));

  std::vector<std::pair<std::string, Literal>> literals;
  TF_ASSERT_OK_AND_ASSIGN(literals, ReadTFRecordIOLiteral(temp_dir.path()));
  EXPECT_EQ(literals.size(), 1);
  EXPECT_EQ(literals[0].first, "op.1");
  EXPECT_EQ(literals[0].second, expected);

  // Verify that append works.
  TF_ASSERT_OK_AND_ASSIGN(module, GetOptimizedModule(hlo));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  TF_ASSERT_OK_AND_ASSIGN(literals, ReadTFRecordIOLiteral(temp_dir.path()));
  EXPECT_EQ(literals.size(), 2);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
