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

#include "xla/service/gpu/autotuner_util.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_results.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"   // IWYU pragma: keep
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::TempDir;
using ::tsl::testing::StatusIs;

class AutotunerUtilTest : public HloTestBase {
 protected:
  static constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  std::string GetUniqueTempFilePath(absl::string_view suffix) {
    std::string filename = TempDir();
    CHECK(tsl::Env::Default()->CreateUniqueFileName(&filename,
                                                    std::string(suffix)));
    return filename;
  }

  std::string ExpectToReadNonEmptyFile(absl::string_view file_path) {
    std::string str;
    tsl::Env* env = tsl::Env::Default();
    TF_EXPECT_OK(tsl::ReadFileToString(env, std::string(file_path), &str));
    EXPECT_THAT(str, Not(IsEmpty()));
    return str;
  }

  std::unique_ptr<stream_executor::StreamExecutor> NewStreamExecutor() {
    stream_executor::Platform* platform =
        stream_executor::PlatformManager::PlatformWithName("Host").value();
    stream_executor::StreamExecutorConfig config(/*ordinal=*/0);
    return platform->GetUncachedExecutor(config).value();
  }
};

TEST_F(AutotunerUtilTest, SerializeAutotuneResultsToFile_TextProto1) {
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());

  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
}

TEST_F(AutotunerUtilTest, SerializeAutotuneResultsToFile_TextProto2) {
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());

  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
}

TEST_F(AutotunerUtilTest, SerializeAutotuneResultsToFile_Protobuf) {
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());

  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  std::string autotune_results_str = ExpectToReadNonEmptyFile(kFilePath);
  AutotuneResults results;
  EXPECT_TRUE(results.ParseFromString(autotune_results_str));
}

TEST_F(AutotunerUtilTest, LoadAutotuneResultsFromFile_TextProto1) {
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath));
}

TEST_F(AutotunerUtilTest, LoadAutotuneResultsFromFile_TextProto2) {
  std::string kFilePath = GetUniqueTempFilePath(".textproto");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath));
}

TEST_F(AutotunerUtilTest, LoadAutotuneResultsFromFile_Protobuf) {
  std::string kFilePath = GetUniqueTempFilePath(".pb");
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  TF_EXPECT_OK(AutotunerUtil::SerializeAutotuneResultsToFile(kFilePath));

  TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(kFilePath));
}

// Test that when complete AOT autotuning is required, and there is cache miss,
// a `NotFound` error will be raised.
TEST_F(AutotunerUtilTest, FailIfRequireCompleteAotAutotuning) {
  std::string kFilePath = GetUniqueTempFilePath(".txt");
  auto hlo_module = GetOptimizedModule(kHloText);
  TF_EXPECT_OK(hlo_module.status());
  std::vector<HloComputation*> computations =
      (*hlo_module)
          ->MakeNonfusionComputations(absl::flat_hash_set<absl::string_view>());
  EXPECT_THAT(computations, Not(IsEmpty()));
  const HloInstruction* instruction = *computations[0]->instructions().begin();
  std::unique_ptr<stream_executor::StreamExecutor> executor =
      NewStreamExecutor();
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);
  AutotuneConfig config(DeviceConfig{executor.get()}, options);
  EXPECT_THAT(
      AutotunerUtil::Autotune(instruction, config,
                              [&] { return AutotuneResult(); }),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr("Complete XLA AOT autotuning results are required, but "
                    "no AOT result was found for key: <key model")));
}

// Test that when JIT autotuning is disabled, but no cache miss due to AOT
// autotuning, `Autotune` still returns Ok status.
TEST_F(AutotunerUtilTest, OkIfJitAutotuningDisabledButAlreadyLoadedAOT) {
  auto hlo_module = GetOptimizedModule(kHloText);
  std::vector<HloComputation*> computations =
      (*hlo_module)
          ->MakeNonfusionComputations(absl::flat_hash_set<absl::string_view>());
  EXPECT_THAT(computations, Not(IsEmpty()));
  const HloInstruction* instruction = *computations[0]->instructions().begin();
  std::unique_ptr<stream_executor::StreamExecutor> executor =
      NewStreamExecutor();

  {
    // By default, JIT autotuning is OK.
    AutotuneConfig config(DeviceConfig{executor.get()}, DebugOptions());
    TF_EXPECT_OK(AutotunerUtil::Autotune(instruction, config, [&] {
                   return AutotuneResult();
                 }).status());
  }

  // Now require complete AOT autotuning results.
  auto options = DebugOptions();
  options.set_xla_gpu_require_complete_aot_autotune_results(true);

  AutotuneConfig config(DeviceConfig{executor.get()}, options);
  // Even though JIT autotuning is disabled, there is no cache miss when running
  // autotuning for the same entry, so no error should be raised either.
  TF_EXPECT_OK(AutotunerUtil::Autotune(instruction, config, [&] {
                 return AutotuneResult();
               }).status());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
