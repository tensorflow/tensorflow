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
#include <optional>
#include <string>
#include <utility>

#include "google/protobuf/duration.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/cpu_symbol_repository.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tools/xla_compile_lib.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "xla/util.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Not;

class XlaCompileLibTest : public HloTestBase {
 protected:
  void SetUp() override {
    const std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                                                   "tools", "data", "add.hlo");
    std::string hlo;
    ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo));
    ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));
  }

  std::unique_ptr<HloModule> module_;
};

TEST_F(XlaCompileLibTest, CompilesForCpu) {
  CompilationResult result;
  EXPECT_THAT(
      CompileExecutable(std::move(module_), BackendType::kCpu,
                        /*gpu_target_config=*/std::nullopt,
                        /*cpu_target_config=*/std::nullopt,
                        /*num_partitions=*/1, /*num_replicas=*/1, result),
      absl_testing::IsOkAndHolds(Not(IsEmpty())));
}

TEST_F(XlaCompileLibTest, ErrorsOnUnexpectedPlatform) {
  XlaCompileOptions options;
  options.platform = "tpu";
  EXPECT_THAT(XlaCompileMain(options),
              absl_testing::StatusIs(tsl::error::UNIMPLEMENTED));
}

TEST_F(XlaCompileLibTest, WriteResultFilePropagatesErrors) {
  TimerStats stats;
  CompilationResult result;
  EXPECT_THAT(WriteResultFile("/does/not/exist", stats, result),
              Not(absl_testing::IsOk()));
}

TEST_F(XlaCompileLibTest, WriteResultFileWritesTheFile) {
  std::string result_output_file;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&result_output_file));

  TimerStats stats;
  {
    absl::MutexLock ml(stats.stats_mutex);
    stats.cumulative_secs = 5.5;
    stats.max_secs = 5.5;
  }

  CompilationResult result;
  google::protobuf::Duration duration;
  duration.set_seconds(5);
  duration.set_nanos(0.5 * tsl::EnvTime::kSecondsToNanos);
  *result.mutable_perf_stats()->mutable_compilation_duration() = duration;
  *result.mutable_perf_stats()->mutable_total_duration() = duration;

  ASSERT_OK(WriteResultFile(result_output_file, stats, result));

  CompilationResult got_result;
  ASSERT_OK(tsl::ReadBinaryProto(tsl::Env::Default(), result_output_file,
                                 &got_result));
  // Sadly EqualsProto isn't OSS, so we inspect a few fields manually.
  // See googletest#1761 and b/229726259.
  EXPECT_EQ(5, got_result.perf_stats().compilation_duration().seconds());
  EXPECT_EQ(0.5 * tsl::EnvTime::kSecondsToNanos,
            got_result.perf_stats().compilation_duration().nanos());
  EXPECT_EQ(5, got_result.perf_stats().total_duration().seconds());
  EXPECT_EQ(0.5 * tsl::EnvTime::kSecondsToNanos,
            got_result.perf_stats().total_duration().nanos());
}

TEST_F(XlaCompileLibTest, LoadModuleErrors) {
  EXPECT_THAT(LoadModule("/does/not/exist"), Not(absl_testing::IsOk()));
}

TEST_F(XlaCompileLibTest, ErrorsOnMissingOutputPaths) {
  XlaCompileOptions options;
  options.platform = "gpu";
  EXPECT_THAT(XlaCompileMain(options),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(XlaCompileLibTest, LoadModuleLoadsTextFormat) {
  const std::string module_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "module.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), module_file,
                                   module_->ToString()));

  EXPECT_THAT(LoadModule(module_file),
              absl_testing::IsOkAndHolds(Not(IsNull())));
}

TEST_F(XlaCompileLibTest, MainForCpu) {
  const std::string module_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "module.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), module_file,
                                   module_->ToString()));

  const std::string output_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "cpu_output");
  const std::string result_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "cpu_result.pb");

  XlaCompileOptions options;
  options.module_path = module_file;
  options.output_file = output_file;
  options.platform = "cpu";
  options.result_output_file = result_file;
  EXPECT_OK(XlaCompileMain(options));

  CompilationResult result;
  ASSERT_OK(tsl::ReadBinaryProto(tsl::Env::Default(), result_file, &result));
  EXPECT_TRUE(result.has_status());
  EXPECT_EQ(result.status().code(), tensorflow::error::OK);
  EXPECT_TRUE(result.has_hlo_module());
}

TEST_F(XlaCompileLibTest, MainForCpuWithTargetConfigPath) {
  const std::string module_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "module_target_config.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), module_file,
                                   module_->ToString()));

  std::string target_config_text;
  tsl::protobuf::TextFormat::PrintToString(
      cpu::TargetMachineOptions().ToProto(), &target_config_text);

  const std::string target_config_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "target_config.pbtxt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), target_config_file,
                                   target_config_text));

  const std::string output_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "cpu_output_config");
  const std::string result_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "cpu_result_config.pb");

  XlaCompileOptions options;
  options.module_path = module_file;
  options.output_file = output_file;
  options.platform = "cpu";
  options.result_output_file = result_file;
  options.cpu_options.cpu_target_config_path = target_config_file;
  EXPECT_OK(XlaCompileMain(options));

  CompilationResult result;
  ASSERT_OK(tsl::ReadBinaryProto(tsl::Env::Default(), result_file, &result));
  EXPECT_TRUE(result.has_status());
  EXPECT_EQ(result.status().code(), tensorflow::error::OK);
  EXPECT_TRUE(result.has_hlo_module());
}

namespace {

class FakeCpuSymbolRepository : public SymbolRepository {
 public:
  explicit FakeCpuSymbolRepository(
      std::unique_ptr<HloModule> module,
      std::optional<cpu::TargetMachineOptions> options = std::nullopt)
      : module_(std::move(module)), options_(std::move(options)) {}

  absl::StatusOr<std::unique_ptr<HloModuleAndMetadata>> Lookup(
      absl::string_view symbol_reference, BackendType backend) const override {
    if (symbol_reference != "valid_cpu_symbol") {
      return absl::NotFoundError("not found");
    }
    auto result = std::make_unique<HloModuleAndMetadata>();
    result->hlo_module = module_->Clone();
    if (backend == BackendType::kCpu) {
      auto data = std::make_unique<cpu::CpuBackendSpecificData>();
      data->target_machine_options = options_;
      result->backend_specific_data = std::move(data);
    }
    return result;
  }

 private:
  std::unique_ptr<HloModule> module_;
  std::optional<cpu::TargetMachineOptions> options_;
};

}  // namespace

TEST_F(XlaCompileLibTest, MainForCpuWithSymbolRepo) {
  auto target_options = cpu::TargetMachineOptions();
  GetGlobalSymbolRepositoryRegistry().Register(
      "test_cpu_repo", std::make_unique<FakeCpuSymbolRepository>(
                           module_->Clone(), target_options));

  const std::string output_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "cpu_repo_output");
  const std::string result_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "cpu_repo_result.pb");

  XlaCompileOptions options;
  options.repo_options.symbol_repo = "test_cpu_repo";
  options.repo_options.symbol_id = "valid_cpu_symbol";
  options.output_file = output_file;
  options.platform = "cpu";
  options.result_output_file = result_file;
  EXPECT_OK(XlaCompileMain(options));

  CompilationResult result;
  ASSERT_OK(tsl::ReadBinaryProto(tsl::Env::Default(), result_file, &result));
  EXPECT_TRUE(result.has_status());
  EXPECT_EQ(result.status().code(), tensorflow::error::OK);
  EXPECT_TRUE(result.has_hlo_module());
}

TEST_F(XlaCompileLibTest, LoadAutotuneDataCpu) {
  HloModuleAndMetadata mod;
  mod.hlo_module = std::move(module_);

  EXPECT_THAT(internal::LoadAutotuneDataFromModule(&mod, BackendType::kCpu),
              absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
