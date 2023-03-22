/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/multihost_hlo_runner/functional_hlo_runner.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::SizeIs;

class FunctionalHloRunnerTest : public ::testing::Test {
 protected:
  std::string GetHloPath(std::string file_name) {
    return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                             "multihost_hlo_runner", "data", file_name);
  }
};

TEST_F(FunctionalHloRunnerTest, SingleDeviceHlo) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::FunctionalHloRunner::CreateGpuClient());

  // Options corresponding to --num_replicas=1 --num_partitions=1
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 1;
  FunctionalHloRunner::RunningOptions running_options;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, preproc_options, raw_compile_options, running_options,
      {GetHloPath("single_device.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, Sharded2Devices) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::FunctionalHloRunner::CreateGpuClient());

  constexpr int kRequiredDeviceCount = 2;
  const int kDeviceCount = client->device_count();
  if (kDeviceCount < kRequiredDeviceCount) {
    GTEST_SKIP() << "Requires " << kRequiredDeviceCount
                 << " devices, but found only " << kDeviceCount;
    return;
  }

  // Options corresponding to:
  // --use_spmd_partitioning=true --num_replicas=1 --num_partitions=2
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 2;
  FunctionalHloRunner::RunningOptions running_options;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, preproc_options, raw_compile_options, running_options,
      {GetHloPath("sharded_2_devices.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, UseZerosAsInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::FunctionalHloRunner::CreateGpuClient());

  constexpr int kRequiredDeviceCount = 2;
  const int kDeviceCount = client->device_count();
  if (kDeviceCount < kRequiredDeviceCount) {
    GTEST_SKIP() << "Requires " << kRequiredDeviceCount
                 << " devices, but found only " << kDeviceCount;
    return;
  }

  // Options corresponding to:
  // --use_spmd_partitioning=true --num_replicas=1 --num_partitions=2
  // --hlo_argument_mode=use_zeros_as_input
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 2;
  FunctionalHloRunner::RunningOptions running_options;
  running_options.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUseZerosAsInput;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, preproc_options, raw_compile_options, running_options,
      {GetHloPath("sharded_2_devices.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, UseUninitializedInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::FunctionalHloRunner::CreateGpuClient());

  constexpr int kRequiredDeviceCount = 2;
  const int kDeviceCount = client->device_count();
  if (kDeviceCount < kRequiredDeviceCount) {
    GTEST_SKIP() << "Requires " << kRequiredDeviceCount
                 << " devices, but found only " << kDeviceCount;
    return;
  }

  // Options corresponding to:
  // --use_spmd_partitioning=true --num_replicas=1 --num_partitions=2
  // --hlo_argument_mode=uninitialized
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 2;
  FunctionalHloRunner::RunningOptions running_options;
  running_options.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUninitialized;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, preproc_options, raw_compile_options, running_options,
      {GetHloPath("sharded_2_devices.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, UseUninitializedInputsWithTupledArguments) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::FunctionalHloRunner::CreateGpuClient());

  // Options corresponding to:
  // --num_replicas=1 --num_partitions=1
  // --hlo_argument_mode=uninitialized
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 1;
  FunctionalHloRunner::RunningOptions running_options;
  running_options.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUninitialized;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, preproc_options, raw_compile_options, running_options,
      {GetHloPath("single_device_tupled.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, CanCompileWithoutHavingEnoughGpus) {
  // This test corresponds to:
  // --use_spmd_partitioning=true --num_replicas=1 --num_partitions=16
  // --run=false --xla_dump_to=dump_dir

  tsl::Env* env = tsl::Env::Default();
  std::string dump_dir;
  ASSERT_TRUE(env->LocalTempFilename(&dump_dir));
  tsl::FileSystem* fs = nullptr;
  TF_ASSERT_OK(env->GetFileSystemForFile(dump_dir, &fs));

  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 16;
  raw_compile_options.xla_dump_to = dump_dir;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::FunctionalHloRunner::CreateGpuClient());
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndCompile(
      *client, preproc_options, raw_compile_options,
      GetHloPath("sharded_16_devices.hlo"), InputFormat::kText));

  // Check that the sharding was done correctly.
  {
    std::vector<std::string> after_opt_hlo_paths;
    TF_ASSERT_OK(
        fs->GetMatchingPaths(fs->JoinPath(dump_dir, "*after_optimizations.txt"),
                             &after_opt_hlo_paths));
    ASSERT_THAT(after_opt_hlo_paths, SizeIs(1));
    std::string after_opt_hlo;
    TF_ASSERT_OK(
        tsl::ReadFileToString(env, after_opt_hlo_paths[0], &after_opt_hlo));
    StatusOr<bool> file_check_result = RunFileCheck(after_opt_hlo, R"(
      // CHECK: param = f32[16,1]{1,0}
      // CHECK: add = f32[16,1]{1,0}
    )");
    TF_ASSERT_OK(file_check_result.status());
    EXPECT_TRUE(file_check_result.value());
  }

  // Check that the LLVM IR has been generated.
  {
    std::vector<std::string> ir_paths;
    TF_ASSERT_OK(fs->GetMatchingPaths(fs->JoinPath(dump_dir, "*ir-no-opt.ll"),
                                      &ir_paths));
    ASSERT_THAT(ir_paths, SizeIs(1));
  }
}

}  // namespace
}  // namespace xla
