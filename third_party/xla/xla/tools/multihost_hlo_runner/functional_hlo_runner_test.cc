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

#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/statusor.h"
#include "xla/tests/filecheck.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/subprocess.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::SizeIs;

bool IsTestingCpu() {
#ifdef XLA_TEST_BACKEND_CPU
  return true;
#endif
  return false;
}

std::string GetHloPath(std::string file_name) {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                           "multihost_hlo_runner", "data", file_name);
}

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetPjRtClient() {
  if (IsTestingCpu()) {
    return xla::FunctionalHloRunner::CreateHostClient();
  }
  return xla::FunctionalHloRunner::CreateGpuClient({});
}

using FunctionalHloRunnerTest = ::testing::Test;

TEST_F(FunctionalHloRunnerTest, SingleDeviceHlo) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  // Options corresponding to --num_replicas=1 --num_partitions=1
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 1;
  FunctionalHloRunner::RunningOptions running_options;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("single_device.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, Sharded2Devices) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  constexpr int kRequiredDeviceCount = 2;
  const int kDeviceCount = client->device_count();
  if (kDeviceCount < kRequiredDeviceCount) {
    GTEST_SKIP() << "Requires " << kRequiredDeviceCount
                 << " devices, but found only " << kDeviceCount;
    return;
  }

  // Options corresponding to:
  // --use_spmd_partitioning=true --num_replicas=1 --num_partitions=2
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 2;
  FunctionalHloRunner::RunningOptions running_options;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("sharded_2_devices.hlo")},
      InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, UseZerosAsInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

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
  xla::DebugOptions debug_options;
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
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("sharded_2_devices.hlo")},
      InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, UseUninitializedInputs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

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
  xla::DebugOptions debug_options;
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
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("sharded_2_devices.hlo")},
      InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, UseUninitializedInputsWithTupledArguments) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  // Options corresponding to:
  // --num_replicas=1 --num_partitions=1
  // --hlo_argument_mode=uninitialized
  xla::DebugOptions debug_options;
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
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("single_device_tupled.hlo")},
      InputFormat::kText));
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

  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 16;
  raw_compile_options.xla_dump_to = dump_dir;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndCompile(
      *client, debug_options, preproc_options, raw_compile_options,
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
    absl::StatusOr<bool> file_check_result = RunFileCheck(after_opt_hlo, R"(
      // CHECK: param{{.*}} = f32[16,1]{1,0}
      // CHECK: add{{.*}} = f32[16,1]{1,0}
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

// Name of the test binary.
static const char* binary_name;
constexpr int kNumNodes = 3;

TEST_F(FunctionalHloRunnerTest, ShardedAutotuningWorks) {
  if (IsTestingCpu()) {
    GTEST_SKIP() << "GPU-only test.";
  }

  tsl::SubProcess child[kNumNodes];
  for (int node_id = 0; node_id < kNumNodes; ++node_id) {
    std::vector<std::string> argv;
    argv.push_back(binary_name);
    argv.push_back("--xla_gpu_shard_autotuning");
    argv.push_back(absl::StrFormat("--node_id=%d", node_id));
    child[node_id].SetProgram(binary_name, argv);
    child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[node_id].Start()) << "node " << node_id;
  }
  for (int node_id = 0; node_id < kNumNodes; ++node_id) {
    std::string stdout_str;
    std::string stderr_str;
    int child_status =
        child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);
    ASSERT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
}

absl::Status ShardedAutotuningWorksTestBody(const int node_id) {
  tsl::setenv("CUDA_VISIBLE_DEVICES", std::to_string(node_id).data(),
              /*overwrite=*/true);
  std::unique_ptr<xla::DistributedRuntimeService> service = nullptr;
  std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      GetPjRtClient("gpu", "127.0.0.1:12345", node_id,
                                    kNumNodes, false, service, kv_store));
  CHECK(kv_store != nullptr);
  TF_RETURN_IF_ERROR(FunctionalHloRunner::LoadAndCompile(
      *client, GetDebugOptionsFromFlags(),
      FunctionalHloRunner::PreprocessingOptions{},
      FunctionalHloRunner::RawCompileOptions{},
      GetHloPath("multiple_gemm_fusions.hlo"), InputFormat::kText));
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(
        std::string results0,
        kv_store->Get("gemm_fusion_autotuning_results_1_0", absl::Seconds(1)));
    CHECK(absl::StrContains(results0, "run_time"));
    TF_ASSIGN_OR_RETURN(
        std::string results1,
        kv_store->Get("gemm_fusion_autotuning_results_1_1", absl::Seconds(1)));
    CHECK(absl::StrContains(results1, "run_time"));
    // First two nodes autotune two different fusions.
    CHECK_NE(results0, results1);
    TF_ASSIGN_OR_RETURN(
        std::string results2,
        kv_store->Get("gemm_fusion_autotuning_results_1_2", absl::Seconds(1)));
    // Third node has nothing to autotune.
    CHECK(!absl::StrContains(results2, "run_time"));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::binary_name = argv[0];
  int node_id = -1;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("node_id", &node_id,
                "Node ID for ShardedAutotuningWorks test."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  if (node_id >= 0) {
    return !xla::ShardedAutotuningWorksTestBody(node_id).ok();
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
