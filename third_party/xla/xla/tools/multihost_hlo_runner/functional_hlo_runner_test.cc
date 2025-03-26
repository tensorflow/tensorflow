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

#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/runtime/large_hlo_snapshot_serialization/serialization.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/file_system_helper.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using ::testing::SizeIs;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;
using HloModuleAndArguments = ::xla::FunctionalHloRunner::HloModuleAndArguments;

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
    return CreateHostClient();
  }
  return CreateGpuClient({});
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

TEST_F(FunctionalHloRunnerTest, SingleDeviceHloThroughStableHlo) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  preproc_options.compile_as_stablehlo = true;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 1;
  FunctionalHloRunner::RunningOptions running_options;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("single_device.hlo")}, InputFormat::kText));
}

TEST_F(FunctionalHloRunnerTest, SingleDeviceHloWithExecutionProfile) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());
  std::vector<ExecutionProfile> profiles;
  FunctionalHloRunner::RunningOptions running_options;
  running_options.num_repeats = 2;
  running_options.execution_profiles = &profiles;
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client,
      /* debug_options= */ {}, /* preproc_options= */ {},
      /* raw_compile_options = */ {}, running_options,
      {GetHloPath("single_device.hlo")}, InputFormat::kText));
  ASSERT_EQ(profiles.size(), 2);
  if (client->platform_name() == "cuda") {
    // CPU backend does not fill the profile at the moment.
    EXPECT_GT(profiles[0].compute_time_ns(), 0);
    EXPECT_GT(profiles[1].compute_time_ns(), 0);
  }
}

TEST_F(FunctionalHloRunnerTest, GPUProfilerWithEmptyDumpPathReturnsError) {
  if (IsTestingCpu()) {
    GTEST_SKIP() << "GPU-only test";
  }
  std::string empty_profile_dump_path = "";
  EXPECT_THAT(
      GPURunnerProfiler::Create(empty_profile_dump_path, /*keep_xspace=*/true),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FunctionalHloRunnerTest, GPUProfilerKeepXSpaceReturnsNonNullXSpace) {
  if (IsTestingCpu()) {
    GTEST_SKIP() << "GPU-only test";
  }
  std::string profile_dump_path =
      tsl::io::JoinPath(testing::TempDir(), "xspace.pb");
  tsl::Env* env = tsl::Env::Default();
  tsl::FileSystem* fs = nullptr;
  TF_ASSERT_OK(env->GetFileSystemForFile(profile_dump_path, &fs));

  FunctionalHloRunner::RunningOptions running_options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto profiler,
      GPURunnerProfiler::Create(profile_dump_path, /*keep_xspace=*/true));
  running_options.profiler = profiler.get();

  profiler->CreateSession();
  profiler->UploadSession();
  EXPECT_NE(profiler->GetXSpace(), nullptr);
  EXPECT_GT(profiler->GetXSpace()->planes_size(), 0);
  TF_EXPECT_OK(env->FileExists(profile_dump_path));
}

TEST_F(FunctionalHloRunnerTest,
       SingleDeviceHloWithGPUProfilerSavesXSpaceToDisk) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());
  std::string profile_dump_path =
      tsl::io::JoinPath(testing::TempDir(), "xspace.pb");

  std::unique_ptr<GPURunnerProfiler> profiler;
  FunctionalHloRunner::RunningOptions running_options;
  TF_ASSERT_OK_AND_ASSIGN(
      profiler,
      GPURunnerProfiler::Create(profile_dump_path, /*keep_xspace=*/true));
  running_options.profiler = profiler.get();

  running_options.num_repeats = 2;
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client,
      /* debug_options= */ {}, /* preproc_options= */ {},
      /* raw_compile_options = */ {}, running_options,
      {GetHloPath("single_device.hlo")}, InputFormat::kText));

  if (client->platform_name() == "cuda") {
    EXPECT_NE(profiler->GetXSpace(), nullptr);
    EXPECT_GT(profiler->GetXSpace()->planes_size(), 0);
  }
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
  // --num_replicas=1 --num_partitions=2
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
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
  // --num_replicas=1 --num_partitions=2
  // --hlo_argument_mode=use_zeros_as_input
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
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
  // --num_replicas=1 --num_partitions=2
  // --hlo_argument_mode=uninitialized
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
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

// ROCM Error:
// E0000 00:00:1737155629.780742  137227 pjrt_stream_executor_client.cc:3045]
// Execution of replica 0 failed: INTERNAL: Failed to end stream capture:
// hipError_t(901)
TEST_F(FunctionalHloRunnerTest, ShardedComputationUnderStreamCapture) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  constexpr int kRequiredDeviceCount = 2;
  const int kDeviceCount = client->device_count();
  if (kDeviceCount < kRequiredDeviceCount) {
    GTEST_SKIP() << "Requires " << kRequiredDeviceCount
                 << " devices, but found only " << kDeviceCount;
    return;
  }

  // NOTE: debug_options sent to FunctionalHloRunner::LoadAndRunAndDump() get
  // lost during the creating of XlaComputation from HloModuleProto in
  // FunctionalHloRunner::Compile
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 2;
  FunctionalHloRunner::RunningOptions running_options;
  running_options.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUseRandomInputs;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("sharded_computation.hlo")},
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

void CompileAndFilecheck(
    absl::string_view hlo_file, absl::string_view pattern,
    const FunctionalHloRunner::PreprocessingOptions& preproc_options,
    const FunctionalHloRunner::HloPassesMode hlo_passes_mode,
    const int num_partitions = 1) {
  tsl::Env* env = tsl::Env::Default();
  std::string dump_dir;
  ASSERT_TRUE(env->LocalTempFilename(&dump_dir));
  tsl::FileSystem* fs = nullptr;
  TF_ASSERT_OK(env->GetFileSystemForFile(dump_dir, &fs));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());
  FunctionalHloRunner::RawCompileOptions opts;
  opts.hlo_passes_mode = hlo_passes_mode;
  opts.num_partitions = num_partitions;
  opts.xla_dump_to = dump_dir;
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndCompile(
      *client, xla::DebugOptions{}, preproc_options, opts, hlo_file,
      InputFormat::kText));

  {
    std::vector<std::string> after_opt_hlo_paths;
    TF_ASSERT_OK(
        fs->GetMatchingPaths(fs->JoinPath(dump_dir, "*after_optimizations.txt"),
                             &after_opt_hlo_paths));
    ASSERT_THAT(after_opt_hlo_paths, SizeIs(1));
    std::string after_opt_hlo;
    TF_ASSERT_OK(
        tsl::ReadFileToString(env, after_opt_hlo_paths[0], &after_opt_hlo));
    EXPECT_THAT(RunFileCheck(after_opt_hlo, pattern), IsOkAndHolds(true));
  }

  // Check that the LLVM IR has been generated.
  if (!IsTestingCpu()) {
    std::vector<std::string> ir_paths;
    TF_ASSERT_OK(fs->GetMatchingPaths(fs->JoinPath(dump_dir, "*ir-no-opt.ll"),
                                      &ir_paths));
    ASSERT_THAT(ir_paths, SizeIs(1));
  }
}

TEST_F(FunctionalHloRunnerTest, CanCompileWithoutHavingEnoughGpus) {
  CompileAndFilecheck(GetHloPath("sharded_16_devices.hlo"),
                      // Check that the sharding was done correctly.
                      R"(
      // CHECK: param{{.*}} = f32[16,1]{1,0}
      // CHECK: add{{.*}} = f32[16,1]{1,0}
    )",
                      /*preproc_options=*/{},
                      FunctionalHloRunner::HloPassesMode::kStandardCompile,
                      /*num_partitions=*/16);
}

TEST_F(FunctionalHloRunnerTest, WhileKnownTripCountGetsCapped) {
  FunctionalHloRunner::PreprocessingOptions opts;
  opts.while_execution_count = 5;
  opts.annotate_while_loop_trip_count = true;
  CompileAndFilecheck(GetHloPath("while_with_known_trip_count.hlo"),
                      R"(
      // CHECK: constant(5)
      // CHECK: "known_trip_count":{"n":"5"}
    )",
                      opts,
                      FunctionalHloRunner::HloPassesMode::kRunXLABackendOnly);
}

// Name of the test binary.
static const char* binary_name;
constexpr int kNumNodes = 2;

TEST_F(FunctionalHloRunnerTest, ShardedAutotuningWorks) {
  if (IsTestingCpu()) {
    GTEST_SKIP() << "GPU-only test.";
  }

  tsl::setenv("TF_CPP_VMODULE", "gemm_fusion_autotuner=2", /*overwrite=*/true);
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
    EXPECT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
}

absl::Status ShardedAutotuningWorksTestBody(const int node_id) {
  GpuClientOptions gpu_options;
  gpu_options.node_id = node_id;
  gpu_options.num_nodes = kNumNodes;
  gpu_options.allowed_devices = {node_id};
  TF_ASSIGN_OR_RETURN(
      PjRtEnvironment env,
      xla::GetPjRtEnvironmentForGpu("127.0.0.1:12345", gpu_options,
                                    /*init_timeout=*/absl::Seconds(120)));
  TF_RET_CHECK(env.kv_store != nullptr);
  TF_RET_CHECK(env.client->device_count() == kNumNodes);
  TF_RET_CHECK(env.client->addressable_device_count() == 1);

  // The logic for exchanging autotuning results is tested using mocks in
  // gemm_fusion_autotuner_test.cc. Here, we just check that compilation
  // actually succeeds, and that the autotuner runs correctly ends up storing
  // results for each node in the key-value store.
  TF_RETURN_IF_ERROR(FunctionalHloRunner::LoadAndCompile(
      *env.client, GetDebugOptionsFromFlags(),
      FunctionalHloRunner::PreprocessingOptions{},
      FunctionalHloRunner::RawCompileOptions{.num_replicas = kNumNodes},
      GetHloPath("multiple_gemm_fusions.hlo"), InputFormat::kText, node_id,
      kNumNodes, /*kv_store=*/nullptr,
      /*use_gpu_count_workaround=*/false));
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(
        std::string results0,
        env.kv_store->Get("gemm_fusion_autotuning_results_"
                          "b190aeb9aa0b9e93e4c08d095726f562_"
                          "iuhMRX2JY-YpaUJD3Pw0h3H3HNGWEzN4xA0s9Q3CoK8_0",
                          absl::Seconds(1)));
    CHECK(absl::StrContains(results0, "run_time"));
    TF_ASSIGN_OR_RETURN(
        std::string results1,
        env.kv_store->Get("gemm_fusion_autotuning_results_"
                          "b190aeb9aa0b9e93e4c08d095726f562_"
                          "iuhMRX2JY-YpaUJD3Pw0h3H3HNGWEzN4xA0s9Q3CoK8_1",
                          absl::Seconds(1)));
    CHECK(absl::StrContains(results1, "run_time"));
    // The nodes autotune different fusions.
    CHECK_NE(results0, results1);
  }
  return absl::OkStatus();
}

absl::Status RunShardedHloWithClient(xla::PjRtClient& client) {
  // This method corresponds to:
  // --num_replicas=1 --num_partitions=16
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 16;

  FunctionalHloRunner::RunningOptions running_options;
  running_options.module_argument_mode =
      FunctionalHloRunner::ModuleArgumentMode::kUseZerosAsInput;

  return FunctionalHloRunner::LoadAndRunAndDump(
      client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("sharded_16_devices.hlo")},
      InputFormat::kText);
}

TEST_F(FunctionalHloRunnerTest, PreservesAutoLayout) {
  TF_ASSERT_OK_AND_ASSIGN(
      HloModuleAndArguments hlo_module_and_arguments,
      FunctionalHloRunner::LoadHloModuleAndArguments(
          GetHloPath("single_gemm_fusion.hlo"), InputFormat::kText));
  const ComputationLayout& layout =
      hlo_module_and_arguments.hlo_module->config().entry_computation_layout();
  EXPECT_EQ(layout.parameter_count(), 2);
  EXPECT_FALSE(layout.parameter_layouts()[0].AnyLayoutIsSet());
  EXPECT_TRUE(layout.parameter_layouts()[1].AnyLayoutIsSet());
  EXPECT_FALSE(layout.result_layout().AnyLayoutIsSet());
}

TEST_F(FunctionalHloRunnerTest, CanRunWithMockCollectives) {
  if (IsTestingCpu()) {
    GTEST_SKIP() << "GPU-only test";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          CreateMockGpuClient(16));

  TF_EXPECT_OK(RunShardedHloWithClient(*client));
}

TEST_F(FunctionalHloRunnerTest, CanCreateMockClientInPjRtEnv) {
  // Tests that the GPU options are propagated correctly to initialize a mock
  // client.
  if (IsTestingCpu()) {
    GTEST_SKIP() << "GPU-only test";
  }

  GpuClientOptions gpu_options;
  gpu_options.node_id = 0;
  gpu_options.num_nodes = 16;
  gpu_options.enable_mock_nccl = true;
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtEnvironment env,
      GetPjRtEnvironmentForGpu("", gpu_options, absl::Seconds(120)));

  TF_EXPECT_OK(RunShardedHloWithClient(*env.client));
}

TEST_F(FunctionalHloRunnerTest, Sharded2DevicesHloUnoptimizedSnapshot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  constexpr int kRequiredDeviceCount = 2;
  int device_count = client->device_count();
  if (device_count < kRequiredDeviceCount) {
    GTEST_SKIP() << "Requires " << kRequiredDeviceCount
                 << " devices, but found only " << device_count;
    return;
  }

  // Options corresponding to:
  // --num_replicas=1 --num_partitions=2
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.num_replicas = 1;
  raw_compile_options.num_partitions = 2;
  FunctionalHloRunner::RunningOptions running_options;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client, debug_options, preproc_options, raw_compile_options,
      running_options, {GetHloPath("sharded_unoptimized_hlo_snapshot.pbtxt")},
      InputFormat::kUnoptimizedSnapshotProtoText,
      tsl::io::JoinPath(std::getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
                        "dump.txt")));
  tsl::Env* env = tsl::Env::Default();
  tsl::FileSystem* fs = nullptr;
  TF_ASSERT_OK(env->GetFileSystemForFile(
      std::getenv("TEST_UNDECLARED_OUTPUTS_DIR"), &fs));

  std::vector<std::string> filenames;
  TF_ASSERT_OK(fs->GetMatchingPaths(
      tsl::io::JoinPath(std::getenv("TEST_UNDECLARED_OUTPUTS_DIR"), "dump.*"),
      &filenames));

  // The test is sharded, so we expect two dump files with the same result.
  ASSERT_THAT(filenames, SizeIs(2));
  std::string result;
  std::string exp_result = R"(s64[8,1] {
  {0},
  {2},
  {4},
  {6},
  {8},
  {10},
  {12},
  {14}
})";
  for (const auto& filename : filenames) {
    TF_ASSERT_OK(tsl::ReadFileToString(env, filename, &result));
    CHECK_EQ(result, exp_result);
  }
}

TEST_F(FunctionalHloRunnerTest, ReadHloUnoptimizedSnapshot) {
  std::string path_to_text_hlo =
      GetHloPath("sharded_unoptimized_hlo_snapshot.pbtxt");
  std::string path_to_binary_hlo =
      tsl::io::JoinPath(std::getenv("TEST_UNDECLARED_OUTPUTS_DIR"),
                        "sharded_unoptimized_hlo_snapshot.pb");
  tsl::Env* env = tsl::Env::Default();

  // Read the text proto
  HloUnoptimizedSnapshot message;
  TF_ASSERT_OK(tsl::ReadTextProto(env, path_to_text_hlo, &message));

  // Dump message in the custom binary format
  std::unique_ptr<tsl::WritableFile> file;
  TF_ASSERT_OK(env->NewWritableFile(path_to_binary_hlo, &file));

  tsl::WritableFileCopyingOutputStream output(file.get());

  tsl::protobuf::io::CopyingOutputStreamAdaptor adaptor(&output);
  TF_ASSERT_OK(SerializeHloUnoptimizedSnapshot(message, &adaptor));
  adaptor.Flush();

  TF_ASSERT_OK(file->Close());

  // Read HloModuleAndArguments from text dump.
  TF_ASSERT_OK_AND_ASSIGN(
      HloModuleAndArguments hlo_module_and_arguments_from_text,
      FunctionalHloRunner::LoadHloModuleAndArguments(
          path_to_text_hlo, InputFormat::kUnoptimizedSnapshotProtoText));
  // Read HloModuleAndArguments from binary dump.
  TF_ASSERT_OK_AND_ASSIGN(
      HloModuleAndArguments hlo_module_and_arguments_from_binary,
      FunctionalHloRunner::LoadHloModuleAndArguments(
          path_to_binary_hlo, InputFormat::kUnoptimizedSnapshotProtoBinary));

  // Compare
  CHECK_EQ(hlo_module_and_arguments_from_binary.arguments.size(), 2);

  CHECK_EQ(hlo_module_and_arguments_from_text.hlo_module->ToString(),
           hlo_module_and_arguments_from_binary.hlo_module->ToString());

  CHECK_EQ(hlo_module_and_arguments_from_text.arguments.size(),
           hlo_module_and_arguments_from_binary.arguments.size());
}

TEST_F(FunctionalHloRunnerTest, FixFakeArguments) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  // Options corresponding to --num_replicas=1 --num_partitions=1
  xla::DebugOptions debug_options;
  FunctionalHloRunner::PreprocessingOptions preproc_options;
  CompileOptions compile_options;
  FunctionalHloRunner::RunningOptions running_options;

  std::minstd_rand0 engine(42);
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRun(
      *client, debug_options, preproc_options, compile_options, running_options,
      {GetHloPath("single_device.hlo")}, InputFormat::kText,
      /*arguments=*/{}, /*engine=*/&engine));
}

TEST(FunctionalHloRunnerTest, TestHloUnoptimizedSnapshotDeSerialization) {
  std::string path_to_text_hlo =
      GetHloPath("sharded_unoptimized_hlo_snapshot.pbtxt");

  // Read the text proto
  HloUnoptimizedSnapshot snapshot;
  TF_ASSERT_OK(
      tsl::ReadTextProto(tsl::Env::Default(), path_to_text_hlo, &snapshot));

  // Serialize the snapshot to string
  std::string output;
  tsl::protobuf::io::StringOutputStream output_stream(&output);
  TF_ASSERT_OK(SerializeHloUnoptimizedSnapshot(snapshot, &output_stream));

  // Read the snapshot back from the string
  tsl::protobuf::io::ArrayInputStream input_stream(output.data(),
                                                   output.size());
  auto maybe_deserialized_snapshot =
      DeserializeHloUnoptimizedSnapshot(&input_stream);
  ASSERT_TRUE(maybe_deserialized_snapshot.ok());

  EXPECT_EQ(snapshot.SerializeAsString(),
            maybe_deserialized_snapshot->SerializeAsString());
}

TEST(FunctionalHloRunnerTest, TestDebugOptionsAreNotOverwrittenByRawOptions) {
  // If xla_dump_to is set in the raw options, then the debug options are
  // overridden in `CreateCompileOptions` and we lose the dumping debug options.
  // This test checks that we don't overwrite if we don't set xla_dump_to in the
  // raw options.
  xla::DebugOptions debug_options;
  debug_options.set_xla_dump_hlo_as_text(true);
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.execution_options = ExecutionOptions();
  *raw_compile_options.execution_options->mutable_debug_options() =
      debug_options;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  TF_ASSERT_OK_AND_ASSIGN(
      CompileOptions compile_options,
      FunctionalHloRunner::CreateCompileOptions(*client, raw_compile_options,
                                                /*task_id=*/0, /*num_nodes=*/1,
                                                /*kv_store=*/nullptr));
  EXPECT_TRUE(compile_options.executable_build_options.debug_options()
                  .xla_dump_hlo_as_text());
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
  testing::InitGoogleTest(&argc, argv);
  if (node_id >= 0) {
    return !xla::ShardedAutotuningWorksTestBody(node_id).ok();
  }
  return RUN_ALL_TESTS();
}
