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

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

inline constexpr size_t kMB = 1024LL * 1024LL;
constexpr int kNumNodes = 2;
static const char* test_binary_name;

struct MultiProcessGpuClientSetup {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  std::unique_ptr<PjRtClient> client;
};

absl::StatusOr<MultiProcessGpuClientSetup> SetUpMultiProcessGpuClient(
    int rank_id, int num_nodes, int port, absl::string_view log_prefix) {
  MultiProcessGpuClientSetup prepared_test;
  std::string coordinator_address = absl::StrFormat("127.0.0.1:%d", port);

  // Rank 0 creates the coordination service.
  if (rank_id == 0) {
    LOG(INFO) << log_prefix << ": creating coordination service on "
              << coordinator_address;
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_nodes;
    ABSL_ASSIGN_OR_RETURN(prepared_test.service,
                     xla::GetDistributedRuntimeService(
                         absl::StrFormat("[::]:%d", port), service_options));
    LOG(INFO) << log_prefix << ": created coordination service";
  }

  // Connect to the coordination service.
  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = rank_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient(coordinator_address, distributed_options);

  LOG(INFO) << log_prefix << ": connecting distributed client";
  ABSL_RETURN_IF_ERROR(distributed_client->Connect());
  LOG(INFO) << log_prefix << ": distributed client connected";

  // Create the GPU client with a single addressable device per process.
  GpuClientOptions options;
  options.node_id = rank_id;
  options.num_nodes = num_nodes;
  options.allowed_devices = {rank_id};
  options.kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");
  options.distributed_client = distributed_client;
  options.allocator_config.kind = xla::GpuAllocatorConfig::Kind::kBFC;
  options.allocator_config.gpu_system_memory_size = 32 * kMB;
  options.allocator_config.collective_memory_size = 0;
  options.use_tfrt_gpu_client = true;

  LOG(INFO) << log_prefix << ": creating PjRtClient";
  ABSL_ASSIGN_OR_RETURN(prepared_test.client, GetXlaPjrtGpuClient(options));
  LOG(INFO) << log_prefix << ": PjRtClient created";

  return prepared_test;
}

absl::Status AllReduceMultiProcessTestBody(int node_id, int port) {
  std::string log_prefix = absl::StrFormat("rank_%d", node_id);
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform, PlatformUtil::GetPlatform("gpu"));
  ABSL_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                   platform->ExecutorForDevice(node_id));
  const auto& desc = executor->GetDeviceDescription();
  if (desc.gpu_compute_capability().IsCuda() && !desc.gpu_compute_capability()
                                                     .cuda_compute_capability()
                                                     ->IsAtLeastAmpere()) {
    LOG(INFO) << log_prefix
              << ": skipping one-shot all-reduce test, requires Ampere+ for "
                 "CUDA.";
    return absl::OkStatus();
  }

  ABSL_ASSIGN_OR_RETURN(
      MultiProcessGpuClientSetup setup,
      SetUpMultiProcessGpuClient(node_id, kNumNodes, port, log_prefix));
  std::unique_ptr<PjRtClient> client = std::move(setup.client);

  TF_RET_CHECK(client->addressable_device_count() == 1)
      << "Expected exactly 1 local addressable device per process.";
  TF_RET_CHECK(client->device_count() == kNumNodes)
      << "Expected " << kNumNodes << " global devices.";

  constexpr absl::string_view kModuleStr = R"(
    HloModule test

    apply_op {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT apply_op = f32[] add(x, y)
    }

    ENTRY test_computation {
      param_0 = f32[128] parameter(0)
      ROOT all-reduce = f32[128] all-reduce(param_0), to_apply=apply_op,
      replica_groups={{0,1}}
    }
  )";

  ABSL_ASSIGN_OR_RETURN(auto hlo_module,
                   ParseAndReturnUnverifiedModule(kModuleStr, /*config=*/{}));
  xla::XlaComputation computation(hlo_module->ToProto());

  xla::CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(kNumNodes);
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(kNumNodes, 1);
  device_assignment(0, 0) = 0;
  device_assignment(1, 0) = 1;
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);

  GpuTopology gpu_topology(
      /*platform_version=*/"",
      /*num_partitions=*/1,
      /*num_hosts_per_partition=*/1,
      /*num_devices_per_host=*/kNumNodes,
      /*gpu_target_config=*/std::nullopt,
      /*host_target_machine_options=*/std::nullopt,
      /*num_devices_per_process=*/kNumNodes);
  compile_options.executable_build_options.set_gpu_topology(gpu_topology);

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_autotune_level(0);
  debug_options.add_xla_gpu_unsupported_use_cross_host_one_shot_kernel(
      DebugOptions::ALLREDUCE);
  *compile_options.executable_build_options.mutable_debug_options() =
      debug_options;

  LOG(INFO) << log_prefix << ": compiling HLO module with one-shot all-reduce";
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                   client->CompileAndLoad(computation, compile_options));
  LOG(INFO) << log_prefix << ": compilation succeeded";

  // Inspect the collective kernel strategy selected in the optimized HLO.
  ABSL_ASSIGN_OR_RETURN(auto hlo_modules,
                   executable->GetExecutable()->GetHloModules());
  TF_RET_CHECK(!hlo_modules.empty());
  const HloModule* optimized_module = hlo_modules.front().get();
  bool found_all_reduce = false;
  for (const HloComputation* comp : optimized_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduce ||
          instr->opcode() == HloOpcode::kAllReduceStart) {
        found_all_reduce = true;
        auto gpu_config = instr->backend_config<gpu::GpuBackendConfig>();
        TF_RET_CHECK(gpu_config.ok())
            << "Failed to get GpuBackendConfig for " << instr->name() << ": "
            << gpu_config.status();
        LOG(INFO)
            << log_prefix << ": AllReduce instruction " << instr->name()
            << " collective kernel strategy: "
            << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                   gpu_config->collective_backend_config().kernel_strategy());
        TF_RET_CHECK(
            gpu_config->collective_backend_config().kernel_strategy() ==
            gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT)
            << "Expected AllReduce to use KERNEL_STRATEGY_TRITON_ONE_SHOT, but "
               "got: "
            << gpu::CollectiveBackendConfig::CollectiveKernelStrategy_Name(
                   gpu_config->collective_backend_config().kernel_strategy());
      }
    }
  }
  TF_RET_CHECK(found_all_reduce)
      << "Expected to find an AllReduce instruction in optimized HLO.";

  // Prepare input literal: rank 0 provides 10.0f, rank 1 provides 20.0f.
  const float input_val = (node_id == 0) ? 10.0f : 20.0f;
  Literal input_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, input_val));

  ABSL_ASSIGN_OR_RETURN(auto* memory_space,
                   client->addressable_devices()[0]->default_memory_space());
  ABSL_ASSIGN_OR_RETURN(auto input_buffer,
                   client->BufferFromHostLiteral(input_literal, memory_space));

  std::vector<std::vector<PjRtBuffer*>> input_ptrs = {{input_buffer.get()}};
  LOG(INFO) << log_prefix << ": executing one-shot all-reduce";
  ABSL_ASSIGN_OR_RETURN(auto results,
                   executable->Execute(input_ptrs, ExecuteOptions()));
  LOG(INFO) << log_prefix << ": execution finished";

  TF_RET_CHECK(results.size() == 1 && results[0].size() == 1);
  Literal result_literal(ShapeUtil::MakeShape(F32, {128}));
  ABSL_RETURN_IF_ERROR(results[0][0]->ToLiteralSync(&result_literal));

  // Expected result is 10.0f + 20.0f = 30.0f across all ranks.
  Literal expected_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(128, 30.0f));
  if (!LiteralTestUtil::Equal(expected_literal, result_literal)) {
    return absl::InternalError(absl::StrFormat(
        "Result literal %s does not match expected %s on rank %d",
        result_literal.ToString(), expected_literal.ToString(), node_id));
  }

  LOG(INFO) << log_prefix << ": verified result successfully";
  return absl::OkStatus();
}

TEST(AllReduceMultiProcessE2ETest, OneShotAllReduce2Processes) {
  absl::StatusOr<se::Platform*> platform = PlatformUtil::GetPlatform("gpu");
  if (!platform.ok() || (*platform)->VisibleDeviceCount() < kNumNodes) {
    GTEST_SKIP() << "Test requires at least " << kNumNodes
                 << " GPU devices, but found "
                 << (platform.ok() ? (*platform)->VisibleDeviceCount() : 0);
  }

  // Re-pass XLA_FLAGS environment variable to child processes.
  const char* xla_flags = std::getenv("XLA_FLAGS");
  if (xla_flags != nullptr) {
    tsl::setenv("XLA_FLAGS", xla_flags, /*overwrite=*/true);
  }

  int port = tsl::testing::PickUnusedPortOrDie();
  tsl::SubProcess child[kNumNodes];

  for (int node_id = 0; node_id < kNumNodes; ++node_id) {
    std::vector<std::string> argv = {
        test_binary_name,
        "--test_to_run=AllReduceMultiProcessHelper",
        absl::StrFormat("--node_id=%d", node_id),
        absl::StrFormat("--port=%d", port),
        "--alsologtostderr",
        "--vmodule=gpu_executable=1,thunk_executor=1,all_reduce_thunk=5,"
        "collective_kernel_thunk=5,collective_memory=5",
    };
    child[node_id].SetProgram(test_binary_name, argv);
    child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[node_id].Start()) << "Failed to start node " << node_id;
  }

  for (int node_id = 0; node_id < kNumNodes; ++node_id) {
    std::string stdout_str, stderr_str;
    int status = child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);

    const char* undeclared_outputs_dir =
        std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (undeclared_outputs_dir != nullptr &&
        undeclared_outputs_dir[0] != '\0') {
      std::string stderr_file = tsl::io::JoinPath(
          undeclared_outputs_dir,
          absl::StrFormat("subprocess_node_%d_stderr.log", node_id));
      absl::Status write_status =
          tsl::WriteStringToFile(tsl::Env::Default(), stderr_file, stderr_str);
      if (!write_status.ok()) {
        LOG(WARNING) << "Failed to write stderr to " << stderr_file << ": "
                     << write_status;
      }
    }

    EXPECT_EQ(status, 0) << "node " << node_id << " failed with status "
                         << status << "\nstdout:\n"
                         << stdout_str << "\nstderr:\n"
                         << stderr_str;
  }
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  std::string test_to_run;
  int node_id = -1;
  int port = -1;
  xla::test_binary_name = argv[0];

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("test_to_run", &test_to_run,
                "The test to run in the child process."),
      tsl::Flag("node_id", &node_id,
                "The node id (rank) for the child process."),
      tsl::Flag("port", &port, "The coordinator port for distributed runtime."),
  };

  xla::AppendDebugOptionsFlags(&flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);

  if (test_to_run.empty()) {
    return RUN_ALL_TESTS();
  }

  absl::Status result = absl::OkStatus();
  if (test_to_run == "AllReduceMultiProcessHelper") {
    result = xla::AllReduceMultiProcessTestBody(node_id, port);
  } else {
    result = absl::InvalidArgumentError(absl::StrFormat(
        "Unrecognized multiprocess test name: %s", test_to_run));
  }

  if (!result.ok()) {
    LOG(ERROR) << "Child process (node_id " << node_id
               << ") failed: " << result;
  }
  return result.raw_code();
}
