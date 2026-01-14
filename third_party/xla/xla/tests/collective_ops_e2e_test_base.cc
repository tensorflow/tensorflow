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

#include "xla/tests/collective_ops_e2e_test_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace {

absl::StatusOr<gpu::GpuTargetConfig> GetGpuTargetConfig(PjRtClient* client) {
  ASSIGN_OR_RETURN(const PjRtTopologyDescription* topology,
                   client->GetTopologyDescription());
  auto it = topology->Attributes().find("target_config");
  if (it == topology->Attributes().end()) {
    return absl::InvalidArgumentError(
        "Topology description does not contain target config");
  }
  if (!std::holds_alternative<std::string>(it->second)) {
    return absl::InvalidArgumentError(
        "Target config is not a string in topology description");
  }
  stream_executor::GpuTargetConfigProto target_config_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(
          std::get<std::string>(it->second), &target_config_proto)) {
    return absl::InvalidArgumentError(
        "Failed to parse target config from topology description");
  }
  return gpu::GpuTargetConfig::FromProto(target_config_proto);
}

}  // namespace

void CollectiveOpsE2ETestBase::SetupHloRunner(size_t memory_size,
                                              size_t collectives_memory_size) {
  xla::GpuClientOptions options;
  options.allocator_config.kind = xla::GpuAllocatorConfig::Kind::kBFC;
  options.allocator_config.gpu_system_memory_size = memory_size;
  options.allocator_config.collective_memory_size = collectives_memory_size;
  options.use_tfrt_gpu_client =
      std::getenv("XLA_TEST_USE_STREAM_EXECUTOR_GPU_CLIENT") == nullptr;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> pjrt_client,
                       xla::GetXlaPjrtGpuClient(options));

  ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                       GetGpuTargetConfig(pjrt_client.get()));
  gpu_compute_capability_ =
      target_config.device_description.gpu_compute_capability();

  hlo_runner_ = std::make_unique<HloRunnerPjRt>(std::move(pjrt_client));
}

absl::StatusOr<CollectiveOpsE2ETestBase::ExecutionResult>
CollectiveOpsE2ETestBase::ExecuteReplicated(std::unique_ptr<HloModule> module) {
  return ExecuteReplicated(std::move(module),
                           /*arguments=*/std::vector<Literal*>(),
                           /*run_hlo_passes=*/true);
}

absl::StatusOr<CollectiveOpsE2ETestBase::ExecutionResult>
CollectiveOpsE2ETestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, const std::vector<Literal*>& arguments,
    bool run_hlo_passes) {
  int64_t num_devices =
      module->config().replica_count() * module->config().num_partitions();

  return ExecuteReplicated(
      std::move(module),
      /*arguments=*/std::vector<std::vector<Literal*>>(num_devices, arguments),
      /*run_hlo_passes=*/run_hlo_passes);
}

absl::StatusOr<CollectiveOpsE2ETestBase::ExecutionResult>
CollectiveOpsE2ETestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const std::vector<std::vector<Literal*>>& arguments, bool run_hlo_passes) {
  ExecutionResult execution_result;

  TF_ASSIGN_OR_RETURN(
      execution_result.executable,
      hlo_runner_->CreateExecutable(std::move(module), run_hlo_passes));

  TF_ASSIGN_OR_RETURN(
      execution_result.optimized_module,
      hlo_runner_->HloModuleFromWrapped(execution_result.executable.get()));

  TF_ASSIGN_OR_RETURN(execution_result.results,
                      ExecuteReplicated(execution_result.executable.get(),
                                        arguments, run_hlo_passes));

  return execution_result;
}

absl::StatusOr<std::vector<Literal>>
CollectiveOpsE2ETestBase::ExecuteReplicated(
    OpaqueExecutable* executable,
    const std::vector<std::vector<Literal*>>& arguments, bool run_hlo_passes) {
  ASSIGN_OR_RETURN(const HloModule* module,
                   hlo_runner_->HloModuleFromWrapped(executable));

  int64_t num_replicas = module->config().replica_count();
  int64_t num_partitions = module->config().num_partitions();

  CHECK(num_replicas > 0 && "expect at least one replica");
  CHECK(num_partitions > 0 && "expect at least one partition");

  DeviceAssignment device_assignment =
      GetDefaultDeviceAssignment(num_replicas, num_partitions);
  int64_t num_devices = num_replicas * num_partitions;

  CHECK(num_devices == arguments.size() &&
        "expect arguments for each replica and partition");

  // TODO(b/441865120): Use designated initializers this once XLA moves to
  // C++20.
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_devices = num_devices;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;

  return hlo_runner_->ExecuteReplicated(
      /*executable_provider=*/
      [&](int64_t) { return executable; },
      /*argument_count_provider=*/
      [&](int64_t) { return arguments.front().size(); },
      /*argument_provider=*/
      [&](int64_t replica_idx, int64_t argument_idx) -> const Literal* {
        return arguments[replica_idx][argument_idx];
      },
      std::move(options),
      /*device_assignment=*/&device_assignment);
}

DebugOptions CollectiveOpsWithFlagsBase::GetDebugOptionsForTest() const {
  DebugOptions debug_options =
      CollectiveOpsE2ETestBase::GetDebugOptionsForTest();

  // Enable or disable all async collectives based on test parameter.
  if (!enable_async_) {
    for (auto option :
         {DebugOptions::NOOP, DebugOptions::ALLREDUCE, DebugOptions::ALLGATHER,
          DebugOptions::REDUCESCATTER, DebugOptions::COLLECTIVEBROADCAST,
          DebugOptions::ALLTOALL, DebugOptions::COLLECTIVEPERMUTE,
          DebugOptions::RAGGEDALLTOALL}) {
      debug_options.add_xla_gpu_disable_async_collectives(option);
    }
  }
  debug_options.add_xla_disable_hlo_passes(
      "gpu-convert-async-collectives-to-sync");
  if (enable_p2p_memcpy_) {
    debug_options.set_xla_gpu_use_memcpy_local_p2p(true);
  }
  return debug_options;
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
CollectiveOpsWithFlagsBase::CreateExecutable(absl::string_view hlo_string,
                                             int64_t num_replicas) {
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/num_replicas);

  TF_ASSIGN_OR_RETURN(auto module,
                      ParseAndReturnVerifiedModule(hlo_string, config));
  return hlo_runner_->CreateExecutable(std::move(module),
                                       /*run_hlo_passes=*/true);
}

}  // namespace xla
