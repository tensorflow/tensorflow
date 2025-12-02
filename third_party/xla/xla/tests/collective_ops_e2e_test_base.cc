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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/integrations/stream_executor_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

std::unique_ptr<tsl::BFCAllocator> CreateAllocator(se::StreamExecutor* executor,
                                                   int64_t device_ordinal,
                                                   bool is_collective,
                                                   size_t memory_size) {
  std::string name_suffix = is_collective ? "_collectives_bfc" : "_bfc";
  tsl::BFCAllocator::Options opts;
  opts.allow_growth = false;
  std::unique_ptr<tsl::SubAllocator> device_mem_allocator;
  if (is_collective) {
    device_mem_allocator = std::make_unique<se::StreamExecutorAllocator>(
        executor->CreateMemoryAllocator(se::MemoryType::kCollective).value(),
        /*memory_type=*/stream_executor::MemoryType::kCollective,
        device_ordinal);
  } else {
    device_mem_allocator = std::make_unique<se::DeviceMemAllocator>(
        executor, tsl::PlatformDeviceId(device_ordinal));
  }
  return std::make_unique<tsl::BFCAllocator>(
      std::move(device_mem_allocator), memory_size,
      absl::StrCat("GPU_", device_ordinal, name_suffix), opts);
}

template <typename Type>
Type CheckStatus(absl::StatusOr<Type> result) {
  CHECK_OK(result);
  return *result;
}

}  // namespace

CollectiveOpsE2ETestBase::CollectiveOpsE2ETestBase() {
  se::Platform* platform = CheckStatus(PlatformUtil::GetPlatform("GPU"));
  se::Platform* reference_platform =
      CheckStatus(PlatformUtil::GetPlatform("GPU"));

  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  constexpr int64_t kGB = 1024LL * 1024LL * 1024LL;
  size_t common_buffers_size = 8 * kGB;   // 8GB
  size_t collectives_buffers_size = kGB;  // 1GB
  for (int64_t i = 0; i < platform->VisibleDeviceCount(); ++i) {
    se::StreamExecutor* executor = CheckStatus(platform->ExecutorForDevice(i));
    // Common memory allocator for device i.
    allocators.emplace_back(
        CreateAllocator(executor, i, /*is_collective=*/false,
                        common_buffers_size),
        nullptr, 0, i, platform);

    // Collectives and symmetric memory allocator for device i.
    allocators.emplace_back(CreateAllocator(executor, i, /*is_collective=*/true,
                                            collectives_buffers_size),
                            nullptr, (int)gpu::MemorySpaceColor::kCollective, i,
                            platform);
  }

  hlo_runner_ =
      std::make_unique<HloRunner>(platform, /*intra_op_parallelism_threads=*/0,
                                  std::make_unique<se::MultiDeviceAdapter>(
                                      platform, std::move(allocators)));
  reference_hlo_runner_ = std::make_unique<HloRunner>(
      reference_platform, /*intra_op_parallelism_threads=*/0);
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
  int64_t num_replicas = module->config().replica_count();
  int64_t num_partitions = module->config().num_partitions();

  CHECK(num_replicas > 0 && "expect at least one replica");
  CHECK(num_partitions > 0 && "expect at least one partition");

  DeviceAssignment device_assignment =
      GetDefaultDeviceAssignment(num_replicas, num_partitions);
  int64_t num_devices = num_replicas * num_partitions;

  CHECK(num_devices == arguments.size() &&
        "expect arguments for each replica and partition");

  ExecutionResult execution_result;

  TF_ASSIGN_OR_RETURN(
      execution_result.executable,
      hlo_runner_->CreateExecutable(std::move(module), run_hlo_passes));

  TF_ASSIGN_OR_RETURN(
      execution_result.optimized_module,
      hlo_runner_->HloModuleFromWrapped(execution_result.executable.get()));

  // TODO(b/441865120): Use designated initializers this once XLA moves to
  // C++20.
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_devices;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;

  TF_ASSIGN_OR_RETURN(
      execution_result.results,
      hlo_runner_->ExecuteReplicated(
          /*executable_provider=*/
          [&](int64_t) { return execution_result.executable.get(); },
          /*argument_count_provider=*/
          [&](int64_t) { return arguments.front().size(); },
          /*argument_provider=*/
          [&](int64_t replica_idx, int64_t argument_idx) -> const Literal* {
            return arguments[replica_idx][argument_idx];
          },
          std::move(options),
          /*device_assignment=*/&device_assignment));

  return execution_result;
}

DebugOptions CollectiveOpsWithFlagsBase::GetDebugOptionsForTest() const {
  DebugOptions debug_options =
      HloHardwareIndependentTestBase::GetDebugOptionsForTest();

  // Disable autotuning which is unnecessary.
  debug_options.set_xla_gpu_autotune_level(0);

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
