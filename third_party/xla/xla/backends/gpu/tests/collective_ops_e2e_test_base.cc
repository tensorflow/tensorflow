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

#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

std::unique_ptr<PjRtClient> CreatePjRtClient(size_t memory_size,
                                             size_t collectives_memory_size) {
  xla::GpuClientOptions options;
  options.allocator_config.kind = xla::GpuAllocatorConfig::Kind::kBFC;
  options.allocator_config.gpu_system_memory_size = memory_size;
  options.allocator_config.collective_memory_size = collectives_memory_size;
  options.use_tfrt_gpu_client = true;

  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> pjrt_client =
      xla::GetXlaPjrtGpuClient(options);
  CHECK_OK(pjrt_client);
  return *std::move(pjrt_client);
}

}  // namespace

CollectiveOpsE2ETestBase::CollectiveOpsE2ETestBase(
    size_t memory_size, size_t collectives_memory_size)
    : HloPjRtGpuTestBase(
          CreatePjRtClient(memory_size, collectives_memory_size)) {}

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

  ABSL_ASSIGN_OR_RETURN(execution_result.executable,
                   CreateExecutable(std::move(module), run_hlo_passes));

  ABSL_ASSIGN_OR_RETURN(
      execution_result.optimized_module,
      test_runner().HloModuleFromWrapped(execution_result.executable.get()));

  ABSL_ASSIGN_OR_RETURN(execution_result.results,
                   ExecuteReplicated(execution_result.executable.get(),
                                     arguments, run_hlo_passes));

  return execution_result;
}

absl::StatusOr<std::vector<Literal>>
CollectiveOpsE2ETestBase::ExecuteReplicated(
    OpaqueExecutable* executable,
    const std::vector<std::vector<Literal*>>& arguments, bool run_hlo_passes) {
  ABSL_ASSIGN_OR_RETURN(const HloModule* module,
                   test_runner().HloModuleFromWrapped(executable));

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

  return test_runner().ExecuteReplicated(
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
  if (enable_async_) {
    debug_options.add_xla_disable_hlo_passes(
        "gpu-convert-async-collectives-to-sync");
  } else {
    for (auto option :
         {DebugOptions::NOOP, DebugOptions::ALLREDUCE, DebugOptions::ALLGATHER,
          DebugOptions::REDUCESCATTER, DebugOptions::COLLECTIVEBROADCAST,
          DebugOptions::ALLTOALL, DebugOptions::COLLECTIVEPERMUTE,
          DebugOptions::RAGGEDALLTOALL}) {
      debug_options.add_xla_gpu_disable_async_collectives(option);
    }
  }

  if (enable_symmetric_buffer_) {
    auto* filter =
        debug_options.add_xla_enable_nccl_symmetric_buffers_for_collectives();
    filter->set_collective(DebugOptions::ALLCOLLECTIVES);
  }

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

  ABSL_ASSIGN_OR_RETURN(auto module,
                   ParseAndReturnVerifiedModule(hlo_string, config));
  return test_runner().CreateExecutable(std::move(module),
                                        /*run_hlo_passes=*/true);
}

absl::StatusOr<CommandBufferThunkCounts> CountThunksInDump(
    absl::string_view dump_dir, absl::string_view thunk_kind_prefix) {
  std::vector<std::string> dump_files;
  ABSL_RETURN_IF_ERROR(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_dir, "*thunk_sequence_after_thunk_passes*.txt"),
      &dump_files));
  if (dump_files.empty()) {
    // When thunk passes make no changes (e.g. no command buffers are formed),
    // only the initial thunk_sequence.txt dump is written.
    ABSL_RETURN_IF_ERROR(tsl::Env::Default()->GetMatchingPaths(
        tsl::io::JoinPath(dump_dir, "*thunk_sequence.txt"), &dump_files));
  }
  if (dump_files.size() != 1) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected exactly one thunk sequence dump in ", dump_dir,
                     ", found ", dump_files.size()));
  }
  std::string dump;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), dump_files[0], &dump));

  // Each thunk line in the dump has the form "<indent><index>: <kind> ...".
  // Thunks nested inside a top-level kCommandBuffer thunk are indented under
  // it; non-thunk lines (such as WhileThunk's "condition:" and "body:" headers)
  // do not start with "<digits>: " and are ignored.
  CommandBufferThunkCounts counts;
  bool in_command_buffer = false;
  for (absl::string_view line : absl::StrSplit(dump, '\n')) {
    absl::string_view thunk = absl::StripLeadingAsciiWhitespace(line);
    size_t pos = thunk.find(": ");
    if (pos == absl::string_view::npos || pos == 0 ||
        !absl::c_all_of(thunk.substr(0, pos),
                        [](char c) { return absl::ascii_isdigit(c); })) {
      continue;
    }
    const bool is_top_level = thunk.size() == line.size();
    thunk.remove_prefix(pos + 2);
    if (is_top_level) {
      in_command_buffer = absl::StartsWith(thunk, "kCommandBuffer");
    }
    if (absl::StartsWith(thunk, thunk_kind_prefix)) {
      ++(in_command_buffer ? counts.in_command_buffer
                           : counts.outside_command_buffer);
    }
  }
  return counts;
}

}  // namespace xla
