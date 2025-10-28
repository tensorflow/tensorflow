/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/core/host_offloading/host_offloading_pjrt_executable.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/core/host_offloading/host_offloading_buffer.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/core/host_offloading/host_offloading_layout_analysis.h"
#include "xla/core/host_offloading/host_offloading_transforms.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

HostOffloadingPjRtExecutable::HostOffloadingPjRtExecutable(
    std::string name, ProgramShape program_shape,
    HloInputOutputAliasConfig alias_config,
    std::unique_ptr<PjRtLoadedExecutable> executable,
    bool needs_layout_conversion)
    : name_(std::move(name)),
      program_shape_(std::move(program_shape)),
      alias_config_(std::move(alias_config)),
      executable_(std::move(executable)),
      needs_layout_conversion_(needs_layout_conversion) {}

namespace {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

// Verifies that parameters and result buffers have aliasing required by
// the alias config requested by the compiled executable.
absl::Status VerifyBufferAliasing(
    absl::Span<const ShapeTree<HostOffloadingBuffer>> parameters,
    const xla::ShapeTree<HostOffloadingBuffer>& result,
    const HloInputOutputAliasConfig& alias_config) {
  for (const auto& [index, buffer] : result.leaves()) {
    auto alias = alias_config.GetAliasedParameter(index);
    if (!alias.has_value()) {
      continue;
    }

    const HostOffloadingBuffer& result_buffer = result.element(index);
    const HostOffloadingBuffer& parameter_buffer =
        parameters.at(alias->parameter_number).element(alias->parameter_index);

    if (!(result_buffer == parameter_buffer)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Result buffer at index %s does not alias with "
                          "parameter %d buffer at index %s",
                          index.ToString(), alias->parameter_number,
                          alias->parameter_index.ToString()));
    }
  }
  return absl::OkStatus();
}

// Sets up host offloading specific options in HLO module config.
void SetHostOffloadingHloModuleConfig(HloModuleConfig& config) {
  auto& debug_options = config.mutable_debug_options();
  debug_options.set_xla_cpu_copy_insertion_use_region_analysis(true);
  // TODO(b/374556751): Megascale custom calls do not have correct data
  // dependencies and can be scheduled in wrong order.
  debug_options.set_xla_cpu_enable_concurrency_optimized_scheduler(false);
}

// A mutex for a global PJRT CPU client initialization.
ABSL_CONST_INIT absl::Mutex host_offloading_client_mutex(absl::kConstInit);

// Returns a global PJRT CPU client for host offloading computations.
absl::StatusOr<PjRtClient*> GetHostOffloadingPjRtClient() {
  static PjRtClient* client = nullptr;

  absl::MutexLock lock(host_offloading_client_mutex);
  if (client != nullptr) {
    return client;
  }

  xla::CpuClientOptions options;
  options.customize_hlo_module_config = SetHostOffloadingHloModuleConfig;

  VLOG(1) << "Create host offloading PjRt client for a current process";
  TF_ASSIGN_OR_RETURN(auto owned_client,
                      xla::GetXlaPjrtCpuClient(std::move(options)));
  return client = owned_client.release();
}

}  // namespace

absl::StatusOr<std::unique_ptr<HostOffloadingPjRtExecutable>>
HostOffloadingPjRtExecutable::LoadFromProto(
    const HostOffloadingExecutableProto& proto) {
  TF_RET_CHECK(proto.executable_type() ==
               HostOffloadingExecutableProto::EXECUTABLE_TYPE_PJRT);

  VLOG(3) << "Load PjRt host offloading executable: name="
          << proto.hlo_module().name();

  TraceMe trace([&] {
    return TraceMeEncode("HostOffloadingPjRtExecutable::LoadFromProto",
                         {{"name", proto.hlo_module().name()}});
  });

  // We keep program shape and alias config of the original HLO module and not
  // the destination-passing-style module with extra output parameters.
  TF_ASSIGN_OR_RETURN(
      ProgramShape program_shape,
      ProgramShape::FromProto(proto.hlo_module().host_program_shape()));
  TF_ASSIGN_OR_RETURN(
      auto alias_config,
      HloInputOutputAliasConfig::CreateFromProto(
          program_shape.result(), proto.hlo_module().input_output_alias()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      HloModule::CreateFromProto(
                          proto.hlo_module(), HloModuleConfig(program_shape)));

  TF_RETURN_IF_ERROR(RewriteToDestinationPassingStyle(
      hlo_module.get(), program_shape, alias_config));

  TF_ASSIGN_OR_RETURN(
      bool needs_layout_conversion,
      HostOffloadingLayoutAnalysis::NeedsLayoutConversion(hlo_module.get()));

  TF_ASSIGN_OR_RETURN(PjRtClient * client, GetHostOffloadingPjRtClient());

  CompileOptions compile_options;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      client->CompileAndLoad(XlaComputation(hlo_module.get()->ToProto()),
                             compile_options));

  return absl::WrapUnique(new HostOffloadingPjRtExecutable(
      proto.hlo_module().name(), std::move(program_shape),
      std::move(alias_config), std::move(executable), needs_layout_conversion));
}

tsl::AsyncValueRef<HostOffloadingExecutable::ExecuteEvent>
HostOffloadingPjRtExecutable::Execute(
    absl::Span<const ShapeTree<HostOffloadingBuffer>> parameters,
    const xla::ShapeTree<HostOffloadingBuffer>& result,
    const ExecuteOptions& execute_options) {
  VLOG(3) << "Execute PjRt host offloading executable: name=" << name_;

  TraceMe trace([&] {
    return TraceMeEncode(
        "HostOffloadingPjRtExecutable::Execute",
        {{"executable", absl::StrFormat("%s (device %d)", name_,
                                        execute_options.device_index)},
         {"launch_id", execute_options.launch_id}});
  });

  // Check that buffer aliasing is compatible with executable alias config.
  TF_RETURN_IF_ERROR(VerifyBufferAliasing(parameters, result, alias_config_));

  // We assume that for host offloading computation we have a single device.
  PjRtDevice* const device = executable_->client()->devices().front();
  TF_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());

  // Convert parameters and result to zero-copy PjRt buffers.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> arguments;

  auto add_argument = [&](const Shape& shape,
                          const HostOffloadingBuffer& buffer,
                          PjRtClient::HostBufferSemantics semantics) {
    DCHECK(shape.IsArray()) << "Buffer shape must be an array";
    TF_ASSIGN_OR_RETURN(
        arguments.emplace_back(),
        executable_->client()->BufferFromHostBuffer(
            buffer.opaque_base(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt, semantics, nullptr, memory_space,
            /*device_layout=*/nullptr));
    return absl::OkStatus();
  };

  static constexpr auto kImmutableZeroCopy =
      PjRtClient::HostBufferSemantics::kImmutableZeroCopy;
  static constexpr auto kMutableZeroCopy =
      PjRtClient::HostBufferSemantics::kMutableZeroCopy;

  for (size_t i = 0; i < parameters.size(); ++i) {
    const ShapeTree<HostOffloadingBuffer>& parameter = parameters[i];
    for (const auto& [index, buffer] : parameter.leaves()) {
      auto shape = ShapeUtil::GetSubshape(parameter.shape(), index);
      // If parameter is aliased with output we create a mutable zero-copy
      // buffer so that PjRtClient can write result into it.
      TF_RETURN_IF_ERROR(add_argument(shape, buffer,
                                      alias_config_.GetAliasedOutput(i, index)
                                          ? kMutableZeroCopy
                                          : kImmutableZeroCopy));
    }
  }

  for (const auto& [index, buffer] : result.leaves()) {
    // If output has an aliased parameter we don't add it to arguments as it
    // was already added above.
    if (alias_config_.OutputHasAlias(index)) {
      continue;
    }
    auto shape = ShapeUtil::GetSubshape(result.shape(), index);
    TF_RETURN_IF_ERROR(add_argument(shape, buffer, kMutableZeroCopy));
  }

  // Convert buffer arguments to non-owning arguments handles.
  absl::InlinedVector<PjRtBuffer*, 4> arguments_handles(arguments.size());
  for (size_t i = 0; i < arguments.size(); ++i) {
    arguments_handles[i] = arguments[i].get();
  }

  // TODO(b/340666998) Add additional context needed to support megascale ops
  ::xla::ExecuteOptions pjrt_execute_options{
      // Forward launch id to the host offloading executable because logically
      // it executes as a part of parent device execution.
      .launch_id = execute_options.launch_id,
      // Forward context to the host offloading executable so it has the
      // metadata required to perform megascale operations.
      .context = execute_options.context,
      // Host offloading executables dispatched into async work runner, and
      // there is no need to introduce another level of asynchronicity.
      .execution_mode = ::xla::ExecuteOptions::ExecutionMode::kSynchronous};

  // We immediately throw away result buffers because all of them must be
  // aliased with parameters or result buffers passed in arguments.
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> results,
                      executable_->ExecuteSharded(arguments_handles, device,
                                                  pjrt_execute_options));

  return tsl::MakeAvailableAsyncValueRef<ExecuteEvent>();
}

}  // namespace xla
