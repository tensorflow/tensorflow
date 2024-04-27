/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/hlo_runner_pjrt.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_util.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

// TODO(b/245550554): Remove the use of PjRtWrappedExecutable.
class PjRtWrappedExecutable : public Executable {
 public:
  explicit PjRtWrappedExecutable(std::shared_ptr<HloModule> hlo_module,
                                 PjRtLoadedExecutable* pjrt_loaded_executable)
      : Executable(hlo_module),
        pjrt_loaded_executable_(pjrt_loaded_executable) {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  PjRtLoadedExecutable* GetPjRtLoadedExecutable() const {
    return pjrt_loaded_executable_;
  }

 private:
  PjRtLoadedExecutable* pjrt_loaded_executable_;
};

absl::StatusOr<ExecutionOutput> PjRtWrappedExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  return Unimplemented(
      "PjRtWrappedExecutable: Unimplemented ExecuteAsyncOnStream");
}

static const int kDeviceIdx = 0;

HloRunnerPjRt::HloRunnerPjRt(
    std::unique_ptr<PjRtClient> pjrt_client,
    DeviceShapeRepresentationFn device_shape_representation_fn)
    : pjrt_client_(std::move(pjrt_client)),
      device_shape_representation_fn_(device_shape_representation_fn) {}

HloRunnerPjRt::~HloRunnerPjRt() = default;

absl::StatusOr<CompileOptions> HloRunnerPjRt::GenerateDefaultCompileOptions(
    HloModule* module, bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(
      auto device_assignment,
      pjrt_client_->GetDefaultDeviceAssignment(
          module->config().replica_count(), module->config().num_partitions()));

  CompileOptions compile_options;

  compile_options.executable_build_options.set_device_assignment(
      device_assignment);
  compile_options.executable_build_options.set_num_partitions(
      module->config().num_partitions());
  compile_options.executable_build_options.set_num_replicas(
      module->config().replica_count());
  compile_options.executable_build_options.set_run_backend_only(
      !run_hlo_passes);
  *compile_options.executable_build_options.mutable_debug_options() =
      module->config().debug_options();

  std::vector<Shape> parameter_shapes;
  parameter_shapes.reserve(
      module->entry_computation_layout().parameter_count());
  for (const ShapeLayout& shape_layout :
       module->entry_computation_layout().parameter_layouts()) {
    parameter_shapes.push_back(shape_layout.shape());
  }
  compile_options.argument_layouts = parameter_shapes;

  return compile_options;
}

absl::StatusOr<Literal> HloRunnerPjRt::TransferLiteralFromDevice(
    PjRtBuffer& buffer) {
  TF_RETURN_IF_ERROR(buffer.GetReadyFuture().Await());

  TF_ASSIGN_OR_RETURN(auto literal, buffer.ToLiteralSync());
  return std::move(*literal);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
HloRunnerPjRt::TransferLiteralToDevice(const Literal& literal) {
  auto devices = pjrt_client_->addressable_devices();

  TF_ASSIGN_OR_RETURN(auto assignment, pjrt_client_->BufferFromHostLiteral(
                                           literal, devices[kDeviceIdx]));

  return std::move(assignment);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::TransferLiteralsToDevice(
    absl::Span<const Literal* const> literals) {
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(literals.size());
  for (const Literal* literal : literals) {
    TF_RET_CHECK(literal != nullptr);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer,
                        TransferLiteralToDevice(*literal));
    TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
    buffers.push_back(std::move(buffer));
  }
  return std::move(buffers);
}

absl::StatusOr<Literal> HloRunnerPjRt::Execute(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  // TODO (b/245550554) : Remove UpdateEntryComputationLayout from runner.
  xla::UpdateEntryComputationLayout(module.get(),
                                    device_shape_representation_fn_);
  TF_ASSIGN_OR_RETURN(auto compile_options, GenerateDefaultCompileOptions(
                                                module.get(), run_hlo_passes));

  TF_ASSIGN_OR_RETURN(auto executable,
                      CreateExecutable(std::move(module), run_hlo_passes));

  return ExecuteWithExecutable(executable.get(), arguments, {});
}

std::vector<PjRtBuffer*> HloRunnerPjRt::BufferVecToPointerVec(
    const std::vector<std::unique_ptr<PjRtBuffer>>& buffer) {
  std::vector<PjRtBuffer*> argument_ptrs;
  argument_ptrs.resize(buffer.size());
  for (int i = 0; i < buffer.size(); ++i) {
    argument_ptrs[i] = buffer[i].get();
  }

  return argument_ptrs;
}

std::vector<std::vector<PjRtBuffer*>> HloRunnerPjRt::BufferMatToPointerMat(
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& buffer) {
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs;
  argument_ptrs.reserve(buffer.size());
  for (int i = 0; i < buffer.size(); ++i) {
    argument_ptrs.push_back(BufferVecToPointerVec(buffer[i]));
  }
  return argument_ptrs;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
HloRunnerPjRt::CreateExecutable(HloModule* module,
                                CompileOptions compile_options) {
  XlaComputation computation(module->ToProto());

  return pjrt_client_->Compile(computation, compile_options);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::ExecuteWithDeviceBuffers(
    PjRtLoadedExecutable* executable,
    const std::vector<std::unique_ptr<PjRtBuffer>>& arguments) {
  ExecuteOptions execute_options;

  std::vector<PjRtBuffer*> argument_ptrs = BufferVecToPointerVec(arguments);

  auto devices = pjrt_client_->addressable_devices();

  std::optional<PjRtFuture<>> returned_future = {};

  TF_ASSIGN_OR_RETURN(
      auto output_buffers,
      executable->ExecuteSharded(argument_ptrs, devices[kDeviceIdx],
                                 execute_options, returned_future, false));

  return output_buffers;
}

absl::StatusOr<Literal> HloRunnerPjRt::ExecuteWithExecutable(
    Executable* executable, absl::Span<const Literal* const> arguments,
    ExecutionProfile* profile) {
  PjRtWrappedExecutable* wrapped_executable =
      static_cast<PjRtWrappedExecutable*>(executable);

  TF_ASSIGN_OR_RETURN(auto argument_handles,
                      TransferLiteralsToDevice(arguments));

  TF_ASSIGN_OR_RETURN(
      auto output_buffer,
      ExecuteWithDeviceBuffers(wrapped_executable->GetPjRtLoadedExecutable(),
                               std::move(argument_handles)));
  // TODO (b/245550554): Support more than 1 output.
  CHECK_EQ(output_buffer.size(), 1);

  return TransferLiteralFromDevice(*output_buffer[0]);
}

absl::StatusOr<std::unique_ptr<Executable>> HloRunnerPjRt::CreateExecutable(
    std::unique_ptr<HloModule> module, bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(auto compile_options, GenerateDefaultCompileOptions(
                                                module.get(), run_hlo_passes));

  TF_ASSIGN_OR_RETURN(auto pjrt_executable,
                      CreateExecutable(module.get(), compile_options));

  auto executable = std::make_unique<PjRtWrappedExecutable>(
      std::shared_ptr<HloModule>(std::move(module)), pjrt_executable.release());

  std::unique_ptr<Executable> exec =
      static_cast<std::unique_ptr<Executable>>(executable.release());
  return exec;
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const HloRunnerInterface::ReplicatedExecuteOptions& options) {
  xla::UpdateEntryComputationLayout(module.get(),
                                    device_shape_representation_fn_);

  TF_ASSIGN_OR_RETURN(
      auto device_assignment,
      pjrt_client_->GetDefaultDeviceAssignment(
          options.num_replicas, module->config().num_partitions()));
  return ExecuteReplicated(std::move(module), options, &device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  module->mutable_config().set_replica_count(options.num_replicas);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));

  return ExecuteReplicated(executable.get(), options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    Executable* executable,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment, ExecutionProfile* profile) {
  return ExecuteReplicatedImpl(
      [&](absl::Span<const std::vector<PjRtBuffer*>>& argument_buffer_slices)
          -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
        PjRtWrappedExecutable* wrapped_executable =
            static_cast<PjRtWrappedExecutable*>(executable);

        TF_ASSIGN_OR_RETURN(
            auto execution_results,
            wrapped_executable->GetPjRtLoadedExecutable()->Execute(
                argument_buffer_slices, {}));

        std::vector<std::unique_ptr<PjRtBuffer>> results;

        for (auto& device_execution_result : execution_results) {
          for (auto& device_buffer : device_execution_result) {
            results.push_back(std::move(device_buffer));
          }
        }

        return results;
      },
      [&](int64_t replica) { return options.arguments.size(); },
      [&](int64_t replica, int64_t index) { return options.arguments[index]; },
      options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::function<Executable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  return Unimplemented("Unimplemeneted ExecuteReplicated");
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicatedImpl(
    std::function<absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>(
        absl::Span<const std::vector<PjRtBuffer*>>&)>
        execution_helper,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  absl::Span<PjRtDevice* const> devices = pjrt_client_->devices();

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffer_slices;
  argument_buffer_slices.reserve(pjrt_client_->addressable_device_count());

  for (int64_t i = 0; i < options.num_replicas; ++i) {
    PjRtDevice* device_ptr = devices[i];

    // Transfer literals to device.
    const int64_t argument_count = argument_count_provider(i);

    std::vector<std::unique_ptr<PjRtBuffer>> replica_buffers;
    replica_buffers.reserve(argument_count);

    for (int64_t arg_index = 0; arg_index < argument_count; arg_index++) {
      const Literal* const argument = argument_provider(i, arg_index);
      TF_RET_CHECK(argument != nullptr);

      TF_ASSIGN_OR_RETURN(auto assignment, pjrt_client_->BufferFromHostLiteral(
                                               *argument, device_ptr));
      replica_buffers.push_back(std::move(assignment));
    }

    argument_buffer_slices.push_back(std::move(replica_buffers));
  }

  TF_RET_CHECK(options.infeed_values.empty() ||
               options.infeed_values.size() == options.num_replicas);

  if (!options.infeed_values.empty()) {
    // TODO(b/245550554): Infeed/Outfeed
  }

  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    // TODO(b/245550554): Infeed/Outfeed
  }

  auto mat = BufferMatToPointerMat(argument_buffer_slices);

  auto span = absl::Span<const std::vector<PjRtBuffer*>>(mat);

  TF_ASSIGN_OR_RETURN(auto results, execution_helper(span));
  std::vector<Literal> exec_results;
  exec_results.reserve(options.num_replicas);

  for (int64_t i = 0; i < options.num_replicas; ++i) {
    TF_ASSIGN_OR_RETURN(Literal literal,
                        TransferLiteralFromDevice(*results[i]));

    exec_results.push_back(std::move(literal));
  }

  return std::move(exec_results);
}

absl::string_view HloRunnerPjRt::Name() const { return "HloRunnerPjRt"; }

}  // namespace xla
