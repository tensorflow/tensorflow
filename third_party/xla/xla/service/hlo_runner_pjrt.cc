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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla {

namespace {

absl::Status SanityCheckParameterLayouts(
    const ComputationLayout& entry_layout) {
  const std::vector<ShapeLayout>& layouts = entry_layout.parameter_layouts();
  bool has_nested_tuples =
      absl::c_any_of(layouts, [](const auto& shape_layout) {
        return ShapeUtil::IsNestedTuple(shape_layout.shape());
      });
  if (has_nested_tuples) {
    return InvalidArgument(
        "PJRT does not support nested tuples as input parameters");
  }
  int num_tuples = absl::c_count_if(layouts, [](const auto& shape_layout) {
    return shape_layout.shape().IsTuple();
  });
  if (num_tuples > 1) {
    return InvalidArgument(
        "PJRT does not support more than one tuple as input parameters"
        " (found %d tuples)",
        num_tuples);
  }
  if (num_tuples == 1 && num_tuples != layouts.size()) {
    return InvalidArgument(
        "PJRT does not support mixing tuples and non-tuples as input "
        "parameters (found 1 tuple out of %d arguments)",
        layouts.size());
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MustFlattenInputTuple(
    const ComputationLayout& entry_layout) {
  TF_RETURN_IF_ERROR(SanityCheckParameterLayouts(entry_layout));
  // Strictly, we only need to flatten tuples with mixed host/device leaves
  // because mixed host/device PjRtBuffer's are not supported.
  // However, splitting all tuples makes the code simpler and is the way
  // PJRT is commonly used by JAX.
  return entry_layout.parameter_count() == 1 &&
         entry_layout.parameter_shape(0).IsTuple();
}

absl::StatusOr<ExecuteOptions> GenerateExecuteOptions(const HloModule& module) {
  ExecuteOptions execute_options;

  // If any output leaf buffer is in host memory, PJRT requires untuple_result.
  auto output_has_tuple_leaf_in_host_memory_space =
      [&module]() -> absl::StatusOr<bool> {
    if (!module.result_shape().IsTuple()) {
      return false;
    }
    TF_ASSIGN_OR_RETURN(
        std::vector<Layout> output_layouts,
        module.entry_computation_layout().FlattenedResultLayouts());
    return absl::c_any_of(output_layouts, [](const Layout& layout) {
      return layout.memory_space() == Layout::kHostMemorySpace;
    });
  };
  TF_ASSIGN_OR_RETURN(execute_options.untuple_result,
                      output_has_tuple_leaf_in_host_memory_space());

  return execute_options;
}

inline PjRtGlobalDeviceId DeviceIdForInvocation(
    const DeviceAssignment& device_assignment, const int64_t i) {
  const int64_t computation_count = device_assignment.computation_count();
  return PjRtGlobalDeviceId(
      device_assignment(i / computation_count, i % computation_count));
}

absl::StatusOr<DeviceAssignment> GetStaticDeviceAssignmentOrComputeDefault(
    const HloModule& module, PjRtClient& client) {
  if (module.config().has_static_device_assignment()) {
    return module.config().static_device_assignment();
  }
  return client.GetDefaultDeviceAssignment(module.config().replica_count(),
                                           module.config().num_partitions());
}

}  // namespace

// TODO(b/245550554): Remove the use of PjRtWrappedExecutable.
class PjRtWrappedExecutable : public Executable {
 public:
  // Takes ownership of the provided executable.
  explicit PjRtWrappedExecutable(std::shared_ptr<HloModule> hlo_module,
                                 PjRtLoadedExecutable* pjrt_loaded_executable)
      : Executable(hlo_module),
        pjrt_loaded_executable_(pjrt_loaded_executable) {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

  PjRtLoadedExecutable* GetPjRtLoadedExecutable() const {
    return pjrt_loaded_executable_.get();
  }

 private:
  std::unique_ptr<PjRtLoadedExecutable> pjrt_loaded_executable_;
};

absl::StatusOr<ExecutionOutput> PjRtWrappedExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  return Unimplemented(
      "PjRtWrappedExecutable: Unimplemented ExecuteAsyncOnStream");
}

static const int kDeviceIdx = 0;

HloRunnerPjRt::HloRunnerPjRt(
    std::unique_ptr<PjRtClient> pjrt_client,
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn,
    const bool use_parameter_layout_on_device)
    : pjrt_client_(std::move(pjrt_client)),
      device_shape_representation_fn_(device_shape_representation_fn),
      device_shape_size_fn_(device_shape_size_fn),
      use_parameter_layout_on_device_(use_parameter_layout_on_device) {}

HloRunnerPjRt::~HloRunnerPjRt() = default;

absl::StatusOr<CompileOptions> HloRunnerPjRt::GenerateDefaultCompileOptions(
    HloModule* module, bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment device_assignment,
      GetStaticDeviceAssignmentOrComputeDefault(*module, *pjrt_client_));

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

  TF_ASSIGN_OR_RETURN(
      bool flatten, MustFlattenInputTuple(module->entry_computation_layout()));
  compile_options.parameter_is_tupled_arguments = flatten;

  compile_options.executable_build_options.set_result_layout(
      module->entry_computation_layout().result_shape());

  return compile_options;
}

absl::StatusOr<Literal> HloRunnerPjRt::TransferLiteralFromDevice(
    PjRtBuffer& buffer) {
  TF_RETURN_IF_ERROR(buffer.GetReadyFuture().Await());

  TF_ASSIGN_OR_RETURN(auto literal, buffer.ToLiteralSync());
  return std::move(*literal);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
HloRunnerPjRt::TransferLiteralToDevice(const Literal& literal,
                                       const Layout& parameter_layout) {
  auto devices = pjrt_client_->addressable_devices();
  PjRtDevice* device = devices[kDeviceIdx];

  if (pjrt_client_->memory_spaces().empty()) {
    TF_ASSIGN_OR_RETURN(
        auto assignment,
        use_parameter_layout_on_device_
            ? pjrt_client_->BufferFromHostLiteral(literal, device,
                                                  &parameter_layout)
            : pjrt_client_->BufferFromHostLiteral(literal, device));
    return std::move(assignment);
  }

  auto get_pjrt_memory_space = [](PjRtDevice* pjrt_device,
                                  int64_t xla_memory_space) {
    if (xla_memory_space == Layout::kHostMemorySpace) {
      return pjrt_device->memory_space_by_kind(PinnedHostMemorySpace::kKind);
    }
    return pjrt_device->default_memory_space();
  };
  TF_ASSIGN_OR_RETURN(
      PjRtMemorySpace * pjrt_memory_space,
      get_pjrt_memory_space(device, parameter_layout.memory_space()));
  TF_ASSIGN_OR_RETURN(
      auto assignment,
      use_parameter_layout_on_device_
          ? pjrt_client_->BufferFromHostLiteral(literal, pjrt_memory_space,
                                                &parameter_layout)
          : pjrt_client_->BufferFromHostLiteral(literal, pjrt_memory_space));
  return std::move(assignment);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::TransferLiteralsToDevice(
    const ComputationLayout& entry_layout,
    absl::Span<const Literal* const> literals) {
  TF_ASSIGN_OR_RETURN(bool flatten, MustFlattenInputTuple(entry_layout));
  TF_ASSIGN_OR_RETURN(std::vector<Layout> parameter_layouts,
                      entry_layout.FlattenedParameterLayouts());

  auto transfer_literals = [&parameter_layouts, this](
                               absl::Span<const Literal* const> input_literals)
      -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
    TF_RET_CHECK(parameter_layouts.size() == input_literals.size());
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    buffers.reserve(input_literals.size());
    for (int i = 0; i < input_literals.size(); ++i) {
      const Literal* literal = input_literals[i];
      TF_RET_CHECK(literal != nullptr);
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> buffer,
          TransferLiteralToDevice(*literal, parameter_layouts[i]));
      TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
      buffers.push_back(std::move(buffer));
    }
    return std::move(buffers);
  };

  if (flatten) {
    Literal cloned_literal = literals[0]->Clone();
    std::vector<Literal> flattened = cloned_literal.DecomposeTuple();
    std::vector<const Literal*> flattened_ptrs;
    flattened_ptrs.reserve(flattened.size());
    for (const Literal& literal : flattened) {
      flattened_ptrs.push_back(&literal);
    }
    return transfer_literals(flattened_ptrs);
  }
  return transfer_literals(literals);
}

absl::StatusOr<Literal> HloRunnerPjRt::Execute(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  // TODO (b/245550554) : Remove UpdateEntryComputationLayout from runner.
  UpdateEntryComputationLayout(module.get());
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

  return pjrt_client_->Compile(computation, std::move(compile_options));
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::ExecuteWithDeviceBuffers(
    PjRtLoadedExecutable* executable, const ExecuteOptions& execute_options,
    const std::vector<std::unique_ptr<PjRtBuffer>>& arguments) {
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

  auto* pjrt_executable = wrapped_executable->GetPjRtLoadedExecutable();
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      pjrt_executable->GetHloModules());
  TF_RET_CHECK(hlo_modules.size() == 1);
  const HloModule& module = *hlo_modules.front();

  TF_ASSIGN_OR_RETURN(ExecuteOptions execute_options,
                      GenerateExecuteOptions(module));
  TF_ASSIGN_OR_RETURN(
      auto argument_handles,
      TransferLiteralsToDevice(module.entry_computation_layout(), arguments));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> output_buffers,
      ExecuteWithDeviceBuffers(wrapped_executable->GetPjRtLoadedExecutable(),
                               execute_options, std::move(argument_handles)));
  if (!execute_options.untuple_result) {
    TF_RET_CHECK(output_buffers.size() == 1)
        << ", got " << output_buffers.size();
    return TransferLiteralFromDevice(*output_buffers[0]);
  }
  std::vector<Literal> result_leaves;
  for (const auto& leaf_buffer : output_buffers) {
    TF_ASSIGN_OR_RETURN(Literal leaf, TransferLiteralFromDevice(*leaf_buffer));
    result_leaves.push_back(std::move(leaf));
  }
  return Literal::MoveIntoTuple(absl::MakeSpan(result_leaves));
}

absl::StatusOr<std::unique_ptr<Executable>> HloRunnerPjRt::CreateExecutable(
    std::unique_ptr<HloModule> module, bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(auto compile_options, GenerateDefaultCompileOptions(
                                                module.get(), run_hlo_passes));

  TF_ASSIGN_OR_RETURN(auto pjrt_executable,
                      CreateExecutable(module.get(), compile_options));

  auto executable = std::make_unique<PjRtWrappedExecutable>(
      std::shared_ptr<HloModule>(
          std::move(pjrt_executable->GetHloModules().value()[0])),
      pjrt_executable.release());

  return executable;
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const HloRunnerInterface::ReplicatedExecuteOptions& options) {
  UpdateEntryComputationLayout(module.get());

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
  TF_RET_CHECK(device_assignment->computation_count() == 1)
      << "Only single-computation execution is supported.";
  return ExecuteReplicatedImpl(
      [&](absl::Span<const std::vector<PjRtBuffer*>>& argument_buffer_slices)
          -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
        TF_RET_CHECK(options.use_threads);

        // The underlying data is modified concurrently. We don't need to
        // protect access as each replica writes only to its own slot.
        std::vector<absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>>
            per_replica_results(options.num_replicas);
        absl::c_fill(per_replica_results,
                     absl::InternalError("No result for replica."));

        {
          // NB: `pool` is joined on destruction.
          tsl::thread::ThreadPool pool(tsl::Env::Default(), "replicas",
                                       options.num_replicas);
          for (int64_t i = 0; i < options.num_replicas; ++i) {
            for (const PjRtBuffer* const buffer : argument_buffer_slices[i]) {
              TF_RET_CHECK(buffer != nullptr);
            }
            PjRtWrappedExecutable* executable =
                tensorflow::down_cast<PjRtWrappedExecutable*>(
                    executable_provider(i));
            if (executable == nullptr) {
              return absl::InternalError(
                  absl::StrFormat("Failed to cast executable for replica %d "
                                  "to PjRtWrappedExecutable.",
                                  i));
            }
            TF_ASSIGN_OR_RETURN(
                PjRtDevice * device_ptr,
                pjrt_client_->LookupDevice(
                    DeviceIdForInvocation(*device_assignment, i)));
            pool.Schedule([&per_replica_results, i, executable,
                           args = argument_buffer_slices[i], device_ptr]() {
              per_replica_results[i] =
                  executable->GetPjRtLoadedExecutable()->ExecuteSharded(
                      args, device_ptr, {});
            });
          }
        }
        // Aggregate results.
        std::vector<std::unique_ptr<PjRtBuffer>> results;
        for (int64_t i = 0; i < options.num_replicas; ++i) {
          absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>&
              replica_result = per_replica_results[i];
          if (!replica_result.ok()) {
            return replica_result.status();
          }
          if (replica_result->size() != 1) {
            return absl::InternalError(absl::StrFormat(
                "Expected a single result for replica %d, got %d results.", i,
                replica_result->size()));
          }
          results.push_back(std::move(std::move(replica_result)->front()));
        }
        return results;
      },
      argument_count_provider, argument_provider, options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicatedImpl(
    std::function<absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>(
        absl::Span<const std::vector<PjRtBuffer*>>&)>
        execution_helper,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffer_slices;
  argument_buffer_slices.reserve(options.num_replicas);

  for (int64_t i = 0; i < options.num_replicas; ++i) {
    TF_ASSIGN_OR_RETURN(PjRtDevice * device_ptr,
                        pjrt_client_->LookupDevice(
                            DeviceIdForInvocation(*device_assignment, i)));

    // Transfer literals to device.
    const int64_t argument_count = argument_count_provider(i);

    std::vector<std::unique_ptr<PjRtBuffer>> replica_buffers;
    replica_buffers.reserve(argument_count);

    for (int64_t arg_index = 0; arg_index < argument_count; arg_index++) {
      const Literal* const argument = argument_provider(i, arg_index);
      TF_RET_CHECK(argument != nullptr);

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> assignment,
          use_parameter_layout_on_device_
              ? pjrt_client_->BufferFromHostLiteral(*argument, device_ptr,
                                                    &argument->shape().layout())
              : pjrt_client_->BufferFromHostLiteral(*argument, device_ptr));
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
