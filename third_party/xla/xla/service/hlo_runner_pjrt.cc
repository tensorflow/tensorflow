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
#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
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

  // PjRt requires untuple_result if any output leaf buffer is in host memory,
  // or if any output leaf buffer is not an array.
  if (module.result_shape().IsTuple()) {
    bool has_array_output_in_host_memory = false;
    bool has_non_array_output = false;
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        module.entry_computation_layout().result_shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (!subshape.IsArray()) {
            if (!subshape.IsTuple()) {
              has_non_array_output = true;
            }
            // Skip token, opaque, and tuple outputs.
            return absl::OkStatus();
          }
          // Arrays require a layout.
          if (!subshape.has_layout()) {
            return absl::InvalidArgumentError(
                "GenerateExecuteOptions requires that all array subshapes of "
                "the result shape have layouts.");
          }
          if (subshape.layout().memory_space() == Layout::kHostMemorySpace) {
            has_array_output_in_host_memory = true;
          }
          return absl::OkStatus();
        }));

    execute_options.untuple_result =
        has_array_output_in_host_memory || has_non_array_output;
  }
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

std::vector<PjRtBuffer*> BufferVecToPointerVec(
    const absl::Span<const std::unique_ptr<PjRtBuffer>> buffer) {
  std::vector<PjRtBuffer*> argument_ptrs;
  argument_ptrs.resize(buffer.size());
  for (int i = 0; i < buffer.size(); ++i) {
    argument_ptrs[i] = buffer[i].get();
  }

  return argument_ptrs;
}

std::vector<std::vector<PjRtBuffer*>> BufferMatToPointerMat(
    const absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> buffer) {
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs;
  argument_ptrs.reserve(buffer.size());
  for (int i = 0; i < buffer.size(); ++i) {
    argument_ptrs.push_back(BufferVecToPointerVec(buffer[i]));
  }
  return argument_ptrs;
}

constexpr int kDeviceIdx = 0;

absl::StatusOr<absl::Nonnull<PjRtMemorySpace*>> GetMemorySpaceFromLayout(
    absl::Nonnull<PjRtDevice*> const device, const Layout& layout) {
  PjRtMemorySpace* memory_space = nullptr;
  if (layout.memory_space() == Layout::kHostMemorySpace) {
    TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind(
                                          PinnedHostMemorySpace::kKind));
  } else {
    TF_ASSIGN_OR_RETURN(memory_space, device->default_memory_space());
  }
  TF_RET_CHECK(memory_space != nullptr)
      << "Memory space " << layout.memory_space()
      << " does not exist on device " << device->id();
  return memory_space;
}

class HloRunnerPjRtExecutable : public OpaqueExecutable {
 public:
  HloRunnerPjRtExecutable(
      absl::Nonnull<const HloRunnerPjRt*> creator,
      std::unique_ptr<PjRtLoadedExecutable> pjrt_loaded_executable)
      : OpaqueExecutable(creator),
        pjrt_loaded_executable_(std::move(pjrt_loaded_executable)) {}

  PjRtLoadedExecutable* pjrt_loaded_executable() const {
    return pjrt_loaded_executable_.get();
  }

  static absl::StatusOr<HloRunnerPjRtExecutable*> TryUnwrap(
      const HloRunnerPjRt& runner,
      absl::Nonnull<OpaqueExecutable*> const wrapped) {
    return OpaqueExecutable::TryUnwrap<HloRunnerPjRtExecutable>(runner,
                                                                wrapped);
  }
  static absl::StatusOr<const HloRunnerPjRtExecutable*> TryUnwrap(
      const HloRunnerPjRt& runner,
      absl::Nonnull<const OpaqueExecutable*> const wrapped) {
    return OpaqueExecutable::TryUnwrap<HloRunnerPjRtExecutable>(runner,
                                                                wrapped);
  }

 private:
  std::unique_ptr<PjRtLoadedExecutable> pjrt_loaded_executable_;
};

}  // namespace

HloRunnerPjRt::HloRunnerPjRt(
    std::unique_ptr<PjRtClient> pjrt_client,
    DeviceShapeRepresentationFn device_shape_representation_fn,
    DeviceShapeSizeFn device_shape_size_fn)
    : pjrt_client_(std::move(pjrt_client)),
      device_shape_representation_fn_(device_shape_representation_fn),
      device_shape_size_fn_(device_shape_size_fn) {}

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

  compile_options.executable_build_options.set_use_spmd_partitioning(
      module->config().use_spmd_partitioning());

  return compile_options;
}

absl::StatusOr<Literal> HloRunnerPjRt::TransferLiteralFromDevice(
    PjRtBuffer& buffer) {
  TF_RETURN_IF_ERROR(buffer.GetReadyFuture().Await());

  // Implementations of ToLiteralSync() do not support empty tuples. Since an
  // empty tuple literal is easy to construct, we do so here.
  if (const Shape& on_device_shape = buffer.on_device_shape();
      on_device_shape.IsTuple() && on_device_shape.tuple_shapes_size() == 0) {
    return LiteralUtil::MakeTuple({});
  }
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, buffer.ToLiteralSync());
  return std::move(*literal);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
HloRunnerPjRt::TransferLiteralToDevice(
    const Literal& literal, absl::Nonnull<PjRtMemorySpace*> const memory_space,
    const Layout& on_device_layout) {
  // Whenever possible, we want to respect the provided on-device layout. This
  // layout was either provided by the user or was inferred by the compiler. The
  // runtime should ideally not select a layout of its own accord.
  //
  // Not all clients implement this functionality.
  if (absl::StatusOr<std::unique_ptr<PjRtBuffer>> buffer =
          pjrt_client_->BufferFromHostLiteral(literal, memory_space,
                                              &on_device_layout);
      buffer.ok() || !absl::IsUnimplemented(buffer.status())) {
    return buffer;
  }
  // Fall back to the two-argument version of BufferFromHostLiteral.
  return pjrt_client_->BufferFromHostLiteral(literal, memory_space);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::TransferLiteralsToDevice(
    const ComputationLayout& entry_layout,
    absl::Span<const Literal* const> literals) {
  // Note: This function is used for single (default) device execution.
  if (pjrt_client_->addressable_device_count() <= kDeviceIdx) {
    return absl::InternalError("No addressable devices available");
  }
  PjRtDevice* device = pjrt_client_->addressable_devices()[kDeviceIdx];
  TF_RET_CHECK(device != nullptr)
      << "Device with ordinal " << kDeviceIdx << " is null.";

  TF_ASSIGN_OR_RETURN(bool flatten, MustFlattenInputTuple(entry_layout));
  TF_ASSIGN_OR_RETURN(std::vector<Layout> parameter_layouts,
                      entry_layout.FlattenedParameterLayouts());

  auto transfer_literals =
      [&, this](absl::Span<const Literal* const> input_literals)
      -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
    TF_RET_CHECK(parameter_layouts.size() == input_literals.size());
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    buffers.reserve(input_literals.size());
    for (int i = 0; i < input_literals.size(); ++i) {
      const Literal* literal = input_literals[i];
      TF_RET_CHECK(literal != nullptr);
      const Layout& on_device_layout = parameter_layouts[i];
      TF_ASSIGN_OR_RETURN(absl::Nonnull<PjRtMemorySpace*> memory_space,
                          GetMemorySpaceFromLayout(device, on_device_layout));
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> buffer,
          TransferLiteralToDevice(*literal, memory_space, on_device_layout));
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
  TF_ASSIGN_OR_RETURN(auto executable,
                      CreateExecutable(std::move(module), run_hlo_passes));

  return ExecuteWithExecutable(executable.get(), arguments, {});
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
    OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
    ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const wrapped_executable,
                      HloRunnerPjRtExecutable::TryUnwrap(*this, executable));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<HloModule>> hlo_modules,
      wrapped_executable->pjrt_loaded_executable()->GetHloModules());
  TF_RET_CHECK(hlo_modules.size() == 1);
  const HloModule& module = *hlo_modules.front();

  TF_ASSIGN_OR_RETURN(ExecuteOptions execute_options,
                      GenerateExecuteOptions(module));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> argument_handles,
      TransferLiteralsToDevice(module.entry_computation_layout(), arguments));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> output_buffers,
      ExecuteWithDeviceBuffers(wrapped_executable->pjrt_loaded_executable(),
                               execute_options, std::move(argument_handles)));
  if (!execute_options.untuple_result) {
    // If not flattened, the tuple should only contain arrays with layouts.
    TF_RET_CHECK(output_buffers.size() == 1)
        << ", got " << output_buffers.size();
    return TransferLiteralFromDevice(*output_buffers[0]);
  }
  std::vector<Literal> result_leaves;
  for (const std::unique_ptr<PjRtBuffer>& leaf_buffer : output_buffers) {
    const Shape& leaf_shape = leaf_buffer->on_device_shape();
    if (leaf_shape.IsArray()) {
      TF_ASSIGN_OR_RETURN(Literal leaf,
                          TransferLiteralFromDevice(*leaf_buffer));
      result_leaves.push_back(std::move(leaf));
    } else {
      // Untupled non-array buffers are not supported by
      // TransferLiteralFromDevice. We skip token, opaque, and tuple outputs.
      // This is similar to how this case is handled for HloRunner:
      // cs/symbol:xla::GenericTransferManager::TransferLiteralFromDevice
      //
      // Since we still need to return a tuple literal, create an empty literal
      // for the non-array buffer.
      result_leaves.push_back(Literal(leaf_shape));
    }
  }
  return Literal::MoveIntoTuple(absl::MakeSpan(result_leaves));
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
HloRunnerPjRt::CreateExecutable(std::unique_ptr<HloModule> module,
                                bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(
      CompileOptions compile_options,
      GenerateDefaultCompileOptions(module.get(), run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> pjrt_executable,
      CreateExecutable(module.get(), std::move(compile_options)));
  return std::make_unique<HloRunnerPjRtExecutable>(this,
                                                   std::move(pjrt_executable));
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const HloRunnerInterface::ReplicatedExecuteOptions& options) {
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
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));

  return ExecuteReplicated(executable.get(), options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    OpaqueExecutable* executable,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment, ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const wrapped_executable,
                      HloRunnerPjRtExecutable::TryUnwrap(*this, executable));

  return ExecuteReplicatedImpl(
      [&](absl::Span<const std::vector<PjRtBuffer*>> argument_buffer_slices)
          -> absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> {
        TF_ASSIGN_OR_RETURN(
            auto execution_results,
            wrapped_executable->pjrt_loaded_executable()->Execute(
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
    std::function<OpaqueExecutable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  TF_RET_CHECK(device_assignment->computation_count() == 1)
      << "Only single-computation execution is supported.";
  return ExecuteReplicatedImpl(
      [&](absl::Span<const std::vector<PjRtBuffer*>> argument_buffer_slices)
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
            TF_ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const executable,
                                HloRunnerPjRtExecutable::TryUnwrap(
                                    *this, executable_provider(i)));
            TF_ASSIGN_OR_RETURN(
                PjRtDevice * device_ptr,
                pjrt_client_->LookupDevice(
                    DeviceIdForInvocation(*device_assignment, i)));
            pool.Schedule([&per_replica_results, i, executable,
                           args = argument_buffer_slices[i], device_ptr]() {
              per_replica_results[i] =
                  executable->pjrt_loaded_executable()->ExecuteSharded(
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
        absl::Span<const std::vector<PjRtBuffer*>>)>
        execution_helper,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  TF_RET_CHECK(options.infeed_values.empty() ||
               options.infeed_values.size() == options.num_replicas);

  std::vector<PjRtDevice*> replica_devices(options.num_replicas, nullptr);
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffer_slices;
  argument_buffer_slices.reserve(options.num_replicas);
  for (int64_t i = 0; i < options.num_replicas; ++i) {
    // Amortize device lookup.
    TF_ASSIGN_OR_RETURN(PjRtDevice* const device_ptr,
                        pjrt_client_->LookupDevice(
                            DeviceIdForInvocation(*device_assignment, i)));
    replica_devices[i] = device_ptr;

    // Transfer literals to device.
    const int64_t argument_count = argument_count_provider(i);
    std::vector<std::unique_ptr<PjRtBuffer>> replica_buffers;
    replica_buffers.reserve(argument_count);
    for (int64_t arg_index = 0; arg_index < argument_count; arg_index++) {
      const Literal* const argument = argument_provider(i, arg_index);
      TF_RET_CHECK(argument != nullptr);
      TF_RET_CHECK(argument->shape().has_layout())
          << "Replica " << i << " argument " << arg_index << " has no layout.";
      TF_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                          device_ptr->default_memory_space());
      TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> assignment,
                          TransferLiteralToDevice(*argument, memory_space,
                                                  argument->shape().layout()));
      replica_buffers.push_back(std::move(assignment));
    }
    argument_buffer_slices.push_back(std::move(replica_buffers));
  }

  // Handle infeed and outfeed.
  const bool has_infeed = !options.infeed_values.empty();
  const bool has_outfeed = ShapeUtil::IsInitialized(options.outfeed_shape);
  std::unique_ptr<tsl::thread::ThreadPool> pool = nullptr;
  absl::Mutex infeed_outfeed_status_mu;
  absl::Status infeed_outfeed_status = absl::OkStatus();
  if (has_infeed || has_outfeed) {
    // One infeed per infeed value and one outfeed per replica.
    const int64_t num_threads =
        options.infeed_values.size() + (has_outfeed ? options.num_replicas : 0);
    pool = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "infeed_outfeed", num_threads);
  }
  if (has_infeed) {
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      pool->Schedule(
          [device = replica_devices[i],
           &infeed_literal = *ABSL_DIE_IF_NULL(options.infeed_values[i]),
           infeed_steps = options.infeed_steps, &infeed_outfeed_status_mu,
           &infeed_outfeed_status]() {
            VLOG(1) << "Starting infeed on device " << device->ToString();
            absl::Status per_feed_status = absl::OkStatus();
            for (int64_t step = 1; infeed_steps < 0 || step <= infeed_steps;
                 ++step) {
              per_feed_status.Update(device->TransferToInfeed(infeed_literal));
              if (step % 100 == 0) {
                VLOG(1) << "Infeed step " << step;
              }
            }
            absl::MutexLock lock(&infeed_outfeed_status_mu);
            infeed_outfeed_status.Update(per_feed_status);
          });
    }
  }
  if (has_outfeed) {
    if (options.outfeed_values != nullptr) {
      options.outfeed_values->resize(options.num_replicas);
    }
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      pool->Schedule([i, device = replica_devices[i],
                      outfeed_values = options.outfeed_values,
                      outfeed_shape = options.outfeed_shape,
                      infeed_steps = options.infeed_steps,
                      &infeed_outfeed_status_mu, &infeed_outfeed_status]() {
        VLOG(1) << "Starting outfeed on device " << device->ToString();
        absl::Status per_feed_status = absl::OkStatus();
        for (int64_t step = 1; infeed_steps < 0 || step <= infeed_steps;
             ++step) {
          Literal literal(outfeed_shape);
          per_feed_status.Update(device->TransferFromOutfeed(&literal));
          if (outfeed_values != nullptr) {
            outfeed_values->at(i) = std::move(literal);
          }
          if (step % 100 == 0) {
            VLOG(1) << "Outfeed step " << step;
          }
        }
        absl::MutexLock lock(&infeed_outfeed_status_mu);
        infeed_outfeed_status.Update(per_feed_status);
      });
    }
  }

  VLOG(1) << "Replicated execution started";
  TF_ASSIGN_OR_RETURN(
      const std::vector<std::unique_ptr<PjRtBuffer>> result_buffers,
      execution_helper(BufferMatToPointerMat(argument_buffer_slices)));
  VLOG(1) << "Replicated execution terminated";

  // Get the result from execution.
  std::vector<Literal> result_literals;
  result_literals.reserve(options.num_replicas);
  for (int64_t i = 0; i < options.num_replicas; ++i) {
    TF_ASSIGN_OR_RETURN(Literal literal,
                        TransferLiteralFromDevice(*result_buffers[i]));
    result_literals.push_back(std::move(literal));
  }

  // Join infeed and outfeed threads, if they exist. The thread pool's threads
  // are joined on destruction. No-op otherwise.
  pool = nullptr;
  TF_RETURN_IF_ERROR(infeed_outfeed_status);

  return std::move(result_literals);
}

absl::string_view HloRunnerPjRt::Name() const { return "HloRunnerPjRt"; }

bool HloRunnerPjRt::HasProperty(const HloRunnerPropertyTag::Type tag) const {
  if (tag == HloRunnerPropertyTag::kUsingGpuRocm) {
    return pjrt_client_->platform_name() == xla::RocmName();
  }
  if (tag == HloRunnerPropertyTag::kCpu) {
    return pjrt_client_->platform_name() == xla::CpuName();
  }
  return false;
}

absl::StatusOr<absl::Nonnull<const HloModule*>>
HloRunnerPjRt::HloModuleFromWrapped(const OpaqueExecutable* wrapped) const {
  TF_ASSIGN_OR_RETURN(
      const HloRunnerPjRtExecutable* const hlo_runner_pjrt_executable,
      HloRunnerPjRtExecutable::TryUnwrap(*this, wrapped));
  const PjRtLoadedExecutable* const executable =
      hlo_runner_pjrt_executable->pjrt_loaded_executable();
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> modules,
                      executable->GetHloModules());
  if (!modules.empty()) {
    return modules.front().get();
  }
  return absl::NotFoundError("PjRtLoadedExecutable has no modules.");
}

}  // namespace xla
