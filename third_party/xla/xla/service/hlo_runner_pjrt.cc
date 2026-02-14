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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/path.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

namespace {

absl::Status SanityCheckParameterLayouts(
    const absl::Span<const ShapeLayout> layouts) {
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
    const absl::Span<const ShapeLayout> layouts) {
  TF_RETURN_IF_ERROR(SanityCheckParameterLayouts(layouts));
  // Strictly, we only need to flatten tuples with mixed host/device leaves
  // because mixed host/device PjRtBuffer's are not supported.
  // However, splitting all tuples makes the code simpler and is the way
  // PJRT is commonly used by JAX.
  return layouts.size() == 1 && layouts[0].shape().IsTuple();
}

absl::StatusOr<std::vector<Layout>> FlattenedParameterLayouts(
    const absl::Span<const ShapeLayout> layouts) {
  std::vector<Layout> result;
  for (const ShapeLayout& layout : layouts) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        layout.shape(),
        [&result](const Shape& subshape,
                  const ShapeIndex& index) -> absl::Status {
          if (subshape.IsTuple()) {
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::UnimplementedError(
                absl::StrFormat("FlattenedParameterLayouts doesn't support "
                                "token or opaque parameters (got: %s)",
                                subshape.ToString(true)));
          }
          if (!subshape.has_layout()) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "FlattenedParameterLayouts can only be called after all "
                "parameters have layouts assigned (got: %s)",
                subshape.ToString(true)));
          }
          result.push_back(subshape.layout());
          return absl::OkStatus();
        }));
  }
  return result;
}

absl::StatusOr<ExecuteOptions> GenerateExecuteOptions(const HloModule& module) {
  ExecuteOptions execute_options;
  execute_options.strict_shape_checking = true;
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

absl::StatusOr<PjRtMemorySpace* absl_nonnull> GetMemorySpaceFromLayout(
    PjRtDevice* absl_nonnull const device, const Layout& layout) {
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
  // Construct an internal executable with a PjRt executable that requires
  // loading prior to execution.
  HloRunnerPjRtExecutable(const HloRunnerPjRt* absl_nonnull creator,
                          std::unique_ptr<PjRtExecutable> executable)
      : OpaqueExecutable(creator), executable_(std::move(executable)) {}
  // Construct an internal executable with a pre-loaded PjRt executable. This
  // only exists to support PjRt clients that do not implement Compile() and
  // DeserializeExecutable() functionality.
  HloRunnerPjRtExecutable(
      const HloRunnerPjRt* absl_nonnull creator,
      std::unique_ptr<PjRtLoadedExecutable> loaded_executable)
      : OpaqueExecutable(creator),
        loaded_executable_(std::move(loaded_executable)) {}

  PjRtExecutable* executable() const {
    if (executable_ != nullptr) {
      return executable_.get();
    }
    return loaded_executable_->GetExecutable();
  }
  absl::StatusOr<PjRtLoadedExecutable* absl_nonnull> GetOrLoadExecutable(
      PjRtClient* const absl_nonnull client) {
    if (loaded_executable_ == nullptr) {
      TF_ASSIGN_OR_RETURN(loaded_executable_,
                          client->Load(std::move(executable_), LoadOptions()));
    }
    return loaded_executable_.get();
  }

  static absl::StatusOr<HloRunnerPjRtExecutable*> TryUnwrap(
      const HloRunnerPjRt& runner,
      OpaqueExecutable* absl_nonnull const wrapped) {
    return OpaqueExecutable::TryUnwrap<HloRunnerPjRtExecutable>(runner,
                                                                wrapped);
  }
  static absl::StatusOr<const HloRunnerPjRtExecutable*> TryUnwrap(
      const HloRunnerPjRt& runner,
      const OpaqueExecutable* absl_nonnull const wrapped) {
    return OpaqueExecutable::TryUnwrap<HloRunnerPjRtExecutable>(runner,
                                                                wrapped);
  }

 private:
  // If the executable is loaded, executable_ is null and loaded_executable_ is
  // non-null. The executable is loaded if and only if required, and is cached
  // thereafter.
  std::unique_ptr<PjRtExecutable> executable_;
  std::unique_ptr<PjRtLoadedExecutable> loaded_executable_;
};

// Obtains the best device assignment for the given executable.
// If the executable was compiled with a device assignment, that assignment is
// returned. Otherwise, the static device assignment is pulled from the module
// and returned instead. If that does not exist either, the default device
// assignment is computed and returned.
absl::StatusOr<DeviceAssignment> GetBestDeviceAssignment(
    HloRunnerPjRtExecutable* const executable, PjRtClient& client) {
  ASSIGN_OR_RETURN(CompileOptions compile_options,
                   executable->executable()->GetCompileOptions());
  if (compile_options.executable_build_options.has_device_assignment()) {
    return compile_options.executable_build_options.device_assignment();
  }

  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->executable()->GetHloModules());
  TF_RET_CHECK(hlo_modules.size() == 1);
  return GetStaticDeviceAssignmentOrComputeDefault(*hlo_modules.front(),
                                                   client);
}

}  // namespace

HloRunnerPjRt::HloRunnerPjRt(std::unique_ptr<PjRtClient> pjrt_client)
    : pjrt_client_(std::move(pjrt_client)) {}

absl::StatusOr<CompileOptions> HloRunnerPjRt::GenerateDefaultCompileOptions(
    HloModule* module, bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment device_assignment,
      GetStaticDeviceAssignmentOrComputeDefault(*module, *pjrt_client_));

  CompileOptions compile_options;

  // LINT.IfChange
  compile_options.executable_build_options.set_num_partitions(
      module->config().num_partitions());
  compile_options.executable_build_options.set_num_replicas(
      module->config().replica_count());
  compile_options.executable_build_options.set_use_spmd_partitioning(
      module->config().use_spmd_partitioning());
  compile_options.executable_build_options
      .set_allow_spmd_sharding_propagation_to_parameters(
          module->config().allow_spmd_sharding_propagation_to_parameters());
  compile_options.executable_build_options
      .set_allow_spmd_sharding_propagation_to_output(
          module->config().allow_spmd_sharding_propagation_to_output());
  compile_options.executable_build_options.set_use_auto_spmd_partitioning(
      module->config().use_auto_spmd_partitioning());
  compile_options.executable_build_options
      .set_auto_spmd_partitioning_mesh_shape(std::vector<int64_t>(
          module->config().auto_spmd_partitioning_mesh_shape().begin(),
          module->config().auto_spmd_partitioning_mesh_shape().end()));
  compile_options.executable_build_options.set_auto_spmd_partitioning_mesh_ids(
      std::vector<int64_t>(
          module->config().auto_spmd_partitioning_mesh_ids().begin(),
          module->config().auto_spmd_partitioning_mesh_ids().end()));
  compile_options.executable_build_options.set_exec_time_optimization_effort(
      module->config().exec_time_optimization_effort());
  compile_options.executable_build_options.set_memory_fitting_effort(
      module->config().memory_fitting_effort());
  compile_options.executable_build_options.set_optimization_level(
      module->config().optimization_level());
  compile_options.executable_build_options.set_memory_fitting_level(
      module->config().memory_fitting_level());
  compile_options.executable_build_options.set_deduplicate_hlo(
      module->config().deduplicate_hlo());
  *compile_options.executable_build_options.mutable_debug_options() =
      module->config().debug_options();
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);
  compile_options.executable_build_options.set_fdo_profile(
      std::string(module->config().fdo_profile()));
  compile_options.executable_build_options.set_alias_passthrough_params(
      module->config().alias_passthrough_params());
  compile_options.executable_build_options.set_device_memory_size(
      module->config().device_memory_size());
  compile_options.executable_build_options.set_use_shardy_partitioner(
      module->config().use_shardy_partitioner());
  // LINT.ThenChange(
  //   //xla/service/hlo_module_util.cc
  // )

  compile_options.executable_build_options.set_run_backend_only(
      !run_hlo_passes);
  *compile_options.executable_build_options.mutable_comp_envs() =
      module->comp_envs();

  std::vector<Shape> parameter_shapes;
  parameter_shapes.reserve(
      module->entry_computation_layout().parameter_count());
  for (const ShapeLayout& shape_layout :
       module->entry_computation_layout().parameter_layouts()) {
    parameter_shapes.push_back(shape_layout.shape());
  }
  compile_options.argument_layouts = parameter_shapes;

  TF_ASSIGN_OR_RETURN(
      bool flatten,
      MustFlattenInputTuple(
          module->entry_computation_layout().parameter_layouts()));
  compile_options.parameter_is_tupled_arguments = flatten;

  compile_options.executable_build_options.set_result_layout(
      module->entry_computation_layout().result_shape());

  return compile_options;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::TransferLiteralsToDevice(
    const absl::Span<const ShapeLayout> layouts,
    const absl::Span<const Literal* const> literals) {
  // Note: This function is used for single (default) device execution.
  if (pjrt_client_->addressable_device_count() <= kDeviceIdx) {
    return absl::InternalError("No addressable devices available");
  }
  PjRtDevice* device = pjrt_client_->addressable_devices()[kDeviceIdx];
  TF_RET_CHECK(device != nullptr)
      << "Device with ordinal " << kDeviceIdx << " is null.";

  TF_ASSIGN_OR_RETURN(bool flatten, MustFlattenInputTuple(layouts));
  TF_ASSIGN_OR_RETURN(std::vector<Layout> parameter_layouts,
                      FlattenedParameterLayouts(layouts));

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
      TF_ASSIGN_OR_RETURN(PjRtMemorySpace* absl_nonnull memory_space,
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

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::TransferLiteralsToDevice(
    absl::Span<const Literal* const> literals) {
  std::vector<ShapeLayout> layouts;
  layouts.reserve(literals.size());
  for (const Literal* literal : literals) {
    layouts.push_back(ShapeLayout(literal->shape()));
  }
  return TransferLiteralsToDevice(layouts, literals);
}

absl::StatusOr<Literal> HloRunnerPjRt::TransferLiteralsFromDevice(
    absl::Span<const std::unique_ptr<PjRtBuffer>> output_buffers,
    const bool untuple_result) {
  if (!untuple_result) {
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

absl::StatusOr<Literal> HloRunnerPjRt::Execute(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(const std::unique_ptr<OpaqueExecutable> executable,
                      CreateExecutable(std::move(module), run_hlo_passes));
  return HloRunnerInterface::ExecuteWithExecutable(executable.get(), arguments);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
HloRunnerPjRt::ExecuteWithDeviceBuffers(
    OpaqueExecutable* executable,
    const std::vector<std::unique_ptr<PjRtBuffer>>& arguments,
    const ExecuteOptions* execute_options) {
  TF_ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const wrapped_executable,
                      HloRunnerPjRtExecutable::TryUnwrap(*this, executable));
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      wrapped_executable->executable()->GetHloModules());
  TF_RET_CHECK(hlo_modules.size() == 1);
  const HloModule& module = *hlo_modules.front();

  std::optional<ExecuteOptions> generated_execute_options = std::nullopt;
  if (execute_options == nullptr) {
    TF_ASSIGN_OR_RETURN(generated_execute_options,
                        GenerateExecuteOptions(module));
    execute_options = &*generated_execute_options;
  }

  TF_ASSIGN_OR_RETURN(
      PjRtLoadedExecutable * pjrt_executable,
      wrapped_executable->GetOrLoadExecutable(pjrt_client_.get()));
  std::vector<PjRtBuffer*> argument_ptrs = BufferVecToPointerVec(arguments);
  std::optional<Future<>> returned_future = {};
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> buffers,
      pjrt_executable->ExecuteSharded(
          argument_ptrs, pjrt_client_->addressable_devices()[kDeviceIdx],
          *execute_options, returned_future, true));
  if (returned_future.has_value()) {
    TF_RETURN_IF_ERROR(returned_future->Await());
  }
  return buffers;
}

absl::StatusOr<std::vector<absl::StatusOr<Literal>>>
HloRunnerPjRt::ExecuteWithExecutable(OpaqueExecutable* executable,
                                     absl::Span<const Literal* const> arguments,
                                     int64_t num_repeats) {
  TF_ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const wrapped_executable,
                      HloRunnerPjRtExecutable::TryUnwrap(*this, executable));

  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      wrapped_executable->executable()->GetHloModules());
  TF_RET_CHECK(hlo_modules.size() == 1);
  const HloModule& module = *hlo_modules.front();

  TF_ASSIGN_OR_RETURN(ExecuteOptions execute_options,
                      GenerateExecuteOptions(module));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> argument_handles,
      TransferLiteralsToDevice(
          module.entry_computation_layout().parameter_layouts(), arguments));

  std::vector<absl::StatusOr<Literal>> results;
  results.reserve(num_repeats);
  for (int64_t i = 0; i < num_repeats; ++i) {
    absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers =
        ExecuteWithDeviceBuffers(wrapped_executable, argument_handles,
                                 &execute_options);
    if (!output_buffers.ok()) {
      results.push_back(output_buffers.status());
      continue;
    }
    results.push_back(TransferLiteralsFromDevice(
        *std::move(output_buffers), module.result_shape().IsTuple()));
  }
  return results;
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
HloRunnerPjRt::CreateExecutable(std::unique_ptr<HloModule> module,
                                bool run_hlo_passes) {
  TF_ASSIGN_OR_RETURN(
      CompileOptions compile_options,
      GenerateDefaultCompileOptions(module.get(), run_hlo_passes));
  XlaComputation computation(module->ToProto());

  // Attempt to compile without loading. If that fails, fall back to compile +
  // load functionality instead.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> pjrt_executable =
      pjrt_client_->Compile(computation, compile_options);
  if (pjrt_executable.ok()) {
    absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> hlo_modules =
        pjrt_executable->get()->GetHloModules();
    if (hlo_modules.ok() && !hlo_modules->empty()) {
      std::shared_ptr<HloModule> exe_module = (*hlo_modules)[0];
      exe_module->mutable_config().set_seed(module->config().seed());
    }
    return std::make_unique<HloRunnerPjRtExecutable>(
        this, *std::move(pjrt_executable));
  }
  if (!absl::IsUnimplemented(pjrt_executable.status())) {
    return pjrt_executable.status();
  }

  // Fall back to compile + load if Compile() was not implemented.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> pjrt_loaded_executable,
      pjrt_client_->CompileAndLoad(computation, std::move(compile_options)));
  if (pjrt_loaded_executable != nullptr) {
    absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> hlo_modules =
        pjrt_loaded_executable->GetHloModules();
    if (hlo_modules.ok() && !hlo_modules->empty()) {
      std::shared_ptr<HloModule> exe_module = (*hlo_modules)[0];
      exe_module->mutable_config().set_seed(module->config().seed());
    }
  }
  return std::make_unique<HloRunnerPjRtExecutable>(
      this, std::move(pjrt_loaded_executable));
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
HloRunnerPjRt::DeserializeExecutable(const absl::string_view serialized) const {
  // TODO: b/237720161 - According to the comment in the base class, the
  // `options` argument is mandatory. However, our implementation is capable of
  // handling the default case where it is not present. The options are
  // serialized with the executable and we can read them from there.
  // Remove this comment once the bug is closed.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> pjrt_executable =
      pjrt_client_->DeserializeExecutable(serialized, /*options=*/std::nullopt);
  if (pjrt_executable.ok()) {
    return std::make_unique<HloRunnerPjRtExecutable>(
        this, *std::move(pjrt_executable));
  }
  if (!absl::IsUnimplemented(pjrt_executable.status())) {
    return pjrt_executable.status();
  }

  // Fall back to deserialize + load if DeserializeExecutable() was not
  // implemented. This is similar to how we handle CreateExecutable above.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> pjrt_loaded_executable,
      pjrt_client_->LoadSerializedExecutable(
          serialized, /*options=*/std::nullopt, LoadOptions()));
  return std::make_unique<HloRunnerPjRtExecutable>(
      this, std::move(pjrt_loaded_executable));
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const HloRunnerInterface::ReplicatedExecuteOptions& options) {
  return ExecuteReplicated(std::move(module), options,
                           /*device_assignment=*/nullptr);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));
  return ExecuteReplicatedWithExecutable(executable.get(), options,
                                         device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerPjRt::ExecuteReplicatedWithExecutable(
    OpaqueExecutable* const absl_nonnull executable,
    const ReplicatedExecuteOptions& options) {
  return ExecuteReplicatedWithExecutable(executable, options,
                                         /*device_assignment=*/nullptr);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerPjRt::ExecuteReplicatedWithExecutable(
    OpaqueExecutable* const absl_nonnull executable,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const wrapped_executable,
                   HloRunnerPjRtExecutable::TryUnwrap(*this, executable));

  // If a device assignment is provided, use it. Otherwise, use the one from the
  // executable, or if that is not available, generate a default one.
  std::optional<DeviceAssignment> device_assignment_storage = std::nullopt;
  if (device_assignment == nullptr) {
    ASSIGN_OR_RETURN(
        device_assignment_storage,
        GetBestDeviceAssignment(wrapped_executable, *pjrt_client_));
    device_assignment = &*device_assignment_storage;
  }
  CHECK_NE(device_assignment, nullptr);

  xla::ExecuteOptions execute_options;
  return ExecuteReplicatedImpl(
      [&](absl::Span<const std::vector<PjRtBuffer*>> argument_buffer_slices,
          absl::AnyInvocable<OpaqueExecutable*(int64_t)>
              executable_provider_arg,
          absl::Span<PjRtDevice* const>)
          -> absl::StatusOr<
              std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> {
        TF_ASSIGN_OR_RETURN(
            PjRtLoadedExecutable * pjrt_executable,
            wrapped_executable->GetOrLoadExecutable(pjrt_client_.get()));
        return pjrt_executable->Execute(argument_buffer_slices,
                                        execute_options);
      },
      [&](int64_t replica) { return wrapped_executable; },
      [&](int64_t replica) { return options.arguments.size(); },
      [&](int64_t replica, int64_t index) { return options.arguments[index]; },
      options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicated(
    absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
    absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
    absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
    const HloRunnerInterface::ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  CHECK_GT(options.num_devices, 0);
  ASSIGN_OR_RETURN(
      HloRunnerPjRtExecutable* const wrapped_executable_device0,
      HloRunnerPjRtExecutable::TryUnwrap(*this, executable_provider(0)));

  // NB: we assume all executables have the same device assignments.  If a
  // device assignment is provided, use it. Otherwise, use the one from the
  // first device's executable, or if that is not available, generate a default
  // one.
  std::optional<DeviceAssignment> device_assignment_storage = std::nullopt;
  if (device_assignment == nullptr) {
    ASSIGN_OR_RETURN(
        device_assignment_storage,
        GetBestDeviceAssignment(wrapped_executable_device0, *pjrt_client_));
    device_assignment = &*device_assignment_storage;
  }
  CHECK_NE(device_assignment, nullptr);

  return ExecuteReplicatedImpl(
      [&](absl::Span<const std::vector<PjRtBuffer*>> argument_buffer_slices,
          absl::AnyInvocable<OpaqueExecutable*(int64_t)>
              executable_provider_arg,
          absl::Span<PjRtDevice* const> id_to_device_ptr)
          -> absl::StatusOr<
              std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> {
        TF_RET_CHECK(options.use_threads);

        // The underlying data is modified concurrently. We don't need to
        // protect access as each replica writes only to its own slot.
        std::vector<absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>>
            per_replica_results(options.num_devices);
        absl::c_fill(per_replica_results,
                     absl::InternalError("No result for replica."));

        {
          RunId run_id = RunId::CreateUniqueId();
          // NB: `pool` is joined on destruction.
          tsl::thread::ThreadPool pool(tsl::Env::Default(), "replicas",
                                       options.num_devices);
          for (int64_t i = 0; i < options.num_devices; ++i) {
            for (const PjRtBuffer* const buffer : argument_buffer_slices[i]) {
              TF_RET_CHECK(buffer != nullptr);
            }
            TF_ASSIGN_OR_RETURN(HloRunnerPjRtExecutable* const executable,
                                HloRunnerPjRtExecutable::TryUnwrap(
                                    *this, executable_provider_arg(i)));
            TF_ASSIGN_OR_RETURN(
                PjRtLoadedExecutable * pjrt_executable,
                executable->GetOrLoadExecutable(pjrt_client_.get()));
            pool.Schedule([&per_replica_results, i, pjrt_executable,
                           // Add 1 to the launch ID to avoid a zero, because
                           // zero means that launch id is not set.
                           launch_id = run_id.ToInt() + 1,
                           args = argument_buffer_slices[i],
                           device_ptr = id_to_device_ptr[i]]() {
              std::optional<Future<>> returned_future = {};
              xla::ExecuteOptions options;
              options.launch_id = launch_id;
              per_replica_results[i] = pjrt_executable->ExecuteSharded(
                  args, device_ptr, options,
                  /*returned_future=*/returned_future,
                  /*fill_future=*/true);
              if (returned_future.has_value()) {
                if (const absl::Status& status = returned_future->Await();
                    !status.ok()) {
                  per_replica_results[i] = status;
                }
              }
            });
          }
        }
        // Aggregate results.
        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results;
        for (int64_t i = 0; i < options.num_devices; ++i) {
          absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>&
              replica_result = per_replica_results[i];
          if (!replica_result.ok()) {
            return replica_result.status();
          }
          results.push_back(*std::move(replica_result));
        }
        return results;
      },
      std::move(executable_provider), std::move(argument_count_provider),
      std::move(argument_provider), options, device_assignment);
}

absl::StatusOr<std::vector<Literal>> HloRunnerPjRt::ExecuteReplicatedImpl(
    absl::AnyInvocable<
        absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(
            absl::Span<const std::vector<PjRtBuffer*>>,
            absl::AnyInvocable<OpaqueExecutable*(int64_t)>,
            absl::Span<PjRtDevice* const>)>
        execution_helper,
    absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
    absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
    absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  TF_RET_CHECK(options.infeed_values.empty() ||
               options.infeed_values.size() == options.num_devices);
  TF_RET_CHECK(device_assignment != nullptr);

  std::vector<PjRtDevice*> id_to_device_ptr(options.num_devices, nullptr);
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffer_slices;
  argument_buffer_slices.reserve(options.num_devices);
  std::vector<bool> is_tuple_result(options.num_devices, false);
  for (int64_t i = 0; i < options.num_devices; ++i) {
    // Amortize device lookup.
    TF_ASSIGN_OR_RETURN(PjRtDevice* const device_ptr,
                        pjrt_client_->LookupDevice(
                            DeviceIdForInvocation(*device_assignment, i)));
    id_to_device_ptr[i] = device_ptr;

    // Get the entry layout.
    OpaqueExecutable* const wrapped_executable = executable_provider(i);
    TF_ASSIGN_OR_RETURN(const HloModule* const module,
                        HloModuleFromWrapped(wrapped_executable));
    const ComputationLayout& ecl = module->entry_computation_layout();
    is_tuple_result[i] = ecl.result_shape().IsTuple();

    // Transfer literals to device.
    const int64_t argument_count = argument_count_provider(i);
    std::vector<std::unique_ptr<PjRtBuffer>> replica_buffers;
    replica_buffers.reserve(argument_count);
    for (int64_t arg_index = 0; arg_index < argument_count; arg_index++) {
      const Literal* const argument = argument_provider(i, arg_index);
      TF_RET_CHECK(argument != nullptr);
      const ShapeLayout& layout = ecl.parameter_layout(arg_index);
      TF_RET_CHECK(layout.LayoutIsSet());

      TF_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                          device_ptr->default_memory_space());
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> assignment,
          TransferLiteralToDevice(*argument, memory_space, layout.layout()));
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
        options.infeed_values.size() + (has_outfeed ? options.num_devices : 0);
    pool = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "infeed_outfeed", num_threads);
  }
  if (has_infeed) {
    for (int64_t i = 0; i < options.num_devices; ++i) {
      pool->Schedule(
          [device = id_to_device_ptr[i],
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
            absl::MutexLock lock(infeed_outfeed_status_mu);
            infeed_outfeed_status.Update(per_feed_status);
          });
    }
  }
  if (has_outfeed) {
    if (options.outfeed_values != nullptr) {
      options.outfeed_values->resize(options.num_devices);
    }
    for (int64_t i = 0; i < options.num_devices; ++i) {
      pool->Schedule([i, device = id_to_device_ptr[i],
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
        absl::MutexLock lock(infeed_outfeed_status_mu);
        infeed_outfeed_status.Update(per_feed_status);
      });
    }
  }

  VLOG(1) << "Replicated execution started";
  TF_ASSIGN_OR_RETURN(
      const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
          result_buffers,
      execution_helper(BufferMatToPointerMat(argument_buffer_slices),
                       std::move(executable_provider), id_to_device_ptr));
  VLOG(1) << "Replicated execution terminated";

  // Get the result from execution.
  std::vector<Literal> result_literals;
  result_literals.reserve(options.num_devices);
  for (int64_t i = 0; i < options.num_devices; ++i) {
    TF_ASSIGN_OR_RETURN(
        Literal literal,
        TransferLiteralsFromDevice(result_buffers[i], is_tuple_result[i]));
    result_literals.push_back(std::move(literal));
  }

  // Join infeed and outfeed threads, if they exist. The thread pool's threads
  // are joined on destruction. No-op otherwise.
  pool = nullptr;
  TF_RETURN_IF_ERROR(infeed_outfeed_status);

  return std::move(result_literals);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
HloRunnerPjRt::TransferLiteralToDevice(
    const Literal& literal, PjRtMemorySpace* absl_nonnull const memory_space,
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

absl::StatusOr<Literal> HloRunnerPjRt::TransferLiteralFromDevice(
    PjRtBuffer& buffer) {
  TF_RETURN_IF_ERROR(buffer.GetReadyFuture().Await());

  // Implementations of ToLiteralSync() do not support empty tuples. Since an
  // empty tuple literal is easy to construct, we do so here.
  if (const Shape& on_device_shape = buffer.on_device_shape();
      on_device_shape.IsTuple() && on_device_shape.tuple_shapes().size() == 0) {
    return LiteralUtil::MakeTuple({});
  }
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                      buffer.ToLiteral().Await());
  return std::move(*literal);
}

absl::string_view HloRunnerPjRt::Name() const { return "HloRunnerPjRt"; }

bool HloRunnerPjRt::HasProperty(const HloRunnerPropertyTag::Type tag) const {
  if (tag == HloRunnerPropertyTag::kUsingGpuRocm) {
    return pjrt_client_->platform_name() == RocmName();
  }
  if (tag == HloRunnerPropertyTag::kCpu) {
    return pjrt_client_->platform_name() == CpuName();
  }
  if (tag == HloRunnerPropertyTag::kUsingGpuCuda) {
    return pjrt_client_->platform_name() == CudaName();
  }
  return false;
}

absl::StatusOr<const HloModule* absl_nonnull>
HloRunnerPjRt::HloModuleFromWrapped(const OpaqueExecutable* wrapped) const {
  TF_ASSIGN_OR_RETURN(
      const HloRunnerPjRtExecutable* const hlo_runner_pjrt_executable,
      HloRunnerPjRtExecutable::TryUnwrap(*this, wrapped));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<HloModule>> modules,
      hlo_runner_pjrt_executable->executable()->GetHloModules());
  if (!modules.empty()) {
    return modules.front().get();
  }
  return absl::NotFoundError("PjRtLoadedExecutable has no modules.");
}

bool HloRunnerPjRt::ExecutablesAreEquivalent(
    const OpaqueExecutable* absl_nonnull lhs,
    const OpaqueExecutable* absl_nonnull rhs) const {
  constexpr auto kFingerprint =
      [](const absl::StatusOr<const HloRunnerPjRtExecutable*> wrapped)
      -> absl::StatusOr<std::string> {
    TF_ASSIGN_OR_RETURN(const HloRunnerPjRtExecutable* const executable,
                        wrapped);
    return executable->executable()->FingerprintExecutable();
  };

  const absl::StatusOr<std::string> lhs_fingerprint =
      kFingerprint(HloRunnerPjRtExecutable::TryUnwrap(*this, lhs));
  if (!lhs_fingerprint.ok()) {
    return false;
  }
  const absl::StatusOr<std::string> rhs_fingerprint =
      kFingerprint(HloRunnerPjRtExecutable::TryUnwrap(*this, rhs));
  if (!rhs_fingerprint.ok()) {
    return false;
  }
  return *lhs_fingerprint == *rhs_fingerprint;
}

absl::StatusOr<DeviceAssignment> HloRunnerPjRt::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return pjrt_client_->GetDefaultDeviceAssignment(num_replicas, num_partitions);
}

// HloRunnerPjRt implementations for AoT use cases (i.e., compilation being a
// phase distinct from execution):

namespace {

std::string SerializeDeterministically(const tsl::protobuf::Message& message) {
  const size_t size = message.ByteSizeLong();
  std::string buffer;
  buffer.resize(size);
  tsl::protobuf::io::StringOutputStream string_stream(&buffer);
  tsl::protobuf::io::CodedOutputStream coded_stream(&string_stream);
  coded_stream.SetSerializationDeterministic(true);
  message.SerializeWithCachedSizes(&coded_stream);
  return buffer;
}

std::string MakeFilename(const HloModule& module, const bool run_hlo_passes) {
  // Fingerprint the HLO module, including its names.
  tsl::Fprint128 fingerprint =
      tsl::Fingerprint128(module.ToString(HloPrintOptions::Default()));
  // Fingerprint the run_hlo_passes flag, as this is an input from the test.
  fingerprint = tsl::FingerprintCat128(
      fingerprint, tsl::Fingerprint128(run_hlo_passes ? "true" : "false"));
  // Fingerprint the HLO module's compilation environment, as this is not
  // captured in the HLO module fingerprint.
  fingerprint = tsl::FingerprintCat128(
      fingerprint, tsl::Fingerprint128(SerializeDeterministically(
                       module.comp_envs().ToProto())));

  // Convert the fingerprint into a hex string and concatenate it with the .bin
  // extension.
  const std::array<char, 16> fingerprint_bytes =
      tsl::Fprint128ToBytes(fingerprint);
  const absl::string_view fingerprint_bytes_view(fingerprint_bytes.data(),
                                                 fingerprint_bytes.size());
  return absl::StrCat(absl::BytesToHexString(fingerprint_bytes_view), ".bin");
}

inline absl::StatusOr<DeviceAssignment> AotDefaultDeviceAssignment(
    int num_replicas, int num_partitions) {
  DeviceAssignment device_assignment(num_replicas, num_partitions);
  device_assignment.FillIota(0);
  return device_assignment;
}

}  // namespace

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
CompilePhaseHloRunnerPjRt::CreateExecutable(std::unique_ptr<HloModule> module,
                                            const bool run_hlo_passes) {
  const std::string path =
      tsl::io::JoinPath(artifact_dir_, MakeFilename(*module, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<OpaqueExecutable> wrapped_executable,
      HloRunnerPjRt::CreateExecutable(std::move(module), run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      HloRunnerPjRtExecutable* const executable,
      HloRunnerPjRtExecutable::TryUnwrap(*this, wrapped_executable.get()));

  TF_ASSIGN_OR_RETURN(const std::string serialized_executable,
                      executable->executable()->SerializeExecutable());
  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), path, serialized_executable));
  return wrapped_executable;
}

absl::StatusOr<DeviceAssignment>
CompilePhaseHloRunnerPjRt::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return AotDefaultDeviceAssignment(num_replicas, num_partitions);
}

absl::StatusOr<std::unique_ptr<OpaqueExecutable>>
ExecutePhaseHloRunnerPjRt::CreateExecutable(std::unique_ptr<HloModule> module,
                                            const bool run_hlo_passes) {
  const std::string filename = MakeFilename(*module, run_hlo_passes);
  const std::string path = tsl::io::JoinPath(artifact_dir_, filename);
  std::string serialized_executable;
  if (const absl::Status status = tsl::ReadFileToString(
          tsl::Env::Default(), path, &serialized_executable);
      !status.ok()) {
    if (!compile_if_not_found_) {
      return absl::NotFoundError(absl::StrCat(
          "Failed to read serialized executable. ", status.message()));
    }
    LOG(INFO) << "Failed to read serialized executable. " << status;
    return HloRunnerPjRt::CreateExecutable(std::move(module), run_hlo_passes);
  }
  LOG(INFO) << "ExecutePhase deserializing " << module->name() << " from "
            << filename;

  // If fail_duplicate_loads_ is enabled, fail if we previously loaded an
  // executable at this path, and the file at this path exists.
  if (fail_duplicate_loads_) {
    if (const auto [unused_it, did_insert] =
            loaded_executable_paths_.insert(path);
        !did_insert) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "ExecutePhaseHloRunnerPjRt::CreateExecutable called with a module "
          "that loads an executable that was previously loaded. The module "
          "name is %s and the filename is %s. If this is intentional, please "
          "set fail_duplicate_loads to false. This error exists to snuff out "
          "accidental duplicate loads originating from fingerprint collisions. "
          "If you intended to load two different executables, this error "
          "indicates that their fingerprints are the same. If you wish to "
          "avoid this issue in a test, you can either force their fingerprints "
          "to be different through some superficial change in the module (e.g. "
          "the module name), or by disabling ahead-of-time compilation by "
          "setting aot_compile_test = False in the corresponding xla_test.",
          module->name(), filename));
    }
  }

  return DeserializeExecutable(serialized_executable);
}

absl::StatusOr<DeviceAssignment>
ExecutePhaseHloRunnerPjRt::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return AotDefaultDeviceAssignment(num_replicas, num_partitions);
}

}  // namespace xla
