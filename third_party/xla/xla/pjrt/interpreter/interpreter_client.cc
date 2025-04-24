/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/interpreter/interpreter_client.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/expanders/cholesky_expander.h"
#include "xla/hlo/transforms/expanders/dynamic_index_splitter.h"
#include "xla/hlo/transforms/expanders/eigh_expander.h"
#include "xla/hlo/transforms/expanders/qr_expander.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/utils.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/triangular_solve_expander.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool ShapesMatch(const Shape& expected_shape, const Shape& actual_shape) {
  if (expected_shape.is_dynamic()) {
    return ShapeUtil::DynamicArrayShapeIsCompatible(actual_shape,
                                                    expected_shape);
  }
  return Shape::Equal().MinorToMajorOnlyInLayout()(expected_shape,
                                                   actual_shape);
}

absl::StatusOr<Shape> ChooseCompactLayoutForShape(const Shape& shape) {
  return LayoutUtil::GetWithDefaultLayout(shape);
}

// Handles custom_call ops during evaluation by routing them through the global
// CPU registry used by other CPU-based backends.
absl::StatusOr<Literal> HandleEvaluatorCustomCall(
    const HloInstruction* custom_call, absl::Span<const Literal*> operands) {
  // Find the target C function in the global registry.
  CustomCallTargetRegistry* const registry = CustomCallTargetRegistry::Global();
  void* const target_fn =
      registry->Lookup(custom_call->custom_call_target(), "Host");
  if (target_fn == nullptr) {
    return NotFound("Custom call target '%s' was not registered",
                    custom_call->custom_call_target());
  }

  // Populate pointers to operand and output literal data.
  std::vector<const void*> operand_data;
  operand_data.reserve(operands.size());
  for (const Literal* const literal : operands) {
    operand_data.push_back(literal->untyped_data());
  }
  Literal output = Literal::CreateFromShape(custom_call->shape());
  void* const output_data = output.untyped_data();

  // Call the target function matching the C ABI used by the CPU backends.
  auto* typed_fn = reinterpret_cast<void (*)(void*, const void**)>(target_fn);
  (*typed_fn)(output_data, operand_data.data());

  return std::move(output);
}

// Extract the input literals from the provided buffers.
//
// If there is a tupled argument and the arguments are not tupled, the extracted
// literals will be reconstituted into a tuple. The second element of the
// returned tuple is storage for the tupled literal, if required. Otherwise it
// is nullptr.
absl::StatusOr<std::tuple<std::vector<Literal*>, std::unique_ptr<Literal>>>
ExtractInterpreterInputLiteralsFromBuffers(
    const absl::Span<PjRtBuffer* const> buffers,
    const HloComputation& entry_computation,
    const bool parameter_is_tupled_arguments, const bool arguments_are_tupled) {
  std::vector<Literal*> literals;
  for (PjRtBuffer* const buffer : buffers) {
    InterpreterLiteralWrapperBuffer* interpreter_buffer =
        dynamic_cast<InterpreterLiteralWrapperBuffer*>(buffer);
    if (interpreter_buffer == nullptr) {
      return absl::InvalidArgumentError(
          "Interpreter only supports InterpreterLiteralWrapperBuffers");
    }
    literals.push_back(&interpreter_buffer->mutable_literal());
  }

  // Return early if arguments don't need to be re-tupled.
  if (!parameter_is_tupled_arguments || arguments_are_tupled) {
    return std::make_tuple(std::move(literals), nullptr);
  }

  if (entry_computation.num_parameters() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Interpreter expected a single tupled entry parameter, but got %d.",
        entry_computation.num_parameters()));
  }

  // Re-tuple input arguments. PjRt is commonly used in a mode where the input
  // tuple (if present) is flattened and passed as a vector of argument
  // buffers. The HloEvaluator expects the input to be tupled in these cases.
  //
  // This process invalidates the input literals and thus the input buffers
  // themselves.
  std::vector<Shape> shapes;
  shapes.reserve(literals.size());
  for (const Literal* literal : literals) {
    shapes.push_back(literal->shape());
  }
  auto tupled_arg_literal = std::make_unique<Literal>(
      ShapeUtil::MakeValidatedTupleShape(shapes).value(),
      /*allocate_arrays=*/false);
  for (int i = 0; i < literals.size(); ++i) {
    TF_RETURN_IF_ERROR(tupled_arg_literal->MoveFrom(std::move(*literals[i]),
                                                    /*dest_shape_index=*/{i}));
  }

  // Replace arg literals with the tupled literal.
  literals.clear();
  literals.push_back(tupled_arg_literal.get());
  return std::make_tuple(std::move(literals), std::move(tupled_arg_literal));
}

// The interpreter is a 1 replica, 1 partition = 1 device system.
inline DeviceAssignment MakeInterpreterDeviceAssignment() {
  DeviceAssignment assignment(1, 1);
  assignment(0, 0) = 0;
  return assignment;
}
}  // namespace

const InterpreterDescription& InterpreterDescription::Singleton() {
  static const InterpreterDescription* const singleton =
      new InterpreterDescription;
  return *singleton;
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
InterpreterLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  if (device_assignment_ == nullptr) {
    return absl::InvalidArgumentError(
        "Execute expects a non-null device_assignment");
  }
  if (argument_handles.size() != addressable_devices_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attempted to execute with %d argument lists when device count is %d "
        "(total replica count: %d, partition count: %d)",
        argument_handles.size(), addressable_devices_.size(), num_replicas(),
        num_partitions()));
  }
  if (addressable_devices_.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Attempted to execute with %d devices, but interpreter "
                        "only supports single device execution.",
                        addressable_devices_.size()));
  }

  std::optional<PjRtFuture<>> returned_future;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> replica_result,
      ExecuteSharded(argument_handles[0], addressable_devices_[0], options,
                     returned_future, returned_futures.has_value()));
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result;
  result.push_back(std::move(replica_result));
  if (returned_futures.has_value()) {
    CHECK(returned_future.has_value())
        << "returned_future must be set because ExecuteSharded was called with "
           "fill_future=true.";
    returned_futures = std::vector<PjRtFuture<>>({*std::move(returned_future)});
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
InterpreterLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  if (device_assignment_ == nullptr) {
    return absl::InvalidArgumentError(
        "ExecuteSharded expects a non-null device_assignment");
  }
  // Since there is only one device, the device should always be the same. Check
  // anyways just to be sure.
  if (!absl::c_any_of(
          addressable_devices_,
          [needle = device](PjRtDevice* const d) { return d == needle; })) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ExecuteShard attempted to execute on device id %d, which is not "
        "addressable by this client.",
        device->global_device_id().value()));
  }

  // Extract the literals from the arguments.
  const HloComputation& computation = *hlo_module_->entry_computation();
  TF_ASSIGN_OR_RETURN(const auto literals_and_storage,
                      ExtractInterpreterInputLiteralsFromBuffers(
                          argument_handles, computation,
                          compile_options_.parameter_is_tupled_arguments,
                          options.arguments_are_tupled));
  const absl::Span<const Literal* const> literals =
      std::get<0>(literals_and_storage);
  if (computation.num_parameters() != literals.size()) {
    return absl::InternalError(absl::StrFormat(
        "Mismatch between argument count (%d) and graph parameter count (%d).",
        literals.size(), computation.num_parameters()));
  }

  // Check that the args have the right shape.
  for (int64_t i = 0; i < computation.num_parameters(); ++i) {
    const Shape& expected_shape = computation.parameter_instruction(i)->shape();
    const Shape& actual_shape = literals[i]->shape();
    if (!ShapesMatch(expected_shape, actual_shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape mismatch on parameter %d. Expected %s but was %s.", i,
          ShapeUtil::HumanStringWithLayout(expected_shape),
          ShapeUtil::HumanStringWithLayout(actual_shape)));
    }
  }

  TF_ASSIGN_OR_RETURN(Literal result_literal, Evaluate(computation, literals));
  // Shrink the generated dynamic shape into static shape.
  result_literal = result_literal.ToStatic();
  if (fill_future) {
    returned_future = PjRtFuture<>(absl::OkStatus());
  }

  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                      device->default_memory_space());

  // Transform the result literal back into a one or more
  // InterpreterLiteralWrapperBuffer.
  std::vector<std::unique_ptr<PjRtBuffer>> result;
  // Untuple result if requested.
  if (options.untuple_result && result_literal.shape().IsTuple()) {
    const int tuple_count = result_literal.shape().tuple_shapes().size();
    result.reserve(tuple_count);
    // DecomposeTuple invalidates result_literal. move(...) to make it obvious.
    std::vector<Literal> tuple_elements =
        std::move(result_literal).DecomposeTuple();
    CHECK(tuple_count == tuple_elements.size())
        << "DecomposedTuple returned the wrong number of elements.";
    for (int i = 0; i < tuple_count; ++i) {
      result.push_back(std::make_unique<InterpreterLiteralWrapperBuffer>(
          client_, memory_space, std::move(tuple_elements[i])));
    }
  } else {
    result.push_back(std::make_unique<InterpreterLiteralWrapperBuffer>(
        client_, memory_space, std::move(result_literal)));
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
InterpreterLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  return absl::UnimplementedError("ExecutePortable is not implemented");
}

absl::StatusOr<Literal> InterpreterLoadedExecutable::Evaluate(
    const HloComputation& computation,
    absl::Span<const Literal* const> arg_literals) {
  absl::MutexLock lock(&hlo_evaluator_lock_);
  hlo_evaluator_->ResetVisitStates();
  return hlo_evaluator_->Evaluate(computation, arg_literals);
}

absl::StatusOr<DeviceAssignment> InterpreterClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_replicas != 1 || num_partitions != 1) {
    return absl::UnimplementedError(
        "Interpreter only supports num_replicas=1 and num_partitions=1.");
  }
  return MakeInterpreterDeviceAssignment();
}

absl::StatusOr<Layout> InterpreterClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  // This is all the GenericTransferManager::ChooseCompactLayoutForShape does.
  Shape shape = ShapeUtil::MakeValidatedShape(element_type, dims).value();
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape.layout();
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::CompileAndLoad(const XlaComputation& computation,
                                  CompileOptions options) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers,
                         /*layout_canonicalization_callback=*/nullptr, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::CompileAndLoad(mlir::ModuleOp module,
                                  CompileOptions options) {
  XlaComputation xla_computation;
  const ExecutableBuildOptions& exec_build_options =
      options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, exec_build_options.use_shardy_partitioner()));

  // If the compile options specify argument layout, then let's
  // fall back to using the options to determine layouts.
  if (options.argument_layouts) {
    return CompileAndLoad(xla_computation, options);
  }

  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                      GetOutputLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                      GetArgMemoryKinds(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                      GetOutputMemoryKinds(module));

  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [&arg_layout_modes, &out_layout_modes,
                          &arg_memory_spaces,
                          &out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces, ChooseCompactLayoutForShape);
  };

  // This call will update result_layout in options.executable_build_options.
  TF_ASSIGN_OR_RETURN(
      auto arg_layouts_and_pointers,
      LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes,
                       arg_memory_spaces, out_memory_spaces,
                       ChooseCompactLayoutForShape,
                       options.executable_build_options));
  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
InterpreterClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                         PjRtMemorySpace* memory_space,
                                         const Layout* device_layout) {
  if (device_layout == nullptr) {
    return std::make_unique<InterpreterLiteralWrapperBuffer>(
        memory_space->client(), memory_space, literal);
  }
  Literal device_literal = literal.Relayout(*device_layout);
  return std::make_unique<InterpreterLiteralWrapperBuffer>(
      memory_space->client(), memory_space, std::move(device_literal));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_shapes,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options) {
  CompileOptions input_options = options;
  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  if (layout_canonicalization_callback != nullptr) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());

  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_shapes, &execution_options,
                         execution_options.num_replicas(),
                         /*num_threads=*/std::nullopt,
                         /*aot_options=*/nullptr));
  // Unoptimized HloModule.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation.proto(), *hlo_module_config));

  if (build_options.num_partitions() != 1) {
    return absl::UnimplementedError(
        "For the time being, only num_partitions=1 is supported.");
  }

  if (!build_options.run_backend_only()) {
    TF_ASSIGN_OR_RETURN(hlo_module, RunHloPasses(std::move(hlo_module)));
  }

  return RunBackend(std::move(hlo_module), options);
}

absl::StatusOr<std::unique_ptr<HloModule>> InterpreterClient::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module) {
  HloPassPipeline pipeline("Interpreter");

  // The TopkDecomposer generates a compare op with type=TOTALORDER and must
  // run before the ComparisonExpander which rewrites such comparisons.
  pipeline.AddPass<TopkDecomposer>();
  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/true,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/true);
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());

  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module.get()).status());
  return hlo_module;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::RunBackend(std::unique_ptr<HloModule> hlo_module,
                              CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(
      DynamicDimensionInference dynamic_dimension_inference,
      DynamicDimensionInference::Run(
          hlo_module.get(),
          /*op_supports_dynamism_handler=*/[&](HloInstruction* hlo) {
            return OpDynamismSupport::kOptional;
          }));
  auto evaluator = hlo_evaluator_factory_();
  evaluator->set_use_fast_path(
      hlo_module->config().debug_options().xla_hlo_evaluator_use_fast_path());
  evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);

  std::shared_ptr<DeviceAssignment> device_assignment = nullptr;
  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  int num_replicas = 0, num_partitions = 0;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options.compile_portable_executable, &options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));
  if (device_assignment == nullptr) {
    return absl::InternalError("device_assignment is nullptr");
  }
  if (num_replicas != 1 || num_partitions != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("num_replicas and num_partitions must be 1. "
                        "num_replicas: %d, num_partitions: %d",
                        num_replicas, num_partitions));
  }
  PjRtLoadedExecutable::LogicalDeviceIds logical_device_ids;
  logical_device_ids.replica = 0;
  logical_device_ids.partition = 0;
  addressable_device_logical_ids.push_back(std::move(logical_device_ids));
  addressable_devices.push_back(&interpreter_device_);

  return std::make_unique<InterpreterLoadedExecutable>(
      this, std::move(hlo_module), std::move(evaluator),
      dynamic_dimension_inference, std::move(device_assignment), options,
      std::move(addressable_device_logical_ids),
      std::move(addressable_devices));
}

}  // namespace xla
