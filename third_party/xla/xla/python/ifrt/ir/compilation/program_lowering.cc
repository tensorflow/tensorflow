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

#include "xla/python/ifrt/ir/compilation/program_lowering.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation/passes.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/debug.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/passes.h"
#include "xla/python/ifrt/ir/utils.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {

using ::tsl::profiler::TraceMe;

namespace {

// An executor that uses `tsl::Env::SchedClosure` to schedule tasks. Used to run
// future callbacks that must not run on fibers.
class FutureExecutor : public tsl::Executor {
 public:
  static FutureExecutor& Get() {
    static absl::NoDestructor<FutureExecutor> instance;
    return *instance;
  }

  void Execute(Task task) override {
    tsl::Env::Default()->SchedClosure(
        [task = std::move(task)]() mutable { std::move(task)(); });
  }
};

enum class AtomLayoutDirection {
  kParameter,
  kOutput,
};

// Returns the device-default `PjRtLayout` for `arg_spec`'s shard shape.
absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> BuildDefaultLayout(
    const ArraySpec& arg_spec, Client* absl_nonnull client) {
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::Shape shard_shape,
                   arg_spec.sharding->GetShardShape(arg_spec.shape));
  return client->GetDefaultPjRtLayout(
      arg_spec.dtype, shard_shape.dims(),
      arg_spec.sharding->devices()->devices().front(),
      arg_spec.sharding->memory_kind());
}

// Looks up the parameter or output `PjRtLayout` at `index` of `call_op`'s
// target executable in `atom_program_executables` or `compiled_atom_programs`.
absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
GetLayoutFromAtomExecutable(
    const AtomExecutableMap& atom_program_executables,
    const CompiledAtomProgramMap* absl_nullable compiled_atom_programs,
    mlir::SymbolTableCollection& symbol_table,
    ifrt::CallLoadedExecutableOp call_op, AtomLayoutDirection direction,
    int index) {
  LoadedExecutableOp loaded_executable_op = call_op.getCalleeOp(symbol_table);
  std::string atom_program_name = loaded_executable_op.getSymName().str();
  const LoadedExecutable* loaded_atom_executable = nullptr;
  const Executable* compiled_atom_executable = nullptr;
  if (auto executable_it = atom_program_executables.find(atom_program_name);
      executable_it != atom_program_executables.end()) {
    loaded_atom_executable = executable_it->second.get();
  } else if (compiled_atom_programs != nullptr) {
    auto compiled_it = compiled_atom_programs->find(atom_program_name);
    if (compiled_it != compiled_atom_programs->end()) {
      compiled_atom_executable = compiled_it->second->executable.get();
    }
  }
  if (loaded_atom_executable == nullptr &&
      compiled_atom_executable == nullptr) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Could not find SPMD executable %s", atom_program_name));
  }

  std::vector<std::shared_ptr<const xla::PjRtLayout>> executable_layouts;
  if (direction == AtomLayoutDirection::kParameter) {
    ABSL_ASSIGN_OR_RETURN(executable_layouts,
                     loaded_atom_executable != nullptr
                         ? loaded_atom_executable->GetParameterLayouts()
                         : compiled_atom_executable->GetParameterLayouts());
  } else {
    ABSL_ASSIGN_OR_RETURN(executable_layouts,
                     loaded_atom_executable != nullptr
                         ? loaded_atom_executable->GetOutputLayouts()
                         : compiled_atom_executable->GetOutputLayouts());
  }
  if (index < 0 || index >= executable_layouts.size()) {
    return absl::InternalError(
        direction == AtomLayoutDirection::kParameter
            ? absl::StrFormat("Parameter operand %d out of range for "
                              "executable %s with %d parameter layouts",
                              index, atom_program_name,
                              executable_layouts.size())
            : absl::StrFormat("Output %d out of range for executable %s with "
                              "%d output layouts",
                              index, atom_program_name,
                              executable_layouts.size()));
  }
  if (executable_layouts[index] == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "%s %d layout for executable %s is null",
        direction == AtomLayoutDirection::kParameter ? "Parameter" : "Output",
        index, atom_program_name));
  }
  return std::move(executable_layouts[index]);
}

// Returns whether `CopyArraysOp` preserves device layout from `input_type` to
// `output_type`.
absl::StatusOr<bool> CanPropagateLayoutAcrossCopyArrays(
    IfrtArrayType input_type, IfrtArrayType output_type,
    const DeviceListRef& device_list) {
  llvm::ArrayRef<int> src_device_ids = input_type.getDevices();
  llvm::ArrayRef<int> dst_device_ids = output_type.getDevices();
  if (src_device_ids.empty() || dst_device_ids.empty()) {
    return absl::InvalidArgumentError(
        "CopyArraysOp array types must have non-empty device lists");
  }
  if (src_device_ids.front() < 0 ||
      src_device_ids.front() >= device_list->devices().size() ||
      dst_device_ids.front() < 0 ||
      dst_device_ids.front() >= device_list->devices().size()) {
    return absl::InvalidArgumentError(
        "CopyArraysOp logical device id is out of range");
  }
  Device* src_device = device_list->devices()[src_device_ids.front()];
  Device* dst_device = device_list->devices()[dst_device_ids.front()];
  if (src_device->PlatformName() != dst_device->PlatformName()) {
    return false;
  }
  if (absl::string_view(input_type.getMemoryKindAttr().getValue()) ==
          xla::UnpinnedHostMemorySpace::kKind ||
      absl::string_view(output_type.getMemoryKindAttr().getValue()) ==
          xla::UnpinnedHostMemorySpace::kKind) {
    return false;
  }
  return true;
}

// Resolves the `PjRtLayout` of `value`, or returns `nullptr` when `value` is
// produced by a `CopyArraysOp` that cannot propagate its input layout.
absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> GetLayoutForValue(
    mlir::Value value, Client* absl_nonnull client,
    const AtomExecutableMap& atom_program_executables,
    const CompiledAtomProgramMap* absl_nullable compiled_atom_programs,
    absl::Span<const ArraySpec> in_specs,
    absl::Span<const std::shared_ptr<const xla::PjRtLayout>> in_layouts,
    mlir::SymbolTableCollection& symbol_table,
    const DeviceListRef& device_list) {
  if (auto block_arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    const unsigned arg_number = block_arg.getArgNumber();
    if (in_layouts[arg_number] != nullptr) {
      return in_layouts[arg_number];
    }
    return BuildDefaultLayout(in_specs[arg_number], client);
  }

  auto op_result = llvm::cast<mlir::OpResult>(value);
  if (auto owner_call_op =
          llvm::dyn_cast<CallLoadedExecutableOp>(op_result.getOwner())) {
    return GetLayoutFromAtomExecutable(
        atom_program_executables, compiled_atom_programs, symbol_table,
        owner_call_op, AtomLayoutDirection::kOutput,
        op_result.getResultNumber());
  }

  if (auto copy_arrays =
          llvm::dyn_cast<ifrt::CopyArraysOp>(op_result.getOwner())) {
    mlir::Value input = copy_arrays.getInputs()[op_result.getResultNumber()];
    IfrtArrayType input_type = GetArrayType(input);
    IfrtArrayType output_type = GetArrayType(op_result);
    ABSL_ASSIGN_OR_RETURN(bool can_propagate,
                     CanPropagateLayoutAcrossCopyArrays(input_type, output_type,
                                                        device_list));
    if (!can_propagate) {
      return nullptr;
    }
    return GetLayoutForValue(input, client, atom_program_executables,
                             compiled_atom_programs, in_specs, in_layouts,
                             symbol_table, device_list);
  }

  return absl::FailedPreconditionError(absl::StrFormat(
      "Layouts are supported only for programs that have outputs produced "
      "by CallLoadedExecutableOp or CopyArraysOp ops. Produced by %s",
      OperationToString(op_result.getOwner(), mlir::OpPrintingFlags())));
}

// Verifies that `mlir_module` defines a public single-block IFRT `@main`
// function and returns it.
absl::StatusOr<mlir::func::FuncOp> ValidateMainFunctionHeader(
    mlir::ModuleOp mlir_module, absl::string_view program_name) {
  mlir::func::FuncOp main_func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      mlir_module.lookupSymbol("main"));
  if (main_func == nullptr || !main_func.isPublic() ||
      !IsIfrtFunction(main_func)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`main` function of IFRT IR program `%s` must be a "
                        "public IFRT function",
                        program_name));
  }
  if (!llvm::hasSingleElement(main_func.getBody())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`main` function of IFRT IR program `%s` must have a single block",
        program_name));
  }
  return main_func;
}

// Validates that all operations in `main_func` are supported by
// `ProgramInterpreter` (or by `createOutlinedAtomProgramsToCompiledPipeline`
// when `allow_precompile_ops` is true) and have valid `ArraySpec`s, shardings,
// logical device IDs, and `RemapPlan`s against `device_list`.
absl::Status ValidateMainFunctionOps(
    mlir::func::FuncOp main_func, absl::string_view program_name,
    Client* absl_nonnull client, const DeviceListRef& device_list,
    const AtomExecutableMap& bound_executables,
    const AtomExecutableFutureMap* absl_nullable atom_executable_future_map,
    const CompiledAtomProgramFutureMap* absl_nullable
        compiled_atom_program_future_map,
    bool allow_precompile_ops) {
  auto validate_array_values = [&](mlir::ValueRange values) -> absl::Status {
    for (const mlir::Value value : values) {
      ABSL_RETURN_IF_ERROR(
          ArraySpecFromMlirType(value.getType(), client, device_list).status());
    }
    return absl::OkStatus();
  };

  mlir::SymbolTableCollection symbol_table;
  for (mlir::Operation& op : main_func.getOps()) {
    if (allow_precompile_ops) {
      if (auto call_op = mlir::dyn_cast<CallOp>(op)) {
        for (const int logical_id : call_op.getDevices()) {
          if (logical_id < 0 || logical_id >= device_list->devices().size()) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Logical device id %d is out of range for device list of "
                "size %d",
                logical_id, device_list->devices().size()));
          }
        }
        ABSL_RETURN_IF_ERROR(validate_array_values(call_op.getInputs()));
        ABSL_RETURN_IF_ERROR(validate_array_values(call_op.getOutputs()));
        continue;
      }
      if (auto reshard_op = mlir::dyn_cast<ReshardOp>(op)) {
        ABSL_RETURN_IF_ERROR(validate_array_values(reshard_op.getInputs()));
        ABSL_RETURN_IF_ERROR(validate_array_values(reshard_op.getOutputs()));
        continue;
      }
    }
    if (auto call_op = mlir::dyn_cast<CallLoadedExecutableOp>(op)) {
      LoadedExecutableOp loaded_exec_op = call_op.getCalleeOp(symbol_table);
      absl::string_view atom_name = loaded_exec_op.getSymName();
      const bool has_bound_or_compiled =
          bound_executables.contains(atom_name) ||
          (atom_executable_future_map != nullptr &&
           atom_executable_future_map->contains(atom_name)) ||
          (compiled_atom_program_future_map != nullptr &&
           compiled_atom_program_future_map->contains(atom_name));
      if (!has_bound_or_compiled) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Could not find bound or compiled executable for `%s`", atom_name));
      }
      ABSL_RETURN_IF_ERROR(validate_array_values(call_op.getInputs()));
      ABSL_RETURN_IF_ERROR(validate_array_values(call_op.getOutputs()));
      continue;
    }
    if (auto remap_op = mlir::dyn_cast<RemapArraysOp>(op)) {
      ABSL_RETURN_IF_ERROR(BuildRemapPlan(client, device_list, remap_op).status());
      continue;
    }
    if (auto bitcast_op = mlir::dyn_cast<BitcastArraysOp>(op)) {
      ABSL_RETURN_IF_ERROR(validate_array_values(bitcast_op.getInputs()));
      ABSL_RETURN_IF_ERROR(validate_array_values(bitcast_op.getOutputs()));
      continue;
    }
    if (auto copy_arrays_op = mlir::dyn_cast<CopyArraysOp>(op)) {
      ABSL_RETURN_IF_ERROR(validate_array_values(copy_arrays_op.getInputs()));
      ABSL_RETURN_IF_ERROR(validate_array_values(copy_arrays_op.getOutputs()));
      continue;
    }
    if (!mlir::isa<mlir::func::ReturnOp>(op)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Found unsupported op `%s` in `main` function of IFRT IR program "
          "`%s`",
          op.getName().getStringRef().str(), program_name));
    }
  }
  return absl::OkStatus();
}

// Waits for any in-flight atom compilation futures in
// `atom_executable_future_map` and `compiled_atom_program_future_map` to settle
// before fulfilling the returned future with `status`.
tsl::Future<LoweredIfrtIrProgramState> PropagateErrorAfterDrainingAtomFutures(
    absl::Status status,
    const AtomExecutableFutureMap* absl_nullable atom_executable_future_map,
    const CompiledAtomProgramFutureMap* absl_nullable
        compiled_atom_program_future_map) {
  CHECK(!status.ok());
  std::vector<tsl::Future<>> pending_futures;
  if (atom_executable_future_map != nullptr) {
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [unused_key, future] : *atom_executable_future_map) {
      if (!future.IsReady()) {
        pending_futures.push_back(future.GetReadyFuture());
      }
    }
  }
  if (compiled_atom_program_future_map != nullptr) {
    for (const auto& [unused_key, future] :
         // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
         *compiled_atom_program_future_map) {
      if (!future.IsReady()) {
        pending_futures.push_back(future.GetReadyFuture());
      }
    }
  }
  if (pending_futures.empty()) {
    return tsl::Future<LoweredIfrtIrProgramState>(std::move(status));
  }
  auto [promise, drained_future] =
      tsl::MakePromise<LoweredIfrtIrProgramState>();
  // Propagate the lowering `status` once all in-flight atom compilations have
  // settled, regardless of whether an atom compilation also failed.
  tsl::JoinFutures(pending_futures)
      .OnReady([promise = std::move(promise), status = std::move(status)](
                   absl::Status unused_join_status) mutable {
        promise.Set(std::move(status));
      });
  return std::move(drained_future);
}

// Runs `createOutlinedAtomProgramsToCompiledPipeline` and constructs
// `LoweredIfrtIrProgramState` once pre-compile `@main` validation has passed.
absl::StatusOr<LoweredIfrtIrProgramState>
FinishLoweringAndDispatchAtomCompilations(
    absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
    std::string program_name, Client* absl_nonnull client,
    absl_nonnull std::shared_ptr<AtomProgramCompiler> atom_program_compiler,
    AtomProgramLoadingMode loading_mode, DeviceListRef device_list,
    absl_nonnull std::shared_ptr<IfrtIRCompileOptions> compile_options,
    const absl_nullable std::shared_ptr<AtomExecutableFutureMap>&
        atom_executable_future_map,
    const absl_nullable std::shared_ptr<CompiledAtomProgramFutureMap>&
        compiled_atom_program_future_map,
    const
    absl_nonnull std::shared_ptr<AtomExecutableMap>& bound_executable_map) {
  mlir::ModuleOp mlir_module = ifrt_ir_program->mlir_module;
  {
    mlir::MLIRContext* context = mlir_module.getContext();
    mlir::PassManager compile_pm(context);
    InitPassManager(compile_pm, "ifrt.compile", compile_options->mlir_dump_to,
                    compile_options->mlir_dump_pass_re,
                    compile_options->mlir_dump_func_re,
                    compile_options->mlir_enable_timing);

    OutlinedAtomProgramsToCompiledPipelineOptions compile_pipeline_options;
    std::vector<std::string>& platform_names =
        compile_pipeline_options.platform_names;
    platform_names.reserve(device_list->devices().size());
    for (Device* const device : device_list->devices()) {
      platform_names.push_back(std::string(device->PlatformName()));
    }
    ABSL_RETURN_IF_ERROR(createOutlinedAtomProgramsToCompiledPipeline(
        compile_pm, std::move(atom_program_compiler), compile_pipeline_options,
        compile_options, atom_executable_future_map,
        compiled_atom_program_future_map, bound_executable_map));

    {
      TraceMe traceme("ProgramCompiler::CompileForInterpreter::RunPasses");
      StatusScopedDiagnosticHandler diag_handler(context);
      if (mlir::failed(compile_pm.run(mlir_module))) {
        return diag_handler.ConsumeStatus();
      }
    }
  }

  // Clone the lowered IFRT IR module into a new MLIR context when atom program
  // loading is deferred (so `LoweredIfrtIrProgramState::ifrt_ir_program` owns
  // the lowered module independently of the caller's input `MLIRContext`) or
  // when `ifrt_ir_program` exclusively owns its `MLIRContext` (to drop interned
  // attributes from deserialized StableHLO programs and reduce host memory).
  if (loading_mode == AtomProgramLoadingMode::kDeferred ||
      ifrt_ir_program->OwnsMlirContext()) {
    auto context = std::make_unique<mlir::MLIRContext>(
        mlir::MLIRContext::Threading::DISABLED);
    context->enableThreadSafeLocking();
    ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> cloned_module,
                     CloneModuleIntoContext(mlir_module, *context));
    ifrt_ir_program = std::make_unique<xla::ifrt::IfrtIRProgram>(
        std::move(context), std::move(cloned_module));
  }

  // Extract input and output specs from the modified `mlir_module`, which has
  // all array shardings specified, and validate lowered `@main` operations.
  ABSL_ASSIGN_OR_RETURN(
      mlir::func::FuncOp main_func,
      ValidateMainFunctionHeader(ifrt_ir_program->mlir_module, program_name));
  std::vector<ArraySpec> in_specs;
  in_specs.reserve(main_func.getNumArguments());
  for (const mlir::Type arg_type : main_func.getArgumentTypes()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec spec,
                     ArraySpecFromMlirType(arg_type, client, device_list));
    in_specs.push_back(std::move(spec));
  }
  std::vector<ArraySpec> out_specs;
  out_specs.reserve(main_func.getNumResults());
  for (const mlir::Type result_type : main_func.getResultTypes()) {
    ABSL_ASSIGN_OR_RETURN(ArraySpec spec,
                     ArraySpecFromMlirType(result_type, client, device_list));
    out_specs.push_back(std::move(spec));
  }
  ABSL_RETURN_IF_ERROR(ValidateMainFunctionOps(
      main_func, program_name, client, device_list, *bound_executable_map,
      atom_executable_future_map.get(), compiled_atom_program_future_map.get(),
      /*allow_precompile_ops=*/false));

  std::vector<int> donatable_input_indices;
  for (const auto [idx, arg] : llvm::enumerate(main_func.getArguments())) {
    if (main_func.getArgAttr(idx, kIfrtDonatedArgAttrName) != nullptr) {
      donatable_input_indices.push_back(idx);
    }
  }

  return LoweredIfrtIrProgramState{
      /*program_name=*/std::move(program_name),
      /*atom_executable_future_map=*/atom_executable_future_map,
      /*compiled_atom_program_future_map=*/compiled_atom_program_future_map,
      /*in_specs=*/std::move(in_specs),
      /*out_specs=*/std::move(out_specs),
      /*donatable_input_indices=*/std::move(donatable_input_indices),
      /*ifrt_ir_program=*/std::move(ifrt_ir_program),
      /*device_list=*/std::move(device_list),
      /*compile_options=*/std::move(compile_options),
  };
}

}  // namespace

tsl::Executor& GetIfrtIrCompilationExecutor() { return FutureExecutor::Get(); }

tsl::Future<LoweredIfrtIrProgramState> LowerIfrtIrProgram(
    absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
    absl_nonnull std::unique_ptr<IfrtIRCompileOptions> ifrt_ir_compile_options,
    Client* absl_nonnull client,
    absl_nonnull std::shared_ptr<AtomProgramCompiler> atom_program_compiler,
    AtomProgramLoadingMode loading_mode) {
  if (ifrt_ir_program == nullptr || ifrt_ir_program->mlir_module == nullptr) {
    return absl::InvalidArgumentError(
        "IFRT IR program must have a non-null MLIR module");
  }
  mlir::ModuleOp mlir_module = ifrt_ir_program->mlir_module;
  std::string program_name = mlir_module.getName().value_or("unknown").str();
  ABSL_RETURN_IF_ERROR(
      ValidateMainFunctionHeader(mlir_module, program_name).status());

  std::shared_ptr<IfrtIRCompileOptions> compile_options =
      std::move(ifrt_ir_compile_options);
  ABSL_ASSIGN_OR_RETURN(DeviceListRef device_list,
                   LookUpDevices(client, compile_options->device_assignments));

  std::shared_ptr<AtomExecutableFutureMap> atom_executable_future_map;
  std::shared_ptr<CompiledAtomProgramFutureMap>
      compiled_atom_program_future_map;
  if (loading_mode == AtomProgramLoadingMode::kDeferred) {
    compiled_atom_program_future_map =
        std::make_shared<CompiledAtomProgramFutureMap>();
  } else {
    atom_executable_future_map = std::make_shared<AtomExecutableFutureMap>();
  }
  // Validate bound executables upfront and, in eager mode, seed
  // `atom_executable_future_map` so they can be used by the interpreter.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [key, executable] : compile_options->loaded_exec_binding) {
    if (executable == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Bound executable `%s` must not be null", key));
    }
    if (atom_executable_future_map != nullptr) {
      atom_executable_future_map->insert({key, executable});
    }
  }
  // Extract bindings.
  auto bound_executable_map = std::make_shared<AtomExecutableMap>(
      compile_options->loaded_exec_binding.begin(),
      compile_options->loaded_exec_binding.end());

  // Run outlining and metadata passes, then validate `@main` operations and
  // `ArraySpec`s before dispatching any atom program compilations.
  {
    mlir::MLIRContext* context = mlir_module.getContext();
    mlir::PassManager outline_pm(context);
    InitPassManager(outline_pm, "ifrt.outline", compile_options->mlir_dump_to,
                    compile_options->mlir_dump_pass_re,
                    compile_options->mlir_dump_func_re,
                    compile_options->mlir_enable_timing);
    createIfrtToOutlinedAtomProgramsPipeline(outline_pm);
    createIfrtPopulateAtomProgramMetadataPipeline(outline_pm);
    {
      TraceMe traceme(
          "ProgramCompiler::CompileForInterpreter::RunOutlinePasses");
      StatusScopedDiagnosticHandler diag_handler(context);
      if (mlir::failed(outline_pm.run(mlir_module))) {
        return diag_handler.ConsumeStatus();
      }
    }
  }
  {
    ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp precompile_main_func,
                     ValidateMainFunctionHeader(mlir_module, program_name));
    for (const mlir::Type arg_type : precompile_main_func.getArgumentTypes()) {
      ABSL_RETURN_IF_ERROR(
          ArraySpecFromMlirType(arg_type, client, device_list).status());
    }
    for (const mlir::Type result_type : precompile_main_func.getResultTypes()) {
      ABSL_RETURN_IF_ERROR(
          ArraySpecFromMlirType(result_type, client, device_list).status());
    }
    ABSL_RETURN_IF_ERROR(ValidateMainFunctionOps(
        precompile_main_func, program_name, client, device_list,
        *bound_executable_map, /*atom_executable_future_map=*/nullptr,
        /*compiled_atom_program_future_map=*/nullptr,
        /*allow_precompile_ops=*/true));
  }

  absl::StatusOr<LoweredIfrtIrProgramState> lowered_state =
      FinishLoweringAndDispatchAtomCompilations(
          std::move(ifrt_ir_program), std::move(program_name), client,
          std::move(atom_program_compiler), loading_mode,
          std::move(device_list), std::move(compile_options),
          atom_executable_future_map, compiled_atom_program_future_map,
          bound_executable_map);
  if (!lowered_state.ok()) {
    return PropagateErrorAfterDrainingAtomFutures(
        std::move(lowered_state).status(), atom_executable_future_map.get(),
        compiled_atom_program_future_map.get());
  }
  return *std::move(lowered_state);
}

absl::Status PopulateIfrtIrProgramLayouts(
    mlir::ModuleOp mlir_module, Client* absl_nonnull client,
    const AtomExecutableMap& atom_program_executables,
    const CompiledAtomProgramMap* absl_nullable compiled_atom_programs,
    absl::Span<ArraySpec> in_specs, absl::Span<ArraySpec> out_specs,
    const DeviceListRef& device_list) {
  TraceMe traceme("PopulateLayouts");

  mlir::func::FuncOp main_func = GetMainFunction(mlir_module);
  mlir::SymbolTableCollection symbol_table;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> in_layouts(
      in_specs.size());
  std::vector<std::shared_ptr<const xla::PjRtLayout>> out_layouts(
      out_specs.size());

  for (mlir::BlockArgument& arg : main_func.getArguments()) {
    std::shared_ptr<const xla::PjRtLayout> parameter_layout;
    if (arg.use_empty()) {
      // The argument is not used. Return device default layout.
      ABSL_ASSIGN_OR_RETURN(
          parameter_layout,
          BuildDefaultLayout(in_specs[arg.getArgNumber()], client));
    } else {
      // Find the layout from the first LoadedExecutableOp consumer or just
      // return any of the users otherwise. Possible users: CopyArraysOp,
      // ReturnOp, and other LoadedExecutableOp.
      for (mlir::OpOperand& use : arg.getUses()) {
        if (llvm::isa<mlir::func::ReturnOp>(use.getOwner()) ||
            llvm::isa<ifrt::CopyArraysOp>(use.getOwner())) {
          continue;
        }

        if (!llvm::isa<ifrt::CallLoadedExecutableOp>(use.getOwner())) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "Layouts are supported only for programs that have parameters "
              "used only by CallLoadedExecutableOp ops. Parameter %d is used "
              "by %s",
              arg.getArgNumber(),
              OperationToString(use.getOwner(), mlir::OpPrintingFlags())));
        }
        auto call_op = llvm::cast<ifrt::CallLoadedExecutableOp>(use.getOwner());
        ABSL_ASSIGN_OR_RETURN(
            std::shared_ptr<const xla::PjRtLayout> consumer_layout,
            GetLayoutFromAtomExecutable(
                atom_program_executables, compiled_atom_programs, symbol_table,
                call_op, AtomLayoutDirection::kParameter,
                use.getOperandNumber()));
        if (parameter_layout == nullptr) {
          parameter_layout = std::move(consumer_layout);
          continue;
        }
        if (*parameter_layout != *consumer_layout) {
          return absl::InternalError(absl::StrFormat(
              "Parameter %d is used by atom programs with incompatible "
              "layouts: %s vs. %s. This happens because support for layout "
              "progation within MPMD programs is limited. Contact "
              "ml-pathways-team@ for help",
              arg.getArgNumber(), parameter_layout->ToString(),
              consumer_layout->ToString()));
        }
      }
      if (parameter_layout == nullptr) {
        // The argument was skipped above, meaning only used by ReturnOp or
        // CopyArraysOp.
        ABSL_ASSIGN_OR_RETURN(
            parameter_layout,
            BuildDefaultLayout(in_specs[arg.getArgNumber()], client));
      }
    }
    in_layouts[arg.getArgNumber()] = std::move(parameter_layout);
  }

  for (mlir::OpOperand& return_operand :
       main_func.front().getTerminator()->getOpOperands()) {
    const unsigned operand_number = return_operand.getOperandNumber();
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<const xla::PjRtLayout> out_layout,
        GetLayoutForValue(return_operand.get(), client,
                          atom_program_executables, compiled_atom_programs,
                          in_specs, in_layouts, symbol_table, device_list));
    if (out_layout == nullptr) {
      ABSL_ASSIGN_OR_RETURN(out_layout,
                       BuildDefaultLayout(out_specs[operand_number], client));
    }
    out_layouts[operand_number] = std::move(out_layout);
  }

  for (size_t i = 0; i < in_specs.size(); ++i) {
    in_specs[i].layout = std::move(in_layouts[i]);
  }
  for (size_t i = 0; i < out_specs.size(); ++i) {
    out_specs[i].layout = std::move(out_layouts[i]);
  }
  return absl::OkStatus();
}

}  // namespace ifrt
}  // namespace xla
