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
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
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

// Looks up the parameter layout for `param_operand_number` of `call_op`'s
// target executable in `atom_program_executables`.
absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
GetParameterLayoutFromAtomExecutable(
    const AtomExecutableMap& atom_program_executables,
    mlir::SymbolTableCollection& symbol_table,
    ifrt::CallLoadedExecutableOp& call_op, int param_operand_number) {
  // The parameter is used by a CallLoadedExecutableOp, return the layout
  // from the atom program executable.
  LoadedExecutableOp loaded_executable_op = call_op.getCalleeOp(symbol_table);
  std::string atom_program_name = loaded_executable_op.getSymName().str();
  auto executable_it = atom_program_executables.find(atom_program_name);
  if (executable_it != atom_program_executables.end()) {
    ABSL_ASSIGN_OR_RETURN(
        std::vector<std::shared_ptr<const xla::PjRtLayout>> executable_layouts,
        executable_it->second->GetParameterLayouts());
    return std::move(executable_layouts[param_operand_number]);
  }
  return absl::FailedPreconditionError(
      absl::StrFormat("Could not find SPMD executable %s", atom_program_name));
}

// Returns whether `CopyArraysOp` preserves device layout from `input_type` to
// `output_type`.
bool CanPropagateLayoutAcrossCopyArrays(IfrtArrayType input_type,
                                        IfrtArrayType output_type,
                                        const DeviceListRef& device_list) {
  llvm::ArrayRef<int> src_device_ids = input_type.getDevices();
  llvm::ArrayRef<int> dst_device_ids = output_type.getDevices();
  CHECK(!src_device_ids.empty());
  CHECK(!dst_device_ids.empty());
  CHECK_LT(src_device_ids.front(), device_list->devices().size());
  CHECK_LT(dst_device_ids.front(), device_list->devices().size());
  Device* src_device = device_list->devices()[src_device_ids.front()];
  Device* dst_device = device_list->devices()[dst_device_ids.front()];
  if (src_device->PlatformName() != dst_device->PlatformName()) {
    return false;
  }
  if (input_type.MemoryKind().value() == xla::UnpinnedHostMemorySpace::kKind ||
      output_type.MemoryKind().value() == xla::UnpinnedHostMemorySpace::kKind) {
    return false;
  }
  return true;
}

// Resolves the `PjRtLayout` of `value`, or returns `nullptr` when `value` is
// produced by a `CopyArraysOp` that cannot propagate its input layout.
absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> GetLayoutForValue(
    mlir::Value value, Client* absl_nonnull client,
    const AtomExecutableMap& atom_program_executables,
    absl::Span<const ArraySpec> in_specs,
    mlir::SymbolTableCollection& symbol_table,
    const DeviceListRef& device_list) {
  if (auto block_arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    if (in_specs[block_arg.getArgNumber()].layout != nullptr) {
      return in_specs[block_arg.getArgNumber()].layout;
    }
    return BuildDefaultLayout(in_specs[block_arg.getArgNumber()], client);
  }

  auto op_result = llvm::cast<mlir::OpResult>(value);
  if (auto owner_call_op =
          llvm::dyn_cast<CallLoadedExecutableOp>(op_result.getOwner())) {
    LoadedExecutableOp loaded_executable_op =
        owner_call_op.getCalleeOp(symbol_table);
    std::string atom_program_name = loaded_executable_op.getSymName().str();
    auto executable_it = atom_program_executables.find(atom_program_name);
    if (executable_it != atom_program_executables.end()) {
      ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<const xla::PjRtLayout>>
                           executable_layouts,
                       executable_it->second->GetOutputLayouts());
      return executable_layouts[op_result.getResultNumber()];
    }
    return absl::FailedPreconditionError(absl::StrFormat(
        "Could not find SPMD executable %s", atom_program_name));
  }

  if (auto copy_arrays =
          llvm::dyn_cast<ifrt::CopyArraysOp>(op_result.getOwner())) {
    mlir::Value input = copy_arrays.getInputs()[op_result.getResultNumber()];
    IfrtArrayType input_type = GetArrayType(input);
    IfrtArrayType output_type = GetArrayType(op_result);
    if (!CanPropagateLayoutAcrossCopyArrays(input_type, output_type,
                                            device_list)) {
      return nullptr;
    }
    return GetLayoutForValue(input, client, atom_program_executables, in_specs,
                             symbol_table, device_list);
  }

  return absl::FailedPreconditionError(absl::StrFormat(
      "Layouts are supported only for programs that have outputs produced "
      "by CallLoadedExecutableOp or CopyArraysOp ops. Produced by %s",
      OperationToString(op_result.getOwner(), mlir::OpPrintingFlags())));
}

}  // namespace

tsl::Executor& GetIfrtIrCompilationExecutor() { return FutureExecutor::Get(); }

absl::StatusOr<LoweredIfrtIrProgramState> LowerIfrtIrProgram(
    absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
    absl_nonnull std::unique_ptr<IfrtIRCompileOptions> ifrt_ir_compile_options,
    Client* absl_nonnull client,
    absl_nonnull std::shared_ptr<AtomProgramCompiler> atom_program_compiler) {
  std::shared_ptr<IfrtIRCompileOptions> compile_options =
      std::move(ifrt_ir_compile_options);

  ABSL_ASSIGN_OR_RETURN(DeviceListRef device_list,
                   LookUpDevices(client, compile_options->device_assignments));

  mlir::ModuleOp mlir_module = ifrt_ir_program->mlir_module;
  std::string program_name = mlir_module.getName().value_or("unknown").str();

  // Add the bound executables to the atom program executable map so that they
  // can be used by the interpreter.
  auto atom_executable_future_map = std::make_shared<AtomExecutableFutureMap>();
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [key, executable] : compile_options->loaded_exec_binding) {
    atom_executable_future_map->insert({key, executable});
  }
  // Extract bindings.
  auto bound_executable_map = std::make_shared<AtomExecutableMap>(
      compile_options->loaded_exec_binding.begin(),
      compile_options->loaded_exec_binding.end());

  // Run lowering passes.
  {
    mlir::MLIRContext* context = mlir_module.getContext();
    mlir::PassManager pm(context);
    InitPassManager(pm, "ifrt.compile", compile_options->mlir_dump_to,
                    compile_options->mlir_dump_pass_re,
                    compile_options->mlir_dump_func_re,
                    compile_options->mlir_enable_timing);

    createIfrtToOutlinedAtomProgramsPipeline(pm);

    createIfrtPopulateAtomProgramMetadataPipeline(pm);

    OutlinedAtomProgramsToCompiledPipelineOptions compile_pipeline_options;
    std::vector<std::string>& platform_names =
        compile_pipeline_options.platform_names;
    platform_names.reserve(device_list->devices().size());
    for (Device* const device : device_list->devices()) {
      platform_names.push_back(std::string(device->PlatformName()));
    }
    ABSL_RETURN_IF_ERROR(createOutlinedAtomProgramsToCompiledPipeline(
        pm, std::move(atom_program_compiler), compile_pipeline_options,
        compile_options, atom_executable_future_map,
        std::move(bound_executable_map)));

    {
      TraceMe traceme("ProgramCompiler::CompileForInterpreter::RunPasses");
      StatusScopedDiagnosticHandler diag_handler(context);
      if (mlir::failed(pm.run(mlir_module))) {
        return diag_handler.ConsumeStatus();
      }
    }
  }

  // If the IfrtIRProgram owns the MLIRContext, we are the exclusive users of
  // this context (e.g., deserialized program), and we can clone the IFRT IR
  // module to a new context in order to drop the temporary interned attributes
  // created while deserializing the StableHLO programs. Notice that cloning the
  // module is relatively fast and allows us to reduce host memory usage.
  if (ifrt_ir_program->OwnsMlirContext()) {
    auto context = std::make_unique<mlir::MLIRContext>(
        mlir::MLIRContext::Threading::DISABLED);
    context->enableThreadSafeLocking();
    ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> cloned_module,
                     CloneModuleIntoContext(mlir_module, *context));
    ifrt_ir_program = std::make_unique<xla::ifrt::IfrtIRProgram>(
        std::move(context), std::move(cloned_module));
  }

  // Extract input and output specs from the modified `mlir_module`, which has
  // all array shardings specified.
  mlir::func::FuncOp main_func = GetMainFunction(ifrt_ir_program->mlir_module);
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
  std::vector<int> donatable_input_indices;
  for (const auto [idx, arg] : llvm::enumerate(main_func.getArguments())) {
    if (main_func.getArgAttr(idx, kIfrtDonatedArgAttrName) != nullptr) {
      donatable_input_indices.push_back(idx);
    }
  }

  return LoweredIfrtIrProgramState{
      /*program_name=*/std::move(program_name),
      /*atom_executable_future_map=*/std::move(atom_executable_future_map),
      /*in_specs=*/std::move(in_specs),
      /*out_specs=*/std::move(out_specs),
      /*donatable_input_indices=*/std::move(donatable_input_indices),
      /*ifrt_ir_program=*/std::move(ifrt_ir_program),
      /*device_list=*/std::move(device_list),
      /*compile_options=*/std::move(compile_options),
  };
}

absl::Status PopulateIfrtIrProgramLayouts(
    mlir::ModuleOp mlir_module, Client* absl_nonnull client,
    const AtomExecutableMap& atom_program_executables,
    absl::Span<ArraySpec> in_specs, absl::Span<ArraySpec> out_specs,
    const DeviceListRef& device_list) {
  TraceMe traceme("PopulateLayouts");

  mlir::func::FuncOp main_func = GetMainFunction(mlir_module);
  mlir::SymbolTableCollection symbol_table;

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
        ABSL_ASSIGN_OR_RETURN(std::shared_ptr<const xla::PjRtLayout> consumer_layout,
                         GetParameterLayoutFromAtomExecutable(
                             atom_program_executables, symbol_table, call_op,
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
    in_specs[arg.getArgNumber()].layout = std::move(parameter_layout);
  }

  for (mlir::OpOperand& return_operand :
       main_func.front().getTerminator()->getOpOperands()) {
    ArraySpec& out_spec = out_specs[return_operand.getOperandNumber()];
    ABSL_ASSIGN_OR_RETURN(out_spec.layout,
                     GetLayoutForValue(return_operand.get(), client,
                                       atom_program_executables, in_specs,
                                       symbol_table, device_list));
    if (out_spec.layout == nullptr) {
      ABSL_ASSIGN_OR_RETURN(out_spec.layout, BuildDefaultLayout(out_spec, client));
    }
  }

  return absl::OkStatus();
}

}  // namespace ifrt
}  // namespace xla
