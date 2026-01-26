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

#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/program_interpreter.h"
#include "xla/python/ifrt/ir/transforms/debug.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {

using ::tsl::profiler::TraceMe;

namespace {

absl::StatusOr<xla::ifrt::ArraySpec> ToArraySpec(
    xla::ifrt::IfrtArrayType array, xla::ifrt::Client* client,
    const std::vector<xla::ifrt::Device*>& devices) {
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::DType dtype,
      xla::ifrt::ToIfrtDType(array.getShape().getElementType()));
  absl::InlinedVector<xla::ifrt::Device*, 1> client_devices;
  client_devices.reserve(array.getDevices().size());
  for (int logical_id : array.getDevices()) {
    TF_RET_CHECK(devices[logical_id] != nullptr);
    client_devices.push_back(devices[logical_id]);
  }
  auto sharding_attr =
      mlir::dyn_cast<xla::ifrt::IfrtShardingParamAttr>(array.getShardingAttr());
  TF_RET_CHECK(sharding_attr != nullptr);
  TF_ASSIGN_OR_RETURN(DeviceListRef device_list,
                      client->MakeDeviceList(client_devices));
  TF_ASSIGN_OR_RETURN(xla::ifrt::ShardingRef sharding,
                      xla::ifrt::ShardingParamSharding::Create(
                          sharding_attr.getSharding(), std::move(device_list),
                          array.MemoryKind()));
  return xla::ifrt::ArraySpec{
      /*dtype=*/dtype,
      /*shape=*/xla::ifrt::Shape(array.getShape().getShape()),
      /*sharding=*/std::move(sharding)};
}

absl::StatusOr<std::vector<xla::ifrt::ArraySpec>> ExtractInSpecs(
    mlir::ModuleOp mlir_module, xla::ifrt::Client* client,
    const std::vector<xla::ifrt::Device*>& devices) {
  auto main_func = mlir_module.lookupSymbol<mlir::func::FuncOp>("main");
  TF_RET_CHECK(main_func != nullptr) << "Can't find `main` function";
  std::vector<xla::ifrt::ArraySpec> in_specs;
  in_specs.reserve(main_func.getNumArguments());
  for (const mlir::Type arg_type : main_func.getArgumentTypes()) {
    auto array_type = mlir::dyn_cast<xla::ifrt::IfrtArrayType>(arg_type);
    TF_RET_CHECK(array_type != nullptr)
        << "Unsupported argument type `" << mlir::debugString(arg_type) << "`";
    TF_ASSIGN_OR_RETURN(xla::ifrt::ArraySpec spec,
                        ToArraySpec(array_type, client, devices));
    in_specs.push_back(spec);
  }
  return in_specs;
}

absl::StatusOr<std::vector<xla::ifrt::ArraySpec>> ExtractOutSpecs(
    mlir::ModuleOp mlir_module, xla::ifrt::Client* client,
    const std::vector<xla::ifrt::Device*>& devices) {
  auto main_func = mlir_module.lookupSymbol<mlir::func::FuncOp>("main");
  TF_RET_CHECK(main_func != nullptr) << "Can't find `main` function";
  std::vector<xla::ifrt::ArraySpec> out_specs;
  out_specs.reserve(main_func.getNumResults());
  for (const mlir::Type result_type : main_func.getResultTypes()) {
    auto array_type = mlir::dyn_cast<xla::ifrt::IfrtArrayType>(result_type);
    TF_RET_CHECK(array_type != nullptr)
        << "Unsupported return type `" << mlir::debugString(result_type) << "`";
    TF_ASSIGN_OR_RETURN(xla::ifrt::ArraySpec spec,
                        ToArraySpec(array_type, client, devices));
    out_specs.push_back(spec);
  }
  return out_specs;
}

absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> BuildDefaultLayout(
    const xla::ifrt::ArraySpec& arg_spec, xla::ifrt::Client* client) {
  TF_ASSIGN_OR_RETURN(auto shard_shape,
                      arg_spec.sharding->GetShardShape(arg_spec.shape));
  return client->GetDefaultPjRtLayout(
      arg_spec.dtype, shard_shape.dims(),
      arg_spec.sharding->devices()->devices().front(),
      arg_spec.sharding->memory_kind());
}

absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
GetParameterLayoutFromLoadedExecutable(
    xla::ifrt::Client* client,
    const AtomExecutableMap& atom_program_executables,
    absl::Span<const xla::ifrt::ArraySpec> in_specs,
    absl::Span<const xla::ifrt::ArraySpec> out_specs,
    mlir::SymbolTableCollection& symbol_table,
    ifrt::CallLoadedExecutableOp& call_op, int param_operand_number) {
  // The parameter is used by a CallLoadedExecutableOp, return the layout
  // from the atom program executable.
  xla::ifrt::LoadedExecutableOp loaded_exec_op =
      call_op.getCalleeOp(symbol_table);
  auto atom_program_name = loaded_exec_op.getSymName().str();
  auto exec_it = atom_program_executables.find(atom_program_name);
  if (exec_it != atom_program_executables.end()) {
    TF_ASSIGN_OR_RETURN(auto exec_layouts,
                        exec_it->second->GetParameterLayouts());
    return std::move(exec_layouts[param_operand_number]);
  } else {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Could not find SPMD executable %s", atom_program_name));
  }
}

absl::Status PopulateLayouts(mlir::ModuleOp mlir_module,
                             xla::ifrt::Client* client,
                             const AtomExecutableMap& atom_program_executables,
                             absl::Span<xla::ifrt::ArraySpec> in_specs,
                             absl::Span<xla::ifrt::ArraySpec> out_specs) {
  auto main_func = xla::ifrt::GetMainFunction(mlir_module);
  mlir::SymbolTableCollection symbol_table;

  for (mlir::BlockArgument& arg : main_func.getArguments()) {
    std::shared_ptr<const xla::PjRtLayout> parameter_layout;
    if (arg.use_empty()) {
      // The argument is not used. Return device default layout.
      TF_ASSIGN_OR_RETURN(
          parameter_layout,
          BuildDefaultLayout(in_specs[arg.getArgNumber()], client));
    } else {
      bool found_copy_arrays_user = false;
      // Find the layout from the first LoadedExecutableOp consumer or just
      // return any of the users otherwise. Possible users: CopyArraysOp,
      // ReturnOp, and other LoadedExecutableOp.
      for (mlir::OpOperand& use : arg.getUses()) {
        if (llvm::isa<mlir::func::ReturnOp>(use.getOwner())) {
          continue;
        }
        if (llvm::isa<ifrt::CopyArraysOp>(use.getOwner())) {
          found_copy_arrays_user = true;
          continue;
        }

        if (!llvm::isa<ifrt::CallLoadedExecutableOp>(use.getOwner())) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "Layouts are supported only for programs that have parameters "
              "used only by CallLoadedExecutableOp ops. Parameter %d is used "
              "by %s",
              arg.getArgNumber(),
              xla::ifrt::OperationToString(use.getOwner(),
                                           mlir::OpPrintingFlags())));
        }
        auto call_op = llvm::cast<ifrt::CallLoadedExecutableOp>(use.getOwner());
        TF_ASSIGN_OR_RETURN(
            std::shared_ptr<const xla::PjRtLayout> consumer_layout,
            GetParameterLayoutFromLoadedExecutable(
                client, atom_program_executables, in_specs, out_specs,
                symbol_table, call_op, use.getOperandNumber()));
        if (!parameter_layout) {
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
      if (parameter_layout && found_copy_arrays_user) {
        // Need to check if the layout is compatible with the CopyArraysOp.
        TF_ASSIGN_OR_RETURN(
            std::shared_ptr<const xla::PjRtLayout> default_layout,
            BuildDefaultLayout(in_specs[arg.getArgNumber()], client));
        if (*parameter_layout != *default_layout) {
          return absl::InternalError(absl::StrFormat(
              "Parameter %d is used by atom program with layout: %s and in "
              "transfer op with layout: %s. This happens because support for "
              "layout progation within MPMD programs is limited. Contact "
              "ml-pathways-team@ for help",
              arg.getArgNumber(), parameter_layout->ToString(),
              default_layout->ToString()));
        }
      }
      if (!parameter_layout) {
        // The argument was skipped above, meaning only used by ReturnOp.
        TF_ASSIGN_OR_RETURN(
            parameter_layout,
            BuildDefaultLayout(in_specs[arg.getArgNumber()], client));
      }
    }
    in_specs[arg.getArgNumber()].layout = std::move(parameter_layout);
  }

  for (mlir::OpOperand& return_operand :
       main_func.front().getTerminator()->getOpOperands()) {
    auto& out_spec = out_specs[return_operand.getOperandNumber()];
    if (mlir::BlockArgument block_arg =
            llvm::dyn_cast<mlir::BlockArgument>(return_operand.get())) {
      // If result is a main func BlockArg, then it has already propagated the
      // layout above.
      out_spec.layout = in_specs[block_arg.getArgNumber()].layout;
      continue;
    }
    auto op_result = llvm::cast<mlir::OpResult>(return_operand.get());
    if (xla::ifrt::CallLoadedExecutableOp owner_call_op =
            llvm::dyn_cast<xla::ifrt::CallLoadedExecutableOp>(
                op_result.getOwner())) {
      xla::ifrt::LoadedExecutableOp loaded_exec_op =
          owner_call_op.getCalleeOp(symbol_table);
      auto atom_program_name = loaded_exec_op.getSymName().str();
      auto exec_it = atom_program_executables.find(atom_program_name);
      if (exec_it != atom_program_executables.end()) {
        TF_ASSIGN_OR_RETURN(auto exec_layouts,
                            exec_it->second->GetOutputLayouts());
        // Since this method is a temporary solution, we are ok with calling
        // GetOutputLayouts for an executable multiple times. In this way, we
        // avoid std::moving the same unique_ptr if an atom program result is
        // returned multiple times.
        out_spec.layout = std::move(exec_layouts[op_result.getResultNumber()]);
      } else {
        return absl::FailedPreconditionError(absl::StrFormat(
            "Could not find SPMD executable %s", atom_program_name));
      }
    } else if (llvm::isa<ifrt::CopyArraysOp>(op_result.getOwner())) {
      // The output is produced by a CopyArraysOp. Must be device
      // default layout.
      TF_ASSIGN_OR_RETURN(out_spec.layout,
                          BuildDefaultLayout(out_spec, client));
    } else {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Layouts are supported only for programs that have outputs produced "
          "by a CallLoadedExecutableOp. Produced by %s",
          xla::ifrt::OperationToString(op_result.getOwner(),
                                       mlir::OpPrintingFlags())));
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CompiledIfrtIrProgram> CompiledIfrtIrProgram::Create(
    std::unique_ptr<xla::ifrt::IfrtIRProgram> ifrt_ir_program,
    std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> ifrt_ir_compile_options,
    xla::ifrt::Client* client,
    std::shared_ptr<xla::ifrt::AtomProgramCompiler> atom_program_compiler) {
  TraceMe traceme([]() { return "ProgramCompiler::CompileForInterpreter"; });

  // Sharing the compile options with the passes and when pipeline is done add
  // it to the CompiledIfrtIrProgram.
  std::shared_ptr<xla::ifrt::IfrtIRCompileOptions> compile_options =
      std::move(ifrt_ir_compile_options);

  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(compile_options->device_assignments.size());
  for (const auto& device_id : compile_options->device_assignments) {
    TF_ASSIGN_OR_RETURN(devices.emplace_back(),
                        client->LookupDevice(device_id));
  }

  mlir::ModuleOp mlir_module = ifrt_ir_program->mlir_module;
  // Load the dialects necessary to compile the IFRT IR module.
  mlir::MLIRContext* context = mlir_module.getContext();
  xla::ifrt::support::RegisterMlirDialects(*context);

  std::string program_name = mlir_module.getName().value_or("unknown").str();

  // Add the bounded executables to the atom program executable map so that
  // they can be used by the interpreter
  std::shared_ptr<xla::ifrt::AtomExecutableMap> atom_executable_map =
      std::make_shared<xla::ifrt::AtomExecutableMap>(
          compile_options->loaded_exec_binding.begin(),
          compile_options->loaded_exec_binding.end());
  // Extract bindings.
  std::shared_ptr<xla::ifrt::AtomExecutableMap> bound_executable_map =
      std::make_shared<xla::ifrt::AtomExecutableMap>(
          compile_options->loaded_exec_binding.begin(),
          compile_options->loaded_exec_binding.end());

  std::vector<xla::ifrt::DeviceId> device_assignments =
      compile_options->device_assignments;

  // Run lowering passes.
  {
    mlir::PassManager pm(context);
    InitPassManager(pm, "ifrt.compile", compile_options->mlir_dump_to,
                    compile_options->mlir_dump_pass_re,
                    compile_options->mlir_dump_func_re,
                    compile_options->mlir_enable_timing);

    xla::ifrt::createIfrtToOutlinedAtomProgramsPipeline(pm);

    xla::ifrt::createIfrtPopulateAtomProgramMetadataPipeline(pm);

    OutlinedAtomProgramsToCompiledPipelineOptions compile_pipeline_options;
    for (const auto device : devices) {
      compile_pipeline_options.platform_names.push_back(
          std::string(device->PlatformName()));
    }
    TF_RETURN_IF_ERROR(xla::ifrt::createOutlinedAtomProgramsToCompiledPipeline(
        pm, std::move(atom_program_compiler), compile_pipeline_options,
        compile_options, atom_executable_map, std::move(bound_executable_map)));

    {
      TraceMe traceme(
          []() { return "ProgramCompiler::CompileForInterpreter::RunPasses"; });
      StatusScopedDiagnosticHandler diag_handler(context);
      if (mlir::failed(pm.run(mlir_module))) {
        return diag_handler.ConsumeStatus();
      }
    }
  }

  // Extract input and output specs from the modified `mlir_module`, which has
  // all array shardings specified.
  TF_ASSIGN_OR_RETURN(std::vector<xla::ifrt::ArraySpec> in_specs,
                      ExtractInSpecs(mlir_module, client, devices));
  TF_ASSIGN_OR_RETURN(std::vector<xla::ifrt::ArraySpec> out_specs,
                      ExtractOutSpecs(mlir_module, client, devices));

  absl::Status layout_status =
      PopulateLayouts(mlir_module, client, *atom_executable_map,
                      absl::MakeSpan(in_specs), absl::MakeSpan(out_specs));
  if (!layout_status.ok()) {
    for (auto& spec : in_specs) {
      spec.layout = nullptr;
    }
    for (auto& spec : out_specs) {
      spec.layout = nullptr;
    }
  }

  mlir::func::FuncOp main_func = xla::ifrt::GetMainFunction(mlir_module);
  std::vector<int> donatable_input_indices;
  for (const auto [idx, arg] : llvm::enumerate(main_func.getArguments())) {
    if (main_func.getArgAttr(idx, xla::ifrt::kIfrtDonatedArgAttrName) !=
        nullptr) {
      donatable_input_indices.push_back(idx);
    }
  }

  TF_ASSIGN_OR_RETURN(DeviceListRef device_list,
                      client->MakeDeviceList(devices));
  TF_ASSIGN_OR_RETURN(
      auto interpreter,
      ProgramInterpreter::Create(client, program_name, mlir_module,
                                 atom_executable_map, std::move(device_list)));
  TF_ASSIGN_OR_RETURN(auto execute_fn, interpreter->BuildExecuteFn());

  return CompiledIfrtIrProgram{
      /*program_name=*/std::move(program_name),
      /*atom_program_executables=*/std::move(atom_executable_map),
      /*in_specs=*/std::move(in_specs),
      /*out_specs=*/std::move(out_specs),
      /*layout_status=*/layout_status,
      /*donatable_input_indices=*/std::move(donatable_input_indices),
      /*program=*/std::move(ifrt_ir_program),
      /*device_assignments=*/std::move(device_assignments),
      /*compile_options=*/compile_options,
      /*execute_fn=*/std::move(execute_fn),
  };
}

}  // namespace ifrt
}  // namespace xla
