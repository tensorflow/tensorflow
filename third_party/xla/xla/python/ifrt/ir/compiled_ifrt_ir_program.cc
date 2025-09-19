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
#include <variant>
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

absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
GetParameterLayoutFromConsumer(
    xla::ifrt::Client* client,
    const AtomExecutableMap& atom_program_executables,
    absl::Span<const xla::ifrt::ArraySpec> in_specs,
    absl::Span<const xla::ifrt::ArraySpec> out_specs,
    mlir::SymbolTableCollection& symbol_table, mlir::OpOperand& param_operand) {
  if (auto call_op = llvm::dyn_cast<xla::ifrt::CallLoadedExecutableOp>(
          param_operand.getOwner())) {
    // The parameter is used by a CallLoadedExecutableOp, return the layout
    // from the atom program executable.
    xla::ifrt::LoadedExecutableOp loaded_exec_op =
        call_op.getCalleeOp(symbol_table);
    auto atom_program_name = loaded_exec_op.getSymName().str();
    auto exec_it = atom_program_executables.find(atom_program_name);
    if (exec_it != atom_program_executables.end()) {
      TF_ASSIGN_OR_RETURN(auto exec_layouts,
                          exec_it->second->GetParameterLayouts());
      return std::move(exec_layouts[param_operand.getOperandNumber()]);
    } else {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Could not find SPMD executable %s", atom_program_name));
    }
  } else if (auto return_op = llvm::dyn_cast<mlir::func::ReturnOp>(
                 param_operand.getOwner())) {
    // TODO(b/382761415): AUTO layouts should be handled during IFRT IR program
    // compilation so that by the time this method is called there should be no
    // AUTO layouts.
    // The parameter is not used by any atom program, return device default
    // layout.
    const auto& out_spec = out_specs[param_operand.getOperandNumber()];
    TF_ASSIGN_OR_RETURN(auto shard_shape,
                        out_spec.sharding->GetShardShape(out_spec.shape));
    return client->GetDefaultLayout(
        out_spec.dtype, shard_shape.dims(),
        out_spec.sharding->devices()->devices().front(),
        out_spec.sharding->memory_kind());
  } else if (auto copy_arrays =
                 llvm::dyn_cast<ifrt::CopyArraysOp>(param_operand.getOwner())) {
    // If the parameter is used by a CopyArraysOp, we assume a default layout.
    if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(param_operand.get())) {
      const auto& arg_spec = in_specs[arg.getArgNumber()];
      TF_ASSIGN_OR_RETURN(auto shard_shape,
                          arg_spec.sharding->GetShardShape(arg_spec.shape));
      return client->GetDefaultLayout(
          arg_spec.dtype, shard_shape.dims(),
          arg_spec.sharding->devices()->devices().front(),
          arg_spec.sharding->memory_kind());
    } else {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Parameter used by CopyArraysOp does not originate from a block "
          "argument. Parameter used by %s",
          xla::ifrt::OperationToString(param_operand.getOwner(),
                                       mlir::OpPrintingFlags())));
    }
  } else {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Layouts are supported only for programs that have parameters used "
        "only by CallLoadedExecutableOp ops. Used by %s",
        xla::ifrt::OperationToString(param_operand.getOwner(),
                                     mlir::OpPrintingFlags())));
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
      const auto& arg_spec = in_specs[arg.getArgNumber()];
      TF_ASSIGN_OR_RETURN(auto shard_shape,
                          arg_spec.sharding->GetShardShape(arg_spec.shape));
      TF_ASSIGN_OR_RETURN(parameter_layout,
                          client->GetDefaultLayout(
                              arg_spec.dtype, shard_shape.dims(),
                              arg_spec.sharding->devices()->devices().front(),
                              arg_spec.sharding->memory_kind()));
    } else {
      mlir::OpOperand& first_use = *arg.getUses().begin();
      TF_ASSIGN_OR_RETURN(parameter_layout,
                          GetParameterLayoutFromConsumer(
                              client, atom_program_executables, in_specs,
                              out_specs, symbol_table, first_use));
      for (mlir::OpOperand& use : llvm::drop_begin(arg.getUses())) {
        TF_ASSIGN_OR_RETURN(auto layout_from_executable,
                            GetParameterLayoutFromConsumer(
                                client, atom_program_executables, in_specs,
                                out_specs, symbol_table, use));
        // Verify that all uses of the parameter have the same layout.
        if (*parameter_layout != *layout_from_executable) {
          return absl::InternalError(absl::StrFormat(
              "Parameter %d is used by atom programs with incompatible "
              "layouts: %s vs. %s. This happens because support for layout "
              "progation within MPMD programs is limited. Contact "
              "ml-pathways-team@ for help",
              arg.getArgNumber(), parameter_layout->ToString(),
              layout_from_executable->ToString()));
        }
      }
    }
    in_specs[arg.getArgNumber()].layout = std::move(parameter_layout);
  }

  for (mlir::OpOperand& return_operand :
       main_func.front().getTerminator()->getOpOperands()) {
    auto& out_spec = out_specs[return_operand.getOperandNumber()];
    if (mlir::BlockArgument block_arg =
            llvm::dyn_cast<mlir::BlockArgument>(return_operand.get())) {
      // The output is an argument of the IFRT IR program. Assume device
      // default layout.
      TF_ASSIGN_OR_RETURN(auto shard_shape,
                          out_spec.sharding->GetShardShape(out_spec.shape));
      TF_ASSIGN_OR_RETURN(out_spec.layout,
                          client->GetDefaultLayout(
                              out_spec.dtype, shard_shape.dims(),
                              out_spec.sharding->devices()->devices().front(),
                              out_spec.sharding->memory_kind()));
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
    std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> compile_options,
    xla::ifrt::Client* client,
    std::shared_ptr<xla::ifrt::AtomProgramCompiler> atom_program_compiler) {
  TraceMe traceme([]() { return "ProgramCompiler::CompileForInterpreter"; });

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
    // We need to ensure that Multithreading is enabled in order to be able
    // to dispatch compilations from multiple threads. Otherwise, we would
    // trigger data races while printing the ModuleOps for creating the
    // compilation cache keys
    // (see llvm/llvm-project/mlir/lib/Support/StorageUniquer.cpp).
    // JAX currently disables Multithreading for all contexts, but temporarily
    // enabling multithreading here is safe because the context is not shared
    // across JAX ModuleOps.
    bool wasMultithreaded = context->isMultithreadingEnabled();
    context->enableMultithreading(true);
    absl::Cleanup reset_multithreading = [&]() {
      context->enableMultithreading(wasMultithreaded);
    };

    xla::ifrt::IfrtToOutlinedAtomProgramsPipelineOptions
        outline_pipeline_options;
    outline_pipeline_options.propagate_shardings =
        compile_options->propagate_shardings;
    xla::ifrt::createIfrtToOutlinedAtomProgramsPipeline(
        pm, outline_pipeline_options);

    xla::ifrt::createIfrtPopulateAtomProgramMetadataPipeline(pm);

    OutlinedAtomProgramsToCompiledPipelineOptions compile_pipeline_options;
    compile_pipeline_options.propagate_shardings =
        compile_options->propagate_shardings;
    for (const auto device : devices) {
      const auto it = device->Attributes().map().find("platform_name");
      if (it != device->Attributes().map().end()) {
        if (auto* const str = std::get_if<xla::ifrt::AttributeMap::StringValue>(
                &it->second)) {
          compile_pipeline_options.platform_names.push_back(str->value);
        } else {
          return absl::FailedPreconditionError(
              "Device platform name is not a string");
        }
      } else {
        compile_pipeline_options.platform_names.push_back(
            std::string(client->platform_name()));
      }
    }
    TF_RETURN_IF_ERROR(xla::ifrt::createOutlinedAtomProgramsToCompiledPipeline(
        pm, std::move(atom_program_compiler), compile_pipeline_options,
        std::move(compile_options), atom_executable_map,
        std::move(bound_executable_map)));

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

  return CompiledIfrtIrProgram{
      /*program_name=*/mlir_module.getName().value_or("unknown").str(),
      /*atom_program_executables=*/std::move(atom_executable_map),
      /*in_specs=*/std::move(in_specs),
      /*out_specs=*/std::move(out_specs),
      /*layout_status=*/layout_status,
      /*donatable_input_indices=*/std::move(donatable_input_indices),
      /*program=*/std::move(ifrt_ir_program),
      /*device_assignments=*/std::move(device_assignments),
  };
}

}  // namespace ifrt
}  // namespace xla
