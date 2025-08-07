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
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
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

  return CompiledIfrtIrProgram{
      /*program_name=*/mlir_module.getName().value_or("unknown").str(),
      /*atom_program_executables=*/std::move(atom_executable_map),
      /*in_specs=*/std::move(in_specs),
      /*out_specs=*/std::move(out_specs),
      /*program=*/std::move(ifrt_ir_program),
      // TODO(b/382761415): Do not store the mlir module once we can get the
      // layouts from the IFRT IR ArrayType. It is currently necessary to clone
      // to avoid accessing the module from different threads.
      /*mlir_module=*/mlir::OwningOpRef<mlir::ModuleOp>(mlir_module.clone()),
      /*device_assignments=*/std::move(device_assignments),
  };
}

}  // namespace ifrt
}  // namespace xla
