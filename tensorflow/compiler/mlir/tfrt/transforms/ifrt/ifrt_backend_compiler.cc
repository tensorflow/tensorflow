/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_backend_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tpu_passes.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
absl::StatusOr<std::vector<ServingExecutableRegistry::Handle>>
CompileAndRegisterIfrtPrograms(absl::string_view model_name,
                               mlir::ModuleOp module,
                               IfrtModelContext& ifrt_model_context) {
  std::vector<ServingExecutableRegistry::Handle> handles;

  // Compile Ifrt programs and register the executables. Outlined Ifrt
  // programs are marked with `tfrt_ifrt_serving.program_id` attributes.
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    int64_t program_id;
    if (auto attr = func->getAttrOfType<mlir::IntegerAttr>(
            "tfrt_ifrt_serving.program_id")) {
      program_id = attr.getInt();
    } else {
      continue;
    }

    mlir::StatusScopedDiagnosticHandler diag_handler(module->getContext());
    auto entry_function_name = func.getSymName();
    auto submodule = mlir::TF::CreatePrunedModule(module, entry_function_name);
    if (mlir::failed(submodule)) {
      return diag_handler.ConsumeStatus();
    }

    // Remove the attribute inherited from saved model loading. They impose
    // additional constraint on public functions that are not necessary.
    submodule->get()->removeAttr("tf_saved_model.semantics");
    submodule->get().walk([&](mlir::func::FuncOp func) {
      if (func.getSymName() == entry_function_name) {
        func.setName("main");
        func.setSymName("main");
        func.setPublic();
      }
    });

    auto executable = std::make_unique<IfrtServingExecutable>(
        model_name, entry_function_name.str(), *std::move(submodule),
        ifrt_model_context.GetClient(), &ifrt_model_context.GetThreadPool(),
        &ifrt_model_context.GetLoadedVariableRegistry(),
        ifrt_model_context.GetDeviceMgr(),
        ifrt_model_context.GetShapeRepresentationFn());

    // Register the Ifrt program to `ServingExecutableRegistry` so that
    // the client TF program can invoke them via `IfrtCall` op.
    TF_ASSIGN_OR_RETURN(auto handle, ServingExecutableRegistry::Register(
                                         program_id, std::move(executable)));

    handles.push_back(std::move(handle));
  }

  return handles;
}

absl::Status CompileTensorflowForIfrtServing(
    absl::string_view model_name, IfrtModelContext& ifrt_model_context,
    mlir::ModuleOp module) {
  tsl::profiler::TraceMe trace_me("CompileTensorflowForIfrtServing");
  mlir::Builder builder(module.getContext());

  TF_RETURN_IF_ERROR(
      RunClusterToIfrtRuntimeOpsPassPipeline(module, model_name));

  TF_ASSIGN_OR_RETURN(
      auto handles,
      CompileAndRegisterIfrtPrograms(model_name, module, ifrt_model_context));

  for (auto& handle : handles) {
    ifrt_model_context.RegisterHandle(std::move(handle));
  }

  return absl::OkStatus();
}

}  // namespace

// Compile ifrt programs in TF dialect into ifrt executables.
// Remove ifrt programs afterwards.
absl::Status IfrtBackendCompiler::CompileTensorflow(
    tensorflow::tfrt_stub::ModelRuntimeContext& model_context,
    mlir::ModuleOp module) const {
  auto ifrt_model_context =
      model_context.resource_context().GetResource<IfrtModelContext>(
          kIfrtModelContextName);
  if (!ifrt_model_context.has_value()) {
    return absl::InternalError(
        "Failed to find model context for ifrt serving.");
  }

  mlir::StatusScopedDiagnosticHandler diag_handler(module->getContext());
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_tpu_bct_conversion_before", module);
  }

  if (tpu_compiler_ != nullptr) {
    // Run backward compat pass so that we can use bridge to do clustering.
    if (mlir::failed(
            tpu_compiler_->RunTPUBackwardCompatConversion(module, {}))) {
      return diag_handler.Combine(
          absl::InternalError("Failed to handle legacy TPU Ops"));
    }
  }
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_tpu_bct_conversion_after", module);
  }

  // Use bridge for cluster formation.
  TF_RETURN_IF_ERROR(tensorflow::tf2xla::v2::RunFunctionTf2xlaClusteringBridge(
      module, /*is_supported_by_replicated_brige*/ true,
      /*is_in_fallback_enabled_mode=*/false));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("before_ifrt_outlining", module);
  }

  // Extract TPU program for IFRT call.
  TF_RETURN_IF_ERROR(CompileTensorflowForIfrtServing(
      model_context.name(), **ifrt_model_context, module));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("after_ifrt_outlining", module);
  }

  // IFRT program is no longer needed.
  llvm::SmallVector<mlir::func::FuncOp> to_erase;
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    if (func->getAttr("tfrt_ifrt_serving.program_id")) {
      to_erase.push_back(func);
    }
  }
  for (auto func : to_erase) {
    func->erase();
  }

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("after_ifrt_program_removal", module);
  }

  if (mlir::failed(mlir::verify(module))) {
    return diag_handler.ConsumeStatus();
  }

  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
