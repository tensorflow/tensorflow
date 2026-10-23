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
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf_ifrt_passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tpu_passes.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

// Points every IFRT call op (tf.IfrtCall or tf.AsyncIfrtCall) with
// `old_program_id` at the already registered `new_program_id`. A program is
// called by only one of the two op kinds, but a module may contain both kinds
// for different programs.
void ReplaceIfrtCallProgramId(mlir::ModuleOp module, int64_t old_program_id,
                              int64_t new_program_id) {
  mlir::Builder builder(module.getContext());
  auto update_program_id = [&](mlir::Operation* op) {
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("program_id")) {
      if (attr.getInt() == old_program_id) {
        op->setAttr("program_id", builder.getI64IntegerAttr(new_program_id));
      }
    }
  };
  module.walk([&](mlir::TF::IfrtCallOp call) {
    update_program_id(call.getOperation());
  });
  module.walk([&](mlir::TF::AsyncIfrtCallOp call) {
    update_program_id(call.getOperation());
  });
}

// Returns the `variable_arg_indices` shared by all call sites of `program_id`
// (which are either all tf.IfrtCall or all tf.AsyncIfrtCall), or std::nullopt
// if the call sites disagree.
std::optional<std::vector<int>> GetVariableArgIndices(mlir::ModuleOp module,
                                                      int64_t program_id) {
  std::optional<std::vector<int>> result;
  bool consistent = true;
  auto collect = [&](auto call) {
    if (call.getProgramId() != program_id) return;
    std::vector<int> indices;
    for (mlir::Attribute attr : call.getVariableArgIndices()) {
      indices.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
    }
    if (!result.has_value()) {
      result = std::move(indices);
    } else if (*result != indices) {
      consistent = false;
    }
  };
  module.walk([&](mlir::TF::IfrtCallOp call) { collect(call); });
  module.walk([&](mlir::TF::AsyncIfrtCallOp call) { collect(call); });
  if (!consistent) {
    return std::nullopt;
  }
  return result.value_or(std::vector<int>());
}

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
    // `tf_ifrt.modified_variable_names` is set on the outer module by
    // SinkVariableAsNamedArrayPass and lists variables written by host-side
    // AssignVariableOps anywhere in the client graph. CreatePrunedModule clones
    // it onto the submodule, but it says nothing about the TPU program. Drop it
    // so that identical TPU clusters extracted from different client graphs
    // (with different host-side writes) still get the same fingerprint below.
    submodule->get()->removeAttr("tf_ifrt.modified_variable_names");
    submodule->get().walk([&](mlir::func::FuncOp func) {
      if (func.getSymName() == entry_function_name) {
        func.setName("main");
        func.setSymName("main");
        func.setPublic();
      }
    });
    // Remove the program id attribute from the submodule because they are not
    // needed and will prevent us generating consistent cache key.
    // program id is already in ifrt_call op's attribute and that part is not
    // touched here.
    submodule->get()->walk([](mlir::func::FuncOp func) {
      func->removeAttr("tfrt_ifrt_serving.program_id");
    });

    const uint64_t submodule_fingerprint =
        MlirModuleFingerprint(submodule->get());
    // The executable binds loaded variables by the call site's
    // `variable_arg_indices`, which are not part of the submodule, so a program
    // is only reused by call sites with the same indices. If call sites of
    // this program disagree, skip the cache.
    const std::optional<std::vector<int>> variable_arg_indices =
        GetVariableArgIndices(module, program_id);
    if (variable_arg_indices.has_value()) {
      if (std::optional<int64_t> existing_program_id =
              ifrt_model_context.LookupProgramId(submodule_fingerprint,
                                                 *variable_arg_indices);
          existing_program_id.has_value()) {
        ReplaceIfrtCallProgramId(module, program_id, *existing_program_id);
        continue;
      }
    }

    if (ifrt_model_context.IsFrozen()) {
      if (ifrt_model_context.HasProgramWithFingerprint(submodule_fingerprint)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Cannot compile IFRT programs after the model is frozen. The TPU "
            "program was compiled during warmup, but with different "
            "variable_arg_indices than this call site [",
            variable_arg_indices.has_value()
                ? absl::StrJoin(*variable_arg_indices, ", ")
                : "inconsistent",
            "]."));
      }
      return absl::FailedPreconditionError(
          "Cannot compile IFRT programs after the model is frozen. Please make "
          "sure warmup covers all signatures by following go/tf-model-warmup.");
    }

    TF_ASSIGN_OR_RETURN(
        auto executable,
        IfrtServingExecutable::Create(
            program_id, model_name, entry_function_name.str(),
            *std::move(submodule), ifrt_model_context.GetClient(),
            &ifrt_model_context.GetThreadPool(),
            &ifrt_model_context.GetLoadedVariableRegistry(),
            &ifrt_model_context.GetRestoreTensorRegistry(),
            ifrt_model_context.checkpoint_loader_queue(),
            ifrt_model_context.GetDeviceMgr(),
            ifrt_model_context.GetShapeRepresentationFn(),
            ifrt_model_context.GetIfrtServingCoreSelector(),
            ifrt_model_context.GetCompilationEnvOrOverrides(),
            ifrt_model_context.GetTfToHloCompiler(),
            ifrt_model_context.GetPersistentCompilationCache(),
            ifrt_model_context.GetH2DTransferExecutorFactory(),
            ifrt_model_context.use_output_arena(),
            ifrt_model_context.use_undonatable_buffer_converter()));

    // Register the Ifrt program to `ServingExecutableRegistry` so that
    // the client TF program can invoke them via `IfrtCall` op.
    TF_ASSIGN_OR_RETURN(auto handle, ServingExecutableRegistry::Register(
                                         program_id, std::move(executable)));

    if (variable_arg_indices.has_value()) {
      ifrt_model_context.RegisterProgramId(submodule_fingerprint,
                                           *variable_arg_indices, program_id);
    }
    handles.push_back(std::move(handle));
  }

  return handles;
}

absl::Status CompileTensorflowForIfrtServing(
    absl::string_view model_name, IfrtModelContext& ifrt_model_context,
    mlir::ModuleOp module, bool enable_async_ifrt) {
  tsl::profiler::TraceMe trace_me("CompileTensorflowForIfrtServing");
  mlir::Builder builder(module.getContext());

  TF_RETURN_IF_ERROR(RunClusterToIfrtRuntimeOpsPassPipeline(
      module, model_name,
      ifrt_model_context.enable_propagate_static_shapes_pass(),
      enable_async_ifrt));

  // Collect the modified-variable report emitted by
  // SinkVariableAsNamedArrayPass. The union of these reports across all
  // compiled modules identifies host-needed variables for
  // IfrtModelContext::Freeze() (freeze-time host variable mode). This is
  // only complete if every signature is compiled before Freeze() is called.
  if (!ifrt_model_context.IsFrozen()) {
    if (auto modified = module->getAttrOfType<mlir::ArrayAttr>(
            "tf_ifrt.modified_variable_names")) {
      for (mlir::Attribute attr : modified) {
        if (auto name = mlir::dyn_cast<mlir::StringAttr>(attr)) {
          // Ignore NOT_FOUND errors: some modified variables may be host-only
          // variables managed directly in ResourceManager and not present in
          // the checkpoint restore registry.
          ifrt_model_context.GetRestoreTensorRegistry()
              .SetUsedByHost(name.str())
              .IgnoreError();
        }
      }
    }
  }

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

  TfrtTpuCompileOptions options;
  options.disable_set_default_tpu_device_and_device_assignment_attributes =
      compile_options_
          .disable_set_default_tpu_device_and_device_assignment_attributes;
  options.support_multi_dims_sharding = true;

  if (tpu_compiler_ != nullptr) {
    // Run backward compat pass so that we can use bridge to do clustering.
    if (mlir::failed(
            tpu_compiler_->RunTPUBackwardCompatConversion(module, options))) {
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
      model_context.name(), **ifrt_model_context, module,
      model_context.graph_execution_options()
          .compile_options.enable_async_ifrt));

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
