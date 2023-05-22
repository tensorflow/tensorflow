/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/debugging/mlir_dump.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace quantization {
namespace {

absl::Status RunPassesOnModuleOp(const absl::string_view mlir_dump_file_name,
                                 mlir::PassManager& pass_manager,
                                 mlir::ModuleOp module_op) {
  mlir::StatusScopedDiagnosticHandler statusHandler(module_op.getContext(),
                                                    /*propagate=*/true);

  const absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> dump_file =
      MaybeEnableIrPrinting(pass_manager, mlir_dump_file_name);
  if (!dump_file.ok()) {
    return dump_file.status();
  }

  if (failed(pass_manager.run(module_op))) {
    return statusHandler.ConsumeStatus();
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status PreprocessAndFreezeGraph(
    const absl::string_view mlir_dump_file_prefix, const bool is_inliner_run,
    const absl::flat_hash_set<std::string>& noinline_functions,
    mlir::ModuleOp module_op, mlir::MLIRContext* context,
    std::optional<Session*> session) {
  mlir::PassManager pm_before_freezing_variables(context);
  mlir::StatusScopedDiagnosticHandler statusHandler(module_op.getContext(),
                                                    /*propagate=*/true);

  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = false;
  mlir::TF::CreateTFStandardPipeline(pm_before_freezing_variables,
                                     standard_pipeline_options);

  pm_before_freezing_variables.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  mlir::PassManager pm_after_freezing_variables(context);
  pm_after_freezing_variables.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm_after_freezing_variables.addPass(mlir::createCanonicalizerPass());

  // Makes certain functions immune to the `InlinerPass`. Used to preserve
  // aliased functions.
  pm_after_freezing_variables.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateMarkFunctionsNoinlinePass(std::vector<std::string>(
          noinline_functions.begin(), noinline_functions.end())));
  if (is_inliner_run) {
    pm_after_freezing_variables.addPass(mlir::createInlinerPass());
  }

  if (const auto pre_variable_freezing_status = RunPassesOnModuleOp(
          /*mlir_dump_file_name=*/absl::StrCat(
              mlir_dump_file_prefix, "_preprocess_pre_variable_freezing"),
          pm_before_freezing_variables, module_op);
      !pre_variable_freezing_status.ok()) {
    return pre_variable_freezing_status;
  }

  if (session.has_value() && failed(mlir::tf_saved_model::FreezeVariables(
                                 module_op, session.value()))) {
    return statusHandler.ConsumeStatus();
  }

  return RunPassesOnModuleOp(
      /*mlir_dump_file_name=*/absl::StrCat(
          mlir_dump_file_prefix, "_preprocess_post_variable_freezing"),
      pm_after_freezing_variables, module_op);
}

}  // namespace quantization
}  // namespace tensorflow
