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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace quantization {

Status PreprocessAndFreezeGraph(mlir::ModuleOp module,
                                mlir::MLIRContext* context,
                                llvm::Optional<Session*> session) {
  mlir::PassManager pm_before_freezing_variables(context);
  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
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
  pm_after_freezing_variables.addPass(mlir::createInlinerPass());

  if (failed(pm_before_freezing_variables.run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (session.has_value() && failed(mlir::tf_saved_model::FreezeVariables(
                                 module, session.getValue()))) {
    return statusHandler.ConsumeStatus();
  }

  if (failed(pm_after_freezing_variables.run(module))) {
    return statusHandler.ConsumeStatus();
  }

  return OkStatus();
}

}  // namespace quantization
}  // namespace tensorflow
