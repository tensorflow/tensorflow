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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mhlo/transforms/passes.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/rename_entrypoint_to_main.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace quantization {

using ::mlir::quant::stablehlo::AddXlaCallModuleOpDeserializationPasses;

// Adds passes that unfuse MHLO ops that do not have their equivalents in
// StableHLO.
void AddUnfuseMhloOpsPasses(mlir::PassManager& pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeEinsumToDotGeneralPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeDotToDotGeneralPass());
  // Unfuse mhlo BatchNorm to primitive ops.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::odml::createUnfuseBatchNormPass());
  // Fuse Conv + Mul to Conv.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::odml::createFuseConvolutionPass());
  // Fold broadcast_in_dim + Mul.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::odml::createFoldBroadcastPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeTorchIndexSelectToGatherPass());
}

// Converts TF SavedModel to StableHLO module. The input TF SavedModel can have
// StableHLO module serialized into a XlaCallModuleOp. (ex: JAX/PyTorch models)
void AddTFToStablehloPasses(
    mlir::PassManager& pm,
    llvm::ArrayRef<llvm::ArrayRef<int64_t>> input_arg_shapes) {
  pm.addPass(mlir::odml::CreateRenameEntrypointToMainPass());
  // TODO: b/230572023 - Consider improving shape inference for While op instead
  // of dropping the attribute. This need not be correct for models not trained
  // on TPU.
  // Extracts the StableHLO module from tf.XlaCallModuleOp if the StableHLO
  // module is serialized in it.
  pm.addPass(mlir::stablehlo::CreateLegalizeTFXlaCallModuleToStablehloPass());

  // Preprocesses TPU-targeting StableHLO module for support in TF Quantizer.
  pm.addPass(mlir::quant::CreateConvertTpuModelToCpuPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::quant::CreateCastBf16OpsToF32Pass());

  // Optimizes the graph via cleanups, merges, rewrites, constant folding,
  // and edge case handling where possible.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateDropWhileShapeInvariantPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Propagates shapes on the TensorFlow graph.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass(input_arg_shapes));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  // FreezeVariables only freezes variables for TF v1 types. Separately handle
  // freezing of TF v2 GlobalTensor ops. (Ref: b/206855389)
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
      /*allow_mutable_tensors=*/true));

  // Generic MLIR optimization passes.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass(input_arg_shapes));

  // Legalizes TF UniformQuantized types into MHLO. Part of the official
  // TF/XLA bridge component.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::stablehlo::CreateConvertTFQuantOpsToMHLOPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // TF -> StableHLO legalization.
  // Skip StatefulPartitionedCall to preserve aliased functions.
  mlir::odml::AddLegalizeTFToStablehloPasses(pm, /*skip_quantization_ops=*/true,
                                             /*skip_resize=*/false,
                                             /*skip_partitioned_calls=*/true);
  // StableHLO -> MHLO legalization for MHLO optimization.
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  // Rewrites legacy StableHLO ops.
  AddUnfuseMhloOpsPasses(pm);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // MHLO -> StableHLO legalization.
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

absl::Status PreprocessAndFreezeGraph(
    const absl::string_view mlir_dump_file_prefix, const bool is_inliner_run,
    const absl::flat_hash_set<std::string>& noinline_functions,
    mlir::ModuleOp module_op, mlir::MLIRContext* context,
    std::optional<Session*> session, const bool run_tf_to_stablehlo,
    const bool deserialize_xla_call_module,
    llvm::ArrayRef<llvm::ArrayRef<int64_t>> input_arg_shapes) {
  mlir::PassManager pm_before_freezing_variables(context);
  mlir::StatusScopedDiagnosticHandler statusHandler(module_op.getContext(),
                                                    /*propagate=*/true);

  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = false;
  mlir::TF::CreateTFStandardPipeline(pm_before_freezing_variables,
                                     standard_pipeline_options);

  // The AddQuantizationUnitLocPass should be added before any other passes.
  pm_before_freezing_variables.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateAddQuantizationUnitLocPass());
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
  if (run_tf_to_stablehlo) {
    // AddLegalizeTFToStablehloPasses expects frozen TF variables when
    // legalizing to stablehlo.constant.
    AddTFToStablehloPasses(pm_after_freezing_variables, input_arg_shapes);
  }

  if (deserialize_xla_call_module) {
    // Deserialize the StableHLO module embedded in tf.XlaCallModule and lifts
    // the StableHLO functions to the top level module. This is needed for
    // StableHLO quantization. Also restores some shape information for
    // XlaCallModuleOps and CustomAggregatorOps lost from the calibration step.
    AddXlaCallModuleOpDeserializationPasses(pm_after_freezing_variables);
  }

  if (const auto pre_variable_freezing_status = RunPassesOnModuleOp(
          /*mlir_dump_file_name=*/absl::StrCat(
              mlir_dump_file_prefix, "_preprocess_pre_variable_freezing"),
          pm_before_freezing_variables, module_op);
      !pre_variable_freezing_status.ok()) {
    return pre_variable_freezing_status;
  }

  if (!session.has_value() || !*session) {
    mlir::PassManager pm_freezing_variables(context);
    // This pass does resource analysis of saved model global tensors and marks
    // those deemed read-only as immutable.
    pm_freezing_variables.addPass(
        mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());

    pm_freezing_variables.addPass(
        mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
            /*allow_mutable_tensors=*/true));

    pm_freezing_variables.addPass(
        mlir::TFL::CreateUnfreezeMutableGlobalTensorsPass());

    if (const auto variable_freezing_status = RunPassesOnModuleOp(
            /*mlir_dump_file_name=*/absl::StrCat(
                mlir_dump_file_prefix, "_preprocess_variable_freezing"),
            pm_freezing_variables, module_op);
        !variable_freezing_status.ok()) {
      return variable_freezing_status;
    }
  } else if (failed(
                 mlir::tf_saved_model::FreezeVariables(module_op, *session))) {
    return statusHandler.ConsumeStatus();
  }

  return RunPassesOnModuleOp(
      /*mlir_dump_file_name=*/absl::StrCat(
          mlir_dump_file_prefix, "_preprocess_post_variable_freezing"),
      pm_after_freezing_variables, module_op);
}

}  // namespace quantization
}  // namespace tensorflow
