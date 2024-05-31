
/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_asset_sinking_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/set_shape_invariant_in_while_ops.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

// Assigns devices so that later passes can utilize device information.
// Device assignment might have not been done by the upstream pipeline, or get
// removed by previous passes. However, we assume most of the device assignment
// has been done by the upstream pipeline, so we simply assign the default
// device to unassigned ops. Specifically, we do assignment for ConstOp first to
// place it on the same device as its user operation, instead of placing it on
// the default device blindly.
// TODO(b/221297389): Figure out a more robust way to handle dropped device
// assignment.
void AddTfDeviceAssignmentPasses(mlir::OpPassManager &pm,
                                 const TfrtPipelineOptions &options) {
  pm.addPass(mlir::TF::CreateConstantOpDeviceAssignmentPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateTFDeviceAssignmentByFuncAttrPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateSimpleTFDeviceAssignmentPass(options.default_device));
}

}  // namespace

void CreateTFExecutorToTFPreInvariantOptimizationPipelineHelper(
    mlir::OpPassManager &pm, const TfrtPipelineOptions &options) {
  // Due to b/191304670, functionalized while ops might not have the
  // shape_invariant attribute set correctly, which leads to failure in shape
  // inference. As a workaround, we conservatively (e.g., we place less
  // restrictions on tf.while which will avoid failures but lead to potentially
  // less exact shape inference) set the shape_invariant attribute in all
  // tf.While ops before performing shape inference.
  //
  // Note that this pass might not work well with TF XLA bridge, but this is
  // fine as TF XLA bridge is run before this pipeline. For CPU ops, less exact
  // shape inference may lead to fewer optimizations but it should be fine as it
  // is limited to while ops currently.
  //
  // TODO(b/191304670): Remove this pass once the shape_invariant attribute is
  // set correctly in the upstream.
  pm.addNestedPass<mlir::func::FuncOp>(
      tfrt_compiler::CreateSetShapeInvariantInWhileOps());

  // We pass the MLIR module through the TF standard pipeline, which for
  // instances does shape inference, canonicalization, inlining, etc.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tf_executor::CreateTFExecutorIslandCoarseningPass());

  AddTfDeviceAssignmentPasses(pm, options);

  pm.addPass(tfrt_compiler::CreateTfrtXlaRewritePass());

  // Here we perform TFRT specific optimization before standard TF optimization,
  // as TFRT-specific optimization may create more opportunities.
  pm.addNestedPass<mlir::func::FuncOp>(
      tfrt_compiler::CreateOptimizeTfForTfrtPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateTFOptimizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  AddTfDeviceAssignmentPasses(pm, options);

  // After the standard pass, we now have MLIR in TF dialect, and now we convert
  // reference variable to resource variables, which is besteffort.
  pm.addPass(CreateConvertReferenceVariableToResourceVariablePass());

  // Move the tf.Assert op to the end of the function, so that it does not
  // impose unnecessary control dependencies on other ops.
  pm.addPass(tfrt_compiler::CreateReorderTfAssertPass());

  // Optimze the side-effects of control flow ops by examining the ops in its
  // callees.
  pm.addPass(tfrt_compiler::CreateOptimizeTfControlFlowSideEffectPass());

  // Remove tf.If ops' operands that are produced by tf.Const ops.
  pm.addPass(tfrt_compiler::CreateRemoveTfIfConstArgsPass());

  // Merge non-side-effecting tf.If ops if their operands are the same.
  pm.addPass(tfrt_compiler::CreateMergeTfIfOpsPass());

  // Lower bound on the number of batch threads in `tf.BatchFunction`.
  pm.addPass(tfrt_compiler::CreateReconfigBatchOpPass(
      {.min_num_batch_threads = options.min_num_batch_threads,
       .min_max_enqueued_batches = options.min_max_enqueued_batches}));

  // Deduplicate functions invoked by tf.BatchFunction with the same
  // shared_name
  pm.addPass(
      tfrt_compiler::CreateDeduplicateFunctionsInovkedByBatchFunctionPass());

  // RemoveUnusedWhileResultsPass operates on the region-based control flow, so
  // the functional control flow is first converted to region-based control
  // flow, which is converted back after the optimization passes are performed.
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateRemoveUnusedWhileResultsPass());
  pm.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());

  // Apply standard optimization after optimizing control flow ops.
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // TODO(b/187876545): An extra shape inference pass is added because it does
  // not work well with tf.Identity op that remove ref type. So we work around
  // by performing shape inference again after reference variable to resource
  // variable conversion. We should remove this after b/187876545 is fixed.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateLaunchToDeviceAttributePass());

  // After all standard passes run layout optimization to assign optimal data
  // format for all layout sensitive operations.
  mlir::TF::LayoutOptimizationPipelineOptions layout_optimization_options;
  layout_optimization_options.force_data_format =
      options.force_data_format.getValue();
  // TODO(b/191304261): Folding transpose in ops is buggy in the layout
  // optimization pass. Disable it to avoid errors in b/191304261. This should
  // not affect CPU performance as it does not change the number of ops, nor
  // does it change the types of the ops.
  layout_optimization_options.skip_fold_transpose_in_ops = true;
  mlir::TF::CreateLayoutOptimizationPipeline(pm.nest<mlir::func::FuncOp>(),
                                             layout_optimization_options);

  // Run canonicalization pipeline to remove unused constants and bypassed
  // transpose operations left in the IR after layout optimization.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  // Decompose resource ops as resource variables will be converted to tensors
  // directly.
  if (options.decompose_resource_ops)
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TFDevice::CreateDecomposeResourceOpsPass());

  AddTfDeviceAssignmentPasses(pm, options);

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateTensorDeviceCopyConversionPass());

  // Rewriter operation sequences to device specific fusions.
  DeviceNameUtils::ParsedName parsed_name;

  // Ignore error.
  bool success =
      DeviceNameUtils::ParseFullName(options.default_device, &parsed_name);
  assert(success && "default device is invalid");
  (void)success;

  if (parsed_name.has_type && parsed_name.type == DEVICE_GPU)
    pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateGpuOpFusionPass());

  if (parsed_name.has_type && parsed_name.type == DEVICE_CPU)
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateFusedKernelMatcherPass());

  if (options.tpu_fuse_ops) {
    pm.addNestedPass<mlir::func::FuncOp>(
        tfrt_compiler::CreateFuseTpuCompileAndExecutePass());
    // Remove ops for the input to _TPUCompileMlirOp, which are no longer needed
    // after CreateFuseTpuCompileAndExecutePass
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  }

  AddTfDeviceAssignmentPasses(pm, options);
}

void CreateTFExecutorToTFInvariantOptimizationPipelineHelper(
    mlir::OpPassManager &pm, const TfrtPipelineOptions &options) {
  if (options.sink_in_invariant_ops) {
    pm.addPass(CreateSinkInInvariantOpsPass());
  }

  if (!options.saved_model_dir.empty()) {
    pm.addPass(
        mlir::tf_saved_model::CreateAssetSinkingPass(options.saved_model_dir));
  }

  pm.addPass(CreateLowerTFSavedModelPass(
      options.hoist_invariant_ops, options.fuse_get_resource_ops_in_hoisting));
}

Status ValidateTfrtPipelineOptions(const TfrtPipelineOptions &options) {
  if (options.target_tpurt && options.target_gpu) {
    return tensorflow::errors::Internal(
        "Invalid pipeline options. Targeting both TPU and GPU is not "
        "supported.");
  }
  return absl::OkStatus();
}

Status CreateTFExecutorToTFPreInvariantOptimizationPipeline(
    mlir::PassManager &pm, const TfrtPipelineOptions &options) {
  TF_RETURN_IF_ERROR(ValidateTfrtPipelineOptions(options));
  if (VLOG_IS_ON(1)) {
    // Print the whole module after each pass, which requires disabling
    // multi-threading as well.
    pm.getContext()->disableMultithreading();
    pm.enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>(
        /*print_module_scope=*/true));
  }
  CreateTFExecutorToTFPreInvariantOptimizationPipelineHelper(pm, options);
  return absl::OkStatus();
}

}  // namespace tensorflow
