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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_

#include <cstdint>
#include <memory>

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tpu_passes.h"
#include "tensorflow/core/platform/status.h"

namespace mlir {
class PassManager;
}

namespace tensorflow {

namespace tfrt_compiler {

// Create a pass to insert kernels that copy fallback tensors when they are
// passed to multiple threads, to avoid atomic contention on their refcounts.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateInsertFallbackTensorCopyPass();

// Create a pass to reorder tf.Assert ops or tf.If ops that contains only
// tf.Assert ops to the end of the function, to avoid unnecessary control
// dependencies to other ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateReorderTfAssertPass();

// Create a pass to optimize the side-effect of control flow ops. eg. if both
// branches of a tf.If op contains only non-side-effecting ops, its
// `is_stateless` attribute will be set to true.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateOptimizeTfControlFlowSideEffectPass();

// Create a pass to remove tf.If ops' operands that are produced by tf.Const
// ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRemoveTfIfConstArgsPass();

// Create a pass to merge non-side-effecting tf.If ops that have the same
// operands.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateMergeTfIfOpsPass();

// Create a pass to deduplicate the function invoked by tf.BatchFunction with
// the same shared_name.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDeduplicateFunctionsInovkedByBatchFunctionPass();

// Create a pass to lower bound the number of threads in tf.BatchFunction.
struct ReconfigBatchOpPassOptions {
  int64_t min_num_batch_threads = 1;
  int64_t min_max_enqueued_batches = 1;
};
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateReconfigBatchOpPass(
    ReconfigBatchOpPassOptions options);

// Create a pass to fuse the TPU Ops for TFRT.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseTpuCompileAndExecutePass();

// Create a pass to optimize TF dialect for TFRT workflow.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateOptimizeTfForTfrtPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateTfrtXlaRewritePass();

// Create a pass to deduplicate results of tf.If ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDeduplicateIfResultPass();

}  // namespace tfrt_compiler

class CoreRTConverter;

// Create a pass that sink in the var handle op to the callee function when
// proper.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSinkInInvariantOpsPass();

// Create a pass that rewrites tf_saved_model dialect's ops according to TFRT's
// requirements.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateLowerTFSavedModelPass(bool hoist_invariant_ops,
                            bool fuse_get_resource_ops);

// Create a pass that converts ref variables to resource variables in a limited
// number of cases.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateConvertReferenceVariableToResourceVariablePass();

// Run *ToCoreRTConversionPassRun as free functions. Useful for
// reusing the pass logic in a custom pass with additional conversions.
mlir::LogicalResult TFSavedModelToCoreRTConversionPassRun(
    mlir::MLIRContext* context, mlir::func::FuncOp func,
    mlir::ConversionTarget* target, mlir::RewritePatternSet* patterns,
    CoreRTConverter* corert_converter);

// Create an operation pass that removes the device attribute from every
// corert.executeop.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRemoveDeviceAttributePass();

// Create an operation pass that inserts corert.transfer op to make sure any
// argument of any op is on the same device of the op itself.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateCrossDeviceTransferPass();

// Create a pass that converts MLIR TF dialect to MLIR TFRT dialect.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToTfrtConversionPass(const TfrtPipelineOptions& options);

// Creates a pipeline of passes that lowers MLIR TF dialect to TFRT dialects.
void CreateTfToTfrtPipeline(mlir::OpPassManager& pm,
                            const TfrtPipelineOptions& options);

// Creates a pipeline of passes that lowers MLIR TF dialect from tf.function to
// TFRT dialect. SavedModel related conversions are not included.
Status CreateTfExecutorToTfrtPipeline(mlir::PassManager& pm,
                                      const TfrtPipelineOptions& options);

// Creates a pipeline of passes that lowers MLIR TF Executor dialect to TF
// dialect for CoreRT purposes.
Status CreateTFExecutorToTFPipeline(mlir::PassManager& pm,
                                    const TfrtPipelineOptions& options);

// TODO(deqiangc): refactor below helpers once mlrt is OSSed.
void CreateTFExecutorToTFPreInvariantOptimizationPipelineHelper(
    mlir::OpPassManager& pm, const TfrtPipelineOptions& options);
void CreateTFExecutorToTFInvariantOptimizationPipelineHelper(
    mlir::OpPassManager& pm, const TfrtPipelineOptions& options);

Status CreateTFExecutorToTFPreInvariantOptimizationPipeline(
    mlir::PassManager& pm, const TfrtPipelineOptions& options);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_
