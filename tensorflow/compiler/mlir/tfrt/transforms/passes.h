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

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tpu_passes.h"

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

// Create a pass to fuse the TPU Ops for TFRT.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseTpuCompileAndExecutePass();

// Create a pass to optimize TF dialect for TFRT workflow.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateOptimizeTfForTfrtPass();

}  // namespace tfrt_compiler

class CoreRTConverter;

// Create a pass that rewrites tf_saved_model dialect's ops according to TFRT's
// requirements.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateLowerTFSavedModelPass(bool hoist_invariant_ops);

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

// Create an operation pass that converts each tfrt_dist.remote_execute_func op
// into a combination of tfrt_dist.register_tfrt_function op and
// tfrt_dist.remote_execute op.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDistRemoteRunEncapsulatePass();

// Create an operation pass that removes the device attribute from every
// corert.executeop.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRemoveDeviceAttributePass();

// Create an operation pass that inserts corert.transfer op to make sure any
// argument of any op is on the same device of the op itself.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateCrossDeviceTransferPass();

struct TfrtPipelineOptions
    : public mlir::PassPipelineOptions<TfrtPipelineOptions> {
  Option<std::string> default_device{
      *this, "default-device", llvm::cl::desc("default device assignment"),
      llvm::cl::init("/job:localhost/replica:0/task:0/device:CPU:0")};
  Option<bool> enable_optimizer{
      *this, "enable-optimizer",
      llvm::cl::desc("run optimization passes on corert dialect"),
      llvm::cl::init(false)};
  Option<bool> decompose_resource_ops{
      *this, "decompose-resource-ops",
      llvm::cl::desc("decompose composite resource ops into ReadVariableOp and "
                     "non-resource ops. This is currently used in TFRT "
                     "savedmodel pipeline."),
      llvm::cl::init(false)};
  Option<std::string> force_data_format{
      *this, "force-data-format",
      llvm::cl::desc("force data format for all layout sensitive operations")};
  // TODO(tfrt-devs): consider making compiler to figure out whether to fold
  // transpose or not instead of exposing the specific option.
  Option<bool> skip_fold_transpose_in_ops{
      *this, "skip-fold-transpose-in-ops",
      llvm::cl::desc("Skip folding transpose operands in Ops which can support "
                     "different layouts.")};
  Option<bool> target_tpurt{*this, "target-tpurt",
                            llvm::cl::desc("target TPURT dialect if true"),
                            llvm::cl::init(false)};
  Option<bool> tpu_use_core_selector{
      *this, "tpu-use-core-selector",
      llvm::cl::desc("If true, use ServingCoreSelector to pick TPU core. "
                     "Otherwise, use the assigned core. Currently we use "
                     "core selector for Servo serving use cases."),
      llvm::cl::init(true)};
  Option<bool> tpu_use_bundled_transfer{
      *this, "tpu-use-bundled-transfer",
      llvm::cl::desc("If true, use BundledTransferToTpuOp to transfer "
                     "variables and input tensors to TPU."),
      llvm::cl::init(true)};
  Option<bool> tpu_lower_to_fallback{
      *this, "tpu-lower-to-fallback",
      llvm::cl::desc("If true, lower an TF op that's placed on TPU device "
                     "to be executed by tfrt_fallback.execute."),
      llvm::cl::init(true)};
  Option<bool> tpu_fuse_ops{
      *this, "tpu-fuse-ops",
      llvm::cl::desc("If true, use the TPU fused compile_and_execute kernel"),
      llvm::cl::init(false)};
  // TODO(b/194081364): remove this option once we unify servo TPU serving
  // result transfer behavior.
  Option<bool> tpu_transfer_result_to_host{
      *this, "tpu-transfer-result-to-host",
      llvm::cl::desc("If true, transfer the result of tpurt.execute from TPU "
                     "to host."),
      llvm::cl::init(true)};
  Option<bool> use_tpu_host_allocator_for_inputs{
      *this, "use-tpu-host-allocator-for-inputs",
      llvm::cl::desc("If true, fallback executeops that produce inputs to tpu "
                     "program will use tpu host allocator."),
      llvm::cl::init(false)};
  Option<bool> enable_native_ops{
      *this, "enable-native-ops",
      llvm::cl::desc(
          "If true, native ops will be used on an opt-in basis instead of "
          "fallback ops. If false, no native ops are used."),
      llvm::cl::init(true)};
  Option<bool> func_use_fallback_tensor{
      *this, "func-use-fallback-tensor",
      llvm::cl::desc(
          "If true, use TF tensor as input/output types in func (and other "
          "control flow) ops."),
      llvm::cl::init(false)};

  Option<bool> enable_while_parallel_iterations{
      *this, "enable-while-parallel-iterations",
      llvm::cl::desc("If true, tf.While op will be parallelized. This is "
                     "currently experimental."),
      llvm::cl::init(false)};

  Option<bool> hoist_invariant_ops{
      *this, "hoist-invariant-ops",
      llvm::cl::desc("If true, invariant ops in savedmodels will be hoisted "
                     "out to run during loading."),
      llvm::cl::init(false)};

  Option<uint64_t> cost_threshold{
      *this, "tfrt-cost-threshold",
      llvm::cl::desc(
          "The cost threshold to decide whether a sequence of operations is "
          "cheap, and then whether it can be executed inline."),
      llvm::cl::init(1)};

  Option<int64_t> upper_cost_threshold{
      *this, "tfrt-upper-cost-threshold",
      llvm::cl::desc(
          "The threshold to limit the merging of dependent sequence."),
      llvm::cl::init(-1)};

  Option<bool> merge_inter_dependent_streams{
      *this, "tfrt-merge-inter-dependent-streams",
      llvm::cl::desc("If true, streams with inter data depenedencies will be "
                     "preferred to be merged for inline execution."),
      llvm::cl::init(false)};

  // A set of flags to control auto-fusion: automatic clustering of Tensorflow
  // operations and compiling outlined regions using MLIR based compilation
  // stack.
  //
  // WARNING: These flags are experimental and are intended for manual testing
  // of different auto-fusion strategies. They will be removed in the future.

  ListOption<std::string> auto_fusion_oplist{
      *this, "auto-fusion-oplist",
      llvm::cl::desc("A list of Tensorflow operations to cluster together for "
                     "JIT compilation. Alternatively use 'tier1', ..., 'all' "
                     "to allow clustering for all operations included in the "
                     "given clustering tier.")};

  Option<int> auto_fusion_min_cluster_size{
      *this, "auto-fusion-min-cluster-size",
      llvm::cl::desc("Minimum size of the cluster that should be outlined for "
                     "compilation"),
      llvm::cl::init(2)};
};

// Create a pass that converts MLIR TF dialect to MLIR TFRT dialect.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTfToTfrtConversionPass(const TfrtPipelineOptions& options);

// Creates a pipeline of passes that lowers MLIR TF Executor dialect to TF
// dialect for CoreRT purposes.
void CreateTFExecutorToTFPipeline(mlir::OpPassManager& pm,
                                  const TfrtPipelineOptions& options);

// Creates a pipeline of passes that lowers MLIR TF dialect from tf.function to
// TFRT dialect. SavedModel related conversions are not included.
void CreateTfExecutorToTfrtPipeline(mlir::PassManager& pm,
                                    const TfrtPipelineOptions& options);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_PASSES_H_
