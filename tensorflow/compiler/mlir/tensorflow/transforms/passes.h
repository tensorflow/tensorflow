/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // TF:local_config_mlir

namespace mlir {

// Creates a pass that breaks up an island with multiple ops into multiple
// islands, each with a single op.
std::unique_ptr<OpPassBase<FuncOp>> CreateBreakUpIslandsPass();

// Creates a pass that converts mlir functions consisting of mlir ops into a
// tf_executor dialect as a single island.
std::unique_ptr<OpPassBase<FuncOp>>
CreateFunctionalToExecutorDialectConversionPass();

namespace TF {
// Transforms functional control flow operations in the standard TensorFlow
// dialect to MLIR Control Flow Graph (CFG) form.
std::unique_ptr<OpPassBase<FuncOp>> CreateTFFunctionalControlFlowToCFG();

// Materialize the MlirPassthroughOp by replacing it with the MLIR module
// attached as an attribute.
std::unique_ptr<OpPassBase<FuncOp>> CreateMaterializePassthroughOpPass();

// Performs Shape Inference on the TensorFlow dialect using the global registry.
std::unique_ptr<OpPassBase<ModuleOp>> CreateTFShapeInferencePass();

// Optimizes Tensorflow graph.
std::unique_ptr<OpPassBase<FuncOp>> CreateTFOptimizePass();

// Propagates the pass manager with the passes involved in transforming or
// optimizing an MLIR graph without any target specialization.
// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager& pm);
}  // namespace TF

namespace TFControlFlow {
// Raises from the "TensorFlow Control Flow" dialect to the standard TensorFlow
// dialect.
std::unique_ptr<OpPassBase<FuncOp>> CreateRaiseTFControlFlowPass();

}  // namespace TFControlFlow

namespace tf_executor {
class GraphOp;

// Returns a pass that folds switch nodes with constant predicates.
std::unique_ptr<OpPassBase<FuncOp>> CreateSwitchFoldPass();

// Create a pass to merge IslandOps from TFExecutor dialect.
std::unique_ptr<OpPassBase<FuncOp>> CreateTFExecutorIslandCoarseningPass();

// Create a pass to prune tf_executor.graph from dead nodes.
std::unique_ptr<OpPassBase<FuncOp>> CreateTFExecutorGraphPruningPass();

// Prunes unreachable operations of a tf_executor.graph operation.
void PruneGraph(GraphOp graph);

// Sink `tf.Const` operations in the LaunchOp region using them. This is
// performed in order to limit the number of values implicitly captured in this
// region before outlining.
std::unique_ptr<OpPassBase<FuncOp>> CreateTFExecutorConstantSinkingPass();

}  // namespace tf_executor

namespace TFDevice {
// Creates a pass that forms clusters from instructions that are assigned to
// same device.
std::unique_ptr<OpPassBase<FuncOp>> CreateClusterFormationPass();

// Creates a pass that outlines regions of tf_device.launch operations.
std::unique_ptr<OpPassBase<ModuleOp>> CreateClusterOutliningPass();

// Creates a pass that lifts operations on external resource variables from
// device computation nested in `tf_device::LaunchOp` out so that resource
// variable load operations are all before device computation while resource
// variable store operations are all after device computation. After this pass,
// device computation no longer interacts with external resource variables.
std::unique_ptr<OpPassBase<FuncOp>> CreateResourceOpLiftingPass();

// Lifts resource variable operations from tf_device.launch_func ops nested in
// `op`.
void LiftResourceOps(Operation* op);

// Creates a pass that hoists invariant operations in a `tf_device.replicate`.
std::unique_ptr<OpPassBase<FuncOp>> CreateReplicateInvariantOpHoistingPass();

// Creates a pass that forms replica `tf_executor.island` from a single
// `tf_device.replicate` island.
std::unique_ptr<OpPassBase<FuncOp>> CreateReplicateToIslandPass();

}  // namespace TFDevice

namespace TFTPU {
// Creates a pass that forms clusters from operations of the same
// `_tpu_replicate` attribute.
std::unique_ptr<OpPassBase<FuncOp>> CreateTPUClusterFormationPass();

// Creates a pass that rewrites `tf_device.launch_func` on TPUs into TPU runtime
// ops.
std::unique_ptr<OpPassBase<ModuleOp>> CreateTPURewritePass();

// Creates a pass that merges device variable reads/updates into the surrounded
// TPUExecute node. This allows the execute node to perform in-place variable
// updates.
std::unique_ptr<OpPassBase<FuncOp>> CreateTPUMergeVariablesWithExecutePass();

// Populates the supplied passmanager with the passes required to run the
// bridge. NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTPUBridge(OpPassManager& pm);

}  // namespace TFTPU

namespace tf_saved_model {

// Creates a pass that uses tf_saved_model dialect linkage information
// to delete unused func's.
std::unique_ptr<OpPassBase<ModuleOp>> CreateDeleteUnusedFuncsPass();

// Creates a pass that optimizes tf_saved_model.global_tensor ops.
std::unique_ptr<OpPassBase<ModuleOp>> CreateOptimizeGlobalTensorsPass();

// Creates a pass that inlines global tensors as tf.Const ops in the function
// body.
std::unique_ptr<OpPassBase<ModuleOp>> CreateInlineGlobalTensorsPass();

}  // namespace tf_saved_model

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
