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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {

// Creates a pass that breaks up an island with multiple ops into multiple
// islands, each with a single op.
std::unique_ptr<OperationPass<ModuleOp>> CreateBreakUpIslandsPass();

// Creates a pass that converts mlir functions consisting of mlir ops into a
// tf_executor dialect as a single island.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateFunctionalToExecutorDialectConversionPass();

// Creates a pass that lifts inner ops of tf_executor.island ops in
// tf_executor.graph into the same block as the tf_executor.graph.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateExecutorDialectToFunctionalConversionPass();

namespace TF {
// Creates a pass that canonicalizes legacy compilation and replication
// attributes.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateCanonicalizeCompileAndReplicateAttributesPass();

// Creates a pass that drops `shape_invariant` attribute from While/WhileRegion
// ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDropWhileShapeInvariantPass();

// Creates a pass that drops `shape_invariant` attribute from While/WhileRegion
// ops within device cluster.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDropWhileShapeInvariantInDeviceClusterPass();

// Creates a pass that moves writes to replicate invariant resource variables
// outside tf_device.replicate op.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateHoistReplicateInvariantResourceWritesPass();

// Transforms functional control flow operations in the TensorFlow dialect to
// MLIR Control Flow Graph (CFG) form.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFFunctionalControlFlowToCFG();

// Transforms functional control flow operations in the TensorFlow dialect to
// their region based counterparts.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFFunctionalControlFlowToRegions();

// Transforms region bases control flow operations in the TensorFlow dialect to
// their functional counterparts.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFRegionControlFlowToFunctional();

// Materialize the MlirPassthroughOp by replacing it with the MLIR module
// attached as an attribute.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateMaterializePassthroughOpPass();

// Replicates the TensorList init op by undoing some CSE needed for correct
// shape assignment in shape_inference.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicateTensorListInitOpsPass();

// Performs Shape Inference on the TensorFlow dialect using the global registry.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFShapeInferencePass();

// Performs TF.data optimizations.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFDataOptimizationPass();

std::unique_ptr<OperationPass<func::FuncOp>> CreateMoveTransposesPass();
std::unique_ptr<OperationPass<func::FuncOp>> CreateLayoutAssignmentPass();

// Guarantee that all FuncOp's have a single use.
std::unique_ptr<OperationPass<ModuleOp>> CreateGuaranteeAllFuncsOneUsePass();

// Optional pass which will unroll BatchMatMul and use only MatMul
std::unique_ptr<OperationPass<func::FuncOp>> CreateUnrollBatchMatMulPassPass();

// Optional pass which will map TF BatchMatMul to TF Einsum
std::unique_ptr<OperationPass<func::FuncOp>> CreateBatchMatMulToEinsumPass();

// Pass that transform Einsum to other TF Ops for the supported variants.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTransformEinsumPass();

// Optimizes Tensorflow graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFOptimizePass();
void RegisterTFOptimizePassPipeline();

// Creates pass to rewrite RecvTPUEmbeddingActivationsOp and
// SendTPUEmbeddingGradients ops to internal variants.
std::unique_ptr<OperationPass<func::FuncOp>> CreateRewriteTPUEmbeddingOpsPass();

// Performs specific fusion for GPU targets.
std::unique_ptr<OperationPass<func::FuncOp>> CreateGpuOpFusionPass();

// Creates a pass that decomposes to be compiled ReduceDataset ops into a while
// loop that iterates the dataset and calls the reduction function.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDecomposeReduceDatasetPass();

// Create a pass that convert ops that copy tensors between devices, e.g.
// tf.Identity.
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
CreateTensorDeviceCopyConversionPass();

// Returns a pass that folds tf.BroadcastTo nodes with subsequent nodes if they
// have built in broadcasting support.
std::unique_ptr<OperationPass<func::FuncOp>> CreateBroadcastFoldPass();

void populateTfControlFlowToScfPatterns(MLIRContext* context,
                                        RewritePatternSet* patterns);
// Create a pass to convert TensorFlow control flow to SCF.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTfControlFlowToScfPass();

struct LayoutOptimizationPipelineOptions
    : public PassPipelineOptions<LayoutOptimizationPipelineOptions> {
  Option<std::string> force_data_format{
      *this, "force-data-format",
      llvm::cl::desc("Force data format for all layout sensitive ops")};
  Option<bool> skip_fold_transpose_in_ops{
      *this, "skip-fold-transpose-in-ops",
      llvm::cl::desc("Skip folding transpose operands in Ops which can support "
                     "different layouts.")};
};

// Layout optimization assigns optimal data layout for layout sensitive
// operations, and cancels all redundant transposes.
void CreateLayoutOptimizationPipeline(
    OpPassManager& pm,  // NOLINT - MLIR contract is pass by mutable reference.
    const LayoutOptimizationPipelineOptions& options);

struct StandardPipelineOptions
    : public PassPipelineOptions<StandardPipelineOptions> {
  Option<bool> enable_inliner{*this, "enable-inliner",
                              llvm::cl::desc("Enable inliner."),
                              llvm::cl::init(false)};
  Option<bool> form_clusters{*this, "form-clusters",
                             llvm::cl::desc("Enable Cluster Formation pass."),
                             llvm::cl::init(false)};
};

// Propagates the pass manager with the passes involved in transforming or
// optimizing an MLIR graph without any target specialization.
// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager& pm,
                              const StandardPipelineOptions& options);

// Propagates device attributes of resources from callers to callees.
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceDeviceInferencePass();

// Creates a pass that promotes resource reads/writes in `functions` to inputs
// and outputs of `functions`, assuming that resource operations have already
// been decomposed and function calls have already been inlined. If `functions`
// is empty, the pass is applied to the main function by default. The pass also
// annotates the input arguments for resources with the indices of their
// aliasing output arguments.
std::unique_ptr<OperationPass<ModuleOp>> CreatePromoteResourcesToArgsPass(
    llvm::ArrayRef<std::string> functions = {});

// Creates a pass that promotes tf.VarHandleOp to resource arguments for all
// functions.
std::unique_ptr<OperationPass<ModuleOp>> CreatePromoteVarHandlesToArgsPass();

// Creates a pass that converts readonly reference variables to the
// corresponding resource variables.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertReadonlyReferenceVariablesToResourceVariablesPass();

// Creates a simple device assignment pass on TF dialect for CoreRT use case.
std::unique_ptr<OperationPass<func::FuncOp>> CreateSimpleTFDeviceAssignmentPass(
    llvm::StringRef default_device = "cpu");

// Creates a pass to perform device assignment for TF dialect ops that do not
// have device assignment, by using the device attribute of the function.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFDeviceAssignmentByFuncAttrPass();

// Performs resource lifting on the function body to hoist resource variable
// accesses outside all control flow statements.
LogicalResult ResourceLiftingForFunctionalControlFlow(func::FuncOp function);

// Converts stack ops into operations on local variables, which can later be
// removed by resource lifting. Requires known maximum sizes of stacks and
// known element shapes of push ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateStackOpsDecompositionPass();

// Creates a pass to strip the "tf._noinline" attribute from the functions in
// the module.
std::unique_ptr<OperationPass<ModuleOp>> CreateStripNoinlineAttributePass();

// Converts tensor list operations into operations on buffers and sizes. Needs
// static shapes and known max element count.
std::unique_ptr<OperationPass<ModuleOp>> CreateTensorListOpsDecompositionPass();

// Converts tensor array ops into operations on local variables, which can later
// be removed by resource lifting. Requires known sizes and known element shapes
// (either defined in TensorArrayV3 or implied in the first write).
std::unique_ptr<OperationPass<ModuleOp>>
CreateTensorArrayOpsDecompositionPass();

// Create a pass that legalize HLO to TF dialect.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeHloToTfPass();

// Create a pass that legalize TFG to TF dialect.
std::unique_ptr<Pass> CreateLegalizeTFGToTFEPass();

// Addds the HLO to TF rewrite patterns to the specified pattern list.
void PopulateLegalizeHloToTfPatterns(RewritePatternSet* patterns,
                                     MLIRContext* context);

// Matches sequence of ops to TensorFlow fused kernels. This pass should not be
// generally used beyond exporting to runtimes that supports these ops. In the
// future these fusions may be codegen'd automatically.
std::unique_ptr<OperationPass<func::FuncOp>> CreateFusedKernelMatcherPass();

// Creates function pass to select device index/fold tf.DeviceIndex.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDeviceIndexSelectorPass();

// Creates function pass to replace InitializeTableFromTextFileV2Ops with
// LookupTableImportV2Op ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateInitTextFileToImportPass(
    std::string saved_model_dir = "");

// Creates function pass to cluster TensorFlow ops by host. The program
// generated by this pass will have one function per host where all operations
// in the same function are placed on the same host. Each result of the per-host
// function will have a "tf.device" attribute which specifies the device
// assignment of the result.
std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateClusterTFOpsByHostPass();

// Creates a pass to insert tf_device.send and tf_device.receive ops to make
// sure any argument of any op is on the same host of the op itself.
std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateCrossHostTransferPass();

// Creates a pass that adds the device attribute to every tf.Const op based on
// the device attribute of the operations that read its result. If the result of
// a tf.Const op is read by operations placed on multiple devices, then the pass
// will replicate the tf.Const op once for each device.
std::unique_ptr<OperationPass<ModuleOp>> CreateConstantOpDeviceAssignmentPass();

// Populates the supplied passmanager with the passes required to export
// to TensorFlow Graph.
void AddGraphExportLoweringPassesV2(OpPassManager& pm);

// Populates the supplied passmanager with the passes required to export
// to TensorFlow Graph.
// ***This is the legacy graph export pipeline, prefer
// AddGraphExportLoweringPassesV2***.
void AddGraphExportLoweringPasses(OpPassManager& pm);

// Returns pass that verifies whether all functions in module are of single
// tf_executor.graph and each tf_executor.island in tf_executor.graph only has a
// single op.
std::unique_ptr<OperationPass<ModuleOp>> CreateVerifySuitableForExportPass();

// Returns pass that prepares TPU computation to be legal for export to
// TensorFlow.
std::unique_ptr<OperationPass<ModuleOp>>
CreatePrepareTpuComputationForTfExportPass();

// Rewrites ops that require quantized inputs or outputs to ops that allow
// non-quantized inputs and outputs.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLowerQuantizedPass();

// Reorders ops so ops of the same dialect are next to each other.
std::unique_ptr<Pass> CreateOrderByDialectPass();

// Groups ops into functions that only contain one dialect.
std::unique_ptr<Pass> CreateGroupByDialectPass();

// Removes unused parameters from functions & their callers.
std::unique_ptr<OperationPass<ModuleOp>> CreateRemoveUnusedArgumentsPass();

// Removes unused results from WhileRegion ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateRemoveUnusedWhileResultsPass();

// Hoists loop invariant ops to the outside of the loop.
std::unique_ptr<OperationPass<func::FuncOp>> CreateHoistLoopInvariantPass();

// Creates VarHandleOps right next to the operations that use them.
std::unique_ptr<OperationPass<ModuleOp>> CreateLocalizeVarHandlesPass();

// Removes all TF attributes
std::unique_ptr<OperationPass<ModuleOp>> CreateStripTfAttributesPass();

// Converts AnonymousIteratorOps to (named) IteratorOps.
std::unique_ptr<OperationPass<ModuleOp>> CreateNameAnonymousIteratorsPass();

// Creates a pass that breaks up an island with multiple ops into multiple
// islands, each with a single op. This pass intentionally does not propagate
// control dependencies across newly created islands, a following pass will
// handle this.
// TODO(b/244596254) Implement followup pass for creating control deps.
std::unique_ptr<OperationPass<func::FuncOp>> CreateSplitIntoIslandPerOpPass();

// Populates the supplied passmanager with the passes required to run the
// CPU/GPU bridge.
void CreateTFXLABridgePipeline(OpPassManager& pm);

}  // namespace TF

namespace tf_executor {

// Creates a pass to chain control outputs of while loop body.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorConvertControlToDataOutputsPass();

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorCheckControlDependenciesPass();

// Creates a pass to merge IslandOps from TFExecutor dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFExecutorIslandCoarseningPass();

// Creates a pass to merge IslandOps for operation marked for execution on TPU.
// This is a V1 backward compatibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandCoarseningPass();

// Creates a pass to outlining TPU clusters from single IslandOp into a nested
// module suitable for being processed as-if it was a V2 module.
// This is a V1 backward compatibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandOutliningPass();

// Creates a pass to inline calls to the nested TPU module, this reverses the
// effect of the `TFExecutorTPUV1IslandOutlining` pass above.
// This is a V1 backward compatibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandInliningPass();

// Creates a pass to prune tf_executor.graph from dead nodes.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFExecutorGraphPruningPass(
    llvm::ArrayRef<std::string> ops_to_preserve = {});

// Creates a pass to update control dependencies.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorUpdateControlDependenciesPass();

}  // namespace tf_executor

namespace TFDevice {
// Creates a pass that forms clusters from instructions that are assigned to
// same device.
std::unique_ptr<OperationPass<ModuleOp>> CreateClusterFormationPass();

// Sinks `tf.Const` operations in the ClusterOp region using them. This is
// performed in order to limit the number of values implicitly captured in this
// region before outlining.
std::unique_ptr<OperationPass<func::FuncOp>> CreateClusterConstantSinkingPass(
    llvm::function_ref<bool(tf_device::ClusterOp, ElementsAttr)> filter = {});

// Creates a pass that outlines regions of tf_device.cluster operations.
std::unique_ptr<OperationPass<ModuleOp>> CreateClusterOutliningPass();

// Creates a pass that outlines regions of tf_device.launch operations.
std::unique_ptr<OperationPass<ModuleOp>> CreateLaunchOutliningPass();

// Creates a pass that converts tf_device::LaunchFuncOp into
// TF::PartitionedCallOp.
std::unique_ptr<OperationPass<ModuleOp>> CreateConvertLaunchFuncToTFCallPass();

// A pass that decomposes composite resource operations into primitive ones like
// ReadVariableOp, AssignVariableOp and other computations to facilitate
// transformations like resource op lifting.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDecomposeResourceOpsPass();

// A pass that decomposes composite resource operations in device cluster
// (tf_device.cluster op) into primitive ones like ReadVariableOp,
// AssignVariableOp and other computations to facilitate transformations like
// resource op lifting.
std::unique_ptr<OperationPass<ModuleOp>>
CreateDecomposeResourceOpsInClusterPass();

// Creates a pass that marks TPU cluster input-output pairs reading and writing
// to same resource variable as aliases.
std::unique_ptr<OperationPass<ModuleOp>> CreateMarkInputOutputAliasesPass();

// Creates a pass that lifts operations on external resource variables from
// device computation nested in `tf_device::LaunchOp` out so that resource
// variable load operations are all before device computation while resource
// variable store operations are all after device computation. After this pass,
// device computation no longer interacts with external resource variables.
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceOpLiftingPass();

// Creates a pass that lifts operations from the main function.
std::unique_ptr<OperationPass<ModuleOp>>
CreateResourceOpLiftingForMainFunctionPass();

// Lifts resource operations from tf_device.launch_func ops nested in `op`
// outside. Returns a failure if there are remaining resource-type values that
// can not be lifted.
LogicalResult LiftResourceOps(Operation* op);

// Creates a pass that hoists invariant operations in a `tf_device.replicate`.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicateInvariantOpHoistingPass();

// Creates a pass that forms replica `tf_executor.island` from a single
// `tf_device.replicate` island.
std::unique_ptr<OperationPass<func::FuncOp>> CreateReplicateToIslandPass(
    bool legacy_graph_export = true);

// Creates a pass that sets the device ordinal attribute of the required op
// using the replica id attribute.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplicaIDToDeviceOrdinalPass();

// Creates a pass that creates `tf_executor.island` from a single
// `tf_device.parallel_execute` island.
std::unique_ptr<OperationPass<func::FuncOp>> CreateParallelExecuteToIslandsPass(
    bool legacy_graph_export = true);

// Creates a pass that annotates whether a LaunchFuncOp's parameters have the
// same data across replicas.
std::unique_ptr<OperationPass<ModuleOp>>
CreateAnnotateParameterReplicationPass();

// Creates a pass that marks unsupported ops in device cluster for outside
// compilation.
std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOpsForOutsideCompilationPass();

// Creates a pass that extracts outside compilation (Host ops inside device
// cluster) at head/tail of Device cluster to run before/after XLA computation.
std::unique_ptr<OperationPass<ModuleOp>>
CreateExtractHeadTailOutsideCompilationPass();

// Creates a pass that extract outside compilation (Host ops inside cevice
// cluster) ops to a separate parallel_execute region to run on CPU.
std::unique_ptr<OperationPass<ModuleOp>> CreateExtractOutsideCompilationPass();

// Creates a pass that merges control flow with similar predicates.
std::unique_ptr<OperationPass<ModuleOp>> CreateMergeControlFlowPass();

// Creates a pass that wraps each TensorFlow dialect with `device` attribute
// in a `tf_device.launch` op with the same `device` attribute.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDeviceAttributeToLaunchPass();

// Creates a pass that hoists a `tf_device.launch` body and assigns a `device`
// attribute to each TensorFlow dialect op in the body based on the `device`
// attribute on the `tf_device.launch`.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLaunchToDeviceAttributePass(
    bool legacy_graph_export = true);

// Creates a pass that extracts ops in tf_device.launch op with host device
// assignment and adds an `_xla_outside_compilation` attribute value.
std::unique_ptr<OperationPass<ModuleOp>>
CreateHostLaunchToOutsideCompiledPass();

// Create a pass that encapsulates StatefulPartitionedCallOp within a cluster.
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaClusterFormationPass();

// Create a pass that inlines the StatefulPartitionedCallOp op based in the
// parent region.
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaInlineDeviceOpsPass();

// Creates a pass that rewrites partitioned calls with `_xla_compile_device
// type` with `tf.XlaLaunch` ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateXlaRewritePass();
}  // namespace TFDevice

namespace TFTPU {
// Creates a pass that converts unified compilation and replication
// attributes back to legacy attributes.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertToLegacyCompileAndReplicateAttributesPass();

// Creates a pass that converts all TPUPartitionedInput to TPUPartitionedInputV2
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUPartitionedOpConversionPass();

// Creates a pass that forms clusters from operations of the same
// `_replication_info` attribute.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUClusterFormationPass();

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUValidateInputsPass();

// Creates a pass that cleans up `_replication_info` attribute on operations
// that are inside a cluster.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUClusterCleanupAttributesPass();

// Creates a pass that removes Identity/IdentityN ops from a cluster.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUIdentityPruningPass();

// Creates a pass that allows TPU program inputs to have layouts determined at
// run time.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUDynamicLayoutPass();

// Creates a pass that adds `tf.ReadVariableOp` to a TPU cluster for resources
// the cluster only writes to.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUResourceReadForWritePass();

// Creates a pass that reorders partitiioned resource reads and replicated
// inputs.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUReorderReplicateAndPartitionedInputsPass();

// Creates a pass that partitions unpartitioned resource read/write to
// partitioned resource variables.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUResourceReadsWritesPartitioningPass();

// Creates a pass that looks for usage of the result of
// TPUCopyWithDynamicShapeOp and annotate these values to be dynamic shape. This
// ensures that the generated tpu program has the correct inputs annotation.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUAnnotateDynamicShapeInputsPass();

// Creates a pass that rewrites `tf_device.launch_func` on TPUs into TPU runtime
// ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPURewritePass();

// Creates a pass that identifies XLASharding ops in launch op for TPU
// computation.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUShardingIdentificationPass();

// Creates a pass that moves `tf.AssignVariableOp` into a
// `tf_device.parallel_execute` region if the `tf.AssignVariableOp` is the
// only consumer of a `tf_device.parallel_execute` result.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUParallelExecuteSinkResourceWritePass();

// Creates a pass that merges device variable reads/updates into the surrounded
// TPUExecute node. This allows the execute node to perform in-place variable
// updates.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUMergeVariablesWithExecutePass();

// Create a pass that extract TPUCopyWithDynamicShapeOp from the host launch op
// and wrap them in device launch op. This allows this op executed on TPU while
// still compiled on host.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateExtractTPUCopyWithDynamicShapeOpPass();

// Creates a pass that wraps ReadVariableOp/AssignVariable op that consumes a
// packed tensor to have same device placement as underlying TPU device.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUColocateCompositeResourceOps();

// Creates a pass that adds ops which perform formatting on variables at
// run-time according to compilation result.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUVariableRuntimeReformattingPass();

// Creates a pass that wraps ops with the same `_xla_outside_compilation`
// attribute value in a tf_device.launch op with host device assignment.
std::unique_ptr<OperationPass<ModuleOp>>
CreateOutsideCompiledToHostLaunchPass();

// Creates a pass that expands outside compilation cluster at the head/tail of
// TPU computation by adding outside compilation attribute to identity/cast ops
// that are only used for host computation.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUHostComputationExpansionPass();

// Creates a pass that updates inputs to TPU embedding layer enqueue ops so that
// correct ops are invoked during training and evaluation.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUUpdateEmbeddingEnqueueOpInputsPass();

// Creates a pass that propagates TPU devices to users.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTPUDevicePropagationPass();

// Populates the supplied passmanager with the passes required to run the
// bridge.
void CreateTPUBridgePipeline(OpPassManager& pm);

// Populates the supplied passmanager with the passes required to run the
// bridge in V1 mode.
void CreateTPUBridgePipelineV1(OpPassManager& pm);

// Creates a pass that replicates the tf._TPUCompileMlir op on each host that
// needs the compiled program. It helps avoid transferring the compiled binary
// between hosts.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
CreateTPUCompileOpReplicationPass();

// Creates a pass that applies space to depth transform
// for the first or frontier convolutions consume host inputs on TPU.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUSpaceToDepthPass();

}  // namespace TFTPU

// Define the registrations in a detail namespace, just so that we can overload
// the main entry point `registerTensorFlowPasses` to inject
// RegisterTFOptimizePassPipeline.
namespace detail {

// Direction in which to move transposes in MoveTransposePass.
enum MoveTransposeDirection { kBegin, kEnd };

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_BATCHMATMULTOEINSUMPASS
#define GEN_PASS_DECL_BREAKUPISLANDSPASS
#define GEN_PASS_DECL_BROADCASTFOLDPASS
#define GEN_PASS_DECL_CANONICALIZECOMPILEANDREPLICATEATTRIBUTESPASS
#define GEN_PASS_DECL_CLUSTERCONSTANTSINKINGPASS
#define GEN_PASS_DECL_CLUSTERFORMATIONPASS
#define GEN_PASS_DECL_CLUSTEROUTLININGPASS
#define GEN_PASS_DECL_CLUSTERTFOPSBYHOSTPASS
#define GEN_PASS_DECL_CONSTANTOPDEVICEASSIGNMENTPASS
#define GEN_PASS_DECL_CONVERTLAUNCHFUNCTOTFCALLPASS
#define GEN_PASS_DECL_CONVERTREADONLYREFERENCEVARIABLESTORESOURCEVARIABLESPASS
#define GEN_PASS_DECL_CONVERTTFCONTROLFLOWTOSCFPASS
#define GEN_PASS_DECL_CONVERTTOLEGACYCOMPILEANDREPLICATEATTRIBUTESPASS
#define GEN_PASS_DECL_DECOMPOSEREDUCEDATASETPASS
#define GEN_PASS_DECL_DEVICEINDEXSELECTORPASS
#define GEN_PASS_DECL_DROPWHILESHAPEINVARIANTINDEVICECLUSTERPASS
#define GEN_PASS_DECL_DROPWHILESHAPEINVARIANTPASS
#define GEN_PASS_DECL_EXECUTORCHECKCONTROLDEPENDENCIESPASS
#define GEN_PASS_DECL_EXECUTORCONVERTCONTROLTODATAOUTPUTSPASS
#define GEN_PASS_DECL_EXECUTORDIALECTTOFUNCTIONALPASS
#define GEN_PASS_DECL_EXECUTORGRAPHPRUNINGPASS
#define GEN_PASS_DECL_EXECUTORISLANDCOARSENINGPASS
#define GEN_PASS_DECL_EXECUTORTPUV1ISLANDINLININGPASS
#define GEN_PASS_DECL_EXECUTORUPDATECONTROLDEPENDENCIESPASS
#define GEN_PASS_DECL_FUNCTIONALCONTROLFLOWTOCFGPASS
#define GEN_PASS_DECL_FUNCTIONALCONTROLFLOWTOREGIONSPASS
#define GEN_PASS_DECL_FUNCTIONALTOEXECUTORDIALECTCONVERSIONPASS
#define GEN_PASS_DECL_FUSEDKERNELMATCHERPASS
#define GEN_PASS_DECL_GROUPBYDIALECTPASS
#define GEN_PASS_DECL_GUARANTEEALLFUNCSONEUSEPASS
#define GEN_PASS_DECL_HOISTREPLICATEINVARIANTRESOURCEWRITESPASS
#define GEN_PASS_DECL_INITTEXTFILETOIMPORTPASS
#define GEN_PASS_DECL_LAUNCHOUTLININGPASS
#define GEN_PASS_DECL_LAYOUTASSIGNMENTPASS
#define GEN_PASS_DECL_LEGALIZEHLOTOTFPASS
#define GEN_PASS_DECL_LEGALIZETFGTOTFPASS
#define GEN_PASS_DECL_LOCALIZEVARHANDLESPASS
#define GEN_PASS_DECL_LOWERQUANTIZEDPASS
#define GEN_PASS_DECL_MARKINPUTOUTPUTALIASESPASS
#define GEN_PASS_DECL_MARKOPSFOROUTSIDECOMPILATIONPASS
#define GEN_PASS_DECL_MATERIALIZEPASSTHROUGHOP
#define GEN_PASS_DECL_MERGECONTROLFLOWPASS
#define GEN_PASS_DECL_MOVETRANSPOSESPASS
#define GEN_PASS_DECL_ORDERBYDIALECTPASS
#define GEN_PASS_DECL_OUTSIDECOMPILEDTOHOSTLAUNCHPASS
#define GEN_PASS_DECL_PARALLELEXECUTETOISLANDSPASS
#define GEN_PASS_DECL_PREPARETPUCOMPUTATIONFORTFEXPORTPASS
#define GEN_PASS_DECL_PROMOTERESOURCESTOARGSPASS
#define GEN_PASS_DECL_PROMOTEVARHANDLESTOARGSPASS
#define GEN_PASS_DECL_REGIONCONTROLFLOWTOFUNCTIONALPASS
#define GEN_PASS_DECL_REMOVEUNUSEDARGUMENTSPASS
#define GEN_PASS_DECL_REMOVEUNUSEDWHILERESULTSPASS
#define GEN_PASS_DECL_REPLICAIDTODEVICEORDINALPASS
#define GEN_PASS_DECL_REPLICATEINVARIANTOPHOISTINGPASS
#define GEN_PASS_DECL_REPLICATETOISLANDPASS
#define GEN_PASS_DECL_RESOURCEDEVICEINFERENCEPASS
#define GEN_PASS_DECL_REWRITETPUEMBEDDINGOPSPASS
#define GEN_PASS_DECL_SIMPLETFDEVICEASSIGNMENTPASS
#define GEN_PASS_DECL_SPLITINTOISLANDPEROPPASS
#define GEN_PASS_DECL_STACKOPSDECOMPOSITIONPASS
#define GEN_PASS_DECL_STRIPNOINLINEATTRIBUTEPASS
#define GEN_PASS_DECL_TFDATAOPTIMIZATIONPASS
#define GEN_PASS_DECL_TFDEVICEASSIGNMENTBYFUNCATTRPASS
#define GEN_PASS_DECL_TPUBRIDGEEXECUTORISLANDOUTLININGPASS
#define GEN_PASS_DECL_TPUCLEANUPCLUSTERATTRIBUTESPASS
#define GEN_PASS_DECL_TPUCLUSTERFORMATIONPASS
#define GEN_PASS_DECL_TPUCOLOCATECOMPOSITERESOURCEOPSPASS
#define GEN_PASS_DECL_TPUDEVICEPROPAGATIONPASS
#define GEN_PASS_DECL_TPUDYNAMICLAYOUTPASS
#define GEN_PASS_DECL_TPUEXTRACTHEADTAILOUTSIDECOMPILATIONPASS
#define GEN_PASS_DECL_TPUEXTRACTOUTSIDECOMPILATIONPASS
#define GEN_PASS_DECL_TPUHOSTCOMPUTATIONEXPANSIONPASS
#define GEN_PASS_DECL_TPUIDENTITYPRUNINGPASS
#define GEN_PASS_DECL_TPUMERGEVARIABLESWITHEXECUTEPASS
#define GEN_PASS_DECL_EXTRACTTPUCOPYWITHDYNAMICSHAPEOPPASS
#define GEN_PASS_DECL_TPUPARALLELEXECUTESINKRESOURCEWRITEPASS
#define GEN_PASS_DECL_TPUREORDERREPLICATEANDPARTITIONEDINPUTSPASS
#define GEN_PASS_DECL_TPURESOURCEREADFORWRITEPASS
#define GEN_PASS_DECL_TPURESOURCEREADSWRITESPARTITIONINGPASS
#define GEN_PASS_DECL_TPUREWRITEPASS
#define GEN_PASS_DECL_TPUSHARDINGIDENTIFICATIONPASS
#define GEN_PASS_DECL_TPUSPACETODEPTHPASS
#define GEN_PASS_DECL_TPUUPDATEEMBEDDINGENQUEUEOPINPUTSPASS
#define GEN_PASS_DECL_TPUVALIDATEINPUTSPASS
#define GEN_PASS_DECL_TPUVARIABLERUNTIMEREFORMATTINGPASS
#define GEN_PASS_DECL_TENSORARRAYOPSDECOMPOSITIONPASS
#define GEN_PASS_DECL_TENSORDEVICECOPYCONVERSIONPASS
#define GEN_PASS_DECL_TENSORFLOWOPTIMIZEPASS
#define GEN_PASS_DECL_TENSORFLOWSHAPEINFERENCEPASS
#define GEN_PASS_DECL_TENSORLISTOPSDECOMPOSITIONPASS
#define GEN_PASS_DECL_TENSORFLOWGPUFUSION
#define GEN_PASS_DECL_TPUV1BRIDGEEXECUTORISLANDCOARSENINGPASS
#define GEN_PASS_DECL_TRANSFORMEINSUMPASS
#define GEN_PASS_DECL_UNROLLBATCHMATMULPASS
#define GEN_PASS_DECL_VERIFYSUITABLEFOREXPORTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"
}  // namespace detail
using namespace detail;  // NOLINT
inline void registerTensorFlowPasses() {
  detail::registerTensorFlowPasses();
  TF::RegisterTFOptimizePassPipeline();
}

namespace TFDevice {
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_ANNOTATEPARAMETERREPLICATIONPASS
#define GEN_PASS_DECL_DECOMPOSERESOURCEOPSINCLUSTERPASS
#define GEN_PASS_DECL_DECOMPOSERESOURCEOPSPASS
#define GEN_PASS_DECL_DEVICEATTRIBUTETOLAUNCHPASS
#define GEN_PASS_DECL_HOSTLAUNCHTOOUTSIDECOMPILEDPASS
#define GEN_PASS_DECL_LAUNCHTODEVICEATTRIBUTEPASS
#define GEN_PASS_DECL_RESOURCEOPLIFTINGFORMAINFUNCTIONPASS
#define GEN_PASS_DECL_RESOURCEOPLIFTINGPASS
#define GEN_PASS_DECL_XLACLUSTERFORMATIONPASS
#define GEN_PASS_DECL_XLAINLINEDEVICEOPSPASS
#define GEN_PASS_DECL_XLAREWRITEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"
}  // namespace TFDevice

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
