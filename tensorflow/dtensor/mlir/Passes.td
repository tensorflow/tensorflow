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

#ifndef TF_DTENSOR_PASSES
#define TF_DTENSOR_PASSES

include "mlir/Pass/PassBase.td"

def DTensorOpToDeviceCluster
    : Pass<"dtensor-op-to-device-cluster", "mlir::func::FuncOp"> {
  let summary = "Creates and wraps tf_device.cluster op for all TF ops";
  let constructor = "CreateDTensorOpToDeviceClusterPass()";
  let dependentDialects = [
  ];
}

def DTensorDeviceMeshClusterCoarsening
    : Pass<"dtensor-device-mesh-cluster-coarsening", "mlir::func::FuncOp"> {
  let summary = "Merges tf_device.cluster op with same mesh attribute.";
  let constructor = "CreateDTensorDeviceMeshClusterCoarsening()";
  let dependentDialects = [
  ];
}

def DTensorConstantFolding
    : Pass<"dtensor-constant-folding", "mlir::func::FuncOp"> {
  let summary = "Folds constants operations.";
  let constructor = "CreateDTensorConstantFolding()";
  let dependentDialects = [
  ];
}

def DTensorAllReduceSumOptimization
    : Pass<"dtensor-allreduce-sum-optimization", "mlir::func::FuncOp"> {
  let summary = "Changes order of add/allreduce to minimize all reduce operations.";
  let constructor = "CreateDTensorAllReduceSumOptimization()";
  let dependentDialects = [
  ];
}

def DTensorAllReduceScatterOptimization
    : Pass<"dtensor-allreduce-scatter-optimization", "mlir::func::FuncOp"> {
  let summary = "Combines allreduce and scatter to reducescatter.";
  let constructor = "CreateDTensorAllReduceScatterOptimization()";
  let dependentDialects = [
  ];
}

def DTensorAllReduceCombineOptimization
    : Pass<"dtensor-allreduce-combine-optimization", "mlir::func::FuncOp"> {
  let summary = "Combine independent all reduce operations.";
  let constructor = "CreateDTensorAllReduceCombineOptimization()";
  let dependentDialects = [
  ];
}

def DTensorMixedPrecisionReduce
    : Pass<"dtensor-mixed-precision-reduce", "mlir::func::FuncOp"> {
  let summary = "Upcast tensors to higher precision type for reduction ops.";
  let constructor = "CreateDTensorMixedPrecisionReducePass()";
  let dependentDialects = [
  ];
}

def DTensorDCE
    : Pass<"dtensor-dce", "mlir::func::FuncOp"> {
  let summary = "Removes unused ops from graph.";
  let constructor = "CreateDTensorDCE()";
  let dependentDialects = [
  ];
}

def DTensorUndoMergeConstAcrossMesh
    : Pass<"dtensor-undo-merge-const-across-mesh", "mlir::func::FuncOp"> {
  let summary = "Undo constant merging across meshes";
  let constructor = "CreateDTensorUndoMergeConstAcrossMesh()";
  let dependentDialects = [
  ];
}

def DTensorElideIdentityBeforeCopyToMesh
    : Pass<"dtensor-elide-identity-before-copy-to-mesh", "mlir::func::FuncOp"> {
  let summary = "Elide IdentityOp right before CopyToMesh style Ops";
  let constructor = "CreateDTensorElideIdentityBeforeCopyToMesh()";
  let dependentDialects = [
  ];
}

def DTensorSetDefaultSharding
    : Pass<"dtensor-set-default-sharding", "mlir::func::FuncOp"> {
  let summary = "Sets default sharding of TPU computation inputs/outputs to maximal.";
  let constructor = "CreateDTensorSetDefaultSharding()";
  let dependentDialects = [
  ];
}

def DTensorDesignateResourceHandleMesh
    : Pass<"dtensor-designate-resource-handle-mesh", "mlir::func::FuncOp"> {
  let summary = "Sets empty mesh attributes for device cluster that creates or destroys resource handles.";
  let constructor = "CreateDTensorDesignateResourceHandleMesh()";
  let dependentDialects = [
  ];
}

def DTensorPropagateDefaultLayout
    : Pass<"dtensor-propagate-default-layout", "mlir::func::FuncOp"> {
  let summary = "Converts layout attributes added by end users to DTensorLayout op.";
  let constructor = "CreateDTensorPropagateDefaultLayout()";
  let dependentDialects = [
  ];
}

def DTensorHandleCrossClusterDependencies
    : Pass<"dtensor-handle_cross_cluster_dependences", "mlir::ModuleOp"> {
  let summary = "Lowers cross mesh cluster data depedences as send/recvs.";
  let constructor = "CreateDTensorHandleCrossClusterDependencies()";
  let dependentDialects = [
  ];
}

def DTensorAnnotateGlobalShape
    : Pass<"dtensor-annotate-global-shape", "mlir::ModuleOp"> {
  let summary = "Mark all operations and function arguments with `_global_shape` attribute to be used during SPMD expansion.";
  let constructor = "CreateDTensorAnnotateGlobalShape()";
  let dependentDialects = [
  ];
}

def DTensorLayoutPropagationV2
    : Pass<"dtensor-layout-propagation-v2", "mlir::ModuleOp"> {
  let summary = "Propagates layout information for all the TF Ops.";
  let constructor = "CreateDTensorLayoutPropagationPassV2()";
  let dependentDialects = [
  ];
}

def DTensorMeshPropagation
    : Pass<"dtensor-mesh-propagation", "mlir::ModuleOp"> {
  let summary = "Propagates mesh information to all clusters.";
  let constructor = "CreateDTensorMeshPropagationPass()";
  let dependentDialects = [
  ];
}

def DTensorSPMDExpansion
    : Pass<"dtensor-spmd-expansion", "mlir::ModuleOp"> {
  let summary = "Converts ops into SPMD expanded form.";
  let constructor = "CreateDTensorSPMDExpansion()";
  let dependentDialects = [
  ];
}

def DTensorAllReduceLowering
    : Pass<"dtensor-all-reduce-lowering", "mlir::ModuleOp"> {
  let summary = "Converts logical AllReduce ops into physical AllReduce ops.";
  let constructor = "CreateDTensorAllReduceLoweringPass()";
  let dependentDialects = [
  ];
}

def DTensorReduceScatterLowering
    : Pass<"dtensor-reduce-scatter-lowering", "mlir::ModuleOp"> {
  let summary = "Converts logical ReduceScatter ops into physical ReduceScatter ops.";
  let constructor = "CreateDTensorReduceScatterLoweringPass()";
  let dependentDialects = [
  ];
}

def DTensorAllGatherLowering
    : Pass<"dtensor-all-gather-lowering", "mlir::ModuleOp"> {
  let summary = "Converts logical AllGather ops into physical AllGather ops.";
  let constructor = "CreateDTensorAllGatherLoweringPass()";
  let dependentDialects = [
  ];
}

def DTensorAllScatterLowering
    : Pass<"dtensor-all-scatter-lowering", "mlir::ModuleOp"> {
  let summary = "Converts logical AllScatter ops into physical Split ops.";
  let constructor = "CreateDTensorAllScatterLoweringPass()";
  let dependentDialects = [
  ];
}

def DTensorClusterFunctionConversion
    : Pass<"dtensor-cluster-function-conversion", "mlir::ModuleOp"> {
  let summary = "Converts tf_device.cluster_func ops into TF StatefulPartitioned call op with mesh attribute.";
  let constructor = "CreateDTensorClusterFunctionConversion()";
  let dependentDialects = [
  ];
}

def DTensorPropagateDeviceIdToFunctionArgs
    : Pass<"dtensor-propagate-device-id-to-function-args", "mlir::ModuleOp"> {
  let summary = "Adds device id as arguments to all private function in graph.";
  let constructor = "CreateDTensorPropagateDeviceIdToFunctionArgs()";
  let dependentDialects = [
  ];
}

def DTensorTPUIntegration
    : Pass<"dtensor-tpu-integration", "mlir::ModuleOp"> {
  let summary = "Adds TPUReplicateMetadata and converts ops that run on TPU's to a single tf_device.cluster to be compatible with following TF2XLA MLIR passes.";
  let constructor = "CreateDTensorTPUIntegration()";
  let dependentDialects = [
  ];
}

def DTensorTpuAddResourceDeviceAttribute
    : Pass<"dtensor-tpu-add-resource-device-attribute", "mlir::ModuleOp"> {
  let summary = "Adds placeholder device attributes to resources accessed by TPU computation to enable buffer aliasing.";
  let constructor = "CreateDTensorTpuAddResourceDeviceAttribute()";
  let dependentDialects = [
  ];
}

def DTensorUpdateTPUMetadata
    : Pass<"dtensor-update-tpu-metadata", "mlir::ModuleOp"> {
  let summary = "Changes metadata on TPU specific ops such as device placement.";
  let constructor = "CreateDTensorUpdateTPUMetadata()";
  let dependentDialects = [
  ];
}

def DTensorEmbedding
    : Pass<"dtensor-embedding", "mlir::ModuleOp"> {
  let summary = "Extracts the ApplyEmbeddingOptimizer and checks if the attached optimizer is usable for TPU embeddings.";
  let constructor = "CreateDTensorEmbeddingPass()";
  let dependentDialects = [
    "::mlir::chlo::ChloDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::mhlo::MhloDialect",
    "::mlir::shape::ShapeDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::sparse_tensor::SparseTensorDialect",
    "::mlir::quant::QuantizationDialect",
  ];
}

def DTensorEmbeddingV2
    : Pass<"dtensor-embedding-v2", "mlir::ModuleOp"> {
  let summary = "Embedding Pass";
  let constructor = "CreateDTensorEmbeddingPassV2()";
  let dependentDialects = [
  ];
}

def DTensorEmbeddingCheckpoint
    : Pass<"dtensor-embedding-checkpoint", "mlir::ModuleOp"> {
  let summary = "Checkpoint pass to save or load TPU embeddings variables.";
  let constructor = "CreateDTensorEmbeddingCheckpointPass()";
  let dependentDialects = [
  ];
}

def DTensorFunctionRenaming
    : Pass<"dtensor-function-renaming", "mlir::ModuleOp"> {
  let summary = "Renames private functions by appending an id to each name. This is used to make private function names unique across modules.";
  let constructor = "CreateFunctionRenamingPass()";
  let dependentDialects = [
  ];
}

def DTensorMergeClusters
    : Pass<"dtensor-merge-clusters", "mlir::ModuleOp"> {
  let summary = "Merges tf_device.Clusters ops with same mesh specification.";
  let constructor = "CreateDTensorMergeClustersPass()";
  let dependentDialects = [
  ];
}

def DTensorLowerSendRecv
    : Pass<"dtensor-lower-send-recv", "mlir::ModuleOp"> {
  let summary = "Lowers DTensorSend/DTensorRecv ops to send/recv ops.";
  let constructor = "CreateDTensorLowerSendRecv()";
  let dependentDialects = [
  ];
}

def DTensorMoveCompilationToHost
    : Pass<"dtensor-move-compilation-to-host", "mlir::ModuleOp"> {
  let summary = "Moves XLA compilation ops to host computation.";
  let constructor = "CreateDTensorMoveCompilationToHost()";
  let dependentDialects = [
  ];
}

def DTensorSparseTensorToDenseTensor
    : Pass<"dtensor-sparse-tensor-to-dense-tensor", "mlir::ModuleOp"> {
  let summary = "Converts SparseTensor inputs to its component tensors inputs and emits a SparseToDenseOp for every op that consumes a SparseTensor.";
  let constructor = "CreateDTensorSparseTensorToDenseTensor()";
  let dependentDialects = [
  ];
}

def DTensorSparseExpansion
    : Pass<"dtensor-sparse-expansion", "mlir::ModuleOp"> {
  let summary = "Convert ops that take in SparseTensor input to its corresponding Sparse or Dense ops.";
  let constructor = "CreateDTensorSparseExpansion()";
  let dependentDialects = [
  ];
}

def DTensorInferShapesForRestoreV2Op
    : Pass<"dtensor-infer-shapes-for-restorev2-op", "mlir::ModuleOp"> {
  let summary = "Infer shapes of the outputs of tf.RestoreV2Op from the AssignVariableOps that consume those outputs. This is used for DTensor integration with TF Checkpoint.";
  let constructor = "CreateDTensorInferShapesForRestoreV2Op()";
  let dependentDialects = [
  ];
}

def DTensorSetHloShardingPass : Pass<"dtensor-set-hlo-sharding", "mlir::ModuleOp"> {
  let summary = "Set `mhlo.sharding` attribute for function inputs and ops.";

  let description = [{
    For each `tf.DTensorLayout` op, the pass sets `mhlo.sharding` attributes for
     related Ops and the related outer layer function's inputs and outputs.
    By default the operation is applied on every DTensorLayout op.
    If `check_layout_use_xla_spmd`is set to true, then the pass checks every
     DTensorLayout must have Mesh config with use_xla_spmd.
    For example, by default the follow code

    ```
    func.func @main(%arg0) -> tensor<8x8xi32> {
      %1 = "tf.DTensorLayout"(%arg0)
      %2 = "tf.Identity"(%1)
      %3 = "tf.DTensorLayout"(%2)
      return %3 : tensor<8x8xi32>
    }
    ```

    will be converted to

    ```
    func.func @main(%arg0 {mhlo.sharding = ""}) -> (tensor<8x8xi32> {mhlo.sharding = ""})
      %1 = "tf.DTensorLayout"(%arg0)
      %2 = "tf.Identity"(%1) {mhlo.sharding = ""}
      %3 = "tf.DTensorLayout"(%2)
      return %3 : tensor<8x8xi32>
    }
    ```

    When `check_layout_use_xla_spmd` is set to true, the pass throws an exception for the above example.
    Meanwhile, if `check_layout_use_xla_spmd` is set to true, the follow code

    ```
    func.func @main(%arg0) -> tensor<8x8xi32> {
      %1 = "tf.DTensorLayout"(%0) {mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>}
      %2 = "tf.Identity"(%1)
      %3 = "tf.DTensorLayout"(%2){mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>}
      return %3 : tensor<8x8xi32>
    }
    ```

    will be converted to

    ```
    func.func @main(%arg0 {mhlo.sharding = ""}) -> (tensor<8x8xi32> {mhlo.sharding = ""})
      %1 = "tf.DTensorLayout"(%0) {mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>}
      %2 = "tf.Identity"(%1)
      %3 = "tf.DTensorLayout"(%2){mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>}
      return %3 : tensor<8x8xi32>
    }
    ```
  }];

  let constructor = "CreateDTensorSetHloShardingPass()";
  let dependentDialects = [
    "::mlir::dtensor::DTensorDialect"
  ];

  let options = [
    Option<"check_layout_use_xla_spmd_", "check_layout_use_xla_spmd", "bool",
     /*default=*/"false",
    "If true, the pass checks every DTensorLayout must have Mesh config with use_xla_spmd."
    "Otherwise, the check is disabled.">
  ];
}

def DTensorReplaceAuxiliaryDTensorLayoutOpPass
    : Pass<"dtensor-replace-auxiliary-layout-op", "mlir::ModuleOp"> {
  let summary = "Replace auxiliary `tf.DTensorLayout` op with `tf.Identity`.";

  let description = [{
    Canonicalizer and DCE transformation passes may remove ops in the graph and
    result in multiple consecutive DTensorLayout ops. The pass detects all such
     cases and replaces unnecessary DTensorLayout ops with Identity ops.

    Removes `tf.DTensorLayouts` and inserts a `tf.Identity`.
    For example:

    ```
      %0 = tf.DTensorLayout(arg0)
      %1 = tf.DTensorLayout(%0)
      %2 = tf.Add(%1, %1)
    ```

    will be converted to:
    ```
      %0 = tf.Identity(arg0)
      %1 = tf.DTensorLayout(%0)
      %2 = tf.Add(%1, %1)
    ```
  }];

  let constructor = "CreateDTensorReplaceAuxiliaryDTensorLayoutOpPass()";
}

def DTensorRemoveDTensorLayoutPass : Pass<"dtensor-remove-dtensorlayout", "mlir::ModuleOp"> {
  let summary = "Remove DTensor `tf.DTensorLayout` ops.";

  let description = [{
    The pass removes DTensor `tf.DTensorLayout` ops.
    For example,

    ```mlir
      %1 = tf.DTensorLayout(%0)
      %2 = tf.SomeOp(%1)
    ```

    will be converted to

    ```mlir
      %2 = tf.SomeOp(%0)
    ```
  }];

  let constructor = "CreateDTensorRemoveDTensorLayoutPass()";
  let dependentDialects = [
    "::mlir::dtensor::DTensorDialect"
  ];
}

#endif // TF_DTENSOR_PASSES
