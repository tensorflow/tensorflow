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

#ifndef TENSORFLOW_DTENSOR_MLIR_CREATE_DTENSOR_MLIR_PASSES_H_
#define TENSORFLOW_DTENSOR_MLIR_CREATE_DTENSOR_MLIR_PASSES_H_

#include <memory>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace dtensor {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorOpToDeviceClusterPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorDeviceMeshClusterCoarsening();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateDTensorDCE();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorUndoMergeConstAcrossMesh();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorElideIdentityBeforeCopyToMesh();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorConstantFolding();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceSumOptimization();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceScatterOptimization();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceCombineOptimization();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorMixedPrecisionReducePass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorSetDefaultSharding();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorDesignateResourceHandleMesh();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorPropagateDefaultLayout();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorHandleCrossClusterDependencies();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAnnotateGlobalShape();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorLayoutPropagationPassV2();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMeshPropagationPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSPMDExpansion();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorClusterFunctionConversion();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorPropagateDeviceIdToFunctionArgs();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorTPUIntegration();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorTpuAddResourceDeviceAttribute();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorUpdateTPUMetadata();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorEmbeddingPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorEmbeddingPassV2();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorEmbeddingCheckpointPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateFunctionRenamingPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMultiDeviceExpansionPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllReduceLoweringPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorReduceScatterLoweringPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllGatherLoweringPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllScatterLoweringPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMergeClustersPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorLowerSendRecv();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMoveCompilationToHost();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSparseTensorToDenseTensor();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSparseExpansion();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorInferShapesForRestoreV2Op();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSetHloShardingPass(
    std::optional<bool> check_layout_use_xla_spmd = std::optional<bool>(false));

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorLayoutToXlaShardingOpPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorReplaceAuxiliaryDTensorLayoutOpPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorRemoveDTensorLayoutPass();

// Creates a pass that replaces `tf.Relayout` with `tf.Identity`.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorReplaceRelayoutWithIdentityPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_CREATE_DTENSOR_MLIR_PASSES_H_
