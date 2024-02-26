/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_CLUSTERING_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_CLUSTERING_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

// Verifies that all MLIR Ops have the expected attributes.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyClusteringPass();

// Creates a pass that forms clusters from operations of the same
// `_replication_info` attribute.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTPUClusterFormationPass(bool strict_clusters = false);

// Creates a pass that extracts outside compilation (Host ops inside device
// cluster) at head/tail of Device cluster to run before/after XLA computation.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateExtractHeadTailOutsideCompilationPass();

// Creates a pass that extract outside compilation (Host ops inside cevice
// cluster) ops to a separate parallel_execute region to run on CPU.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateExtractOutsideCompilationPass();

// Create a pass that encapsulates StatefulPartitionedCallOp within a cluster.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateXlaClusterFormationPass();

// Create a pass that rewrites entry functions with `_xla_compile_device` into a
// `tf.StatefulPartitionedCall` to the original function.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateXlaOutlineEntryFunctionsPass();

// Creates a pass that marks unsupported ops in device cluster for outside
// compilation.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateMarkOpsForOutsideCompilationPass();

// Creates a pass that hoists reads out of a replicate that are on a variable
// whose value is broacast to all replicas.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateHoistBroadcastReadPass();

// Creates a pass that moves broadcasts from TF host ops to XLA code, encoded as
// XlaAllReduces. This enables use of the device network for broadcasts, which
// is faster.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateXlaBroadcastPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_MARKOPSFOROUTSIDECOMPILATIONPASS
#define GEN_PASS_DECL_TPUCLUSTERFORMATIONPASS
#define GEN_PASS_DECL_TPUEXTRACTHEADTAILOUTSIDECOMPILATIONPASS
#define GEN_PASS_DECL_TPUEXTRACTOUTSIDECOMPILATIONPASS
#define GEN_PASS_DECL_VERIFYCLUSTERINGPASS
#define GEN_PASS_DECL_XLACLUSTERFORMATIONPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_CLUSTERING_PASSES_H_
