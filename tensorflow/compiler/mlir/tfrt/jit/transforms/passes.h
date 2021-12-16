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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace tensorflow {

// Pass for trivial buffer forwarding for the linalg.generic operations.
std::unique_ptr<mlir::FunctionPass> CreateLinalgTrivialBufferForwardingPass();

// Pass for trivial copy removal of linalg.copy operations.
std::unique_ptr<mlir::FunctionPass> CreateLinalgTrivialCopyRemovalPass();

// Pass to optimize padding in tiled loops by peeling the final loop iteration.
std::unique_ptr<mlir::FunctionPass> CreatePeelTiledLoopsPass();

// Pass to tile and fuse linalg.generic on tensors that models reduction.
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForReductionPass();
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForReductionPass(
    int64_t reduction_1d_tile_size,
    llvm::ArrayRef<int64_t> reduction_2d_tile_sizes);

// Pass to fuse `linalg.fill` into a tiled reduction.
std::unique_ptr<mlir::FunctionPass> CreateFuseFillIntoTiledReductionPass();

// Pass to replace 'i1' tensor types with 'i8' tensor types. This pass is a
// temporary workaround to avoid the problem of vectorizing 'i1' tensors (see
// b/205714705).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateCpuRtLegalizeI1TypesPass();

// Pass to sink unused outputs of `tiled_loop` into the loop body.
std::unique_ptr<mlir::FunctionPass> CreateSinkUnusedOutputs();

// Pass to vectorize linalg ops.
std::unique_ptr<mlir::FunctionPass> CreateVectorizeTiledOpsPass();

// Pass to tile elementwise ops on tensors.
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForCWisePass();
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForCWisePass(
    int64_t cwise_tile_size);

// Pass to split _Fused Tensorflow kernels into primitives.
std::unique_ptr<mlir::FunctionPass> CreateFissionPass();

// Pass to fuse Linalg generic operations on Tensors.
std::unique_ptr<mlir::FunctionPass> CreateFusionPass();

// Pass to optimize broadcasts based on the symbolic shape constraints.
std::unique_ptr<mlir::FunctionPass> CreateSymbolicShapeOptimizationPass(
    bool constraints_only = false);

// Pass to replace 0-d tensor inputs to LinalgOp with extracted elements.
std::unique_ptr<mlir::FunctionPass> CreateDetensorizeLinalgPass();

// Creates `tf_device.cluster` operations according to the TF CPURT clustering
// policy.
std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass();
std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass(
    llvm::ArrayRef<std::string> oplist, int min_cluster_size);

// Pass to replace math ops with approximations.
std::unique_ptr<mlir::FunctionPass> CreateMathApproximationPass(
    llvm::ArrayRef<std::string> oplist = {});

// Returns true if the `value` type is a memref that is contiguous in memory.
bool IsContiguousMemref(mlir::Value value);

// Detects the combiner in the body of LinalgOp if any. Currently, only
// ops with a single combiner are supported.
mlir::FailureOr<mlir::Operation *> DetectCombiner(
    mlir::linalg::LinalgOp linalg_op);

}  // namespace tensorflow

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/passes.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_PASSES_H_
