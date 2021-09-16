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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
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

// Pass to tile, promote and vectorize linalg.matmul on buffers.
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForMatMulPass();

// Pass to optimize padding in tiled loops by peeling the final loop iteration.
std::unique_ptr<mlir::FunctionPass> CreatePeelTiledLoopsPass();

// Pass to tile and fuse linalg.generic on tensors that models reduction.
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForReductionPass();

// Pass to pad linalg ops.
std::unique_ptr<mlir::FunctionPass> CreatePadTiledOpsPass();

// Pass to vectorize linalg ops.
std::unique_ptr<mlir::FunctionPass> CreateVectorizeTiledOpsPass();

// Pass to tile elementwise ops on tensors.
std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForCWisePass();

// Pass to specialize linalg.matmul to dot, matvec or vecmat.
std::unique_ptr<mlir::FunctionPass> CreateLinalgMatmulSpecializationPass();

// Pass to split _Fused Tensorflow kernels into primitives.
std::unique_ptr<mlir::FunctionPass> CreateFissionPass();

// Pass to optimize broadcasts based on the symbolic shape constraints.
std::unique_ptr<mlir::FunctionPass> CreateSymbolicShapeOptimizationPass(
    bool constraints_only = false);

// Creates `tf_device.cluster` operations according to the TF CPURT clustering
// policy.
std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass();
std::unique_ptr<mlir::FunctionPass> CreateTfCpurtClusteringPass(
    llvm::ArrayRef<std::string> oplist, int min_cluster_size);

// Returns true if the `value` type is a memref that is contiguous in memory.
bool IsContiguousMemref(mlir::Value value);

}  // namespace tensorflow

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_PASSES_H_
