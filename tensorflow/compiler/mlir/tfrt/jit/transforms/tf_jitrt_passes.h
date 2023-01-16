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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TRANSFORMS_TF_JITRT_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TRANSFORMS_TF_JITRT_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/xla/mlir_hlo/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace tensorflow {
#define GEN_PASS_DECL_MATHAPPROXIMATION
#define GEN_PASS_DECL_CLUSTERING
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

// Pass for trivial buffer forwarding for the linalg.generic operations.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLinalgTrivialBufferForwardingPass();

// Pass for trivial copy removal of memref.copy operations.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLinalgTrivialCopyRemovalPass();

// Pass to replace 'i1' tensor types with 'i8' tensor types. This pass is a
// temporary workaround to avoid the problem of vectorizing 'i1' tensors (see
// b/205714705).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateJitRtLegalizeI1TypesPass();

// Pass to split _Fused Tensorflow kernels into primitives.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateFissionPass();

// Pass to fuse Linalg generic operations on Tensors.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateFusionPass();

// Creates `tf_device.cluster` operations according to the TF JitRt clustering
// policy.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfJitRtClusteringPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfJitRtClusteringPass(llvm::ArrayRef<std::string> oplist,
                            int min_cluster_size);

// Pass to replace math ops with approximations.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateMathApproximationPass(llvm::ArrayRef<std::string> oplist = {});

// Returns true if the `value` type is a memref that is contiguous in memory.
bool IsContiguousMemref(mlir::Value value);

}  // namespace tensorflow

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TRANSFORMS_TF_JITRT_PASSES_H_
