/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

// Pass to replace some of the Standard ops with TF Framework ops.
// * adds tf_framework::OpKernelContextType argument to the function
// * std.alloc becomes tf_framework.alloc_raw
// * std.dealloc becomes tf_framework.dealloc_raw
// * std.assert becomes tf_framework.assert
std::unique_ptr<OperationPass<ModuleOp>> CreateEmbedTFFrameworkPass();

// Pass to convert tf_framework.assert operations to calls to
// tf_framework.report_error and create the required control flow to abort the
// function on failed execution.
std::unique_ptr<OperationPass<ModuleOp>> CreateRewriteTFFrameworkAssert();

}  // namespace tf_framework

namespace transforms {

// Pass to find and annotate candidates for buffer reuse.
std::unique_ptr<OperationPass<func::FuncOp>> CreateBufferReusePass();

// Pass to rewrite all TF operations to JIT invocations through the TF
// framework.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTFToJITInvocationPass(
    llvm::ArrayRef<int64_t> tile_sizes = {},
    llvm::ArrayRef<int64_t> unroll_factors = {}, int64_t max_supported_rank = 5,
    bool enable_ftz = false, bool index_64bit = false, bool cpu_codegen = false,
    bool jit_i64_indexed_for_large_tensors = false);

// Pass for applying LLVM legalization patterns.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFKernelToLLVMPass(
    mlir::StringRef blob_annotation = {});

// Pass to tranform shape computations in shape dialect to standard and scf
// using memref descriptors.
std::unique_ptr<OperationPass<ModuleOp>> CreateShapeToDescriptorsPass();

// Pass to convert scf::ParallelOp to scf::ForOp.
std::unique_ptr<OperationPass<func::FuncOp>> CreateParallelLoopsToSequential();

// Pass to annotate GPU Module with its PTX.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> CreateGpuKernelToBlobPass(
    mlir::StringRef blob_annotation = {},
    ArrayRef<std::string> architectures = {}, bool print_ptx = false,
    bool print_llvmir = false, bool enable_ftz = false);

// Pass to propagate tensorflow runtime ABI knowledge across kernel boundaries.
std::unique_ptr<OperationPass<func::FuncOp>>
CreatePropagateTfAbiKnowledgeToKernels();

// Pass to propagate shape equalities across kernel boundaries.
std::unique_ptr<OperationPass<func::FuncOp>>
CreatePropagateShapeKnowledgeToKernels();

// Pass to print content of memrefs.
std::unique_ptr<OperationPass<ModuleOp>> CreateEmbedMemRefPrintsPass();

/// Greedily maps loops to GPU hardware dimensions.
std::unique_ptr<mlir::OperationPass<func::FuncOp>> CreateMapParallelLoopsPass();

/// We need to direct fusion to the inner loops. This cannot be done with
/// a passmanager alone ATM, as nested pass managers require operations to
/// be closed from above.
std::unique_ptr<mlir::OperationPass<func::FuncOp>>
CreateFuseInnerParallelLoopsPass();

// Pass to create vectorized code for CPU.
std::unique_ptr<OperationPass<func::FuncOp>> CreateVectorizationPass();

// Pass to remove unneeded code generated in VectorizationPass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateVectorizationCleanupPass();

// Pass to remove copies which are consumed by a GenericOp.
std::unique_ptr<OperationPass<func::FuncOp>> CreateCopyCleanupPass();

std::unique_ptr<OperationPass<ModuleOp>> CreateKernelgenFinalBufferizePass();

}  // namespace transforms

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_
