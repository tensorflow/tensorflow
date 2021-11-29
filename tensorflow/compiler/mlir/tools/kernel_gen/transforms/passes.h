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
std::unique_ptr<FunctionPass> CreateBufferReusePass();

// Pass to rewrite all TF operations to JIT invocations through the TF
// framework.
std::unique_ptr<FunctionPass> CreateTFToJITInvocationPass(
    llvm::ArrayRef<int64_t> tile_sizes = {},
    llvm::ArrayRef<int64_t> unroll_factors = {}, int64_t max_supported_rank = 5,
    bool enable_ftz = false, bool cpu_codegen = false);

// Pass for applying LLVM legalization patterns.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFKernelToLLVMPass(
    mlir::StringRef blob_annotation = {});

// Pass to tranform shape computations in shape dialect to standard and scf
// using memref descriptors.
std::unique_ptr<OperationPass<ModuleOp>> CreateShapeToDescriptorsPass();

// Pass to tranform compute computations (hlo and linalg) on values to their
// corresponding counterparts on buffers. Also bufferizes function signatures.
std::unique_ptr<OperationPass<ModuleOp>> CreateComputeOpAndFuncBufferizePass();

// Pass to bufferize `linalg.tiled_loop` including the operations contained in
// its body.
std::unique_ptr<FunctionPass> CreateTiledLoopBufferizePass();

// Pass to tranform computations on values to their corresponding parts on
// buffers.
std::unique_ptr<OperationPass<ModuleOp>> CreateFinalBufferizePass();

// Pass to replace unsigned types with signless integers.
std::unique_ptr<OperationPass<ModuleOp>> CreateConvertToSignlessPass();

// Pass to convert scf::ParallelOp to scf::ForOp.
std::unique_ptr<FunctionPass> CreateParallelLoopsToSequential();

// Pass to annotate GPU Module with its PTX.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> CreateGpuKernelToBlobPass(
    mlir::StringRef blob_annotation = {},
    ArrayRef<std::string> architectures = {}, bool print_ptx = false,
    bool print_llvmir = false, bool enable_ftz = false);

// Pass to propagate tensorflow runtime ABI knowledge across kernel boundaries.
std::unique_ptr<FunctionPass> CreatePropagateTfAbiKnowledgeToKernels();

// Pass to propagate shape equalities across kernel boundaries.
std::unique_ptr<FunctionPass> CreatePropagateShapeKnowledgeToKernels();

// Pass to print content of memrefs.
std::unique_ptr<FunctionPass> CreateEmbedMemRefPrintsPass();

/// Greedily maps loops to GPU hardware dimensions.
std::unique_ptr<mlir::FunctionPass> CreateMapParallelLoopsPass();

/// We need to direct fusion to the inner loops. This cannot be done with
/// a passmanager alone ATM, as nested pass managers require operations to
/// be closed from above.
std::unique_ptr<mlir::FunctionPass> CreateFuseInnerParallelLoopsPass();

/// Pass that transforms gpu modules in standard dialect to NNVM.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
CreateGpuKernelToNvvmPass();

/// Pass that transforms gpu modules in standard dialect to ROCDL.
std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
CreateGpuKernelToRocdlPass();

// Pass to lower index cast on tensors to tensor dialect.
std::unique_ptr<FunctionPass> CreateLowerIndexCastPass();

// Pass to simplify shape ops.
std::unique_ptr<FunctionPass> CreateShapeSimplification();

// Pass to create vectorized code for CPU.
std::unique_ptr<FunctionPass> CreateVectorizationPass();

// Pass to remove unneeded code generated in VectorizationPass.
std::unique_ptr<FunctionPass> CreateVectorizationCleanupPass();

// Pass to remove copies which are consumed by a GenericOp.
std::unique_ptr<FunctionPass> CreateCopyCleanupPass();

}  // namespace transforms

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_
