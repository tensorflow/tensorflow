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

#ifndef TF_KERNEL_GEN_PASSES
#define TF_KERNEL_GEN_PASSES

include "mlir/Pass/PassBase.td"

def TFKernelToLLVMPass : Pass<"tf-kernel-to-llvm", "ModuleOp"> {
  let summary = "Pass for applying LLVM legalization patterns.";
  let constructor = "transforms::CreateTFKernelToLLVMPass()";
  let options = [
      Option<"blob_annotation_", "blob-annotation", "std::string",
           /*default=*/"\"gpu.binary_blob\"", "Blob attribute name">,
  ];
}

def EmbedTFFrameworkPass
    : Pass<"embed-tf-framework", "ModuleOp"> {
  let summary = "Pass to embed TF Framework for allocation and assertions,";
  let constructor = "tf_framework::CreateEmbedTFFrameworkPass()";
}

def RewriteTFFrameworkAssert : Pass<"rewrite-tf-framework-assert", "ModuleOp"> {
  let summary = "Pass to rewrite TFAssertOps to CondBranchOp";
  let constructor = "tf_framework::CreateRewriteTFFrameworkAssert()";
}

def TFToJITInvocationPass : Pass<"tf-to-jit-invocation", "mlir::func::FuncOp"> {
  let summary = "Pass to rewrite all TF operations to JIT invocations";
  let constructor = "transforms::CreateTFToJITInvocationPass()";
  let options = [
      ListOption<"tile_sizes_", "tile-sizes", "int64_t", "Tiling sizes",
                 "llvm::cl::ZeroOrMore">,
      ListOption<"unroll_factors_", "unroll-factors", "int64_t",
                 "Unrolling in each tile dimension", "llvm::cl::ZeroOrMore">,
      Option<"max_supported_rank_", "max-supported-rank", "int64_t",
             /*default=*/"", "Max rank that this kernel supports">,
      Option<"enable_ftz_", "enable-ftz", "bool", /*default=*/"",
             "Enable the denormal flush to zero mode when generating code">,
      Option<"index_64bit_", "index_64bit", "bool", /*default=*/"",
             "Enable the 64 bit indexing for GPU kernels">,
      Option<"cpu_codegen_", "cpu-codegen", "bool", /*default=*/"",
             "CPU codegen (false implies GPU)">,
      Option<"jit_i64_indexed_for_large_tensors_",
             "jit_i64_indexed_for_large_tensors", "bool", /*default=*/"false",
             "Enable JIT compilation of i64-indexed kernels for large input "
             "tensors.">,
  ];
}

def BufferReusePass : Pass<"buffer-reuse", "mlir::func::FuncOp"> {
  let summary = "Pass to find and annotate candidates for buffer reuse.";
  let constructor = "transforms::CreateBufferReusePass()";
}

def ShapeToDescriptorsPass : Pass<"shape-to-descriptors", "ModuleOp"> {
  let summary = "Pass to transform shape computations to descriptors";
  let constructor = "transforms::CreateShapeToDescriptorsPass()";
}

def KernelgenFinalBufferizePass : Pass<"kernelgen-final-bufferize", "ModuleOp"> {
  let summary = "Pass to transform late operations on values to buffer based "
                "ones.";
  let constructor = "transforms::CreateKernelgenFinalBufferizePass()";
}

def GpuKernelToNVVMPass : Pass<"gpu-kernel-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Pass to transform a gpu module to nvvm.";
  let constructor = "transforms::CreateGpuKernelToNvvmPass()";
}

def GpuKernelToROCDLPass : Pass<"gpu-kernel-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Pass to transform a gpu module to rocdl.";
  let constructor = "transforms::CreateGpuKernelToRocdlPass()";
}

def GpuKernelToBlobPass : Pass<"gpu-kernel-to-blob", "gpu::GPUModuleOp"> {
  let summary = "Pass to annotate GPU Module with its PTX";
  let options = [
    Option<"blob_annotation_", "blob-annotation", "std::string",
           /*default=*/"\"gpu.binary_blob\"", "Blob attribute name">,
    ListOption<"architectures_", "arch", "std::string", "GPU architectures">,
    Option<"generate_fatbin_", "generate-fatbin", "bool", /*default=*/"true",
           "Bundle machine code for the different architectures in one "
           "fatbin.">,
    Option<"print_ptx_", "print-ptx", "bool", /*default=*/"false",
           "Print generated PTX code per target architecture.">,
    Option<"print_llvmir_", "print-llvmir", "bool", /*default=*/"false",
           "Print llvm ir when lowering code per target architecture.">,
  ];
  let constructor = "transforms::CreateGpuKernelToBlobPass()";
}

def ParallelLoopsToSequential : Pass<"parallel-loops-to-sequential", "mlir::func::FuncOp"> {
  let summary = "Pass to convert scf::ParallelOp to scf::ForOp";
  let constructor = "transforms::CreateParallelLoopsToSequential()";
}

def PropagateTfAbiKnowledgeToKernels
    : Pass<"propagate-tf-abi-knowledge-to-kernels", "mlir::func::FuncOp"> {
  let summary = "Pass to propagate tensorflow ABI knowledge to kernels";
  let constructor = "transforms::CreatePropagateTfAbiKnowledgeToKernels()";
}

def PropagateShapeKnowledgeToKernels
    : Pass<"propagate-shape-knowledge-to-kernels", "mlir::func::FuncOp"> {
  let summary = "Pass to propagate shape information into kernels";
  let constructor = "transforms::CreatePropagateShapeKnowledgeToKernels()";
}

def EmbedMemRefPrintsPass : Pass<"embed-memref-prints", "mlir::ModuleOp"> {
  let summary = "Pass to print content of memrefs";
  let constructor = "transforms::CreateEmbedMemRefPrintsPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def MapParallelLoopsPass
    : Pass<"map-parallel-loops-to-gpu", "mlir::func::FuncOp"> {
  let summary = "Greedily maps loops to GPU hardware dimensions.";
  let constructor = "transforms::CreateMapParallelLoopsPass()";
  let description = [{
    Greedily maps loops to GPU hardware dimensions.
  }];
}

def FuseInnerParallelLoopsPass
    : Pass<"fuse-inner-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Limited pass to forward stores to loads.";
  let constructor = "transforms::CreateFuseInnerParallelLoopsPass()";
  let description = [{
    Directs parallel loop fusion to the inner loops. This cannot be done with
    a passmanager alone ATM, as nested pass managers require operations to
    be closed from above.
  }];
}

def VectorizationPass : Pass<"vectorization", "mlir::func::FuncOp"> {
  let summary = "Pass to create vectorized operations for CPU.";
  let constructor = "transforms::CreateVectorizationPass()";
  let description = [{
    Takes linalg loops on buffers and vectorizes the loads/stores and operations
    in between. This is intended for the CPU backend.
  }];
}

def VectorizationCleanupPass : Pass<"vectorization-cleanup", "mlir::func::FuncOp"> {
  let summary = "Pass to remove dead code in VectorizationPass.";
  let constructor = "transforms::CreateVectorizationCleanupPass()";
  let description = [{
    VectorizationPass creates some memory transfers that are unneeded but cannot
    be safely removed by current generic infrastructure. This removes the
    specific operations.
  }];
}

def CopyCleanupPass : Pass<"copy-cleanup", "mlir::func::FuncOp"> {
  let summary = "Pass to remove copies which are consumed by a GenericOp.";
  let constructor = "transforms::CreateCopyCleanupPass()";
  let description = [{
    We can have GenericOps which have operands that are a copy, but the copy is
    not used by any other op. In this case, the GenericOp can just use the
    buffer which is the source of the copy, and we can remove the Alloc and the
    copy.
  }];
}

#endif // TF_KERNEL_GEN_PASSES
