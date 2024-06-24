/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_PASSES_TD_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_PASSES_TD_

include "mlir/Pass/PassBase.td"

def PropagateSliceIndicesPass :
   Pass<"xla-gpu-propagate-slice-indices", "mlir::ModuleOp"> {
  let summary = "Propagates slice indices from the entry function to all callees.";

  let description = [{
      Propagates xla.slice_index attributes from the function with the xla.entry
      attribute to all other functions.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];

  let constructor = "CreatePropagateSliceIndicesPass()";
}

def ConvertPureCallOpsPass
    : Pass<"xla-gpu-convert-pure-call-ops", "mlir::func::FuncOp"> {
  let summary = "Converts xla_gpu.pure_call to func.call";
  let description = [{
      We use xla_gpu.pure_call ops for calls to enable CSE and other
      transformations (e.g. LICM). This pass rewrites our custom ops to standard
      ops.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "xla::gpu::XlaGpuDialect"
  ];
  let constructor = "CreateConvertPureCallOpsPass()";
}

def LowerTensorsPass : Pass<"xla-gpu-lower-tensors", "mlir::ModuleOp"> {
  let summary = "Lowers tensors to llvm pointers and loads/stores.";

  let description = [{
      Lowers tensors to LLVM. We cannot use the memref lowerings because they
      are not compatible with XLA's ABI.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "xla::gpu::XlaGpuDialect",
  ];
  let options = [
    Option<"is_amd_gpu_", "is_amd_gpu", "bool", /*default=*/"false",
           "True if AMD GPU.">,
    Option<"gpu_arch_", "gpu_arch", "std::string", /*default=*/"",
           "CUDA or ROCm compute capability.">,
  ];
  let constructor = "CreateLowerTensorsPass()";
}

def MergePointersToSameSlicePass :
   Pass<"xla-gpu-merge-pointers", "mlir::ModuleOp"> {
  let summary = "Merges pointers that share slices.";

  let description = [{
      When a function has multiple pointer arguments with the same slice index,
      merges them.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];

  let constructor = "CreateMergePointersToSameSlicePass()";
}

def SimplifyArithPass : Pass<"xla-gpu-simplify-arith", "mlir::func::FuncOp"> {
  let summary = "Simplifies arith using XLA's range-aware simplifier.";

  let description = [{
      We often emit bounds checks that are statically known to be satisfied.
      This pass removes them.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
  ];

  let constructor = "CreateSimplifyArithPass()";
}

def SimplifyAffinePass : Pass<"xla-gpu-simplify-affine", "mlir::ModuleOp"> {
  let summary = "Simplifies affine.apply using XLA's range-aware simplifier.";

  let description = [{
      The standard affine canonicalizer cannot simplify all expressions, since
      it is unaware of range information. This pass uses `xla.range` attributes
      on arguments and ops for simplification. It also lowers floordiv and mod
      to simpler expressions than lower-affine. This pass only works for
      expressions for which we can prove the LHS of mod and div is nonnegative.
  }];

  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
  ];

  let constructor = "CreateSimplifyAffinePass()";
}

def ExpandFloatOpsPass : Pass<"xla-gpu-expand-float-ops", "mlir::ModuleOp"> {
  let summary = "Expands float ops that are not natively supported.";

  let description = [{
     Not all float ops are natively supported, either because they don't exist
     in hardware or they are too inaccurate.

     This pass replaces these ops with alternative implementations.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::math::MathDialect",
    "mlir::mhlo::MhloDialect"
  ];

  let options = [
    Option<"pre_ampere_", "pre-ampere", "bool", /*default=*/"false",
           "Rewrite ops that are not supported on architectures before Ampere">,
  ];
}

def LowerXlaGpuToScfPass :
   Pass<"xla-gpu-lower-xla-gpu-to-scf", "mlir::ModuleOp"> {
  let summary = "Lowers xla_gpu to SCF.";

  let dependentDialects = [
    "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect", "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect", "xla::gpu::XlaGpuDialect",
  ];

  let constructor = "CreateLowerXlaGpuToScfPass()";
}

def EraseDeadFunctionsPass : Pass<"xla-erase-dead-functions", "mlir::ModuleOp"> {
  let summary = "Deletes unused functions";

  let description = [{
      Deletes functions that are not called.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "xla::gpu::XlaGpuDialect"
  ];

  let constructor = "CreateEraseDeadFunctionsPass()";
}

def LowerToLLVMPass :
   Pass<"xla-gpu-lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers to LLVM.";

  let description = [{
    Lowers the rest to LLVM
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"
  ];

  let constructor = "CreateLowerToLLVMPass()";
}

def VectorizeLoadsAndStoresPass :
   Pass<"xla-gpu-vectorize-loads-stores", "mlir::func::FuncOp"> {
  let summary = "Vectorizes loads and stores.";

  let description = [{
    Rewrites tensor.extract and tensor.insert ops inside loops to their vector
    equivalents (vector.transfer_read and vector.transfer_write + vector.extract
    and vector.insert).
  }];

  let dependentDialects = [
    "mlir::vector::VectorDialect",
  ];

  let constructor = "CreateVectorizeLoadsAndStoresPass()";
}

def OptimizeLoopsPass :
   Pass<"xla-gpu-optimize-loops", "mlir::func::FuncOp"> {
  let summary = "Unrolls and pipelines loops.";

  let description = [{
    Unrolls loops with a small trip count. Pipelines loops with a large trip
    count.
  }];

  let dependentDialects = [
    "mlir::vector::VectorDialect",
    "xla::gpu::XlaGpuDialect",
  ];

  let constructor = "CreateOptimizeLoopsPass()";
}

def UnswitchLoopsPass :
   Pass<"xla-gpu-unswitch-loops", "mlir::func::FuncOp"> {
  let summary = "Swaps scf.if and scf.for.";

  let description = [{
      Extracts `scf.if` ops with conditions that are independent of the loop
      variable from `scf.for` by doing the following rewrite:

      Before:

      %cond = some_cond() : i1
      %results = scf.for {
        %some_val = scf.if %cond  {
        } else {
        }
        scf.yield %some_val
      }

      After:

      %cond = some_cond() : i1
      %results = scf.if %cond {
         %results = scf.for {
            %some_val = scf.if %true  {
            } else {
            }
         }
         yield %results
      } else {
         %results = scf.for {
            %some_val = scf.if %false  {
            } else {
            }
         }
         yield %results
      }

      This only triggers if there is a single `scf.if` op in the loop body (and
      nothing else).
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect", "mlir::scf::SCFDialect"
  ];

  let constructor = "CreateUnswitchLoopsPass()";
}

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_PASSES_TD_
