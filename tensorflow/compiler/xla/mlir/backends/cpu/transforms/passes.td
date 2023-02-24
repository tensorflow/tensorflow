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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Auxiliary passes for lowering to XLA Cpu runtime.
//===----------------------------------------------------------------------===//

def ConvertLmhloToCpuRuntimePass :
    Pass<"xla-lmhlo-to-cpu-runtime", "mlir::ModuleOp"> {
  let summary = "Converts lmhlo operations to XLA Cpu runtime custom calls";

  let description = [{
      Converts lmhlo dialect operations to XLA Cpu runtime custom calls.
  }];

  let constructor = "createConvertLmhloToCpuRuntimePass()";
}

def LegalizeXlaAbiPass :
    Pass<"xla-legalize-abi", "mlir::ModuleOp"> {
  let summary = "Converts layouts and data formats at ABI boundaries";

  let description = [{
      Converts layouts defined in entry_computation_layout to the default
      layout assumed by the MLIR pipeline. The same is done for custom calls.

      This pass should run immediately after expand-hlo-tuples.
  }];

  let dependentDialects = [
      "mlir::func::FuncDialect", "mlir::mhlo::MhloDialect"
  ];

  let constructor = "createXlaAbiLegalizationPass()";
}

def LegalizeCollectiveOpsPass :
   Pass<"xla-legalize-collective-ops", "mlir::func::FuncOp"> {
  let summary = "Legalizes collective ops to AllToAll and regular ops.";

  let description = [{
      Lowers collective ops to xla_cpu ops.
  }];

  let dependentDialects = [
    "mlir::mhlo::MhloDialect", "mlir::tensor::TensorDialect",
    "mlir::xla_cpu::XlaCpuDialect"
  ];

  let constructor = "createLegalizeCollectiveOpsPass()";
}

def LegalizeI1VectorTransferOpsPass :
   Pass<"xla-legalize-i1-vector-transfers", "mlir::func::FuncOp"> {
  let summary = "Legalizes transfer ops operating on vectors of i1.";

  let description = [{
      Replaces transfers involving vector<i1>s with memref casts to i8, and
      vector comparisons.
  }];

  let dependentDialects = [
    "mlir::vector::VectorDialect", "mlir::xla_cpu::XlaCpuDialect"
  ];

  let constructor = "createLegalizeI1VectorTransferOpsPass()";
}

def ConvertXlaCpuMemRefElementCastToLLVMPass :
   Pass<"xla-convert-memref-element-cast-to-llvm", "mlir::func::FuncOp"> {
  let summary = "Converts xla_cpu.memref_element_cast ops to LLVM.";

  let description = [{
      Rewrites xla_cpu.memref_elements_cast ops as a new memref descriptor,
      where the allocated and aligned pointers are updated.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];

  let constructor = "createConvertXlaCpuMemRefElementCastToLLVMPass()";
}

def RemoveCopiesToOutParamsPass :
   Pass<"xla-remove-copies-to-out-params", "mlir::func::FuncOp"> {
  let summary = "Removes redundant alloc/copy pairs to out params.";

  let description = [{
      Removes redundant alloc/alloca + copy pairs that can remain after running
      bufferization's BufferResultsToOutParams pass.
  }];

  let constructor = "createRemoveCopiesToOutParamsPass()";
}

def SparseCustomCallToPackPass :
    Pass<"xla-sparse-custom-call-to-pack", "mlir::func::FuncOp"> {
  let summary = "Converts CustomCallOp to sparse_tensor_pack/unpack";

  let description = [{
      Converts CustomCallOp to sparse_tensor_pack/unpack to the corresponding
      sparse_tensor::PackOp and sparse_tensor::UnpackOp.

      The pass only converts CustomCallOp to a specific set of targets that take
      or return MLIR sparse tensors.
  }];

  let dependentDialects = [
      "mlir::sparse_tensor::SparseTensorDialect"
  ];

  let constructor = "createSparseCustomCallToPackUnpackOpPass()";
}

def RewriteReallocToAllocPass :
    Pass<"xla-rewrite-realloc-to-alloc", "mlir::func::FuncOp"> {
  let summary = "Rewrites realloc to alloc + copy";

  let description = [{
    Rewrites realloc to alloc + copy.

    This pass is actually undesirable. However, the current buffer-deallocation
    pass fails to handle reallocation correctly, and causes double free issues
    when using memref::ReallocOp. We should remove this pass after the bug is
    fixed upstream.
  }];

  let dependentDialects = [
      "mlir::memref::MemRefDialect"
  ];

  let constructor = "createRewriteReallocToAllocPass()";
}

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_
