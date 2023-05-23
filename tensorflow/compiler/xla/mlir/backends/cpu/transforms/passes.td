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

def ConvertXlaCpuToCpuRuntimePass :
    Pass<"xla-cpu-to-cpu-runtime", "mlir::ModuleOp"> {
  let summary = "Converts xla_cpu operations to XLA Cpu runtime custom calls";

  let description = [{
      Converts xla_cpu dialect operations to XLA Cpu runtime custom calls.
  }];

  let constructor = "createConvertXlaCpuToCpuRuntimePass()";
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

def LegalizeLibraryOpsPass :
   Pass<"xla-legalize-library-ops", "mlir::func::FuncOp"> {
  let summary = "Legalizes ops that map to runtime library calls.";

  let description = [{
      Lowers ops that map to a runtime library to xla_cpu ops.
  }];

  let dependentDialects = [
    "mlir::mhlo::MhloDialect", "mlir::tensor::TensorDialect",
    "mlir::xla_cpu::XlaCpuDialect"
  ];

  let constructor = "createLegalizeLibraryOpsPass()";
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

def SparseCustomCallRewritingPass :
    Pass<"xla-sparse-custom-call-to-pack", "mlir::func::FuncOp"> {
  let summary = "Converts CustomCall operations to sparse operations";

  let description = [{
      Converts CustomCallOp operations to sparse operations
      to avoid dropping sparsity information during the
      HLO roundtrip.
  }];

  let dependentDialects = [
      "mlir::sparse_tensor::SparseTensorDialect",
      "mlir::chlo::ChloDialect",
  ];

  let constructor = "createSparseCustomCallRewritingPass()";
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
