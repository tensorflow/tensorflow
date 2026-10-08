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

include "mlir/Pass/PassBase.td"

def LowerXTileEntryPass : Pass<"xtile-cpu-lower-xtile-entry", "mlir::ModuleOp"> {
  let summary = "Lowers the entry function into the form required by the CPU runtime";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "xla::cpu::XlaCpuDialect",
    "xla::xtile::XTileDialect"
  ];

  let options = [
    Option<"prefer_vector_width_", "prefer_vector_width", "int32_t",
           /*default=*/"0",
           "prefer-vector-width value to set on the kernel function. "
           "Not set if 0.">,
  ];
}

def ShloToVectorPass : Pass<"xtile-cpu-shlo-to-vector", "mlir::ModuleOp"> {
  let summary = "Lowering satble hlo ops to vector ops";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
  ];
}

def LinalgElementwiseToVectorPass : Pass<"xtile-cpu-linalg-elementwise-to-vector",
                                       "mlir::ModuleOp"> {
  let summary = "Convert elementwise linalg ops to vector ops";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
}

def TensorOpsToBufferizablePass : Pass<"xtile-cpu-tensor-ops-to-bufferizable",
                                 "mlir::ModuleOp"> {
  let summary = "Lowering tensor dialect ops to bufferizable ops";

  let description = [{
    Some tensor ops such as bitcast are not directly bufferizable. This pass
    lowers such ops to ops that have bufferization support.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
}


def MemrefCopyToLoopsPass : Pass<"xtile-cpu-memref-copy-to-loops",
                                 "mlir::ModuleOp"> {
  let summary = "Rewrite mmeref.copy to loops.";

  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

def FuseElementwisePass : Pass<"xtile-cpu-fuse-elementwise"> {
  let summary = "Fuse linalg elementwise ops.";

  let description = [{
    This pass fuses multiple linalg elementwise ops into linalg elementwise ops
    that contain multiple instructions. This allows for fewer matrializations
    and fewer temporary allocations in bufferization.
  }];
}

def VectorToScalarPass : Pass<"xtile-cpu-vector-to-scalar"> {
  let summary = "Convert vector ops to scalar ops where possible.";

  let description = [{
    This pass converts elementwise vector ops to scalar ops if the operation
    acts on a single element.
  }];

  let dependentDialects = [
    "::mlir::vector::VectorDialect",
  ];
}

def DropVectorUnitDimsPass : Pass<"xtile-cpu-drop-vector-unit-dims"> {
  let summary = "Drop unit dimensions from vector operations.";
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def UnrollVectorsPass : Pass<"xtile-cpu-unroll-vectors"> {
  let summary = "Unroll multi-dimensional vectors to 1D vectors.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
  ];
}


def LegalizeNarrowFloatStoragePass
    : Pass<"xtile-cpu-legalize-narrow-float-storage", "mlir::ModuleOp"> {
  let summary = "Moves data movement of narrow floats onto same-width integers.";

  let description = [{
    x86 has no legal vector register class for float types narrower than 32
    bits unless the matching ISA extension is present (AVX512-BF16,
    AVX512-FP16, ...). SelectionDAG therefore scalarizes any such vector that
    is not immediately extended or truncated, promoting every element to f32
    and paying a libcall (e.g. `__truncsfbf2`) per element on the way back.
    This happens for phis, shuffles, masked loads and stores and selects of
    narrow float vectors, i.e. for exactly the ops that only move data around.

    This pass treats 8- and 16-bit floats as storage-only types:

    * Tile accesses to narrow float buffers go through an
      `xtile.memref_bitcast` view of the buffer as the same-width integer type,
      followed by a `tensor.bitcast` back to the float type.
    * These bitcasts are then moved through data movement ops (transpose,
      reshape, broadcast, slice, concatenate, mask, select) towards the ops that
      compute on the float values, e.g. `arith.extf`/`arith.truncf`, where the
      remaining pairs cancel.

    Ops the pass does not know simply stop the bitcasts, so it never changes
    semantics; at worst the float type survives on that op, as before.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::xla::xtile::XTileDialect",
  ];
}

def LowerMemRefBitcastPass
    : Pass<"xtile-cpu-lower-memref-bitcast", "mlir::ModuleOp"> {
  let summary = "Lowers xtile.memref_bitcast to LLVM.";

  let description = [{
    The memref descriptor holds an opaque pointer, so its LLVM type does not
    depend on the element type and the op lowers to nothing.
  }];

  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def UnpackSubByteVectorWritePass : Pass<"xtile-cpu-unpack-sub-byte-vector-write"> {
  let summary = "Unpacks sub-byte vector writes.";

  let description = [{
    The default lowering of vector.transfer_write of subbyte types reflects that
    of llvm vector store instructions, i.e it writes the packed data directly.
    The semantics we use in the CPU lowering is that it will unpack the data
    into a series of bytes and store them separately. This pass implements that
    by using unrolling the transfer_write into individual elements.
  }];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
  ];
}

def InitializeAllocsPass : Pass<"xtile-cpu-initialize-allocs"> {
  let summary = "Initializes allocations.";

  let description = [{
    We don't currently have the infrastructure to instrument JIT kernels fully
    with msan, therefore to avoid false positives we initialize all allocated
    memory to set values.

    Integer values will be initialized with all high bits and floats with NaNs.

    This pass will also add a call to __msan_unpoison for each allocation as
    optimizations can remove the set values.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::LLVM::LLVMDialect",
  ];
}

def HoistAllocaPass : Pass<"xtile-cpu-hoist-alloca", "mlir::func::FuncOp"> {
  let summary = "Hoist memref.alloca ops to the entry block of the function.";
  let dependentDialects = [
    "::mlir::memref::MemRefDialect"
  ];
}

def VectorizeXTilePass : Pass<"xtile-cpu-vectorize-xtile", "mlir::ModuleOp"> {
  let summary = "Vectorize arith and math ops on tensors";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::vector::VectorDialect",
    "mlir::tensor::TensorDialect",
    "xla::XlaDialect",
  ];
}

