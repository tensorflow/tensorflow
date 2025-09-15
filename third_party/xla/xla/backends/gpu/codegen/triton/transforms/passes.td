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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_

include "mlir/Pass/PassBase.td"

def TritonXLAExtractInsertToTritonPass : Pass<"triton-xla-extract-insert-to-triton", "mlir::ModuleOp"> {
  let summary = "Convert Triton XLA extract and insert ops to Triton ops.";
  let description = [{
    This pass converts `triton_xla.extract` and `triton_xla.insert` ops to
    Triton ops. It also rewrites `func` args to `tt.ptr` types and removes
    function return args.
  }];
  let options = [
    Option<"allow_tma_", "allow_tma", "bool", "false",
           "Whether to permit lowering to TMA.">,
  ];
  let dependentDialects = [
    "triton::TritonDialect",
    "::xla::XlaDialect"
  ];
  let constructor = "CreateTritonXLAExtractInsertToTritonPass()";
}

def TritonXLASqueezeDimsPass : Pass<"triton-xla-squeeze-dims", "mlir::ModuleOp"> {
  let summary = "Remove superfluous size-1 dimensions.";
  let description = [{
    This pass tries to remove size-1 dimensions from tensors.
  }];
  let dependentDialects = [
    "::mlir::triton::xla::XlaTritonDialect",
    "triton::TritonDialect"
  ];
  let options = [
    Option<"finalize_", "finalize", "bool", "true",
           "Convert squeeze_dims back to reshape. Disable for testing only.">,
  ];
  let constructor = "CreateTritonXLASqueezeDimsPass()";
}

def TritonXLAFoldTransposePass : Pass<"triton-xla-fold-transpose", "mlir::ModuleOp"> {
  let summary = "Folds transposes into loads.";
  let description = [{
    This pass tries to remove transposes by folding them into loads.
  }];
  let dependentDialects = [
    "::mlir::triton::xla::XlaTritonDialect"
  ];
  let constructor = "CreateTritonXLAFoldTransposePass()";
}

def GeneralizeKernelSignaturePass
    : Pass<"generalize-kernel-signature"> {
  let summary = "Rewrite kernels to use generic data pointer arguments.";
  let description = [{
    Rewrite signatures of kernel functions from global pointers to generic
    pointers and cast them to global ones within the kernel.
  }];
  let constructor = "CreateGeneralizeKernelSignaturePass()";
}

def ExtractTmaInfoPass
    : Pass<"extract-tma-info", "mlir::ModuleOp"> {
  let summary = "Extract TMA info during Triton lowering.";
  let description = [{
    Some information needed from TMA is created mid-pipeline by Triton. This
    pass extracts the information while it still exists and copies it into XLA's
    tma_descriptor attribute to be read at the LLVM IR level.
  }];
  let constructor = "CreateExtractTmaInfoPass()";
}

def LoadInt4RewritePass
    : Pass<"int4-to-packed-int4-rewrite", "mlir::ModuleOp"> {
  let summary = "Converts ops with int4 tensors to the ops with int4 packed to int8 tensors.";
  let description = [{
    This pass replaces the int4 tensors with the int4 packed to int8 tensor of
    the twice smaller size. It also replaces the plain ExtSIOp upcast to the
    int8 tensor with the unpack sequence.
  }];
  let dependentDialects = [
    "triton::TritonDialect"
  ];
  let options = [
    Option<"enable_bf16x2_", "enable_bf16x2", "bool", "true",
           "Whether to enable bf16x2.">,
  ];
}

def RoundF32ToTF32ForTf32DotRewritePass
    : Pass<"round-f32-to-tf32-for-tf32-dot-rewrite", "mlir::ModuleOp"> {
  let summary = "dot with tf32 algorithm requires explicit rounding.";
  let description = [{
    This pass adds explicit rounding from f32 to tf32 for the dot with tf32 algorithm.
    This is required because mma instruction does not have explicit rounding and
    by default does truncation. As a result, the dot with tf32 algorithm has too
    small precision. It is even less than for the dot with BF16 arguments.
  }];
  let constructor = "CreateRoundF32ToTF32ForTf32DotRewritePass()";
}

def TritonXLALowerGetTidPass
    : Pass<"triton-xla-get-tid", "mlir::ModuleOp"> {
  let summary = "Lower get_tid to the PTX intrinsic.";
  let description = [{
    This pass lowers get_tid to the PTX intrinsic by loading the thread ID from
    the register and returning it.
  }];
  let constructor = "CreateTritonXLALowerGetTidPass()";
}

def TritonXLALowerAtomicsPass
    : Pass<"triton-xla-atomics", "mlir::ModuleOp"> {
  let summary = "Lower triton_xla.atomic operations.";
  let description = [{
    This pass lowers atomic operations to the PTX intrinsic with the given
    memory semantics.
  }];
  let constructor = "CreateTritonXLALowerAtomicsPass()";
}

def TritonXLALowerBlockBarrierPass
    : Pass<"triton-xla-block-barrier", "mlir::ModuleOp"> {
  let summary = "Lower a block barrier to triton_xla/triton operations.";
  let description = [{
    This pass lowers a block barrier to Triton IR using atomic write and
    spin-wait operations.
  }];
  let constructor = "CreateTritonXLALowerBlockBarrierPass()";
}

def TritonXLAConvertUnsupportedTypesPass
    : Pass<"convert-unsupported-types", "mlir::ModuleOp"> {
  let summary = "Converts types unsupported by Triton into their supported equivalents.";
  let description = [{
    This pass converts types unsupported by Triton into their supported
    equivalents.

    This is essentially useful to support e.g. flavors of float8 types that
    Triton decides to represent as int8s from the very beginning, but that have
    both a proper representation in HLO and upstream MLIR.

    The important invariant of this pass is that the types it converts to have
    the same bitwidth as the original type.

    This pass may not be replaced by a simple `TypeConverter`, because the type
    replacement it performs do not preserve semantics in a vacuum (but may for
    specific ops, such as loads, stores, or bitcasts).
  }];
  let constructor = "CreateTritonXLAConvertUnsupportedTypesPass()";
}

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_
