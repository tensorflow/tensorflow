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

def XTileToVectorPass : Pass<"xtile-cpu-xtile-to-vector", "mlir::ModuleOp"> {
  let summary = "Lowering xtile ops to vector ops";

  let constructor = "CreateXTileToVectorPass()";

  let dependentDialects = [
    "mlir::vector::VectorDialect",
    "xla::xtile::XTileDialect",
  ];
}

def LowerXTileEntryPass : Pass<"xtile-cpu-lower-xtile-entry", "mlir::ModuleOp"> {
  let summary = "Lowers the entry function into the form required by the CPU runtime";

  let constructor = "CreateLowerXTileEntryPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "xla::cpu::XlaCpuDialect",
    "xla::xtile::XTileDialect"
  ];
}

def ShloToVectorPass : Pass<"xtile-cpu-shlo-to-vector", "mlir::ModuleOp"> {
  let summary = "Lowering satble hlo ops to vector ops";

  let constructor = "CreateShloToVectorPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
  ];
}

def ElementalTensorToVectorPass : Pass<"xtile-cpu-elemental-tensor-to-vector",
                                       "mlir::ModuleOp"> {
  let summary = "Lowering arith & math ops with tensor types to vector types";

  let constructor = "CreateElementalTensorToVectorPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
  ];
}

def TensorOpsToVectorPass : Pass<"xtile-cpu-tensor-ops-to-vector",
                                 "mlir::ModuleOp"> {
  let summary = "Lowering tensor dialect ops to vector ops";

  let constructor = "CreateTensorOpsToVectorPass()";

  let dependentDialects = [
    "mlir::vector::VectorDialect",
  ];
}

def RewriteDynamicVectorExtractPass : Pass<"xtile-cpu-rewrite-dynamic-vector-extract",
                                 "mlir::ModuleOp"> {
  let summary = "Rewrite vector.extracts with dynamic indices.";

  let constructor = "CreateRewriteDynamicVectorExtractPass()";

  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
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
