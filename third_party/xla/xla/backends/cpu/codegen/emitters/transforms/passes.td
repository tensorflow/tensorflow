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

def LowerToLLVMPass : Pass<"xla-cpu-lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowering from tensors + xla_cpu to LLVM";

  let description = [{
      This is a pass for lowering tensors and xla_cpu operations to LLVM
      dialect and pointers.
  }];

  let options = [
    Option<"prefer_vector_width_", "prefer_vector_width", "int32_t", "256",
           "prefer-vector-width value to set on entry function">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::tensor::TensorDialect",
    "xla::cpu::XlaCpuDialect",
    "xla::XlaDialect",
  ];
}

def LowerXlaSharedPass :
   Pass<"xla-cpu-lower-xla-shared", "mlir::func::FuncOp"> {
  let summary = "Lowers XLA shared dialect ops to XLA cpu compatable ops.";

  let description = [{
    Lowers XLA shared dialect ops to XLA cpu compatable ops.
  }];

  let dependentDialects = [
    "xla::XlaDialect",
    "mlir::scf::SCFDialect",
  ];

  let constructor = "CreateLowerXlaSharedPass()";
}

def ExpandFloatOpsPass : Pass<"xla-cpu-expand-float-ops", "mlir::func::FuncOp"> {
  let summary = "Expands float ops that are not natively supported on cpu.";

  let description = [{
     Not all float ops are natively supported, either because they don't exist
     in hardware or they are too inaccurate.

     This pass replaces these ops with alternative implementations.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];

  let constructor = "CreateExpandFloatOpsPass()";
}

def AddReductionFastMathFlagsPass : Pass<"xla-cpu-add-reduction-fast-math-flags", "mlir::ModuleOp"> {
  let summary = "Add fast math flags to reduction functions.";

  let description = [{
     It has been shown that the XLA:CPU compiler benefits significantly from
     having fast-math-flags on reduction functions, this pass adds them.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];

  let constructor = "CreateAddReductionFastMathFlagsPass()";
}
