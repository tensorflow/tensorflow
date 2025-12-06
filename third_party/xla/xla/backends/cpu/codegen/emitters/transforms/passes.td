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
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "xla::XlaDialect",
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
    "mlir::math::MathDialect",
    "mlir::func::FuncDialect",
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

def AddLoopUnrollFlagsPass : Pass<"xla-cpu-add-loop-unroll-flags", "mlir::func::FuncOp"> {
  let summary = "Add loop unroll flags to large nested loops.";

  let description = [{
     Disable loop unrolling for loops that have nested loops larger that the given threshold.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
  ];

  let options = [
    Option<"max_nested_iterations_", "max_nested_iterations", "int32_t", "1",
           "The maximum number of iterations accessed in a nested loop before disabling unrolling">,
  ];

  let constructor = "CreateAddLoopUnrollFlagsPass()";
}


def PeelWorkgroupLoopPass : Pass<"xla-cpu-peel-workgroup-loop", "mlir::func::FuncOp"> {
  let summary = "Split the loop into two, creating one with no constraints.";

  let description = [{
    Having a loop with no constraints makes vectorization much harder, by
    creating two loops, the first having no constraints in the body helps with
    autovectorization significantly.
  }];

  let dependentDialects = [
    "xla::XlaDialect",
    "mlir::scf::SCFDialect",
  ];
  let constructor = "CreatePeelWorkgroupLoopPass()";
}

