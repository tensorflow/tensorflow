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

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "xla::cpu::XlaCpuDialect",
  ];

  let constructor = "::xla::cpu::CreateLowerToLLVMPass()";
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
