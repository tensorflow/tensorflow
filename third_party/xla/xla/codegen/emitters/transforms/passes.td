/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def EraseDeadFunctionsPass : Pass<"xla-erase-dead-functions", "mlir::ModuleOp"> {
  let summary = "Deletes unused functions";

  let description = [{
      Deletes functions that are not called.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "xla::XlaDialect",
  ];

  let constructor = "CreateEraseDeadFunctionsPass()";
}

def FlattenTensorsPass : Pass<"xla-flatten-tensors", "mlir::ModuleOp"> {
  let summary = "Flatten tensors.";

  let description = [{
    Linearizes all tensors loads and stores.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "xla::gpu::XlaGpuDialect",
    "xla::XlaDialect",
  ];
  let constructor = "CreateFlattenTensorsPass()";
}

def LowerTensorsPass : Pass<"xla-lower-tensors", "mlir::ModuleOp"> {
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
    "xla::XlaDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto.">,
    Option<"target_type_", "target_type", "std::string", /*default=*/"\"gpu\"",
           "Whether the pass targets a 'cpu' or 'gpu'. If 'cpu', gpu_device_info_ must be empty.">,
  ];
  let constructor = "CreateLowerTensorsPass()";
}

def ExpandFloatOpsPass : Pass<"xla-expand-float-ops", "mlir::ModuleOp"> {
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

  let constructor = "CreateExpandFloatOpsPass()";
}

def LowerToLLVMPass :
   Pass<"xla-lower-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers to LLVM.";

  let description = [{
    Lowers the rest to LLVM
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
  ];

  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto.">,
    Option<"target_type_", "target_type", "std::string", /*default=*/"\"gpu\"",
           "Whether the pass targets a 'cpu' or 'gpu'. If 'cpu', gpu_device_info_ must be empty.">,
  ];
  let constructor = "CreateLowerToLLVMPass()";
}

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD_
