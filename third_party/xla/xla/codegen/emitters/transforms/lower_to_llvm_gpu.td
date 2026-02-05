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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TO_LLVM_GPU_TD_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TO_LLVM_GPU_TD_

include "mlir/Pass/PassBase.td"

def LowerToLLVMGPUPass :
   Pass<"xla-lower-to-llvm-gpu", "mlir::ModuleOp"> {
  let summary = "Lowers to LLVM (GPU Version).";

  let description = [{
    Lowers the rest to LLVM (GPU Version)
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
  ];

  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto.">,
  ];
  let constructor = "CreateLowerToLLVMGPUPass()";
}

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWER_TO_LLVM_GPU_TD_
