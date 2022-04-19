/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

def ConvertLmhloToGpuPass : Pass<"lmhlo-to-gpu", "mlir::ModuleOp"> {
  let summary = "Convert lmhlo ops to gpu dialect.";
  let description = [{
    Some lmhlo ops are converted to gpu dialect. Others are moved inside
    tfrt_gpu_conversion.execute ops and converted to tfrt_gpu dialect.
  }];
  let constructor = "createConvertLmhloToGpuPass()";
}

def ConvertLmhloToGpuBinaryPass :
    Pass<"lmhlo-to-gpu-binary", "mlir::ModuleOp"> {
  let summary = "Compile lmhlo ops to gpu device code.";
  let description = [{
    Replace device code ops (only lmhlo.fusion at the moment) with 
    gpu.launch_func referring to a gpu.module with annotated binary gpu device
    code.
  }];
  let constructor = "createConvertLmhloToGpuBinaryPass()";
}
