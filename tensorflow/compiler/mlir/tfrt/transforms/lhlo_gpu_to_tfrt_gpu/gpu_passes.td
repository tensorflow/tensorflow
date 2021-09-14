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

def LmhloGpuAsyncConversionPass : FunctionPass<"lmhlo-to-gpu"> {
  let summary = "Convert lmhlo_gpu ops to tfrt_gpu dialect.";
  let description = [{
    Move lmhlo_gpu ops inside tfrt_gpu_conversion.execute ops and convert them
    to tfrt_gpu dialect.
  }];
  let constructor = "createLmhloGpuAsyncConversionPass()";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::tfrt::compiler::TFRTDialect",
    "::tfrt::gpu::GpuDialect",
    "::tfrt::gpu::conversion::GpuConversionDialect",
    "::xla::gpu::XlirDialect"
  ];
}
