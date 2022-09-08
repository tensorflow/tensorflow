/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

def ConvertLmhloGpuToJitRtPass :
    Pass<"lmhlo-gpu-to-jitrt", "mlir::ModuleOp"> {
  let summary = "Converts lmhlo_gpu ops to JitRt Gpu custom calls.";

  let description = [{
    Converts operations in the lmhlo_gpu dialect (e.g. lmhlo_gpu.gemm) to the
    JitRt custom calls (custom call handlers implemented as library calls, e.g.
    cuBLAS for gemm).
  }];

  let constructor = "createConvertLmhloGpuToJitRtPass()";
}
