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

def ConvertLmhloConstantToArgPass :
    Pass<"lmhlo-constant-to-arg", "mlir::ModuleOp"> {
  let summary = "Converts memref.get_global corresponding to lmhlo constants";

  let description = [{
    Replaces `memref.get_global` operations corresponding to the lmhlo constant
    arguments (arguments marked with `lmhlo.constant_name` attribute) to use
    the constant arguments directly.

    Once we used global constants for constant folding, we no longer need to
    keep them in the module, because they'll be in the binary constant section
    on the host, and we need them on the device.
  }];

  let constructor = "createConvertLmhloConstantToArgPass()";
}

def ConvertGpuToJitRtPass :
    Pass<"gpu-to-jitrt", "mlir::ModuleOp"> {
  let summary = "Converts gpu operations to JitRt Gpu custom calls.";

  let description = [{
    Converts gpu operations (function launch, memcpy) to the JitRt custom calls,
    that will be handled at runtime by the XLA JitRt custom call handlers.
  }];

  let constructor = "createConvertGpuToJitRtPass()";
}

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
