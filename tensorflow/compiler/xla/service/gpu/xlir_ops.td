// Copyright 2021 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- xlir_ops.td --------------------------------------------------------===//
//
// Operation definitions for XLIR functionality.
//
//===----------------------------------------------------------------------===//

#ifdef XLIR_OPS
#else
#define XLIR_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tfrt/gpu/kernels/gpu_ops_base.td"
include "tfrt/tfrt_op_base.td"

// "xlir" dialect
def XLIR_Dialect : Dialect {
  let name = "xlir";

  let description = [{
    The XLIR dialect.

    This dialect contains common GPU operations for CUDA and ROCm.
  }];

  let cppNamespace = "xla::gpu";
}

// Base class for XLIR dialect ops.
class XLIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<XLIR_Dialect, mnemonic, traits> {
  let summary = !strconcat("xlir.", mnemonic, " operation");
  let assemblyFormat = "operands attr-dict";
}

def XLIR_CclCreateOp : XLIR_Op<"ccl.create"> {
  let description = [{
    xlir.ccl.create returns a ccl.handle, which can be populated with individual
    collective ops (e.g., using tfrt_gpu.ccl.all_reduce) to be executed as a
    single fused operation.
  }];
  let arguments = (ins GPU_ContextType);
  let results = (outs GPU_CclHandleType);
}

#endif  // XLIR_OPS
