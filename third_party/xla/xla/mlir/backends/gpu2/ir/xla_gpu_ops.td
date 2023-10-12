/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifdef XLA_GPU_OPS
#else
#define XLA_GPU_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

include "xla/mlir/backends/gpu2/ir/xla_gpu_dialect.td"

//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

class XLAGPU_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaGpuDialect, mnemonic, traits> {
}

#endif // XLA_GPU_OPS