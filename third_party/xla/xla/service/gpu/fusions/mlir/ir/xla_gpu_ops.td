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


#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_OPS
#define XLA_SERVICE_GPU_FUSIONS_MLIR_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def XlaGpuDialect : Dialect {
  let name = "xla_gpu";

  let description = [{
    This dialect contains ops required for lowering HLO to LLVM.
  }];

  let cppNamespace = "::xla::gpu";
}

class XLAGPU_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaGpuDialect, mnemonic, traits> {
}

def XLAGPU_AllocateSharedOp : XLAGPU_Op<"allocate_shared"> {
  let summary = "Allocates a shared memory tile.";

  let description = [{
    Allocates a shared memory tensor. The tensor is shared among all threads in
    a block.

    ```mlir
    %shared = xla_gpu.allocate_shared : tensor<32x32xf32>
    ```
  }];

  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def XLAGPU_SyncThreadsOp : XLAGPU_Op<"sync_threads"> {
  let summary = "Synchronizes threads.";

  let description = [{
    Synchronizes threads, taking any number of distributed tensors and returning
    the synchronized state.
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$operands);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS
