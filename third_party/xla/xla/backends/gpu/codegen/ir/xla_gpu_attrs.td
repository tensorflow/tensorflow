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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS
#define XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "xla/backends/gpu/codegen/ir/xla_gpu_dialect.td"
include "xla/codegen/ir/xla_attrs.td"

class XLAGPU_Attr<string name, list<Trait> traits = []> :
      AttrDef<XlaGpuDialect, name, traits> {
}

//===----------------------------------------------------------------------===//
// Tensor layout attribute
//===----------------------------------------------------------------------===//

def XLAGPU_MemorySpace : I32EnumAttr<"MemorySpace",
    "element-wise op type", [
    I32EnumAttrCase<"kRegisters", 0, "registers">,
    I32EnumAttrCase<"kSharedMemory", 1, "shmem">
  ]> {
  let cppNamespace = "::xla::gpu";
  let genSpecializedAttr = 0;
}

def XLAGPU_MemorySpaceAttr : EnumAttr<
  XlaGpuDialect, XLAGPU_MemorySpace, "memory_space"> {
  let assemblyFormat = "`<` $value `>`";
}

def XLAGPU_LayoutAttr : XLAGPU_Attr<"Layout"> {
  let mnemonic = "layout";
  let summary = "Layout consists of a thread ID indexing map + memory space.";
  let description = [{
    This attribute is used as an encoding for RankedTensorType. It indicates in
    which memory space the tensor is stored and the access pattern from the
    warps/threads.
    ```mlir
    tensor<42xf32, #xla_gpu.layout<"shmem", (d0) -> (), domain: d0 in [0, 42]>>
    ```
  }];
  let parameters = (ins
      AttrParameter<"MemorySpaceAttr", "memory_space">:$memory_space,
      AttrParameter<"IndexingMapAttr", "thread_map">:$thread_map
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS
