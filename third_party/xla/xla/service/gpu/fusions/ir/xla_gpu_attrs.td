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
include "xla/service/gpu/fusions/ir/xla_gpu_dialect.td"

class XLAGPU_Attr<string name, list<Trait> traits = []> :
      AttrDef<XlaGpuDialect, name, traits> {
}

def XLAGPU_AffineMapParameter :
    AttrOrTypeParameter<"::mlir::AffineMap", ""> {
}

def XLAGPU_IndexingMapVariableParameter
  : ArrayRefParameter<"::xla::gpu::IndexingMap::Variable",
                      "IndexingMapVariableArray"> {
}

def XLAGPU_ConstraintsParameter :
    ArrayRefParameter<"::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>",
                      "ContraintsArray"> {
}

def XLAGPU_IndexingMapAttr : XLAGPU_Attr<"IndexingMap"> {
  let summary = "An Attribute representing an indexing map.";
  let mnemonic = "indexing_map";
  let description = [{This attribute stores an indexing map. See
    https://openxla.org/xla/indexing for more details.
  }];
  let parameters = (ins XLAGPU_AffineMapParameter:$map,
                        XLAGPU_IndexingMapVariableParameter:$dim_vars,
                        XLAGPU_IndexingMapVariableParameter:$range_vars,
                        XLAGPU_ConstraintsParameter:$constraints);
  let hasCustomAssemblyFormat = 1;
  let builders =  [
    AttrBuilder<(ins "const ::xla::gpu::IndexingMap&":$indexing_map)>,
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // Returns the indexing map constructed from IndexingMapAttr.
    xla::gpu::IndexingMap getIndexingMap() const;

    // Returns the number of indexing map results.
    int64_t getNumResults() const;
  }];
}

def XLAGPU_LaunchGridAttr : XLAGPU_Attr<"LaunchGrid"> {
  let summary = "An attribute representing a launch grid.";
  let mnemonic = "launch_grid";
  let description = [{This attribute stores a launch grid.}];
  let parameters = (ins ArrayRefParameter<"int64_t", "">:$block_counts,
                        ArrayRefParameter<"int64_t", "">:$thread_counts);
  let assemblyFormat = [{
    `<`
       `block_counts` `=` `[` $block_counts `]` `,`
       `thread_counts` `=` `[` $thread_counts `]`
    `>`
  }];
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
