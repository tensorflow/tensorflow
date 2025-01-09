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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_TYPES
#define XLA_SERVICE_GPU_FUSIONS_MLIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "xla/backends/gpu/codegen/ir/xla_gpu_dialect.td"
include "xla/codegen/ir/xla_attrs.td"

class XLAGPU_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<XlaGpuDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def XLAGPU_IndexedVectorType : XLAGPU_Type<"IndexedVector", "indexed_vector",
    [ShapedTypeInterface, ValueSemantics]> {
  let summary = "Vector type with a specified layout";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    XLA_IndexingMapAttr:$indexing_map_attr
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    IndexedVectorType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                                mlir::Type elementType) const {
        return IndexedVectorType::get(getContext(), shape.value_or(getShape()),
                                      elementType, getIndexingMapAttr());
    }

    bool hasRank() const { return true; }
  }];
}

#endif // XLA_SERVICE_GPU_FUSIONS_MLIR_TYPES