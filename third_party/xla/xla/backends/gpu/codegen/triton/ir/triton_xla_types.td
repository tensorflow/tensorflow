/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_TYPES_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_TYPES_TD_

include "xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.td"
include "mlir/IR/BuiltinTypes.td" // ValueSemantics
include "mlir/IR/BuiltinTypeInterfaces.td"

// -----------------------------------------------------------------------------
// TiledTensorType
// -----------------------------------------------------------------------------

class TTXLA_Type<string name, string typeMnemonic, list<Trait> traits = []> :
      TypeDef<XlaTritonDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTXLA_TiledTensorType : TTXLA_Type<"TiledTensor", "tiled_tensor", [
    ValueSemantics]> {
  let summary = "A tile of a tensor.";
  let description = [{
    Usage:
      This type will typically be constructed via triton_xla.tile op. The intent
      is to capture tiling information and pass it along to other ops such as
      triton_xla.extract and triton_xla.insert. Refer to the ops for examples.

    Example:
      ```
      !triton_xla.tiled_tensor<16x64|320x512xbf16>
      ```
      this type represents a 16x64 tile of a 320x512 bf16 tensor.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile_shape,
    ArrayRefParameter<"int64_t">:$original_shape,
    "Type":$element_type
  );

 let builders = [
    TypeBuilder<(ins
      "RankedTensorType":$tile_type,
      "RankedTensorType":$original_type),
    [{
      assert(original_type.getElementType() == tile_type.getElementType());
      return $_get($_ctxt, tile_type.getShape(), original_type.getShape(),
                           original_type.getElementType());
    }]>
  ];
  let extraClassDeclaration = [{
    RankedTensorType getOriginalType() const {
      return RankedTensorType::get(getOriginalShape(), getElementType());
    }
    RankedTensorType getTileType() const {
      return RankedTensorType::get(getTileShape(), getElementType());
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_TYPES_TD_
