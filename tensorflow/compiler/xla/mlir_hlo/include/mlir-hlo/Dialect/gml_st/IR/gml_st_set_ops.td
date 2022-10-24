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

// This is the operation definition file for GmlSt Subset ops.

#ifndef GML_ST_SET_OPS
#define GML_ST_SET_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"

// Base class of all subset types.
class GMLST_Set<string name> : TypeDef<GmlSt_Dialect, name> { }

def GMLST_TileType : GMLST_Set<"Tile"> {
  let mnemonic = "tile";
  let summary = "Type that represents a tile of an index space.";
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<ShapeTypeDimensionsList>($shape) `>`";
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumElements() const {
      return ShapedType::getNumElements(getShape());
    }
  }];
}

def AnySet : Type<Or<[GMLST_TileType.predicate]>, "subset type">;

def GMLST_TileOp : GMLST_Op<"tile", [
    Pure,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       I64ArrayAttr:$static_offsets,
                       I64ArrayAttr:$static_sizes,
                       I64ArrayAttr:$static_strides);
  let results = (outs GMLST_TileType:$result);
  let assemblyFormat = [{
    custom<DynamicIndexList>($offsets, $static_offsets,
                               "ShapedType::kDynamicStrideOrOffset")
    custom<DynamicIndexList>($sizes, $static_sizes,
                               "ShapedType::kDynamicSize")
    custom<DynamicIndexList>($strides, $static_strides,
                               "ShapedType::kDynamicStrideOrOffset")
    attr-dict `:` qualified(type($result))
  }];
  let builders = [
   OpBuilder<(ins "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];
  let extraClassDeclaration = [{
    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getType().cast<TileType>().getRank();
      return {rank, rank, rank};
    }
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 0; }
  }];
  let hasVerifier = 1;
}

#endif // GML_ST_SET_OPS
