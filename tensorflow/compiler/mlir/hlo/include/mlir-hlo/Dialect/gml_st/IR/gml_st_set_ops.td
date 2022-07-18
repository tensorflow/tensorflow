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

// This is the operation definition file for ST ops.

#ifndef GML_ST_SET_OPS
#define GML_ST_SET_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"
include "mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.td"

// Base class of all subset types.
class GMLST_Set<string name> : TypeDef<GmlSt_Dialect, name> { }

def GMLST_TileType : GMLST_Set<"Tile"> {
  let mnemonic = "tile";
  let summary = "Type that represents a tile of an index space.";
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<ShapeTypeDimensionsList>($shape) `>`";
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
  }];
}

def GMLST_PointType : GMLST_Set<"Point">,
    BuildableType<"$_builder.getType<::mlir::gml_st::PointType>()"> {
  let mnemonic = "point";
  let summary = "Type that represents a point of an index space.";
  let assemblyFormat = "";
}

// Whether a type is a subset type.
def IsSetTypePred : Or<[GMLST_TileType.predicate, GMLST_PointType.predicate]>;

def AnySet : Type<IsSetTypePred, "subset type">;

def GMLST_SpaceOp : GMLST_Op<"space", [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<Index>:$dynamic_sizes,
                       I64ArrayAttr:$static_sizes);
  let results = (outs GMLST_TileType:$result);
  let assemblyFormat = [{
    custom<OperandsOrIntegersSizesList>($dynamic_sizes, $static_sizes)
    attr-dict `:` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    unsigned getNumDynamicEntriesUpToIdx(unsigned idx);
    mlir::Value getDynamicSize(unsigned idx);
  }];
  let hasVerifier = 1;
}

def GMLST_PointOp : GMLST_Op<"point", [
    NoSideEffect,
    DeclareOpInterfaceMethods<ComposeSetInterface>]> {
  let arguments = (ins AnySet:$superset,
                       Variadic<Index>:$dynamic_indices,
                       I64ArrayAttr:$static_indices);
  let results = (outs GMLST_PointType:$result);
  let assemblyFormat = [{
    $superset
    custom<OperandsOrIntegersOffsetsOrStridesList>($dynamic_indices, $static_indices)
    attr-dict `:` qualified(type($superset)) `to` qualified(type($result))
  }];
  let extraClassDeclaration = [{
    unsigned getRank() { return static_indices().size(); }
  }];
  let hasVerifier = 1;
}

def GMLST_TileOp : GMLST_Op<"tile", [
    NoSideEffect,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ComposeSetInterface>]> {
  let arguments = (ins GMLST_TileType:$superset,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       I64ArrayAttr:$static_offsets,
                       I64ArrayAttr:$static_sizes,
                       I64ArrayAttr:$static_strides);
  let results = (outs GMLST_TileType:$result);
  let assemblyFormat = [{
    $superset
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` qualified(type($superset)) `to` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = superset().getType().cast<TileType>().getRank();
      return {rank, rank, rank};
    }
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
  let hasVerifier = 1;
}

def GMLST_DropDimsOp : GMLST_Op<"drop_dims", [
    NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Drops dimensions of a subset.
  }];
  let description = [{
    Selects specific dimensions of a subset and drops the remaining ones. The
    result is a tile or point in a space of smaller rank, which is implicitly
    defined by the root space.
  }];
  let arguments = (ins
    AnySet:$superset,
    DenseI64ArrayAttr:$remaining_dims);
  let results = (outs AnySet:$result);
  let assemblyFormat = [{
    $superset `,` $remaining_dims attr-dict `:`
    qualified(type($superset)) `to` qualified(type($result))
  }];
  let hasVerifier = 0;
}

def GMLST_TransposeTileOp : GMLST_Op<"transpose_tile", [
    NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ComposeSetInterface>]> {
  let summary = "Transposes a tile.";
  let description = [{
    Transposes the argument tile by applying the permutation to the dimensions,
    offsets and strides of the operand tile.
  }];
  let arguments = (ins
    GMLST_TileType:$superset,
    DenseI64ArrayAttr:$permutation);
  let results = (outs GMLST_TileType:$result);
  let assemblyFormat = [{
    $superset `,` $permutation attr-dict `:`
    qualified(type($superset)) `to` qualified(type($result))
  }];
  let hasVerifier = 1;
}

#endif // GML_ST_SET_OPS
