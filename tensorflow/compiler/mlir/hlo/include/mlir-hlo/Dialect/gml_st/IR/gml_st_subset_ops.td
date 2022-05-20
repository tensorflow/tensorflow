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

#ifndef GML_ST_SUBSET_OPS
#define GML_ST_SUBSET_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"

// Base class of all subset types.
class GMLST_Subset<string name> : TypeDef<GmlSt_Dialect, name> { }

// TODO: Have a type trait for all subset.
def GMLST_TileType : GMLST_Subset<"Tile"> {
  let mnemonic = "tile";
  let summary = "Type that represents a tile of an index space.";
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<ShapeTypeDimensionsList>($shape) `>`";
}

def GMLST_PointType : GMLST_Subset<"Point"> {
  let mnemonic = "point";
  let summary = "Type that represents a point of an index space.";
  let assemblyFormat = "";
}

// Whether a type is a subset type.
def IsSubsetTypePred : CPred<"$_self.isa<::mlir::gml_st::PointType, ::mlir::gml_st::TileType>()">;

def AnySubset : Type<IsSubsetTypePred, "subset type">;

def GMLST_SpaceOp : GMLST_Op<"space", []> {
  let arguments = (ins Variadic<Index>:$shapes,
                       I64ArrayAttr:$static_shapes);
  let results = (outs AnySubset:$result);
  let assemblyFormat = [{
    custom<OperandsOrIntegersOffsetsOrStridesList>($shapes, $static_shapes)
    attr-dict `:` type($result)
  }];
  let hasVerifier = 0;
}

def GMLST_PointOp : GMLST_Op<"point", []> {
  let arguments = (ins AnySubset:$subset,
                       Variadic<Index>:$indices,
                       I64ArrayAttr:$static_indices);
  let results = (outs AnySubset:$result);
  let assemblyFormat = [{
    $subset
    custom<OperandsOrIntegersOffsetsOrStridesList>($indices, $static_indices)
    attr-dict `:` type($subset) `to` type($result)
  }];
  let hasVerifier = 0;
}

def GMLST_TileOp : GMLST_Op<"tile", [AttrSizedOperandSegments]> {
  let arguments = (ins AnySubset:$subset,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       I64ArrayAttr:$static_offsets,
                       I64ArrayAttr:$static_sizes,
                       I64ArrayAttr:$static_strides);
  let results = (outs AnySubset:$result);
  let assemblyFormat = [{
    $subset
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($subset) `to` type($result)
  }];
  let hasVerifier = 0;
}

def GMLST_OperandTileForDynamicBroadcastInDimOp : GMLST_Op<
    "operand_tile_for_dynamic_broadcast_in_dim", []> {
  let summary = [{
    Computes the minimal operand tile for `dynamic_broadcast_in_dim`
  }];
  let description = [{
    Based on a subset of interest for the result of a `dynamic_broadcast_in_dim`
    op, computes the minimal operand subset that is needed to perform the tiled
    broadcast.
  }];
  let arguments = (ins
    AnySubset:$dst_tile,
    AnySubset:$src_space,
    AnySubset:$dst_space,
    I64ElementsAttr:$broadcast_dimensions,
    OptionalAttr<I64ElementsAttr>:$known_expanding_dimensions,
    OptionalAttr<I64ElementsAttr>:$known_nonexpanding_dimensions);
  let results = (outs AnySubset:$result);
  let assemblyFormat = [{
    $dst_tile `,` $src_space `,` $dst_space `,`
    custom<I64ElementsAttr>($broadcast_dimensions)  attr-dict `:`
    type($dst_tile) `,` type($src_space) `,` type($dst_space) `->` type($result)
  }];
  let hasVerifier = 0;
}

#endif // GML_ST_SUBSET_OPS
