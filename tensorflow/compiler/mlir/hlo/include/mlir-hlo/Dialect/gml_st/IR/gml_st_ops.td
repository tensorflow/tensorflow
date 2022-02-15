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

#ifndef GML_ST_OPS
#define GML_ST_OPS

include "mlir/IR/OpBase.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class GMLST_Op<string mnemonic, list<Trait> traits> :
    Op<GmlSt_Dialect, mnemonic, traits> {
  let verifier = [{ return Verify(*this); }];
}

// Whether a type is a subset type.
def IsSubsetTypePred : CPred<"$_self.isa<::mlir::gml_st::PointType, ::mlir::gml_st::TileType>()">;

def AnySubset : Type<IsSubsetTypePred, "subset type">;

// TODO: Have a type trait for all subset.

// Base class of all subset types.
class GMLST_Subset<string name> : TypeDef<GmlSt_Dialect, name> { }

def GMLST_TileType : GMLST_Subset<"Tile"> {
  let mnemonic = "tile";
  let summary = "Type that represents a tile of an index space.";
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` $shape `>`";
}

def GMLST_PointType : GMLST_Subset<"Point"> {
  let mnemonic = "point";
  let summary = "Type that represents a point of an index space.";
  let assemblyFormat = "";
}

def GMLST_PointOp : GMLST_Op<"point", []> {
  let arguments = (ins Variadic<Index>:$offsets,
                       I64ArrayAttr:$static_offsets);
  let results = (outs AnySubset:$result);
  let assemblyFormat = [{
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    attr-dict `:` type($result)
  }];
  let verifier = ?;
}

def GMLST_TileOp : GMLST_Op<"tile", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       I64ArrayAttr:$static_offsets,
                       I64ArrayAttr:$static_sizes,
                       I64ArrayAttr:$static_strides);
  let results = (outs AnySubset:$result);
  let assemblyFormat = [{
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($result)
  }];
  let verifier = ?;
}

def GMLST_MaterializeOp : GMLST_Op<"materialize", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyShaped:$source, AnySubset:$subset);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source `at` $subset attr-dict `:` type($source) `at` type($subset)";
  let verifier = ?;
}

#endif // GML_ST_OPS

