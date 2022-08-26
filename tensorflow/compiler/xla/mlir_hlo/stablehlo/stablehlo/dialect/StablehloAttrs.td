/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_STABLEHLO_ATTRS
#define STABLEHLO_DIALECT_STABLEHLO_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"

def StableHLO_Dim : ArrayRefParameter<"int64_t", "Dimension">;

def StableHLO_ScatterDimensionNumbers : AttrDef<StableHLO_Dialect, "ScatterDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "scatter";
  let summary = "Attribute that models the dimension information for scatter";
  let parameters = (ins
      StableHLO_Dim:$updateWindowDims,
      StableHLO_Dim:$insertedWindowDims,
      StableHLO_Dim:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_GatherDimensionNumbers : AttrDef<StableHLO_Dialect, "GatherDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "gather";
  let summary = "Attribute that models the dimension information for gather";
  let parameters = (ins
      StableHLO_Dim:$offsetDims,
      StableHLO_Dim:$collapsedSliceDims,
      StableHLO_Dim:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_DotDimensionNumbers : AttrDef<StableHLO_Dialect, "DotDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "dot";
  let summary = "Attribute that models the dimension information for dot.";
  let parameters = (ins
      StableHLO_Dim:$lhsBatchingDimensions,
      StableHLO_Dim:$rhsBatchingDimensions,
      StableHLO_Dim:$lhsContractingDimensions,
      StableHLO_Dim:$rhsContractingDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_ConvDimensionNumbers : AttrDef<StableHLO_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "conv";
  let summary = "Structure of dimension information for conv op";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    StableHLO_Dim:$inputSpatialDimensions,
    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    StableHLO_Dim:$kernelSpatialDimensions,
    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    StableHLO_Dim:$outputSpatialDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_ArgResultAlias : AttrDef<StableHLO_Dialect, "ArgResultAlias"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "result_alias";
  let summary =
    "Attribute that models the alias relationship of entry function argument";
  let description = [{
    This attribute captures the alias relationship of a main function
    argument to one of the results, denoted by `resultIndex`. The
    `argTupleIndices` and `resultTupleIndices` are used to index into nested
    tuples in operand and result respectively. If `isMustAlias` is true then the
    operand-result pair must alias.

    This is meant to be used as an attribute on a function argument.
    For example, in the following code it expresses that `%arg1` may alias 0-th
    result.

    ```mlir
    func @main(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>
      {stablehlo.result_alias = stablehlo.result_alias<result_index = [2], ...>}
      ) -> tensor<2xf32>, tensor<3xf32> {
      // function body ...
    }
    ```
  }];
  let parameters = (ins
    StableHLO_Dim:$argTupleIndices,
    "int64_t":$resultIndex,
    StableHLO_Dim:$resultTupleIndices,
    "bool":$isMustAlias
  );
  let hasCustomAssemblyFormat = 1;
}

// Represents a unique identifier for each Send/Recv instruction pair or
// optionally for collective instructions (AllReduce, CollectivePermute,
// AllToAll). Non-positive channel_id handle is equivalent to no channel id.
def StableHLO_ChannelHandle : AttrDef<StableHLO_Dialect, "ChannelHandle"> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let summary = "two 64-bit integers 'handle' and 'type'";
  let assemblyFormat = "`<` struct(params) `>`";
}

// Note: This is an experimental attribute and shouldn't be relied upon for
// production.
def StableHLO_TypeExtensions : AttrDef<StableHLO_Dialect, "TypeExtensions", [
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>,
    DeclareAttrInterfaceMethods<HLO_BoundedAttrInterface>]> {
  let cppNamespace = "::mlir::stablehlo";
  let mnemonic = "type_extensions";

  // TODO(b/238903065): Move sparsity related info here from the standalone
  // attribute. That will allow composition of bounds and sparsity info.
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$bounds
  );

  let summary = "Attribute that extends tensor type with StableHLO type properties.";

  let description = [{
    This attribute is used to extend MLIR tensor type with StableHLO tensor
    specific properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    See `HLO_BoundedAttrInterface` for documentation for `bounds`.
  }];
  let assemblyFormat = "`<` `bounds` `=` `[` $bounds `]` `>`";
}

// A layout attribute (1D tensor of index type)
def StableHLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{$_self.cast<::mlir::DenseIntElementsAttr>().getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def StableHLO_ArrayOfLayoutAttr : TypedArrayAttrBase<StableHLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def StableHLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

def StableHLO_BoolElementsAttr :
    ElementsAttrBase<
      And<[CPred<"$_self.isa<::mlir::DenseIntOrFPElementsAttr>()">,
           CPred<"$_self.cast<::mlir::DenseIntOrFPElementsAttr>().getType().getElementType().isInteger(1)">]>,
      "constant boolean vector/tensor attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def StableHLO_ConvolutionAttributes {
  dag attributes = (ins
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: two zeros for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    // Default value: false for each of the spatial dimension.
    OptionalAttr<StableHLO_BoolElementsAttr>:$window_reversal,
    StableHLO_ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    StableHLO_PrecisionConfigAttr:$precision_config
  );
}

#endif // STABLEHLO_DIALECT_STABLEHLO_ATTRS
