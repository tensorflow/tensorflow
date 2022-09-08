/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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


#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ATTRS
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"

def HLODim : ArrayRefParameter<"int64_t", "Dimension">;

def ScatterDimensionNumbers : AttrDef<HLO_Dialect, "ScatterDimensionNumbers"> {
  let mnemonic = "scatter";
  let summary = "Attribute that models the dimension information for scatter";
  let parameters = (ins
      HLODim:$updateWindowDims,
      HLODim:$insertedWindowDims,
      HLODim:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def GatherDimensionNumbers : AttrDef<HLO_Dialect, "GatherDimensionNumbers"> {
  let mnemonic = "gather";
  let summary = "Attribute that models the dimension information for gather";
  let parameters = (ins
      HLODim:$offsetDims,
      HLODim:$collapsedSliceDims,
      HLODim:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def DotDimensionNumbers : AttrDef<HLO_Dialect, "DotDimensionNumbers"> {
  let mnemonic = "dot";
  let summary = "Attribute that models the dimension information for dot.";
  let parameters = (ins
      HLODim:$lhsBatchingDimensions,
      HLODim:$rhsBatchingDimensions,
      HLODim:$lhsContractingDimensions,
      HLODim:$rhsContractingDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def ConvDimensionNumbers : AttrDef<HLO_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::mhlo";
  let mnemonic = "conv";
  let summary = "Structure of dimension information for conv op";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    HLODim:$inputSpatialDimensions,
    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    HLODim:$kernelSpatialDimensions,
    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    HLODim:$outputSpatialDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def OutputOperandAlias : AttrDef<HLO_Dialect, "OutputOperandAlias"> {
  let cppNamespace = "::mlir::mhlo";
  let mnemonic = "output_operand_alias";
  let summary =
    "Attribute that models the alias relationship of output and operand of a CustomCall op";
  let description = [{
    This attribute captures the alias relationship of the output to one of the
    operands for a CustomCall op, denoted by `operand_index`. The
    `output_tuple_indices` and `operand_tuple_indices` are used to index into
    output and operand types. These indices lists are empty if the corresponding
    types are not tuple types, and can be arbitrarily long in case of
    arbitrarily nested tuple types.

    See https://www.tensorflow.org/xla/aliasing.

    Example when used as array with in mhlo.custom-call:

    ```mlir
    %0 = "mhlo.custom_call"(%arg0, %arg1) {
      // other attributes
      output_operand_alias = [
        #mhlo.output_operand_alias<output_tuple_indices = [0],
                                   operand_index = 0,
                                   operand_tuple_indices = [1]>
      ]
    } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>

    The output and the 0th operand are both tuples. The aliasing shows the
    relationship between the 0th element in output tuple with the 1st element in
    the 0th operand. And both of them are of the same type: tensor<2x3xf32>.
    ```
  }];
  let parameters = (ins
    HLODim:$outputTupleIndices,
    "int64_t":$operandIndex,
    HLODim:$operandTupleIndices
  );
  let assemblyFormat = "`<` "
                       "`output_tuple_indices` `=` `[` $outputTupleIndices `]` `,`"
                       "`operand_index` `=` $operandIndex `,`"
                       "`operand_tuple_indices` `=` `[` $operandTupleIndices `]`"
                       "`>`";
}

def ArgResultAlias : AttrDef<HLO_Dialect, "ArgResultAlias"> {
  let cppNamespace = "::mlir::mhlo";
  let mnemonic = "result_alias";
  let summary =
    "Attribute that models the alias relationship of entry function argument";
  let description = [{
    This attribute captures the alias relationship of an MHLO main function
    argument to one of the results, denoted by `resultIndex`. The
    `argTupleIndices` and `resultTupleIndices` are used to index into nested
    tuples in operand and result respectively. If `isMustAlias` is true then the
    operand-result pair must alias.

    This is meant to be used as an attribute on a function argument in MHLO.
    For example, in the following code it expresses that `%arg1` may alias 0-th
    result.

    ```mlir
    func @main(%arg0: tensor<2xf32>, %arg1: tensor<3xf32> {mhlo.result_alias =
        mhlo.result_alias<result_index = [2], ...>}
      ) -> tensor<2xf32>, tensor<3xf32> {
      // function body ...
    }
    ```
  }];
  let parameters = (ins
    HLODim:$argTupleIndices,
    "int64_t":$resultIndex,
    HLODim:$resultTupleIndices,
    "bool":$isMustAlias
  );
  let hasCustomAssemblyFormat = 1;
}

// Represents a unique identifier for each Send/Recv instruction pair or
// optionally for collective instructions (AllReduce, CollectivePermute,
// AllToAll). Non-positive channel_id handle is equivalent to no channel id.
def ChannelHandle : AttrDef<HLO_Dialect, "ChannelHandle"> {
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let summary = "two 64-bit integers 'handle' and 'type'";
  let assemblyFormat = "`<` struct(params) `>`";
}

// Note: This is an experimental attribute and shouldn't be relied upon for
// production.
def TypeExtensions : AttrDef<HLO_Dialect, "TypeExtensions", [
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>,
    DeclareAttrInterfaceMethods<HLO_BoundedAttrInterface>]> {
  let mnemonic = "type_extensions";

  // TODO(b/238903065): Move sparsity related info here from the standalone
  // attribute. That will allow composition of bounds and sparsity info.
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$bounds
  );

  let summary = "Attribute that extends tensor type with MHLO type properties.";

  let description = [{
    This attribute is used to extend MLIR tensor type with MHLO tensor specific
    properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    See `HLO_BoundedAttrInterface` for documentation for `bounds`.
  }];
  let assemblyFormat = "`<` `bounds` `=` `[` $bounds `]` `>`";
}

// A layout attribute (1D tensor of index type)
def HLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{$_self.cast<::mlir::DenseIntElementsAttr>().getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def HLO_ArrayOfLayoutAttr : TypedArrayAttrBase<HLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def HLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

//===----------------------------------------------------------------------===//
// Common convolution attributes
//===----------------------------------------------------------------------===//

def BoolElementsAttr :
    ElementsAttrBase<
      And<[CPred<"$_self.isa<::mlir::DenseIntOrFPElementsAttr>()">,
           CPred<"$_self.cast<::mlir::DenseIntOrFPElementsAttr>().getType().getElementType().isInteger(1)">]>,
      "constant boolean vector/tensor attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def ConvolutionAttributes {
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
    OptionalAttr<BoolElementsAttr>:$window_reversal,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    HLO_PrecisionConfigAttr:$precision_config
  );
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ATTRS
