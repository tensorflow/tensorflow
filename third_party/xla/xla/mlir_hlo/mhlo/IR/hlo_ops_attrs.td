/* Copyright 2021 The OpenXLA Authors.

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

def MHLO_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "parseDimSizes($_parser)";
  let printer = "printDimSizes($_printer, $_self)";
}

def MHLO_ScatterDimensionNumbers : AttrDef<MHLO_Dialect, "ScatterDimensionNumbers"> {
  let mnemonic = "scatter";
  let summary = "Attribute that models the dimension information for scatter";
  let parameters = (ins
      MHLO_Dims:$updateWindowDims,
      MHLO_Dims:$insertedWindowDims,
      MHLO_Dims:$inputBatchingDims,
      MHLO_Dims:$scatterIndicesBatchingDims,
      MHLO_Dims:$scatterDimsToOperandDims,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def MHLO_GatherDimensionNumbers : AttrDef<MHLO_Dialect, "GatherDimensionNumbers"> {
  let mnemonic = "gather";
  let summary = "Attribute that models the dimension information for gather";
  let parameters = (ins
      MHLO_Dims:$offsetDims,
      MHLO_Dims:$collapsedSliceDims,
      MHLO_Dims:$operandBatchingDims,
      MHLO_Dims:$startIndicesBatchingDims,
      MHLO_Dims:$startIndexMap,
      "int64_t":$indexVectorDim
  );
  let hasCustomAssemblyFormat = 1;
}

def MHLO_DotAlgorithmAttr : AttrDef<MHLO_Dialect, "DotAlgorithm"> {
  let mnemonic = "dot_algorithm";
  let summary = "Attribute that models the algorithm constraints to use for computing dot.";
  let parameters = (ins
      "Type":$lhsPrecisionType,
      "Type":$rhsPrecisionType,
      "Type":$accumulationType,
      "int64_t":$lhsComponentCount,
      "int64_t":$rhsComponentCount,
      "int64_t":$numPrimitiveOperations,
      "bool":$allowImpreciseAccumulation
  );
  let assemblyFormat = [{
    `<`
      `lhs_precision_type` `=` $lhsPrecisionType `,`
      `rhs_precision_type` `=` $rhsPrecisionType `,`
      `accumulation_type` `=` $accumulationType `,`
      `lhs_component_count` `=` $lhsComponentCount `,`
      `rhs_component_count` `=` $rhsComponentCount `,`
      `num_primitive_operations` `=` $numPrimitiveOperations `,`
      `allow_imprecise_accumulation` `=` $allowImpreciseAccumulation
    `>`
  }];
  let genVerifyDecl = 1;
}

def MHLO_DotDimensionNumbers : AttrDef<MHLO_Dialect, "DotDimensionNumbers"> {
  let mnemonic = "dot";
  let summary = "Attribute that models the dimension information for dot.";
  let parameters = (ins
      MHLO_Dims:$lhsBatchingDimensions,
      MHLO_Dims:$rhsBatchingDimensions,
      MHLO_Dims:$lhsContractingDimensions,
      MHLO_Dims:$rhsContractingDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def MHLO_ConvDimensionNumbers : AttrDef<MHLO_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::mhlo";
  let mnemonic = "conv";
  let summary = "Structure of dimension information for conv op";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    MHLO_Dims:$inputSpatialDimensions,
    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    MHLO_Dims:$kernelSpatialDimensions,
    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    MHLO_Dims:$outputSpatialDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

def MHLO_OutputOperandAlias : AttrDef<MHLO_Dialect, "OutputOperandAlias"> {
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
    MHLO_Dims:$outputTupleIndices,
    "int64_t":$operandIndex,
    MHLO_Dims:$operandTupleIndices
  );
  let assemblyFormat = [{
    `<` `output_tuple_indices` `=` $outputTupleIndices `,`
        `operand_index` `=` $operandIndex `,`
        `operand_tuple_indices` `=` $operandTupleIndices `>`
  }];
}

def MHLO_ArgResultAlias : AttrDef<MHLO_Dialect, "ArgResultAlias"> {
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
    MHLO_Dims:$argTupleIndices,
    "int64_t":$resultIndex,
    MHLO_Dims:$resultTupleIndices,
    "bool":$isMustAlias
  );
  let hasCustomAssemblyFormat = 1;
}

// Represents a unique identifier for each Send/Recv instruction pair or
// optionally for collective instructions (AllToAll, AllReduce,
// CollectiveBroadcast, and CollectivePermute). Non-positive channel_id
// handle is equivalent to no channel id.
def MHLO_ChannelHandle : AttrDef<MHLO_Dialect, "ChannelHandle"> {
  let mnemonic = "channel_handle";
  let parameters = (ins "int64_t":$handle, "int64_t":$type);
  let summary = "two 64-bit integers 'handle' and 'type'";
  let assemblyFormat = "`<` struct(params) `>`";
}

def MHLO_CrossProgramPrefetch : AttrDef<MHLO_Dialect, "CrossProgramPrefetch"> {
  let mnemonic = "cross_program_prefetch";
  let parameters = (ins "int64_t":$parameter, MHLO_Dims:$indices, "std::optional<int64_t>":$offset);
  let summary = "Argument that is prefetched from another program";
  let description = [{
    This attribute captures an argument that is prefetched from another program.
    For a given `CrossProgramPrefetchAttr`, `parameter` tells us which argument
    of the `main` function of the module is prefetched, and `indices` is a shape
    index telling us what subshape of that argument is prefetched.

    A shape has a subshape iff it is a tuple. In that case, the subshape of
    the tuple by `indices` is the shape achieved after indexing by each
    element of `indices` in turn. For example, the [1,0] subshape of
    `tuple<tuple<token, token>, tuple<tensor<i32>, token>>` is `tensor<i32>`.

    An empty value for `indices` means the whole shape is prefetched.

    For example,

    ```mlir
    module attributes { mhlo.cross_program_prefetch = [ #mhlo.cross_program_prefetch< parameter = 0, indices = [0]> ]} {
      func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
        %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch}
        return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
      }
      func.func @main(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
        %1 = "mhlo.async_start"(%arg0) {called_computation=@copy}
        %2 = "mhlo.async_done"(%1) {called_computation=@copy}
        return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
      }
    }
    ```

    The `parameter = 0` tells us that the async copy of the `0`th parameter is
    a `cross_program_prefetch`, while the `index` of `[0]` tells us that the
    `0`th element of the tuple is prefetched while the other element of the
    tuple is not.
  }];
  let assemblyFormat = "`<` struct(params) `>`";
}

// Note: This is an experimental attribute and shouldn't be relied upon for
// production.
def MHLO_TypeExtensions : AttrDef<MHLO_Dialect, "TypeExtensions", [
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

  let assemblyFormat = "`<` `bounds` `=` ` ` custom<DimSizes>($bounds) `>`";
}

// A layout attribute (1D tensor of index type)
def MHLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{mlir::cast<::mlir::DenseIntElementsAttr>($_self).getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def MHLO_ArrayOfLayoutAttr : TypedArrayAttrBase<MHLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def MHLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

// Sparsity descriptor attribute
def MHLO_SparsityDescriptor : AttrDef<MHLO_Dialect, "SparsityDescriptor"> {
  let mnemonic = "sparsity";
  let summary = "Describes structured (N:M) sparsity configuration";
  let description = [{
    This attribute is defined for a sparse dot operation with a structured
    sparse input tensor. With (N=2,M=4), every 4 consecutive logical elements
    have exactly 2 non-zero physical elements in the input tensor.

    $dimension defines the index of the contracting dimension that is sparse
    (it has to be the most minor dimension). The additional metadata operand
    in the sparse dot operation defines which logical elements are zeroed out.
  }];
  let parameters = (ins
    "int64_t":$dimension,
    "int64_t":$n,
    "int64_t":$m);
  let assemblyFormat = "`<` struct(params) `>`";
}

def MHLO_RaggedDotDimensionNumbers : AttrDef<MHLO_Dialect, "RaggedDotDimensionNumbers"> {
  let mnemonic = "ragged_dot";
  let summary = "Attribute that models the dimension information for ragged dot.";
  let parameters = (ins
      MHLO_DotDimensionNumbers:$dotDimensionNumbers,
      MHLO_Dims:$lhsRaggedDimensions,
      MHLO_Dims:$rhsGroupDimensions
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Common convolution attributes
//===----------------------------------------------------------------------===//

def MHLO_BoolElementsAttr :
    ElementsAttrBase<
      And<[CPred<"mlir::isa<::mlir::DenseIntOrFPElementsAttr>($_self)">,
           CPred<"mlir::cast<::mlir::DenseIntOrFPElementsAttr>($_self).getType().getElementType().isInteger(1)">]>,
      "constant boolean vector/tensor attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_ATTRS
