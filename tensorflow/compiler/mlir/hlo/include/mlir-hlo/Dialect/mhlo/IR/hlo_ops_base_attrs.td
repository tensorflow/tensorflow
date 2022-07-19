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


#ifndef HLO_OPS_BASE_ATTRS
#define HLO_OPS_BASE_ATTRS

include "mlir/IR/OpBase.td"

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
def TypeExtensions : AttrDef<HLO_Dialect, "TypeExtensions"> {
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

    Bounds parameter specifies the maximum runtime dimension size for
    dimensions with unknown size. If the bound are not known for any of the
    dimensions, it is set to ShapedType::kDynamicSize. Bound is undefined for
    static dimensions. Bounds array has the same size as rank of the tensor.
  }];
  let assemblyFormat = "`<` `bounds` `=` `[` $bounds `]` `>`";

  // TODO(hinsu): Add verification for bounds that it has the same size as rank
  // of the tensor and static dimensions don't have bounds.
}

#endif // HLO_OPS_BASE_ATTRS
