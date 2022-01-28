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
}

def ArgResultAlias : AttrDef<HLO_Dialect, "ArgResultAlias"> {
  let cppNamespace = "::mlir::mhlo";
  let mnemonic = "result_alias";
  let summary = [{
    Attribute that models the alias relationship of an MHLO main function
    argument to one of the results (denoted by `resultIndex`). The
    `argTupleIndices` and `resultTupleIndices` are used to index into nested
    tuples in operand and result respectively. If `isMustAlias` is true then the
    operand-result pair must alias.

    This is meant to be used as an attribute on a function argument in MHLO.
    For example, in the following code it expresses that `%arg1` may alias 0-th
    result.

    ```
    func @main(%arg0: tensor<2xf32>,
               %arg1: tensor<3xf32> {mhlo.result_alias = mhlo.result_alias<result_index = [2], ...>}) -> tensor<2xf32>, tensor<3xf32> {
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
}

#endif // HLO_OPS_BASE_ATTRS
