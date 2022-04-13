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

// This is the struct definition file for TensorFlow.

#ifndef TFL_STRUCT
#define TFL_STRUCT

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"

def DimensionMetadataAttr : StructAttr<"DimensionMetadataAttr", TFL_Dialect, [
                            StructFieldAttr<"format", TFL_DimensionTypeAttr>,
                            StructFieldAttr<"dense_size", I32Attr>,
                            StructFieldAttr<"segments", I32ArrayAttr>,
                            StructFieldAttr<"indices", I32ArrayAttr>] > {
  let summary = "Dimension metadata.";
}

def DimensionMetadataArrayAttr : TypedArrayAttrBase<DimensionMetadataAttr,
    "Array of DimensionMetadata">{}

def SparsityParameterAttr : StructAttr<"SparsityParameterAttr", TFL_Dialect, [
                            StructFieldAttr<"traversal_order", I32ArrayAttr>,
                            StructFieldAttr<"block_map", I32ArrayAttr>,
                            StructFieldAttr<"dim_metadata", DimensionMetadataArrayAttr>]> {
  let summary = "Sparsity parameter.";
  let storageType = [{ TFL::SparsityParameterAttr }];
}

#endif // TFL_STRUCT
