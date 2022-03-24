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

// This is the operation enums definition file for TensorFlow Lite.

#ifndef TFL_OP_ENUMS
#define TFL_OP_ENUMS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td"

// Attributes used for encoding sparse tensors.
// Please find detailed explanation of these parameters in the TFLite schema.
def TFL_DT_Dense : I32EnumAttrCase<"DENSE", 0>;
def TFL_DT_SparseCSR : I32EnumAttrCase<"SPARSE_CSR", 1>;

def TFL_DimensionType : I32EnumAttr<
    "DimensionType", "dimension_type", [TFL_DT_Dense, TFL_DT_SparseCSR]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::TFL";
}
def TFL_DimensionTypeAttr : EnumAttr<TFL_Dialect, TFL_DimensionType,
    "dimension_type_attr">;

#endif // TFL_OP_ENUMS
