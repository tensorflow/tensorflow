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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_INTERFACE_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_INTERFACE_H

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// Include generated definitions.
#include "gml_st/interfaces/tiling_interface.h.inc"

namespace mlir {
namespace gml_st {

// Helpers to emit either materialize(tile) or extract_slice.
Value materializeSlice(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes,
                       ArrayRef<OpFoldResult> strides, bool useExtractSlice);

Value materializeSlice(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes, bool useExtractSlice);

Value materializeIdentitySlice(OpBuilder &b, Location loc, Value valueToTile,
                               bool useExtractSlice);

// Extracts a point using materialize(tile) or extract(extract_slice).
Value materializePoint(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets, bool useExtractSlice);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_INTERFACE_H
