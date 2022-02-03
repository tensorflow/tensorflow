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

// This file defines the operations used in the ST dialect.

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// Generated dialect definitions.
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_dialect.cc.inc"

// Generated op classes.
#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"

// Generated type classes.
#define GET_TYPEDEF_CLASSES
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc"

namespace mlir {
namespace gml_st {

void GmlStDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_types.cc.inc"
      >();
}

LogicalResult MaterializeOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  MaterializeOp::Adaptor adaptor(operands, attributes);

  ShapedType sourceType = adaptor.source().getType().cast<ShapedType>();
  Type subsetType = adaptor.subset().getType();

  if (auto tileType = subsetType.dyn_cast<TileType>()) {
    if (auto memrefType = sourceType.dyn_cast<MemRefType>()) {
      inferredReturnTypes.push_back(
          MemRefType::get(tileType.getShape(), sourceType.getElementType()));
    } else if (auto tensorType = sourceType.dyn_cast<RankedTensorType>()) {
      inferredReturnTypes.push_back(RankedTensorType::get(
          tileType.getShape(), sourceType.getElementType()));
    } else {
      return failure();
    }
  } else if (subsetType.isa<PointType>()) {
    inferredReturnTypes.push_back(sourceType.getElementType());
  } else {
    return failure();
  }
  return success();
}

}  // namespace gml_st
}  // namespace mlir
