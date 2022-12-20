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

#include "gml_st/interfaces/tiling_interface.h"

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace gml_st {

#include "gml_st/interfaces/tiling_interface.cc.inc"

Value materializeSlice(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes,
                       ArrayRef<OpFoldResult> strides, bool useExtractSlice) {
  if (useExtractSlice) {
    return b.create<tensor::ExtractSliceOp>(loc, valueToTile, offsets, sizes,
                                            strides);
  }
  Value tile = b.create<TileOp>(loc, offsets, sizes, strides);
  return b.create<MaterializeOp>(loc, valueToTile, tile);
}

Value materializeSlice(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets,
                       ArrayRef<OpFoldResult> sizes, bool useExtractSlice) {
  SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
  return materializeSlice(b, loc, valueToTile, offsets, sizes, strides,
                          useExtractSlice);
}

Value materializeIdentitySlice(OpBuilder &b, Location loc, Value valueToTile,
                               bool useExtractSlice) {
  if (useExtractSlice) return valueToTile;

  int64_t rank = valueToTile.getType().cast<RankedTensorType>().getRank();
  SmallVector<OpFoldResult> valueToTileSizes{
      tensor::getMixedSizes(b, loc, valueToTile)};
  SmallVector<OpFoldResult> zeros(rank, b.getI64IntegerAttr(0));
  SmallVector<OpFoldResult> ones(rank, b.getI64IntegerAttr(1));
  return materializeSlice(b, loc, valueToTile, zeros, valueToTileSizes, ones,
                          useExtractSlice);
}

Value materializePoint(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets, bool useExtractSlice) {
  auto tensorType = valueToTile.getType().cast<RankedTensorType>();
  int64_t rank = tensorType.getRank();

  IntegerAttr oneAttr = b.getIndexAttr(1);
  SmallVector<OpFoldResult> sizes(rank, oneAttr);
  SmallVector<OpFoldResult> strides(rank, oneAttr);

  if (useExtractSlice) {
    Value slice = b.create<tensor::ExtractSliceOp>(loc, valueToTile, offsets,
                                                   sizes, strides);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    return b.create<tensor::ExtractOp>(loc, slice,
                                       SmallVector<Value>(rank, zero));
  }
  Value tile = b.create<TileOp>(loc, offsets, sizes, strides);
  return b.create<MaterializeOp>(loc, tensorType.getElementType(), valueToTile,
                                 tile);
}

}  // namespace gml_st
}  // namespace mlir
