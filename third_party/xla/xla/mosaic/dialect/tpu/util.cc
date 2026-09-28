/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/mosaic/dialect/tpu/util.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/layout.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

FailureOr<SmallVector<int>> computeSqueezedDimsChecked(
    Operation* op, ArrayRef<int64_t> source_shape,
    ArrayRef<int64_t> target_shape) {
  SmallVector<int> squeezed;
  int source_index = source_shape.size() - 1;
  int target_index = target_shape.size() - 1;

  while (source_index >= 0 || target_index >= 0) {
    int64_t target_dim = (target_index >= 0) ? target_shape[target_index] : -1;
    if (source_index < 0) {
      op->emitError() << llvm::formatv(
          "Target shape is not valid. Source: {0}, Target: {1}.",
          shapeToString(source_shape), shapeToString(target_shape));
      return failure();
    }
    int64_t source_dim = source_shape[source_index];
    if (source_dim == target_dim) {
      source_index--;
      target_index--;
    } else {
      if (source_dim != 1) {
        op->emitError() << llvm::formatv(
            "Target shape is not valid. Source: {0}, Target: {1}.",
            shapeToString(source_shape), shapeToString(target_shape));
        return failure();
      }
      squeezed.push_back(source_index);
      source_index--;
    }
  }

  if (source_index != -1 || target_index != -1) {
    op->emitError() << "Shape mismatch after traversal. Source shape: "
                    << shapeToString(source_shape)
                    << ", target shape: " << shapeToString(target_shape);
    return failure();
  }
  std::reverse(squeezed.begin(), squeezed.end());
  return squeezed;
}

bool HasMemorySpace(MemRefType ty, tpu::MemorySpace space,
                    std::optional<CoreType> type) {
  auto memory_space =
      dyn_cast_or_null<tpu::MemorySpaceAttr>(ty.getMemorySpace());
  if (!memory_space || memory_space.getValue() != space) {
    return false;
  }
  return !type.has_value() || memory_space.getCoreType() == type;
}

SmallVector<Value> fillPositions(ValueRange values, ArrayRef<int32_t> positions,
                                 int size, Value missing) {
  SmallVector<Value> result(size, missing);
  for (const auto [value, position] : llvm::zip(values, positions)) {
    result[position] = value;
  }
  return result;
}

FailureOr<tpu::TiledLayoutAttr> getTiledLayout(Location loc,
                                               MemRefType memref_type) {
  if (auto tiled_layout =
          dyn_cast<tpu::TiledLayoutAttr>(memref_type.getLayout());
      tiled_layout != nullptr) {
    return tiled_layout;
  }
  if (memref_type.getLayout().isIdentity()) {
    return tpu::TiledLayoutAttr::getContiguous(
        memref_type.getContext(), /*tiles=*/{}, memref_type.getShape());
  }
  return emitError(loc, "Not implemented: Unsupported layout");
}

bool keepsMinorDimSequentialAndContiguous(ArrayRef<int64_t> shape,
                                          tpu::TiledLayoutAttr layout) {
  if (shape.empty()) {
    return true;
  }
  const ArrayRef<xla::Tile> tiles = layout.getTiles();
  const ArrayRef<int64_t> tile_strides = layout.getTileStrides();
  CHECK(!tile_strides.empty());
  const bool is_untiled = llvm::all_of(
      tiles, [](xla::Tile tile) { return tile.dimensions().empty(); });
  // The memref is untiled and has unit minor stride (or minor shape <= 1).
  if (is_untiled) {
    return tile_strides.back() == 1 ||
           (!ShapedType::isDynamic(shape.back()) && shape.back() <= 1);
  }
  if (tiles.size() == 1 && !tiles[0].dimensions().empty()) {
    const int64_t minor_tile_dim = tiles[0].dimensions().back();
    const int64_t minor_shape_dim = shape.back();
    // If the memref does not span more than one tile along the minormost
    // dimension, the minormost dimension is contiguous within that single tile
    // regardless of tile strides or other tile dimensions (e.g. 2D tiles).
    if (!ShapedType::isDynamic(minor_shape_dim) &&
        minor_shape_dim <= minor_tile_dim) {
      return true;
    }
    // The memref is tiled along the minormost dimension only with consecutive
    // tiles laid out adjacently (unit minor tile stride).
    const ArrayRef<int64_t> tile_dims(tiles[0].dimensions().data(),
                                      tiles[0].dimensions().size());
    const bool is_single_1d_tile =
        llvm::all_of(tile_dims.drop_back(), llvm::equal_to<int64_t>(1));
    return is_single_1d_tile && tile_strides.back() == 1;
  }
  return false;
}

}  // namespace mlir::tpu
