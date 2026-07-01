/* Copyright 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/layout.h"

namespace mlir {
namespace {

using ::mlir::tpu::TiledLayoutAttr;

TEST(TiledLayoutTest, NumTrailingDimsWithContiguousTilesTest) {
  DialectRegistry registry;
  registry.insert<tpu::TPUDialect>();
  MLIRContext ctx(registry);
  ctx.loadDialect<tpu::TPUDialect>();

  SmallVector<xla::Tile> tiles = {xla::Tile({2, 2})};
  TiledLayoutAttr tiled_layout;
  tiled_layout = TiledLayoutAttr::get(&ctx, tiles, {2, 1});  // 2D
  // Simple contiguity
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles({4, 4}), 2);
  // ...now make the major dimension dynamic. Should still be contiguous.
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles(
                {ShapedType::kDynamic, 4}),
            2);
  tiled_layout = TiledLayoutAttr::get(&ctx, tiles, {4, 2, 1});  // 3D
  // ...now add a leading singleton dimension. Should still be contiguous.
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles(
                {1, ShapedType::kDynamic, 4}),
            3);
  // ...now make it non-singleton. Should no longer be contiguous.
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles(
                {2, ShapedType::kDynamic, 4}),
            2);

  // Check that the strides are ignored for single-tile dimensions.
  tiled_layout = TiledLayoutAttr::get(&ctx, tiles, {16, 1});
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles({2, 4}), 2);

  // Check contiguity for a layout with non-decreasing strides.
  // (note that the minor dimension has a single tile)
  tiled_layout = TiledLayoutAttr::get(&ctx, tiles, {1, 4});
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles({3, 2}),
            2);

  // Check some completely non-contiguous layouts.
  tiled_layout = TiledLayoutAttr::get(&ctx, {}, {10, 2});
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles({4, 10}), 0);
  EXPECT_EQ(tiled_layout.getNumTrailingDimsWithContiguousTiles(
                {4, ShapedType::kDynamic}),
            0);
}

}  // namespace
}  // namespace mlir
