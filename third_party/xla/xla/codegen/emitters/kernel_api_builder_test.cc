/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/kernel_api_builder.h"

#include <gtest/gtest.h>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/layout_util.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/runtime/work_tile_size.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::emitters {
namespace {

TEST(DefaultWorkItemIndexingMap, MultiDimensionTile) {
  mlir::MLIRContext mlir_context;
  RegisterSymbolicExprStorage(&mlir_context);
  mlir_context.loadDialect<mlir::affine::AffineDialect>();

  WorkDimensions work_dimensions{NumWorkClusters{}, NumWorkGroups{2},
                                 NumWorkItems{3}, WorkTileSize{{4, 5, 6}}};

  Shape shape(PrimitiveType::F32, {6, 4, 5, 6});
  *shape.mutable_layout() = LayoutUtil::GetDefaultLayoutForShape(shape);

  IndexingMap indexing_map =
      GetDefaultWorkItemIndexingMap(work_dimensions, shape, &mlir_context);

  // The shape is the same as the number of elements work dimensions, so there
  // are no constraints.
  EXPECT_EQ(indexing_map.GetConstraintsCount(), 0);

  // Chunk id is unused so it is removed.
  EXPECT_EQ(indexing_map.GetSymbolCount(), 3);

  // 6 dimensions: 3 work groups + 3 work items.
  EXPECT_EQ(indexing_map.GetDimensionCount(), 6);

  mlir::AffineMap affine_map = indexing_map.GetAffineMap();

  mlir::AffineExpr work_item_sym = mlir::getAffineDimExpr(0, &mlir_context);
  mlir::AffineExpr work_group_sym = mlir::getAffineDimExpr(3, &mlir_context);

  EXPECT_EQ(affine_map.getResult(0), 3 * work_group_sym + work_item_sym);

  mlir::AffineExpr tile_sym_x = mlir::getAffineSymbolExpr(0, &mlir_context);
  EXPECT_EQ(affine_map.getResult(1), tile_sym_x);

  mlir::AffineExpr tile_sym_y = mlir::getAffineSymbolExpr(1, &mlir_context);
  EXPECT_EQ(affine_map.getResult(2), tile_sym_y);

  mlir::AffineExpr tile_sym_z = mlir::getAffineSymbolExpr(2, &mlir_context);
  EXPECT_EQ(affine_map.getResult(3), tile_sym_z);
}

}  // namespace
}  // namespace xla::emitters
