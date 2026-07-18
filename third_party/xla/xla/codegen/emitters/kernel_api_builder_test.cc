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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/layout_util.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/runtime/work_tile_size.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
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

  SymbolicMap symbolic_map = indexing_map.GetSymbolicMap();

  SymbolicExpr work_item_sym = CreateDimExpr(0, &mlir_context);
  SymbolicExpr work_group_sym = CreateDimExpr(3, &mlir_context);

  EXPECT_EQ(symbolic_map.GetResult(0), work_group_sym * 3 + work_item_sym);

  SymbolicExpr tile_sym_x =
      CreateSymbolExpr(0, indexing_map.GetDimensionCount(), &mlir_context);
  EXPECT_EQ(symbolic_map.GetResult(1), tile_sym_x);

  SymbolicExpr tile_sym_y =
      CreateSymbolExpr(1, indexing_map.GetDimensionCount(), &mlir_context);
  EXPECT_EQ(symbolic_map.GetResult(2), tile_sym_y);

  SymbolicExpr tile_sym_z =
      CreateSymbolExpr(2, indexing_map.GetDimensionCount(), &mlir_context);
  EXPECT_EQ(symbolic_map.GetResult(3), tile_sym_z);
}

using KernelApiBuilderTest = HloHardwareIndependentTestBase;

TEST_F(KernelApiBuilderTest, NoInvariantOperandsFrontendAttribute) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
e {
  a = s8[0] parameter(0)
  b = s8[0] parameter(1)
  c = s8[0] add(a, b), frontend_attributes={xla.no_invariant_operands="0"}
})"));
  AliasInfo alias_info;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
          },
          &alias_info, [](LogicalBuffer::Color) { return 0; },
          BufferAssigner::Options{.allocate_buffers_for_constants = true}));

  ASSERT_OK_AND_ASSIGN(
      KernelSpec spec,
      GetKernelSpec("test", *module->entry_computation()->root_instruction(),
                    assignment.get(), WorkDimensions{}));

  EXPECT_THAT(spec.invariant_arguments(), ::testing::ElementsAre(1));
}

}  // namespace
}  // namespace xla::emitters
