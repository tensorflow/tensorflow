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

#include "tensorflow/core/ir/ops.h"

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {

template <typename OpT>
OpT findOp(ModuleOp module) {
  OpT result;
  module.walk([&](OpT op) {
    result = op;
    return WalkResult::interrupt();
  });
  assert(result);
  return result;
}

//===----------------------------------------------------------------------===//
// Unit tests for TFG region ops RegionBranchOpInterface API.
//===----------------------------------------------------------------------===//

TEST(TestTFGRegionOps, TestIfLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>) {
      %IfRegion, %ctl = IfRegion %arg0 then {
        yield(%arg1) : tensor<f32>
      } else {
        yield(%arg1) : tensor<f32>
      } : (tensor<i1>) -> (tensor<f32>)
      return(%IfRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<IfRegionOp>(*module);

  // Test region -> parent
  SmallVector<RegionSuccessor> regions;
  for (unsigned index = 0; index <= 1; ++index, regions.clear()) {
    op.getSuccessorRegions(index, /*operands=*/{Attribute()}, regions);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_TRUE(regions.front().isParent());
  }

  // Test parent -> regions
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  EXPECT_EQ(regions.size(), 2u);
  regions.clear();

  // Test parent -> regions with known branch
  Builder b(&context);
  ShapedType tensor_type = RankedTensorType::get(/*shape=*/{}, b.getI1Type());
  Attribute cond = DenseElementsAttr::get(tensor_type, /*value=*/true);
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{cond}, regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.getThenRegion());
}

TEST(TestTFGRegionOps, TestCaseLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>) {
      %CaseRegion, %ctl = CaseRegion %arg0 {
        yield(%arg1) : tensor<f32>
      }, {
        yield(%arg1) : tensor<f32>
      } : (tensor<i32>) -> (tensor<f32>)
      return(%CaseRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<CaseRegionOp>(*module);

  // Test region -> parent
  SmallVector<RegionSuccessor> regions;
  for (unsigned index = 0; index < op.getNumRegions();
       ++index, regions.clear()) {
    op.getSuccessorRegions(index, /*operands=*/{Attribute()}, regions);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_TRUE(regions.front().isParent());
  }

  // Test parent -> region
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  EXPECT_EQ(regions.size(), 2u);
  regions.clear();

  // Test parent -> region with known branch
  Builder b(&context);
  ShapedType tensor_type = RankedTensorType::get(/*shape=*/{}, b.getI32Type());
  Attribute branch = DenseElementsAttr::get(tensor_type, /*value=*/1);
  op.getSuccessorRegions(/*index=*/llvm::None, {branch}, regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.getBranches()[1]);
}

TEST(TestTFGRegionOps, TestWhileLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<f32>) -> (tensor<f32>) {
      %WhileRegion, %ctl = WhileRegion(%arg0) {
      ^bb0(%arg1: tensor<f32>, %arg2: !tf_type.control):
        %Cond, %ctl = Cond : () -> (tensor<i1>)
        condition %Cond : tensor<i1> (%arg1) : tensor<f32>
      } do {
      ^bb0(%arg1: tensor<f32>, %arg2: !tf_type.control):
        yield(%arg1) : tensor<f32>
      } {parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<f32>)
      return(%WhileRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<WhileRegionOp>(*module);

  // Test parent -> cond
  SmallVector<RegionSuccessor> regions;
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.getCondRegion());
  regions.clear();

  // Test cond -> parent or body
  op.getSuccessorRegions(/*index=*/0, /*operands=*/{Attribute()}, regions);
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_TRUE(regions.front().isParent() ^ regions.back().isParent());
  regions.clear();

  // Test body -> cond
  op.getSuccessorRegions(/*index=*/1, /*operands=*/{Attribute()}, regions);
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.front().getSuccessor(), &op.getCondRegion());
  regions.clear();
}

TEST(TestTFGRegionOps, TestForLikeRegionOpSuccessorRegions) {
  const char *const code = R"mlir(
    tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>) {
      %ForRegion, %ctl = ForRegion(%arg1) from %arg0 to %arg0 by %arg0 {
        ^bb0(%arg2: tensor<i32>, %arg3: tensor<f32>,
             %arg4: !tf_type.control, %arg5: !tf_type.control):
        yield(%arg3) : tensor<f32>
      } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>) -> (tensor<f32>)
      return(%ForRegion) : tensor<f32>
    }
  )mlir";
  MLIRContext context;
  context.getOrLoadDialect<TFGraphDialect>();
  OwningOpRef<ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(code, &context);
  ASSERT_TRUE(module);
  auto op = findOp<ForRegionOp>(*module);

  // Test parent -> body
  SmallVector<RegionSuccessor> regions;
  op.getSuccessorRegions(/*index=*/llvm::None, /*operands=*/{Attribute()},
                         regions);
  EXPECT_EQ(regions.size(), 1u);
  regions.clear();

  // Test body -> body or parent
  op.getSuccessorRegions(/*index=*/0, /*operands=*/{Attribute()}, regions);
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_TRUE(regions.front().isParent() ^ regions.back().isParent());
}

}  // namespace
}  // namespace tfg
}  // namespace mlir
