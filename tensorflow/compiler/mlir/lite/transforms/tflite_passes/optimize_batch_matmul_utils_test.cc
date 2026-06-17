/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/optimize_batch_matmul_utils.h"

#include <cstdint>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

TEST(OptimizeBatchMatmulUtilsTest, BatchMatMulDimensionsInfo) {
  mlir::MLIRContext context;
  mlir::ShapedType type = mlir::RankedTensorType::get(
      {1, 2, 3, 4, 5}, mlir::Float32Type::get(&context));
  BatchMatMulDimensionsInfo lhs_info(type, /*is_lhs=*/true);
  EXPECT_EQ(lhs_info.batch_dimensions().AxesArray(),
            llvm::ArrayRef<int64_t>({0, 1, 2}));
  EXPECT_EQ(lhs_info.batch_dimensions().SizesArray(),
            llvm::ArrayRef<int64_t>({1, 2, 3}));
  EXPECT_EQ(lhs_info.contracting_dimensions().AxesArray(),
            llvm::ArrayRef<int64_t>({4}));
  EXPECT_EQ(lhs_info.contracting_dimensions().SizesArray(),
            llvm::ArrayRef<int64_t>({5}));
  EXPECT_EQ(lhs_info.out_dimensions().AxesArray(),
            llvm::ArrayRef<int64_t>({3}));
  EXPECT_EQ(lhs_info.out_dimensions().SizesArray(),
            llvm::ArrayRef<int64_t>({4}));
  EXPECT_TRUE(lhs_info.is_lhs());

  BatchMatMulDimensionsInfo rhs_info(type, /*is_lhs=*/false);
  EXPECT_EQ(rhs_info.batch_dimensions().AxesArray(),
            llvm::ArrayRef<int64_t>({0, 1, 2}));
  EXPECT_EQ(rhs_info.batch_dimensions().SizesArray(),
            llvm::ArrayRef<int64_t>({1, 2, 3}));
  EXPECT_EQ(rhs_info.contracting_dimensions().AxesArray(),
            llvm::ArrayRef<int64_t>({3}));
  EXPECT_EQ(rhs_info.contracting_dimensions().SizesArray(),
            llvm::ArrayRef<int64_t>({4}));
  EXPECT_EQ(rhs_info.out_dimensions().AxesArray(),
            llvm::ArrayRef<int64_t>({4}));
  EXPECT_EQ(rhs_info.out_dimensions().SizesArray(),
            llvm::ArrayRef<int64_t>({5}));
  EXPECT_FALSE(rhs_info.is_lhs());
}

TEST(OptimizeBatchMatmulUtilsTest, HasFlattenedContractingDims) {
  mlir::MLIRContext context;
  mlir::ShapedType type = mlir::RankedTensorType::get(
      {1, 2, 3, 4, 50}, mlir::Float32Type::get(&context));
  BatchMatMulDimensionsInfo lhs_info(type, /*is_lhs=*/true);
  EXPECT_TRUE(HasFlattenedContractingDims({1, 2, 3, 4, 5, 10}, lhs_info));
  EXPECT_FALSE(HasFlattenedContractingDims({1, 2, 3, 4, 10}, lhs_info));

  type = mlir::RankedTensorType::get({1, 2, 12, 5},
                                     mlir::Float32Type::get(&context));
  BatchMatMulDimensionsInfo rhs_info(type, /*is_lhs=*/false);
  EXPECT_TRUE(HasFlattenedContractingDims({1, 2, 3, 4, 5}, rhs_info));
  EXPECT_FALSE(HasFlattenedContractingDims({1, 2, 3, 4, 10}, rhs_info));

  type = mlir::RankedTensorType::get({4, 50}, mlir::Float32Type::get(&context));
  lhs_info = BatchMatMulDimensionsInfo(type, /*is_lhs=*/true);
  EXPECT_TRUE(HasFlattenedContractingDims({4, 5, 10}, lhs_info));
  EXPECT_FALSE(HasFlattenedContractingDims({4, 10}, lhs_info));

  type = mlir::RankedTensorType::get({12, 5}, mlir::Float32Type::get(&context));
  rhs_info = BatchMatMulDimensionsInfo(type, /*is_lhs=*/false);
  EXPECT_TRUE(HasFlattenedContractingDims({3, 4, 5}, rhs_info));
  EXPECT_FALSE(HasFlattenedContractingDims({3, 4, 10}, rhs_info));
}

TEST(OptimizeBatchMatmulUtilsTest, HasFlattenedOutDims) {
  mlir::MLIRContext context;
  mlir::ShapedType type = mlir::RankedTensorType::get(
      {1, 2, 12, 5}, mlir::Float32Type::get(&context));
  BatchMatMulDimensionsInfo lhs_info(type, /*is_lhs=*/true);
  EXPECT_TRUE(HasFlattenedOutDims({1, 2, 3, 4, 5}, lhs_info));
  EXPECT_FALSE(HasFlattenedOutDims({1, 2, 3, 4, 10}, lhs_info));

  type = mlir::RankedTensorType::get({1, 2, 12, 10},
                                     mlir::Float32Type::get(&context));
  BatchMatMulDimensionsInfo rhs_info(type, /*is_lhs=*/false);
  EXPECT_TRUE(HasFlattenedOutDims({1, 2, 12, 5, 2}, rhs_info));
  EXPECT_FALSE(HasFlattenedOutDims({1, 2, 3, 4, 10}, rhs_info));

  type = mlir::RankedTensorType::get({12, 5}, mlir::Float32Type::get(&context));
  lhs_info = BatchMatMulDimensionsInfo(type, /*is_lhs=*/true);
  EXPECT_TRUE(HasFlattenedOutDims({3, 4, 5}, lhs_info));
  EXPECT_FALSE(HasFlattenedOutDims({3, 4, 10}, lhs_info));

  type =
      mlir::RankedTensorType::get({12, 10}, mlir::Float32Type::get(&context));
  rhs_info = BatchMatMulDimensionsInfo(type, /*is_lhs=*/false);
  EXPECT_TRUE(HasFlattenedOutDims({12, 5, 2}, rhs_info));
  EXPECT_FALSE(HasFlattenedOutDims({3, 4, 10}, rhs_info));
}

TEST(OptimizeBatchMatmulUtilsTest, GetTransposedGroupsIndexRange) {
  EXPECT_EQ(GetTransposedGroupsIndexRange({0, 1, 2, 6, 7, 8, 3, 4, 5}),
            std::make_tuple(std::make_pair(3, 5), std::make_pair(6, 8)));
  EXPECT_EQ(GetTransposedGroupsIndexRange({2, 0, 1}),
            std::make_tuple(std::make_pair(0, 0), std::make_pair(1, 2)));
  EXPECT_EQ(GetTransposedGroupsIndexRange({0, 1, 2, 3, 7, 8, 4, 5, 6}),
            std::make_tuple(std::make_pair(4, 5), std::make_pair(6, 8)));
  EXPECT_EQ(GetTransposedGroupsIndexRange({0, 1, 2, 3, 8, 7, 4, 5, 6}),
            std::make_tuple(std::make_pair(-1, -1), std::make_pair(-1, -1)));
  EXPECT_EQ(GetTransposedGroupsIndexRange({0, 1, 2}),
            std::make_tuple(std::make_pair(-1, -1), std::make_pair(-1, -1)));
  EXPECT_EQ(GetTransposedGroupsIndexRange({0, 1, 2}),
            std::make_tuple(std::make_pair(-1, -1), std::make_pair(-1, -1)));
  EXPECT_EQ(GetTransposedGroupsIndexRange({}),
            std::make_tuple(std::make_pair(-1, -1), std::make_pair(-1, -1)));
}

TEST(OptimizeBatchMatmulUtilsTest, HasTransposedContractingAndOutDims) {
  mlir::MLIRContext context;
  mlir::ShapedType type = mlir::RankedTensorType::get(
      {1, 2, 3, 504, 120}, mlir::Float32Type::get(&context));
  BatchMatMulDimensionsInfo lhs_info(type, /*is_lhs=*/true);
  EXPECT_TRUE(HasTransposedContractingAndOutDims(
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 6, 7, 8, 3, 4, 5}, lhs_info));
  EXPECT_FALSE(HasTransposedContractingAndOutDims(
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 8, 7, 6, 4, 5, 3}, lhs_info));

  BatchMatMulDimensionsInfo rhs_info(type, /*is_lhs=*/false);
  EXPECT_TRUE(HasTransposedContractingAndOutDims(
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 6, 7, 8, 3, 4, 5}, rhs_info));
  EXPECT_FALSE(HasTransposedContractingAndOutDims(
      {1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 8, 7, 6, 4, 5, 3}, rhs_info));

  type =
      mlir::RankedTensorType::get({504, 120}, mlir::Float32Type::get(&context));
  lhs_info = BatchMatMulDimensionsInfo(type, /*is_lhs=*/true);
  EXPECT_TRUE(HasTransposedContractingAndOutDims({4, 5, 6, 7, 8, 9},
                                                 {3, 4, 5, 0, 1, 2}, lhs_info));
  EXPECT_FALSE(HasTransposedContractingAndOutDims(
      {4, 5, 6, 7, 8, 9}, {5, 4, 3, 1, 2, 0}, lhs_info));

  rhs_info = BatchMatMulDimensionsInfo(type, /*is_lhs=*/false);
  EXPECT_TRUE(HasTransposedContractingAndOutDims({4, 5, 6, 7, 8, 9},
                                                 {3, 4, 5, 0, 1, 2}, rhs_info));
  EXPECT_FALSE(HasTransposedContractingAndOutDims(
      {4, 5, 6, 7, 8, 9}, {5, 4, 3, 1, 2, 0}, rhs_info));
}

}  // namespace
}  // namespace TFL
}  // namespace mlir
