//===- OpDefinitionTest.cpp - Op definition unit tests --------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "gmock/gmock.h"

using namespace mlir;
using namespace mlir::OpTrait::impl;

namespace {

#define FILE_LOC                                                               \
  FileLineColLoc::get(UniquedFilename::get(__FILE__, &context), __LINE__, 0,   \
                      &context)

// TODO: Replace with regular test once this trait is used by operation in core.
// TODO(b/132891206): Replace with dialect test.
TEST(OpDefinitionTest, SameOperandAndResultElementType) {
  MLIRContext context;
  Builder b(&context);
  auto *operandtF32x10x10 = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getTensorType({10, 10}, b.getF32Type())},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);
  auto *operandtF32x1 = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getTensorType({1}, b.getF32Type())},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);
  auto *operandvF32x1 = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getVectorType({1}, b.getF32Type())},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);
  auto *operandtI32x1 = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getTensorType({1}, b.getIntegerType(32))},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);

  // Verifies whether an op with x and y as inputs and resultType satisfies the
  // SameOperandAndResultElementType trait.
  auto valid = [&](Location loc, Operation *x, Operation *y, Type resultType) {
    auto op = Operation::create(loc, OperationName("some_op", &context),
                                /*operands=*/{x->getResult(0), y->getResult(0)},
                                /*resultTypes=*/{resultType},
                                /*attributes=*/llvm::None, /*successors=*/{},
                                /*numRegions=*/0,
                                /*resizableOperandList=*/false, &context);
    return succeeded(verifySameOperandsAndResultElementType(op));
  };

  EXPECT_TRUE(valid(FILE_LOC, operandtF32x1, operandtF32x1,
                    b.getTensorType({12}, b.getF32Type())));
  EXPECT_TRUE(valid(FILE_LOC, operandtF32x10x10, operandtF32x1,
                    b.getTensorType({5}, b.getF32Type())));
  EXPECT_FALSE(valid(FILE_LOC, operandtF32x10x10, operandtI32x1,
                     b.getTensorType({7}, b.getF32Type())));
  EXPECT_FALSE(valid(FILE_LOC, operandtF32x10x10, operandtF32x1,
                     b.getTensorType({12}, b.getIntegerType(32))));
  EXPECT_FALSE(valid(FILE_LOC, operandtF32x10x10, operandtI32x1,
                     b.getTensorType({9}, b.getIntegerType(32))));
  EXPECT_TRUE(valid(FILE_LOC, operandtF32x10x10, operandtF32x1,
                    b.getVectorType({9}, b.getF32Type())));
  EXPECT_TRUE(valid(FILE_LOC, operandtF32x10x10, operandvF32x1,
                    b.getVectorType({9}, b.getF32Type())));
  EXPECT_TRUE(valid(FILE_LOC, operandtF32x1, operandvF32x1,
                    b.getTensorType({5}, b.getF32Type())));
  EXPECT_FALSE(valid(FILE_LOC, operandtI32x1, operandvF32x1,
                     b.getTensorType({5}, b.getF32Type())));
}

TEST(OpDefinitionTest, SameOperandAndResultShape) {
  MLIRContext context;
  Builder b(&context);
  auto *operandtF32x10x10 = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getTensorType({10, 10}, b.getF32Type())},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);
  auto *operandtF32x1 = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getTensorType({1}, b.getF32Type())},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);
  auto *operandtF32xunranked = Operation::create(
      FILE_LOC, OperationName("some_const", &context), /*operands=*/{},
      /*resultTypes=*/{b.getTensorType(b.getF32Type())},
      /*attributes=*/llvm::None, /*successors=*/{}, /*numRegions=*/0,
      /*resizableOperandList=*/false, &context);

  // SameOperandAndResultShape trait.
  auto valid = [&](Location loc, Operation *x, Operation *y, Type resultType) {
    auto op = Operation::create(loc, OperationName("some_op", &context),
                                /*operands=*/{x->getResult(0), y->getResult(0)},
                                /*resultTypes=*/{resultType},
                                /*attributes=*/llvm::None, /*successors=*/{},
                                /*numRegions=*/0,
                                /*resizableOperandList=*/false, &context);
    return succeeded(verifySameOperandsAndResultShape(op));
  };

  EXPECT_TRUE(valid(FILE_LOC, operandtF32x1, operandtF32x1,
                    b.getTensorType({1}, b.getF32Type())));
  EXPECT_FALSE(valid(FILE_LOC, operandtF32x1, operandtF32x1,
                     b.getTensorType({12}, b.getF32Type())));
  EXPECT_FALSE(valid(FILE_LOC, operandtF32x1, operandtF32x10x10,
                     b.getTensorType({1}, b.getF32Type())));
  EXPECT_TRUE(valid(FILE_LOC, operandtF32x1, operandtF32xunranked,
                    b.getTensorType({1}, b.getF32Type())));
}

#undef FILE_LOC
} // end namespace
