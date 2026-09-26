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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {
namespace {

class DivisibilityCheckTest : public ::testing::Test {
 protected:
  DivisibilityCheckTest()
      : context_([]() {
          DialectRegistry registry;
          registry.insert<arith::ArithDialect, func::FuncDialect, TPUDialect>();
          return registry;
        }()),
        builder_(&context_) {
    context_.loadAllAvailableDialects();
  }

  ~DivisibilityCheckTest() override {
    for (int i = ops_.size() - 1; i >= 0; --i) {
      ops_[i]->erase();
    }
  }

  template <typename OpTy, typename... Args>
  OpTy Create(Args&&... args) {
    OpTy op = OpTy::create(builder_, std::forward<Args>(args)...);
    ops_.push_back(op.getOperation());
    return op;
  }

  MLIRContext context_;
  OpBuilder builder_;
  std::vector<Operation*> ops_;
};

TEST_F(DivisibilityCheckTest, ConstantAddIOpIsDivisible) {
  Location loc = builder_.getUnknownLoc();
  auto c1_a = Create<arith::ConstantIndexOp>(loc, 1);
  auto c1_b = Create<arith::ConstantIndexOp>(loc, 1);

  // add = 1 + 1 = 2
  auto add = Create<arith::AddIOp>(loc, c1_a, c1_b);

  std::optional<bool> result = isDivisible(add, /*divisor=*/2);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

// Tests that getRemainder directly returns non-zero remainders for sums
// ((1 + 1) mod 3 = 2).
TEST_F(DivisibilityCheckTest, ConstantAddIOpGetRemainder) {
  Location loc = builder_.getUnknownLoc();
  auto c1_a = Create<arith::ConstantIndexOp>(loc, 1);
  auto c1_b = Create<arith::ConstantIndexOp>(loc, 1);

  // add = 1 + 1 = 2
  auto add = Create<arith::AddIOp>(loc, c1_a, c1_b);

  std::optional<int64_t> rem = getRemainder(add, /*divisor=*/3);
  ASSERT_TRUE(rem.has_value());
  EXPECT_EQ(*rem, 2);
}

// Tests that dynamic terms like (x * 8) + (y * 8) are proven divisible by 8.
TEST_F(DivisibilityCheckTest, DynamicMulIOpAddIsDivisible) {
  Location loc = builder_.getUnknownLoc();
  Type index_type = builder_.getIndexType();
  FunctionType func_type =
      builder_.getFunctionType({index_type, index_type}, {});
  auto func = Create<func::FuncOp>(loc, "test_func", func_type);
  Block* entry = func.addEntryBlock();

  Value x = entry->getArgument(0);
  Value y = entry->getArgument(1);

  builder_.setInsertionPointToStart(entry);
  auto c8_a = Create<arith::ConstantIndexOp>(loc, 8);
  auto c8_b = Create<arith::ConstantIndexOp>(loc, 8);

  auto mul_x = Create<arith::MulIOp>(loc, x, c8_a);
  auto mul_y = Create<arith::MulIOp>(loc, y, c8_b);
  auto add = Create<arith::AddIOp>(loc, mul_x, mul_y);

  std::optional<bool> result = isDivisible(add, /*divisor=*/8);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
}

// The remainder of a negative constant is reported in [0, divisor).
TEST_F(DivisibilityCheckTest, NegativeConstantGetRemainder) {
  Location loc = builder_.getUnknownLoc();
  auto cm3 = Create<arith::ConstantIndexOp>(loc, -3);
  auto c8 = Create<arith::ConstantIndexOp>(loc, 8);
  // add = 8 + (-3) = 5
  auto add = Create<arith::AddIOp>(loc, c8, cm3);

  EXPECT_EQ(getRemainder(cm3, /*divisor=*/8), std::optional<int64_t>(5));
  EXPECT_EQ(getRemainder(add, /*divisor=*/8), std::optional<int64_t>(5));
}

// Pallas computes indices in i32 and casts them to index last: the remainder
// of (x * 8 + 3) modulo 8 is known through the cast.
TEST_F(DivisibilityCheckTest, IndexCastGetRemainder) {
  Location loc = builder_.getUnknownLoc();
  Type i32 = builder_.getI32Type();
  FunctionType func_type = builder_.getFunctionType({i32}, {});
  auto func = Create<func::FuncOp>(loc, "test_func", func_type);
  Block* entry = func.addEntryBlock();
  Value x = entry->getArgument(0);

  builder_.setInsertionPointToStart(entry);
  auto aligned = Create<tpu::AssumeMultipleOp>(loc, x, /*multiple=*/8);
  auto c3 = Create<arith::ConstantIntOp>(loc, i32, 3);
  auto add = Create<arith::AddIOp>(loc, aligned, c3);
  auto cast = Create<arith::IndexCastOp>(loc, builder_.getIndexType(), add);
  auto cast_ui =
      Create<arith::IndexCastUIOp>(loc, builder_.getIndexType(), add);

  EXPECT_EQ(getRemainder(cast, /*divisor=*/8), std::optional<int64_t>(3));
  EXPECT_EQ(getRemainder(cast_ui, /*divisor=*/8), std::optional<int64_t>(3));
  EXPECT_EQ(getRemainder(cast, /*divisor=*/2), std::optional<int64_t>(1));
  EXPECT_EQ(isDivisible(cast, /*divisor=*/8), std::optional<bool>(false));
}

// A cast only keeps the remainder modulo a power of two.
TEST_F(DivisibilityCheckTest, IndexCastNonPowerOfTwoDivisorUnknown) {
  Location loc = builder_.getUnknownLoc();
  auto c5 = Create<arith::ConstantIntOp>(loc, builder_.getI32Type(), 5);
  auto cast = Create<arith::IndexCastOp>(loc, builder_.getIndexType(), c5);

  EXPECT_EQ(getRemainder(cast, /*divisor=*/4), std::optional<int64_t>(1));
  EXPECT_EQ(getRemainder(cast, /*divisor=*/3), std::nullopt);
}

// A cast keeps the low bits of its operand, so a divisor wider than the
// operand is unknown: the sign extended constant would give 7 and 456.
TEST_F(DivisibilityCheckTest, IndexCastNarrowOperandKeepsItsBits) {
  Location loc = builder_.getUnknownLoc();
  Type index = builder_.getIndexType();
  auto c_true = Create<arith::ConstantIntOp>(loc, builder_.getI1Type(), 1);
  auto cast_i1 = Create<arith::IndexCastUIOp>(loc, index, c_true);
  auto c200 = Create<arith::ConstantIntOp>(loc, builder_.getI8Type(), 200);
  auto cast_i8 = Create<arith::IndexCastUIOp>(loc, index, c200);

  EXPECT_EQ(getRemainder(cast_i1, /*divisor=*/2), std::optional<int64_t>(1));
  EXPECT_EQ(getRemainder(cast_i1, /*divisor=*/8), std::nullopt);
  EXPECT_EQ(getRemainder(cast_i8, /*divisor=*/256),
            std::optional<int64_t>(200));
  EXPECT_EQ(getRemainder(cast_i8, /*divisor=*/512), std::nullopt);
}

}  // namespace
}  // namespace mlir::tpu
