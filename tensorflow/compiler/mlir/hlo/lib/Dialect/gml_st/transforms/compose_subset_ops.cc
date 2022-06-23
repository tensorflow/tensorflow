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

#include <iterator>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

struct OperandOrIntegerRange {
  OperandOrIntegerRange(ValueRange dynamicValues, ArrayAttr staticValues,
                        int64_t dynamicIntPlaceholder)
      : dynamicValues(dynamicValues),
        staticValues(staticValues),
        dynamicIntPlaceholder(dynamicIntPlaceholder) {}

  struct Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = OpFoldResult;
    using pointer = value_type*;
    using reference = value_type&;

   private:
    using StaticValuesIteratorTy = const mlir::Attribute*;
    using DynamicValuesIteratorTy = llvm::detail::indexed_accessor_range_base<
        ValueRange,
        llvm::PointerUnion<const Value*, OpOperand*,
                           mlir::detail::OpResultImpl*>,
        Value, Value, Value>::iterator;

   public:
    Iterator(DynamicValuesIteratorTy dynamicValuesIterator,
             StaticValuesIteratorTy staticValuesIterator,
             int64_t dynamicIntPlaceholder)
        : dynamicValuesIterator(dynamicValuesIterator),
          staticValuesIterator(staticValuesIterator),
          dynamicIntPlaceholder(dynamicIntPlaceholder) {}

    OpFoldResult operator*() const {
      if (staticValuesIterator->cast<IntegerAttr>().getInt() ==
          dynamicIntPlaceholder) {
        return *dynamicValuesIterator;
      }
      return *staticValuesIterator;
    }

    // Increments.
    Iterator& operator++() {
      int64_t integer = staticValuesIterator->cast<IntegerAttr>().getInt();
      if (integer == dynamicIntPlaceholder) dynamicValuesIterator++;
      staticValuesIterator++;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    // Equivalence.
    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.staticValuesIterator == b.staticValuesIterator &&
             a.dynamicValuesIterator == b.dynamicValuesIterator;
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return !(a == b);
    }

   private:
    DynamicValuesIteratorTy dynamicValuesIterator;
    StaticValuesIteratorTy staticValuesIterator;
    int64_t dynamicIntPlaceholder;
  };

 public:
  Iterator begin() {
    return Iterator(dynamicValues.begin(), staticValues.begin(),
                    dynamicIntPlaceholder);
  }
  Iterator end() {
    return Iterator(dynamicValues.end(), staticValues.end(),
                    dynamicIntPlaceholder);
  }

 private:
  ValueRange dynamicValues;
  ArrayAttr staticValues;
  int64_t dynamicIntPlaceholder;
};

OpFoldResult multiplyOperandsOrIntegers(PatternRewriter& rewriter, Location loc,
                                        OpFoldResult lhs, OpFoldResult rhs) {
  // Both operands are static.
  if (lhs.is<Attribute>() && rhs.is<Attribute>()) {
    return rewriter.getI64IntegerAttr(
        lhs.get<Attribute>().cast<IntegerAttr>().getInt() *
        rhs.get<Attribute>().cast<IntegerAttr>().getInt());
  }

  // Exploit commutativity and move static operand to the left (if any).
  if (rhs.is<Attribute>()) std::swap(lhs, rhs);

  // Create constant if needed.
  if (lhs.is<Attribute>()) {
    int64_t lhsInt = lhs.get<Attribute>().cast<IntegerAttr>().getInt();

    // Exploit static operand if possible.
    if (lhsInt == 0) return lhs;
    if (lhsInt == 1) return rhs;

    lhs = rewriter.create<arith::ConstantIndexOp>(loc, lhsInt).getResult();
  }

  // Multiply.
  return rewriter.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs.get<Value>())
      .getResult();
}

OpFoldResult addOperandsOrIntegers(PatternRewriter& rewriter, Location loc,
                                   OpFoldResult lhs, OpFoldResult rhs) {
  // Both operands are static.
  if (lhs.is<Attribute>() && rhs.is<Attribute>()) {
    return rewriter.getI64IntegerAttr(
        lhs.get<Attribute>().cast<IntegerAttr>().getInt() +
        rhs.get<Attribute>().cast<IntegerAttr>().getInt());
  }

  // Exploit commutativity and move static operand to the left (if any).
  if (rhs.is<Attribute>()) std::swap(lhs, rhs);

  // Create constant if needed.
  if (lhs.is<Attribute>()) {
    int64_t lhsInt = lhs.get<Attribute>().cast<IntegerAttr>().getInt();

    // Exploit static operand if possible.
    if (lhsInt == 0) return rhs;

    lhs = rewriter.create<arith::ConstantIndexOp>(loc, lhsInt).getResult();
  }

  // Add.
  return rewriter.create<arith::AddIOp>(loc, lhs.get<Value>(), rhs.get<Value>())
      .getResult();
}

// Compose offsets with newOffset = argOffset + argStride * offset.
std::pair<SmallVector<Value>, ArrayAttr> composeOffsets(
    ValueRange dynamicOffsets, ArrayAttr staticOffsets,
    ValueRange dynamicStrides, ArrayAttr staticStrides,
    ValueRange argDynamicOffsets, ArrayAttr argStaticOffsets, Location loc,
    PatternRewriter& rewriter) {
  // Create ranges.
  OperandOrIntegerRange offsets(dynamicOffsets, staticOffsets,
                                ShapedType::kDynamicStrideOrOffset);
  OperandOrIntegerRange argStrides(dynamicStrides, staticStrides,
                                   ShapedType::kDynamicStrideOrOffset);
  OperandOrIntegerRange argOffsets(argDynamicOffsets, argStaticOffsets,
                                   ShapedType::kDynamicStrideOrOffset);

  // Compose.
  SmallVector<Value> composedDynamicOffsets;
  SmallVector<int64_t> composedStaticOffsets;
  for (auto it : llvm::zip(argOffsets, argStrides, offsets)) {
    auto composed = addOperandsOrIntegers(
        rewriter, loc, std::get<0>(it),
        multiplyOperandsOrIntegers(rewriter, loc, std::get<1>(it),
                                   std::get<2>(it)));
    if (composed.is<Attribute>()) {
      composedStaticOffsets.push_back(
          composed.get<Attribute>().cast<IntegerAttr>().getInt());
    } else {
      composedStaticOffsets.push_back(ShapedType::kDynamicStrideOrOffset);
      composedDynamicOffsets.push_back(composed.get<Value>());
    }
  }
  return {composedDynamicOffsets,
          rewriter.getI64ArrayAttr(composedStaticOffsets)};
}

// Compose strides with newStride = argStride * stride.
std::pair<SmallVector<Value>, ArrayAttr> composeStrides(
    PatternRewriter& rewriter, Location loc, ValueRange argDynamicStrides,
    ArrayAttr argStaticStrides, ValueRange dynamicStrides,
    ArrayAttr staticStrides) {
  // Create ranges.
  OperandOrIntegerRange argStrides(argDynamicStrides, argStaticStrides,
                                   ShapedType::kDynamicStrideOrOffset);
  OperandOrIntegerRange strides(dynamicStrides, staticStrides,
                                ShapedType::kDynamicStrideOrOffset);

  // Compose.
  SmallVector<Value> composedDynamicStrides;
  SmallVector<int64_t> composedStaticStrides;
  for (auto it : llvm::zip(argStrides, strides)) {
    auto product = multiplyOperandsOrIntegers(rewriter, loc, std::get<0>(it),
                                              std::get<1>(it));
    if (product.is<Attribute>()) {
      composedStaticStrides.push_back(
          product.get<Attribute>().cast<IntegerAttr>().getInt());
    } else {
      composedStaticStrides.push_back(ShapedType::kDynamicStrideOrOffset);
      composedDynamicStrides.push_back(product.get<Value>());
    }
  }
  return {composedDynamicStrides,
          rewriter.getI64ArrayAttr(composedStaticStrides)};
}

struct ComposeTilesPattern : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter& rewriter) const override {
    auto argOp = llvm::dyn_cast_or_null<TileOp>(op.subset().getDefiningOp());
    if (!argOp) return failure();

    // Compose offsets with newOffset = argOffset + argStride * offset.
    auto loc = op.getLoc();
    auto composedOffsets =
        composeOffsets(op.offsets(), op.static_offsets(), argOp.strides(),
                       argOp.static_strides(), argOp.offsets(),
                       argOp.static_offsets(), loc, rewriter);

    // Reuse sizes.
    std::pair composedSizes = {op.sizes(), op.static_sizes()};

    // Compose strides with newStride = argStride * stride.
    auto newStrides =
        composeStrides(rewriter, loc, argOp.strides(), argOp.static_strides(),
                       op.strides(), op.static_strides());

    // Build the composed tile op.
    rewriter.replaceOpWithNewOp<TileOp>(
        op, argOp.subset(), composedOffsets.first, composedSizes.first,
        newStrides.first, composedOffsets.second, composedSizes.second,
        newStrides.second);
    return success();
  }
};

class ComposeSubsetOpsPass
    : public ComposeSubsetOpsPassBase<ComposeSubsetOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithmeticDialect, GmlStDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ComposeTilesPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createComposeSubsetOpsPass() {
  return std::make_unique<ComposeSubsetOpsPass>();
}

}  // namespace gml_st
}  // namespace mlir
