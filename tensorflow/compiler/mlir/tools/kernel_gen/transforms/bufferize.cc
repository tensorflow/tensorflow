/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for translating mixed IR to buffer form.

#include "mlir/Transforms/Bufferize.h"  // from @llvm-project

#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

class BufferizeConstantOp : public OpConversionPattern<ConstantOp> {
 public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // We only need to bufferize tensor constants.
    Location loc = op.getLoc();
    auto result_type = op.getType().dyn_cast<RankedTensorType>();
    int64_t result_rank = result_type.getRank();
    if (!result_type || !result_type.hasStaticShape() || result_rank > 1)
      return failure();

    auto memref_type =
        MemRefType::get(result_type.getShape(), result_type.getElementType());
    auto elements_attr = op.value().cast<DenseElementsAttr>();

    if (result_rank == 0) {
      Value buffer = rewriter.create<AllocOp>(loc, memref_type);
      Value constant =
          rewriter.create<ConstantOp>(loc, elements_attr.getValue({}));
      rewriter.create<StoreOp>(loc, constant, buffer);
      rewriter.replaceOp(op, {buffer});
      return success();
    }

    Value buffer = rewriter.create<AllocaOp>(loc, memref_type);

    bool all_same_elems = elements_attr.isSplat();
    Value value;
    if (all_same_elems)
      value = rewriter.create<ConstantOp>(loc, elements_attr.getSplatValue());
    for (auto en : llvm::enumerate(elements_attr.getAttributeValues())) {
      if (!all_same_elems) value = rewriter.create<ConstantOp>(loc, en.value());
      Value index = rewriter.create<ConstantIndexOp>(loc, en.index());
      rewriter.create<StoreOp>(loc, value, buffer, index);
    }
    rewriter.replaceOp(op, {buffer});
    return success();
  }
};

class BufferizeDimOp : public OpConversionPattern<DimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      DimOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    DimOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<DimOp>(op, adaptor.memrefOrTensor(),
                                       adaptor.index());
    return success();
  }
};

class BufferizeAndConvertMinimumBroadcastShapesOp
    : public OpConversionPattern<chlo::MinimumBroadcastShapesOp> {
 public:
  using OpConversionPattern<
      chlo::MinimumBroadcastShapesOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      chlo::MinimumBroadcastShapesOp broadcast_shapes_op,
      ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    chlo::MinimumBroadcastShapesOp::Adaptor adaptor(operands);
    auto loc = broadcast_shapes_op.getLoc();
    ImplicitLocOpBuilder lb(loc, rewriter);
    Value zero = lb.create<ConstantIndexOp>(0);
    SmallVector<Value> shapes = adaptor.shapes();
    size_t k = shapes.size();
    SmallVector<Value> ranks;
    ranks.reserve(k);
    SmallVector<Value> real_ranks;
    real_ranks.reserve(k);
    SmallVector<Value> leading_ones;
    leading_ones.reserve(k);

    // Determine the "real" rank of each operand shape by counting leading 1's.
    for (size_t i = 0; i < k; ++i) {
      Value rank = lb.create<DimOp>(loc, shapes[i], zero);
      ranks.push_back(rank);
      leading_ones.push_back(CountLeadingOnes(lb, shapes[i], rank));
      Value real_rank = lb.create<SubIOp>(rank, leading_ones[i]);
      real_ranks.push_back(real_rank);
    }

    // Determine the maximum real rank of the operands.
    Value max_rank = real_ranks[0];
    for (size_t i = 1; i < k; ++i) {
      Value rank_is_greater =
          lb.create<CmpIOp>(CmpIPredicate::ugt, real_ranks[i], max_rank);
      max_rank = lb.create<SelectOp>(rank_is_greater, real_ranks[i], max_rank);
    }

    // Allocate buffers for the return values and initialize them with 1's.
    SmallVector<Value> result_shapes;
    result_shapes.reserve(k);
    auto result_type =
        MemRefType::get({ShapedType::kDynamicSize}, lb.getIndexType());
    Value one = lb.create<ConstantIndexOp>(1);
    for (size_t i = 0; i < k; ++i) {
      // We assume the buffer will be small, so we allocate it on the stack.
      // TODO(b/181654096): Replace AllocaOp with AllocOp.
      auto result = lb.create<AllocaOp>(result_type, real_ranks[i]);
      lb.create<scf::ForOp>(zero, real_ranks[i], one, llvm::None,
                            [&one, &result](OpBuilder &b, Location l, Value idx,
                                            ValueRange /*vr*/) {
                              b.create<StoreOp>(l, one, result, idx);
                              b.create<scf::YieldOp>(l, llvm::None);
                            });
      result_shapes.push_back(result);
    }

    // Iterate through the dimensions and determine which adjacent dimensions
    // can be combined. Keep a running product of the dimensions that can be
    // combined as iteration variable (initialized to 1), and the current
    // dimension offset in the result shapes. We iterate through the shapes
    // backward, because the broadcasting semantics mean that the last
    // dimensions of each shape (the least significant ones) are matched
    // together.
    Value two = lb.create<ConstantIndexOp>(2);
    Value max_rank_plus_two = lb.create<AddIOp>(loc, max_rank, two);
    Value constant_false =
        lb.create<ConstantOp>(lb.getI1Type(), lb.getBoolAttr(false));
    SmallVector<Value> init_values;
    init_values.reserve(k + 3);
    // Initially, all values are marked as not broadcasted.
    for (int i = 0; i < k; ++i) {
      init_values.push_back(constant_false);
    }
    // The running product is initially 1.
    init_values.push_back(one);
    // The current dimension offset is initially 0.
    init_values.push_back(zero);
    // Whether the broadcasting is invalid.
    init_values.push_back(constant_false);

    // Iterate from 1 to max_rank + 1 (inclusive). This iteration variable is
    // used as an offset from the end of each shape vector. We iterate until
    // max_rank + 1 to handle the case that we have a running_product > 1 left
    // when we have processed all dimensions of the largest shape.
    auto main_loop = lb.create<scf::ForOp>(
        one, max_rank_plus_two, one, init_values,
        [&](OpBuilder &b, Location l, Value v, ValueRange vr) {
          // 'same_size' should track what the size of the dimension is to which
          // the 1-sized dimensions are broadcasted. If all of the dimensions
          // are 1, it will stay 1.
          Value same_size = one;
          // 'result_dimensions' stores the current dimension with an offset of
          // 'leading_ones' to make it easier to check whether we are in-bounds
          // with respect to the "real" shape with leading 1's removed.
          SmallVector<Value> result_dimensions;
          result_dimensions.reserve(k);
          // 'no_broadcasting' stores boolean flags that encode whether the
          // corresponding shape does not need broadcasting at the current
          // position.
          SmallVector<Value> no_broadcasting;
          no_broadcasting.reserve(k + 3);
          // The first k loop carried values are the previous broadcasting
          // state.
          auto prev_no_broadcasting = vr.take_front(k);

          // This loop checks which shapes need broadcasting at the current
          // dimension. A shape needs broadcasting if it is indexed out of
          // bounds, or its current dimension size is 1.
          Value current_dimension_has_invalid_broadcast = constant_false;
          for (size_t i = 0; i < k; ++i) {
            // Determine the size of the current dimension. If the dimension is
            // out of bounds, we choose the value 'one'.
            Value is_out_of_bounds =
                b.create<CmpIOp>(l, CmpIPredicate::ult, real_ranks[i], v);
            Value dimension = b.create<SubIOp>(l, ranks[i], v);
            Value result_dimension =
                b.create<SubIOp>(l, dimension, leading_ones[i]);
            result_dimensions.push_back(result_dimension);
            Value current_size =
                b.create<scf::IfOp>(
                     l, TypeRange{b.getIndexType()}, is_out_of_bounds,
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, one);
                     },
                     [&](OpBuilder &b, Location l) {
                       // Using IfOp instead of SelectOp makes sure that we
                       // don't try to load if the dimension is out of bounds.
                       Value size = b.create<LoadOp>(l, shapes[i], dimension);
                       b.create<scf::YieldOp>(l, size);
                     })
                    .getResult(0);
            // Compute whether the current dimension does require broadcasting.
            Value current_size_is_not_one =
                b.create<CmpIOp>(l, CmpIPredicate::ne, current_size, one);
            no_broadcasting.push_back(current_size_is_not_one);
            Value new_same_size = b.create<SelectOp>(l, current_size_is_not_one,
                                                     current_size, same_size);
            Value same_size_was_not_one =
                b.create<CmpIOp>(l, CmpIPredicate::ne, same_size, one);
            Value is_different_size = b.create<CmpIOp>(
                l, CmpIPredicate::ne, same_size, new_same_size);
            // The broadcast is invalid if the size of the current dimension
            // is not equal to the expected size, unless the expected size was
            // still the initial value 1.
            Value is_invalid =
                b.create<AndOp>(l, same_size_was_not_one, is_different_size);
            current_dimension_has_invalid_broadcast = b.create<OrOp>(
                l, current_dimension_has_invalid_broadcast, is_invalid);
            same_size = new_same_size;
          }

          // Check whether we have at least one shape that has a different
          // status regarding whether it needs broadcasting at the current
          // dimension versus whether it needs broadcasting at the previous
          // dimension.
          Value same_size_is_one =
              b.create<CmpIOp>(l, CmpIPredicate::eq, same_size, one);
          Value different_broadcasting_set = constant_false;
          for (size_t i = 0; i < k; ++i) {
            // If all dimensions are 1, we preserve the status whether a shape
            // needs broadcasting or not, because in that case the dimension can
            // just be ignored.
            no_broadcasting[i] =
                b.create<SelectOp>(l, same_size_is_one, prev_no_broadcasting[i],
                                   no_broadcasting[i]);
            // Compare whether the current shape changes its status regarding
            // whether it needs broadcasting at the current dimension.
            Value broadcasting_is_different =
                b.create<CmpIOp>(l, CmpIPredicate::ne, prev_no_broadcasting[i],
                                 no_broadcasting[i]);
            different_broadcasting_set = b.create<OrOp>(
                l, different_broadcasting_set, broadcasting_is_different);
          }
          Value running_product = vr[k];
          Value current_dimension_offset = vr[k + 1];

          // We need to stop combining dimensions if the set of shapes which
          // need broadcasting at the current dimension changes compared to the
          // set of shapes needing broadcasting at the previous dimension.
          Value is_last_iteration =
              b.create<CmpIOp>(l, CmpIPredicate::sgt, v, max_rank);
          Value stop_combining_dimensions =
              b.create<OrOp>(l, is_last_iteration, different_broadcasting_set);
          auto if_stop_combining_dimensions = b.create<scf::IfOp>(
              l, TypeRange{b.getIndexType(), b.getIndexType()},
              stop_combining_dimensions,
              [&](OpBuilder &b, Location l) {
                // If the running product is not 1, add one dimension of size
                // 'running_product' to each shape that didn't need
                // broadcasting, otherwise add a 1 dimension if it was
                // previously indexed in-bounds.
                Value running_product_not_one = b.create<CmpIOp>(
                    l, CmpIPredicate::ne, running_product, one);
                Value new_dimension_offset =
                    b.create<scf::IfOp>(
                         l, TypeRange{b.getIndexType()},
                         running_product_not_one,
                         [&](OpBuilder &b, Location l) {
                           Value new_dimension_offset = b.create<AddIOp>(
                               l, current_dimension_offset, one);
                           Value minus_one = lb.create<ConstantIndexOp>(-1);
                           for (size_t i = 0; i < k; ++i) {
                             Value was_in_bounds = b.create<CmpIOp>(
                                 l, CmpIPredicate::sge, result_dimensions[i],
                                 minus_one);
                             Value should_store_dimension = b.create<OrOp>(
                                 l, was_in_bounds, prev_no_broadcasting[i]);
                             b.create<scf::IfOp>(
                                 l, should_store_dimension,
                                 [&](OpBuilder &b, Location l) {
                                   Value output_dimension = b.create<SubIOp>(
                                       l, real_ranks[i], new_dimension_offset);
                                   // If the shape needed broadcasting at the
                                   // previous dimension, we set the output size
                                   // to 1, otherwise to 'running_product'.
                                   Value output_size = b.create<SelectOp>(
                                       l, prev_no_broadcasting[i],
                                       running_product, one);
                                   b.create<StoreOp>(l, output_size,
                                                     result_shapes[i],
                                                     output_dimension);
                                   b.create<scf::YieldOp>(l, llvm::None);
                                 });
                           }
                           b.create<scf::YieldOp>(l, new_dimension_offset);
                         },
                         [&](OpBuilder &b, Location l) {
                           b.create<scf::YieldOp>(l, current_dimension_offset);
                         })
                        .getResult(0);
                b.create<scf::YieldOp>(
                    l, ValueRange{same_size, new_dimension_offset});
              },
              [&](OpBuilder &b, Location l) {
                Value new_running_product =
                    b.create<MulIOp>(l, running_product, same_size);
                b.create<scf::YieldOp>(l, ValueRange{new_running_product,
                                                     current_dimension_offset});
              });
          // Add the remaining results.
          no_broadcasting.push_back(if_stop_combining_dimensions.getResult(0));
          no_broadcasting.push_back(if_stop_combining_dimensions.getResult(1));
          Value is_invalid = vr.back();
          is_invalid = b.create<OrOp>(l, is_invalid,
                                      current_dimension_has_invalid_broadcast);
          no_broadcasting.push_back(is_invalid);
          b.create<scf::YieldOp>(l, no_broadcasting);
        });
    Value is_invalid = main_loop.getResults().back();
    for (size_t i = 0; i < k; ++i) {
      result_shapes[i] =
          RemoveLeadingOnesFrom1DMemref(lb, result_shapes[i], real_ranks[i]);
      result_shapes[i] =
          lb.create<SelectOp>(is_invalid, shapes[i], result_shapes[i]);
    }
    rewriter.replaceOp(broadcast_shapes_op, result_shapes);
    return success();
  }

 private:
  Value CountLeadingOnes(ImplicitLocOpBuilder &lb, Value extent_memref,
                         Value rank) const {
    // Count leading 1's. Use two iteration variables for that: one with a
    // boolean flag for whether every size so far was 1, one with the number of
    // leading 1's.
    Value constant_true =
        lb.create<ConstantOp>(lb.getI1Type(), lb.getBoolAttr(true));
    Value zero = lb.create<ConstantIndexOp>(0);
    Value one = lb.create<ConstantIndexOp>(1);
    auto leading_ones_loop = lb.create<scf::ForOp>(
        zero, rank, one, ValueRange{constant_true, zero},
        [&](OpBuilder &b, Location l, Value idx, ValueRange vr) {
          auto size = b.create<LoadOp>(l, extent_memref, idx);
          auto is_equal_to_one =
              b.create<CmpIOp>(l, CmpIPredicate::eq, size, one);
          auto all_ones = b.create<AndOp>(l, vr.front(), is_equal_to_one);
          auto increased_value = b.create<AddIOp>(l, vr.back(), one);
          auto number_of_leading_ones =
              b.create<SelectOp>(l, all_ones, increased_value, vr.back());
          b.create<scf::YieldOp>(l,
                                 ValueRange{all_ones, number_of_leading_ones});
        });
    return leading_ones_loop.results()[1];
  }

  Value RemoveLeadingOnesFrom1DMemref(ImplicitLocOpBuilder &lb,
                                      Value extent_memref, Value rank) const {
    Value leading_ones = CountLeadingOnes(lb, extent_memref, rank);
    Value new_rank = lb.create<SubIOp>(rank, leading_ones);
    auto result_type =
        MemRefType::get({ShapedType::kDynamicSize}, lb.getIndexType());
    // Ideally we would use SubView here to return a MemRef with 'leading_ones'
    // as offset, but several things related to MemRef with offsets are
    // currently broken, so instead we just allocate another buffer of the
    // desired size and copy the elements over. We assume the buffer will be
    // small, so we allocate it on the stack.
    // TODO(b/181654096): Replace AllocaOp with AllocOp.
    Value result = lb.create<AllocaOp>(result_type, new_rank);
    Value zero = lb.create<ConstantIndexOp>(0);
    Value one = lb.create<ConstantIndexOp>(1);
    lb.create<scf::ForOp>(
        zero, new_rank, one, llvm::None,
        [&](OpBuilder &b, Location l, Value idx, ValueRange /*vr*/) {
          Value idx_with_offset = b.create<AddIOp>(l, idx, leading_ones);
          auto size = b.create<LoadOp>(l, extent_memref, idx_with_offset);
          b.create<StoreOp>(l, size, result, idx);
          b.create<scf::YieldOp>(l, llvm::None);
        });
    return result;
  }
};

class BufferizeRankOp : public OpConversionPattern<RankOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      RankOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    RankOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<RankOp>(op, adaptor.memrefOrTensor());
    return success();
  }
};
}  // namespace

void populateExtraStdBufferizePattern(MLIRContext *context,
                                      BufferizeTypeConverter *converter,
                                      OwningRewritePatternList *patterns) {
  patterns
      ->insert<BufferizeConstantOp, BufferizeDimOp,
               BufferizeAndConvertMinimumBroadcastShapesOp, BufferizeRankOp>(
          *converter, context);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
