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

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/LoopOps/LoopOps.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {
namespace {

// Clones and adapts the code in `lhlo_block` that works on buffers and has a
// single output buffer to make it compatible with `operands` that have element
// types of the respective buffers. Returns the computed value.
//
// Example. For `operands` with (f32, i32) types and a block with LHLO ops and
// with signature:
//   ^bb(%lhs: memref<f32>, %rhs: memref<i32>, %res: memref<i1>):
//     <LHLO_ops>
//
// inserts necessary alloc and store ops to compute and return result that has
// `i1` type.
Value ApplySingleResultLhloCode(Location loc, ValueRange operands,
                                Block* lhlo_block, OpBuilder* b) {
  SmallVector<Value, 2> arg_bufs;
  for (auto arg_type : lhlo_block->getArgumentTypes()) {
    arg_bufs.push_back(b->create<AllocOp>(loc, arg_type.cast<MemRefType>()));
  }
  for (auto operand : llvm::enumerate(operands)) {
    b->create<StoreOp>(loc, operand.value(), arg_bufs[operand.index()]);
  }
  // Clone the ops from `lhlo_block`.
  BlockAndValueMapping mapping;
  mapping.map(lhlo_block->getArguments(), arg_bufs);
  for (auto& nested : lhlo_block->without_terminator()) {
    auto clone = b->clone(nested, mapping);
    mapping.map(nested.getResults(), clone->getResults());
  }
  return b->create<LoadOp>(loc, arg_bufs.back());
}

// Converts a block with LHLO ops and with signature:
//   ^bb(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
// into a reduction operator of loop.reduce by doing buffer allocation for
// scalar arguments and the result of `loop.reduce` to make it compatible with
// LHLO ops.
void ConvertToReductionOperator(Location loc, loop::ReduceOp reduce_op,
                                Block* lhlo_block, OpBuilder* b) {
  Block& loop_reduce_op_body = reduce_op.reductionOperator().front();
  OpBuilder::InsertionGuard guard(*b);
  b->setInsertionPointToStart(&loop_reduce_op_body);
  b->create<loop::ReduceReturnOp>(
      loc, ApplySingleResultLhloCode(loc, loop_reduce_op_body.getArguments(),
                                     lhlo_block, b));
}

// Returns result of ConstantOp if `dim` is static, otherwise uses DimOp to
// extract dimension at runtime.
Value GetStaticOrDynamicDim(mlir::Location loc, Value shaped_value,
                            size_t dim_index, int64_t dim, OpBuilder* b) {
  return dim == ShapedType::kDynamicSize
             ? b->create<DimOp>(loc, shaped_value, dim_index).getResult()
             : b->create<ConstantIndexOp>(loc, dim);
}

struct MappedIvs {
  // False if the mapped indices are in the padding area, true otherwise.
  Value in_bounds;
  // Mapped indices.
  SmallVector<Value, 2> ivs;
};

MappedIvs MapWindowIvsToInput(ReduceWindowOp op, ValueRange ivs,
                              ValueRange window_ivs, OpBuilder* b) {
  MappedIvs mapped_ivs;

  if (!op.window_strides().hasValue()) {
    op.emitOpError("No window strides specified.");
  }
  auto window_strides = op.window_strides().getValue();

  if (!op.padding().hasValue()) {
    op.emitOpError("No padding specified.");
  }
  auto padding = op.padding().getValue();

  auto loc = op.getLoc();
  auto operand = op.operand();
  auto operand_shape = operand.getType().cast<MemRefType>().getShape();

  // `in_bounds` is false when the mapped indices are in the padding area.
  mapped_ivs.in_bounds = b->create<mlir::ConstantOp>(
      loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));
  for (unsigned i = 0, e = ivs.size(); i < e; ++i) {
    auto stride = window_strides.getValue<llvm::APInt>(i);
    auto pad_low = padding.getValue<llvm::APInt>({i, 0});

    Value stride_val = b->create<ConstantIndexOp>(loc, stride.getSExtValue());
    Value pad_low_val = b->create<ConstantIndexOp>(loc, pad_low.getSExtValue());

    Value center = b->create<MulIOp>(loc, ivs[i], stride_val);
    Value offset = b->create<SubIOp>(loc, window_ivs[i], pad_low_val);
    Value index = b->create<AddIOp>(loc, center, offset);
    Value upper_bound =
        GetStaticOrDynamicDim(loc, operand, i, operand_shape[i], b);
    // We must check whether 0 <= index_i < shape_i, as otherwise we are in
    // the pad and then we have to use the neutral element for reduction.
    // Equivalently, it can be computed as the unsigned comparison index_i <
    // shape_i, since a negative value wraps to a large positive value.
    mapped_ivs.in_bounds = b->create<mlir::AndOp>(
        loc, mapped_ivs.in_bounds,
        b->create<CmpIOp>(loc, CmpIPredicate::ult, index, upper_bound));
    mapped_ivs.ivs.push_back(index);
  }
  return mapped_ivs;
}

// Returns loop::Parallel over a shaped value with static or dynamic shape.
loop::ParallelOp MakeLoopOverShape(Location loc, Value shaped_value,
                                   OpBuilder* b) {
  Value zero = b->create<ConstantIndexOp>(loc, 0);
  Value one = b->create<ConstantIndexOp>(loc, 1);

  ArrayRef<int64_t> shape =
      shaped_value.getType().cast<ShapedType>().getShape();
  SmallVector<Value, 2> lower, upper, step;
  for (auto dim : llvm::enumerate(shape)) {
    upper.push_back(
        GetStaticOrDynamicDim(loc, shaped_value, dim.index(), dim.value(), b));
    lower.push_back(zero);
    step.push_back(one);
  }
  return b->create<loop::ParallelOp>(loc, lower, upper, step);
}

// Converts `xla_lhlo.ReduceOp` into two loop::ParallelOp and a loop::ReduceOp.
// The outper `ParallelOp` refers to the parallel loops if there are
// any. The inner `ParalleOp` refers to the reduction loops and `ReduceOp`
// contains the reduction operator.
//
// Example:
//
//  "xla_lhlo.reduce"(%buffer, %init_buf, %result) ( {
//    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
//      <LHLO ops>
//    } ) {dimensions = dense<[1]> : tensor<1xi64>}
//      : (memref<100x10x5xf32>, memref<f32>, memref<100x5xf32>) -> ()
//
//  is roughly converted into:
//
//  %init = load %init_buf[] : memref<f32>
//  loop.parallel (%i, %k) = (%c0, %c0) to (%c100, %c5) step (%c1, %c1) {
//    %result = loop.parallel (%j) = (%c0) to (%c10) step (%c1) init (%init) {
//      %elem_to_reduce = load %buffer[%i, %j, %k] : memref<100x10x5xf32>
//      loop.reduce(%elem_to_reduce)  {
//        ^bb0(%elem: f32, %acc: f32):   // no predecessors
//          elem_buf = alloc() : memref<f32>
//          store %elem, elem_buf[] : memref<f32>
//          acc_buf = alloc() : memref<f32>
//          store %acc, acc_buf[] : memref<f32>
//          <LHLO_ops>
//          %acc_result = load acc_buf[] : memref<f32>
//          loop.reduce.return %acc_result : f32
//      } : f32
//      loop.yield
//    } : f32
//    loop.yield
//  }
class ReduceOpConverter : public OpConversionPattern<xla_lhlo::ReduceOp> {
 public:
  using OpConversionPattern<xla_lhlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::ReduceOp xla_reduce_op, ArrayRef<Value> /*args*/,
      ConversionPatternRewriter& rewriter) const final {
    // TODO(b/137624192) Implement variadic reduce.
    if (xla_reduce_op.out().size() != 1) return failure();

    loop::ReduceOp reduce_op =
        CreateReduceOpInNestedParallelLoops(xla_reduce_op, &rewriter);
    ConvertToReductionOperator(xla_reduce_op.getLoc(), reduce_op,
                               &xla_reduce_op.body().front(), &rewriter);
    rewriter.replaceOp(xla_reduce_op, llvm::None);
    return success();
  }

 private:
  // Creates nested `loop.parallel` ops with `loop.reduce`. The outer ParallelOp
  // refers to the parallel dimensions of `xla_reduce_op` if any and the inner
  // ParallelOp refers to the reduction dimensions. The loop.reduce op is
  // returned.
  //
  // If the reduction argument is a memref<100x10x5xf32> and the
  // reduction is performed along dimension 1 then this method will generate
  //
  //  %init = load %init_buf[] : memref<f32>
  //  loop.parallel (%i, %k) = (%c0, %c0) to (%c100, %c5) step (%c1, %c1) {
  //    %result = loop.parallel (%j) = (%c0) to (%c10) step (%c1) init (%init) {
  //      %elem_to_reduce = load %buffer[%i, %j, %k] : memref<100x10x5xf32>
  //      loop.reduce(%elem_to_reduce)  {
  //        <THE BLOCK PTR TO BE RETURNED>
  //      } : f32
  //      loop.yield
  //    } : f32
  //    loop.yield
  //  }
  loop::ReduceOp CreateReduceOpInNestedParallelLoops(
      xla_lhlo::ReduceOp xla_reduce_op,
      ConversionPatternRewriter* rewriter) const {
    auto loc = xla_reduce_op.getLoc();
    DenseSet<int> reducing_dims;
    for (auto rdim : xla_reduce_op.dimensions().getIntValues()) {
      reducing_dims.insert(rdim.getSExtValue());
    }

    Value operand = *xla_reduce_op.operands().begin();
    Value out = *xla_reduce_op.out().begin();
    SmallVector<Value, 2> parallel_lower, parallel_upper, parallel_step;
    SmallVector<Value, 2> reduce_lower, reduce_upper, reduce_step;
    auto operand_shape = operand.getType().cast<MemRefType>().getShape();
    for (auto dim : llvm::enumerate(operand_shape)) {
      const bool is_reducing_dim = reducing_dims.count(dim.index());

      Value ub = GetStaticOrDynamicDim(loc, operand, dim.index(), dim.value(),
                                       rewriter);
      Value lb = rewriter->create<ConstantIndexOp>(loc, 0);
      Value step = rewriter->create<ConstantIndexOp>(loc, 1);
      (is_reducing_dim ? reduce_lower : parallel_lower).push_back(lb);
      (is_reducing_dim ? reduce_upper : parallel_upper).push_back(ub);
      (is_reducing_dim ? reduce_step : parallel_step).push_back(step);
    }
    // Load initial value from memref<element_type>.
    SmallVector<Value, 1> init_value = {
        rewriter->create<LoadOp>(loc, *xla_reduce_op.init_values().begin())};
    // Outer ParallelOp is not needed if it is a reduction across all dims.
    loop::ParallelOp outer;
    if (!parallel_lower.empty()) {
      outer = rewriter->create<loop::ParallelOp>(loc, parallel_lower,
                                                 parallel_upper, parallel_step);
      rewriter->setInsertionPointToStart(outer.getBody());
    }
    loop::ParallelOp inner = rewriter->create<loop::ParallelOp>(
        loc, reduce_lower, reduce_upper, reduce_step, init_value);
    Value reduction_result = *inner.getResults().begin();

    SmallVector<Value, 1> out_indices;
    if (outer != nullptr) {
      out_indices.reserve(outer.getNumLoops());
      for (Value iv : outer.getInductionVars()) {
        out_indices.push_back(iv);
      }
    } else {
      out_indices.push_back(rewriter->create<ConstantIndexOp>(loc, 0));
    }

    rewriter->create<StoreOp>(loc, reduction_result, out, out_indices);

    // Load the element to reduce.
    SmallVector<Value, 2> indices;
    indices.reserve(operand_shape.size());

    if (outer) {
      auto inner_ivs_it = inner.getInductionVars().begin();
      auto outer_ivs_it = outer.getInductionVars().begin();
      for (unsigned i = 0, e = operand_shape.size(); i < e; ++i) {
        indices.push_back(reducing_dims.count(i) ? *inner_ivs_it++
                                                 : *outer_ivs_it++);
      }
    } else {
      indices = inner.getInductionVars();
    }

    rewriter->setInsertionPointToStart(inner.getBody());
    Value elem = rewriter->create<mlir::LoadOp>(
        loc, *xla_reduce_op.operands().begin(), indices);
    return rewriter->create<loop::ReduceOp>(loc, elem);
  }
};

// Pseudocode:
// for each index O in output
//   accumulator = neutral_value
//   in_bounds = true
//   for each index W in window
//     for each dimension i from 0 to rank - 1
//       index = O[i] * stride[i] + W[i] - pad_low[i]
//       in_bounds = inbounds && (index `ult` shape[i])
//       I[i] = index
//     if (in_bounds)
//       value = input[I]
//     else
//       value = neutral_value
//     accumulator = reduction_operator(output[O], value)
//   output[O] = accumulator
//
// Converts `xla_lhlo.ReduceWindowOp` into two loop::ParallelOp and a
// loop::ReduceOp.
// The outper `ParallelOp` refers to the parallel loops that traverese output
// buffer. The inner `ParalleOp` refers to the reduction loops that traverse
// reduction windows and `ReduceOp` contains the reduction operator.
//
// Example:
//
// func @reduce_window(%arg: memref<112x112xf32>,
//              %init: memref<f32>,
//              %result: memref<56x56xf32>) {
//   "xla_lhlo.reduce_window"(%arg, %init, %result) ( {
//     ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
//       "xla_lhlo.maximum"(%lhs, %rhs, %res)
//         : (memref<f32>, memref<f32>, memref<f32>) -> ()
//       "xla_lhlo.terminator"() : () -> ()
//     }) {
//       padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
//       window_dimensions = dense<[3, 3]> : tensor<2xi64>,
//       window_strides = dense<[2, 2]> : tensor<2xi64>
//     } : (memref<112x112xf32>, memref<f32>, memref<56x56xf32>) -> ()
//   return
// }
//
// is roughly converted into:
//
//    %neutral_elem = load %init_buf[] : memref<f32>
//    loop.parallel (%i, %j) = (%c0, %c0) to (%c56, %c56) step (%c1, %c1) {
//      %result = loop.parallel (%iw, %jw) = (%c0, %c0)
//                  to (%c3, %c3) step (%c1, %c1) neutral_elem (%0) -> f32 {
//        %in_bounds = <COMPUTE IF INDEX IS IN OPERAND'S pad>
//        %elem = load %operand[%computed_i, %computed_j]
//        %elem_or_neutral = select %in_bounds, %elem, %neutral_elem : f32
//        loop.reduce(%elem_to_reduce)  : f32 {
//          ^bb0(%arg7: f32, %arg8: f32):
//            <LHLO ops>
//        }
//        loop.yield
//      }
//      store %result, %output_buffer[%i, %j] : memref<56x56xf32>
//      loop.yield
//    }
//    return
//  }
class ReduceWindowOpConverter
    : public OpConversionPattern<xla_lhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern<xla_lhlo::ReduceWindowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::ReduceWindowOp xla_reduce_window_op, ArrayRef<Value> /*args*/,
      ConversionPatternRewriter& rewriter) const final {
    loop::ParallelOp output_loop, window_loop;
    std::tie(output_loop, window_loop) =
        CreateParallelLoopsToTraverseOutputAndWindow(xla_reduce_window_op,
                                                     &rewriter);

    loop::ReduceOp reduce_op = CreateReduceOpInNestedParallelLoops(
        xla_reduce_window_op, output_loop, window_loop, &rewriter);

    ConvertToReductionOperator(xla_reduce_window_op.getLoc(), reduce_op,
                               &xla_reduce_window_op.body().front(), &rewriter);
    rewriter.replaceOp(xla_reduce_window_op, llvm::None);
    return success();
  }

 private:
  std::pair<loop::ParallelOp, loop::ParallelOp>
  CreateParallelLoopsToTraverseOutputAndWindow(
      xla_lhlo::ReduceWindowOp xla_reduce_window_op,
      ConversionPatternRewriter* rewriter) const {
    auto loc = xla_reduce_window_op.getLoc();
    Value init_value =
        rewriter->create<LoadOp>(loc, xla_reduce_window_op.init_value());

    Value zero = rewriter->create<ConstantIndexOp>(loc, 0);
    Value one = rewriter->create<ConstantIndexOp>(loc, 1);

    // Create an outer parallel loop that spans the output of ReduceWindowOp.
    Value xla_output = xla_reduce_window_op.out();
    auto output_loop = MakeLoopOverShape(loc, xla_output, rewriter);

    // Create a nested loop that traverses the window.
    SmallVector<Value, 2> window_lower, window_upper, window_step;
    rewriter->setInsertionPointToStart(output_loop.getBody());
    for (const auto& window_dim : xla_reduce_window_op.window_dimensions()) {
      window_step.push_back(one);
      window_lower.push_back(zero);
      window_upper.push_back(
          rewriter->create<ConstantIndexOp>(loc, window_dim.getSExtValue()));
    }
    auto window_loop = rewriter->create<loop::ParallelOp>(
        loc, window_lower, window_upper, window_step, init_value);

    Value reduction_result = *window_loop.getResults().begin();
    auto output_ivs = output_loop.getInductionVars();
    rewriter->create<StoreOp>(loc, reduction_result, xla_output, output_ivs);
    return std::make_pair(output_loop, window_loop);
  }

  loop::ReduceOp CreateReduceOpInNestedParallelLoops(
      xla_lhlo::ReduceWindowOp xla_reduce_window_op,
      loop::ParallelOp output_loop, loop::ParallelOp window_loop,
      ConversionPatternRewriter* rewriter) const {
    rewriter->setInsertionPointToStart(window_loop.getBody());
    auto loc = xla_reduce_window_op.getLoc();

    if (xla_reduce_window_op.base_dilations().hasValue() ||
        xla_reduce_window_op.window_dilations().hasValue()) {
      xla_reduce_window_op.emitRemark(
          "Lowering to parallel loops does not support `base_dilations` or "
          "`window_dilations` attributes yet. The attributes will be ignored.");
    }

    Value xla_operand = xla_reduce_window_op.operand();
    auto xla_operand_type = xla_operand.getType().cast<MemRefType>();

    MappedIvs mapped_ivs = MapWindowIvsToInput(
        xla_reduce_window_op, output_loop.getInductionVars(),
        window_loop.getInductionVars(), rewriter);

    auto elem_or_init = rewriter->create<loop::IfOp>(
        loc, xla_operand_type.getElementType(), mapped_ivs.in_bounds,
        /*withElseRegion=*/true);

    OpBuilder then_builder = elem_or_init.getThenBodyBuilder();
    Value elem = then_builder.create<mlir::LoadOp>(
        loc, xla_reduce_window_op.operand(), mapped_ivs.ivs);
    then_builder.create<loop::YieldOp>(loc, elem);

    OpBuilder else_builder = elem_or_init.getElseBodyBuilder();
    else_builder.create<loop::YieldOp>(loc, *window_loop.initVals().begin());

    return rewriter->create<loop::ReduceOp>(loc,
                                            *elem_or_init.results().begin());
  }
};

struct LhloLegalizeToParallelLoops
    : public PassWrapper<LhloLegalizeToParallelLoops, FunctionPass> {
  void runOnFunction() override {
    auto func = getFunction();

    OwningRewritePatternList patterns;
    // clang-format off
    patterns.insert<
        ReduceOpConverter,
        ReduceWindowOpConverter
      >(func.getContext());
    // clang-format on

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           loop::LoopOpsDialect, XlaLhloDialect>();
    target.addIllegalOp<xla_lhlo::ReduceOp>();
    target.addIllegalOp<xla_lhlo::ReduceWindowOp>();

    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToParallelLoopsPass() {
  return absl::make_unique<LhloLegalizeToParallelLoops>();
}

static PassRegistration<LhloLegalizeToParallelLoops> legalize_lhlo_pass(
    "lhlo-legalize-to-parallel-loops",
    "Legalize from LHLO dialect to parallel loops.");

}  // namespace xla_lhlo
}  // namespace mlir
