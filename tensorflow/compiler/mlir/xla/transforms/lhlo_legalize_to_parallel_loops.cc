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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // TF:llvm-project
#include "mlir/Dialect/LoopOps/LoopOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {
namespace {

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
//  is converted into:
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
      xla_lhlo::ReduceOp xla_reduce_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    // TODO(b/137624192) Implement variadic reduce.
    if (xla_reduce_op.out().size() != 1) return failure();

    loop::ReduceOp reduce_op =
        CreateParallelLoopsWithReduceOp(xla_reduce_op, args, &rewriter);
    ConvertReductionOperator(xla_reduce_op,
                             &reduce_op.reductionOperator().front(), &rewriter);
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
  loop::ReduceOp CreateParallelLoopsWithReduceOp(
      xla_lhlo::ReduceOp xla_reduce_op, ArrayRef<Value> args,
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
    Type index_type = rewriter->getIndexType();
    for (auto dim : llvm::enumerate(operand_shape)) {
      const bool is_reducing_dim = reducing_dims.count(dim.index());

      Value ub =
          dim.value() == ShapedType::kDynamicSize
              ? rewriter->create<DimOp>(loc, operand, dim.index()).getResult()
              : rewriter->create<mlir::ConstantOp>(
                    loc, index_type,
                    rewriter->getIntegerAttr(index_type, dim.value()));
      Value lb = rewriter->create<mlir::ConstantOp>(
          loc, index_type, rewriter->getIntegerAttr(index_type, 0));
      Value step = rewriter->create<mlir::ConstantOp>(
          loc, index_type, rewriter->getIntegerAttr(index_type, 1));
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
      for (auto& iv : outer.getInductionVars()) {
        out_indices.push_back(iv);
      }
    } else {
      out_indices.push_back(rewriter->create<mlir::ConstantOp>(
          loc, index_type, rewriter->getIntegerAttr(index_type, 0)));
    }

    rewriter->create<StoreOp>(loc, reduction_result, out, out_indices);

    // Load the element to reduce.
    SmallVector<Value, 2> indices;
    indices.reserve(operand_shape.size());
    Block::args_iterator outer_ivs_it =
        outer ? outer.getInductionVars().begin() : nullptr;
    Block::args_iterator inner_ivs_it = inner.getInductionVars().begin();
    for (unsigned i = 0, e = operand_shape.size(); i < e; ++i) {
      indices.push_back(reducing_dims.count(i) ? *inner_ivs_it++
                                               : *outer_ivs_it++);
    }

    rewriter->setInsertionPointToStart(inner.getBody());
    Value elem = rewriter->create<mlir::LoadOp>(
        loc, *xla_reduce_op.operands().begin(), indices);
    return rewriter->create<loop::ReduceOp>(loc, elem);
  }

  // Converts `xla_lhlo.reduce` reduction operator into `loop.reduce` op by
  // doing buffer allocation for scalar arguments and the result of
  // `loop.reduce` to make it compatible with LHLO ops.
  void ConvertReductionOperator(xla_lhlo::ReduceOp xla_reduce_op,
                                Block* loop_reduce_op_body,
                                ConversionPatternRewriter* rewriter) const {
    rewriter->setInsertionPointToStart(loop_reduce_op_body);

    // Allocate buffers to hold arguments of reduction operator block to stay
    // compatible with the LHLO dialect ops in the reduction body.
    auto loc = xla_reduce_op.getLoc();
    Value elem_arg = xla_reduce_op.body().front().getArgument(0);
    Value elem_buf =
        rewriter->create<AllocOp>(loc, elem_arg.getType().cast<MemRefType>());
    rewriter->create<StoreOp>(loc, loop_reduce_op_body->getArgument(0),
                              elem_buf);
    Value acc_arg = xla_reduce_op.body().front().getArgument(1);
    Value acc_buf =
        rewriter->create<AllocOp>(loc, acc_arg.getType().cast<MemRefType>());
    rewriter->create<StoreOp>(loc, loop_reduce_op_body->getArgument(1),
                              acc_buf);

    // Clone the ops from `xla_lhlo.reduce` into reduction operator block.
    BlockAndValueMapping mapping;
    mapping.map(xla_reduce_op.body().front().getArguments(),
                ValueRange{elem_buf, acc_buf, acc_buf});
    for (auto& nested : xla_reduce_op.body().front().without_terminator()) {
      auto clone = rewriter->clone(nested, mapping);
      mapping.map(nested.getResults(), clone->getResults());
    }
    Value acc_result = rewriter->create<LoadOp>(loc, acc_buf);
    rewriter->create<loop::ReduceReturnOp>(loc, acc_result);
  }
};

struct LhloLegalizeToParallelLoops
    : public FunctionPass<LhloLegalizeToParallelLoops> {
  void runOnFunction() override {
    auto func = getFunction();

    OwningRewritePatternList patterns;
    patterns.insert<ReduceOpConverter>(func.getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           loop::LoopOpsDialect, XlaLhloDialect>();
    target.addIllegalOp<xla_lhlo::ReduceOp>();

    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeLhloToParallelLoopsPass() {
  return absl::make_unique<LhloLegalizeToParallelLoops>();
}

static PassRegistration<LhloLegalizeToParallelLoops> legalize_lhlo_pass(
    "lhlo-legalize-to-parallel-loops",
    "Legalize from LHLO dialect to parallel loops.");

}  // namespace xla_lhlo
}  // namespace mlir
