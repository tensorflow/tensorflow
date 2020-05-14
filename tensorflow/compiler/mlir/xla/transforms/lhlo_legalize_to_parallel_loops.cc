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
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
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
// into a reduction operator of scf.reduce by doing buffer allocation for
// scalar arguments and the result of `scf.reduce` to make it compatible with
// LHLO ops.
void ConvertToReductionOperator(Location loc, scf::ReduceOp reduce_op,
                                Block* lhlo_block, OpBuilder* b) {
  Block& loop_reduce_op_body = reduce_op.reductionOperator().front();
  OpBuilder::InsertionGuard guard(*b);
  b->setInsertionPointToStart(&loop_reduce_op_body);
  b->create<scf::ReduceReturnOp>(
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

template <typename OpTy>
MappedIvs MapWindowIvsToInput(OpTy op, ValueRange ivs, ValueRange window_ivs,
                              OpBuilder* b) {
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
  auto operand_shape = operand.getType().template cast<MemRefType>().getShape();

  // `in_bounds` is false when the mapped indices are in the padding area.
  mapped_ivs.in_bounds = b->create<mlir::ConstantOp>(
      loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));
  for (unsigned i = 0, e = ivs.size(); i < e; ++i) {
    auto stride = window_strides.template getValue<llvm::APInt>(i);
    auto pad_low = padding.template getValue<llvm::APInt>({i, 0});

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

// Returns scf::Parallel over a shaped value with static or dynamic shape.
scf::ParallelOp MakeLoopOverShape(Location loc, Value shaped_value,
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
  return b->create<scf::ParallelOp>(loc, lower, upper, step);
}

// Converts `xla_lhlo.ReduceOp` into two scf::ParallelOp and a scf::ReduceOp.
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
//  scf.parallel (%i, %k) = (%c0, %c0) to (%c100, %c5) step (%c1, %c1) {
//    %result = scf.parallel (%j) = (%c0) to (%c10) step (%c1) init (%init) {
//      %elem_to_reduce = load %buffer[%i, %j, %k] : memref<100x10x5xf32>
//      scf.reduce(%elem_to_reduce)  {
//        ^bb0(%elem: f32, %acc: f32):   // no predecessors
//          elem_buf = alloc() : memref<f32>
//          store %elem, elem_buf[] : memref<f32>
//          acc_buf = alloc() : memref<f32>
//          store %acc, acc_buf[] : memref<f32>
//          <LHLO_ops>
//          %acc_result = load acc_buf[] : memref<f32>
//          scf.reduce.return %acc_result : f32
//      } : f32
//      scf.yield
//    } : f32
//    scf.yield
//  }
class ReduceOpConverter : public OpConversionPattern<xla_lhlo::ReduceOp> {
 public:
  using OpConversionPattern<xla_lhlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::ReduceOp xla_reduce_op, ArrayRef<Value> /*args*/,
      ConversionPatternRewriter& rewriter) const final {
    // TODO(b/137624192) Implement variadic reduce.
    if (xla_reduce_op.out().size() != 1) return failure();

    scf::ReduceOp reduce_op =
        CreateReduceOpInNestedParallelLoops(xla_reduce_op, &rewriter);
    ConvertToReductionOperator(xla_reduce_op.getLoc(), reduce_op,
                               &xla_reduce_op.body().front(), &rewriter);
    rewriter.replaceOp(xla_reduce_op, llvm::None);
    return success();
  }

 private:
  // Creates nested `scf.parallel` ops with `scf.reduce`. The outer ParallelOp
  // refers to the parallel dimensions of `xla_reduce_op` if any and the inner
  // ParallelOp refers to the reduction dimensions. The scf.reduce op is
  // returned.
  //
  // If the reduction argument is a memref<100x10x5xf32> and the
  // reduction is performed along dimension 1 then this method will generate
  //
  //  %init = load %init_buf[] : memref<f32>
  //  scf.parallel (%i, %k) = (%c0, %c0) to (%c100, %c5) step (%c1, %c1) {
  //    %result = scf.parallel (%j) = (%c0) to (%c10) step (%c1) init (%init) {
  //      %elem_to_reduce = load %buffer[%i, %j, %k] : memref<100x10x5xf32>
  //      scf.reduce(%elem_to_reduce)  {
  //        <THE BLOCK PTR TO BE RETURNED>
  //      } : f32
  //      scf.yield
  //    } : f32
  //    scf.yield
  //  }
  scf::ReduceOp CreateReduceOpInNestedParallelLoops(
      xla_lhlo::ReduceOp xla_reduce_op,
      ConversionPatternRewriter* rewriter) const {
    auto loc = xla_reduce_op.getLoc();
    DenseSet<int> reducing_dims;
    for (const auto& rdim : xla_reduce_op.dimensions().getIntValues()) {
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
    scf::ParallelOp outer;
    if (!parallel_lower.empty()) {
      outer = rewriter->create<scf::ParallelOp>(loc, parallel_lower,
                                                parallel_upper, parallel_step);
      rewriter->setInsertionPointToStart(outer.getBody());
    }
    scf::ParallelOp inner = rewriter->create<scf::ParallelOp>(
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
    return rewriter->create<scf::ReduceOp>(loc, elem);
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
// Converts `xla_lhlo.ReduceWindowOp` into two scf::ParallelOp and a
// scf::ReduceOp.
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
//    scf.parallel (%i, %j) = (%c0, %c0) to (%c56, %c56) step (%c1, %c1) {
//      %result = scf.parallel (%iw, %jw) = (%c0, %c0)
//                  to (%c3, %c3) step (%c1, %c1) neutral_elem (%0) -> f32 {
//        %in_bounds = <COMPUTE IF INDEX IS IN OPERAND'S pad>
//        %elem = load %operand[%computed_i, %computed_j]
//        %elem_or_neutral = select %in_bounds, %elem, %neutral_elem : f32
//        scf.reduce(%elem_to_reduce)  : f32 {
//          ^bb0(%arg7: f32, %arg8: f32):
//            <LHLO ops>
//        }
//        scf.yield
//      }
//      store %result, %output_buffer[%i, %j] : memref<56x56xf32>
//      scf.yield
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
    scf::ParallelOp output_loop, window_loop;
    std::tie(output_loop, window_loop) =
        CreateParallelLoopsToTraverseOutputAndWindow(xla_reduce_window_op,
                                                     &rewriter);

    scf::ReduceOp reduce_op = CreateReduceOpInNestedParallelLoops(
        xla_reduce_window_op, output_loop, window_loop, &rewriter);

    ConvertToReductionOperator(xla_reduce_window_op.getLoc(), reduce_op,
                               &xla_reduce_window_op.body().front(), &rewriter);
    rewriter.replaceOp(xla_reduce_window_op, llvm::None);
    return success();
  }

 private:
  std::pair<scf::ParallelOp, scf::ParallelOp>
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
    auto window_loop = rewriter->create<scf::ParallelOp>(
        loc, window_lower, window_upper, window_step, init_value);

    Value reduction_result = *window_loop.getResults().begin();
    auto output_ivs = output_loop.getInductionVars();
    rewriter->create<StoreOp>(loc, reduction_result, xla_output, output_ivs);
    return std::make_pair(output_loop, window_loop);
  }

  scf::ReduceOp CreateReduceOpInNestedParallelLoops(
      xla_lhlo::ReduceWindowOp xla_reduce_window_op,
      scf::ParallelOp output_loop, scf::ParallelOp window_loop,
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

    // Compute ivs in 'arg' buffer and whether these ivs are in pad area or not.
    MappedIvs mapped_ivs = MapWindowIvsToInput(
        xla_reduce_window_op, output_loop.getInductionVars(),
        window_loop.getInductionVars(), rewriter);

    auto elem_or_init = rewriter->create<scf::IfOp>(
        loc, xla_operand_type.getElementType(), mapped_ivs.in_bounds,
        /*withElseRegion=*/true);

    OpBuilder then_builder = elem_or_init.getThenBodyBuilder();
    Value elem = then_builder.create<mlir::LoadOp>(
        loc, xla_reduce_window_op.operand(), mapped_ivs.ivs);
    then_builder.create<scf::YieldOp>(loc, elem);

    OpBuilder else_builder = elem_or_init.getElseBodyBuilder();
    else_builder.create<scf::YieldOp>(loc, *window_loop.initVals().begin());

    return rewriter->create<scf::ReduceOp>(loc,
                                           *elem_or_init.results().begin());
  }
};

// See the operation semantics in
// https://www.tensorflow.org/xla/operation_semantics#selectandscatter
//
// Pseudocode:
//  scf.parallel(coordinates O in the output):
//    output[O] = init
//  scf.parallel(coordinates S in the source):
//    selected_ivs = 0
//    selected_val = 0
//    initialized_flag = false
//    scf.for (first dim W_1 in the window)
//         iter_args (selected_ivs, selected_val, initialized_flag):
//    ...
//      scf.for (last dim W_N in the window):
//           iter_args (selected_ivs, selected_val, initialized_flag):
//        I = S * stride + W - pad_low
//        if I within bounds of operand:
//          if (initialized_flag):
//            pred = select(selected_value, operand(I))):
//            if (pred)
//              selected_value = operand(I)
//              selected_index = I
//          else
//              selected_value = operand(I)
//              selected_index = I
//              initialized_flag = true
//    output(selected_index) = scatter(output(selected_index), source(S))
class SelectAndScatterOpConverter
    : public OpConversionPattern<xla_lhlo::SelectAndScatterOp> {
 public:
  using OpConversionPattern<xla_lhlo::SelectAndScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::SelectAndScatterOp s_and_s_op, ArrayRef<Value> /*args*/,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = s_and_s_op.getLoc();
    InitializeOutput(s_and_s_op, &rewriter);
    scf::ParallelOp loop_over_src =
        MakeLoopOverShape(loc, s_and_s_op.source(), &rewriter);
    rewriter.setInsertionPointToStart(loop_over_src.getBody());

    // Compute indices of the selected element in the window.
    auto selected_ivs = SelectIvs(s_and_s_op, loop_over_src, &rewriter);

    // Load `source[selected_ivs]`.
    auto src_elem = rewriter.create<LoadOp>(loc, s_and_s_op.source(),
                                            loop_over_src.getInductionVars());

    // Compute `out[selected_ivs]` = scatter(out[selected_ivs], src_element)`.
    auto rmw = rewriter.create<GenericAtomicRMWOp>(loc, s_and_s_op.out(),
                                                   selected_ivs);
    OpBuilder rmw_builder = OpBuilder::atBlockEnd(rmw.getBody());
    auto acc_result =
        ApplySingleResultLhloCode(loc, {src_elem, rmw.getCurrentValue()},
                                  &s_and_s_op.scatter().front(), &rmw_builder);
    rmw_builder.create<AtomicYieldOp>(loc, acc_result);

    rewriter.replaceOp(s_and_s_op, llvm::None);
    return success();
  }

 private:
  void InitializeOutput(xla_lhlo::SelectAndScatterOp s_and_s_op,
                        OpBuilder* b) const {
    auto loc = s_and_s_op.getLoc();
    Value init_value = b->create<LoadOp>(loc, s_and_s_op.init_value());

    scf::ParallelOp loop_over_output =
        MakeLoopOverShape(loc, s_and_s_op.out(), b);
    OpBuilder::InsertionGuard guard(*b);
    b->setInsertionPointToStart(loop_over_output.getBody());
    b->create<StoreOp>(loc, init_value, s_and_s_op.out(),
                       loop_over_output.getInductionVars());
  }

  struct WindowLoops {
    SmallVector<Value, 2> selected_ivs;
    SmallVector<Value, 2> window_ivs;
    scf::ForOp inner_loop;
  };
  WindowLoops InsertWindowLoops(xla_lhlo::SelectAndScatterOp s_and_s_op,
                                scf::ParallelOp loop_over_src,
                                OpBuilder* b) const {
    auto loc = s_and_s_op.getLoc();
    Value zero = b->create<ConstantIndexOp>(loc, 0);
    Value one = b->create<ConstantIndexOp>(loc, 1);

    auto element_type =
        s_and_s_op.out().getType().cast<MemRefType>().getElementType();
    auto rank = loop_over_src.getNumLoops();

    // `iter_args` = [iv_1, ..., iv_N, selected_value, is_initialized]
    SmallVector<Value, 4> iter_args(rank, zero);
    iter_args.push_back(b->create<mlir::ConstantOp>(
        loc, element_type, b->getFloatAttr(element_type, 0)));
    iter_args.push_back(b->create<mlir::ConstantOp>(
        loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 0)));

    // Create a nested loop that traverses the window.
    OpBuilder::InsertPoint ip;
    WindowLoops result;
    for (const auto& window_dim :
         s_and_s_op.window_dimensions()->getIntValues()) {
      Value upper = b->create<ConstantIndexOp>(loc, window_dim.getSExtValue());
      result.inner_loop =
          b->create<scf::ForOp>(loc, zero, upper, one, iter_args);
      if (b->getInsertionBlock() == loop_over_src.getBody()) {
        ip = b->saveInsertionPoint();
        result.selected_ivs = result.inner_loop.getResults().take_front(rank);
      } else {
        b->create<scf::YieldOp>(loc, result.inner_loop.getResults());
      }
      b->setInsertionPointToStart(result.inner_loop.getBody());
      iter_args = ValueRange{result.inner_loop.getRegionIterArgs()};
      result.window_ivs.push_back(result.inner_loop.getInductionVar());
    }
    b->restoreInsertionPoint(ip);
    return result;
  }

  // Adapter to store iteration arguments of sequential loops that perform
  // select in a window.
  class IterArgs {
   public:
    explicit IterArgs(ValueRange ivs_val_flag) : ivs_val_flag_(ivs_val_flag) {}
    IterArgs(ValueRange ivs, Value value, Value flag) {
      ivs_val_flag_ = ivs;
      ivs_val_flag_.push_back(value);
      ivs_val_flag_.push_back(flag);
    }

    ArrayRef<Value> to_vector() const { return ivs_val_flag_; }

    // Indices of the currently selected value.
    ArrayRef<Value> ivs() const { return to_vector().drop_back(2); }
    // Currently selected value w.r.t. select() function.
    Value value() const { return ivs_val_flag_.end()[-2]; }
    // i1 flag if value() and ivs() were initialized.
    Value is_init() const { return ivs_val_flag_.back(); }

   private:
    // Vector that stores iv_1, ..., iv_N, value, init.
    SmallVector<Value, 4> ivs_val_flag_;
  };

  SmallVector<Value, 2> SelectIvs(xla_lhlo::SelectAndScatterOp s_and_s_op,
                                  scf::ParallelOp loop_over_src,
                                  OpBuilder* b) const {
    auto loc = s_and_s_op.getLoc();

    WindowLoops window_loops = InsertWindowLoops(s_and_s_op, loop_over_src, b);
    auto inner_loop_b =
        OpBuilder::atBlockEnd(window_loops.inner_loop.getBody());

    // Compute ivs in 'arg' buffer and whether these ivs are in the pad area.
    MappedIvs mapped_ivs =
        MapWindowIvsToInput(s_and_s_op, loop_over_src.getInductionVars(),
                            window_loops.window_ivs, &inner_loop_b);

    IterArgs ivs_val_flag(window_loops.inner_loop.getRegionIterArgs());

    auto if_in_bounds = inner_loop_b.create<scf::IfOp>(
        loc, window_loops.inner_loop.getResultTypes(), mapped_ivs.in_bounds,
        /*withElseRegion=*/true);

    // Case when we are inside boundaries of 'arg' and not in the pad area.
    {
      OpBuilder in_bounds_then_b = if_in_bounds.getThenBodyBuilder();
      auto select_or_init_results = SelectOrInitialize(
          s_and_s_op, mapped_ivs.ivs, &ivs_val_flag, &in_bounds_then_b);
      in_bounds_then_b.create<scf::YieldOp>(loc, select_or_init_results);
    }

    // Case when we are in the pad.
    {
      OpBuilder in_bounds_else_b = if_in_bounds.getElseBodyBuilder();
      in_bounds_else_b.create<scf::YieldOp>(loc, ivs_val_flag.to_vector());
    }

    inner_loop_b.create<scf::YieldOp>(loc, if_in_bounds.getResults());
    return window_loops.selected_ivs;
  }

  SmallVector<Value, 4> SelectOrInitialize(
      xla_lhlo::SelectAndScatterOp s_and_s_op, ArrayRef<Value> operand_ivs,
      IterArgs* ivs_val_flag, OpBuilder* b) const {
    auto loc = s_and_s_op.getLoc();
    Value true_i1 = b->create<mlir::ConstantOp>(
        loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));

    TypeRange iter_arg_types{ivs_val_flag->to_vector()};
    Value operand_elem =
        b->create<LoadOp>(loc, s_and_s_op.operand(), operand_ivs);
    auto if_init =
        b->create<scf::IfOp>(loc, iter_arg_types, ivs_val_flag->is_init(),
                             /*withElseRegion=*/true);
    // Init == true, i.e. iter args are already initialized with a selected
    // element in boundaries of the operand. Select function has to be computed
    // here.
    {
      OpBuilder if_init_then_b = if_init.getThenBodyBuilder();

      auto& lhlo_select = s_and_s_op.select().front();
      Value pred =
          ApplySingleResultLhloCode(loc, {operand_elem, ivs_val_flag->value()},
                                    &lhlo_select, &if_init_then_b);

      auto if_pred = if_init_then_b.create<scf::IfOp>(loc, iter_arg_types, pred,
                                                      /*withElseRegion=*/true);

      // Pred == true, therefore pack newly selected ivs, val and init flag back
      // to iter_args and return.
      {
        OpBuilder if_pred_then_b = if_pred.getThenBodyBuilder();
        if_pred_then_b.create<scf::YieldOp>(
            loc, IterArgs{operand_ivs, operand_elem, true_i1}.to_vector());
      }

      // Pred == false, therefore return old iter_args.
      {
        OpBuilder if_pred_else_b = if_pred.getElseBodyBuilder();
        if_pred_else_b.create<scf::YieldOp>(loc, ivs_val_flag->to_vector());
      }

      if_init_then_b.create<scf::YieldOp>(loc, if_pred.getResults());
    }
    // Init == false, i.e. only pad was visited before and this is the first
    // element in the boundaries of the operand.
    {
      OpBuilder if_init_else_b = if_init.getElseBodyBuilder();

      if_init_else_b.create<scf::YieldOp>(
          loc, IterArgs{operand_ivs, operand_elem, true_i1}.to_vector());
    }
    return if_init.getResults();
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
        ReduceWindowOpConverter,
        SelectAndScatterOpConverter
      >(func.getContext());
    // clang-format on

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           scf::SCFDialect, XlaLhloDialect>();
    target.addIllegalOp<xla_lhlo::ReduceOp, xla_lhlo::ReduceWindowOp,
                        xla_lhlo::SelectAndScatterOp>();

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
