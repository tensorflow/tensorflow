/* Copyright 2020 The OpenXLA Authors.

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

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace lmhlo {

#define GEN_PASS_DEF_LHLOLEGALIZETOPARALLELLOOPSPASS
#include "lhlo/transforms/lmhlo_passes.h.inc"

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
Value applySingleResultLhloCode(Location loc, ValueRange operands,
                                Block* lhloBlock, OpBuilder* b) {
  SmallVector<Value, 2> argBufs;
  for (auto argType : lhloBlock->getArgumentTypes()) {
    argBufs.push_back(
        b->create<memref::AllocOp>(loc, argType.cast<MemRefType>()));
  }
  for (const auto& operand : llvm::enumerate(operands)) {
    b->create<memref::StoreOp>(loc, operand.value(), argBufs[operand.index()]);
  }
  // Clone the ops from `lhlo_block`.
  IRMapping mapping;
  mapping.map(lhloBlock->getArguments(), argBufs);
  for (auto& nested : lhloBlock->without_terminator()) {
    auto* clone = b->clone(nested, mapping);
    mapping.map(nested.getResults(), clone->getResults());
  }
  return b->create<memref::LoadOp>(loc, argBufs.back());
}

// Converts a block with LHLO ops and with signature:
//   ^bb(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
// into a reduction operator of scf.reduce by doing buffer allocation for
// scalar arguments and the result of `scf.reduce` to make it compatible with
// LHLO ops.
void convertToReductionOperator(Location loc, Block& loopReduceOpBody,
                                Block* lhloBlock, OpBuilder* b) {
  OpBuilder::InsertionGuard guard(*b);
  b->setInsertionPointToStart(&loopReduceOpBody);
  b->create<scf::ReduceReturnOp>(
      loc, applySingleResultLhloCode(loc, loopReduceOpBody.getArguments(),
                                     lhloBlock, b));
}

// Returns result of arith::ConstantOp if `dim` is static, otherwise uses DimOp
// to extract dimension at runtime.
Value getStaticOrDynamicDim(mlir::Location loc, Value shapedValue,
                            size_t dimIndex, int64_t dim, OpBuilder* b) {
  return dim == ShapedType::kDynamic
             ? (Value)b->create<memref::DimOp>(loc, shapedValue, dimIndex)
             : (Value)b->create<arith::ConstantIndexOp>(loc, dim);
}

struct MappedIvs {
  // False if the mapped indices are in the padding area, true otherwise.
  Value inBounds;
  // Mapped indices.
  SmallVector<Value, 2> ivs;
};

template <typename OpTy>
MappedIvs mapWindowIvsToInput(OpTy op, Value operand, ValueRange ivs,
                              ValueRange windowIvs, OpBuilder* b) {
  MappedIvs mappedIvs;

  if (!op.getWindowStrides().has_value()) {
    op.emitOpError("No window strides specified.");
  }
  auto windowStrides = op.getWindowStrides().value();

  if (!op.getPadding().has_value()) {
    op.emitOpError("No padding specified.");
  }
  auto padding = op.getPadding().value();

  auto loc = op.getLoc();
  auto operandShape = operand.getType().template cast<MemRefType>().getShape();

  // `in_bounds` is false when the mapped indices are in the padding area.
  mappedIvs.inBounds = b->create<mlir::arith::ConstantOp>(
      loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));
  for (unsigned i = 0, e = ivs.size(); i < e; ++i) {
    auto stride = windowStrides.template getValues<llvm::APInt>()[i];
    auto padLow = padding.template getValues<llvm::APInt>()[{i, 0}];

    Value strideVal =
        b->create<arith::ConstantIndexOp>(loc, stride.getSExtValue());
    Value padLowVal =
        b->create<arith::ConstantIndexOp>(loc, padLow.getSExtValue());

    Value center = b->create<arith::MulIOp>(loc, ivs[i], strideVal);
    Value offset = b->create<arith::SubIOp>(loc, windowIvs[i], padLowVal);
    Value index = b->create<arith::AddIOp>(loc, center, offset);
    Value upperBound =
        getStaticOrDynamicDim(loc, operand, i, operandShape[i], b);
    // We must check whether 0 <= index_i < shape_i, as otherwise we are in
    // the pad and then we have to use the neutral element for reduction.
    // Equivalently, it can be computed as the unsigned comparison index_i <
    // shape_i, since a negative value wraps to a large positive value.
    mappedIvs.inBounds = b->create<mlir::arith::AndIOp>(
        loc, mappedIvs.inBounds,
        b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, index,
                                 upperBound));
    mappedIvs.ivs.push_back(index);
  }
  return mappedIvs;
}

// Returns scf::Parallel over a shaped value with static or dynamic shape.
scf::ParallelOp makeLoopOverShape(Location loc, Value shapedValue,
                                  OpBuilder* b) {
  Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
  Value one = b->create<arith::ConstantIndexOp>(loc, 1);

  ArrayRef<int64_t> shape = shapedValue.getType().cast<ShapedType>().getShape();
  SmallVector<Value, 2> lower, upper, step;
  for (const auto& dim : llvm::enumerate(shape)) {
    upper.push_back(
        getStaticOrDynamicDim(loc, shapedValue, dim.index(), dim.value(), b));
    lower.push_back(zero);
    step.push_back(one);
  }
  return b->create<scf::ParallelOp>(loc, lower, upper, step);
}

// Converts `lmhlo.ReduceOp` into two scf::ParallelOp and a scf::ReduceOp.
// The outper `ParallelOp` refers to the parallel loops if there are
// any. The inner `ParalleOp` refers to the reduction loops and `ReduceOp`
// contains the reduction operator.
//
// Example:
//
//  "lmhlo.reduce"(%buffer, %init_buf, %result) ({
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
//        ^bb0(%elem: f32, %acc: f32):
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
class ReduceOpConverter : public OpConversionPattern<lmhlo::ReduceOp> {
 public:
  using OpConversionPattern<lmhlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ReduceOp reduceOp, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
    // TODO(b/183977252) : Handle variadic ReduceOp/ReduceWindowOp
    if (reduceOp.getOut().size() != 1) return failure();

    scf::ReduceOp scfReduceOp =
        createReduceOpInNestedParallelLoops(reduceOp, &rewriter);
    convertToReductionOperator(reduceOp.getLoc(),
                               scfReduceOp.getReductions().front().front(),
                               &reduceOp.getBody().front(), &rewriter);
    rewriter.replaceOp(reduceOp, std::nullopt);
    return success();
  }

 private:
  // Creates nested `scf.parallel` ops with `scf.reduce`. The outer ParallelOp
  // refers to the parallel dimensions of `reduce_op` if any and the inner
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
  scf::ReduceOp createReduceOpInNestedParallelLoops(
      lmhlo::ReduceOp reduceOp, ConversionPatternRewriter* rewriter) const {
    auto loc = reduceOp.getLoc();
    DenseSet<int> reducingDims;
    for (const auto& rdim : reduceOp.getDimensions().getValues<APInt>()) {
      reducingDims.insert(rdim.getSExtValue());
    }

    Value operand = reduceOp.getInputs().front();
    Value out = reduceOp.getOut().front();
    SmallVector<Value, 2> parallelLower, parallelUpper, parallelStep;
    SmallVector<Value, 2> reduceLower, reduceUpper, reduceStep;
    auto operandShape = operand.getType().cast<MemRefType>().getShape();
    for (const auto& dim : llvm::enumerate(operandShape)) {
      const bool isReducingDim = reducingDims.count(dim.index());

      Value ub = getStaticOrDynamicDim(loc, operand, dim.index(), dim.value(),
                                       rewriter);
      Value lb = rewriter->create<arith::ConstantIndexOp>(loc, 0);
      Value step = rewriter->create<arith::ConstantIndexOp>(loc, 1);
      (isReducingDim ? reduceLower : parallelLower).push_back(lb);
      (isReducingDim ? reduceUpper : parallelUpper).push_back(ub);
      (isReducingDim ? reduceStep : parallelStep).push_back(step);
    }
    // Load initial value from memref<element_type>.
    SmallVector<Value, 1> initValue = {rewriter->create<memref::LoadOp>(
        loc, *reduceOp.getInitValues().begin())};
    // Outer ParallelOp is not needed if it is a reduction across all dims.
    scf::ParallelOp outer;
    if (!parallelLower.empty()) {
      outer = rewriter->create<scf::ParallelOp>(loc, parallelLower,
                                                parallelUpper, parallelStep);
      rewriter->setInsertionPointToStart(outer.getBody());
    }
    scf::ParallelOp inner = rewriter->create<scf::ParallelOp>(
        loc, reduceLower, reduceUpper, reduceStep, ValueRange(initValue));
    Value reductionResult = *inner.getResults().begin();

    SmallVector<Value, 1> outIndices;
    if (outer != nullptr) {
      outIndices.reserve(outer.getNumLoops());
      for (Value iv : outer.getInductionVars()) {
        outIndices.push_back(iv);
      }
    } else {
      outIndices.push_back(rewriter->create<arith::ConstantIndexOp>(loc, 0));
    }

    rewriter->create<memref::StoreOp>(loc, reductionResult, out, outIndices);

    // Load the element to reduce.
    SmallVector<Value, 2> indices;
    indices.reserve(operandShape.size());

    if (outer) {
      auto innerIvsIt = inner.getInductionVars().begin();
      auto outerIvsIt = outer.getInductionVars().begin();
      for (unsigned i = 0, e = operandShape.size(); i < e; ++i) {
        indices.push_back(reducingDims.count(i) ? *innerIvsIt++
                                                : *outerIvsIt++);
      }
    } else {
      indices = inner.getInductionVars();
    }

    rewriter->setInsertionPointToStart(inner.getBody());
    Value elem = rewriter->create<mlir::memref::LoadOp>(
        loc, reduceOp.getInputs().front(), indices);
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
//     accumulator = reduction_operator(accumulator, value)
//   output[O] = accumulator
//
// Converts `lmhlo.ReduceWindowOp` into two scf::ParallelOp and a
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
//   "lmhlo.reduce_window"(%arg, %init, %result) ({
//     ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
//       "lmhlo.maximum"(%lhs, %rhs, %res)
//         : (memref<f32>, memref<f32>, memref<f32>) -> ()
//       "lmhlo.terminator"() : () -> ()
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
    : public OpConversionPattern<lmhlo::ReduceWindowOp> {
 public:
  using OpConversionPattern<lmhlo::ReduceWindowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ReduceWindowOp reduceWindowOp, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
    // TODO(b/183977252) : Handle variadic ReduceOp/ReduceWindowOp
    if (reduceWindowOp.getOut().size() != 1) return failure();

    scf::ParallelOp outputLoop, windowLoop;
    std::tie(outputLoop, windowLoop) =
        createParallelLoopsToTraverseOutputAndWindow(reduceWindowOp, &rewriter);

    scf::ReduceOp reduceOp = createReduceOpInNestedParallelLoops(
        reduceWindowOp, outputLoop, windowLoop, &rewriter);

    convertToReductionOperator(reduceWindowOp.getLoc(),
                               reduceOp.getReductions().front().front(),
                               &reduceWindowOp.getBody().front(), &rewriter);
    rewriter.replaceOp(reduceWindowOp, std::nullopt);
    return success();
  }

 private:
  std::pair<scf::ParallelOp, scf::ParallelOp>
  createParallelLoopsToTraverseOutputAndWindow(
      lmhlo::ReduceWindowOp reduceWindowOp,
      ConversionPatternRewriter* rewriter) const {
    auto loc = reduceWindowOp.getLoc();
    Value initValue = rewriter->create<memref::LoadOp>(
        loc, reduceWindowOp.getInitValues()[0]);

    Value zero = rewriter->create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter->create<arith::ConstantIndexOp>(loc, 1);

    // Create an outer parallel loop that spans the output of ReduceWindowOp.
    Value output = reduceWindowOp.getOut()[0];
    auto outputLoop = makeLoopOverShape(loc, output, rewriter);

    // Create a nested loop that traverses the window.
    SmallVector<Value, 2> windowLower, windowUpper, windowStep;
    rewriter->setInsertionPointToStart(outputLoop.getBody());
    for (const auto& windowDim : reduceWindowOp.getWindowDimensions()) {
      windowStep.push_back(one);
      windowLower.push_back(zero);
      windowUpper.push_back(rewriter->create<arith::ConstantIndexOp>(
          loc, windowDim.getSExtValue()));
    }
    auto windowLoop = rewriter->create<scf::ParallelOp>(
        loc, windowLower, windowUpper, windowStep, ValueRange(initValue));

    Value reductionResult = *windowLoop.getResults().begin();
    auto outputIvs = outputLoop.getInductionVars();
    rewriter->create<memref::StoreOp>(loc, reductionResult, output, outputIvs);
    return std::make_pair(outputLoop, windowLoop);
  }

  scf::ReduceOp createReduceOpInNestedParallelLoops(
      lmhlo::ReduceWindowOp reduceWindowOp, scf::ParallelOp outputLoop,
      scf::ParallelOp windowLoop, ConversionPatternRewriter* rewriter) const {
    rewriter->setInsertionPointToStart(windowLoop.getBody());
    auto loc = reduceWindowOp.getLoc();

    if (reduceWindowOp.getBaseDilations().has_value() ||
        reduceWindowOp.getWindowDilations().has_value()) {
      reduceWindowOp.emitRemark(
          "Lowering to parallel loops does not support `base_dilations` or "
          "`window_dilations` attributes yet. The attributes will be ignored.");
    }

    Value input = reduceWindowOp.getInputs()[0];
    auto inputType = input.getType().cast<MemRefType>();

    // Compute ivs in 'arg' buffer and whether these ivs are in pad area or not.
    MappedIvs mappedIvs = mapWindowIvsToInput(
        reduceWindowOp, input, outputLoop.getInductionVars(),
        windowLoop.getInductionVars(), rewriter);

    auto elemOrInit = rewriter->create<scf::IfOp>(
        loc, inputType.getElementType(), mappedIvs.inBounds,
        /*withElseRegion=*/true);

    OpBuilder thenBuilder =
        elemOrInit.getThenBodyBuilder(rewriter->getListener());
    Value elem =
        thenBuilder.create<mlir::memref::LoadOp>(loc, input, mappedIvs.ivs);
    thenBuilder.create<scf::YieldOp>(loc, elem);

    OpBuilder elseBuilder =
        elemOrInit.getElseBodyBuilder(rewriter->getListener());
    elseBuilder.create<scf::YieldOp>(loc, *windowLoop.getInitVals().begin());

    return rewriter->create<scf::ReduceOp>(loc,
                                           *elemOrInit.getResults().begin());
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
    : public OpConversionPattern<lmhlo::SelectAndScatterOp> {
 public:
  using OpConversionPattern<lmhlo::SelectAndScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::SelectAndScatterOp sAndSOp, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = sAndSOp.getLoc();
    initializeOutput(sAndSOp, &rewriter);
    scf::ParallelOp loopOverSrc =
        makeLoopOverShape(loc, sAndSOp.getSource(), &rewriter);
    rewriter.setInsertionPointToStart(loopOverSrc.getBody());

    // Compute indices of the selected element in the window.
    auto selectedIvs = selectIvs(sAndSOp, loopOverSrc, &rewriter);

    // Load `source[selected_ivs]`.
    auto srcElem = rewriter.create<memref::LoadOp>(
        loc, sAndSOp.getSource(), loopOverSrc.getInductionVars());

    // Compute `out[selected_ivs]` = scatter(out[selected_ivs], src_element)`.
    auto rmw = rewriter.create<memref::GenericAtomicRMWOp>(
        loc, sAndSOp.getOut(), selectedIvs);
    OpBuilder rmwBuilder = OpBuilder::atBlockEnd(rmw.getBody());
    auto accResult =
        applySingleResultLhloCode(loc, {srcElem, rmw.getCurrentValue()},
                                  &sAndSOp.getScatter().front(), &rmwBuilder);
    rmwBuilder.create<memref::AtomicYieldOp>(loc, accResult);

    rewriter.replaceOp(sAndSOp, std::nullopt);
    return success();
  }

 private:
  void initializeOutput(lmhlo::SelectAndScatterOp sAndSOp, OpBuilder* b) const {
    auto loc = sAndSOp.getLoc();
    Value initValue = b->create<memref::LoadOp>(loc, sAndSOp.getInitValue());

    scf::ParallelOp loopOverOutput =
        makeLoopOverShape(loc, sAndSOp.getOut(), b);
    OpBuilder::InsertionGuard guard(*b);
    b->setInsertionPointToStart(loopOverOutput.getBody());
    b->create<memref::StoreOp>(loc, initValue, sAndSOp.getOut(),
                               loopOverOutput.getInductionVars());
  }

  struct WindowLoops {
    SmallVector<Value, 2> selectedIvs;
    SmallVector<Value, 2> windowIvs;
    scf::ForOp innerLoop;
  };
  WindowLoops insertWindowLoops(lmhlo::SelectAndScatterOp sAndSOp,
                                scf::ParallelOp loopOverSrc,
                                OpBuilder* b) const {
    auto loc = sAndSOp.getLoc();
    Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
    Value one = b->create<arith::ConstantIndexOp>(loc, 1);

    auto elementType =
        sAndSOp.getOut().getType().cast<MemRefType>().getElementType();
    auto rank = loopOverSrc.getNumLoops();

    // `iter_args` = [iv_1, ..., iv_N, selected_value, is_initialized]
    SmallVector<Value, 4> iterArgs(rank, zero);
    iterArgs.push_back(b->create<mlir::arith::ConstantOp>(
        loc, elementType, b->getFloatAttr(elementType, 0)));
    iterArgs.push_back(b->create<mlir::arith::ConstantOp>(
        loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 0)));

    // Create a nested loop that traverses the window.
    OpBuilder::InsertPoint ip;
    WindowLoops result;
    for (const auto& windowDim :
         sAndSOp.getWindowDimensions()->getValues<APInt>()) {
      Value upper =
          b->create<arith::ConstantIndexOp>(loc, windowDim.getSExtValue());
      result.innerLoop = b->create<scf::ForOp>(loc, zero, upper, one, iterArgs);
      if (b->getInsertionBlock() == loopOverSrc.getBody()) {
        ip = b->saveInsertionPoint();
        result.selectedIvs = result.innerLoop.getResults().take_front(rank);
      } else {
        b->create<scf::YieldOp>(loc, result.innerLoop.getResults());
      }
      b->setInsertionPointToStart(result.innerLoop.getBody());
      iterArgs = ValueRange{result.innerLoop.getRegionIterArgs()};
      result.windowIvs.push_back(result.innerLoop.getInductionVar());
    }
    b->restoreInsertionPoint(ip);
    return result;
  }

  // Adapter to store iteration arguments of sequential loops that perform
  // select in a window.
  class IterArgs {
   public:
    explicit IterArgs(ValueRange ivsValFlag) : ivsValFlag(ivsValFlag) {}
    IterArgs(ValueRange ivs, Value value, Value flag) {
      ivsValFlag = ivs;
      ivsValFlag.push_back(value);
      ivsValFlag.push_back(flag);
    }

    ArrayRef<Value> toVector() const { return ivsValFlag; }

    // Indices of the currently selected value.
    ArrayRef<Value> ivs() const { return toVector().drop_back(2); }
    // Currently selected value w.r.t. select() function.
    Value value() const { return ivsValFlag.end()[-2]; }
    // i1 flag if value() and ivs() were initialized.
    Value isInit() const { return ivsValFlag.back(); }

   private:
    // Vector that stores iv_1, ..., iv_N, value, init.
    SmallVector<Value, 4> ivsValFlag;
  };

  SmallVector<Value, 2> selectIvs(lmhlo::SelectAndScatterOp sAndSOp,
                                  scf::ParallelOp loopOverSrc,
                                  OpBuilder* b) const {
    auto loc = sAndSOp.getLoc();

    WindowLoops windowLoops = insertWindowLoops(sAndSOp, loopOverSrc, b);
    auto innerLoopB = OpBuilder::atBlockEnd(windowLoops.innerLoop.getBody());

    // Compute ivs in 'arg' buffer and whether these ivs are in the pad area.
    MappedIvs mappedIvs = mapWindowIvsToInput(
        sAndSOp, sAndSOp.getOperand(), loopOverSrc.getInductionVars(),
        windowLoops.windowIvs, &innerLoopB);

    IterArgs ivsValFlag(windowLoops.innerLoop.getRegionIterArgs());

    auto ifInBounds = innerLoopB.create<scf::IfOp>(
        loc, windowLoops.innerLoop.getResultTypes(), mappedIvs.inBounds,
        /*withElseRegion=*/true);

    // Case when we are inside boundaries of 'arg' and not in the pad area.
    {
      OpBuilder inBoundsThenB = ifInBounds.getThenBodyBuilder(b->getListener());
      auto selectOrInitResults = selectOrInitialize(
          sAndSOp, mappedIvs.ivs, &ivsValFlag, &inBoundsThenB);
      inBoundsThenB.create<scf::YieldOp>(loc, selectOrInitResults);
    }

    // Case when we are in the pad.
    {
      OpBuilder inBoundsElseB = ifInBounds.getElseBodyBuilder(b->getListener());
      inBoundsElseB.create<scf::YieldOp>(loc, ivsValFlag.toVector());
    }

    innerLoopB.create<scf::YieldOp>(loc, ifInBounds.getResults());
    return windowLoops.selectedIvs;
  }

  SmallVector<Value, 4> selectOrInitialize(lmhlo::SelectAndScatterOp sAndSOp,
                                           ArrayRef<Value> operandIvs,
                                           IterArgs* ivsValFlag,
                                           OpBuilder* b) const {
    auto loc = sAndSOp.getLoc();
    Value trueI1 = b->create<mlir::arith::ConstantOp>(
        loc, b->getI1Type(), b->getIntegerAttr(b->getI1Type(), 1));

    const TypeRange iterArgTypes{ValueRange{ivsValFlag->toVector()}};
    Value operandElem =
        b->create<memref::LoadOp>(loc, sAndSOp.getOperand(), operandIvs);
    auto ifInit = b->create<scf::IfOp>(loc, iterArgTypes, ivsValFlag->isInit(),
                                       /*withElseRegion=*/true);
    // Init == true, i.e. iter args are already initialized with a selected
    // element in boundaries of the operand. Select function has to be computed
    // here.
    {
      OpBuilder ifInitThenB = ifInit.getThenBodyBuilder(b->getListener());

      auto& lhloSelect = sAndSOp.getSelect().front();
      Value pred = applySingleResultLhloCode(
          loc, {operandElem, ivsValFlag->value()}, &lhloSelect, &ifInitThenB);

      auto ifPred = ifInitThenB.create<scf::IfOp>(loc, iterArgTypes, pred,
                                                  /*withElseRegion=*/true);

      // Pred == true, therefore pack newly selected ivs, val and init flag back
      // to iter_args and return.
      {
        OpBuilder ifPredThenB = ifPred.getThenBodyBuilder(b->getListener());
        ifPredThenB.create<scf::YieldOp>(
            loc, IterArgs{operandIvs, operandElem, trueI1}.toVector());
      }

      // Pred == false, therefore return old iter_args.
      {
        OpBuilder ifPredElseB = ifPred.getElseBodyBuilder(b->getListener());
        ifPredElseB.create<scf::YieldOp>(loc, ivsValFlag->toVector());
      }

      ifInitThenB.create<scf::YieldOp>(loc, ifPred.getResults());
    }
    // Init == false, i.e. only pad was visited before and this is the first
    // element in the boundaries of the operand.
    {
      OpBuilder ifInitElseB = ifInit.getElseBodyBuilder(b->getListener());

      ifInitElseB.create<scf::YieldOp>(
          loc, IterArgs{operandIvs, operandElem, trueI1}.toVector());
    }
    return ifInit.getResults();
  }
};

struct LhloLegalizeToParallelLoopsPass
    : public impl::LhloLegalizeToParallelLoopsPassBase<
          LhloLegalizeToParallelLoopsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<
        ReduceOpConverter,
        ReduceWindowOpConverter,
        SelectAndScatterOpConverter
      >(func.getContext());
    // clang-format on

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           scf::SCFDialect, LmhloDialect>();
    target.addIllegalOp<lmhlo::ReduceOp, lmhlo::ReduceWindowOp,
                        lmhlo::SelectAndScatterOp>();

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeLhloToParallelLoopsPass() {
  return std::make_unique<LhloLegalizeToParallelLoopsPass>();
}

}  // namespace lmhlo
}  // namespace mlir
