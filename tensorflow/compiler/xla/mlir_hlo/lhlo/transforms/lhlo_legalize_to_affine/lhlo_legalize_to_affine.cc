/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering LHLO dialect to Affine dialect.

#include <utility>

#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace lmhlo {

#define GEN_PASS_DEF_LHLOLEGALIZETOAFFINEPASS
#include "lhlo/transforms/lmhlo_passes.h.inc"

namespace {

// Builds an affine loop nest iterating from zeros to "upper_bounds" with unit
// steps, and populates the body of the innermost loop using "body_builder".
static void buildBoundedAffineLoopNest(
    OpBuilder& builder, Location location, ArrayRef<int64_t> upperBounds,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilder) {
  SmallVector<int64_t, 3> lowerBounds(upperBounds.size(), /*Value=*/0);
  SmallVector<int64_t, 3> steps(upperBounds.size(), /*Value=*/1);
  buildAffineLoopNest(builder, location, lowerBounds, upperBounds, steps,
                      bodyBuilder);
}

struct DotOpConverter : public OpRewritePattern<DotOp> {
  using OpRewritePattern<DotOp>::OpRewritePattern;

  // Supports only rank-2 tensors for LHS and RHS.
  LogicalResult matchAndRewrite(DotOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    MemRefType lhsType = lhs.getType().cast<MemRefType>();
    MemRefType rhsType = rhs.getType().cast<MemRefType>();
    Type elementType = lhsType.getElementType();
    ArrayRef<int64_t> shapeLhs = lhsType.getShape();
    ArrayRef<int64_t> shapeRhs = rhsType.getShape();

    if ((lhsType.getRank() != 2) || (rhsType.getRank() != 2)) {
      return failure();
    }

    // We don't currently support batching dimensions, or multiple contraction
    // dimensions.
    mhlo::DotDimensionNumbersAttr dotDimensionNumbers =
        op.getDotDimensionNumbers();
    if (!dotDimensionNumbers.getLhsBatchingDimensions().empty() ||
        !dotDimensionNumbers.getRhsBatchingDimensions().empty())
      return failure();
    if (dotDimensionNumbers.getLhsContractingDimensions().size() != 1 ||
        *dotDimensionNumbers.getLhsContractingDimensions().begin() != 1 ||
        dotDimensionNumbers.getRhsContractingDimensions().size() != 1 ||
        *dotDimensionNumbers.getRhsContractingDimensions().begin() != 0) {
      return failure();
    }

    LogicalResult mapStatus = success();
    auto bodyBuilder = [&](OpBuilder& builder, Location loc, ValueRange ivs) {
      SmallVector<Value, 2> lhsIndices{ivs[0], ivs[2]},
          rhsIndices{ivs[2], ivs[1]}, resultIndices{ivs[0], ivs[1]};

      auto l = builder.create<AffineLoadOp>(loc, lhs, lhsIndices);
      auto r = builder.create<AffineLoadOp>(loc, rhs, rhsIndices);
      auto result =
          rewriter.create<AffineLoadOp>(loc, op.getOutput(), resultIndices);
      Value opResult = lmhlo::LhloOpToStdScalarOp::map<DotOp>(
          op, elementType, {l, r, result}, &builder);
      mapStatus = success(opResult != nullptr);
      if (failed(mapStatus)) return;
      builder.create<AffineStoreOp>(loc, opResult, op.getOutput(),
                                    resultIndices);
    };

    buildBoundedAffineLoopNest(rewriter, op.getLoc(),
                               {shapeLhs[0], shapeRhs[1], shapeRhs[0]},
                               bodyBuilder);
    if (failed(mapStatus)) return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// Concat Operation Example (2D):
/// Given inpA[2][1], inpB[2][2], concat_dimension = 1.
/// Compute output[x1][x2].
/// Implementation Pseudocode:
/// s = 0
/// for a in range(0, 2):
///   for b in range(0, 1):
///     output[a][b] = inpA[a][b - s]
/// s = 1
/// for a in range(0, 2):
///   for b in range(1, 3):
///     output[a][b] = inpB[a][b - s]
///
/// Concatenate composes an array from multiple array operands. The array is of
/// the same rank as each of the input array operands (which must be of the same
/// rank as each other) and contains the arguments in the order that they were
/// specified.
struct ConcatOpConverter : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern<ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value output = op.getOutput();
    MemRefType outputType = output.getType().cast<MemRefType>();
    unsigned outputRank = outputType.getRank();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    ValueRange operands = op.getVal();
    uint64_t concatDim = op.getDimension();
    int64_t prevBound = 0;

    for (Value operand : operands) {
      MemRefType operandType = operand.getType().cast<MemRefType>();
      ArrayRef<int64_t> operandShape = operandType.getShape();

      // TODO(pashu123): Extend support for dynamic dimensions.
      if (!operandType.hasStaticShape()) return failure();

      // Only for the concatenation dimension, the value is dimension -
      // prevBound.
      SmallVector<AffineExpr, 4> expr;
      for (unsigned i = 0; i < outputRank; i++) {
        AffineExpr d0 = (i == concatDim)
                            ? rewriter.getAffineDimExpr(concatDim) - prevBound
                            : rewriter.getAffineDimExpr(i);

        expr.push_back(d0);
      }
      AffineMap map =
          AffineMap::get(outputRank, 0, expr, rewriter.getContext());

      // Create multiple for loop nests iterating along the concatenation
      // dimension.
      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value, 3> indices;
      AffineForOp forOp;
      for (unsigned i = 0; i < outputRank; i++) {
        if (i == concatDim) {
          forOp = rewriter.create<AffineForOp>(loc, prevBound,
                                               prevBound + operandShape[i]);
          prevBound += operandShape[i];
          indices.push_back(forOp.getInductionVar());
        } else {
          forOp = rewriter.create<AffineForOp>(loc, 0, outputShape[i]);
          indices.push_back(forOp.getInductionVar());
        }
        rewriter.setInsertionPointToStart(forOp.getBody());
      }
      Value storeVal =
          rewriter.create<AffineLoadOp>(loc, operand, map, indices);
      rewriter.create<AffineStoreOp>(loc, storeVal, output, indices);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Returns a zero value of type `type`. `type` is expected to be either
/// int or float.
static Value getZeroValue(Type type, Location loc, PatternRewriter& rewriter) {
  assert(type.isIntOrFloat() && "Expected int or float");

  if (IntegerType intType = type.dyn_cast<IntegerType>())
    return rewriter.create<mlir::arith::ConstantIntOp>(loc, 0,
                                                       intType.getWidth());

  FloatType floatType = type.cast<FloatType>();
  return rewriter.create<mlir::arith::ConstantFloatOp>(
      loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
}

/// Emits a nest to fill the given `buffer` of memref type with `fillValue`.
static void fillBuffer(Location loc, Value buffer, Value fillValue,
                       PatternRewriter& builder) {
  OpBuilder::InsertionGuard guard(builder);
  MemRefType bufType = buffer.getType().cast<MemRefType>();
  unsigned rank = bufType.getRank();
  SmallVector<Value, 4> dimSizes;
  dimSizes.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dimSizes.push_back(builder.create<mlir::memref::DimOp>(loc, buffer, i));

  AffineMap idSymMap = builder.getSymbolIdentityMap();
  AffineMap lbMap = builder.getConstantAffineMap(0);
  SmallVector<Value, 4> ivs(rank);
  AffineForOp forOp;
  for (unsigned i = 0; i < rank; ++i) {
    forOp = builder.create<AffineForOp>(loc, llvm::None, lbMap, dimSizes[i],
                                        idSymMap);
    builder.setInsertionPointToStart(forOp.getBody());
    ivs[i] = forOp.getInductionVar();
  }
  Type fillValueType = fillValue.getType();
  auto fillMemRefType = fillValueType.dyn_cast<MemRefType>();
  assert(((fillMemRefType && fillMemRefType.getRank() == 0) ||
          fillValueType.isIntOrFloat()) &&
         "init value has to be a 0-d memref or int or fp");
  Value initVal = fillMemRefType ? builder.create<AffineLoadOp>(
                                       loc, fillValue, /*indices=*/llvm::None)
                                 : fillValue;
  builder.create<AffineStoreOp>(loc, initVal, buffer, ivs);
}

/// Converts GatherOp to Affine nest form.
/// Pseudocode:
///   1. Fill a temporary output tensor with 0.
///   2. Repeat the following for each batch dimension :-
///      1. For each indices in 'operand' :-
///        1. Get hold of start indices from 'start_indices'.
///        2. Add offset to the start indices to get the final indices.
///        3. Load value from 'operand' tensor : 'operand_val'.
///        4. Load value from temporary output : 'prev_val'.
///        5. If the final indices match current indices of 'operand' :
///             'prev_val' = 'prev_val' + 'operand_val'
///        6. Store 'prev_val' back to the temporary output.
class GatherOpConverter : public OpRewritePattern<GatherOp> {
 public:
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter& rewriter) const final {
    Location loc = op.getLoc();

    // Operand array.
    Value operand = op.getOperand();
    MemRefType operandType = operand.getType().cast<MemRefType>();
    unsigned operandRank = operandType.getRank();
    ArrayRef<int64_t> operandShape = operandType.getShape();

    // Start_indices array.
    Value startIndices = op.getStartIndices();
    MemRefType startIndicesType = startIndices.getType().cast<MemRefType>();
    unsigned startIndicesRank = startIndicesType.getRank();
    ArrayRef<int64_t> startIndicesShape = startIndicesType.getShape();

    // Output array.
    Value output = op.getOutput();
    MemRefType outputType = output.getType().cast<MemRefType>();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    if (!operandType.hasStaticShape() || !startIndicesType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static shaped type allowed");

    mhlo::GatherDimensionNumbersAttr gatherDim = op.getDimensionNumbers();

    auto collapsedSliceDims = gatherDim.getCollapsedSliceDims();
    auto offsetDims = gatherDim.getOffsetDims();
    auto startIndexMap = gatherDim.getStartIndexMap();
    int64_t indexVectorDim = gatherDim.getIndexVectorDim();

    // Slice_sizes.
    DenseIntElementsAttr sliceSizesAttr = op.getSliceSizesAttr();
    SmallVector<int64_t, 4> sliceSizes;
    for (const APInt& dim : sliceSizesAttr.getValues<APInt>())
      sliceSizes.push_back(dim.getSExtValue());

    // Creating constants with 0 value. We need the Integer type constant value
    // because the indices type will be Integer.
    Value zeroIntVal = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, 0, startIndicesType.getElementType());
    Type elementType = outputType.getElementType();
    Value zeroLoadValue = getZeroValue(elementType, loc, rewriter);
    // Initializing the output buffer with 0.
    fillBuffer(loc, output, zeroLoadValue, rewriter);

    // We fetch the shape of start_indices at index_vector_dim. In case
    // index_vector_dim is equal to the rank of start_indices, we implicitly
    // consider start_indices to have a trailing 1 dimension.
    unsigned startIndicesNumbers = (indexVectorDim == startIndicesRank)
                                       ? 1
                                       : startIndicesShape[indexVectorDim];
    // We create integer constants till start_incides_index which help us to
    // fetch start_indices in affine transformation.
    SmallVector<Value, 4> startIndicesIndex;
    for (unsigned i = 0; i < startIndicesNumbers; i++) {
      Value iVal = rewriter.create<mlir::arith::ConstantIntOp>(
          loc, i, startIndicesType.getElementType());
      iVal = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                 iVal);
      startIndicesIndex.push_back(iVal);
    }

    // S_in contains the multiple indices that form a starting index used in the
    // input/operand tensor. O_in contains the multiple offsets of corresponding
    // starting index used in the input/operand tensor. We initialize both of
    // them with 0.
    SmallVector<Value, 4> sIn;
    SmallVector<Value, 4> oIn;
    Value zeroIndexVal = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), zeroIntVal);
    for (unsigned i = 0; i < operandRank; i++) {
      sIn.push_back(zeroIndexVal);
      oIn.push_back(zeroIndexVal);
    }

    // batch_induction_vars stores the loop induction variables pertaining to
    // the batches of start indices.
    SmallVector<Value, 4> batchInductionVars;
    // output_induction_vars stores the loop induction variables pertaining to
    // both batches and offsets within the output tensor.
    SmallVector<Value, 4> outputInductionVars;
    // Create loops to iterate over each batch of starting index.
    for (unsigned i = 0; i < startIndicesRank; i++) {
      // ith dimension of start_indices doesn't form a batch if it is equal to
      // index_vector_dim.
      if (i == indexVectorDim) continue;
      AffineForOp forOp =
          rewriter.create<AffineForOp>(loc, 0, startIndicesShape[i]);
      batchInductionVars.push_back(forOp.getInductionVar());
      outputInductionVars.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Create loops to iterate over each offset dimension within the output
    // tensor.
    for (unsigned i = 0, k = 0, e = offsetDims.size(); i < e; i++) {
      AffineForOp forOp =
          rewriter.create<AffineForOp>(loc, 0, outputShape[offsetDims[i]]);
      rewriter.setInsertionPointToStart(forOp.getBody());
      // We try to fetch the first non-collapsed dimension.
      while (k < collapsedSliceDims.size() && collapsedSliceDims[k] == i) k++;
      // Remapping the offset_dim[i] to the non-collapsed dimension.
      oIn[k++] = forOp.getInductionVar();
      // We assume offset_dims to be sorted. So when inserted to
      // output_induction_vars the loop induction variable gets inserted at the
      // correct position.
      outputInductionVars.insert(outputInductionVars.begin() + offsetDims[i],
                                 forOp.getInductionVar());
    }

    // Create loops to iterate over all dimensions within the operand tensor.
    SmallVector<Value, 4> operandIndex;
    for (unsigned i = 0, k = 0; i < operandRank; i++) {
      // We assume start_index_map to have sorted dimensions. We only include
      // those dimensions of operand tensor which are present in
      // start_index_map.
      if (k < startIndexMap.size() && i == startIndexMap[k++]) {
        AffineForOp forOp =
            rewriter.create<AffineForOp>(loc, 0, operandShape[i]);
        operandIndex.push_back(forOp.getInductionVar());
        rewriter.setInsertionPointToStart(forOp.getBody());
      } else {
        operandIndex.push_back(oIn[i]);
      }
    }

    // In case index_vector_dim is not equal to start_indices shape then we
    // create another loop to iterate over starting index and update
    // batch_induction_vars.
    if (indexVectorDim != startIndicesRank) {
      for (unsigned i = 0; i < startIndicesNumbers; i++) {
        batchInductionVars.insert(batchInductionVars.begin() + indexVectorDim,
                                  startIndicesIndex[i]);
        Value startIndex = rewriter.create<AffineLoadOp>(loc, startIndices,
                                                         batchInductionVars);
        startIndex = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), startIndex);
        sIn[startIndexMap[i]] = startIndex;
        batchInductionVars.erase(batchInductionVars.begin() + indexVectorDim);
      }
    } else {
      // Since index_vector_dim is equal to start_indicesRank we can directly
      // fetch the start_index from batch_induction_vars.
      Value startIndex =
          rewriter.create<AffineLoadOp>(loc, startIndices, batchInductionVars);
      startIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), startIndex);
      sIn[0] = startIndex;
    }

    // We load value at a particular operand index and populate the output
    // tensor if the index constraints match.
    Value loadValue = rewriter.create<AffineLoadOp>(loc, operand, operandIndex);
    SmallVector<Value, 4> predicates;
    // Adding offsets to the corresponding starting index and comparing it with
    // the corresponding operand index.
    for (unsigned k = 0, i = 0; k < startIndexMap.size(); k++) {
      i = startIndexMap[k];
      Value addStartIndexOffset = rewriter.create<mlir::arith::AddIOp>(
          loc, rewriter.getIndexType(), sIn[i], oIn[i]);
      Value predicate = rewriter.create<mlir::arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, addStartIndexOffset, operandIndex[i]);
      predicates.push_back(predicate);
    }

    // Since the no. of predicates is equal to start_index_map.size() we
    // iterate over pairs of predicates and join them with arith::AndIOp.
    // We store the final predicate formed by joining other predicates with
    // arith::AndIOp in result_predicate.
    Value resultPredicate = nullptr;
    for (unsigned i = 0; i < predicates.size() - 1; i += 2) {
      Value predicateA = predicates[i];
      Value predicateB = predicates[i + 1];
      Value andPredicate =
          rewriter.create<mlir::arith::AndIOp>(loc, predicateA, predicateB);
      resultPredicate = (i == 0) ? andPredicate
                                 : rewriter.create<mlir::arith::AndIOp>(
                                       loc, resultPredicate, andPredicate);
    }
    // We fetch the last predicate value. In case this is the only predicate
    // we let result_predicate be equal to this predicate value. Else if there
    // are odd number of predicates we join it to other predicates using
    // arith::AndIOp.
    Value predicate = predicates.back();
    if (!resultPredicate) resultPredicate = predicate;
    // In case there are odd number of predicates we join the last predicate
    // to the result_predicate using arith::AndIOp.
    else if (startIndexMap.size() % 2 == 1)
      resultPredicate =
          rewriter.create<mlir::arith::AndIOp>(loc, resultPredicate, predicate);

    // We use the loaded value if the index computed by adding offsets to
    // starting index is equal to the current operand index. We use 0 as a value
    // otherwise.
    Value selectLoad = rewriter.create<mlir::arith::SelectOp>(
        loc, resultPredicate, loadValue, zeroLoadValue);
    // We load value at output array.
    Value outputValue =
        rewriter.create<AffineLoadOp>(loc, output, outputInductionVars);

    // The selected value is added to the previous value stored in output array.
    if (elementType.isa<FloatType>())
      outputValue = rewriter.create<arith::AddFOp>(loc, elementType, selectLoad,
                                                   outputValue);
    else
      outputValue = rewriter.create<arith::AddIOp>(loc, elementType, selectLoad,
                                                   outputValue);
    rewriter.create<AffineStoreOp>(loc, outputValue, output,
                                   outputInductionVars);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Converts PadOp to Affine nest form.
/// Pseudocode:
///   1. Fill `output` tensor with `padding_value`.
///   2. Compute AffineMap for store into `output`.
///      out_idx = edge_padding_low +
///                operand_idx * (1 + interior_padding)
///   3. Create nested loop from `operand` shape.
///      3.1 load from `operand`.
///      3.2 store into `output`.
/// NOTE: The lowering handles only ranked shapes and bails out in case any of
///       output type/edge_padding_low/edge_padding_high/interior_padding size
///       doesn't match that of the operand's rank.
/// Limitation for now:
///   interior_padding == 0 && edge_padding_* >= 0
struct PadOpConverter : public OpRewritePattern<PadOp> {
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter& rewriter) const override {
    Value operand = op.getOperand();
    Value paddingValue = op.getPaddingValue();
    Value output = op.getOutput();

    auto operandType = operand.getType().dyn_cast<ShapedType>();
    auto outputType = output.getType().dyn_cast<ShapedType>();
    // We allow lowering for only ranked input/output.
    if (!(operandType && outputType && operandType.hasRank() &&
          outputType.hasRank()))
      return failure();
    unsigned rank = operandType.getRank();

    auto edgePadLowRanges = op.getEdgePaddingLow().getValues<int64_t>();
    auto edgePadHighRanges = op.getEdgePaddingHigh().getValues<int64_t>();
    auto interiorPadRanges = op.getInteriorPadding().getValues<int64_t>();
    // Check whether the constraints for the lowering are satisfied :-
    //   1. interior_padding[i] == 0
    //   2. edge_padding_*[i] >= 0
    for (auto paddings :
         llvm::zip(edgePadLowRanges, edgePadHighRanges, interiorPadRanges)) {
      // Only handle non-negative edge padding.
      if (std::get<0>(paddings) < 0 || std::get<1>(paddings) < 0)
        return rewriter.notifyMatchFailure(
            op, "expected non-negative edge padding");
      // Only handle interior padding being zero for now.
      if (std::get<2>(paddings) != 0)
        return rewriter.notifyMatchFailure(op,
                                           "expected zero interior padding");
    }

    SmallVector<int64_t> edgePaddingLow(edgePadLowRanges.begin(),
                                        edgePadLowRanges.end());
    SmallVector<int64_t> edgePaddingHigh(edgePadHighRanges.begin(),
                                         edgePadHighRanges.end());
    SmallVector<int64_t> interiorPadding(interiorPadRanges.begin(),
                                         interiorPadRanges.end());

    ArrayRef<int64_t> operandShape = operandType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    // Mapping the `operand` index to the `output` index.
    SmallVector<AffineExpr, 4> expr;
    for (unsigned i = 0; i < rank; i++) {
      AffineExpr dimExpr = rewriter.getAffineDimExpr(i);
      expr.push_back(dimExpr + edgePaddingLow[i]);
    }
    AffineMap map =
        AffineMap::get(rank, /*symbolCount=*/0, expr, rewriter.getContext());

    SmallVector<Value, 4> indices;

    Location loc = op.getLoc();
    // Set padding_value to output.
    {
      OpBuilder::InsertionGuard regionGuard(rewriter);
      Value scalarPaddingValue = rewriter.create<AffineLoadOp>(
          loc, paddingValue, SmallVector<Value, 4>());
      AffineForOp initForOp;
      for (unsigned i = 0; i < rank; i++) {
        initForOp = rewriter.create<AffineForOp>(loc, 0, outputShape[i]);
        indices.push_back(initForOp.getInductionVar());
        rewriter.setInsertionPointToStart(initForOp.getBody());
      }
      rewriter.create<AffineStoreOp>(loc, scalarPaddingValue, output, indices);
    }

    // Store `operand` into `output`, loop upper bounds from `operand` shape.
    indices.clear();
    AffineForOp padForOp;
    for (unsigned i = 0; i < rank; i++) {
      padForOp = rewriter.create<AffineForOp>(loc, 0, operandShape[i]);
      indices.push_back(padForOp.getInductionVar());
      rewriter.setInsertionPointToStart(padForOp.getBody());
    }
    Value storeVal = rewriter.create<AffineLoadOp>(loc, operand, indices);
    rewriter.create<AffineStoreOp>(loc, storeVal, output, map, indices);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename LhloOpTy>
struct BinaryOpConverter : public OpRewritePattern<LhloOpTy> {
  using OpRewritePattern<LhloOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LhloOpTy op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    const auto& lhsType = lhs.getType().template cast<MemRefType>();
    const auto& rhsType = rhs.getType().template cast<MemRefType>();
    const auto& elementType = lhsType.getElementType();

    if (lhsType.getShape() != rhsType.getShape()) {
      return failure();
    }

    LogicalResult mapStatus = success();
    auto bodyBuilder = [&](OpBuilder& builder, Location loc,
                           ValueRange inductionVars) {
      auto l = builder.create<AffineLoadOp>(loc, lhs, inductionVars);
      auto r = builder.create<AffineLoadOp>(loc, rhs, inductionVars);
      Value opResult = lmhlo::LhloOpToStdScalarOp::map<LhloOpTy>(
          op, elementType, {l, r}, &builder);
      mapStatus = success(opResult != nullptr);
      if (failed(mapStatus)) return;
      rewriter.create<AffineStoreOp>(loc, opResult, op.getOut(), inductionVars);
    };

    buildBoundedAffineLoopNest(rewriter, op.getLoc(), lhsType.getShape(),
                               bodyBuilder);
    if (failed(mapStatus)) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Conversion for unary operations i.e. tanh sin cos log log1p etc.
template <typename LhloOpTy>
struct UnaryOpConverter : public OpRewritePattern<LhloOpTy> {
  using OpRewritePattern<LhloOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LhloOpTy op,
                                PatternRewriter& rewriter) const override {
    Value input = op.getInput();
    auto inputType = input.getType().cast<MemRefType>();
    auto elementType = inputType.getElementType();
    ArrayRef<int64_t> shape = inputType.getShape();

    SmallVector<Value, 4> inductionVars;

    LogicalResult mapStatus = success();
    auto bodyBuilder = [&](OpBuilder& builder, Location loc,
                           ValueRange inductionVars) {
      Value loadInput = builder.create<AffineLoadOp>(loc, input, inductionVars);
      Value opResult = lmhlo::LhloOpToStdScalarOp::map<LhloOpTy>(
          op, elementType, {loadInput}, &builder);
      mapStatus = success(opResult != nullptr);
      if (failed(mapStatus)) return;
      rewriter.create<AffineStoreOp>(loc, opResult, op.getOutput(),
                                     inductionVars);
    };
    buildBoundedAffineLoopNest(rewriter, op.getLoc(), shape, bodyBuilder);
    if (failed(mapStatus)) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLHLOToAffineConversionPattern(MLIRContext* context,
                                           RewritePatternSet* patterns) {
  // clang-format off
  patterns->add<
      BinaryOpConverter<lmhlo::AddOp>,
      BinaryOpConverter<lmhlo::AndOp>,
      BinaryOpConverter<lmhlo::DivOp>,
      BinaryOpConverter<lmhlo::MaxOp>,
      BinaryOpConverter<lmhlo::MinOp>,
      BinaryOpConverter<lmhlo::MulOp>,
      BinaryOpConverter<lmhlo::SubtractOp>,
      ConcatOpConverter,
      DotOpConverter,
      GatherOpConverter,
      PadOpConverter,
      UnaryOpConverter<lmhlo::LogOp>>(context);
  // clang-format on
}

struct LhloLegalizeToAffinePass
    : public impl::LhloLegalizeToAffinePassBase<LhloLegalizeToAffinePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, math::MathDialect>();
  }
  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateLHLOToAffineConversionPattern(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLhloLegalizeToAffinePass() {
  return std::make_unique<LhloLegalizeToAffinePass>();
}

}  // namespace lmhlo
}  // namespace mlir
