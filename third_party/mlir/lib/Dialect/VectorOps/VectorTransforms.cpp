//===- VectorToLoops.cpp - Conversion within the Vector dialect -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites as 1->N patterns.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Dialect/VectorOps/Utils.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/Dialect/VectorOps/VectorTransforms.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;
using llvm::dbgs;
using mlir::functional::zipMap;

/// Given a shape with sizes greater than 0 along all dimensions,
/// returns the distance, in number of elements, between a slice in a dimension
/// and the next slice in the same dimension.
///   e.g. shape[3, 4, 5] -> linearization_basis[20, 5, 1]
static SmallVector<int64_t, 8> computeStrides(ArrayRef<int64_t> shape) {
  if (shape.empty())
    return {};
  SmallVector<int64_t, 8> tmp;
  tmp.reserve(shape.size());
  int64_t running = 1;
  for (auto size : llvm::reverse(shape)) {
    assert(size > 0 && "size must be nonnegative");
    tmp.push_back(running);
    running *= size;
  }
  return SmallVector<int64_t, 8>(tmp.rbegin(), tmp.rend());
}

static int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  if (basis.empty())
    return 0;
  int64_t res = 1;
  for (auto b : basis)
    res *= b;
  return res;
}

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
static int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(offsets.size() == basis.size());
  int64_t linearIndex = 0;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex += offsets[idx] * basis[idx];
  return linearIndex;
}

/// Given a shape with sizes greater than 0 along all dimensions, returns the
/// delinearized components of linearIndex along shape.
static SmallVector<int64_t, 8> delinearize(int64_t linearIndex,
                                           ArrayRef<int64_t> basis) {
  SmallVector<int64_t, 8> res;
  res.reserve(basis.size());
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx) {
    assert(basis[idx] > 0);
    res.push_back(linearIndex / basis[idx]);
    linearIndex %= basis[idx];
  }
  // Sanity check.
  assert(linearIndex == 0 && "linear index remainder must be 0");
  return res;
}

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(PatternRewriter &builder,
                                              Location loc, Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return builder.createOperation(res);
}

static Value makeSplatZero(Location loc, PatternRewriter &rewriter,
                           VectorType vt) {
  auto t = vt.getElementType();
  Value f = nullptr;
  if (t.isBF16() || t.isF16())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF64FloatAttr(0.0f));
  else if (t.isF32())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF32FloatAttr(0.0f));
  else if (t.isF64())
    f = rewriter.create<ConstantOp>(loc, t, rewriter.getF64FloatAttr(0.0f));
  if (f)
    return rewriter.create<SplatOp>(loc, vt, f);
  llvm_unreachable("Unsupported type in `makeSplatZero`");
}

// Populates 'resultElements[indexMap[i]]' with elements from 'inputElements[i]'
// for each index 'i' in inputElements with a valid mapping in 'indexMap'.
static void getMappedElements(const DenseMap<int64_t, int64_t> &indexMap,
                              ArrayRef<int64_t> inputElements,
                              SmallVectorImpl<int64_t> &resultElements) {
  assert(indexMap.size() == resultElements.size());
  assert(inputElements.size() >= resultElements.size());
  for (unsigned i = 0, e = inputElements.size(); i < e; ++i) {
    auto it = indexMap.find(i);
    if (it != indexMap.end())
      resultElements[it->second] = inputElements[i];
  }
}

// Returns a tuple type with vector element types for each resulting slice
// of 'vectorType' unrolled by 'sizes' and 'strides'.
// TODO(andydavis) Move this to a utility function and share it with
// Extract/InsertSlicesOp verification.
static TupleType generateExtractSlicesOpResultType(VectorType vectorType,
                                                   ArrayRef<int64_t> sizes,
                                                   ArrayRef<int64_t> strides,
                                                   PatternRewriter &builder) {
  assert(llvm::all_of(strides, [](int64_t s) { return s == 1; }));
  unsigned rank = vectorType.getRank();
  assert(sizes.size() == rank);
  assert(strides.size() == rank);

  // Compute shape ratio of 'shape' and 'sizes'.
  auto shape = vectorType.getShape();
  auto maybeDimSliceCounts = shapeRatio(shape, sizes);
  assert(maybeDimSliceCounts.hasValue());
  auto sliceDimCounts = *maybeDimSliceCounts;

  // Compute strides w.r.t number of slices in each dimension.
  auto basis = computeStrides(sliceDimCounts);
  int64_t sliceCount = computeMaxLinearIndex(sliceDimCounts);
  SmallVector<Type, 4> vectorTypes(sliceCount);
  for (unsigned i = 0; i < sliceCount; ++i) {
    // De-linearize w.r.t. 'basis'.
    auto vectorOffsets = delinearize(i, basis);
    // Convert from unrolled vector-space offsets to element-space offsets.
    auto offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; },
                          vectorOffsets, sizes);
    // Initialize 'sliceSizes' to target 'sizes'
    SmallVector<int64_t, 4> sliceSizes(sizes.begin(), sizes.end());
    for (unsigned j = 0; j < rank; ++j) {
      // Based on 'offsets' and 'shape' clip some dim sizes for partial tiles.
      sliceSizes[j] = std::min(sliceSizes[j], shape[j] - offsets[j]);
    }
    // Create Vector type and add to 'vectorTypes[i]'.
    vectorTypes[i] = VectorType::get(sliceSizes, vectorType.getElementType());
  }
  return TupleType::get(vectorTypes, builder.getContext());
}

// UnrolledVectorState aggregates per-operand/result vector state required for
// unrolling.
struct UnrolledVectorState {
  SmallVector<int64_t, 4> unrolledShape;
  SmallVector<int64_t, 4> unrollFactors;
  SmallVector<int64_t, 8> basis;
  int64_t numInstances;
  Value slicesTuple;
};

// Populates 'state' with unrolled shape, unroll factors, basis and
// num unrolled instances for 'vectorType'.
static void initUnrolledVectorState(VectorType vectorType, Value initValue,
                                    const DenseMap<int64_t, int64_t> &indexMap,
                                    ArrayRef<int64_t> targetShape,
                                    UnrolledVectorState &state,
                                    PatternRewriter &builder) {
  // Compute unrolled shape of 'vectorType'.
  state.unrolledShape.resize(vectorType.getRank());
  getMappedElements(indexMap, targetShape, state.unrolledShape);
  // Compute unroll factors for unrolled shape.
  auto maybeUnrollFactors =
      shapeRatio(vectorType.getShape(), state.unrolledShape);
  assert(maybeUnrollFactors.hasValue());
  state.unrollFactors = *maybeUnrollFactors;
  // Compute 'basis' and 'numInstances' based on 'state.unrollFactors'.
  state.basis = computeStrides(state.unrollFactors);
  state.numInstances = computeMaxLinearIndex(state.unrollFactors);
  state.slicesTuple = nullptr;
  if (initValue != nullptr) {
    // Create ExtractSlicesOp.
    SmallVector<int64_t, 4> sizes(state.unrolledShape);
    SmallVector<int64_t, 4> strides(state.unrollFactors.size(), 1);
    auto tupleType =
        generateExtractSlicesOpResultType(vectorType, sizes, strides, builder);
    state.slicesTuple = builder.create<vector::ExtractSlicesOp>(
        initValue->getLoc(), tupleType, initValue, sizes, strides);
  }
}

// Computes and returns the linear index of the unrolled vector at
// 'vectorOffsets' within the vector represented by 'state'.
static int64_t
getUnrolledVectorLinearIndex(UnrolledVectorState &state,
                             ArrayRef<int64_t> vectorOffsets,
                             DenseMap<int64_t, int64_t> &indexMap) {
  // Compute vector offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, vectorOffsets, sliceOffsets);
  // Compute and return linear index of 'sliceOffsets' w.r.t 'state.basis'.
  return linearize(sliceOffsets, state.basis);
}

// Returns an unrolled vector at 'vectorOffsets' within the vector
// represented by 'state'. The vector is created from a slice of 'initValue'
// if not present in 'cache'.
static Value getOrCreateUnrolledVectorSlice(
    Location loc, UnrolledVectorState &state, ArrayRef<int64_t> vectorOffsets,
    ArrayRef<int64_t> offsets, DenseMap<int64_t, int64_t> &indexMap,
    Value initValue, SmallVectorImpl<Value> &cache, PatternRewriter &builder) {
  // Compute slice offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, offsets, sliceOffsets);
  // TODO(b/144845578) Support non-1 strides.
  SmallVector<int64_t, 4> sliceStrides(state.unrolledShape.size(), 1);
  // Compute linear index of 'sliceOffsets' w.r.t 'state.basis'.
  int64_t sliceLinearIndex =
      getUnrolledVectorLinearIndex(state, vectorOffsets, indexMap);
  assert(sliceLinearIndex < static_cast<int64_t>(cache.size()));
  auto valueSlice = cache[sliceLinearIndex];
  if (valueSlice == nullptr) {
    // Return tuple element at 'sliceLinearIndex'.
    auto tupleIndex = builder.getI64IntegerAttr(sliceLinearIndex);
    auto initValueType = initValue->getType().cast<VectorType>();
    auto vectorType =
        VectorType::get(state.unrolledShape, initValueType.getElementType());
    // Initialize 'cache' with slice from 'initValue'.
    valueSlice = builder.create<vector::TupleGetOp>(
        loc, vectorType, state.slicesTuple, tupleIndex);
    // Store value back to 'cache'.
    cache[sliceLinearIndex] = valueSlice;
  }
  return valueSlice;
}

// VectorState aggregates per-operand/result vector state required for
// creating slices of vector operands, and clones of the operation being
// unrolled.
struct VectorState {
  // The type of this vector.
  VectorType type;
  // Map from iteration space index to vector dimension index.
  DenseMap<int64_t, int64_t> indexMap;
  // Index of this value in operation's operand list (-1 if not an operand).
  int64_t operandIndex = -1;
  // Accumulator iterator flag.
  bool isAcc = false;
};

//
// unrollSingleResultStructuredOp
//
// Returns a value representing the result of structured operation 'op'
// with iteration bounds 'iterationBounds' unrolled to 'targetShape'.
// A list of VectorState objects must be specified in 'vectors', where
// each VectorState in the list represents a vector operand or vector result
// (if the operation does not have an accumulator operand).
// The VectorState at index 'resultIndex' in the list must be the state
// associated with the operations single result (i.e. either its accumulator
// operand or vector result value).
//
// Example:
//
//  // Before unrolling
//
//   operand0                operand1                operand2
//       \                      |                      /
//        -------------------- opA --------------------
//
//  // After unrolling by 2
//
//   operand0                operand1                operand2
//   /      \                /      \                /      \
// slice00  slice01       slice10  slice11        slice20  slice21
//   \         |            |          |            /          |
//    -------------------- opA0 --------------------           |
//             |            |          |                       |
//              \           |          |                      /
//               -------------------- opA1 -------------------
//                          |          |
//                           \        /
//                           insertslice
//                                |

// TODO(andydavis) Add the following canonicalization/simplifcation patterns:
// *) Add pattern which matches InsertStridedSlice -> StridedSlice and forwards
//    InsertStridedSlice operand to StridedSlice.
// *) Add pattern which matches SourceOp -> StridedSlice -> UserOp which checks
//    if there are duplicate identical StridedSlice ops from SourceOp, and
//    rewrites itself to use the first duplicate. This transformation should
//    cause users of identifical StridedSlice ops to reuse the same StridedSlice
//    operation, and leave the duplicate StridedSlice ops with no users
//    (removable with DCE).

// TODO(andydavis) Generalize this to support structured ops beyond
// vector ContractionOp, and merge it with 'unrollSingleResultOpMatchingType'
static Value unrollSingleResultStructuredOp(Operation *op,
                                            ArrayRef<int64_t> iterationBounds,
                                            std::vector<VectorState> &vectors,
                                            unsigned resultIndex,
                                            ArrayRef<int64_t> targetShape,
                                            PatternRewriter &builder) {
  auto shapedType = op->getResult(0)->getType().dyn_cast_or_null<ShapedType>();
  if (!shapedType || !shapedType.hasStaticShape())
    assert(false && "Expected a statically shaped result type");

  // Compute unroll factors for 'iterationBounds' based on 'targetShape'
  auto maybeUnrollFactors = shapeRatio(iterationBounds, targetShape);
  if (!maybeUnrollFactors.hasValue())
    assert(false && "Failed to compute unroll factors for target shape");
  auto unrollFactors = *maybeUnrollFactors;

  // Compute unrolled vector state for each vector in 'vectors'.
  unsigned numVectors = vectors.size();
  SmallVector<UnrolledVectorState, 3> unrolledVectorState(numVectors);
  for (unsigned i = 0; i < numVectors; ++i) {
    int64_t operandIndex = vectors[i].operandIndex;
    auto operand = operandIndex >= 0 ? op->getOperand(operandIndex) : nullptr;
    initUnrolledVectorState(vectors[i].type, operand, vectors[i].indexMap,
                            targetShape, unrolledVectorState[i], builder);
  }
  // Compute number of total unrolled instances.
  auto numUnrolledInstances = computeMaxLinearIndex(unrollFactors);
  auto basis = computeStrides(unrollFactors);

  auto &resultValueState = unrolledVectorState[resultIndex];
  auto unrolledResultType = VectorType::get(resultValueState.unrolledShape,
                                            shapedType.getElementType());

  // Initialize caches for intermediate vector results.
  std::vector<SmallVector<Value, 4>> caches(numVectors);
  for (unsigned i = 0; i < numVectors; ++i)
    caches[i].resize(unrolledVectorState[i].numInstances);

  // Unroll 'numUnrolledInstances' of 'op', storing results in 'caches'.
  for (unsigned i = 0; i < numUnrolledInstances; ++i) {
    // De-linearize w.r.t. 'basis'.
    auto vectorOffsets = delinearize(i, basis);
    // Convert from unrolled vector-space offsets to element-space offsets.
    auto offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; },
                          vectorOffsets, targetShape);
    // Get cached slice (or create slice) for each operand at 'offsets'.
    SmallVector<Value, 3> operands;
    operands.resize(op->getNumOperands());
    for (unsigned i = 0; i < numVectors; ++i) {
      int64_t operandIndex = vectors[i].operandIndex;
      if (operandIndex < 0)
        continue; // Output
      auto operand = op->getOperand(operandIndex);
      operands[operandIndex] = getOrCreateUnrolledVectorSlice(
          op->getLoc(), unrolledVectorState[i], vectorOffsets, offsets,
          vectors[i].indexMap, operand, caches[i], builder);
    }
    // Create op on sliced vector arguments.
    auto resultVector =
        cloneOpWithOperandsAndTypes(builder, op->getLoc(), op, operands,
                                    unrolledResultType)
            ->getResult(0);

    // Compute linear result index.
    int64_t linearIndex = getUnrolledVectorLinearIndex(
        resultValueState, vectorOffsets, vectors[resultIndex].indexMap);
    // Update result cache at 'linearIndex'.
    caches[resultIndex][linearIndex] = resultVector;
  }

  // Create TupleOp of unrolled result vectors.
  SmallVector<Type, 4> vectorTupleTypes(resultValueState.numInstances);
  SmallVector<Value, 4> vectorTupleValues(resultValueState.numInstances);
  for (unsigned i = 0; i < resultValueState.numInstances; ++i) {
    vectorTupleTypes[i] = caches[resultIndex][i]->getType().cast<VectorType>();
    vectorTupleValues[i] = caches[resultIndex][i];
  }
  TupleType tupleType = builder.getTupleType(vectorTupleTypes);
  Value tupleOp = builder.create<vector::TupleOp>(op->getLoc(), tupleType,
                                                  vectorTupleValues);

  // Create InsertSlicesOp(Tuple(result_vectors)).
  auto resultVectorType = op->getResult(0)->getType().cast<VectorType>();
  SmallVector<int64_t, 4> sizes(resultValueState.unrolledShape);
  SmallVector<int64_t, 4> strides(resultValueState.unrollFactors.size(), 1);

  Value insertSlicesOp = builder.create<vector::InsertSlicesOp>(
      op->getLoc(), resultVectorType, tupleOp, builder.getI64ArrayAttr(sizes),
      builder.getI64ArrayAttr(strides));
  return insertSlicesOp;
}

static void getVectorContractionOpUnrollState(
    vector::ContractionOp contractionOp, ArrayRef<int64_t> targetShape,
    SmallVectorImpl<int64_t> &iterationBounds,
    std::vector<VectorState> &vectors, unsigned &resultIndex) {
  // Get contraction op iteration bounds.
  contractionOp.getIterationBounds(iterationBounds);
  assert(iterationBounds.size() == targetShape.size());
  // Get map from iteration space index to lhs/rhs/result shape index.
  std::vector<DenseMap<int64_t, int64_t>> iterationIndexMapList;
  contractionOp.getIterationIndexMap(iterationIndexMapList);
  unsigned numIterators = iterationIndexMapList.size();
  vectors.resize(numIterators);
  unsigned accOperandIndex = vector::ContractionOp::getAccOperandIndex();
  for (unsigned i = 0; i < numIterators; ++i) {
    vectors[i].type = contractionOp.getOperand(i)->getType().cast<VectorType>();
    vectors[i].indexMap = iterationIndexMapList[i];
    vectors[i].operandIndex = i;
    vectors[i].isAcc = i == accOperandIndex ? true : false;
  }

  if (llvm::size(contractionOp.masks()) == 2) {
    // Add vectors for lhs/rhs vector mask arguments. Masks have the
    // same vector shape lhs/rhs args, so copy their index maps.
    vectors.push_back({contractionOp.getLHSVectorMaskType(),
                       vectors[0].indexMap, accOperandIndex + 1, false});
    vectors.push_back({contractionOp.getRHSVectorMaskType(),
                       vectors[1].indexMap, accOperandIndex + 2, false});
  }
  // Unroll 'op' 'iterationBounds' to 'targetShape'.
  // TODO(andydavis) Use linalg style 'args_in'/'args_out' to partition
  // 'vectors' instead of 'resultIndex'.
  resultIndex = accOperandIndex;
}

static void
getVectorElementwiseOpUnrollState(Operation *op, ArrayRef<int64_t> targetShape,
                                  SmallVectorImpl<int64_t> &iterationBounds,
                                  std::vector<VectorState> &vectors,
                                  unsigned &resultIndex) {
  // Verify that operation and operands all have the same vector shape.
  auto resultType = op->getResult(0)->getType().dyn_cast_or_null<VectorType>();
  assert(resultType && "Expected op with vector result type");
  auto resultShape = resultType.getShape();
  // Verify that all operands have the same vector type as result.
  assert(llvm::all_of(op->getOperandTypes(),
                      [=](Type type) { return type == resultType; }));
  // Populate 'iterationBounds' with 'resultShape' for elementwise operations.
  iterationBounds.assign(resultShape.begin(), resultShape.end());

  // Create trivial elementwise identity index map based on 'resultShape'.
  DenseMap<int64_t, int64_t> indexMap;
  indexMap.reserve(resultShape.size());
  for (unsigned i = 0; i < resultShape.size(); ++i)
    indexMap[i] = i;

  // Create VectorState each operand and single result.
  unsigned numVectors = op->getNumOperands() + op->getNumResults();
  vectors.resize(numVectors);
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    vectors[i] = {resultType, indexMap, i, false};
  vectors[numVectors - 1] = {resultType, indexMap, -1, false};
  resultIndex = numVectors - 1;
}

// Entry point for unrolling declarative pattern rewrites.
Value mlir::vector::unrollSingleResultOpMatchingType(
    PatternRewriter &builder, Operation *op, ArrayRef<int64_t> targetShape) {
  assert(op->getNumResults() == 1 && "Expected single result operation");

  // Populate 'iterationBounds', 'vectors' and 'resultIndex' to unroll 'op'.
  SmallVector<int64_t, 6> iterationBounds;
  std::vector<VectorState> vectors;
  unsigned resultIndex;

  if (auto contractionOp = dyn_cast<vector::ContractionOp>(op)) {
    // Popultate state for vector ContractionOp.
    getVectorContractionOpUnrollState(contractionOp, targetShape,
                                      iterationBounds, vectors, resultIndex);
  } else {
    // Populate state for vector elementwise op.
    getVectorElementwiseOpUnrollState(op, targetShape, iterationBounds, vectors,
                                      resultIndex);
  }

  // Unroll 'op' with 'iterationBounds' to 'targetShape'.
  return unrollSingleResultStructuredOp(op, iterationBounds, vectors,
                                        resultIndex, targetShape, builder);
}

// Generates slices of 'vectorType' according to 'sizes' and 'strides, and
// calls 'fn' with linear index and indices for each slice.
static void
generateTransferOpSlices(VectorType vectorType, TupleType tupleType,
                         ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides,
                         ArrayRef<Value> indices, PatternRewriter &rewriter,
                         function_ref<void(unsigned, ArrayRef<Value>)> fn) {
  // Compute strides w.r.t. to slice counts in each dimension.
  auto maybeDimSliceCounts = shapeRatio(vectorType.getShape(), sizes);
  assert(maybeDimSliceCounts.hasValue());
  auto sliceDimCounts = *maybeDimSliceCounts;
  auto basis = computeStrides(sliceDimCounts);

  int64_t numSlices = tupleType.size();
  unsigned numSliceIndices = indices.size();
  auto *ctx = rewriter.getContext();
  for (unsigned i = 0; i < numSlices; ++i) {
    // De-linearize w.r.t. 'basis'.
    auto vectorOffsets = delinearize(i, basis);
    // Convert from unrolled vector-space offsets to element-space offsets.
    auto offsets = zipMap([](int64_t v1, int64_t v2) { return v1 * v2; },
                          vectorOffsets, sizes);
    // Compute 'sliceIndices' by adding 'sliceOffsets[i]' to 'indices[i]'.
    SmallVector<Value, 4> sliceIndices(numSliceIndices);
    for (auto it : llvm::enumerate(indices)) {
      auto expr = getAffineDimExpr(0, ctx) +
                  getAffineConstantExpr(offsets[it.index()], ctx);
      auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
      sliceIndices[it.index()] = rewriter.create<AffineApplyOp>(
          it.value()->getLoc(), map, ArrayRef<Value>(it.value()));
    }
    // Call 'fn' to generate slice 'i' at 'sliceIndices'.
    fn(i, sliceIndices);
  }
}

// Splits vector TransferReadOp into smaller TransferReadOps based on slicing
// scheme of its unique ExtractSlicesOp user.
struct SplitTransferReadOp : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::TransferReadOp xferReadOp,
                                     PatternRewriter &rewriter) const override {
    // TODO(andydavis, ntv) Support spliting TransferReadOp with non-identity
    // permutation maps. Repurpose code from MaterializeVectors transformation.
    if (!xferReadOp.permutation_map().isIdentity())
      return matchFailure();
    // Return unless the unique 'xferReadOp' user is an ExtractSlicesOp.
    Value xferReadResult = xferReadOp.getResult();
    auto extractSlicesOp =
        dyn_cast<vector::ExtractSlicesOp>(*xferReadResult->getUsers().begin());
    if (!xferReadResult->hasOneUse() || !extractSlicesOp)
      return matchFailure();

    // Get 'sizes' and 'strides' parameters from ExtractSlicesOp user.
    auto sourceVectorType = extractSlicesOp.getSourceVectorType();
    auto resultTupleType = extractSlicesOp.getResultTupleType();
    SmallVector<int64_t, 4> sizes;
    extractSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    extractSlicesOp.getStrides(strides);
    assert(llvm::all_of(strides, [](int64_t s) { return s == 1; }));

    Location loc = xferReadOp.getLoc();
    int64_t numSlices = resultTupleType.size();
    SmallVector<Value, 4> vectorTupleValues(numSlices);
    SmallVector<Value, 4> indices(xferReadOp.indices().begin(),
                                  xferReadOp.indices().end());
    auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
      // Get VectorType for slice 'i'.
      auto sliceVectorType = resultTupleType.getType(index);
      // Create split TransferReadOp for 'sliceUser'.
      vectorTupleValues[index] = rewriter.create<vector::TransferReadOp>(
          loc, sliceVectorType, xferReadOp.memref(), sliceIndices,
          xferReadOp.permutation_map(), xferReadOp.padding());
    };
    generateTransferOpSlices(sourceVectorType, resultTupleType, sizes, strides,
                             indices, rewriter, createSlice);

    // Create tuple of splice xfer read operations.
    Value tupleOp = rewriter.create<vector::TupleOp>(loc, resultTupleType,
                                                     vectorTupleValues);
    // Replace 'xferReadOp' with result 'insertSlicesResult'.
    rewriter.replaceOpWithNewOp<vector::InsertSlicesOp>(
        xferReadOp, sourceVectorType, tupleOp, extractSlicesOp.sizes(),
        extractSlicesOp.strides());
    return matchSuccess();
  }
};

// Splits vector TransferWriteOp into smaller TransferWriteOps for each source.
struct SplitTransferWriteOp : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::TransferWriteOp xferWriteOp,
                                     PatternRewriter &rewriter) const override {
    // TODO(andydavis, ntv) Support spliting TransferWriteOp with non-identity
    // permutation maps. Repurpose code from MaterializeVectors transformation.
    if (!xferWriteOp.permutation_map().isIdentity())
      return matchFailure();
    // Return unless the 'xferWriteOp' 'vector' operand is an 'InsertSlicesOp'.
    auto *vectorDefOp = xferWriteOp.vector()->getDefiningOp();
    auto insertSlicesOp = dyn_cast_or_null<vector::InsertSlicesOp>(vectorDefOp);
    if (!insertSlicesOp)
      return matchFailure();

    // Get TupleOp operand of 'insertSlicesOp'.
    auto tupleOp = dyn_cast_or_null<vector::TupleOp>(
        insertSlicesOp.vectors()->getDefiningOp());
    if (!tupleOp)
      return matchFailure();

    // Get 'sizes' and 'strides' parameters from InsertSlicesOp user.
    auto sourceTupleType = insertSlicesOp.getSourceTupleType();
    auto resultVectorType = insertSlicesOp.getResultVectorType();
    SmallVector<int64_t, 4> sizes;
    insertSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    insertSlicesOp.getStrides(strides);

    Location loc = xferWriteOp.getLoc();
    SmallVector<Value, 4> indices(xferWriteOp.indices().begin(),
                                  xferWriteOp.indices().end());
    auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
      // Create split TransferWriteOp for source vector 'tupleOp.operand[i]'.
      rewriter.create<vector::TransferWriteOp>(
          loc, tupleOp.getOperand(index), xferWriteOp.memref(), sliceIndices,
          xferWriteOp.permutation_map());
    };
    generateTransferOpSlices(resultVectorType, sourceTupleType, sizes, strides,
                             indices, rewriter, createSlice);

    // Erase old 'xferWriteOp'.
    rewriter.eraseOp(xferWriteOp);
    return matchSuccess();
  }
};

// Patter rewrite which forward tuple elements to their users.
// User(TupleGetOp(ExtractSlicesOp(InsertSlicesOp(TupleOp(Producer)))))
//   -> User(Producer)
struct TupleGetFolderOp : public OpRewritePattern<vector::TupleGetOp> {
  using OpRewritePattern<vector::TupleGetOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::TupleGetOp tupleGetOp,
                                     PatternRewriter &rewriter) const override {
    // Return if 'tupleGetOp.vectors' arg was not defined by ExtractSlicesOp.
    auto extractSlicesOp = dyn_cast_or_null<vector::ExtractSlicesOp>(
        tupleGetOp.vectors()->getDefiningOp());
    if (!extractSlicesOp)
      return matchFailure();

    // Return if 'extractSlicesOp.vector' arg was not defined by InsertSlicesOp.
    auto insertSlicesOp = dyn_cast_or_null<vector::InsertSlicesOp>(
        extractSlicesOp.vector()->getDefiningOp());
    if (!insertSlicesOp)
      return matchFailure();

    // Return if 'insertSlicesOp.vectors' arg was not defined by TupleOp.
    auto tupleOp = dyn_cast_or_null<vector::TupleOp>(
        insertSlicesOp.vectors()->getDefiningOp());
    if (!tupleOp)
      return matchFailure();

    // Forward Value from 'tupleOp' at 'tupleGetOp.index'.
    Value tupleValue = tupleOp.getOperand(tupleGetOp.getIndex());
    rewriter.replaceOp(tupleGetOp, tupleValue);
    return matchSuccess();
  }
};

// TODO(andydavis) Add pattern to rewrite ExtractSlices(ConstantMaskOp).
// TODO(andydavis) Add this as DRR pattern.
void mlir::vector::populateVectorToVectorTransformationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<SplitTransferReadOp, SplitTransferWriteOp, TupleGetFolderOp>(
      context);
}
