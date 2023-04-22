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

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace lmhlo {
namespace {

// Builds an affine loop nest iterating from zeros to "upper_bounds" with unit
// steps, and populates the body of the innermost loop using "body_builder".
static void BuildBoundedAffineLoopNest(
    OpBuilder& builder, Location location, ArrayRef<int64_t> upper_bounds,
    function_ref<void(OpBuilder&, Location, ValueRange)> body_builder) {
  SmallVector<int64_t, 3> lower_bounds(upper_bounds.size(), /*Value=*/0);
  SmallVector<int64_t, 3> steps(upper_bounds.size(), /*Value=*/1);
  buildAffineLoopNest(builder, location, lower_bounds, upper_bounds, steps,
                      body_builder);
}

struct DotOpConverter : public OpRewritePattern<DotOp> {
  using OpRewritePattern<DotOp>::OpRewritePattern;

  // Supports only rank-2 tensors for LHS and RHS.
  LogicalResult matchAndRewrite(DotOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    MemRefType lhs_type = lhs.getType().cast<MemRefType>();
    MemRefType rhs_type = rhs.getType().cast<MemRefType>();
    Type element_type = lhs_type.getElementType();
    ArrayRef<int64_t> shape_lhs = lhs_type.getShape();
    ArrayRef<int64_t> shape_rhs = rhs_type.getShape();

    if ((lhs_type.getRank() != 2) || (rhs_type.getRank() != 2)) {
      return failure();
    }

    // We don't currently support batching dimensions, or multiple contraction
    // dimensions.
    mhlo::DotDimensionNumbers dot_dimension_numbers =
        op.dot_dimension_numbers();
    if (dot_dimension_numbers.lhs_batching_dimensions().size() > 0 ||
        dot_dimension_numbers.rhs_batching_dimensions().size() > 0)
      return failure();
    if (dot_dimension_numbers.lhs_contracting_dimensions().size() != 1 ||
        *dot_dimension_numbers.lhs_contracting_dimensions().begin() != 1 ||
        dot_dimension_numbers.rhs_contracting_dimensions().size() != 1 ||
        *dot_dimension_numbers.rhs_contracting_dimensions().begin() != 0) {
      return failure();
    }

    LogicalResult map_status = success();
    auto body_builder = [&](OpBuilder& builder, Location loc, ValueRange ivs) {
      SmallVector<Value, 2> lhs_indices{ivs[0], ivs[2]},
          rhs_indices{ivs[2], ivs[1]}, result_indices{ivs[0], ivs[1]};

      auto l = builder.create<AffineLoadOp>(loc, lhs, lhs_indices);
      auto r = builder.create<AffineLoadOp>(loc, rhs, rhs_indices);
      auto result =
          rewriter.create<AffineLoadOp>(loc, op.output(), result_indices);
      Value op_result = lmhlo::HloOpToStdScalarOp::map<DotOp>(
          op, element_type, {l, r, result}, &builder);
      map_status = success(op_result != nullptr);
      if (failed(map_status)) return;
      builder.create<AffineStoreOp>(loc, op_result, op.output(),
                                    result_indices);
    };

    BuildBoundedAffineLoopNest(rewriter, op.getLoc(),
                               {shape_lhs[0], shape_rhs[1], shape_rhs[0]},
                               body_builder);
    if (failed(map_status)) return failure();

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
    Value output = op.output();
    MemRefType outputType = output.getType().cast<MemRefType>();
    unsigned outputRank = outputType.getRank();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    ValueRange operands = op.val();
    uint64_t concatDim = op.dimension();
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
    return rewriter.create<mlir::ConstantIntOp>(loc, 0, intType.getWidth());

  FloatType floatType = type.cast<FloatType>();
  return rewriter.create<mlir::ConstantFloatOp>(
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
    Value operand = op.operand();
    MemRefType operand_type = operand.getType().cast<MemRefType>();
    unsigned operand_rank = operand_type.getRank();
    ArrayRef<int64_t> operand_shape = operand_type.getShape();

    // Start_indices array.
    Value start_indices = op.start_indices();
    MemRefType start_indices_type = start_indices.getType().cast<MemRefType>();
    unsigned start_indices_rank = start_indices_type.getRank();
    ArrayRef<int64_t> start_indices_shape = start_indices_type.getShape();

    // Output array.
    Value output = op.output();
    MemRefType output_type = output.getType().cast<MemRefType>();
    ArrayRef<int64_t> output_shape = output_type.getShape();

    if (!operand_type.hasStaticShape() ||
        !start_indices_type.hasStaticShape() || !output_type.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static shaped type allowed");

    mhlo::GatherDimensionNumbers gather_dim = op.dimension_numbersAttr();

    // Collapsed_slice_dim.
    DenseIntElementsAttr collapsed_slice_dims_attr =
        gather_dim.collapsed_slice_dims();
    SmallVector<int64_t, 4> collapsed_slice_dims;
    for (const APInt& dim : collapsed_slice_dims_attr.getIntValues())
      collapsed_slice_dims.push_back(dim.getSExtValue());

    // Offset_dim.
    DenseIntElementsAttr offset_dims_attr = gather_dim.offset_dims();
    SmallVector<int64_t, 4> offset_dims;
    for (const APInt& dim : offset_dims_attr.getIntValues())
      offset_dims.push_back(dim.getSExtValue());

    // Start_index_map.
    DenseIntElementsAttr start_index_map_attr = gather_dim.start_index_map();
    SmallVector<int64_t, 4> start_index_map;
    for (const APInt& dim : start_index_map_attr.getIntValues())
      start_index_map.push_back(dim.getSExtValue());

    // Index_vector_dim.
    IntegerAttr index_vector_dim_attr = gather_dim.index_vector_dim();
    int64_t index_vector_dim = index_vector_dim_attr.getValue().getSExtValue();

    // Slice_sizes.
    DenseIntElementsAttr slice_sizes_attr = op.slice_sizesAttr();
    SmallVector<int64_t, 4> slice_sizes;
    for (const APInt& dim : slice_sizes_attr.getIntValues())
      slice_sizes.push_back(dim.getSExtValue());

    // Creating constants with 0 value. We need the Integer type constant value
    // because the indices type will be Integer.
    Value zero_int_val = rewriter.create<mlir::ConstantIntOp>(
        loc, 0, start_indices_type.getElementType());
    Type element_type = output_type.getElementType();
    Value zero_load_value = getZeroValue(element_type, loc, rewriter);
    // Initializing the output buffer with 0.
    fillBuffer(loc, output, zero_load_value, rewriter);

    // We fetch the shape of start_indices at index_vector_dim. In case
    // index_vector_dim is equal to the rank of start_indices, we implicitly
    // consider start_indices to have a trailing 1 dimension.
    unsigned start_indices_numbers =
        (index_vector_dim == start_indices_rank)
            ? 1
            : start_indices_shape[index_vector_dim];
    // We create integer constants till start_incides_index which help us to
    // fetch start_indices in affine transformation.
    SmallVector<Value, 4> start_indices_index;
    for (unsigned i = 0; i < start_indices_numbers; i++) {
      Value i_val = rewriter.create<mlir::ConstantIntOp>(
          loc, i, start_indices_type.getElementType());
      i_val = rewriter.create<IndexCastOp>(loc, i_val, rewriter.getIndexType());
      start_indices_index.push_back(i_val);
    }

    // S_in contains the multiple indices that form a starting index used in the
    // input/operand tensor. O_in contains the multiple offsets of corresponding
    // starting index used in the input/operand tensor. We initialize both of
    // them with 0.
    SmallVector<Value, 4> S_in;
    SmallVector<Value, 4> O_in;
    Value zero_index_val = rewriter.create<IndexCastOp>(
        loc, zero_int_val, rewriter.getIndexType());
    for (unsigned i = 0; i < operand_rank; i++) {
      S_in.push_back(zero_index_val);
      O_in.push_back(zero_index_val);
    }

    // batch_induction_vars stores the loop induction variables pertaining to
    // the batches of start indices.
    SmallVector<Value, 4> batch_induction_vars;
    // output_induction_vars stores the loop induction variables pertaining to
    // both batches and offsets within the output tensor.
    SmallVector<Value, 4> output_induction_vars;
    // Create loops to iterate over each batch of starting index.
    for (unsigned i = 0; i < start_indices_rank; i++) {
      // ith dimension of start_indices doesn't form a batch if it is equal to
      // index_vector_dim.
      if (i == index_vector_dim) continue;
      AffineForOp for_op =
          rewriter.create<AffineForOp>(loc, 0, start_indices_shape[i]);
      batch_induction_vars.push_back(for_op.getInductionVar());
      output_induction_vars.push_back(for_op.getInductionVar());
      rewriter.setInsertionPointToStart(for_op.getBody());
    }

    // Create loops to iterate over each offset dimension within the output
    // tensor.
    for (unsigned i = 0, k = 0, e = offset_dims.size(); i < e; i++) {
      AffineForOp for_op =
          rewriter.create<AffineForOp>(loc, 0, output_shape[offset_dims[i]]);
      rewriter.setInsertionPointToStart(for_op.getBody());
      // We try to fetch the first non-collapsed dimension.
      while (k < collapsed_slice_dims.size() && collapsed_slice_dims[k] == i)
        k++;
      // Remapping the offset_dim[i] to the non-collapsed dimension.
      O_in[k++] = for_op.getInductionVar();
      // We assume offset_dims to be sorted. So when inserted to
      // output_induction_vars the loop induction variable gets inserted at the
      // correct position.
      output_induction_vars.insert(
          output_induction_vars.begin() + offset_dims[i],
          for_op.getInductionVar());
    }

    // Create loops to iterate over all dimensions within the operand tensor.
    SmallVector<Value, 4> operand_index;
    for (unsigned i = 0, k = 0; i < operand_rank; i++) {
      // We assume start_index_map to have sorted dimensions. We only include
      // those dimensions of operand tensor which are present in
      // start_index_map.
      if (k < start_index_map.size() && i == start_index_map[k++]) {
        AffineForOp for_op =
            rewriter.create<AffineForOp>(loc, 0, operand_shape[i]);
        operand_index.push_back(for_op.getInductionVar());
        rewriter.setInsertionPointToStart(for_op.getBody());
      } else {
        operand_index.push_back(O_in[i]);
      }
    }

    // In case index_vector_dim is not equal to start_indices shape then we
    // create another loop to iterate over starting index and update
    // batch_induction_vars.
    if (index_vector_dim != start_indices_rank) {
      for (unsigned i = 0; i < start_indices_numbers; i++) {
        batch_induction_vars.insert(
            batch_induction_vars.begin() + index_vector_dim,
            start_indices_index[i]);
        Value start_index = rewriter.create<AffineLoadOp>(loc, start_indices,
                                                          batch_induction_vars);
        start_index = rewriter.create<IndexCastOp>(loc, start_index,
                                                   rewriter.getIndexType());
        S_in[start_index_map[i]] = start_index;
        batch_induction_vars.erase(batch_induction_vars.begin() +
                                   index_vector_dim);
      }
    } else {
      // Since index_vector_dim is equal to start_indicesRank we can directly
      // fetch the start_index from batch_induction_vars.
      Value start_index = rewriter.create<AffineLoadOp>(loc, start_indices,
                                                        batch_induction_vars);
      start_index = rewriter.create<IndexCastOp>(loc, start_index,
                                                 rewriter.getIndexType());
      S_in[0] = start_index;
    }

    // We load value at a particular operand index and populate the output
    // tensor if the index constraints match.
    Value load_value =
        rewriter.create<AffineLoadOp>(loc, operand, operand_index);
    SmallVector<Value, 4> predicates;
    // Adding offsets to the corresponding starting index and comparing it with
    // the corresponding operand index.
    for (unsigned k = 0, i = 0; k < start_index_map.size(); k++) {
      i = start_index_map[k];
      Value add_start_index_offset = rewriter.create<mlir::AddIOp>(
          loc, rewriter.getIndexType(), S_in[i], O_in[i]);
      Value predicate = rewriter.create<mlir::CmpIOp>(
          loc, CmpIPredicate::eq, add_start_index_offset, operand_index[i]);
      predicates.push_back(predicate);
    }

    // Since the no. of predicates is equal to start_index_map.size() we
    // iterate over pairs of predicates and join them with AndOp.
    unsigned num_equality_checks = start_index_map.size() / 2;
    // We store the final predicate formed by joining other predicates with
    // AndOp in result_predicate.
    Value result_predicate = nullptr;
    for (unsigned i = 0; i < num_equality_checks; i += 2) {
      Value predicateA = predicates[i];
      Value predicateB = predicates[i + 1];
      Value and_predicate =
          rewriter.create<mlir::AndOp>(loc, predicateA, predicateB);
      result_predicate = (i == 0) ? and_predicate
                                  : rewriter.create<mlir::AndOp>(
                                        loc, result_predicate, and_predicate);
    }
    // We fetch the last predicate value. In case this is the only predicate
    // we let result_predicate be equal to this predicate value. Else if there
    // are odd number of predicates we join it to other predicates using AndOp.
    Value predicate = predicates.back();
    if (!result_predicate) result_predicate = predicate;
    // In case there are odd number of predicates we join the last predicate
    // to the result_predicate using AndOp.
    else if (start_index_map.size() % 2 == 1)
      result_predicate =
          rewriter.create<mlir::AndOp>(loc, result_predicate, predicate);

    // We use the loaded value if the index computed by adding offsets to
    // starting index is equal to the current operand index. We use 0 as a value
    // otherwise.
    Value select_load = rewriter.create<mlir::SelectOp>(
        loc, result_predicate, load_value, zero_load_value);
    // We load value at output array.
    Value output_value =
        rewriter.create<AffineLoadOp>(loc, output, output_induction_vars);

    // The selected value is added to the previous value stored in output array.
    if (element_type.isa<FloatType>())
      output_value =
          rewriter.create<AddFOp>(loc, element_type, select_load, output_value);
    else
      output_value =
          rewriter.create<AddIOp>(loc, element_type, select_load, output_value);
    rewriter.create<AffineStoreOp>(loc, output_value, output,
                                   output_induction_vars);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename LhloOpTy>
struct BinaryOpConverter : public OpRewritePattern<LhloOpTy> {
  using OpRewritePattern<LhloOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(LhloOpTy op,
                                PatternRewriter& rewriter) const override {
    const auto& lhs = op.lhs();
    const auto& rhs = op.rhs();
    const auto& lhs_type = lhs.getType().template cast<MemRefType>();
    const auto& rhs_type = rhs.getType().template cast<MemRefType>();
    const auto& element_type = lhs_type.getElementType();

    if (lhs_type.getShape() != rhs_type.getShape()) {
      return failure();
    }

    LogicalResult map_status = success();
    auto body_builder = [&](OpBuilder& builder, Location loc,
                            ValueRange induction_vars) {
      auto l = builder.create<AffineLoadOp>(loc, lhs, induction_vars);
      auto r = builder.create<AffineLoadOp>(loc, rhs, induction_vars);
      Value op_result = lmhlo::HloOpToStdScalarOp::map<LhloOpTy>(
          op, element_type, {l, r}, &builder);
      map_status = success(op_result != nullptr);
      if (failed(map_status)) return;
      rewriter.create<AffineStoreOp>(loc, op_result, op.out(), induction_vars);
    };

    BuildBoundedAffineLoopNest(rewriter, op.getLoc(), lhs_type.getShape(),
                               body_builder);
    if (failed(map_status)) return failure();
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
    Value input = op.input();
    auto inputType = input.getType().cast<MemRefType>();
    auto elementType = inputType.getElementType();
    ArrayRef<int64_t> shape = inputType.getShape();

    SmallVector<Value, 4> induction_vars;
    Location loc = op.getLoc();

    LogicalResult map_status = success();
    auto body_builder = [&](OpBuilder& builder, Location loc,
                            ValueRange induction_vars) {
      Value loadInput =
          builder.create<AffineLoadOp>(loc, input, induction_vars);
      Value opResult = lmhlo::HloOpToStdScalarOp::map<LhloOpTy>(
          op, elementType, {loadInput}, &builder);
      map_status = success(opResult != nullptr);
      if (failed(map_status)) return;
      rewriter.create<AffineStoreOp>(loc, opResult, op.output(),
                                     induction_vars);
    };
    BuildBoundedAffineLoopNest(rewriter, op.getLoc(), shape, body_builder);
    if (failed(map_status)) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

void populateLHLOToAffineConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      BinaryOpConverter<lmhlo::AddOp>,
      BinaryOpConverter<lmhlo::AndOp>,
      BinaryOpConverter<lmhlo::DivOp>,
      BinaryOpConverter<lmhlo::MaxOp>,
      BinaryOpConverter<lmhlo::MinOp>,
      BinaryOpConverter<lmhlo::MulOp>,
      BinaryOpConverter<lmhlo::SubOp>,
      ConcatOpConverter,
      DotOpConverter,
      GatherOpConverter,
      UnaryOpConverter<lmhlo::LogOp>>(context);
  // clang-format on
}

struct LhloLegalizeToAffinePass
    : public LhloLegalizeToAffinePassBase<LhloLegalizeToAffinePass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, math::MathDialect>();
  }
  void runOnFunction() override {
    auto func = getFunction();
    OwningRewritePatternList patterns(&getContext());
    populateLHLOToAffineConversionPattern(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLhloLegalizeToAffinePass() {
  return std::make_unique<LhloLegalizeToAffinePass>();
}

}  // namespace lmhlo
}  // namespace mlir
