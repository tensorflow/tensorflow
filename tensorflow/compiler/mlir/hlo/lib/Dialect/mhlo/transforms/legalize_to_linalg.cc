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

// This file implements logic for lowering HLO/LHLO dialect to Linalg dialect.

#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<StringRef, 3> GetParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<StringRef, 3> res(nLoops - nReduction,
                                getParallelIteratorTypeName());
  res.append(nReduction, getReductionIteratorTypeName());
  return res;
}

SmallVector<StringRef, 3> GetNParallelLoopsAttrs(unsigned nParallelLoops) {
  return GetParallelAndReductionIterators(nParallelLoops, 0);
}

template <bool isLHLO = true>
Value GetResultValue(Operation* op) {
  return isLHLO ? op->getOperand(op->getNumOperands() - 1) : op->getResult(0);
}

template <bool isLHLO = true>
ShapedType GetHloOpResultType(Operation* op) {
  return GetResultValue<isLHLO>(op).getType().template cast<ShapedType>();
}

template <bool isLHLO = true>
bool VerifyHloOpBufferOrTensorSemantics(Operation* op) {
  auto verify_type = [&](Value val) -> bool {
    return (isLHLO && val.getType().isa<MemRefType>()) ||
           (!isLHLO && val.getType().isa<RankedTensorType>());
  };
  if (!llvm::all_of(op->getOperands(), verify_type)) return false;
  return isLHLO ? op->getResults().empty()
                : llvm::all_of(op->getResults(), verify_type);
}

Value GetInitTensor(OpBuilder& b, Location loc, ShapedType type,
                    ArrayRef<Value> dyn_sizes) {
  return b.create<linalg::InitTensorOp>(loc, dyn_sizes, type.getShape(),
                                        type.getElementType());
}

SmallVector<Value, 2> ExtractDynamicSizes(OpBuilder& b, Location loc,
                                          Value tensor,
                                          Value shape_tensor = nullptr) {
  auto tensor_type = tensor.getType().dyn_cast<RankedTensorType>();
  if (!tensor_type) return {};
  SmallVector<Value, 2> dyn_sizes;
  for (auto& en : llvm::enumerate(tensor_type.getShape())) {
    if (en.value() != ShapedType::kDynamicSize) continue;
    // If a shape tensor is present extract from there.
    if (shape_tensor) {
      Value extract = b.create<tensor::ExtractOp>(
          loc, shape_tensor,
          ValueRange{b.create<ConstantIndexOp>(loc, en.index())});
      dyn_sizes.push_back(
          b.create<IndexCastOp>(loc, b.getIndexType(), extract));
    } else {
      dyn_sizes.push_back(b.create<memref::DimOp>(loc, tensor, en.index()));
    }
  }
  return dyn_sizes;
}

SmallVector<int64_t, 4> Extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t, 4> ret;
  for (const APInt& element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
}

/// Returns the constant value associated with the init value if the defining
/// operation is a constant.
Attribute GetInitValueAsConst(Value init) {
  DenseElementsAttr attr;
  if (!matchPattern(init, m_Constant(&attr))) return {};
  auto type = attr.getType().dyn_cast<ShapedType>();
  if (!type || type.getRank() != 0) return {};
  return attr.getValue({});
}

/// Returns a permutation AffineMap that puts all reduction dimensions to the
/// last. The order of parallel loops and reduction loops are all sorted. E.g.,
/// if `rank` is 4 and `reductionDims` is {1, 3}, then
/// "(d0, d1, d2, d3) -> (d0, d2, d1, d3)" is used. The inverse permutation of
/// the AffineMap is returned.
AffineMap GetTransposeMapForReduction(MLIRContext* context, int rank,
                                      ArrayRef<int64_t> reduction_dims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reduction_dims) s.insert(dim);

  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i)
    if (!s.count(i)) permutation.push_back(i);
  for (auto dim : reduction_dims) permutation.push_back(dim);

  auto map = AffineMap::getPermutationMap(permutation, context);
  return inversePermutation(map);
}

/// Returns true if the given `attr` is a splat of the given `value`.
bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

/// Returns true if the given `dimensionNumbers` from a mhlo.convolution op
/// follows a canonical form:
///
/// * Input dimensions have order: (batch_count, spatial_dims,
///   input_channel_count).
/// * Filter dimensions have order: (spatial_dims, input_channel_count,
///   output_channel_count).
/// * Output dimensions have order: (batch_count, spatial_dims,
///   output_channel_count).
template <typename DimensionNumbersTy>
static bool HasCanonicalDimensionNumbers(
    const DimensionNumbersTy& dimension_numbers) {
  const int input_spatial_rank =
      llvm::size(dimension_numbers.input_spatial_dimensions());
  // The dimensions for input should follow the order of
  // batch_count, spatial_dims..., input_feature_count.
  if (dimension_numbers.input_batch_dimension().getInt() != 0 ||
      dimension_numbers.input_feature_dimension().getInt() !=
          (input_spatial_rank + 1)) {
    return false;
  }

  const int kernel_spatial_rank =
      llvm::size(dimension_numbers.kernel_spatial_dimensions());
  // The dimensions for filter should follow the order of
  // spatial_dims..., input_feature_count, num_output_feature_count.
  if (dimension_numbers.kernel_input_feature_dimension().getInt() !=
          kernel_spatial_rank ||
      dimension_numbers.kernel_output_feature_dimension().getInt() !=
          (kernel_spatial_rank + 1)) {
    return false;
  }

  const int output_spatial_rank =
      llvm::size(dimension_numbers.output_spatial_dimensions());
  // The dimensions for output should follow the order of
  // batch_count, spatial_dims.., output_feature_count.
  if (dimension_numbers.output_batch_dimension().getInt() != 0 ||
      dimension_numbers.output_feature_dimension().getInt() !=
          (output_spatial_rank + 1)) {
    return false;
  }

  if (input_spatial_rank != output_spatial_rank ||
      input_spatial_rank != kernel_spatial_rank) {
    return false;
  }

  auto input_spatial_dim = dimension_numbers.input_spatial_dimensions().begin();
  auto kernel_spatial_dim =
      dimension_numbers.kernel_spatial_dimensions().begin();
  auto output_spatial_dim =
      dimension_numbers.output_spatial_dimensions().begin();
  // Check spatial dims are ordered correctly.
  for (int i = 0; i < input_spatial_rank; ++i) {
    const int dim = i + 1;
    if ((*input_spatial_dim++).getZExtValue() != dim ||
        (*output_spatial_dim++).getZExtValue() != dim ||
        (*kernel_spatial_dim++).getZExtValue() != i) {
      return false;
    }
  }

  return true;
}

template <typename OpTy, bool isLHLO = true>
class PointwiseToLinalgConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    ShapedType t0 = args[0].getType().template dyn_cast<ShapedType>();
    if (!t0) return failure();

    unsigned nloops = t0.getRank();
    auto fail = [&](ShapedType t) {
      return !t || !t.hasRank() || t.getRank() != nloops ||
             !(t.getElementType().isSignlessIntOrFloat() ||
               t.getElementType().isa<ComplexType>());
    };
    if (llvm::any_of(op.getOperation()->getResultTypes(), [&](Type t) {
          return fail(this->typeConverter->convertType(t)
                          .template dyn_cast<ShapedType>());
        })) {
      return rewriter.notifyMatchFailure(
          op, "mismatched operand/result types or iterator count");
    }

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Type, 4> body_arg_types, body_result_types, op_result_types;

    // This doesnt account for implicit broadcast, but the working assumption
    // in HLO/LHLO is that are broadcasts are made explicit.

    if (isLHLO && !nloops) return failure();

    int num_inputs = (isLHLO ? args.size() - 1 : args.size());

    ValueRange inputs(args.take_front(num_inputs));
    for (Value in : inputs)
      body_arg_types.emplace_back(getElementTypeOrSelf(in.getType()));

    SmallVector<Value, 4> output_buffers;
    if (isLHLO) {
      output_buffers.append(args.begin() + num_inputs, args.end());
    } else {
      Type result_type = this->typeConverter->convertType(
          op.getOperation()->getResult(0).getType());
      auto dyn_sizes = ExtractDynamicSizes(rewriter, loc, args[0]);
      output_buffers.push_back(GetInitTensor(
          rewriter, loc, result_type.cast<ShapedType>(), dyn_sizes));
      op_result_types.push_back(result_type);
    }
    body_result_types = llvm::to_vector<4>(llvm::map_range(
        output_buffers, [](Value v) { return getElementTypeOrSelf(v); }));

    AffineMap common_indexing_map =
        nloops ? rewriter.getMultiDimIdentityMap(nloops)
               : AffineMap::get(nloops, 0, rewriter.getContext());
    SmallVector<AffineMap, 2> indexing_maps(args.size() + (isLHLO ? 0 : 1),
                                            common_indexing_map);

    bool failed = false;
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, op_result_types, inputs, output_buffers, indexing_maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          // TODO(ravishankarm) : For now use the method in lmhlo namespace.
          // That method needs to be moved out of there.
          Value op_result = lmhlo::HloOpToStdScalarOp::map<OpTy>(
              op, body_result_types,
              llvm::to_vector<2>(args.take_front(inputs.size())), &rewriter);
          if (op_result == nullptr) {
            failed = true;
          } else {
            nested_builder.create<linalg::YieldOp>(loc, op_result);
          }
        });
    if (failed) return failure();
    rewriter.replaceOp(op, linalg_op.getOperation()->getResults());
    return success();
  }
};

template <typename LhloOp>
class ScalarPointwiseToStandardConverter : public OpConversionPattern<LhloOp> {
 public:
  using OpConversionPattern<LhloOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LhloOp lhlo_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = lhlo_op.getLoc();
    auto arg_type =
        lhlo_op.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.getElementType().isSignlessIntOrFloat() ||
        (arg_type.getRank() != 0)) {
      return failure();
    }

    // Create two loads from the input.
    auto lhs = rewriter.create<memref::LoadOp>(loc, lhlo_op.lhs());
    auto rhs = rewriter.create<memref::LoadOp>(loc, lhlo_op.rhs());
    // TODO(ravishankarm) : Move this method out of lmhlo namespace.
    Value op_result = lmhlo::HloOpToStdScalarOp::map<LhloOp>(
        lhlo_op, arg_type.getElementType(), llvm::ArrayRef<Value>{lhs, rhs},
        &rewriter);
    rewriter.create<memref::StoreOp>(loc, op_result, lhlo_op.out());
    rewriter.eraseOp(lhlo_op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// lmhlo.convolution conversion pattern.
//===----------------------------------------------------------------------===//

/// Converts lmhlo.convolution operation to a linalg.conv op.
struct ConvToLinalgConverter : public OpConversionPattern<lmhlo::ConvOp> {
 public:
  using OpConversionPattern<lmhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!HasCanonicalDimensionNumbers(op.dimension_numbers())) return failure();

    // TODO: LHS dilation for deconvolution not supported yet.
    // TODO(jurahul): Window reversal is not supported yet.
    if (op.lhs_dilation() || op.hasWindowReversal()) {
      return failure();
    }

    llvm::SmallVector<Attribute, 4> strides;
    if (auto window_strides = op.window_strides()) {
      auto range = window_strides->getAttributeValues();
      strides.assign(range.begin(), range.end());
    }
    auto strides_arg = ArrayAttr::get(op.getContext(), strides);

    llvm::SmallVector<Attribute, 2> dilation;
    if (auto rhs_dilation = op.rhs_dilation()) {
      auto range = rhs_dilation->getAttributeValues();
      dilation.assign(range.begin(), range.end());
    } else {
      // Default dilation of 1.
      dilation.resize(2, IntegerAttr::get(rewriter.getIntegerType(64), 1));
    }
    auto dilation_arg = ArrayAttr::get(op.getContext(), dilation);

    // Set padding only if it is non-zero.
    DenseIntElementsAttr padding = op.paddingAttr();
    if (!padding ||
        !llvm::any_of(padding.getValues<APInt>(),
                      [](APInt int_val) { return !int_val.isNullValue(); })) {
      padding = nullptr;
    }

    // The order of input and filter are switched with linalg.conv.
    rewriter.replaceOpWithNewOp<linalg::ConvOp>(
        op, args[1], args[0], args[2], strides_arg, dilation_arg, padding);
    return success();
  }
};

/// Base class for lowering HLO operations that have one operand and one result,
/// and are semantically equivalent to a copy of the input to the output (like
/// transpose, some reshape, etc.). The derived classes need to provide a method
/// `getIndexingMaps` that returns AffineMaps for the index maps of the input
/// and the output.
template <typename Derived, typename OpTy, bool isLHLO = true>
class DataMovementOpConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics<isLHLO>(op)) return failure();
    auto result_type = GetHloOpResultType<isLHLO>(op);

    SmallVector<AffineMap, 2> indexing_maps =
        Derived::getIndexingMaps(op, &rewriter);
    if (indexing_maps.empty()) return failure();

    auto nloops = result_type.getRank();
    auto loc = op.getLoc();
    // TODO(pifon): technically, the op itself could have size operands (e.g.
    // broadcast into a dynamic dimension).Handle this case.
    auto dyn_sizes = isLHLO ? SmallVector<Value, 2>()
                            : ExtractDynamicSizes(rewriter, loc, args[0]);
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/isLHLO ? ArrayRef<Type>{} : result_type,
        /*inputs=*/args.front(),
        /*outputBuffers=*/
        isLHLO
            ? ValueRange{args.back()}
            : ValueRange{GetInitTensor(rewriter, loc, result_type, dyn_sizes)},
        indexing_maps, GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          nested_builder.create<linalg::YieldOp>(loc, *args.begin());
        });
    rewriter.replaceOp(op, linalg_op.getOperation()->getResults());
    return success();
  }
};

/// Pattern to convert BroadcastOp to Linalg ops.
template <typename OpTy, bool isLHLO = true>
class BroadcastConverter
    : public DataMovementOpConverter<BroadcastConverter<OpTy, isLHLO>, OpTy,
                                     isLHLO> {
 public:
  using DataMovementOpConverter<BroadcastConverter, OpTy,
                                isLHLO>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy broadcast_op,
                                                   Builder* b) {
    ShapedType input_type =
        broadcast_op.operand().getType().template cast<ShapedType>();
    unsigned input_rank = input_type.getRank();
    unsigned nloops = GetHloOpResultType<isLHLO>(broadcast_op).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned num_prepended_dims = llvm::size(broadcast_op.broadcast_sizes());
    SmallVector<AffineExpr, 4> input_dim_exprs;
    input_dim_exprs.reserve(input_rank);
    for (unsigned i = 0; i < input_rank; ++i) {
      input_dim_exprs.push_back(b->getAffineDimExpr(num_prepended_dims + i));
    }

    AffineMap input_map;
    MLIRContext* context = b->getContext();
    if (input_dim_exprs.empty()) {
      // The input is a scalar, i.e. this is a scalar broadcast op.
      input_map = AffineMap::get(nloops, /*symbolCount=*/0, context);
    } else {
      input_map =
          AffineMap::get(nloops, /*symbolCount=*/0, input_dim_exprs, context);
    }
    return {input_map, b->getMultiDimIdentityMap(nloops)};
  }
};

class HloBroadcastInDimConverter
    : public DataMovementOpConverter<HloBroadcastInDimConverter,
                                     mhlo::BroadcastInDimOp, false> {
 public:
  using DataMovementOpConverter<HloBroadcastInDimConverter,
                                mhlo::BroadcastInDimOp,
                                false>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(
      mhlo::BroadcastInDimOp broadcast_op, Builder* b) {
    auto result_type = GetHloOpResultType<false>(broadcast_op);
    auto operand_type =
        broadcast_op.operand().getType().template cast<ShapedType>();
    unsigned nloops = result_type.getRank();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operand_type.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    auto operand_shape = operand_type.getShape();
    SmallVector<AffineExpr, 4> dim_exprs;
    dim_exprs.reserve(nloops);

    if (broadcast_op.broadcast_dimensions()) {
      for (const auto& broadcastDim :
           enumerate(broadcast_op.broadcast_dimensions().getIntValues())) {
        int size = broadcastDim.value().getSExtValue();
        bool expansion_needed = operand_shape[broadcastDim.index()] == 1 &&
                                result_type.getShape()[size] != 1;
        dim_exprs.push_back(expansion_needed ? b->getAffineConstantExpr(0)
                                             : b->getAffineDimExpr(size));
      }
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dim_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

class HloDynamicBroadcastInDimConverter
    : public OpConversionPattern<mhlo::DynamicBroadcastInDimOp> {
 public:
  using OpConversionPattern<mhlo::DynamicBroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    // If the input has a static shape we know exactly when the broadcast must
    // expand (the dimension is 1, which also trivially expands to 1) or will
    // never expand (the dimension is not 1). This means we can lower the
    // broadcast just as we would lower a fully static broadcast and go directly
    // to linalg.generic. This also covers the important case of broadcasting a
    // scalar.

    // Ideally the pattern (`mhlo.constant` -> `mhlo.dynamic_broadcast_in_dim`)
    // should be converted to an Tensor-dialect op similar to TF ConstantLikeOp.

    mhlo::DynamicBroadcastInDimOp::Adaptor adaptor(op);
    Value operand = adaptor.operand();
    auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
    if (!operand_type || !operand_type.hasStaticShape()) return failure();

    Value shape = adaptor.output_dimensions();
    auto shape_type = shape.getType().cast<RankedTensorType>();
    int64_t result_rank = shape_type.getDimSize(0);
    // HLO dimension types can be any integer, as well as index.
    bool convert_to_index =
        shape_type.getElementType() != rewriter.getIndexType();

    auto result_type = op.getType().dyn_cast<RankedTensorType>();
    if (!result_type) return failure();

    SmallVector<Value, 2> dyn_dims;
    Location loc = op.getLoc();
    for (int i = 0; i < result_rank; ++i) {
      if (!result_type.isDynamicDim(i)) continue;
      Value index = rewriter.create<ConstantIndexOp>(loc, i);
      Value dim = rewriter.create<tensor::ExtractOp>(loc, shape, index);
      if (convert_to_index) {
        dim = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), dim);
      }
      dyn_dims.push_back(dim);
    }

    int64_t nloops = result_type.getRank();
    auto operand_shape = operand_type.getShape();
    SmallVector<AffineExpr, 4> dim_exprs;
    dim_exprs.reserve(nloops);

    if (op.broadcast_dimensions()) {
      for (const auto& broadcast_dim :
           enumerate(op.broadcast_dimensions().getIntValues())) {
        int64_t size = broadcast_dim.value().getSExtValue();
        bool expansion_needed = operand_shape[broadcast_dim.index()] == 1;
        dim_exprs.push_back(expansion_needed ? rewriter.getAffineConstantExpr(0)
                                             : rewriter.getAffineDimExpr(size));
      }
    }

    Value init = rewriter.create<linalg::InitTensorOp>(
        loc, dyn_dims, result_type.getShape(), result_type.getElementType());
    Operation* generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{init.getType()}, ValueRange{operand},
        /*outputBuffers=*/ValueRange{init},
        llvm::makeArrayRef(
            {AffineMap::get(/*dimCount=*/nloops, /*symbolCount=*/0, dim_exprs,
                            rewriter.getContext()),
             rewriter.getMultiDimIdentityMap(nloops)}),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          nested_builder.create<linalg::YieldOp>(loc, *args.begin());
        });
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

class LhloBroadcastInDimConverter
    : public OpConversionPattern<lmhlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern<lmhlo::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::BroadcastInDimOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    lmhlo::BroadcastInDimOp::Adaptor operand_adaptor(args);
    auto result_type = operand_adaptor.output().getType().cast<MemRefType>();
    auto result_shape = result_type.getShape();

    auto operand_and_dims = InsertReshapeIfNecessary(op, args, rewriter);

    Value operand = std::get<0>(operand_and_dims);
    auto broadcast_dims = std::get<1>(operand_and_dims);

    auto loc = op.getLoc();
    auto nloops = result_type.getRank();
    auto operand_type = operand.getType().cast<MemRefType>();

    // For a degenerate case, i.e. broadcasting with expansion of
    // memref<1xELEMENT_TYPE>, the operand is not passed to `linalg.generic`.
    // Instead the value is loaded and used directly in `linalg.yield`.
    if (operand_type.getRank() == 1 &&
        operand_type.getDimSize(0) <
            result_type.getDimSize(broadcast_dims.front())) {
      Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
      Value val = rewriter.create<memref::LoadOp>(loc, operand,
                                                  llvm::makeArrayRef({zero}));
      rewriter.create<linalg::GenericOp>(
          loc, /*inputs=*/ValueRange{},
          /*outputBuffers=*/ValueRange{operand_adaptor.output()},
          llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
          GetNParallelLoopsAttrs(nloops),
          [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
            nested_builder.create<linalg::YieldOp>(loc, val);
          });

    } else {
      auto indexing_maps = getIndexingMaps(op, broadcast_dims, result_shape,
                                           operand_type, &rewriter);
      rewriter.create<linalg::GenericOp>(
          loc, /*inputs=*/ValueRange{operand},
          /*outputBuffers=*/ValueRange{operand_adaptor.output()}, indexing_maps,
          GetNParallelLoopsAttrs(nloops),
          [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
            nested_builder.create<linalg::YieldOp>(loc, *args.begin());
          });
    }
    rewriter.replaceOp(op, llvm::None);
    return success();
  }

  // Inserts 'linalg.reshape' if there is a size-1 dim expansion.
  std::pair<Value, SmallVector<int64_t, 2>> InsertReshapeIfNecessary(
      lmhlo::BroadcastInDimOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const {
    lmhlo::BroadcastInDimOp::Adaptor operand_adaptor(args);
    Value operand = operand_adaptor.operand();
    auto operand_type = operand_adaptor.operand().getType().cast<MemRefType>();
    auto operand_shape = operand_type.getShape();

    Value result = operand_adaptor.output();
    auto result_type = result.getType().cast<MemRefType>();
    auto result_shape = result_type.getShape();

    SmallVector<int64_t, 2> operand_strides;
    int64_t operand_offset;
    if (failed(getStridesAndOffset(operand_type, operand_strides,
                                   operand_offset))) {
      op.emitOpError() << "Failed to get offset and strides.";
    }

    SmallVector<int64_t, 2> new_shape, new_strides, broadcast_dims;
    SmallVector<linalg::ReassociationIndices, 4> collapsed_dims_list;
    linalg::ReassociationIndices collapsed_dims;
    for (const auto& item :
         enumerate(op.broadcast_dimensions().getIntValues())) {
      size_t index = item.index();
      int dim = item.value().getSExtValue();

      collapsed_dims.push_back(index);

      bool expansion_needed =
          operand_shape[index] == 1 && result_shape[dim] != 1;
      if (expansion_needed) {
        continue;
      }
      new_shape.push_back(operand_shape[index]);
      new_strides.push_back(operand_strides[index]);
      broadcast_dims.push_back(dim);

      collapsed_dims_list.push_back(collapsed_dims);
      collapsed_dims.clear();
    }
    // If `collapsed_dims_list` is empty, then the memref has shape [1, ..., 1]
    // and all dimensions need expansion. Such memref will be reshaped to a 1D
    // memref with a single element. New shape and strides needs to be updated
    // accordingly.
    if (collapsed_dims_list.empty()) {
      collapsed_dims_list.push_back({});
      new_shape.push_back(1);
      new_strides.push_back(1);
      broadcast_dims.push_back(0);
    }
    for (const auto& dims : collapsed_dims) {
      collapsed_dims_list.back().push_back(dims);
    }

    // `linalg.reshape` is inserted only if necessary, i.e. when the rank can be
    // reduced.
    if (new_shape.size() < operand_shape.size()) {
      auto new_memref_type = MemRefType::get(
          new_shape, operand_type.getElementType(),
          makeStridedLinearLayoutMap(new_strides, operand_offset,
                                     rewriter.getContext()));
      operand = rewriter.create<linalg::ReshapeOp>(op.getLoc(), new_memref_type,
                                                   operand_adaptor.operand(),
                                                   collapsed_dims_list);
    }
    return std::make_pair(operand, broadcast_dims);
  }

  SmallVector<AffineMap, 2> getIndexingMaps(lmhlo::BroadcastInDimOp op,
                                            ArrayRef<int64_t> broadcast_dims,
                                            ArrayRef<int64_t> result_shape,
                                            MemRefType operand_type,
                                            Builder* b) const {
    unsigned nloops = result_shape.size();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operand_type.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    auto operand_shape = operand_type.getShape();
    SmallVector<AffineExpr, 4> dim_exprs;
    dim_exprs.reserve(nloops);

    for (const auto& broadcast_dim : llvm::enumerate(broadcast_dims)) {
      int size = broadcast_dim.value();
      bool expansion_needed =
          operand_shape[broadcast_dim.index()] == 1 && result_shape[size] != 1;
      if (expansion_needed) {
        op.emitOpError(
            "BroadcastInDimOp lowering to Linalg does not support size-1 "
            "dimensions expansion.");
      }
      dim_exprs.push_back(b->getAffineDimExpr(size));
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dim_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

template <typename OpTy, bool isLHLO = true>
class TransposeConverter
    : public DataMovementOpConverter<TransposeConverter<OpTy, isLHLO>, OpTy,
                                     isLHLO> {
 public:
  using DataMovementOpConverter<TransposeConverter<OpTy, isLHLO>, OpTy,
                                isLHLO>::DataMovementOpConverter;
  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy op, Builder* b) {
    auto result_type =
        GetHloOpResultType<isLHLO>(op).template cast<ShapedType>();
    auto nloops = result_type.getRank();
    SmallVector<AffineExpr, 2> input_exprs;
    input_exprs.resize(result_type.getRank());
    for (auto permutation : llvm::enumerate(op.permutation())) {
      input_exprs[permutation.value().getZExtValue()] =
          b->getAffineDimExpr(permutation.index());
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, input_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

// Converts reshape ops that can be proven to be either a collapse of dimensions
// or expansion of dimensions of the operand.
template <typename OpTy, bool isLHLO = true>
class ReshapeOpConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy reshape_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics<isLHLO>(reshape_op))
      return failure();
    typename OpTy::Adaptor operands(args);
    ShapedType operand_type =
        operands.operand().getType().template cast<ShapedType>();
    ShapedType result_type = GetHloOpResultType<isLHLO>(reshape_op);

    if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
      return failure();

    result_type = this->typeConverter->convertType(result_type)
                      .template cast<ShapedType>();

    // Compute the reassociation maps for the linalg operation.
    ArrayRef<int64_t> src_shape =
        (operand_type.getRank() > result_type.getRank()
             ? operand_type.getShape()
             : result_type.getShape());
    ArrayRef<int64_t> dst_shape =
        (operand_type.getRank() > result_type.getRank()
             ? result_type.getShape()
             : operand_type.getShape());
    unsigned curr_src_dim = 0, curr_dst_dim = 0;
    SmallVector<linalg::ReassociationExprs, 4> reassociation_map(
        dst_shape.size());

    // First scan all dimensions in the source shapes to see whether we have a
    // perfect case where consecutive dimensions in source are collapsed. For
    // such case we can just generate one single linalg.reshape.
    bool is_collapsing_source = true;
    while (curr_src_dim < src_shape.size() && curr_dst_dim < dst_shape.size()) {
      int64_t dst_size = dst_shape[curr_dst_dim];
      int64_t src_size = src_shape[curr_src_dim];
      while (src_size < dst_size && curr_src_dim < src_shape.size()) {
        reassociation_map[curr_dst_dim].push_back(
            rewriter.getAffineDimExpr(curr_src_dim++));
        src_size *= src_shape[curr_src_dim];
      }
      if (src_size == dst_size) {
        reassociation_map[curr_dst_dim].push_back(
            rewriter.getAffineDimExpr(curr_src_dim++));
        // If the next dim in dst_shape is not 1, treat subsequent dims in
        // src_shape which are 1 to be collapsed.
        if (curr_dst_dim == dst_shape.size() - 1 ||
            dst_shape[curr_dst_dim + 1] != 1) {
          while (curr_src_dim < src_shape.size() &&
                 src_shape[curr_src_dim] == 1) {
            reassociation_map[curr_dst_dim].push_back(
                rewriter.getAffineDimExpr(curr_src_dim++));
          }
        }
      } else {
        is_collapsing_source = false;
        break;
      }
      curr_dst_dim++;
    }
    // Rank 0 can always use the direct lowering.
    if (!src_shape.empty() && !dst_shape.empty() &&
        (curr_src_dim != src_shape.size() || curr_dst_dim != dst_shape.size()))
      is_collapsing_source = false;

    // Otherwise, we need to first reduce all source dimensions into one and
    // then expand to the destination dimensions.
    if (!is_collapsing_source) {
      auto get_identity_exprs = [&rewriter](int n) {
        SmallVector<AffineExpr, 4> exprs;
        for (int i = 0; i < n; ++i)
          exprs.push_back(rewriter.getAffineDimExpr(i));
        return exprs;
      };
      Location loc = reshape_op.getLoc();
      int64_t total_elems = std::accumulate(src_shape.begin(), src_shape.end(),
                                            1, std::multiplies<int64_t>());
      auto elem_type = operand_type.getElementType();
      SmallVector<linalg::ReassociationExprs, 4> collapsing_map = {
          // Use operand_type here because we need to collapse all operands
          // dimensions.
          get_identity_exprs(operand_type.getShape().size())};
      SmallVector<linalg::ReassociationExprs, 4> expanding_map = {
          // Use result_type here because we need to expand to all result
          // dimensions.
          get_identity_exprs(result_type.getShape().size())};

      if (isLHLO) {
        auto collapsed_type = MemRefType::get({total_elems}, elem_type);
        Value collapsed_op = rewriter.create<linalg::ReshapeOp>(
            loc, collapsed_type, args[0], collapsing_map);
        Value reshape_buffer = rewriter.create<linalg::ReshapeOp>(
            loc, result_type, collapsed_op, expanding_map);
        rewriter.replaceOpWithNewOp<linalg::CopyOp>(reshape_op, reshape_buffer,
                                                    args[1]);
      } else {
        auto collapsed_type = RankedTensorType::get({total_elems}, elem_type);
        Value collapsed_op = rewriter.create<linalg::TensorReshapeOp>(
            loc, collapsed_type, args[0], collapsing_map);
        rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
            reshape_op, result_type, collapsed_op, expanding_map);
      }
      return success();
    }

    if (isLHLO) {
      Value reshape_buffer = rewriter.create<linalg::ReshapeOp>(
          reshape_op.getLoc(), result_type, args[0], reassociation_map);
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(reshape_op, reshape_buffer,
                                                  args[1]);
    } else {
      rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
          reshape_op, result_type, args[0], reassociation_map);
    }
    return success();
  }
};

template <typename OpTy, bool isLHLO = true>
class IotaConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy iota_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    ShapedType result_shaped_type = GetHloOpResultType<isLHLO>(iota_op);
    if (!result_shaped_type) return failure();

    auto result_element_type = result_shaped_type.getElementType();
    if (!result_element_type.isSignlessIntOrFloat()) return failure();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = result_shaped_type.getRank();

    Location loc = iota_op.getLoc();
    // If this is a dynamic iota, the first argument will be a shape tensor.
    Value shape_tensor = args.size() > (isLHLO ? 1 : 0) ? args[0] : nullptr;
    auto dyn_sizes =
        isLHLO
            ? SmallVector<Value, 2>()
            : ExtractDynamicSizes(
                  rewriter, loc, GetResultValue<isLHLO>(iota_op), shape_tensor);
    auto linalg_op = rewriter.create<linalg::IndexedGenericOp>(
        loc,
        /*resultTensorTypes=*/
        isLHLO ? ArrayRef<Type>{} : ArrayRef<Type>{result_shaped_type},
        /*inputs=*/ValueRange{},
        /*outputBuffers=*/
        isLHLO ? ValueRange{args.back()}
               : ValueRange{GetInitTensor(rewriter, loc, result_shaped_type,
                                          dyn_sizes)},
        llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange ivs,
            ValueRange args) {
          Value cast_op = nested_builder.create<IndexCastOp>(
              nested_loc, ivs[iota_op.iota_dimension()],
              nested_builder.getIntegerType(
                  result_element_type.getIntOrFloatBitWidth()));
          if (result_element_type.template isa<FloatType>()) {
            cast_op = nested_builder.create<SIToFPOp>(nested_loc, cast_op,
                                                      result_element_type);
          }
          nested_builder.create<linalg::YieldOp>(nested_loc, cast_op);
        });
    if (isLHLO)
      rewriter.replaceOp(iota_op, llvm::None);
    else
      rewriter.replaceOp(iota_op, linalg_op.result_tensors());
    return success();
  }
};

/// Converts mhlo.concatenate operation to a linalg.generic op.
struct ConcatenateConverter : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const override {
    // Shortcut the one-operand case, simplifies code below.
    if (args.size() == 1) {
      rewriter.replaceOp(op, args[0]);
      return success();
    }

    auto result_type =
        this->typeConverter->convertType(op.getResult().getType())
            .dyn_cast<RankedTensorType>();
    if (!result_type) return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    uint64_t dim = op.dimension();
    int64_t rank = result_type.getRank();
    Value zero = b.create<ConstantIndexOp>(0);
    SmallVector<Value, 3> sizes;
    for (int64_t i = 0; i < rank; ++i) {
      sizes.push_back(i == dim ? Value() : b.create<memref::DimOp>(args[0], i));
    }

    // Calculate the size of the concatenated dimension.
    Value result_dim_size;
    for (auto arg : args) {
      Value size = b.create<memref::DimOp>(arg, dim);
      result_dim_size =
          result_dim_size ? b.create<AddIOp>(result_dim_size, size) : size;
    }
    sizes[dim] = result_dim_size;

    // Allocate the output tensor with init_tensor.
    SmallVector<Value, 3> dyn_sizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (result_type.isDynamicDim(i)) dyn_sizes.push_back(sizes[i]);
    }
    Value result = b.create<linalg::InitTensorOp>(
        dyn_sizes, result_type.getShape(), result_type.getElementType());

    // Generate a generic op to gather the elements of the concatenate. This is
    // awkward standalone but allows fusion with other generic ops.
    unsigned nloops = result_type.getRank();
    auto linalg_op = b.create<linalg::IndexedGenericOp>(
        /*resultTensorTypes=*/result_type,
        /*inputs=*/ValueRange{}, /*outputBuffers=*/result,
        llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location loc, ValueRange ivs,
            ValueRange) {
          OpBuilder b = nested_builder;
          Value concat_dim_size = zero;
          Value result;
          auto extract_indices = llvm::to_vector<4>(ivs);
          for (const Value& arg : args) {
            Value new_concat_dim_size;
            scf::IfOp if_op;
            if (&arg != &args.back()) {
              // Calculate how far along we have iterated along the concatenate
              // dimension. That way we can tell which input to select.
              new_concat_dim_size = b.create<AddIOp>(
                  loc, concat_dim_size, b.create<memref::DimOp>(loc, arg, dim));
              Value cmp = b.create<CmpIOp>(loc, rewriter.getI1Type(),
                                           CmpIPredicate::ult, ivs[dim],
                                           new_concat_dim_size);
              if_op = b.create<scf::IfOp>(loc, result_type.getElementType(),
                                          cmp, true);
              if (result) {
                b.create<scf::YieldOp>(loc, if_op->getResults()[0]);
              } else {
                result = if_op->getResults()[0];
              }

              b = if_op.getThenBodyBuilder(b.getListener());
            }

            // Now adjust the index for the concatenated dimension to fit into
            // the selected tensor and do an extract at that position.
            extract_indices[dim] =
                b.create<SubIOp>(loc, ivs[dim], concat_dim_size);
            Value extract =
                b.create<tensor::ExtractOp>(loc, arg, extract_indices);
            b.create<scf::YieldOp>(loc, extract);

            if (if_op) {
              b = if_op.getElseBodyBuilder(b.getListener());
              concat_dim_size = new_concat_dim_size;
            }
          }
          nested_builder.create<linalg::YieldOp>(loc, result);
        });
    rewriter.replaceOp(op, linalg_op.result_tensors());
    return success();
  }
};

template <typename OpTy>
class ConstConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy const_op, ArrayRef<Value> /*args*/,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = const_op.getLoc();
    auto value_attr = const_op.value().template cast<DenseElementsAttr>();
    if (value_attr.getType().getRank() != 0) return failure();
    ReplaceConstOp(loc, const_op, value_attr, rewriter);
    return success();
  }

 private:
  void ReplaceConstOp(Location loc, mhlo::ConstOp op,
                      DenseElementsAttr value_attr,
                      ConversionPatternRewriter& rewriter) const {
    Value std_tensor_const = rewriter.create<mlir::ConstantOp>(loc, value_attr);
    rewriter.replaceOp(op, {std_tensor_const});
  }
  void ReplaceConstOp(Location loc, lmhlo::ConstOp op,
                      DenseElementsAttr value_attr,
                      ConversionPatternRewriter& rewriter) const {
    Value std_scalar_const =
        rewriter.create<mlir::ConstantOp>(loc, value_attr.getValue({}));
    rewriter.create<mlir::AffineStoreOp>(loc, std_scalar_const, op.getOperand(),
                                         llvm::None);
    rewriter.eraseOp(op);
  }
};

class ReduceConverter : public OpConversionPattern<lmhlo::ReduceOp> {
 public:
  using OpConversionPattern<lmhlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ReduceOp reduce_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = reduce_op.getLoc();
    lmhlo::ReduceOp::Adaptor adaptor(args);
    auto operand_shape =
        adaptor.inputs()[0].getType().template dyn_cast<ShapedType>();
    if (!operand_shape || !operand_shape.hasRank()) {
      return rewriter.notifyMatchFailure(reduce_op, "expects known-rank args");
    }

    // First fill the output buffer with the init value.
    Value init_value =
        rewriter.create<memref::LoadOp>(loc, adaptor.init_values()[0]);
    rewriter.create<linalg::FillOp>(loc, adaptor.out()[0], init_value);

    DenseIntElementsAttr dimensions_attr = reduce_op.dimensions();
    SmallVector<int, 4> reduction_dims;
    for (const auto& dim : dimensions_attr.getIntValues()) {
      reduction_dims.push_back(dim.getSExtValue());
    }

    SmallVector<AffineExpr, 2> src_exprs;
    SmallVector<AffineExpr, 2> dst_exprs;
    SmallVector<StringRef, 4> types;
    for (int i = 0, rank = operand_shape.getRank(); i != rank; ++i) {
      bool is_reduced = llvm::is_contained(reduction_dims, i);
      types.push_back(is_reduced ? getReductionIteratorTypeName()
                                 : getParallelIteratorTypeName());

      src_exprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (!is_reduced) {
        dst_exprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      }
    }

    auto maps = AffineMap::inferFromExprList({src_exprs, dst_exprs});

    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/ArrayRef<Type>{},
        /*inputs=*/adaptor.inputs(), /*outputBuffers=*/adaptor.out(), maps,
        types);
    rewriter.inlineRegionBefore(reduce_op.body(), linalg_op.region(),
                                linalg_op.region().end());
    {
      OpBuilder::InsertionGuard region_guard(rewriter);
      Block* block = linalg_op.getBody();
      rewriter.setInsertionPoint(&block->front());

      // The incoming region is operating on buffers, while linalg.generic
      // expects scalar SSA values. Add some allocs around the original op to
      // make it compatible.
      auto arg_type = block->getArgument(0).getType().cast<MemRefType>();
      Value alloc_a = rewriter.create<memref::AllocaOp>(loc, arg_type);
      Value alloc_b = rewriter.create<memref::AllocaOp>(loc, arg_type);
      Value alloc_res = rewriter.create<memref::AllocaOp>(loc, arg_type);

      // Now turn the existing signature
      //   (memref<X>, memref<X>, memref<X>) -> ()
      // into
      //   (X, X) -> X
      TypeConverter::SignatureConversion signature_converter(3);
      signature_converter.remapInput(0, alloc_a);
      signature_converter.remapInput(1, alloc_b);
      signature_converter.remapInput(2, alloc_res);
      signature_converter.addInputs(
          {arg_type.getElementType(), arg_type.getElementType()});
      Block* entry_block = rewriter.applySignatureConversion(
          &linalg_op.region(), signature_converter);

      // Store the arguments into the newly allocated buffers.
      rewriter.setInsertionPointAfter(alloc_res.getDefiningOp());
      rewriter.create<memref::StoreOp>(loc, entry_block->getArgument(0),
                                       alloc_a);
      rewriter.create<memref::StoreOp>(loc, entry_block->getArgument(1),
                                       alloc_b);
      rewriter.replaceOp(entry_block->getTerminator(), {});

      // Load & yield the result.
      rewriter.setInsertionPointToEnd(entry_block);
      auto load_res = rewriter.create<memref::LoadOp>(loc, alloc_res);
      rewriter.create<linalg::YieldOp>(loc, ValueRange{load_res});
    }

    rewriter.replaceOp(reduce_op, linalg_op.getOperation()->getResults());
    return success();
  }
};

// TODO(b/156787842): Support the lowering for dynamic shapes.
template <typename OpTy, bool isLHLO = true>
class ReverseConverter
    : public DataMovementOpConverter<ReverseConverter<OpTy, isLHLO>, OpTy,
                                     isLHLO> {
 public:
  using DataMovementOpConverter<ReverseConverter<OpTy, isLHLO>, OpTy,
                                isLHLO>::DataMovementOpConverter;
  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy op, Builder* b) {
    auto result_type =
        GetHloOpResultType<isLHLO>(op).template cast<ShapedType>();
    auto nloops = result_type.getRank();
    SmallVector<AffineExpr, 2> input_exprs;
    input_exprs.reserve(nloops);
    for (int i = 0; i < nloops; ++i)
      input_exprs.push_back(b->getAffineDimExpr(i));
    for (auto dim : op.dimensions()) {
      int i = dim.getZExtValue();
      if (result_type.isDynamicDim(i)) return {};
      int n = result_type.getShape()[i];
      input_exprs[i] = b->getAffineConstantExpr(n - 1) - input_exprs[i];
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, input_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

template <typename OpTy, bool isLHLO = true>
class SliceConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy slice_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = slice_op.getLoc();
    auto arg_type = args[0].getType().template dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasRank()) {
      return rewriter.notifyMatchFailure(slice_op, "expects known-rank args");
    }

    SmallVector<OpFoldResult, 3> offsets, sizes, strides;
    for (int i = 0, e = arg_type.getRank(); i < e; ++i) {
      offsets.push_back(rewriter.getI64IntegerAttr(
          slice_op.start_indices().template getValue<int64_t>(i)));
      sizes.push_back(rewriter.getI64IntegerAttr(
          slice_op.limit_indices().template getValue<int64_t>(i) -
          slice_op.start_indices().template getValue<int64_t>(i)));
      strides.push_back(rewriter.getI64IntegerAttr(
          slice_op.strides().template getValue<int64_t>(i)));
    }
    if (isLHLO) {
      auto linalg_op = rewriter.create<memref::SubViewOp>(loc, args[0], offsets,
                                                          sizes, strides);
      rewriter.create<linalg::CopyOp>(loc, linalg_op, args[1]);
      rewriter.eraseOp(slice_op);
    } else {
      rewriter.replaceOpWithNewOp<SubTensorOp>(slice_op, args[0], offsets,
                                               sizes, strides);
    }
    return success();
  }
};

class DynamicSliceConverter : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern<mhlo::DynamicSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp dynamic_slice_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = dynamic_slice_op.getLoc();
    mhlo::DynamicSliceOp::Adaptor adaptor(args);
    auto arg_type = adaptor.operand().getType().dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasRank()) {
      return rewriter.notifyMatchFailure(dynamic_slice_op,
                                         "require known-rank args");
    }

    auto index_type = rewriter.getIndexType();
    SmallVector<OpFoldResult, 3> start_indices, sizes;
    Value zero = rewriter.create<ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.start_indices()[0]
                                      .getType()
                                      .cast<RankedTensorType>()
                                      .getElementType()));
    for (auto en : llvm::enumerate(
             llvm::zip(adaptor.start_indices(),
                       dynamic_slice_op.slice_sizes().getValues<int64_t>()))) {
      int64_t size = std::get<1>(en.value());
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      // By mhlo.DynamicSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - size_indices[i])`
      Value start_index =
          rewriter.create<tensor::ExtractOp>(loc, std::get<0>(en.value()));
      Value ub = rewriter.createOrFold<memref::DimOp>(loc, adaptor.operand(),
                                                      en.index());
      // ClampOp lowering does not support index type, so cast it into integer
      // type.
      ub = rewriter.createOrFold<IndexCastOp>(loc, start_index.getType(), ub);
      ub = rewriter.createOrFold<SubIOp>(
          loc, ub,
          rewriter.create<ConstantOp>(
              loc, rewriter.getIntegerAttr(start_index.getType(), size)));
      // TODO(hanchung): This is a workaround to use the method because only
      // lmhlo version is defined. The implementation in
      // map_lmhlo_to_scalar_op.h requires to pass a mhlo op. It will convert it
      // to an lmhlo op and call the lmhlo implementation.
      start_index = lmhlo::HloOpToStdScalarOp::map<lmhlo::ClampOp>(
          loc, start_index.getType(),
          ArrayRef<Type>{start_index.getType(), start_index.getType(),
                         start_index.getType()},
          ArrayRef<Value>{zero, start_index, ub}, &rewriter);
      start_indices.push_back(
          rewriter.create<IndexCastOp>(loc, index_type, start_index)
              .getResult());
    }

    int64_t rank = arg_type.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));

    auto result_type =
        this->typeConverter->convertType(dynamic_slice_op.getType())
            .cast<RankedTensorType>();

    rewriter.replaceOpWithNewOp<SubTensorOp>(dynamic_slice_op, result_type,
                                             adaptor.operand(), start_indices,
                                             sizes, strides);
    return success();
  }
};

enum class DotOperationType {
  kVectorDot = 0,
  kMatrixVector = 1,
  kMatrixMatrix = 2,
  kUnsupported = 3
};

DotOperationType GetDotOperationType(mhlo::DotOp dot_op) {
  ArrayRef<int64_t> lhs_shape =
      dot_op.lhs().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhs_shape =
      dot_op.rhs().getType().cast<ShapedType>().getShape();
  auto shape_matches = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize ||
           a == b;
  };
  if (lhs_shape.size() == 1 && rhs_shape.size() == 1 &&
      shape_matches(lhs_shape[0], rhs_shape[0])) {
    return DotOperationType::kVectorDot;
  }
  if (lhs_shape.size() == 2 && rhs_shape.size() == 1 &&
      shape_matches(lhs_shape[1], rhs_shape[0])) {
    return DotOperationType::kMatrixVector;
  }
  if (rhs_shape.size() == 2 && rhs_shape.size() == 2 &&
      shape_matches(lhs_shape[1], rhs_shape[0])) {
    return DotOperationType::kMatrixMatrix;
  }
  return DotOperationType::kUnsupported;
}

SmallVector<Value, 2> GetDotOpInitTensorDynSizes(OpBuilder& b, Location loc,
                                                 Value lhs, Value rhs,
                                                 DotOperationType type) {
  SmallVector<Value, 2> dyn_shape;
  switch (type) {
    case DotOperationType::kMatrixMatrix: {
      if (lhs.getType().cast<ShapedType>().isDynamicDim(0))
        dyn_shape.push_back(b.create<memref::DimOp>(loc, lhs, 0));
      if (rhs.getType().cast<ShapedType>().isDynamicDim(1))
        dyn_shape.push_back(b.create<memref::DimOp>(loc, rhs, 1));
      break;
    }
    case DotOperationType::kMatrixVector: {
      if (lhs.getType().cast<ShapedType>().isDynamicDim(0))
        dyn_shape.push_back(b.create<memref::DimOp>(loc, lhs, 0));
      break;
    }
    case DotOperationType::kVectorDot:
    case DotOperationType::kUnsupported:
    default: {
      break;
    }
  }
  return dyn_shape;
}

template <DotOperationType op_type, typename LinalgOp>
class DotOpOnTensorsConversion : public OpConversionPattern<mhlo::DotOp> {
 public:
  using OpConversionPattern<mhlo::DotOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::DotOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics</*isLHLO=*/false>(op)) {
      return failure();
    }
    if (GetDotOperationType(op) != op_type) return failure();

    mhlo::DotOp::Adaptor adaptor(args);

    Location loc = op.getLoc();
    auto output_type = op.getType().cast<ShapedType>();
    auto output_el_type = output_type.getElementType();
    auto zero_attr = rewriter.getZeroAttr(output_el_type);
    Value zero = rewriter.create<ConstantOp>(loc, zero_attr);
    SmallVector<Value, 2> dyn_shape = GetDotOpInitTensorDynSizes(
        rewriter, loc, adaptor.lhs(), adaptor.rhs(), op_type);
    auto init_tensor = GetInitTensor(rewriter, loc, output_type, dyn_shape);
    Value zero_tensor =
        rewriter.create<linalg::FillOp>(loc, init_tensor, zero).getResult(0);
    rewriter.replaceOpWithNewOp<LinalgOp>(
        op, TypeRange{op.getType()}, ValueRange{adaptor.lhs(), adaptor.rhs()},
        ValueRange{zero_tensor});
    return success();
  }
};

SmallVector<Value, 8> GetDotGeneralOpInitTensorDynSizes(
    OpBuilder& b, Location loc, Value lhs, Value rhs, ShapedType result_type) {
  SmallVector<Value, 8> dyn_shape;
  if (result_type.isDynamicDim(0))
    dyn_shape.push_back(b.create<memref::DimOp>(loc, lhs, 0));
  if (result_type.isDynamicDim(1))
    dyn_shape.push_back(b.create<memref::DimOp>(loc, lhs, 1));
  if (result_type.isDynamicDim(2))
    dyn_shape.push_back(b.create<memref::DimOp>(loc, rhs, 2));
  return dyn_shape;
}

class DotGeneralOpOnTensorsConversion
    : public OpConversionPattern<mhlo::DotGeneralOp> {
 public:
  using OpConversionPattern<mhlo::DotGeneralOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics</*isLHLO=*/false>(op)) {
      return failure();
    }

    mhlo::DotDimensionNumbers dim_numbers = op.dot_dimension_numbers();
    auto lhs_bathcing_dims =
        Extract1DVector(dim_numbers.lhs_batching_dimensions());
    auto rhs_bathcing_dims =
        Extract1DVector(dim_numbers.rhs_batching_dimensions());
    auto lhs_contracting_dims =
        Extract1DVector(dim_numbers.lhs_contracting_dimensions());
    auto rhs_contracting_dims =
        Extract1DVector(dim_numbers.rhs_contracting_dimensions());
    if (lhs_bathcing_dims.size() != 1 || lhs_bathcing_dims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs batching dimensions exactly {0}");
    }
    if (rhs_bathcing_dims.size() != 1 || rhs_bathcing_dims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs batching dimensions exactly {0}");
    }
    if (lhs_contracting_dims.size() != 1 || lhs_contracting_dims[0] != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs contracting dimensions exactly {2}");
    }
    if (rhs_contracting_dims.size() != 1 || rhs_contracting_dims[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs contracting dimensions exactly {1}");
    }

    mhlo::DotGeneralOp::Adaptor adaptor(args);

    Location loc = op.getLoc();
    auto output_type = op.getType().cast<ShapedType>();
    auto output_el_type = output_type.getElementType();
    SmallVector<Value, 8> dyn_shape = GetDotGeneralOpInitTensorDynSizes(
        rewriter, loc, adaptor.lhs(), adaptor.rhs(), output_type);
    auto zero_attr = rewriter.getZeroAttr(output_el_type);
    Value zero = rewriter.create<ConstantOp>(loc, zero_attr);
    auto init_tensor = GetInitTensor(rewriter, loc, output_type, dyn_shape);
    Value zero_tensor =
        rewriter.create<linalg::FillOp>(loc, init_tensor, zero).getResult(0);
    Operation* linalg_op = rewriter.create<linalg::BatchMatmulOp>(
        loc, /*resultTensorTypes=*/TypeRange{op.getType()},
        /*inputs=*/ValueRange{adaptor.lhs(), adaptor.rhs()},
        /*outputBuffers=*/ValueRange{zero_tensor});

    rewriter.replaceOp(op, linalg_op->getResults());
    return success();
  }
};

template <typename OpTy>
struct ReduceRegionXLAOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    // Only convert the body of reduction ops to std ops.
    auto parent_op = op.getOperation()->getParentRegion()->getParentOp();
    if (!isa<mhlo::ReduceOp, linalg::GenericOp, linalg::IndexedGenericOp>(
            parent_op)) {
      return failure();
    }
    if (!op.getResult().getType().template isa<TensorType>()) return failure();
    if (llvm::all_of(args, [](Value arg) {
          return arg.getType().template isa<TensorType>();
        })) {
      return failure();
    }
    Value result = lmhlo::HloOpToStdScalarOp::map<OpTy>(
        op, getElementTypeOrSelf(op.getType()), args, &rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

SmallVector<Value, 8> GetReduceOpInitTensorDynSizes(
    OpBuilder& b, Location loc, Value arg, ShapedType result_type,
    ArrayRef<int64_t> reduction_dims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reduction_dims) s.insert(dim);

  SmallVector<unsigned, 4> parallel_dims;
  SmallVector<Value, 8> dyn_shape;
  int rank = arg.getType().cast<RankedTensorType>().getRank();
  for (int i = 0, j = 0; i < rank; ++i) {
    if (s.count(i)) continue;
    if (!result_type.isDynamicDim(j++)) continue;
    dyn_shape.push_back(b.create<memref::DimOp>(loc, arg, i));
  }

  return dyn_shape;
}

class ReduceRegionReturnOpConversion
    : public OpConversionPattern<mhlo::ReturnOp> {
 public:
  using OpConversionPattern<mhlo::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, args);
    return success();
  }
};

class ReduceOnTensorsConversion : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern<mhlo::ReduceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();
    mhlo::ReduceOp::Adaptor adaptor(args);

    int num_inputs = static_cast<int>(adaptor.inputs().size());
    auto src_type = adaptor.inputs()[0].getType().cast<ShapedType>();
    int src_rank = src_type.getRank();
    if (!src_rank) {
      return rewriter.notifyMatchFailure(op, "expects known-rank args");
    }

    SmallVector<int64_t, 4> reduction_dims = Extract1DVector(op.dimensions());

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap, 3> indexing_maps;
    for (int i = 0; i < num_inputs; ++i) {
      Value src = adaptor.inputs()[i];
      if (src.getType() != src_type) return failure();

      // Check if init_value is constant. If so, inline the value into the
      // region.
      Value init_value = adaptor.init_values()[i];
      Attribute init_const_val = GetInitValueAsConst(init_value);
      if (init_const_val) {
        init_value = rewriter.create<ConstantOp>(
            init_value.getDefiningOp()->getLoc(), init_const_val);
      } else {
        init_value = rewriter.create<tensor::ExtractOp>(loc, init_value);
      }

      inputs.push_back(src);
      auto result_type = op.getResult(i).getType().cast<ShapedType>();
      SmallVector<Value, 8> dyn_shape = GetReduceOpInitTensorDynSizes(
          rewriter, loc, src, result_type, reduction_dims);
      auto init_tensor = GetInitTensor(rewriter, loc, result_type, dyn_shape);
      Value filled_tensor =
          rewriter.create<linalg::FillOp>(loc, init_tensor, init_value)
              .result();
      outputs.push_back(filled_tensor);
    }

    // Prepare indexing maps for linalg generic op. The elements are for src
    // and dst. Transpose `src` to make the reduction loops be the innermost,
    // because it's easier to fully utilize processors.
    indexing_maps.append(
        num_inputs, GetTransposeMapForReduction(rewriter.getContext(), src_rank,
                                                reduction_dims));

    // The indexing map of `dst` should drop the reduction loops. Since the
    // reduction loops now are all in the innermost, drops
    // `reduction_dims.size()` dimensions. We don't need an inverse
    // permutation here because they are the same.
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0, e = src_rank - reduction_dims.size(); i < e; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    indexing_maps.append(num_inputs,
                         AffineMap::get(src_rank, /*symbolCount=*/0, exprs,
                                        rewriter.getContext()));

    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/op.getResultTypes(), inputs,
        /*outputBuffers=*/ValueRange{outputs}, indexing_maps,
        GetParallelAndReductionIterators(src_rank, reduction_dims.size()));

    // Convert the signature of the body. The reduce op region apply function
    // has a signature (lhs, rhs) -> output, all of the same tensor type t.
    // This is converted to a function with the same signature but with
    // element types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will
    // be converted to "(f32, f32, f32)".
    Region& region = linalg_op.region();
    rewriter.inlineRegionBefore(op.body(), region, region.end());
    TypeConverter::SignatureConversion signature_converter(num_inputs * 2);
    for (int i = 0; i < num_inputs * 2; ++i)
      signature_converter.addInputs(i, src_type.getElementType());
    rewriter.applySignatureConversion(&region, signature_converter);
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

/// Converts mhlo.pad operation to linalg.pad_tensor op.
struct PadOpOnTensorsConversion : public OpConversionPattern<mhlo::PadOp> {
  using OpConversionPattern<mhlo::PadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::PadOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const override {
    mhlo::PadOp::Adaptor adaptor(args);
    if (llvm::any_of(
            op.interior_padding().getValues<APInt>(),
            [](const APInt& int_val) { return int_val.getZExtValue() != 0; })) {
      return rewriter.notifyMatchFailure(op, "expected no interior padding");
    }

    auto loc = op.getLoc();
    Value padding_val =
        rewriter.createOrFold<tensor::ExtractOp>(loc, adaptor.padding_value());

    const auto& edge_padding_low = op.edge_padding_low();
    const auto& edge_padding_high = op.edge_padding_high();
    SmallVector<OpFoldResult, 4> low, high;
    for (auto it : llvm::zip(edge_padding_low, edge_padding_high)) {
      low.push_back(rewriter.createOrFold<ConstantIndexOp>(
          loc, std::get<0>(it).getZExtValue()));
      high.push_back(rewriter.createOrFold<ConstantIndexOp>(
          loc, std::get<1>(it).getZExtValue()));
    }
    Type result_type = op.getResult().getType();
    auto pad_tensor_op = linalg::PadTensorOp::createPadScalarOp(
        result_type, adaptor.operand(), padding_val, low, high, loc, rewriter);
    rewriter.replaceOp(op, pad_tensor_op.getResult());
    return success();
  }
};

/// Converts mhlo.conv operation to linalg named op. This only covers normal
/// convolution cases. The op must have canonical dimension numbers. Depthwise
/// convolution and pointwise convolution are not handled in the conversion.
struct NormalConvOpOnTensorsConversion
    : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern<mhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const override {
    if (!HasCanonicalDimensionNumbers(op.dimension_numbers())) return failure();
    if (op.feature_group_count() != 1u) return failure();

    mhlo::ConvOp::Adaptor adaptor(args);
    Location loc = op.getLoc();
    Value input = adaptor.lhs();
    Value filter = adaptor.rhs();
    auto result_type = op.getResult().getType().cast<ShapedType>();
    int64_t rank = result_type.getRank();

    // Check if padding is zero.
    DenseIntElementsAttr padding = op.paddingAttr();
    if (padding && !isSplatValue(*op.padding(), 0)) {
      return rewriter.notifyMatchFailure(op, "expected no padding");
    }

    // The output shape is N spatial_dims F.
    SmallVector<Value, 8> dyn_sizes;
    if (result_type.isDynamicDim(0)) {
      dyn_sizes.push_back(rewriter.create<memref::DimOp>(loc, input, 0));
    }
    for (int64_t i = 1, e = rank - 1; i < e; ++i) {
      if (result_type.isDynamicDim(i)) {
        return rewriter.notifyMatchFailure(
            op, "expected output spatial dims to be static shapes");
      }
    }
    if (result_type.isDynamicDim(rank - 1)) {
      dyn_sizes.push_back(
          rewriter.create<memref::DimOp>(loc, filter, rank - 1));
    }
    Value init_tensor = rewriter.create<linalg::InitTensorOp>(
        loc, dyn_sizes, result_type.getShape(), result_type.getElementType());
    auto zero_attr = rewriter.getZeroAttr(result_type.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zero_attr);
    Value zero_tensor =
        rewriter.create<linalg::FillOp>(loc, init_tensor, zero).getResult(0);
    linalg::LinalgOp res;
    Attribute strides = op.window_stridesAttr();
    // TODO(ataei): Only support dilated kernel right now. We need to consider
    // input dilation for deconvolution cases.
    Attribute dilations = op.rhs_dilationAttr();
    switch (rank) {
      case 3: {
        res = rewriter.create<linalg::ConvInputNWCFilterWCFOp>(
            loc, result_type, ValueRange{input, filter},
            ValueRange{zero_tensor}, dilations, strides);
        break;
      }
      case 4: {
        res = rewriter.create<linalg::ConvInputNHWCFilterHWCFOp>(
            loc, result_type, ValueRange{input, filter},
            ValueRange{zero_tensor}, dilations, strides);
        break;
      }
      case 5: {
        res = rewriter.create<linalg::ConvInputNDHWCFilterDHWCFOp>(
            loc, result_type, ValueRange{input, filter},
            ValueRange{zero_tensor}, dilations, strides);
        break;
      }
      default:
        return rewriter.notifyMatchFailure(op, "expected 1/2/3D conv op");
    }
    rewriter.replaceOp(op, res.getOperation()->getResults());
    return success();
  }
};

/// Converts mhlo.convolution operation to
/// linalg.depthwise_conv_2d_input_nhwc_filter_hwcf op or
/// depthwise_conv_2d_input_nhwc_filter_hwc op.
struct DepthwiseConvOpOnTensorsConversion
    : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern<mhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const override {
    if (op.batch_group_count() != 1) return failure();

    if (op.padding() && !isSplatValue(*op.padding(), 0)) {
      return rewriter.notifyMatchFailure(op,
                                         "non-zero padding unsupported yet");
    }

    if ((op.lhs_dilation() && !isSplatValue(*op.lhs_dilation(), 1)) ||
        (op.rhs_dilation() && !isSplatValue(*op.rhs_dilation(), 1))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-one dialation unsupported yet");
    }

    if (const mhlo::ConvDimensionNumbers& dimension_numbers =
            op.dimension_numbers()) {
      // Make sure that this is 2-D convolution.
      const auto spatial_rank =
          llvm::size(dimension_numbers.input_spatial_dimensions());
      if (spatial_rank != 2) {
        return rewriter.notifyMatchFailure(op,
                                           "only support 2-D cases for now");
      }

      // Make sure that this is depthwise convolution.
      int64_t input_feature_dim =
          dimension_numbers.input_feature_dimension().getInt();
      int64_t input_feature_count =
          op.lhs().getType().cast<ShapedType>().getDimSize(input_feature_dim);
      if (op.feature_group_count() != input_feature_count) {
        return rewriter.notifyMatchFailure(op, "not depth-wise convolution");
      }

      // Make sure that this convolution has a canonical form.
      if (!HasCanonicalDimensionNumbers(dimension_numbers)) {
        return rewriter.notifyMatchFailure(op, "does not have canonical form");
      }
    }

    DenseIntElementsAttr window_strides;
    if (op.window_strides()) {
      window_strides = op.window_strides().getValue();
    } else {
      window_strides = rewriter.getI64VectorAttr({1, 1});
    }

    mhlo::ConvOp::Adaptor adaptor(args);
    Location loc = op.getLoc();
    Value input = adaptor.lhs();
    Value filter = adaptor.rhs();
    auto result_type = op.getResult().getType().cast<RankedTensorType>();
    if (!result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected output has static shapes");
    }

    auto filter_dims =
        llvm::to_vector<4>(op.rhs().getType().cast<ShapedType>().getShape());

    auto get_indices_vector = [](int start, int end) {
      return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
    };

    if (filter_dims[2] * filter_dims[3] != op.feature_group_count()) {
      // For cases where channel multiplier != 1
      auto output_dims = result_type.getShape();
      auto channel_multiplier = filter_dims[3];
      SmallVector<int64_t> reshaped_output_dims;
      reshaped_output_dims.assign(output_dims.begin(), output_dims.end());
      reshaped_output_dims.push_back(channel_multiplier);
      reshaped_output_dims[3] /= channel_multiplier;

      Value init_tensor = rewriter.create<linalg::InitTensorOp>(
          loc, reshaped_output_dims, result_type.getElementType());
      auto zero_attr = rewriter.getZeroAttr(result_type.getElementType());
      Value zero = rewriter.create<ConstantOp>(loc, zero_attr);
      Value zero_tensor =
          rewriter.create<linalg::FillOp>(loc, init_tensor, zero).getResult(0);

      auto reshaped_output_type = RankedTensorType::get(
          reshaped_output_dims, result_type.getElementType());
      auto conv = rewriter.create<linalg::DepthwiseConvInputNHWCFilterHWCFOp>(
          op.getLoc(), reshaped_output_type, ValueRange{input, filter},
          ValueRange{zero_tensor}, window_strides);

      // Create a Linalg reshape op that converts the output from 5 dimensions
      // into 4 dimensions (by collapsing the last two dimensions). This is
      // needed because linalg.depthwise_conv_2d_input_nhwc_filter_hwcf returns
      // 5 dimensions for the output.
      SmallVector<linalg::ReassociationIndices, 4> collapsed_dim_list = {
          get_indices_vector(0, 1), get_indices_vector(1, 2),
          get_indices_vector(2, 3), get_indices_vector(3, 5)};
      rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
          op, result_type, conv.getResult(0), collapsed_dim_list);
    } else {
      // For cases where channel multiplier == 1
      Value init_tensor = rewriter.create<linalg::InitTensorOp>(
          loc, result_type.getShape(), result_type.getElementType());
      auto zero_attr = rewriter.getZeroAttr(result_type.getElementType());
      Value zero = rewriter.create<ConstantOp>(loc, zero_attr);
      Value zero_tensor =
          rewriter.create<linalg::FillOp>(loc, init_tensor, zero).getResult(0);

      // Create a Linalg reshape op that converts the filter from 4 dimensions
      // into 3 dimensions (by droping the unit dimension). This is needed
      // because linalg.depthwise_conv_2d_input_nhwc_filter_hwc expects 3
      // dimensions for the filter.

      filter_dims[2] = static_cast<int64_t>(op.feature_group_count());
      filter_dims.pop_back();

      RankedTensorType filter_shape =
          RankedTensorType::get(filter_dims, op.getType().getElementType());

      SmallVector<linalg::ReassociationIndices, 4> collapsed_dim_list = {
          get_indices_vector(0, 1), get_indices_vector(1, 2),
          get_indices_vector(2, 4)};

      Value reshaped_filter = rewriter.create<linalg::TensorReshapeOp>(
          loc, filter_shape, filter, collapsed_dim_list);

      rewriter.replaceOpWithNewOp<linalg::DepthwiseConvInputNHWCFilterHWCOp>(
          op, result_type, ValueRange{input, reshaped_filter},
          ValueRange{zero_tensor}, window_strides);
    }

    return success();
  }
};

struct ReduceWindowOpOnTensorsConversion
    : public OpConversionPattern<mhlo::ReduceWindowOp> {
  using OpConversionPattern<mhlo::ReduceWindowOp>::OpConversionPattern;

  /// mhlo.reduce_window is mapped to a linalg.pooling operation. The type of
  /// the pooling is determined based on the body of the reduce window
  /// operation. This class enumerates the different variants.
  enum class PoolingType {
    kInvalid,
    kMin,
    kMax,
    kAdd,
  };

  static PoolingType getPoolingType(mhlo::ReduceWindowOp reduce_op,
                                    int result_index) {
    if (Operation* op = reduce_op.getReductionOp(result_index)) {
      if (isa<mhlo::MinOp>(*op)) return PoolingType::kMin;
      if (isa<mhlo::MaxOp>(*op)) return PoolingType::kMax;
      if (isa<mhlo::AddOp>(*op)) return PoolingType::kAdd;
    }
    return PoolingType::kInvalid;
  }

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    int rank = op.getResultTypes()[0].cast<ShapedType>().getRank();
    if (rank != 4) {
      return rewriter.notifyMatchFailure(op, "expected NHWC pooling-based op");
    }

    if (op.padding() && !isSplatValue(*op.padding(), 0)) {
      return rewriter.notifyMatchFailure(op, "require paddings are all zero");
    }

    SmallVector<int64_t, 2> shapes;
    shapes.push_back(op.window_dimensions().getValue<int64_t>(1));
    shapes.push_back(op.window_dimensions().getValue<int64_t>(2));

    if (op.window_strides() &&
        (op.window_strides().getValue().getValue<int64_t>(0) != 1 ||
         op.window_strides().getValue().getValue<int64_t>(3) != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_strides to be [1,x,y,1]");
    }
    if (op.window_dimensions() &&
        (op.window_dimensions().getValue<int64_t>(0) != 1 ||
         op.window_dimensions().getValue<int64_t>(3) != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_dimensions to be [1,x,y,1]");
    }

    Attribute strides;
    if (op.window_stridesAttr()) {
      strides = rewriter.getI64VectorAttr(
          {op.window_strides().getValue().getValue<int64_t>(1),
           op.window_strides().getValue().getValue<int64_t>(2)});
    } else {
      strides = rewriter.getI64VectorAttr({1, 1});
    }
    Attribute dilations;
    if (op.window_dilations()) {
      dilations = rewriter.getI64VectorAttr(
          {op.window_dilations().getValue().getValue<int64_t>(1),
           op.window_dilations().getValue().getValue<int64_t>(2)});
    } else {
      dilations = rewriter.getI64VectorAttr({1, 1});
    }

    SmallVector<Value> pooling_ops;

    ArrayRef<Value> inputs = args.take_front(op.inputs().size());
    ArrayRef<Value> init_values = args.drop_front(op.inputs().size());
    for (auto it : llvm::zip(op.getResults(), inputs, init_values)) {
      OpResult result = std::get<0>(it);
      Value input = std::get<1>(it);
      Value init_value = std::get<2>(it);
      auto result_type = result.getType().cast<ShapedType>();
      if (!input.getType().cast<ShapedType>().getElementType().isF32()) {
        return rewriter.notifyMatchFailure(op,
                                           "expected element type to be f32");
      }

      // Create a fake window dimension.
      auto fake_window_dims = rewriter.create<linalg::InitTensorOp>(
          loc, shapes, result_type.getElementType());
      Value init_tensor = rewriter.create<linalg::InitTensorOp>(
          loc, result_type.getShape(), result_type.getElementType());
      init_value = rewriter.create<tensor::ExtractOp>(loc, init_value);
      Value filled_init_tensor =
          rewriter.create<linalg::FillOp>(loc, init_tensor, init_value)
              .getResult(0);
      auto create_op = [&](auto* type_ptr) -> linalg::LinalgOp {
        return cast<linalg::LinalgOp>(
            rewriter
                .create<std::remove_pointer_t<decltype(type_ptr)>>(
                    loc, ArrayRef<Type>{result_type},
                    ValueRange{args[0], fake_window_dims.getResult()},
                    filled_init_tensor, dilations, strides)
                .getOperation());
      };
      linalg::LinalgOp pooling_op;
      PoolingType pooling_type = getPoolingType(op, result.getResultNumber());
      switch (pooling_type) {
        case PoolingType::kMin: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNHWCMinFOp*>(nullptr));
          break;
        }
        case PoolingType::kMax: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNHWCMaxFOp*>(nullptr));
          break;
        }
        case PoolingType::kAdd: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNHWCSumFOp*>(nullptr));
          break;
        }
        case PoolingType::kInvalid:
          return rewriter.notifyMatchFailure(op, "unknown reduction operation");
      }
      pooling_ops.push_back(pooling_op->getResult(0));
    }
    rewriter.replaceOp(op, pooling_ops);
    return success();
  }
};

/// Converts xla-hlo.torch_index_select op to a linalg.indexed_generic op.
struct TorchIndexSelectOpOnTensorsConversion
    : public OpConversionPattern<mhlo::TorchIndexSelectOp> {
  using OpConversionPattern<mhlo::TorchIndexSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::TorchIndexSelectOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    mhlo::TorchIndexSelectOp::Adaptor adaptor(args);
    int axis = static_cast<int>(op.dim());
    int batch = static_cast<int>(op.batch_dims());
    auto index_shaped_type = adaptor.index().getType().cast<ShapedType>();
    int num_indices = static_cast<int>(index_shaped_type.getRank());
    auto input_shaped_type = adaptor.input().getType().cast<ShapedType>();
    if (axis < 0) axis += static_cast<int>(input_shaped_type.getRank());
    if (batch < 0) batch += num_indices;

    Location loc = op.getLoc();
    auto result_type =
        this->typeConverter->convertType(op.getResult().getType())
            .cast<ShapedType>();
    int rank = static_cast<int>(result_type.getRank());

    SmallVector<AffineMap, 2> indexing_maps;
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0; i < batch; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int i = 0, e = num_indices - batch; i < e; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }
    indexing_maps.emplace_back(
        AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
    indexing_maps.emplace_back(rewriter.getMultiDimIdentityMap(rank));

    // The output shape is
    //   `params[:axis] + indices[batch_dims:] + params[axis + 1:]`
    SmallVector<Value, 4> dyn_sizes;
    for (int i = 0; i < rank; ++i) {
      if (!result_type.isDynamicDim(i)) continue;
      if (i < axis) {
        dyn_sizes.push_back(
            rewriter.create<memref::DimOp>(loc, adaptor.input(), i));
      } else if (i < (axis + num_indices - batch)) {
        int idx = i - axis + batch;
        dyn_sizes.push_back(
            rewriter.create<memref::DimOp>(loc, adaptor.index(), idx));
      } else {
        int idx = i - (axis + num_indices - batch) + axis + 1;
        dyn_sizes.push_back(
            rewriter.create<memref::DimOp>(loc, adaptor.input(), idx));
      }
    }
    Value init_op = rewriter.create<linalg::InitTensorOp>(
        loc, dyn_sizes, result_type.getShape(), result_type.getElementType());
    auto linalg_op = rewriter.create<linalg::IndexedGenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{result_type},
        /*inputs=*/adaptor.index(),
        /*outputs=*/init_op, indexing_maps, GetNParallelLoopsAttrs(rank));

    SmallVector<Type, 4> body_arg_types;
    SmallVector<Value, 2> linalg_op_args = {adaptor.index()};
    // Add a block to the region.
    auto* region = &linalg_op.region();
    auto* block = rewriter.createBlock(region, region->end());
    body_arg_types.append(rank, rewriter.getIndexType());
    for (auto block_args : linalg_op_args) {
      body_arg_types.push_back(
          block_args.getType().cast<ShapedType>().getElementType());
    }
    block->addArguments(body_arg_types);
    block->addArguments(result_type.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    SmallVector<Value, 4> indices;
    Value casted_value = rewriter.create<IndexCastOp>(
        loc, block->getArgument(rank), rewriter.getIndexType());
    for (int i = 0; i < axis; ++i) {
      indices.push_back(block->getArgument(i));
    }
    indices.push_back(casted_value);
    for (int i = axis + num_indices - batch; i < rank; ++i) {
      indices.push_back(block->getArgument(i));
    }

    Value res =
        rewriter.create<tensor::ExtractOp>(loc, adaptor.input(), indices);
    rewriter.create<linalg::YieldOp>(loc, res);

    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

struct ScatterUpdateOnTensorsConversion
    : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    mhlo::ScatterOp::Adaptor adaptor(args);

    // Check if it is a tensor_scatter_nd_update-like op.
    auto& body_ops = op.getRegion().front().getOperations();
    if (body_ops.size() != 1) return failure();
    auto ret_arg = body_ops.front().getOperand(0).dyn_cast<BlockArgument>();
    if (!ret_arg || ret_arg.getArgNumber() != 1) return failure();

    auto operand_ty = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    auto indices_ty =
        adaptor.scatter_indices().getType().dyn_cast<RankedTensorType>();
    if (!operand_ty || !indices_ty) return failure();

    // Linalg operations put all the computation to the innermost loop. Since we
    // also iterate over scatter_indices() with some loops, we can only check
    // one scatter index in one iteration. If there are multiple indices (ie,
    // the index depth is greater than 1), we don't have a way to keep the
    // comparison state. E.g., if the index_depth is 2, like indices = [[0, 1]],
    // we should use the update value only if (i == 0 and j == 1). However, we
    // can not get both indices in one iteration unless we pack them together.
    auto index_vector_dim =
        op.scatter_dimension_numbers().index_vector_dim().getInt();
    if (indices_ty.getDimSize(index_vector_dim) != 1)
      return rewriter.notifyMatchFailure(op, "require index depth to be 1");
    if (index_vector_dim != indices_ty.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "require index_vector_dim to be the last dim");
    }

    // One of indices dims is index depth vector.
    int64_t nloops = operand_ty.getRank() + indices_ty.getRank() - 1;
    SmallVector<AffineMap, 3> indexing_maps;
    {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = 0, e = operand_ty.getRank(); i < e; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      indexing_maps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                             rewriter.getContext()));
    }
    {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = operand_ty.getRank(); i < nloops; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      // The index depth is 1.
      exprs.push_back(rewriter.getAffineConstantExpr(0));
      indexing_maps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                             rewriter.getContext()));

      exprs.pop_back();
      auto update_window_dims =
          Extract1DVector(op.scatter_dimension_numbers().update_window_dims());
      for (auto d : update_window_dims)
        exprs.push_back(rewriter.getAffineDimExpr(d));
      indexing_maps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                             rewriter.getContext()));
    }
    indexing_maps.push_back(indexing_maps.front());

    auto result_ty = this->typeConverter->convertType(op.getResult().getType())
                         .cast<ShapedType>();
    auto scatter_dims_to_operand_dims = Extract1DVector(
        op.scatter_dimension_numbers().scatter_dims_to_operand_dims());
    assert(scatter_dims_to_operand_dims.size() == 1);
    // Do not need init_tensor because we'd like to initialize the output as
    // operand.
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        op.getLoc(), /*resultTensors=*/ArrayRef<Type>{result_ty},
        /*inputs=*/
        ValueRange{adaptor.operand(), adaptor.scatter_indices(),
                   adaptor.updates()},
        /*outputs=*/adaptor.operand(), indexing_maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& b, Location loc, ValueRange args) {
          Value cmp_idx =
              b.create<linalg::IndexOp>(loc, scatter_dims_to_operand_dims[0]);
          Value idx = b.create<IndexCastOp>(loc, b.getIndexType(), args[1]);
          Value pred = b.create<CmpIOp>(loc, b.getI1Type(), CmpIPredicate::eq,
                                        cmp_idx, idx);
          // Use the output arg, so some update values won't be init value
          // again.
          Value res = b.create<SelectOp>(loc, args[2].getType(), pred, args[2],
                                         args[3]);
          b.create<linalg::YieldOp>(loc, res);
        });
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

void populateLHLOToLinalgConversionPattern(MLIRContext* context,
                                           TypeConverter& typeConverter,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<BroadcastConverter<lmhlo::BroadcastOp>,
                   ConstConverter<lmhlo::ConstOp>,
                   ConvToLinalgConverter,
                   IotaConverter<lmhlo::IotaOp>,
                   LhloBroadcastInDimConverter,
                   PointwiseToLinalgConverter<lmhlo::AbsOp>,
                   PointwiseToLinalgConverter<lmhlo::AddOp>,
                   PointwiseToLinalgConverter<lmhlo::AndOp>,
                   PointwiseToLinalgConverter<lmhlo::Atan2Op>,
                   PointwiseToLinalgConverter<lmhlo::CeilOp>,
                   PointwiseToLinalgConverter<lmhlo::ClampOp>,
                   PointwiseToLinalgConverter<lmhlo::CompareOp>,
                   PointwiseToLinalgConverter<lmhlo::ComplexOp>,
                   PointwiseToLinalgConverter<lmhlo::ConvertOp>,
                   // TODO(ataei): Remove this pattern, CopyOp is folded away.
                   PointwiseToLinalgConverter<lmhlo::CopyOp>,
                   PointwiseToLinalgConverter<lmhlo::CosOp>,
                   PointwiseToLinalgConverter<lmhlo::DivOp>,
                   PointwiseToLinalgConverter<lmhlo::ExpOp>,
                   PointwiseToLinalgConverter<lmhlo::Expm1Op>,
                   PointwiseToLinalgConverter<lmhlo::FloorOp>,
                   PointwiseToLinalgConverter<lmhlo::ImagOp>,
                   PointwiseToLinalgConverter<lmhlo::IsFiniteOp>,
                   PointwiseToLinalgConverter<lmhlo::LogOp>,
                   PointwiseToLinalgConverter<lmhlo::LogisticOp>,
                   PointwiseToLinalgConverter<lmhlo::Log1pOp>,
                   PointwiseToLinalgConverter<lmhlo::MaxOp>,
                   PointwiseToLinalgConverter<lmhlo::MinOp>,
                   PointwiseToLinalgConverter<lmhlo::MulOp>,
                   PointwiseToLinalgConverter<lmhlo::NegOp>,
                   PointwiseToLinalgConverter<lmhlo::NotOp>,
                   PointwiseToLinalgConverter<lmhlo::OrOp>,
                   PointwiseToLinalgConverter<lmhlo::PowOp>,
                   PointwiseToLinalgConverter<lmhlo::RealOp>,
                   PointwiseToLinalgConverter<lmhlo::RemOp>,
                   PointwiseToLinalgConverter<lmhlo::RsqrtOp>,
                   PointwiseToLinalgConverter<lmhlo::SelectOp>,
                   PointwiseToLinalgConverter<lmhlo::ShiftLeftOp>,
                   PointwiseToLinalgConverter<lmhlo::ShiftRightArithmeticOp>,
                   PointwiseToLinalgConverter<lmhlo::ShiftRightLogicalOp>,
                   PointwiseToLinalgConverter<lmhlo::SignOp>,
                   PointwiseToLinalgConverter<lmhlo::SinOp>,
                   PointwiseToLinalgConverter<lmhlo::SqrtOp>,
                   PointwiseToLinalgConverter<lmhlo::SubOp>,
                   PointwiseToLinalgConverter<lmhlo::TanhOp>,
                   PointwiseToLinalgConverter<lmhlo::XorOp>,
                   ReduceConverter,
                   ReshapeOpConverter<lmhlo::ReshapeOp>,
                   ReverseConverter<lmhlo::ReverseOp>,
                   ScalarPointwiseToStandardConverter<lmhlo::AddOp>,
                   ScalarPointwiseToStandardConverter<lmhlo::MaxOp>,
                   SliceConverter<lmhlo::SliceOp>,
                   TransposeConverter<lmhlo::TransposeOp>
                  >(typeConverter, context);
  // clang-format on
}

// Converter that turns signed/unsigned integers types into signless types.
class RemoveSignTypeConverter : public TypeConverter {
 public:
  RemoveSignTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion(convertInteger);
    addConversion(convertShapedType);

    addArgumentMaterialization(materializeCastFromIllegal);
    addSourceMaterialization(materializeCastToIllegal);
    addTargetMaterialization(materializeCastFromIllegal);
  }

 private:
  static Type convertInteger(IntegerType int_type) {
    return IntegerType::get(int_type.getContext(),
                            int_type.getIntOrFloatBitWidth());
  }

  static Type convertShapedType(ShapedType shaped_type) {
    if (auto int_type = shaped_type.getElementType().dyn_cast<IntegerType>())
      return shaped_type.clone(convertInteger(int_type));
    return shaped_type;
  }

  static llvm::Optional<Value> materializeCastFromIllegal(OpBuilder& builder,
                                                          Type type,
                                                          ValueRange inputs,
                                                          Location loc) {
    Type from_type = getElementTypeOrSelf(inputs[0].getType());
    Type to_type = getElementTypeOrSelf(type);
    if ((!from_type.isSignedInteger() && !from_type.isUnsignedInteger()) ||
        !to_type.isSignlessInteger())
      return llvm::None;
    // Use unrealized conversion casts to do signful->signless conversions.
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
  }

  static llvm::Optional<Value> materializeCastToIllegal(OpBuilder& builder,
                                                        Type type,
                                                        ValueRange inputs,
                                                        Location loc) {
    Type from_type = getElementTypeOrSelf(inputs[0].getType());
    Type to_type = getElementTypeOrSelf(type);
    if (!from_type.isSignlessInteger() ||
        (!to_type.isSignedInteger() && !to_type.isUnsignedInteger()))
      return llvm::None;
    // Use unrealized conversion casts to do signless->signful conversions.
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
        ->getResult(0);
  }
};

// Converts LHLO ops to Linalg generic.
// Sample result for lmhlo::AddOp.
//
// "lmhlo.add"(%arg1, %arg2, %out) :
//      (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//
// will be converted to
//
// #map0 = (d0, d1) -> (d0, d1)
// "linalg.generic"(%arg1, %arg2, %out) ( {
//   ^bb0(%arg4: f32, %arg5: f32):
//     %0 = addf %arg4, %arg5 : f32
//     "linalg.yield"(%0) : (f32) -> ()
// }) {
//     indexing_maps = [#map0, #map0, #map0],
//     iterator_types = ["parallel", "parallel"],
// } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
struct LhloLegalizeToLinalgPass
    : public PassWrapper<LhloLegalizeToLinalgPass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<AffineDialect, complex::ComplexDialect, linalg::LinalgDialect,
                math::MathDialect, memref::MemRefDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<complex::ComplexDialect, linalg::LinalgDialect,
                           math::MathDialect, memref::MemRefDialect,
                           StandardOpsDialect, AffineDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RemoveSignTypeConverter type_converter;
    auto func = getFunction();
    populateLHLOToLinalgConversionPattern(func.getContext(), type_converter,
                                          &patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct HloLegalizeToLinalgPass
    : public PassWrapper<HloLegalizeToLinalgPass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<linalg::LinalgDialect, scf::SCFDialect, complex::ComplexDialect,
                math::MathDialect, memref::MemRefDialect>();
  }

  void runOnFunction() override {
    MLIRContext& ctx = getContext();
    OwningRewritePatternList patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<complex::ComplexDialect, linalg::LinalgDialect,
                           math::MathDialect, StandardOpsDialect,
                           tensor::TensorDialect, scf::SCFDialect>();

    // TODO: DimOp shouldn't be in MemRefDialect
    target.addLegalOp<memref::DimOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    RemoveSignTypeConverter type_converter;
    auto func = getFunction();
    mhlo::populateHLOToLinalgConversionPattern(&ctx, type_converter, &patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace lmhlo {
std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToLinalgPass() {
  return std::make_unique<LhloLegalizeToLinalgPass>();
}
}  // namespace lmhlo

namespace mhlo {

void populateHLOToLinalgConversionPattern(MLIRContext* context,
                                          TypeConverter& type_converter,
                                          OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      BroadcastConverter<mhlo::BroadcastOp, false>, ConcatenateConverter,
      ConstConverter<mhlo::ConstOp>, HloDynamicBroadcastInDimConverter,
      HloBroadcastInDimConverter, IotaConverter<mhlo::IotaOp, false>,
      IotaConverter<mhlo::DynamicIotaOp, false>,
      PointwiseToLinalgConverter<mhlo::AbsOp, false>,
      PointwiseToLinalgConverter<mhlo::AddOp, false>,
      PointwiseToLinalgConverter<mhlo::AndOp, false>,
      PointwiseToLinalgConverter<mhlo::Atan2Op, false>,
      PointwiseToLinalgConverter<mhlo::CeilOp, false>,
      PointwiseToLinalgConverter<mhlo::ClampOp, false>,
      PointwiseToLinalgConverter<mhlo::CompareOp, false>,
      PointwiseToLinalgConverter<mhlo::ComplexOp, false>,
      PointwiseToLinalgConverter<mhlo::ConvertOp, false>,
      PointwiseToLinalgConverter<mhlo::CopyOp, false>,
      PointwiseToLinalgConverter<mhlo::CosOp, false>,
      PointwiseToLinalgConverter<mhlo::DivOp, false>,
      PointwiseToLinalgConverter<mhlo::ExpOp, false>,
      PointwiseToLinalgConverter<mhlo::Expm1Op, false>,
      PointwiseToLinalgConverter<mhlo::FloorOp, false>,
      PointwiseToLinalgConverter<mhlo::ImagOp, false>,
      PointwiseToLinalgConverter<mhlo::IsFiniteOp, false>,
      PointwiseToLinalgConverter<mhlo::LogOp, false>,
      PointwiseToLinalgConverter<mhlo::LogisticOp, false>,
      PointwiseToLinalgConverter<mhlo::Log1pOp, false>,
      PointwiseToLinalgConverter<mhlo::MaxOp, false>,
      PointwiseToLinalgConverter<mhlo::MinOp, false>,
      PointwiseToLinalgConverter<mhlo::MulOp, false>,
      PointwiseToLinalgConverter<mhlo::NegOp, false>,
      PointwiseToLinalgConverter<mhlo::NotOp, false>,
      PointwiseToLinalgConverter<mhlo::OrOp, false>,
      PointwiseToLinalgConverter<mhlo::PowOp, false>,
      PointwiseToLinalgConverter<mhlo::RealOp, false>,
      PointwiseToLinalgConverter<mhlo::RemOp, false>,
      PointwiseToLinalgConverter<mhlo::RsqrtOp, false>,
      PointwiseToLinalgConverter<mhlo::SelectOp, false>,
      PointwiseToLinalgConverter<mhlo::ShiftLeftOp, false>,
      PointwiseToLinalgConverter<mhlo::ShiftRightArithmeticOp, false>,
      PointwiseToLinalgConverter<mhlo::ShiftRightLogicalOp, false>,
      PointwiseToLinalgConverter<mhlo::SignOp, false>,
      PointwiseToLinalgConverter<mhlo::SinOp, false>,
      PointwiseToLinalgConverter<mhlo::SqrtOp, false>,
      PointwiseToLinalgConverter<mhlo::SubOp, false>,
      PointwiseToLinalgConverter<mhlo::TanhOp, false>,
      PointwiseToLinalgConverter<mhlo::XorOp, false>,
      ReshapeOpConverter<mhlo::ReshapeOp, false>,
      ReverseConverter<mhlo::ReverseOp, false>,
      SliceConverter<mhlo::SliceOp, false>,
      DynamicSliceConverter,
      TransposeConverter<mhlo::TransposeOp, false>,
      DotOpOnTensorsConversion<DotOperationType::kMatrixMatrix,
                               linalg::MatmulOp>,
      DotOpOnTensorsConversion<DotOperationType::kMatrixVector,
                               linalg::MatvecOp>,
      DotOpOnTensorsConversion<DotOperationType::kVectorDot, linalg::DotOp>,
      DotGeneralOpOnTensorsConversion,
      NormalConvOpOnTensorsConversion,
      DepthwiseConvOpOnTensorsConversion,
      ReduceOnTensorsConversion,
      ReduceWindowOpOnTensorsConversion,
      ScatterUpdateOnTensorsConversion,
      TorchIndexSelectOpOnTensorsConversion,
      PadOpOnTensorsConversion>(type_converter, context);
  // clang-format on
  patterns->insert<ReduceRegionXLAOpConversion<mhlo::AddOp>,
                   ReduceRegionXLAOpConversion<mhlo::MinOp>,
                   ReduceRegionXLAOpConversion<mhlo::MaxOp>,
                   ReduceRegionXLAOpConversion<mhlo::AndOp>,
                   ReduceRegionXLAOpConversion<mhlo::OrOp>,
                   ReduceRegionXLAOpConversion<mhlo::SelectOp>,
                   ReduceRegionXLAOpConversion<mhlo::CompareOp>,
                   ReduceRegionReturnOpConversion>(context,
                                                   PatternBenefit(1000));
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeHloToLinalgPass() {
  return std::make_unique<HloLegalizeToLinalgPass>();
}

std::unique_ptr<TypeConverter> createHloToLinalgSignedIntegerConverter() {
  return std::make_unique<RemoveSignTypeConverter>();
}

}  // namespace mhlo
}  // namespace mlir
