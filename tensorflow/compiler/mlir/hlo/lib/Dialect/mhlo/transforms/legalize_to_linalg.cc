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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

SmallVector<StringRef, 3> GetNParallelLoopsAttrs(unsigned nParallelLoops) {
  static constexpr StringRef kParallelIterType = "parallel";
  return SmallVector<StringRef, 3>(nParallelLoops, kParallelIterType);
}

template <bool isLHLO = true>
Value getResultValue(Operation* op) {
  return isLHLO ? op->getOperand(op->getNumOperands() - 1) : op->getResult(0);
}

template <bool isLHLO = true>
ShapedType getHloOpResultType(Operation* op) {
  return getResultValue<isLHLO>(op).getType().template cast<ShapedType>();
}

template <bool isLHLO = true>
bool verifyHloOpBufferOrTensorSemantics(Operation* op) {
  auto verify_type = [&](Value val) -> bool {
    return (isLHLO && val.getType().isa<MemRefType>()) ||
           (!isLHLO && val.getType().isa<RankedTensorType>());
  };
  if (!llvm::all_of(op->getOperands(), verify_type)) return false;
  return isLHLO ? op->getResults().empty()
                : llvm::all_of(op->getResults(), verify_type);
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
    if (llvm::any_of(args,
                     [&](Value v) {
                       return fail(v.getType().dyn_cast<ShapedType>());
                     }) ||
        llvm::any_of(op.getOperation()->getResultTypes(),
                     [&](Type t) { return fail(t.dyn_cast<ShapedType>()); }))
      return emitError(loc,
                       "lhlo to linalg conversion expects ranked args of "
                       "signless int, float or complex element type with ")
             << nloops << " parallel iterators: " << *(op.getOperation());

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Type, 4> body_arg_types, body_result_types, op_result_types;

    // This doesnt account for implicit broadcast, but the working assumption
    // in HLO/LHLO is that are broadcasts are made explicit.

    if (isLHLO && !nloops) return failure();

    int num_inputs = (isLHLO ? args.size() - 1 : args.size());

    ValueRange inputs(args.take_front(num_inputs));
    for (Value in : inputs)
      body_arg_types.emplace_back(getElementTypeOrSelf(in.getType()));

    ValueRange output_buffers(args.take_back(args.size() - num_inputs));
    for (Value out : output_buffers)
      body_result_types.emplace_back(getElementTypeOrSelf(out.getType()));

    if (!isLHLO) {
      // HLO operations have return as tensor types.
      assert(body_result_types.empty() &&
             "When lowering HLO ops result can't be part of arguments");
      Value result = op.getOperation()->getResult(0);
      body_result_types.push_back(getElementTypeOrSelf(result));
      op_result_types.push_back(result.getType());
    }

    AffineMap common_indexing_map =
        nloops ? rewriter.getMultiDimIdentityMap(nloops)
               : AffineMap::get(nloops, 0, rewriter.getContext());
    SmallVector<AffineMap, 2> indexing_maps(args.size() + (isLHLO ? 0 : 1),
                                            common_indexing_map);

    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, op_result_types, inputs, output_buffers,
        /*initTensors=*/ValueRange{}, indexing_maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          // TODO(ravishankarm) : For now use the method in lmhlo namespace.
          // That method needs to be moved out of there.
          Value op_result = lmhlo::HloOpToStdScalarOp::map<OpTy>(
              op, body_result_types,
              llvm::to_vector<2>(args.take_front(inputs.size())), &rewriter);
          nested_builder.create<linalg::YieldOp>(loc, op_result);
        });
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
    auto lhs = rewriter.create<LoadOp>(loc, lhlo_op.lhs());
    auto rhs = rewriter.create<LoadOp>(loc, lhlo_op.rhs());
    // TODO(ravishankarm) : Move this method out of lmhlo namespace.
    Value op_result = lmhlo::HloOpToStdScalarOp::map<LhloOp>(
        lhlo_op, arg_type.getElementType(), llvm::ArrayRef<Value>{lhs, rhs},
        &rewriter);
    rewriter.create<StoreOp>(loc, op_result, lhlo_op.out());
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

  //  This code has been adapted from IREE's
  //  (https://github.com/google/iree/) mhlo -> linalg conversion.
  LogicalResult matchAndRewrite(
      lmhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    // Check validity of dimension information.
    if (const mhlo::ConvDimensionNumbers& dimension_numbers =
            op.dimension_numbers()) {
      const int input_spatial_rank =
          llvm::size(dimension_numbers.input_spatial_dimensions());
      // The dimensions for input should follow the order of
      // batch_count, spatial_dims..., input_feature_count.
      if (dimension_numbers.input_batch_dimension().getInt() != 0 ||
          dimension_numbers.input_feature_dimension().getInt() !=
              (input_spatial_rank + 1))
        return failure();

      const int kernel_spatial_rank =
          llvm::size(dimension_numbers.kernel_spatial_dimensions());
      // The dimensions for filter should follow the order of
      // spatial_dims..., input_feature_count, num_output_feature_count.
      if (dimension_numbers.kernel_input_feature_dimension().getInt() !=
              kernel_spatial_rank ||
          dimension_numbers.kernel_output_feature_dimension().getInt() !=
              (kernel_spatial_rank + 1))
        return failure();

      const int output_spatial_rank =
          llvm::size(dimension_numbers.output_spatial_dimensions());
      // The dimensions for output should follow the order of
      // batch_count, spatial_dims.., output_feature_count.
      if (dimension_numbers.output_batch_dimension().getInt() != 0 ||
          dimension_numbers.output_feature_dimension().getInt() !=
              (output_spatial_rank + 1))
        return failure();

      if (input_spatial_rank != output_spatial_rank ||
          input_spatial_rank != kernel_spatial_rank)
        return failure();

      auto input_spatial_dim =
          dimension_numbers.input_spatial_dimensions().begin();
      auto kernel_spatial_dim =
          dimension_numbers.kernel_spatial_dimensions().begin();
      auto output_spatial_dim =
          dimension_numbers.output_spatial_dimensions().begin();
      // Check if spatial dims are ordered correctly.
      for (int i = 0; i < input_spatial_rank; ++i) {
        const int dim = i + 1;
        if ((*input_spatial_dim++).getZExtValue() != dim ||
            (*output_spatial_dim++).getZExtValue() != dim ||
            (*kernel_spatial_dim++).getZExtValue() != i)
          return failure();
      }
    }

    // TODO: LHS dilation for deconvolution not supported yet.
    if (op.lhs_dilation()) {
      return failure();
    }

    llvm::SmallVector<Attribute, 4> strides;
    if (auto window_strides = op.window_strides()) {
      auto range = window_strides->getAttributeValues();
      strides.assign(range.begin(), range.end());
    }
    auto strides_arg = ArrayAttr::get(strides, op.getContext());

    llvm::SmallVector<Attribute, 2> dilation;
    if (auto rhs_dilation = op.rhs_dilation()) {
      auto range = rhs_dilation->getAttributeValues();
      dilation.assign(range.begin(), range.end());
    } else {
      // Default dilation of 1.
      dilation.resize(2, IntegerAttr::get(rewriter.getIntegerType(64), 1));
    }
    auto dilation_arg = ArrayAttr::get(dilation, op.getContext());

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
    if (!verifyHloOpBufferOrTensorSemantics<isLHLO>(op)) return failure();
    auto result_type = getHloOpResultType<isLHLO>(op);

    SmallVector<AffineMap, 2> indexing_maps =
        Derived::getIndexingMaps(op, &rewriter);
    if (indexing_maps.empty()) return failure();

    auto nloops = result_type.getRank();
    auto loc = op.getLoc();
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/isLHLO ? ArrayRef<Type>{} : result_type,
        /*inputs=*/args.front(),
        /*outputBuffers=*/isLHLO ? ValueRange{args.back()} : ValueRange{},
        /*initTensor=*/ValueRange{}, indexing_maps,
        GetNParallelLoopsAttrs(nloops),
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
    unsigned nloops = getHloOpResultType<isLHLO>(broadcast_op).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned num_prepended_dims = llvm::size(broadcast_op.broadcast_sizes());
    SmallVector<AffineExpr, 4> input_dim_exprs;
    input_dim_exprs.reserve(input_rank);
    for (int i = 0; i < input_rank; ++i) {
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
    auto result_type = getHloOpResultType<false>(broadcast_op);
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
      Value val =
          rewriter.create<LoadOp>(loc, operand, llvm::makeArrayRef({zero}));
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
        getHloOpResultType<isLHLO>(op).template cast<ShapedType>();
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
    if (!verifyHloOpBufferOrTensorSemantics<isLHLO>(reshape_op))
      return failure();
    ShapedType operand_type =
        reshape_op.operand().getType().template cast<ShapedType>();
    ShapedType result_type = getHloOpResultType<isLHLO>(reshape_op);

    if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
      return failure();

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
    bool is_expanding_or_collapsing = true;
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
        is_expanding_or_collapsing = false;
        break;
      }
      curr_dst_dim++;
    }
    if (curr_src_dim != src_shape.size() || curr_dst_dim != dst_shape.size())
      is_expanding_or_collapsing = false;

    if (!is_expanding_or_collapsing) {
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
          get_identity_exprs(dst_shape.size())};
      SmallVector<linalg::ReassociationExprs, 4> expanding_map = {
          get_identity_exprs(src_shape.size())};

      if (isLHLO) {
        auto collapsed_type = MemRefType::get({total_elems}, elem_type);
        Value collapsed_op = rewriter.create<linalg::ReshapeOp>(
            loc, collapsed_type, args[0], collapsing_map);
        Value reshape_buffer = rewriter.create<linalg::ReshapeOp>(
            loc, result_type, collapsed_op, expanding_map);
        rewriter.replaceOpWithNewOp<linalg::CopyOp>(
            reshape_op, reshape_buffer, args[1], /*inputPermutation =*/nullptr,
            /*outputPermutation =*/nullptr);
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
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(
          reshape_op, reshape_buffer, args[1], /*inputPermutation =*/nullptr,
          /*outputPermutation =*/nullptr);
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
    ShapedType result_shaped_type = getHloOpResultType<isLHLO>(iota_op);
    if (!result_shaped_type) return failure();

    auto result_element_type = result_shaped_type.getElementType();
    if (!result_element_type.isSignlessIntOrFloat()) return failure();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = result_shaped_type.getRank();

    auto linalg_op = rewriter.create<linalg::IndexedGenericOp>(
        iota_op.getLoc(),
        /*resultTensorTypes=*/
        isLHLO ? ArrayRef<Type>{} : ArrayRef<Type>{result_shaped_type},
        /*inputs=*/ValueRange{},
        /*outputBuffers=*/isLHLO ? ValueRange{args} : ValueRange{},
        /*initTensors=*/ValueRange{},
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

class ConstConverter : public OpConversionPattern<lmhlo::ConstOp> {
 public:
  using OpConversionPattern<lmhlo::ConstOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::ConstOp const_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = const_op.getLoc();
    auto value_attr = const_op.value().cast<DenseElementsAttr>();
    if (value_attr.getType().getRank() != 0) return failure();
    auto std_const_op =
        rewriter.create<mlir::ConstantOp>(loc, value_attr.getValue({}));
    rewriter.create<mlir::AffineStoreOp>(loc, std_const_op,
                                         const_op.getOperand(), ValueRange());
    rewriter.eraseOp(const_op);
    return success();
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
        adaptor.operands()[0].getType().template dyn_cast<ShapedType>();
    if (!operand_shape || !operand_shape.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects known-rank args");
      return failure();
    }

    // First fill the output buffer with the init value.
    Value init_value = rewriter.create<LoadOp>(loc, adaptor.init_values()[0]);
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
        /*inputs=*/adaptor.operands(), /*outputBuffers=*/adaptor.out(),
        /*initTensors=*/ValueRange{}, maps, types);
    linalg_op.region().takeBody(reduce_op.body());
    {
      OpBuilder::InsertionGuard region_guard(rewriter);
      Block* block = linalg_op.getBody();
      rewriter.setInsertionPoint(&block->front());

      // The incoming region is operating on buffers, while linalg.generic
      // expects scalar SSA values. Add some allocs around the original op to
      // make it compatible.
      auto arg_type = block->getArgument(0).getType().cast<MemRefType>();
      Value alloc_a = rewriter.create<AllocaOp>(loc, arg_type);
      Value alloc_b = rewriter.create<AllocaOp>(loc, arg_type);
      Value alloc_res = rewriter.create<AllocaOp>(loc, arg_type);

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
      rewriter.create<StoreOp>(loc, entry_block->getArgument(0), alloc_a);
      rewriter.create<StoreOp>(loc, entry_block->getArgument(1), alloc_b);
      rewriter.replaceOp(entry_block->getTerminator(), {});

      // Load & yield the result.
      rewriter.setInsertionPointToEnd(entry_block);
      auto load_res = rewriter.create<LoadOp>(loc, alloc_res);
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
        getHloOpResultType<isLHLO>(op).template cast<ShapedType>();
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

class SliceConverter : public OpConversionPattern<lmhlo::SliceOp> {
 public:
  using OpConversionPattern<lmhlo::SliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lmhlo::SliceOp slice_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = slice_op.getLoc();
    auto arg_type =
        slice_op.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects known-rank args");
      return failure();
    }

    SmallVector<Value, 3> ranges;
    for (int i = 0, e = arg_type.getRank(); i < e; ++i) {
      Value start_index = rewriter.create<ConstantIndexOp>(
          loc, slice_op.start_indices().getValue<int64_t>(i));
      Value limit_index = rewriter.create<ConstantIndexOp>(
          loc, slice_op.limit_indices().getValue<int64_t>(i));
      Value stride = rewriter.create<ConstantIndexOp>(
          loc, slice_op.strides().getValue<int64_t>(i));
      ranges.push_back(rewriter.create<linalg::RangeOp>(loc, start_index,
                                                        limit_index, stride));
    }
    auto linalg_slice =
        rewriter.create<linalg::SliceOp>(loc, slice_op.getOperand(0), ranges);
    rewriter.create<linalg::CopyOp>(loc, linalg_slice, slice_op.getOperand(1));
    rewriter.eraseOp(slice_op);
    return success();
  }
};

void populateLHLOToLinalgConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<BroadcastConverter<lmhlo::BroadcastOp>,
                   ConstConverter,
                   ConvToLinalgConverter,
                   IotaConverter<lmhlo::IotaOp>,
                   LhloBroadcastInDimConverter,
                   PointwiseToLinalgConverter<lmhlo::AbsOp>,
                   PointwiseToLinalgConverter<lmhlo::AddOp>,
                   PointwiseToLinalgConverter<lmhlo::AndOp>,
                   PointwiseToLinalgConverter<lmhlo::Atan2Op>,
                   PointwiseToLinalgConverter<lmhlo::CeilOp>,
                   PointwiseToLinalgConverter<lmhlo::CompareOp>,
                   PointwiseToLinalgConverter<lmhlo::ComplexOp>,
                   PointwiseToLinalgConverter<lmhlo::ConvertOp>,
                   // TODO(ataei): Remove this pattern, CopyOp is folded away.
                   PointwiseToLinalgConverter<lmhlo::CopyOp>,
                   PointwiseToLinalgConverter<lmhlo::CosOp>,
                   PointwiseToLinalgConverter<lmhlo::DivOp>,
                   PointwiseToLinalgConverter<lmhlo::ExpOp>,
                   PointwiseToLinalgConverter<lmhlo::FloorOp>,
                   PointwiseToLinalgConverter<lmhlo::ImagOp>,
                   PointwiseToLinalgConverter<lmhlo::IsFiniteOp>,
                   PointwiseToLinalgConverter<lmhlo::LogOp>,
                   PointwiseToLinalgConverter<lmhlo::MaxOp>,
                   PointwiseToLinalgConverter<lmhlo::MinOp>,
                   PointwiseToLinalgConverter<lmhlo::MulOp>,
                   PointwiseToLinalgConverter<lmhlo::NegOp>,
                   PointwiseToLinalgConverter<lmhlo::NotOp>,
                   PointwiseToLinalgConverter<lmhlo::OrOp>,
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
                   SliceConverter,
                   TransposeConverter<lmhlo::TransposeOp>
                  >(context);
  // clang-format on
}

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
    registry.insert<AffineDialect, linalg::LinalgDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           AffineDialect>();

    auto func = getFunction();
    populateLHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct HloLegalizeToLinalgPass
    : public PassWrapper<HloLegalizeToLinalgPass, FunctionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    mhlo::populateHLOToLinalgConversionPattern(func.getContext(), &patterns);
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
                                          OwningRewritePatternList* patterns) {
  patterns
      ->insert<BroadcastConverter<mhlo::BroadcastOp, false>,
               HloBroadcastInDimConverter, IotaConverter<mhlo::IotaOp, false>,
               PointwiseToLinalgConverter<mhlo::AbsOp, false>,
               PointwiseToLinalgConverter<mhlo::AddOp, false>,
               PointwiseToLinalgConverter<mhlo::AndOp, false>,
               PointwiseToLinalgConverter<mhlo::Atan2Op, false>,
               PointwiseToLinalgConverter<mhlo::CeilOp, false>,
               PointwiseToLinalgConverter<mhlo::CompareOp, false>,
               PointwiseToLinalgConverter<mhlo::ComplexOp, false>,
               PointwiseToLinalgConverter<mhlo::ConvertOp, false>,
               PointwiseToLinalgConverter<mhlo::CopyOp, false>,
               PointwiseToLinalgConverter<mhlo::CosOp, false>,
               PointwiseToLinalgConverter<mhlo::DivOp, false>,
               PointwiseToLinalgConverter<mhlo::ExpOp, false>,
               PointwiseToLinalgConverter<mhlo::FloorOp, false>,
               PointwiseToLinalgConverter<mhlo::ImagOp, false>,
               PointwiseToLinalgConverter<mhlo::IsFiniteOp, false>,
               PointwiseToLinalgConverter<mhlo::LogOp, false>,
               PointwiseToLinalgConverter<mhlo::MaxOp, false>,
               PointwiseToLinalgConverter<mhlo::MinOp, false>,
               PointwiseToLinalgConverter<mhlo::MulOp, false>,
               PointwiseToLinalgConverter<mhlo::NegOp, false>,
               PointwiseToLinalgConverter<mhlo::NotOp, false>,
               PointwiseToLinalgConverter<mhlo::OrOp, false>,
               PointwiseToLinalgConverter<mhlo::RealOp, false>,
               PointwiseToLinalgConverter<mhlo::RemOp, false>,
               PointwiseToLinalgConverter<mhlo::RsqrtOp, false>,
               PointwiseToLinalgConverter<mhlo::SelectOp, false>,
               PointwiseToLinalgConverter<mhlo::ShiftLeftOp, false>,
               PointwiseToLinalgConverter<mhlo::ShiftRightArithmeticOp, false>,
               PointwiseToLinalgConverter<mhlo::ShiftRightLogicalOp, false>,
               PointwiseToLinalgConverter<mhlo::SinOp, false>,
               PointwiseToLinalgConverter<mhlo::SqrtOp, false>,
               PointwiseToLinalgConverter<mhlo::SubOp, false>,
               PointwiseToLinalgConverter<mhlo::TanhOp, false>,
               PointwiseToLinalgConverter<mhlo::XorOp, false>,
               ReshapeOpConverter<mhlo::ReshapeOp, false>,
               ReverseConverter<mhlo::ReverseOp, false>,
               TransposeConverter<mhlo::TransposeOp, false>>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeHloToLinalgPass() {
  return std::make_unique<HloLegalizeToLinalgPass>();
}
}  // namespace mhlo
}  // namespace mlir
