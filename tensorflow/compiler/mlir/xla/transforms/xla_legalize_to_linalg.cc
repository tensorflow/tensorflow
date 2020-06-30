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

#include "absl/memory/memory.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

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
ShapedType getXLAOpResultType(Operation* op) {
  return getResultValue<isLHLO>(op).getType().template cast<ShapedType>();
}

template <bool isLHLO = true>
bool verifyXLAOpBufferOrTensorSemantics(Operation* op) {
  auto verifyType = [&](Value val) -> bool {
    return (isLHLO && val.getType().isa<MemRefType>()) ||
           (!isLHLO && val.getType().isa<RankedTensorType>());
  };
  if (!llvm::all_of(op->getOperands(), verifyType)) return false;
  return isLHLO ? op->getResults().empty()
                : llvm::all_of(op->getResults(), verifyType);
}

template <typename OpTy, bool isLHLO = true>
class PointwiseToLinalgConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto argType =
        op.getOperation()->getOperand(0).getType().template cast<ShapedType>();
    if (!argType.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects ranked args");
      return failure();
    }
    auto elemTy = argType.getElementType();
    if (!elemTy.isSignlessIntOrFloat() && !elemTy.template isa<ComplexType>()) {
      return failure();
    }

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<AffineMap, 2> indexing_maps;
    SmallVector<Type, 4> bodyArgTypes, bodyResultTypes, opResultTypes;

    // This doesnt account for implicit broadcast, but the working assumption
    // here is that are broadcasts have been made explicit.
    unsigned nloops = argType.getRank();

    if (isLHLO && !nloops) return failure();

    int operandCount = (isLHLO ? args.size() - 1 : args.size());
    auto verifyArgOrResultType = [&](Value val) -> ShapedType {
      auto shapedType = val.getType().dyn_cast<ShapedType>();
      if (!shapedType ||
          (!shapedType.isa<MemRefType>() &&
           !shapedType.isa<RankedTensorType>()) ||
          shapedType.getRank() != nloops)
        return nullptr;
      indexing_maps.emplace_back(
          nloops ? rewriter.getMultiDimIdentityMap(nloops)
                 : AffineMap::get(nloops, 0, rewriter.getContext()));
      return shapedType;
    };
    for (const auto& arg : llvm::enumerate(args)) {
      auto shapedType = verifyArgOrResultType(arg.value());
      if (!shapedType) return failure();
      auto& result_or_body_arg =
          arg.index() < operandCount ? bodyArgTypes : bodyResultTypes;
      result_or_body_arg.emplace_back(shapedType.getElementType());
    }
    if (!isLHLO) {
      // HLO operations have return as tensor types.
      assert(bodyResultTypes.empty() &&
             "When lowering HLO ops result can't be part of arguments");
      Value result = op.getOperation()->getResult(0);
      auto shapedType = verifyArgOrResultType(result);
      if (!shapedType) return failure();
      bodyResultTypes.push_back(shapedType.getElementType());
      opResultTypes.push_back(shapedType);
    }

    int64_t args_count = bodyArgTypes.size();
    int64_t results_count = bodyResultTypes.size();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, opResultTypes, args, args_count, results_count, indexing_maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
          // TODO(ravishankarm) : For now use the method in xla_lhlo namespace.
          // That method needs to be moved out of there.
          Value opResult = xla_lhlo::XlaOpToStdScalarOp::map<OpTy>(
              op, bodyResultTypes,
              llvm::to_vector<2>(args.take_front(args_count)), &rewriter);
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
        });
    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
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
    auto argType =
        lhlo_op.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.getElementType().isSignlessIntOrFloat() ||
        (argType.getRank() != 0)) {
      return failure();
    }

    // Create two loads from the input.
    auto lhs = rewriter.create<LoadOp>(loc, lhlo_op.lhs());
    auto rhs = rewriter.create<LoadOp>(loc, lhlo_op.rhs());
    // TODO(ravishankarm) : Move this method out of xla_lhlo namespace.
    Value opResult = xla_lhlo::XlaOpToStdScalarOp::map<LhloOp>(
        lhlo_op, argType.getElementType(), llvm::ArrayRef<Value>{lhs, rhs},
        &rewriter);
    rewriter.create<StoreOp>(loc, opResult, lhlo_op.out());
    rewriter.eraseOp(lhlo_op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// xla_lhlo.convolution conversion pattern.
//===----------------------------------------------------------------------===//

/// Converts xla_lhlo.convolution operation to a linalg.conv op.
struct ConvToLinalgConverter : public OpConversionPattern<xla_lhlo::ConvOp> {
 public:
  using OpConversionPattern<xla_lhlo::ConvOp>::OpConversionPattern;

  //  This code has been adapted from IREE's
  //  (https://github.com/google/iree/) xla_hlo -> linalg conversion.
  LogicalResult matchAndRewrite(
      xla_lhlo::ConvOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    // Check validity of dimension information.
    if (const xla_lhlo::ConvDimensionNumbers& dimensionNumbers =
            op.dimension_numbers()) {
      const int inputSpatialRank =
          llvm::size(dimensionNumbers.input_spatial_dimensions());
      // The dimensions for input should follow the order of
      // batch_count, spatial_dims..., input_feature_count.
      if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
          dimensionNumbers.input_feature_dimension().getInt() !=
              (inputSpatialRank + 1))
        return failure();

      const int kernelSpatialRank =
          llvm::size(dimensionNumbers.kernel_spatial_dimensions());
      // The dimensions for filter should follow the order of
      // spatial_dims..., input_feature_count, num_output_feature_count.
      if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
              kernelSpatialRank ||
          dimensionNumbers.kernel_output_feature_dimension().getInt() !=
              (kernelSpatialRank + 1))
        return failure();

      const int outputSpatialRank =
          llvm::size(dimensionNumbers.output_spatial_dimensions());
      // The dimensions for output should follow the order of
      // batch_count, spatial_dims.., output_feature_count.
      if (dimensionNumbers.output_batch_dimension().getInt() != 0 ||
          dimensionNumbers.output_feature_dimension().getInt() !=
              (outputSpatialRank + 1))
        return failure();

      if (inputSpatialRank != outputSpatialRank ||
          inputSpatialRank != kernelSpatialRank)
        return failure();

      auto inputSpatialDim =
          dimensionNumbers.input_spatial_dimensions().begin();
      auto kernelSpatialDim =
          dimensionNumbers.kernel_spatial_dimensions().begin();
      auto outputSpatialDim =
          dimensionNumbers.output_spatial_dimensions().begin();
      // Check if spatial dims are ordered correctly.
      for (int i = 0; i < inputSpatialRank; ++i) {
        const int dim = i + 1;
        if ((*inputSpatialDim++).getZExtValue() != dim ||
            (*outputSpatialDim++).getZExtValue() != dim ||
            (*kernelSpatialDim++).getZExtValue() != i)
          return failure();
      }
    }

    // TODO: LHS dilation for deconvolution not supported yet.
    if (op.lhs_dilation()) {
      return failure();
    }

    llvm::SmallVector<Attribute, 4> strides;
    if (auto windowStrides = op.window_strides()) {
      auto range = windowStrides->getAttributeValues();
      strides.assign(range.begin(), range.end());
    }
    auto stridesArg = ArrayAttr::get(strides, op.getContext());

    llvm::SmallVector<Attribute, 2> dilation;
    if (auto rhsDilation = op.rhs_dilation()) {
      auto range = rhsDilation->getAttributeValues();
      dilation.assign(range.begin(), range.end());
    } else {
      // Default dilation of 1.
      dilation.resize(2, IntegerAttr::get(rewriter.getIntegerType(64), 1));
    }
    auto dilationArg = ArrayAttr::get(dilation, op.getContext());

    // Set padding only if it is non-zero.
    DenseIntElementsAttr padding = op.paddingAttr();
    if (!padding || !llvm::any_of(padding.getValues<APInt>(), [](APInt intVal) {
          return !intVal.isNullValue();
        })) {
      padding = nullptr;
    }

    // The order of input and filter are switched with linalg.conv.
    rewriter.replaceOpWithNewOp<linalg::ConvOp>(
        op, args[1], args[0], args[2], stridesArg, dilationArg, padding);
    return success();
  }
};

/// Base class for lowering xla operations that have one operand and one result,
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
    if (!verifyXLAOpBufferOrTensorSemantics<isLHLO>(op)) return failure();
    auto resultType = getXLAOpResultType<isLHLO>(op);

    SmallVector<AffineMap, 2> indexing_maps =
        Derived::getIndexingMaps(op, &rewriter);
    if (indexing_maps.empty()) return failure();

    auto nloops = resultType.getRank();
    auto loc = op.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, isLHLO ? ArrayRef<Type>{} : resultType, args, /*inputCount=*/1,
        /*outputCount=*/1, indexing_maps, GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
        });

    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
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

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy broadcastOp,
                                                   Builder* b) {
    ShapedType inputType =
        broadcastOp.operand().getType().template cast<ShapedType>();
    unsigned inputRank = inputType.getRank();
    unsigned nloops = getXLAOpResultType<isLHLO>(broadcastOp).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned numPrependedDims = llvm::size(broadcastOp.broadcast_sizes());
    SmallVector<AffineExpr, 4> inputDimExprs;
    inputDimExprs.reserve(inputRank);
    for (int i = 0; i < inputRank; ++i) {
      inputDimExprs.push_back(b->getAffineDimExpr(numPrependedDims + i));
    }

    AffineMap inputMap;
    MLIRContext* context = b->getContext();
    if (inputDimExprs.empty()) {
      // The input is a scalar, i.e. this is a scalar broadcast op.
      inputMap = AffineMap::get(nloops, /*symbolCount=*/0, context);
    } else {
      inputMap =
          AffineMap::get(nloops, /*symbolCount=*/0, inputDimExprs, context);
    }
    return {inputMap, b->getMultiDimIdentityMap(nloops)};
  }
};

class HloBroadcastInDimConverter
    : public DataMovementOpConverter<HloBroadcastInDimConverter,
                                     xla_hlo::BroadcastInDimOp, false> {
 public:
  using DataMovementOpConverter<HloBroadcastInDimConverter,
                                xla_hlo::BroadcastInDimOp,
                                false>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(
      xla_hlo::BroadcastInDimOp broadcastOp, Builder* b) {
    auto resultType = getXLAOpResultType<false>(broadcastOp);
    auto operandType =
        broadcastOp.operand().getType().template cast<ShapedType>();
    unsigned nloops = resultType.getRank();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operandType.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    auto operandShape = operandType.getShape();
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(nloops);

    if (broadcastOp.broadcast_dimensions()) {
      for (const auto& broadcastDim :
           enumerate(broadcastOp.broadcast_dimensions().getIntValues())) {
        int size = broadcastDim.value().getSExtValue();
        bool expansion_needed = operandShape[broadcastDim.index()] == 1 &&
                                resultType.getShape()[size] != 1;
        dimExprs.push_back(expansion_needed ? b->getAffineConstantExpr(0)
                                            : b->getAffineDimExpr(size));
      }
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dimExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

class LhloBroadcastInDimConverter
    : public OpConversionPattern<xla_lhlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern<xla_lhlo::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::BroadcastInDimOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    xla_lhlo::BroadcastInDimOp::Adaptor operand_adaptor(args);
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
          loc, llvm::None, llvm::makeArrayRef(operand_adaptor.output()),
          /*inputCount=*/0, /*outputCount=*/1,
          llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
          GetNParallelLoopsAttrs(nloops),
          [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(loc, val);
          });

    } else {
      auto indexing_maps = getIndexingMaps(op, broadcast_dims, result_shape,
                                           operand_type, &rewriter);
      rewriter.create<linalg::GenericOp>(
          loc, llvm::None,
          llvm::makeArrayRef({operand, operand_adaptor.output()}),
          /*inputCount=*/1, /*outputCount=*/1, indexing_maps,
          GetNParallelLoopsAttrs(nloops),
          [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
            nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
          });
    }
    rewriter.replaceOp(op, llvm::None);
    return success();
  }

  // Inserts 'linalg.reshape' if there is a size-1 dim expansion.
  std::pair<Value, SmallVector<int64_t, 2>> InsertReshapeIfNecessary(
      xla_lhlo::BroadcastInDimOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const {
    xla_lhlo::BroadcastInDimOp::Adaptor operand_adaptor(args);
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

  SmallVector<AffineMap, 2> getIndexingMaps(xla_lhlo::BroadcastInDimOp op,
                                            ArrayRef<int64_t> broadcastDims,
                                            ArrayRef<int64_t> resultShape,
                                            MemRefType operandType,
                                            Builder* b) const {
    unsigned nloops = resultShape.size();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operandType.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    auto operandShape = operandType.getShape();
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(nloops);

    for (const auto& broadcastDim : llvm::enumerate(broadcastDims)) {
      int size = broadcastDim.value();
      bool expansion_needed =
          operandShape[broadcastDim.index()] == 1 && resultShape[size] != 1;
      if (expansion_needed) {
        op.emitOpError(
            "BroadcastInDimOp lowering to Linalg does not support size-1 "
            "dimensions expansion.");
      }
      dimExprs.push_back(b->getAffineDimExpr(size));
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dimExprs, b->getContext()),
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
    auto resultType =
        getXLAOpResultType<isLHLO>(op).template cast<ShapedType>();
    auto nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultType.getRank());
    for (auto permutation : llvm::enumerate(op.permutation())) {
      inputExprs[permutation.value().getZExtValue()] =
          b->getAffineDimExpr(permutation.index());
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
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
      OpTy reshapeOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    if (!verifyXLAOpBufferOrTensorSemantics<isLHLO>(reshapeOp))
      return failure();
    ShapedType operandType =
        reshapeOp.operand().getType().template cast<ShapedType>();
    ShapedType resultType = getXLAOpResultType<isLHLO>(reshapeOp);

    if (!operandType.hasStaticShape() || !resultType.hasStaticShape())
      return failure();

    // Compute the reassociation maps for the linalg operation.
    ArrayRef<int64_t> srcShape =
        (operandType.getRank() > resultType.getRank() ? operandType.getShape()
                                                      : resultType.getShape());
    ArrayRef<int64_t> dstShape =
        (operandType.getRank() > resultType.getRank() ? resultType.getShape()
                                                      : operandType.getShape());
    unsigned currSrcDim = 0, currDstDim = 0;
    SmallVector<linalg::ReassociationExprs, 4> reassociationMap(
        dstShape.size());
    while (currSrcDim < srcShape.size() && currDstDim < dstShape.size()) {
      int64_t dstSize = dstShape[currDstDim];
      int64_t srcSize = srcShape[currSrcDim];
      while (srcSize < dstSize && currSrcDim < srcShape.size()) {
        reassociationMap[currDstDim].push_back(
            rewriter.getAffineDimExpr(currSrcDim++));
        srcSize *= srcShape[currSrcDim];
      }
      if (srcSize == dstSize) {
        reassociationMap[currDstDim].push_back(
            rewriter.getAffineDimExpr(currSrcDim++));
        // If the next dim in dstShape is not 1, treat subsequent dims in
        // srcShape which are 1 to be collapsed.
        if (currDstDim == dstShape.size() - 1 ||
            dstShape[currDstDim + 1] != 1) {
          while (currSrcDim < srcShape.size() && srcShape[currSrcDim] == 1) {
            reassociationMap[currDstDim].push_back(
                rewriter.getAffineDimExpr(currSrcDim++));
          }
        }
      } else {
        return failure();
      }
      currDstDim++;
    }
    if (currSrcDim != srcShape.size()) return failure();

    if (isLHLO) {
      Value reshapeBuffer = rewriter.create<linalg::ReshapeOp>(
          reshapeOp.getLoc(), resultType, args[0], reassociationMap);
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(
          reshapeOp, reshapeBuffer, args[1], /*inputPermutation =*/nullptr,
          /*outputPermutation =*/nullptr);
    } else {
      rewriter.replaceOpWithNewOp<linalg::TensorReshapeOp>(
          reshapeOp, resultType, args[0], reassociationMap);
    }
    return success();
  }
};

class IotaConverter : public OpConversionPattern<xla_lhlo::IotaOp> {
 public:
  using OpConversionPattern<xla_lhlo::IotaOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::IotaOp iotaOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto resultMemrefType =
        iotaOp.getOperand().getType().dyn_cast<MemRefType>();
    if (!resultMemrefType) return failure();

    auto resultElementType = resultMemrefType.getElementType();
    if (!resultElementType.isSignlessIntOrFloat()) return failure();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = resultMemrefType.getRank();

    rewriter.create<linalg::IndexedGenericOp>(
        iotaOp.getLoc(), ArrayRef<Type>{}, args,
        0,  // args_in
        1,  // args_out
        llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange ivs,
            ValueRange args) {
          Value castOp = nestedBuilder.create<IndexCastOp>(
              nestedLoc, ivs[iotaOp.iota_dimension().getZExtValue()],
              nestedBuilder.getIntegerType(
                  resultElementType.getIntOrFloatBitWidth()));
          if (resultElementType.isa<FloatType>()) {
            castOp = nestedBuilder.create<SIToFPOp>(nestedLoc, castOp,
                                                    resultElementType);
          }
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, castOp);
        });

    rewriter.replaceOp(iotaOp, llvm::None);
    return success();
  }
};

class ConstConverter : public OpConversionPattern<xla_lhlo::ConstOp> {
 public:
  using OpConversionPattern<xla_lhlo::ConstOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::ConstOp constOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = constOp.getLoc();
    auto valueAttr = constOp.value().cast<DenseElementsAttr>();
    if (valueAttr.getType().getRank() != 0) return failure();
    auto stdConstOp =
        rewriter.create<mlir::ConstantOp>(loc, valueAttr.getValue({}));
    rewriter.create<mlir::StoreOp>(loc, stdConstOp, constOp.getOperand());
    rewriter.eraseOp(constOp);
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
    auto resultType =
        getXLAOpResultType<isLHLO>(op).template cast<ShapedType>();
    auto nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.reserve(nloops);
    for (int i = 0; i < nloops; ++i)
      inputExprs.push_back(b->getAffineDimExpr(i));
    for (auto dim : op.dimensions()) {
      int i = dim.getZExtValue();
      if (resultType.isDynamicDim(i)) return {};
      int n = resultType.getShape()[i];
      inputExprs[i] = b->getAffineConstantExpr(n - 1) - inputExprs[i];
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

class SliceConverter : public OpConversionPattern<xla_lhlo::SliceOp> {
 public:
  using OpConversionPattern<xla_lhlo::SliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      xla_lhlo::SliceOp sliceOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = sliceOp.getLoc();
    auto argType =
        sliceOp.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects known-rank args");
      return failure();
    }

    SmallVector<Value, 3> ranges;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      Value start_index = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.start_indices().getValue<int64_t>(i));
      Value limit_index = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.limit_indices().getValue<int64_t>(i));
      Value stride = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.strides().getValue<int64_t>(i));
      ranges.push_back(rewriter.create<linalg::RangeOp>(loc, start_index,
                                                        limit_index, stride));
    }
    auto linalg_slice =
        rewriter.create<linalg::SliceOp>(loc, sliceOp.getOperand(0), ranges);
    rewriter.create<linalg::CopyOp>(loc, linalg_slice, sliceOp.getOperand(1));
    rewriter.eraseOp(sliceOp);
    return success();
  }
};

void populateLHLOToLinalgConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<BroadcastConverter<xla_lhlo::BroadcastOp>,
                   ConstConverter,
                   ConvToLinalgConverter,
                   IotaConverter,
                   LhloBroadcastInDimConverter,
                   PointwiseToLinalgConverter<xla_lhlo::AbsOp>,
                   PointwiseToLinalgConverter<xla_lhlo::AddOp>,
                   PointwiseToLinalgConverter<xla_lhlo::AndOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CeilOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CompareOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ComplexOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ConvertOp>,
                   // TODO(ataei): Remove this pattern, CopyOp is folded away.
                   PointwiseToLinalgConverter<xla_lhlo::CopyOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CosOp>,
                   PointwiseToLinalgConverter<xla_lhlo::DivOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ExpOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ImagOp>,
                   PointwiseToLinalgConverter<xla_lhlo::LogOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MaxOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MinOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MulOp>,
                   PointwiseToLinalgConverter<xla_lhlo::NegOp>,
                   PointwiseToLinalgConverter<xla_lhlo::RealOp>,
                   PointwiseToLinalgConverter<xla_lhlo::RemOp>,
                   PointwiseToLinalgConverter<xla_lhlo::RsqrtOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SelectOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SignOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SinOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SqrtOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SubOp>,
                   PointwiseToLinalgConverter<xla_lhlo::TanhOp>,
                   ReshapeOpConverter<xla_lhlo::ReshapeOp>,
                   ReverseConverter<xla_lhlo::ReverseOp>,
                   ScalarPointwiseToStandardConverter<xla_lhlo::AddOp>,
                   SliceConverter
                  >(context);
  // clang-format on
}

// Converts LHLO ops to Linalg generic.
// Sample result for xla_lhlo::AddOp.
//
// "xla_lhlo.add"(%arg1, %arg2, %out) :
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
//     args_in = 2,
//     args_out = 1,
//     indexing_maps = [#map0, #map0, #map0],
//     iterator_types = ["parallel", "parallel"],
// } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
struct LhloLegalizeToLinalg
    : public PassWrapper<LhloLegalizeToLinalg, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    populateLHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

struct HloLegalizeToLinalg
    : public PassWrapper<HloLegalizeToLinalg, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    xla_hlo::populateHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace xla_lhlo {
std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToLinalgPass() {
  return absl::make_unique<LhloLegalizeToLinalg>();
}

static PassRegistration<LhloLegalizeToLinalg> legalize_lhlo_pass(
    "lhlo-legalize-to-linalg", "Legalize from LHLO dialect to Linalg dialect");
}  // namespace xla_lhlo

namespace xla_hlo {

void populateHLOToLinalgConversionPattern(MLIRContext* context,
                                          OwningRewritePatternList* patterns) {
  patterns->insert<BroadcastConverter<xla_hlo::BroadcastOp, false>,
                   HloBroadcastInDimConverter,
                   PointwiseToLinalgConverter<xla_hlo::AbsOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AddOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AndOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CeilOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CompareOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::ComplexOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::ConvertOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CopyOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CosOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::DivOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::ExpOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::ImagOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::LogOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MaxOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MinOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MulOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::NegOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::RealOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::RemOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::RsqrtOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SelectOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SinOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SqrtOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SubOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::TanhOp, false>,
                   ReshapeOpConverter<xla_hlo::ReshapeOp, false>,
                   ReverseConverter<xla_hlo::ReverseOp, false>,
                   TransposeConverter<xla_hlo::TransposeOp, false>>(context);
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeHloToLinalgPass() {
  return absl::make_unique<HloLegalizeToLinalg>();
}

static PassRegistration<HloLegalizeToLinalg> legalize_hlo_pass(
    "hlo-legalize-to-linalg", "Legalize from HLO dialect to Linalg dialect");
}  // namespace xla_hlo
}  // namespace mlir
