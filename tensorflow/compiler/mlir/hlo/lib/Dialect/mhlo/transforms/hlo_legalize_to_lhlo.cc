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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <algorithm>
#include <utility>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_hlo_to_lhlo_op.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

Value insertDynamicAlloc(Location loc, Value result, Value shapeOperand,
                         ConversionPatternRewriter* rewriter) {
  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!resultType) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects ranked results";
  }
  auto memrefType =
      MemRefType::get(resultType.getShape(), resultType.getElementType());

  // Extract the required element out of the vector.
  SmallVector<Value, 4> dynamicOperands;
  for (const auto& shapeElement : llvm::enumerate(resultType.getShape())) {
    if (shapeElement.value() != ShapedType::kDynamicSize) continue;
    Value index =
        rewriter->create<arith::ConstantIndexOp>(loc, shapeElement.index());
    Value allocOperand =
        rewriter->create<tensor::ExtractOp>(loc, shapeOperand, index);
    if (!allocOperand.getType().isIndex()) {
      allocOperand = rewriter->create<arith::IndexCastOp>(
          loc, rewriter->getIndexType(), allocOperand);
    }
    dynamicOperands.push_back(allocOperand);
  }

  return rewriter->create<memref::AllocOp>(loc, memrefType, dynamicOperands);
}

Value insertAlloc(Location loc, OpResult result,
                  ConversionPatternRewriter* rewriter) {
  auto resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!resultType || !resultType.hasStaticShape()) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects statically shaped results";
  }
  auto memrefType =
      MemRefType::get(resultType.getShape(), resultType.getElementType());
  OpBuilder::InsertionGuard guard(*rewriter);
  rewriter->setInsertionPoint(result.getDefiningOp());
  auto alloc = rewriter->create<memref::AllocOp>(loc, memrefType);
  return alloc;
}

/// Converts the results of the operation `op` to memref types and append them
/// to the `results` vector.
LogicalResult convertResults(Operation* op, SmallVectorImpl<Value>& results,
                             ConversionPatternRewriter& rewriter) {
  size_t numOperands = results.size();
  SmallVector<Value, 2> tensorOperands;
  for (const auto& result : llvm::enumerate(op->getResults())) {
    RankedTensorType resultType =
        result.value().getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    if (resultType.hasStaticShape()) {
      results.push_back(insertAlloc(op->getLoc(), result.value(), &rewriter));
      continue;
    }
    auto shapeTypeOp = dyn_cast<InferShapedTypeOpInterface>(op);
    if (!shapeTypeOp) return failure();

    if (tensorOperands.empty()) {
      for (auto operand : ArrayRef<Value>(results).take_front(numOperands)) {
        auto operandType = operand.getType().dyn_cast<MemRefType>();
        if (!operandType) return failure();
        tensorOperands.push_back(rewriter.create<bufferization::ToTensorOp>(
            op->getLoc(),
            RankedTensorType::get(operandType.getShape(),
                                  operandType.getElementType()),
            operand));
      }
    }

    SmallVector<Value, 1> resultsShape;
    auto status = shapeTypeOp.reifyReturnTypeShapes(rewriter, tensorOperands,
                                                    resultsShape);
    if (failed(status)) return failure();
    results.push_back(insertDynamicAlloc(
        op->getLoc(), result.value(), resultsShape[result.index()], &rewriter));
  }
  return success();
}

template <typename HloOpTy>
class HloToLhloOpConverter : public BaseOpConversion<HloOpTy> {
 public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    SmallVector<Value, 4> bufferArgs(adaptor.getOperands());
    if (failed(convertResults(op, bufferArgs, rewriter))) return failure();
    rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(op->getLoc(), llvm::None,
                                                bufferArgs, op->getAttrs());
    rewriter.replaceOp(op, llvm::makeArrayRef(bufferArgs)
                               .drop_front(adaptor.getOperands().size()));
    return success();
  }
};

// This specialization exists so that LMHLO's Dot can be given a specific set of
// dimension numbers, when lowering from MHLO's Dot, which does not have
// dimension numbers (it uses DotGeneral for this generalized notion of dot
// products). When these two dialects are in sync with respect to the
// Dot/DotGeneral issue, this specialization should be deleted.
template <>
class HloToLhloOpConverter<mhlo::DotOp> : public BaseOpConversion<mhlo::DotOp> {
 public:
  using BaseOpConversion<mhlo::DotOp>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      mhlo::DotOp hloOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    SmallVector<Value, 2> bufferArgs(adaptor.getOperands());
    if (failed(convertResults(op, bufferArgs, rewriter))) return failure();

    auto dotOp = rewriter.create<lmhlo::DotOp>(op->getLoc(), llvm::None,
                                               bufferArgs, op->getAttrs());
    // MHLO's Dot uses rank-2 operands, of the form ([N, M], [M, O]) -> [N, O].
    auto dimensionNumbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{}, /*lhsContractingDimensions=*/{1},
        /*rhsContractingDimensions=*/{0});
    dotOp.dot_dimension_numbersAttr(dimensionNumbers);
    rewriter.replaceOp(
        op, ArrayRef<Value>(bufferArgs).slice(adaptor.getOperands().size()));
    return success();
  }
};

struct HloToLhloCustomCallOpConverter
    : public BaseOpConversion<mhlo::CustomCallOp> {
 public:
  using BaseOpConversion<mhlo::CustomCallOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp hloOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    SmallVector<Value, 2> bufferArgs(adaptor.getOperands());
    if (failed(convertResults(op, bufferArgs, rewriter))) return failure();

    auto lhloOp = rewriter.create<lmhlo::CustomCallOp>(
        op->getLoc(), llvm::None, bufferArgs, op->getAttrs());
    // Setup AttrSizedOperandSegments attribute to indicate number of operands
    // for args and outputs.
    const int32_t segments[2] = {
        static_cast<int32_t>(adaptor.getOperands().size()),
        static_cast<int32_t>(op->getNumResults())};
    lhloOp->setAttr(lhloOp.getOperandSegmentSizeAttr(),
                    rewriter.getI32VectorAttr(segments));

    rewriter.replaceOp(
        op, ArrayRef<Value>(bufferArgs).slice(adaptor.getOperands().size()));
    return success();
  }
};

struct HloToLhloDotGeneralOpConverter
    : public BaseOpConversion<mhlo::DotGeneralOp> {
  using BaseOpConversion<mhlo::DotGeneralOp>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp dotGeneralOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = dotGeneralOp.getOperation();

    if (op->getResults().empty()) return failure();
    OpResult result = op->getResults()[0];
    RankedTensorType resultType = result.getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    // The third buffer argument will be filled with what used to be the return
    // type of the DotGeneral.
    if (adaptor.getOperands().size() != 2) return failure();
    std::array<Value, 3> bufferArgs = {
        adaptor.getOperands()[0], adaptor.getOperands()[1], {}};

    if (resultType.hasStaticShape()) {
      bufferArgs[2] = insertAlloc(op->getLoc(), result, &rewriter);
    } else {
      SmallVector<Value, 1> resultsShape;
      auto shapeTypeOp = dyn_cast<InferShapedTypeOpInterface>(op);
      if (failed(shapeTypeOp.reifyReturnTypeShapes(
              rewriter, adaptor.getOperands(), resultsShape)))
        return failure();

      bufferArgs[2] = insertDynamicAlloc(op->getLoc(), result,
                                         resultsShape.front(), &rewriter);
    }

    rewriter.create<lmhlo::DotOp>(op->getLoc(), llvm::None, bufferArgs,
                                  op->getAttrs());
    rewriter.replaceOp(op, bufferArgs[2]);
    return success();
  }
};

template <typename HloOpTy>
struct HloToLhloReduceLikeOpConverter : public BaseOpConversion<HloOpTy> {
 public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    auto loc = op->getLoc();
    if (!llvm::hasSingleElement(hloOp.body())) {
      return op->emitOpError()
             << "tensor to buffer conversion expects a single block "
                "in the region containing the operation";
    }
    SmallVector<Value, 4> bufferArgs(adaptor.getOperands());
    if (failed(convertResults(op, bufferArgs, rewriter))) return failure();
    auto newOp = rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(
        loc, llvm::None, bufferArgs, op->getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(hloOp.body(), newOp.body(), newOp.body().end());

    // Convert the region signature to memref and add extra result.
    auto& entryBlock = newOp.body().front();
    TypeConverter::SignatureConversion sigConversion(
        adaptor.getOperands().size());
    for (auto arg : entryBlock.getArguments()) {
      auto oldType = arg.getType().template cast<TensorType>();
      auto newType =
          MemRefType::get(oldType.getShape(), oldType.getElementType());
      sigConversion.addInputs(arg.getArgNumber(), newType);
    }
    auto returnOp = cast<mhlo::ReturnOp>(entryBlock.getTerminator());
    if (auto tupleTy = returnOp.results()
                           .front()
                           .getType()
                           .template dyn_cast<TupleType>()) {
      auto* tupleOp = returnOp.getODSOperands(0).front().getDefiningOp();
      returnOp.getOperation()->dropAllReferences();
      rewriter.eraseOp(tupleOp);
      returnOp.getOperation()->setOperands(tupleOp->getOperands());
      for (auto ty : tupleTy) {
        auto tensorTy = ty.template cast<TensorType>();
        sigConversion.addInputs(
            MemRefType::get(tensorTy.getShape(), tensorTy.getElementType()));
      }
    } else {
      for (auto result : returnOp.results()) {
        auto resultType = result.getType().template cast<TensorType>();
        sigConversion.addInputs({MemRefType::get(resultType.getShape(),
                                                 resultType.getElementType())});
      }
    }
    rewriter.applySignatureConversion(&newOp.body(), sigConversion);

    rewriter.replaceOp(
        op, ArrayRef<Value>(bufferArgs).slice(adaptor.getOperands().size()));

    return success();
  }
};

// Legalize mhlo.return to a lmhlo.copy and lmhlo.terminator.
struct HloToLhloReturnOpConverter : public BaseOpConversion<mhlo::ReturnOp> {
 public:
  using BaseOpConversion<mhlo::ReturnOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto& entryBlock = op->getParentRegion()->front();
    auto numArguments = entryBlock.getNumArguments();
    if (adaptor.getOperands().size() > numArguments) {
      return op.emitError(
          "The number of operands that need Copy operations is more "
          "than the number of target function arguments.");
    }

    // The index of the first output block argument.
    auto destArgIdx = numArguments - adaptor.getOperands().size();

    // Create a lmhlo.copy for each operand of mhlo.return.
    for (Value operand : adaptor.getOperands()) {
      rewriter.create<lmhlo::CopyOp>(loc, operand,
                                     entryBlock.getArgument(destArgIdx));
      ++destArgIdx;
    }
    rewriter.replaceOpWithNewOp<lmhlo::TerminatorOp>(op);
    return success();
  }
};

// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
// buffers if necessary.
//
// Example fusion with HLO ops.
//
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "lmhlo.fusion"() ({
//     %0 = bufferization.to_tensor %arg1 : memref<2x2xf32>
//     %1 = bufferization.to_tensor %arg2 : memref<2x2xf32>
//     %2 = "mhlo.add"(%0, %1) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %3 = bufferization.to_tensor %arg0 : memref<2x2xf32>
//     %4 = "mhlo.multiply"(%2, %3) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     tensor_store %4, %arg3 : memref<2x2xf32>
//     "lmhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return
// }
//
// Transformed fusion with LHLO ops.
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "lmhlo.fusion"() ({
//     %0 = alloc() : memref<2x2xf32>
//     "lmhlo.add"(%arg1, %arg2, %0) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "lmhlo.multiply"(%0, %arg0, %arg3) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "lmhlo.terminator"() : () -> ()
//   }) : () -> ()
//   return
// }
//
// FuncOp signature conversion example:
//
// func @func_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
//   %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) ->
//   tensor<4xf32> %1 = "mhlo.add"(%arg0, %0)  : (tensor<4xf32>,
//   tensor<4xf32>) -> tensor<4xf32> return %1 : tensor<4xf32>
// }
//
// Transformed function with an extra argument for the result. The types have
// been converted from tensor to memref.
//
// func @func_op(%arg0: memref<4xf32>,
//               %arg1: memref<4xf32>,
//               %arg2: memref<4xf32>) {
//   %0 = alloc() : memref<4xf32>

//   "lmhlo.maximum"(%arg0, %arg1, %0) :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   %1 = alloc() : memref<4xf32>
//   "lmhlo.add"(%arg0, %0, %1) :
//         (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
//   "lmhlo.copy"(%1, %arg2) : (memref<4xf32>, memref<4xf32>) -> ()
//   "lmhlo.terminator"() : () -> ()
// }

struct HloLegalizeToLhlo : public HloLegalizeToLhloPassBase<HloLegalizeToLhlo> {
  using HloLegalizeToLhloPassBase<HloLegalizeToLhlo>::HloLegalizeToLhloPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<bufferization::BufferizationDialect, lmhlo::LmhloDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
    shape::registerBufferizableOpInterfaceExternalModels(registry);
  }

 public:
  HloLegalizeToLhlo() = default;

  LogicalResult runOpInterfaceBufferization() {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    RewritePatternSet patterns(&getContext());
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.opFilter.allowDialect<shape::ShapeDialect>();
    return bufferization::bufferizeOp(getOperation(), options);
  }

  void runOnOperation() override {
    if (failed(runOpInterfaceBufferization())) {
      signalPassFailure();
      return;
    }

    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, bufferization::BufferizationDialect,
        lmhlo::LmhloDialect, memref::MemRefDialect, shape::ShapeDialect,
        func::FuncDialect, tensor::TensorDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    // bufferization.to_memref is illegal if it has uses.
    // TODO(b/175670649) Make bufferization.to_memref illegal.
    target.addDynamicallyLegalOp<mlir::bufferization::ToMemrefOp>(
        [](auto op) { return op->use_empty(); });

    bufferization::BufferizeTypeConverter converter;
    auto isMemRefType = [](Type type) { return type.isa<BaseMemRefType>(); };
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType) &&
             std::all_of(op.result_type_begin(), op.result_type_end(),
                         isMemRefType);
    });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                             isMemRefType);
        });

    populateHloToLhloConversionPattern(&context, &converter, &patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

// Simply lowers all mhlo ops to their lmhlo counterparts.
void populateDynamicHloToLhloConversionPattern(
    MLIRContext* context, bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns) {
  // clang-format off
  patterns->add<HloToLhloOpConverter<mhlo::DynamicBroadcastInDimOp>,
                   HloToLhloOpConverter<mhlo::DynamicGatherOp>,
                   HloToLhloOpConverter<mhlo::DynamicIotaOp>,
                   HloToLhloOpConverter<mhlo::DynamicPadOp>,
                   HloToLhloOpConverter<mhlo::DynamicReshapeOp>,
                   HloToLhloOpConverter<mhlo::RealDynamicSliceOp>
  >(*converter, context);
  // clang-format on
}

void populateHloToLhloConversionPattern(
    MLIRContext* context, bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns) {
  populateDynamicHloToLhloConversionPattern(context, converter, patterns);

  // clang-format off
  patterns->add<
      HloToLhloCustomCallOpConverter,
      HloToLhloDotGeneralOpConverter,
      HloToLhloOpConverter<mhlo::AbsOp>,
      HloToLhloOpConverter<mhlo::AddOp>,
      HloToLhloOpConverter<mhlo::AndOp>,
      HloToLhloOpConverter<mhlo::Atan2Op>,
      HloToLhloOpConverter<mhlo::BatchNormGradOp>,
      HloToLhloOpConverter<mhlo::BatchNormTrainingOp>,
      HloToLhloOpConverter<mhlo::BroadcastInDimOp>,
      HloToLhloOpConverter<mhlo::CeilOp>,
      HloToLhloOpConverter<mhlo::ClampOp>,
      HloToLhloOpConverter<mhlo::CompareOp>,
      HloToLhloOpConverter<mhlo::ComplexOp>,
      HloToLhloOpConverter<mhlo::ConcatenateOp>,
      HloToLhloOpConverter<mhlo::ConstOp>,
      HloToLhloOpConverter<mhlo::ConvOp>,
      HloToLhloOpConverter<mhlo::ConvertOp>,
      HloToLhloOpConverter<mhlo::CopyOp>,
      HloToLhloOpConverter<mhlo::CosOp>,
      HloToLhloOpConverter<mhlo::DivOp>,
      HloToLhloOpConverter<mhlo::DotOp>,
      HloToLhloOpConverter<mhlo::ExpOp>,
      HloToLhloOpConverter<mhlo::Expm1Op>,
      HloToLhloOpConverter<mhlo::FloorOp>,
      HloToLhloOpConverter<mhlo::GatherOp>,
      HloToLhloOpConverter<mhlo::ImagOp>,
      HloToLhloOpConverter<mhlo::IotaOp>,
      HloToLhloOpConverter<mhlo::IsFiniteOp>,
      HloToLhloOpConverter<mhlo::LogOp>,
      HloToLhloOpConverter<mhlo::LogisticOp>,
      HloToLhloOpConverter<mhlo::MaxOp>,
      HloToLhloOpConverter<mhlo::MinOp>,
      HloToLhloOpConverter<mhlo::MulOp>,
      HloToLhloOpConverter<mhlo::NegOp>,
      HloToLhloOpConverter<mhlo::NotOp>,
      HloToLhloOpConverter<mhlo::OrOp>,
      HloToLhloOpConverter<mhlo::PowOp>,
      HloToLhloOpConverter<mhlo::RealOp>,
      HloToLhloOpConverter<mhlo::RemOp>,
      HloToLhloOpConverter<mhlo::RsqrtOp>,
      HloToLhloOpConverter<mhlo::ReshapeOp>,
      HloToLhloOpConverter<mhlo::SelectOp>,
      HloToLhloOpConverter<mhlo::ShiftLeftOp>,
      HloToLhloOpConverter<mhlo::ShiftRightArithmeticOp>,
      HloToLhloOpConverter<mhlo::ShiftRightLogicalOp>,
      HloToLhloOpConverter<mhlo::SignOp>,
      HloToLhloOpConverter<mhlo::SinOp>,
      HloToLhloOpConverter<mhlo::SliceOp>,
      HloToLhloOpConverter<mhlo::SqrtOp>,
      HloToLhloOpConverter<mhlo::SubOp>,
      HloToLhloOpConverter<mhlo::TanhOp>,
      HloToLhloOpConverter<mhlo::TransposeOp>,
      HloToLhloOpConverter<mhlo::XorOp>,
      HloToLhloReduceLikeOpConverter<mhlo::ReduceOp>,
      HloToLhloReduceLikeOpConverter<mhlo::ReduceWindowOp>,
      HloToLhloReturnOpConverter
  >(*converter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToLhloPass() {
  return std::make_unique<HloLegalizeToLhlo>();
}

}  // namespace mhlo
}  // namespace mlir