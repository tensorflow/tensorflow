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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_hlo_to_lhlo_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
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
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

Value InsertDynamicAlloc(Location loc, Value result, Value shape_operand,
                         ConversionPatternRewriter* rewriter) {
  auto result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects ranked results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());

  // Extract the required element out of the vector.
  SmallVector<Value, 4> dynamic_operands;
  for (auto shape_element : llvm::enumerate(result_type.getShape())) {
    if (shape_element.value() != ShapedType::kDynamicSize) continue;
    Value index = rewriter->create<ConstantIndexOp>(loc, shape_element.index());
    Value alloc_operand =
        rewriter->create<tensor::ExtractOp>(loc, shape_operand, index);
    if (!alloc_operand.getType().isIndex()) {
      alloc_operand = rewriter->create<IndexCastOp>(loc, alloc_operand,
                                                    rewriter->getIndexType());
    }
    dynamic_operands.push_back(alloc_operand);
  }

  return rewriter->create<memref::AllocOp>(loc, memref_type, dynamic_operands);
}

Value InsertAlloc(Location loc, OpResult result,
                  ConversionPatternRewriter* rewriter) {
  auto result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type || !result_type.hasStaticShape()) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects statically shaped results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());
  OpBuilder::InsertionGuard guard(*rewriter);
  rewriter->setInsertionPoint(result.getDefiningOp());
  auto alloc = rewriter->create<memref::AllocOp>(loc, memref_type);
  return alloc;
}

/// Converts the results of the operation `op` to memref types and append them
/// to the `results` vector.
LogicalResult ConvertResults(Operation* op, SmallVectorImpl<Value>& results,
                             ConversionPatternRewriter& rewriter) {
  size_t num_operands = results.size();
  SmallVector<Value, 2> tensor_operands;
  for (auto result : llvm::enumerate(op->getResults())) {
    RankedTensorType resultType =
        result.value().getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    if (resultType.hasStaticShape()) {
      results.push_back(InsertAlloc(op->getLoc(), result.value(), &rewriter));
      continue;
    }
    auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
    if (!shape_type_op) return failure();

    if (tensor_operands.empty()) {
      for (auto operand : ArrayRef<Value>(results).take_front(num_operands)) {
        auto operand_type = operand.getType().dyn_cast<MemRefType>();
        if (!operand_type) return failure();
        tensor_operands.push_back(rewriter.create<memref::TensorLoadOp>(
            op->getLoc(),
            RankedTensorType::get(operand_type.getShape(),
                                  operand_type.getElementType()),
            operand));
      }
    }

    SmallVector<Value, 1> results_shape;
    auto status = shape_type_op.reifyReturnTypeShapes(rewriter, tensor_operands,
                                                      results_shape);
    if (failed(status)) return failure();
    results.push_back(InsertDynamicAlloc(op->getLoc(), result.value(),
                                         results_shape[result.index()],
                                         &rewriter));
  }
  return success();
}

template <typename HloOpTy>
class HloToLhloOpConverter : public BaseOpConversion<HloOpTy> {
 public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      HloOpTy hloOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();
    rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(op->getLoc(), llvm::None,
                                                buffer_args, op->getAttrs());
    rewriter.replaceOp(
        op, llvm::makeArrayRef(buffer_args).drop_front(operands.size()));
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
      mhlo::DotOp hloOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    SmallVector<Value, 2> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();

    // TODO(silvasean): Move this helper to MLIR core.
    auto make_elements_attr = [&rewriter](ArrayRef<int64_t> integers) {
      auto type = RankedTensorType::get({static_cast<int64_t>(integers.size())},
                                        rewriter.getIntegerType(64));
      return DenseIntElementsAttr::get(type, integers);
    };
    auto dotOp = rewriter.create<lmhlo::DotOp>(op->getLoc(), llvm::None,
                                               buffer_args, op->getAttrs());
    // MHLO's Dot uses rank-2 operands, of the form ([N, M], [M, O]) -> [N, O].
    auto dimension_numbers = mhlo::DotDimensionNumbers::get(
        make_elements_attr({}), make_elements_attr({}), make_elements_attr({1}),
        make_elements_attr({0}), rewriter.getContext());
    dotOp.dot_dimension_numbersAttr(dimension_numbers);
    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));
    return success();
  }
};

struct HloToLhloCustomCallOpConverter
    : public BaseOpConversion<mhlo::CustomCallOp> {
 public:
  using BaseOpConversion<mhlo::CustomCallOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp hloOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = hloOp.getOperation();
    SmallVector<Value, 2> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();

    auto lhloOp = rewriter.create<lmhlo::CustomCallOp>(
        op->getLoc(), llvm::None, buffer_args, op->getAttrs());
    // Setup AttrSizedOperandSegments attribute to indicate number of operands
    // for args and outputs.
    const int32_t segments[2] = {static_cast<int32_t>(operands.size()),
                                 static_cast<int32_t>(op->getNumResults())};
    lhloOp->setAttr(lhloOp.getOperandSegmentSizeAttr(),
                    rewriter.getI32VectorAttr(segments));

    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));
    return success();
  }
};

struct HloToLhloDotGeneralOpConverter
    : public BaseOpConversion<mhlo::DotGeneralOp> {
  using BaseOpConversion<mhlo::DotGeneralOp>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp dotGeneralOp, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    Operation* op = dotGeneralOp.getOperation();

    if (op->getResults().empty()) return failure();
    OpResult result = op->getResults()[0];
    RankedTensorType resultType = result.getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    // The third buffer argument will be filled with what used to be the return
    // type of the DotGeneral.
    if (operands.size() != 2) return failure();
    std::array<Value, 3> bufferArgs = {operands[0], operands[1], {}};

    if (resultType.hasStaticShape()) {
      bufferArgs[2] = InsertAlloc(op->getLoc(), result, &rewriter);
    } else {
      SmallVector<Value, 1> results_shape;
      auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
      if (failed(shape_type_op.reifyReturnTypeShapes(rewriter, operands,
                                                     results_shape)))
        return failure();

      bufferArgs[2] = InsertDynamicAlloc(op->getLoc(), result,
                                         results_shape.front(), &rewriter);
    }

    rewriter.create<lmhlo::DotOp>(op->getLoc(), llvm::None, bufferArgs,
                                  op->getAttrs());
    rewriter.replaceOp(op, bufferArgs[2]);
    return success();
  }
};

struct HloToLhloReduceOpConverter : public BaseOpConversion<mhlo::ReduceOp> {
 public:
  using BaseOpConversion<mhlo::ReduceOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    // TODO(b/137624192) Implement variadic reduce.
    if (op.getNumResults() != 1) return failure();
    if (!llvm::hasSingleElement(op.body())) {
      return op.emitOpError()
             << "tensor to buffer conversion expects a single block "
                "in the region containing the operation";
    }
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();
    auto new_op = rewriter.create<lmhlo::ReduceOp>(loc, llvm::None, buffer_args,
                                                   op->getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.body(), new_op.body(), new_op.body().end());

    // Convert the region signature to memref and add extra result.
    auto& entry_block = new_op.body().front();
    TypeConverter::SignatureConversion sig_conversion(
        entry_block.getNumArguments() + 1);
    for (auto arg : entry_block.getArguments()) {
      auto old_type = arg.getType().cast<TensorType>();
      auto new_type =
          MemRefType::get(old_type.getShape(), old_type.getElementType());
      sig_conversion.addInputs(arg.getArgNumber(), new_type);
    }
    auto return_op = cast<mhlo::ReturnOp>(entry_block.getTerminator());
    auto result_type = return_op.results().front().getType().cast<TensorType>();
    sig_conversion.addInputs({MemRefType::get(result_type.getShape(),
                                              result_type.getElementType())});
    rewriter.applySignatureConversion(&new_op.body(), sig_conversion);

    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));

    return success();
  }
};

// Legalize mhlo.return to a lmhlo.copy and lmhlo.terminator.
struct HloToLhloReturnOpConverter : public BaseOpConversion<mhlo::ReturnOp> {
 public:
  using BaseOpConversion<mhlo::ReturnOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto& entry_block = op->getParentRegion()->front();
    auto num_arguments = entry_block.getNumArguments();
    if (operands.size() > num_arguments) {
      return op.emitError(
          "The number of operands that need Copy operations is more "
          "than the number of target function arguments.");
    }

    // The index of the first output block argument.
    auto dest_arg_idx = num_arguments - operands.size();

    // Create a lmhlo.copy for each operand of mhlo.return.
    for (Value operand : operands) {
      rewriter.create<lmhlo::CopyOp>(loc, operand,
                                     entry_block.getArgument(dest_arg_idx));
      ++dest_arg_idx;
    }
    rewriter.replaceOpWithNewOp<lmhlo::TerminatorOp>(op);
    return success();
  }
};

// TODO(b/175789537) Remove this pattern.
class HloToLhloTensorStoreOpLegacyConverter
    : public BaseOpConversion<mlir::memref::TensorStoreOp> {
 public:
  using BaseOpConversion<mlir::memref::TensorStoreOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mlir::memref::TensorStoreOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<lmhlo::CopyOp>(op, llvm::None, operands.front(),
                                               operands.back());
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
//     %0 = tensor_load %arg1 : memref<2x2xf32>
//     %1 = tensor_load %arg2 : memref<2x2xf32>
//     %2 = "mhlo.add"(%0, %1) :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %3 = tensor_load %arg0 : memref<2x2xf32>
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
//   "lmhlo.fusion"() ( {
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
    registry.insert<lmhlo::LmhloDialect, memref::MemRefDialect,
                    shape::ShapeDialect>();
  }

 public:
  HloLegalizeToLhlo() = default;
  HloLegalizeToLhlo(const HloLegalizeToLhlo& o) {}

  void runOnOperation() override {
    auto& context = getContext();
    OwningRewritePatternList patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<lmhlo::LmhloDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    // Declare tensor_store illegal. tensor_load may be used to reify output
    // shape computation during dialect conversion and will be handled later.
    target.addIllegalOp<mlir::memref::TensorStoreOp>();
    // buffer_cast is illegal if it has uses.
    // TODO(b/175670649) Make buffer_cast illegal.
    target.addDynamicallyLegalOp<mlir::memref::BufferCastOp>(
        [](auto op) { return op->use_empty(); });

    BufferizeTypeConverter converter;
    auto isMemRefType = [](Type type) { return type.isa<BaseMemRefType>(); };
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType) &&
             std::all_of(op.result_type_begin(), op.result_type_end(),
                         isMemRefType);
    });
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType);
    });

    populateHLOToLHLOConversionPattern(&context, &converter, &patterns);
    populateFuncOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);

    populateShapeStructuralTypeConversionsAndLegality(converter, patterns,
                                                      target);

    // TODO(b/175789537) Remove this pattern.
    patterns.insert<HloToLhloTensorStoreOpLegacyConverter>(&context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

// Simply lowers all mhlo ops to their lmhlo counterparts.
void populateDynamicHLOToLHLOConversionPattern(
    MLIRContext* context, BufferizeTypeConverter* converter,
    OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<HloToLhloOpConverter<mhlo::DynamicBroadcastInDimOp>,
                   HloToLhloOpConverter<mhlo::DynamicGatherOp>,
                   HloToLhloOpConverter<mhlo::DynamicIotaOp>,
                   HloToLhloOpConverter<mhlo::DynamicPadOp>,
                   HloToLhloOpConverter<mhlo::DynamicReshapeOp>,
                   HloToLhloOpConverter<mhlo::RealDynamicSliceOp>
  >(*converter, context);
  // clang-format on
}

void populateHLOToLHLOConversionPattern(MLIRContext* context,
                                        BufferizeTypeConverter* converter,
                                        OwningRewritePatternList* patterns) {
  populateDynamicHLOToLHLOConversionPattern(context, converter, patterns);

  // clang-format off
  patterns->insert<
      HloToLhloCustomCallOpConverter,
      HloToLhloDotGeneralOpConverter,
      HloToLhloOpConverter<mhlo::AbsOp>,
      HloToLhloOpConverter<mhlo::AddOp>,
      HloToLhloOpConverter<mhlo::AndOp>,
      HloToLhloOpConverter<mhlo::Atan2Op>,
      HloToLhloOpConverter<mhlo::BroadcastInDimOp>,
      HloToLhloOpConverter<mhlo::CeilOp>,
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
      HloToLhloReduceOpConverter,
      HloToLhloReturnOpConverter
  >(*converter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToLhloPass() {
  return std::make_unique<HloLegalizeToLhlo>();
}

}  // namespace mhlo
}  // namespace mlir
