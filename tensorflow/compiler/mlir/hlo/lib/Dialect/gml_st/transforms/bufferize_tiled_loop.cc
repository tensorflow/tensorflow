/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements conversion of `gml_st.loop` to buffer form.

#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/rewriters.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

using bufferization::ToMemrefOp;
using bufferization::ToTensorOp;
using gml_st::LoopOp;
using linalg::InitTensorOp;
using memref::SubViewOp;
using tensor::ExtractSliceOp;
using tensor::InsertSliceOp;
using vector::TransferReadOp;
using vector::TransferWriteOp;

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

// TODO(pifon): Remove as soon as https://reviews.llvm.org/D93126 is landed.
class CustomBufferizeTypeConverter
    : public bufferization::BufferizeTypeConverter {
 public:
  CustomBufferizeTypeConverter() {
    // Keep all types unchanged.
    addConversion([](Type type) { return type; });
    // Convert RankedTensorType to MemRefType.
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    // Convert UnrankedTensorType to UnrankedMemRefType.
    addConversion([](UnrankedTensorType type) -> Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    addArgumentMaterialization(materializeToTensor);
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memref_type = inputs[0].getType().dyn_cast<MemRefType>()) {
        assert(!memref_type.getLayout().isIdentity());
        return inputs[0];
      }
      assert(inputs[0].getType().isa<TensorType>());
      return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
    });
  }
};

/// Convert `tensor.extract_slice` to `memref.subview` in-place.
struct BufferizeExtractSliceOp : public OpConversionPattern<ExtractSliceOp> {
  using OpConversionPattern<ExtractSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<LoopOp>()) return failure();

    rewriter.replaceOpWithNewOp<SubViewOp>(
        op, adaptor.source(), op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    return success();
  }
};

/// Convert `linalg.init_tensor` of `memref.alloc`.
struct BufferizeInitTensorOp : public OpConversionPattern<InitTensorOp> {
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InitTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<LoopOp>()) return failure();

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

bool IsBlockArgOfTiledLoop(Value value) {
  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return isa<LoopOp>(block_arg.getOwner()->getParentOp());
  return false;
}

// Attempts to find an existing `memref.subview` of `destMemRef` in the tiled
// loop. The assumption is that in `gml_st.loop` the tile of the output
// tensor that we read and the tile that we write to are the same.
Value FindExistingSubview(Value destMemRef) {
  if (auto to_memref = destMemRef.getDefiningOp<ToMemrefOp>()) {
    if (auto to_tensor = to_memref.tensor().getDefiningOp<ToTensorOp>()) {
      if (!IsBlockArgOfTiledLoop(to_tensor.memref())) return Value{};
      // Scan through users of the block argument to find `subview` op.
      for (Operation *tensor_user : to_memref.tensor().getUsers()) {
        if (auto another_cast = mlir::dyn_cast<ToMemrefOp>(tensor_user)) {
          for (Operation *memref_user : another_cast.memref().getUsers()) {
            if (auto subview = mlir::dyn_cast<SubViewOp>(memref_user)) {
              if (subview.source() == destMemRef) return subview;
            }
          }
        }
      }
    }
  }
  return Value{};
}

/// Convert `tensor.insert_slice` to `memref.subview` in-place.
struct BufferizeInsertSliceOp : public OpConversionPattern<InsertSliceOp> {
 public:
  using OpConversionPattern<InsertSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InsertSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Value sourceMemRef = adaptor.source();
    assert(sourceMemRef.getType().isa<MemRefType>());

    Value destMemRef = adaptor.dest();
    assert(destMemRef.getType().isa<MemRefType>());

    if (!op->getParentOfType<LoopOp>()) return failure();

    Value subview = FindExistingSubview(destMemRef);
    if (!subview) {
      subview = rewriter.create<SubViewOp>(
          op.getLoc(), destMemRef, op.getMixedOffsets(), op.getMixedSizes(),
          op.getMixedStrides());
    }
    rewriter.create<memref::CopyOp>(op.getLoc(), sourceMemRef, subview);
    rewriter.replaceOp(op, destMemRef);
    return success();
  }
};

/// Create linalg op on buffers given the original tensor-based operation and
/// the buffers for the outputs.
linalg::LinalgOp createLinalgOpOnBuffers(ConversionPatternRewriter &rewriter,
                                         linalg::LinalgOp linalgOp,
                                         ValueRange inputs,
                                         ValueRange outputs) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto *newOp = linalgOp.cloneWithoutRegions(rewriter, linalgOp.getLoc(),
                                             /*resultTypes=*/ArrayRef<Type>{},
                                             newOperands);
  for (auto regions : llvm::zip(linalgOp->getRegions(), newOp->getRegions())) {
    auto &oldRegion = std::get<0>(regions);
    auto &newRegion = std::get<1>(regions);
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.begin());
  }
  return newOp;
}

/// Get a variadic operand segment.
ValueRange GetVariadicOperands(DenseIntElementsAttr size_attr,
                               ValueRange operands, unsigned index) {
  const uint32_t *size_it = &*size_attr.value_begin<uint32_t>();
  if (size_attr.isSplat()) return operands.slice(*size_it * index, *size_it);

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i) start += size_it[i];
  return operands.slice(start, size_it[index]);
}

// Bufferize LinalgOps in-place.
struct BufferizeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
  using OpInterfaceConversionPattern<
      linalg::LinalgOp>::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::LinalgOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<LoopOp>()) return failure();

    // An op with two variadic operand groups expects a segment size attribute.
    auto operand_segments =
        op->getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
    if (!operand_segments) return failure();

    const auto getOperands = [&](unsigned index) {
      return GetVariadicOperands(operand_segments, operands, index);
    };
    createLinalgOpOnBuffers(rewriter, op, getOperands(0), getOperands(1));
    rewriter.replaceOp(op, getOperands(1));
    return success();
  }
};

// Convert `gml_st.yield` terminator of `gml_st.loop` to `gml_st.yield` with no
// arguments.
struct BufferizeLinalgYieldOp : public OpConversionPattern<gml_st::YieldOp> {
  using OpConversionPattern<gml_st::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      gml_st::YieldOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!mlir::dyn_cast<LoopOp>(op->getParentOp()) ||
        adaptor.getOperands().empty())
      return failure();

    rewriter.replaceOpWithNewOp<gml_st::YieldOp>(op);
    return success();
  }
};

// FuncOp-like bufferization pattern for `gml_st.loop` that inserts
// `memref.tensor_load` ops for every memref block argument.
//
// TODO(b/230082413): This code has to go away if we migrate to one-shot
// bufferization.
struct BufferizeLoopOp : public OpConversionPattern<LoopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LoopOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0) return failure();

    // Allocate new buffers for results if it is used by multiple uses.
    SmallVector<Value, 4> operands = adaptor.getOperands();
    for (auto &en : llvm::enumerate(op.outputs())) {
      Value output = en.value();

      auto to_tensor = output.getDefiningOp<bufferization::ToTensorOp>();
      if (!to_tensor || to_tensor->hasOneUse()) continue;

      auto alloc = to_tensor.memref().getDefiningOp<memref::AllocOp>();
      if (!alloc) continue;

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(op);
      auto new_alloc = rewriter.clone(*alloc.getOperation());
      operands[op.getNumControlOperands() + op.getNumInputs() + en.index()] =
          new_alloc->getResult(0);
    }

    SmallVector<NamedAttribute> attr_list;
    for (auto &item : adaptor.getAttributes()) {
      attr_list.push_back(item);
    }
    auto newOp = rewriter.create<LoopOp>(op.getLoc(), mlir::TypeRange{},
                                         operands, attr_list);
    // Take the region from the old op and put it in the new op.
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    // Convert the type of the entry block of the LoopOp's body.
    if (failed(rewriter.convertRegionTypes(&newOp.getLoopBody(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }

    rewriter.replaceOp(op, newOp.outputs());
    return success();
  }
};

// TODO(b/199045477): The pattern for vector.transfer_read/write have to be
// moved out of Linalg bufferization to a VectorOps bufferization pass.
struct BufferizeVectorTransferReadOp
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp readOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (readOp.getShapedType().isa<MemRefType>()) return failure();
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getType(), adaptor.getSource(), adaptor.getIndices(),
        adaptor.getPermutationMapAttr(), adaptor.getPadding(),
        adaptor.getMask(),
        adaptor.getInBounds() ? adaptor.getInBoundsAttr() : ArrayAttr());
    return success();
  }
};

struct BufferizeVectorTransferWriteOp
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp writeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (writeOp.getShapedType().isa<MemRefType>()) return failure();
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), adaptor.getVector(), adaptor.getSource(),
        adaptor.getIndices(), adaptor.getPermutationMapAttr(),
        adaptor.getInBounds() ? adaptor.getInBoundsAttr() : ArrayAttr());
    rewriter.replaceOp(writeOp, adaptor.getSource());
    return success();
  }
};

}  // namespace

namespace gml_st {
struct TiledLoopBufferizePass
    : public TiledLoopBufferizePassBase<TiledLoopBufferizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface. This could be switched to
    // One-Shot Bufferize in the future.
    mlir::RewritePatternSet patterns(&getContext());
    mlir::bufferization::BufferizationOptions options =
        mlir::bufferization::getPartialBufferizationOptions();
    // TODO(springerm): Add dialects to this filter as more and more dialects
    // will be migrated to BufferizableOpInterface-based bufferization.
    options.opFilter.allowDialect<shape::ShapeDialect>();
    if (failed(mlir::bufferization::bufferizeOp(getOperation(), options))) {
      signalPassFailure();
      return;
    }

    // Bufferize the remaining IR with dialect conversion. This will disappear
    // eventually once all bufferization is done via BufferizableOpInterface.
    if (failed(runDialectConversionBasedBufferization())) signalPassFailure();
  }

 private:
  LogicalResult runDialectConversionBasedBufferization() {
    mlir::RewritePatternSet patterns(&getContext());
    auto &context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<
        mlir::arith::ArithmeticDialect,
        mlir::bufferization::BufferizationDialect,
        mlir::complex::ComplexDialect, mlir::lmhlo::LmhloDialect,
        mlir::AffineDialect, mlir::vector::VectorDialect,
        mlir::memref::MemRefDialect, mlir::func::FuncDialect,
        mlir::tensor::TensorDialect, mlir::math::MathDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalOp<tensor::ExtractSliceOp, tensor::InsertSliceOp>();

    CustomBufferizeTypeConverter converter;
    mlir::mhlo::RemoveSignTypeConverter remove_sign_converter;

    // Configure bufferize pattern.
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    mlir::bufferization::populateBufferizeMaterializationLegality(target);
    populateTiledLoopBufferizePattern(&getContext(), &converter, &patterns);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);
    // Configure legality.
    auto isLegalOp = [&](Operation *op) { return converter.isLegal(op); };
    target.addDynamicallyLegalDialect<mlir::linalg::LinalgDialect>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp, gml_st::LoopOp,
                                 gml_st::YieldOp, mlir::LLVM::InlineAsmOp,
                                 mlir::vector::TransferWriteOp,
                                 mlir::vector::TransferReadOp>(isLegalOp);

    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }
};

void populateTiledLoopBufferizePattern(
    mlir::MLIRContext *context,
    mlir::bufferization::BufferizeTypeConverter *converter,
    mlir::RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    BufferizeExtractSliceOp,
    BufferizeInitTensorOp,
    BufferizeInsertSliceOp,
    BufferizeLinalgOp,
    BufferizeLinalgYieldOp,
    BufferizeLoopOp,
    BufferizeVectorTransferReadOp,
    BufferizeVectorTransferWriteOp
  >(*converter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTiledLoopBufferizePass() {
  return std::make_unique<TiledLoopBufferizePass>();
}

}  // namespace gml_st
}  // namespace mlir
