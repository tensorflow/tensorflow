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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

using bufferization::ToMemrefOp;
using bufferization::ToTensorOp;
using gml_st::LoopOp;
using linalg::FillOp;
using linalg::InitTensorOp;
using memref::SubViewOp;
using tensor::ExtractSliceOp;
using tensor::InsertSliceOp;
using vector::TransferReadOp;
using vector::TransferWriteOp;

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
static linalg::LinalgOp createLinalgOpOnBuffers(
    ConversionPatternRewriter &rewriter, linalg::LinalgOp linalgOp,
    ValueRange inputs, ValueRange outputs) {
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

// Bufferize LinalgOps in-place.
struct BufferizeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
  using OpInterfaceConversionPattern<
      linalg::LinalgOp>::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::LinalgOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<LoopOp>()) return failure();

    // GenericOpAdaptor below expects an `operand_segment_sizes` attribute.
    if (!op->hasAttr("operand_segment_sizes")) return failure();

    // TODO(b/199046880): Replace this with LinalgOp::Adaptor or equivalent.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

    createLinalgOpOnBuffers(rewriter, op, adaptor.inputs(), adaptor.outputs());
    rewriter.replaceOp(op, adaptor.outputs());
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
struct BufferizeLoopOp : public OpConversionPattern<LoopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LoopOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0) return failure();

    SmallVector<NamedAttribute> attr_list;
    for (auto &item : adaptor.getAttributes()) {
      attr_list.push_back(item);
    }
    auto newOp = rewriter.create<LoopOp>(op.getLoc(), mlir::TypeRange{},
                                         adaptor.getOperands(), attr_list);
    // Take the region from the old op and put it in the new op.
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    // Convert the type of the entry block of the LoopOp's body.
    if (failed(rewriter.convertRegionTypes(&newOp.getLoopBody(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }

    // Change the clone to use the updated operands. We could have cloned with
    // a BlockAndValueMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());

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
        readOp, readOp.getType(), adaptor.source(), adaptor.indices(),
        adaptor.permutation_mapAttr(), adaptor.padding(), adaptor.mask(),
        adaptor.in_bounds() ? adaptor.in_boundsAttr() : ArrayAttr());
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
        writeOp.getLoc(), adaptor.vector(), adaptor.source(), adaptor.indices(),
        adaptor.permutation_mapAttr(),
        adaptor.in_bounds() ? adaptor.in_boundsAttr() : ArrayAttr());
    rewriter.replaceOp(writeOp, adaptor.source());
    return success();
  }
};

}  // namespace

void populateTiledLoopBufferizePattern(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns) {
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

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
