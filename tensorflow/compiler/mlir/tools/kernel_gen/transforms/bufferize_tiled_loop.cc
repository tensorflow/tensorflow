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

// This file implements conversion of `linalg.tiled_loop` to buffer form.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

using bufferization::ToMemrefOp;
using bufferization::ToTensorOp;
using linalg::FillOp;
using linalg::InitTensorOp;
using linalg::TiledLoopOp;
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
    if (!op->getParentOfType<TiledLoopOp>()) return failure();

    rewriter.replaceOpWithNewOp<SubViewOp>(
        op, adaptor.source(), op.getMixedOffsets(), op.getMixedSizes(),
        op.getMixedStrides());
    return success();
  }
};

/// Convert `linalg.fill` on tensors to `linalg.fill` on buffers.
struct BufferizeFillOp : public OpConversionPattern<FillOp> {
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FillOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<TiledLoopOp>()) return failure();

    rewriter.create<FillOp>(op.getLoc(), adaptor.value(), adaptor.output());
    rewriter.replaceOp(op, adaptor.output());
    return success();
  }
};

/// Convert `linalg.init_tensor` of `memref.alloc`.
struct BufferizeInitTensorOp : public OpConversionPattern<InitTensorOp> {
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InitTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<TiledLoopOp>()) return failure();

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, getTypeConverter()->convertType(op.getType()).cast<MemRefType>(),
        adaptor.sizes());
    return success();
  }
};

bool IsBlockArgOfTiledLoop(Value value) {
  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return isa<TiledLoopOp>(block_arg.getOwner()->getParentOp());
  return false;
}

// Attempts to find an existing `memref.subview` of `destMemRef` in the tiled
// loop. The assumption is that in `linalg.tiled_loop` the tile of the output
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

    if (!op->getParentOfType<TiledLoopOp>()) return failure();

    Value subview = FindExistingSubview(destMemRef);
    if (!subview) {
      subview = rewriter.create<SubViewOp>(
          op.getLoc(), destMemRef, op.getMixedOffsets(), op.getMixedSizes(),
          op.getMixedStrides());
    }
    rewriter.create<linalg::CopyOp>(op.getLoc(), sourceMemRef, subview);
    rewriter.replaceOp(op, destMemRef);
    return success();
  }
};

// Bufferize LinalgOps in-place.
struct BufferizeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
  using OpInterfaceConversionPattern<
      linalg::LinalgOp>::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::LinalgOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    if (!op->getParentOfType<TiledLoopOp>()) return failure();

    // GenericOpAdaptor below expects an `operand_segment_sizes` attribute.
    if (!op->hasAttr("operand_segment_sizes")) return failure();

    // TODO(b/199046880): Replace this with LinalgOp::Adaptor or equivalent.
    linalg::GenericOpAdaptor adaptor(operands, op->getAttrDictionary());

    mlir::linalg::createLinalgOpOnBuffers(rewriter, op, adaptor.inputs(),
                                          adaptor.outputs());
    rewriter.replaceOp(op, adaptor.outputs());
    return success();
  }
};

// Convert `linalg.yield` terminator of `linalg.tiled_loop` to `linalg.yield`
// with no arguments.
struct BufferizeLinalgYieldOp : public OpConversionPattern<linalg::YieldOp> {
  using OpConversionPattern<linalg::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::YieldOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (!mlir::dyn_cast<TiledLoopOp>(op->getParentOp()) ||
        adaptor.getOperands().empty())
      return failure();

    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op);
    return success();
  }
};

// FuncOp-like bufferization pattern for `linalg.tiled_loop` that inserts
// `memref.tensor_load` ops for every memref block argument.
struct BufferizeTiledLoopOp : public OpConversionPattern<TiledLoopOp> {
  using OpConversionPattern<TiledLoopOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TiledLoopOp loop, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    if (loop.getNumResults() == 0) return failure();
    // The following code to set distribution_type is due to the following bug
    // causing distribution_types to return an ArrayAttr instead of an
    // Optional<ArrayAttr>. https://bugs.llvm.org/show_bug.cgi?id=51622
    llvm::Optional<ArrayAttr> distribution_types = adaptor.distribution_types();
    if (!distribution_types.getValue()) distribution_types = llvm::None;

    auto new_loop = rewriter.create<TiledLoopOp>(
        loop.getLoc(), adaptor.lowerBound(), adaptor.upperBound(),
        adaptor.step(), adaptor.inputs(), adaptor.outputs(),
        adaptor.iterator_types(), distribution_types);

    Location loc = loop.getLoc();
    BlockAndValueMapping bvm;
    bvm.map(loop.getInductionVars(), new_loop.getInductionVars());

    OpBuilder innerBuilder =
        OpBuilder::atBlockEnd(new_loop.getBody(), rewriter.getListener());

    // Map input tensors block arguments of the pre-bufferized loop to the
    // `tensor.tensor_load` results of the bufferized loop.
    SmallVector<Value, 2> inputs;
    for (auto en :
         llvm::zip(new_loop.getRegionInputArgs(), loop.getRegionInputArgs())) {
      Value newArg = std::get<0>(en);
      if (!newArg.getType().isa<ShapedType>()) {
        inputs.push_back(newArg);
        continue;
      }
      inputs.push_back(innerBuilder.create<ToTensorOp>(loc, std::get<0>(en)));
    }
    bvm.map(loop.getRegionInputArgs(), inputs);

    // Map output tensors block arguments of the pre-bufferized loop to the
    // `tensor.tensor_load` results of the bufferized loop.
    SmallVector<Value, 2> outputs;
    for (auto en : llvm::zip(new_loop.getRegionOutputArgs(),
                             loop.getRegionOutputArgs())) {
      Value newArg = std::get<0>(en);
      if (!newArg.getType().isa<ShapedType>()) {
        outputs.push_back(newArg);
        continue;
      }
      outputs.push_back(innerBuilder.create<ToTensorOp>(loc, std::get<0>(en)));
    }
    bvm.map(loop.getRegionOutputArgs(), outputs);

    // Clone the region.
    for (auto &op : loop.getBody()->getOperations())
      innerBuilder.clone(op, bvm);
    rewriter.replaceOp(loop, new_loop.outputs());
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
        adaptor.permutation_map(), adaptor.padding(), adaptor.mask(),
        adaptor.in_bounds() ? adaptor.in_bounds() : ArrayAttr());
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
        adaptor.permutation_map(),
        adaptor.in_bounds() ? adaptor.in_bounds() : ArrayAttr());
    rewriter.replaceOp(writeOp, adaptor.source());
    return success();
  }
};

}  // namespace

void populateTiledLoopBufferizePattern(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns) {
  // clang-format off
  patterns->insert<
    BufferizeExtractSliceOp,
    BufferizeFillOp,
    BufferizeInitTensorOp,
    BufferizeInsertSliceOp,
    BufferizeLinalgOp,
    BufferizeLinalgYieldOp,
    BufferizeTiledLoopOp,
    BufferizeVectorTransferReadOp,
    BufferizeVectorTransferWriteOp
  >(*converter, context);
  // clang-format on
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
