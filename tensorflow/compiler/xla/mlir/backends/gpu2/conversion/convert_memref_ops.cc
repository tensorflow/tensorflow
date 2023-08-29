/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/convert_memref_ops.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"

namespace xla {
namespace gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

using NotifyMatchFailure = std::function<LogicalResult(const char *)>;

// Reinterpret-casts block argument as a tensor type converted from the memref
// type via exporting/importing tensor to/from HAL buffer.
//
// In XLA:GPU all block arguments are memrefs (tensors) if `i8` type that sliced
// into memrefs/tensors of correct type with `memref.view` operations. There is
// no corresponding operation in tensor/flow/hal dialect that allows type
// conversion, so we implement "tensor.view" through conversion to HAL buffers.
FailureOr<IREE::Input::TensorImportOp> reinterpretCastTensor(
    ImplicitLocOpBuilder &b, const TypeConverter &converter,
    BlockArgument source, int64_t offset, MemRefType memref_ty,
    NotifyMatchFailure match_failure) {
  // Source type must be statically shaped `i8` tensor.
  auto source_ty = source.getType().cast<RankedTensorType>();
  auto source_length = source_ty.getNumElements();
  assert(source_ty.getElementType().isInteger(8) && "unsupported source type");

  // Target type must be statically shaped tensor.
  auto target_ty = converter.convertType(memref_ty).cast<RankedTensorType>();
  auto target_length = std::max(1u, target_ty.getElementTypeBitWidth() / 8) *
                       target_ty.getNumElements();

  // Always create buffer casting at the top of the block to prevent buffer
  // operations from splitting a command buffer.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(source.getOwner());

  // Export source tensor as !iree_input.buffer.
  auto buffer_ty = b.getType<IREE::Input::BufferType>();
  auto export_op = b.create<IREE::Input::TensorExportOp>(
      buffer_ty, source, /*source_dims=*/ValueRange());

  // If target length matches source length use exported buffer directly,
  // otherwise construct a buffer subspan.
  Value buffer = export_op.getResult();
  if (source_length != target_length) {
    buffer = b.create<IREE::Input::BufferSubspanOp>(
        buffer, b.create<arith::ConstantIndexOp>(offset),
        b.create<arith::ConstantIndexOp>(target_length));
  }

  // Import buffer back as a properly typed tensor.
  return b.create<IREE::Input::TensorImportOp>(target_ty, buffer,
                                               /*target_dims=*/ValueRange());
}

//===----------------------------------------------------------------------===//
// Converts memref.view op to a iree_input.tensor.{export,import}
//===----------------------------------------------------------------------===//

struct ConvertMemrefViewOp : public OpConversionPattern<memref::ViewOp> {
  ConvertMemrefViewOp(TypeConverter &converter, MLIRContext *ctx,
                      DeBufferization &state)
      : OpConversionPattern(converter, ctx), state(state) {}

  LogicalResult matchAndRewrite(
      memref::ViewOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    NotifyMatchFailure match_failure = [&](const char *msg) {
      return rewriter.notifyMatchFailure(op, msg);
    };

    auto memref = op.getType().dyn_cast<MemRefType>();
    if (!memref)
      return rewriter.notifyMatchFailure(op, "expected a memref result");

    IntegerAttr offset;
    if (!matchPattern(adaptor.getByteShift(), m_Constant(&offset)))
      return rewriter.notifyMatchFailure(op, "byte shift must be a constant");

    auto source = adaptor.getSource().cast<BlockArgument>();
    auto tensor_import = reinterpretCastTensor(
        b, *getTypeConverter(), source, offset.getInt(), memref, match_failure);
    if (failed(tensor_import)) return failure();
    rewriter.replaceOp(op, tensor_import->getResult());

    // Update de-bufferization state to track imported memref and a tensor.
    state.imported[source].push_back(op.getResult());
    state.remapped[op->getBlock()][op.getResult()] =
        cast<TypedValue<TensorType>>(tensor_import->getResult());

    return success();
  }

  DeBufferization &state;
};

//===----------------------------------------------------------------------===//
// Converts memref.reinterpret_cast op
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We have to keep track of buffer layout in a side data
// structure to pass it to custom calls that care about this information. All
// device kernels have layout hard coded into the kernel itself, and we pass
// only a pointer to device memory.

struct ConvertMemrefReinterpretCastOp
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::ReinterpretCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Converts memref.get_global op to memref.view op
//===----------------------------------------------------------------------===//

// Returns a function argument corresponding to the constant name.
//
// Example:
//
//   memref.global "private" constant @cst : memref<2x3xf32>
//   func @get_global(%arg0: memref<24xi8> {lmhlo.constant_name = "cst"})
//
// All memref.get_global operations will be replaced by constant arguments
// corresponding to the global constant.
std::optional<BlockArgument> getConstantArg(func::FuncOp func,
                                            std::string_view constant_name) {
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto cst = func.getArgAttrOfType<StringAttr>(i, "lmhlo.constant_name");
        cst && cst.getValue().equals(constant_name))
      return func.getArgument(i);
  }
  return std::nullopt;
}

struct ConvertMemrefGetGlobalOp
    : public OpConversionPattern<memref::GetGlobalOp> {
  ConvertMemrefGetGlobalOp(TypeConverter &converter, MLIRContext *ctx,
                           DeBufferization &state)
      : OpConversionPattern(converter, ctx), state(state) {}

  LogicalResult matchAndRewrite(
      memref::GetGlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    NotifyMatchFailure match_failure = [&](const char *msg) {
      return rewriter.notifyMatchFailure(op, msg);
    };

    // Find function argument corresponding to the global.
    auto func = op->getParentOfType<func::FuncOp>();
    auto arg = getConstantArg(func, op.getName());
    if (!arg.has_value())
      return rewriter.notifyMatchFailure(
          op, "can't find a constant argument corresponding to the global");

    MemRefType memref = op.getResult().getType();

    // For identity layouts we can replace all loads from a global with a view
    // operation from the corresponding argument.
    if (memref.getLayout().isIdentity()) {
      auto tensor_import =
          reinterpretCastTensor(b, *getTypeConverter(), *arg,
                                /*offset=*/0, memref, match_failure);
      if (failed(tensor_import)) return failure();

      rewriter.replaceOp(op, tensor_import->getResult());

      // Update de-bufferization state to track imported memref and a tensor.
      state.imported[*arg].push_back(op.getResult());
      state.remapped[op->getBlock()][op.getResult()] =
          cast<TypedValue<TensorType>>(tensor_import->getResult());

      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported memref layout");
  }

  DeBufferization &state;
};

//===----------------------------------------------------------------------===//
// Erase memref.global op
//===----------------------------------------------------------------------===//

// In XLA:GPU we can only have memref.global ops corresponding to constant
// arguments, and all loads from these globals should be replaced with casts
// from function arguments, so it's safe to erase them.
struct ConvertMemrefGlobalOp : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::GlobalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateMemrefConversionPatterns(RewritePatternSet &patterns,
                                      TypeConverter &converter,
                                      DeBufferization &state) {
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertMemrefViewOp>(converter, ctx, state);
  patterns.insert<ConvertMemrefReinterpretCastOp>(converter, ctx);
  patterns.insert<ConvertMemrefGetGlobalOp>(converter, ctx, state);
  patterns.insert<ConvertMemrefGlobalOp>(converter, ctx);
}

}  // namespace gpu
}  // namespace xla
