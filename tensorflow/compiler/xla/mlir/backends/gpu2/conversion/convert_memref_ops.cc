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
#include <functional>
#include <optional>
#include <string_view>

#include "third_party/iree/llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/Input/InputOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"

namespace xla {
namespace gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

using NotifyMatchFailure = std::function<LogicalResult(const char *)>;

// Reinterpret-casts block argument as a tensor type converted from the memref
// type via exporting/importing tensor to/from HAL buffer view.
//
// In XLA:GPU all block arguments are memrefs (tensors) if `i8` type that sliced
// into memrefs/tensors of correct type with `memref.view` operations. There is
// no corresponding operation in tensor/flow/hal dialect that allows type
// conversion, so we implement "tensor.view" through conversion to HAL buffers.
FailureOr<IREE::Input::TensorImportOp> reinterpretCastTensor(
    ImplicitLocOpBuilder &b, TypeConverter &converter, Value source,
    Value offset, MemRefType memref_ty, NotifyMatchFailure match_failure) {
  auto tensor_ty = converter.convertType(memref_ty).cast<RankedTensorType>();

  // Export tensor as !iree_input.buffer.
  auto buffer_ty = b.getType<IREE::Input::BufferType>();
  auto export_op = b.create<IREE::Input::TensorExportOp>(
      buffer_ty, source, /*source_dims=*/ValueRange());

  // Element type must be supported by the IREE HAL ABI.
  auto abi_ty = IREE::Input::getElementTypeValue(tensor_ty.getElementType());
  if (!abi_ty.has_value()) return match_failure("unsupported element type");

  // Create a buffer view with given offset, type and shape.
  auto buffer_view_size = std::max(1u, tensor_ty.getElementTypeBitWidth() / 8) *
                          tensor_ty.getNumElements();

  auto buffer_view_shape = llvm::to_vector(
      llvm::map_range(tensor_ty.getShape(), [&](int64_t dim) -> Value {
        return b.create<arith::ConstantIndexOp>(dim);
      }));

  auto view_op = b.create<IREE::Input::BufferViewCreateOp>(
      export_op, offset, b.create<arith::ConstantIndexOp>(buffer_view_size),
      *abi_ty,
      /*IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR*/ 1, buffer_view_shape);

  // Import it back as a properly typed tensor.
  return b.create<IREE::Input::TensorImportOp>(tensor_ty, view_op.getResult(),
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

    auto source = adaptor.getSource().cast<BlockArgument>();
    auto tensor_import =
        reinterpretCastTensor(b, *getTypeConverter(), source,
                              adaptor.getByteShift(), memref, match_failure);
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
      auto offset = b.create<arith::ConstantIndexOp>(0);
      auto tensor_import = reinterpretCastTensor(b, *getTypeConverter(), *arg,
                                                 offset, memref, match_failure);
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
