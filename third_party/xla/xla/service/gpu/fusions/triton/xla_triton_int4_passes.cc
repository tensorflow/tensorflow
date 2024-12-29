/* Copyright 2024 The OpenXLA Authors.

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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

using ::xla::llvm_ir::DumpToString;

namespace mt = ::mlir::triton;
namespace ma = ::mlir::arith;

#define GEN_PASS_DEF_LOADINT4REWRITEPASS
#include "xla/service/gpu/fusions/triton/xla_triton_passes.h.inc"

class I4ToI8Converter : public TypeConverter {
 public:
  static Type convertIntegerType(IntegerType type) {
    VLOG(10) << "I4ToI8Converter: converting IntegerType for "
             << DumpToString(type);
    if (type.getWidth() == 4) {
      auto new_type = IntegerType::get(type.getContext(), 8);
      VLOG(10) << "  ->  I4ToI8Converter: IntegerType converted to "
               << DumpToString(new_type);
      return new_type;
    }
    return type;
  }
  static Type convertRankedTensorType(RankedTensorType type) {
    VLOG(10) << "I4ToI8Converter: RankedTensorType for " << DumpToString(type);
    if (!type.getElementType().isInteger(4)) return type;

    auto shape = type.getShape();
    if (shape[0] == ShapedType::kDynamic)
      return type;  // Only handle static shapes for simplicity

    std::vector<int64_t> newShape(shape.begin(), shape.end());
    newShape[0] /= 2;
    auto new_type =
        RankedTensorType::get(newShape, IntegerType::get(type.getContext(), 8));
    VLOG(10) << "  ->  I4ToI8Converter: RankedTensorType converted to "
             << DumpToString(new_type);
    return new_type;
  }

  PointerType convertPointerType(PointerType ptr_type) {
    VLOG(10) << "I4ToI8Converter: converting PointerType for "
             << DumpToString(ptr_type);
    auto pointee_type = ptr_type.getPointeeType();
    auto new_pointee_type = convertType(pointee_type);
    auto new_ptr_type =
        PointerType::get(new_pointee_type, ptr_type.getAddressSpace());
    VLOG(10) << "  ->  I4ToI8Converter: converted PointerType to "
             << DumpToString(new_ptr_type);
    return new_ptr_type;
  }
  Type convertFunctionType(FunctionType func_type) {
    VLOG(10) << "I4ToI8Converter: converting FunctionType "
             << DumpToString(func_type);

    SmallVector<Type> inputs;
    if (failed(convertTypes(func_type.getInputs(), inputs))) return func_type;

    SmallVector<Type> results;
    if (failed(convertTypes(func_type.getResults(), results))) return func_type;

    auto new_func_type =
        FunctionType::get(func_type.getContext(), inputs, results);
    VLOG(10) << "  ->  I4ToI8Converter: converted FunctionType to "
             << DumpToString(new_func_type);
    return new_func_type;
  }

  I4ToI8Converter() {
    // Passthrough for other types.
    addConversion([](Type type) {
      VLOG(10) << "I4ToI8Converter: passthrough for " << DumpToString(type);
      return type;
    });

    // Convert i4 to i8
    addConversion(
        [this](IntegerType type) { return this->convertIntegerType(type); });

    // Convert tensor<AxBxi4> to tensor<A/2xBxi8>
    addConversion([this](RankedTensorType type) {
      return this->convertRankedTensorType(type);
    });

    // Convert !tt.ptr<tensor<AxBxi4>> to !tt.ptr<tensor<A/2xBxi8>>
    addConversion(
        [this](PointerType type) { return this->convertPointerType(type); });

    // Convert function type to function type
    addConversion(
        [this](FunctionType type) { return this->convertFunctionType(type); });
  }
};

class MakeTensorPtrOpConversionPattern
    : public OpConversionPattern<MakeTensorPtrOp> {
 public:
  using OpConversionPattern<MakeTensorPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MakeTensorPtrOp op,
      OpConversionPattern<MakeTensorPtrOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    // Convert the tensor type using the TypeConverter
    auto new_type = getTypeConverter()->convertType(op.getType());
    if (op.getType() == new_type) {
      return r.notifyMatchFailure(op, "no conversion needed");
    }

    auto loc = op.getLoc();
    Value c2 =
        r.create<ma::ConstantOp>(loc, r.getIntegerAttr(r.getI64Type(), 2));
    SmallVector<Value, 2> shape{adaptor.getShape().begin(),
                                adaptor.getShape().end()};
    // The packing dim is major and it should twice smaller.
    shape[0] = r.create<arith::DivSIOp>(loc, shape[0], c2);

    // The packing dim is major and the other stride should be half of the
    // original one.
    SmallVector<Value, 2> new_strides = adaptor.getStrides();
    new_strides[1] = r.create<arith::DivSIOp>(loc, new_strides[1], c2);

    r.replaceOpWithNewOp<MakeTensorPtrOp>(
        op, new_type, adaptor.getBase(), shape, new_strides,
        adaptor.getOffsets(), adaptor.getOrderAttr());

    return success();
  }
};

class AddPtrOpConversionPattern : public OpConversionPattern<AddPtrOp> {
 public:
  using OpConversionPattern<AddPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddPtrOp op, OpConversionPattern<AddPtrOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    // Convert the tensor type using the TypeConverter
    auto new_type = getTypeConverter()->convertType(op.getType());
    if (op.getType() == new_type) {
      return r.notifyMatchFailure(op, "no conversion needed");
    }

    // The increment for the next stripe of tiles along K dimension should be
    // twice smaller.
    auto ptr = adaptor.getOperands()[0];
    auto offset = adaptor.getOperands()[1];
    auto offset_type = offset.getType();
    Value c2 =
        r.create<ma::ConstantOp>(op.getLoc(), r.getIntegerAttr(offset_type, 2));
    auto new_offset =
        r.create<arith::DivSIOp>(op.getLoc(), offset_type, offset, c2);

    r.replaceOpWithNewOp<AddPtrOp>(op, new_type, ptr, new_offset);

    return success();
  }
};

template <typename OpType>
class OpTypeConversionPattern : public OpConversionPattern<OpType> {
 public:
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpType op, typename OpConversionPattern<OpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    VLOG(10) << "OpTypeConversionPattern: matching\n"
             << DumpToString(static_cast<Operation *>(op.getOperation()));
    // Convert the tensor type using the TypeConverter
    auto new_type =
        OpConversionPattern<OpType>::getTypeConverter()->convertType(
            op.getType());
    if (op.getType() == new_type) {
      VLOG(10) << "OpTypeConversionPattern: no conversion needed for "
               << DumpToString(op.getType());
      return r.notifyMatchFailure(op, "no conversion needed");
    }

    r.replaceOpWithNewOp<OpType>(op, new_type, adaptor.getOperands(),
                                 op->getAttrs());
    return success();
  }
};

// The pattern converts the ExtSIOp that converts i4 tensor to i8 tensor to the
// unpack sequence with ShLIOp, ShRSIOp, JoinOp, TransOp and ReshapeOp that does
// the same thing.
class ExtSIInt4ToInt8Pattern : public OpConversionPattern<ma::ExtSIOp> {
 public:
  using OpConversionPattern<ma::ExtSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ma::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto i4_tensor = cast<ShapedType>(op.getType());
    const auto &operand_type = cast<ShapedType>(op.getIn().getType());

    auto i4_type = r.getI4Type();
    auto i8_type = r.getI8Type();

    if (operand_type.getElementType() != i4_type) {
      return r.notifyMatchFailure(op, "not i4 operand");
    }

    // Make a new i8 tensor with the shape that is half of the int4 tensor.
    SmallVector<int64_t> result_shape(i4_tensor.getShape());
    result_shape[0] /= 2;
    auto i8_tensor = RankedTensorType::get(result_shape, i8_type);

    auto loc = op.getLoc();

    Value shift4_const =
        r.create<ma::ConstantOp>(loc, r.getIntegerAttr(i8_type, 4));
    Value shift4 = r.create<mt::SplatOp>(loc, i8_tensor, shift4_const);
    Value shifted_lo =
        r.create<ma::ShLIOp>(loc, i8_tensor, adaptor.getIn(), shift4);
    Value lo = r.create<ma::ShRSIOp>(loc, i8_tensor, shifted_lo, shift4);
    Value hi = r.create<ma::ShRSIOp>(loc, i8_tensor, adaptor.getIn(), shift4);
    Value hi_lo = r.create<mt::JoinOp>(loc, hi, lo);
    auto trans_attr = r.getDenseI32ArrayAttr({0, 2, 1});

    Value trans_hi_lo = r.create<mt::TransOp>(loc, hi_lo, trans_attr);

    r.replaceOpWithNewOp<mt::ReshapeOp>(op, i4_tensor, trans_hi_lo,
                                        /*allow_reorder=*/false);
    return success();
  }
};

struct PlainInt4ToPackedInt4RewritePass
    : public impl::LoadInt4RewritePassBase<PlainInt4ToPackedInt4RewritePass> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();

    ConversionTarget target(*ctx);

    VLOG(10) << "before TypeRewrite rewrite";
    {
      I4ToI8Converter converter;
      ConversionTarget target(*ctx);
      target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        if (auto func_op = dyn_cast<FuncOp>(op)) {
          VLOG(10) << "check funcOp: " << DumpToString(func_op);
          if (func_op.getFunctionType() !=
              converter.convertType(func_op.getFunctionType())) {
            VLOG(10) << "funcOp not legal: " << DumpToString(func_op);
            return false;
          }
        }
        bool is_legal = converter.isLegal(op);
        VLOG(10) << "is_legal: " << is_legal << " for " << DumpToString(op);
        return is_legal;
      });
      RewritePatternSet patterns(ctx);
      scf::populateSCFStructuralTypeConversions(converter, patterns);
      patterns.add<ExtSIInt4ToInt8Pattern>(ctx);
      patterns.add<OpTypeConversionPattern<LoadOp>>(converter, ctx);
      patterns.add<OpTypeConversionPattern<AdvanceOp>>(converter, ctx);
      patterns.add<AddPtrOpConversionPattern>(converter, ctx);
      patterns.add<MakeTensorPtrOpConversionPattern>(converter, ctx);
      populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                               converter);
      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        VLOG(10) << "failed to apply partial conversion";
        signalPassFailure();
      }
    }
    VLOG(10) << "after TypeRewrite Module: " << DumpToString(module);
  }
};

// The pass converts the types like tensor<AxBxi4> to tensor<A/2xBxi8> in the
// Triton dialect and replaces the ExtSIOp with the unpack sequence that accepts
// twice smaller i8 tensor and convert it to the twice bigger i8 tensor where
// every i4 element uses i8 space. At the end the module accepts the tt.ptr<i8>
// to the packed i4 tensor, and unpacks it to the i8 tensor for the further
// processing. It expects that the i4 tensor is packed along the major
// dimension.
std::unique_ptr<Pass> CreateInt4ToPackedInt4RewritePass() {
  return std::make_unique<PlainInt4ToPackedInt4RewritePass>();
}

}  // namespace mlir::triton::xla
