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
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

class I4ToI8Converter : public TypeConverter {
 public:
  Type convertIntegerType(IntegerType type) const {
    VLOG(2) << "I4ToI8Converter: converting IntegerType for "
            << DumpToString(type);
    if (type.getWidth() == 4) {
      auto new_type = IntegerType::get(type.getContext(), 8);
      VLOG(2) << "  ->  I4ToI8Converter: IntegerType converted to "
              << DumpToString(new_type);
      return new_type;
    }
    return type;
  }

  Type convertRankedTensorType(RankedTensorType type) const {
    VLOG(2) << "I4ToI8Converter: RankedTensorType for " << DumpToString(type);
    if (!type.getElementType().isInteger(4)) return type;

    auto shape = type.getShape();
    if (shape[0] == ShapedType::kDynamic)
      return type;  // Only handle static shapes for simplicity

    std::vector<int64_t> new_shape = shape;
    new_shape[packed_dimension()] /= 2;

    auto new_type = RankedTensorType::get(
        new_shape, IntegerType::get(type.getContext(), 8));
    VLOG(2) << "  ->  I4ToI8Converter: RankedTensorType converted to "
            << DumpToString(new_type);
    return new_type;
  }

  PointerType convertPointerType(PointerType ptr_type) const {
    VLOG(2) << "I4ToI8Converter: converting PointerType for "
            << DumpToString(ptr_type);
    auto pointee_type = ptr_type.getPointeeType();
    auto new_pointee_type = convertType(pointee_type);
    auto new_ptr_type =
        PointerType::get(new_pointee_type, ptr_type.getAddressSpace());
    VLOG(2) << "  ->  I4ToI8Converter: converted PointerType to "
            << DumpToString(new_ptr_type);
    return new_ptr_type;
  }

  Type convertFunctionType(FunctionType func_type) const {
    VLOG(2) << "I4ToI8Converter: converting FunctionType "
            << DumpToString(func_type);

    SmallVector<Type> inputs;
    if (failed(convertTypes(func_type.getInputs(), inputs))) return func_type;

    SmallVector<Type> results;
    if (failed(convertTypes(func_type.getResults(), results))) return func_type;

    auto new_func_type =
        FunctionType::get(func_type.getContext(), inputs, results);
    VLOG(2) << "  ->  I4ToI8Converter: converted FunctionType to "
            << DumpToString(new_func_type);
    return new_func_type;
  }

  explicit I4ToI8Converter(int packed_dimension)
      : packed_dimension_(packed_dimension) {
    // Passthrough for other types.
    addConversion([](Type type) {
      VLOG(2) << "I4ToI8Converter: passthrough for " << DumpToString(type);
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
  int packed_dimension() const { return packed_dimension_; }

 private:
  int packed_dimension_;
};

// Divides a value by an integer constant.
Value div(ConversionPatternRewriter &r, Value value, int64_t constant) {
  auto const_attr = r.getIntegerAttr(value.getType(), constant);
  auto const_op = r.template create<ma::ConstantOp>(value.getLoc(), const_attr);
  return r.template create<arith::DivSIOp>(value.getLoc(), value, const_op);
}

// Divides a value by an integer constant.
Value ceilDiv(ConversionPatternRewriter &r, Value value, int64_t constant) {
  auto const_attr = r.getIntegerAttr(value.getType(), constant);
  auto const_op = r.template create<ma::ConstantOp>(value.getLoc(), const_attr);
  return r.template create<arith::CeilDivSIOp>(value.getLoc(), value, const_op);
}

// Returns the integer value of a constant op.
// Returns std::nullopt if the value is not a constant op or the constant op
// does not have an integer value.
std::optional<int64_t> GetConstValue(Value value) {
  if (auto const_op = value.getDefiningOp<ma::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(const_op.getValue())) {
      return attr.getInt();
    }
  }
  return std::nullopt;
}

class MakeTensorPtrOpConversionPattern
    : public OpConversionPattern<MakeTensorPtrOp> {
 public:
  using OpConversionPattern<MakeTensorPtrOp>::OpConversionPattern;

  MakeTensorPtrOpConversionPattern(const I4ToI8Converter &converter,
                                   MLIRContext *context)
      : OpConversionPattern<MakeTensorPtrOp>(converter, context),
        converter_(converter) {}

  LogicalResult matchAndRewrite(
      MakeTensorPtrOp op,
      OpConversionPattern<MakeTensorPtrOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    // Convert the tensor type using the TypeConverter
    auto new_type = getTypeConverter()->convertType(op.getType());
    if (op.getType() == new_type) {
      return r.notifyMatchFailure(op, "no conversion needed");
    }

    SmallVector<Value, 2> shape = adaptor.getShape();
    int packed_dimension = converter_.packed_dimension();
    // The shape of the i8 tensor is half of the i4 tensor but at least 1.
    shape[packed_dimension] = ceilDiv(r, shape[packed_dimension], 2);

    // The stride of the i8 tensor is half of the i4 tensor but at least 1.
    SmallVector<Value, 2> new_strides = adaptor.getStrides();
    for (int i = 0; i < new_strides.size(); ++i) {
      new_strides[i] = ceilDiv(r, new_strides[i], 2);
    }

    r.replaceOpWithNewOp<MakeTensorPtrOp>(
        op, new_type, adaptor.getBase(), shape, new_strides,
        adaptor.getOffsets(), adaptor.getOrderAttr());

    return success();
  }

 private:
  const I4ToI8Converter &converter_;
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
    auto new_offset = div(r, offset, 2);

    r.replaceOpWithNewOp<AddPtrOp>(op, new_type, ptr, new_offset);

    return success();
  }
};

class AdvanceOpConversionPattern : public OpConversionPattern<AdvanceOp> {
 public:
  using OpConversionPattern<AdvanceOp>::OpConversionPattern;

  AdvanceOpConversionPattern(const I4ToI8Converter &converter,
                             MLIRContext *context)
      : OpConversionPattern<AdvanceOp>(converter, context),
        converter_(converter) {}
  LogicalResult matchAndRewrite(
      AdvanceOp op, typename OpConversionPattern<AdvanceOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    VLOG(2) << "AvanceOpConversionPattern: matching\n"
            << DumpToString(op.getOperation());
    // Convert the tensor type using the TypeConverter
    auto new_type = converter_.convertType(op.getType());
    if (op.getType() == new_type) {
      VLOG(2) << "AdvanceOpConversionPattern: no conversion needed for "
              << DumpToString(op.getType());
      return r.notifyMatchFailure(op, "no conversion needed");
    }
    SmallVector<Value, 2> offsets = adaptor.getOffsets();
    int packed_dimension = converter_.packed_dimension();
    offsets[packed_dimension] = div(r, offsets[packed_dimension], 2);
    auto new_op = r.replaceOpWithNewOp<AdvanceOp>(op, new_type,
                                                  adaptor.getPtr(), offsets);
    VLOG(2) << "AdvanceOpConversionPattern: replaced "
            << DumpToString(op.getOperation()) << " with "
            << DumpToString(new_op.getOperation());
    return success();
  }

 private:
  const I4ToI8Converter &converter_;
};

// The generic converter for the ops that requires only type conversion.
template <typename OpType>
class OpTypeConversionPattern : public OpConversionPattern<OpType> {
 public:
  using OpConversionPattern<OpType>::OpConversionPattern;

  OpTypeConversionPattern(const I4ToI8Converter &converter,
                          MLIRContext *context)
      : OpConversionPattern<OpType>(converter, context),
        converter_(converter) {}
  LogicalResult matchAndRewrite(
      OpType op, typename OpConversionPattern<OpType>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    VLOG(2) << "OpTypeConversionPattern: matching\n"
            << DumpToString(static_cast<Operation *>(op.getOperation()));
    // Convert the tensor type using the TypeConverter
    auto new_type = converter_.convertType(op.getType());
    if (op.getType() == new_type) {
      VLOG(2) << "OpTypeConversionPattern: no conversion needed for "
              << DumpToString(op.getType());
      return r.notifyMatchFailure(op, "no conversion needed");
    }

    r.replaceOpWithNewOp<OpType>(op, new_type, adaptor.getOperands(),
                                 op->getAttrs());
    return success();
  }

 private:
  const I4ToI8Converter &converter_;
};

// The pattern converts the ExtSIOp that converts i4 tensor to i8 tensor to an
// unpack sequence that uses ShLIOp, ShRSIOp, JoinOp, TransOp and ReshapeOp to
// do the same thing.
class ExtSIInt4ToInt8Pattern : public OpConversionPattern<ma::ExtSIOp> {
 public:
  ExtSIInt4ToInt8Pattern(const I4ToI8Converter &converter, MLIRContext *context)
      : OpConversionPattern<ma::ExtSIOp>(converter, context),
        converter_(converter) {}

  using OpConversionPattern<ma::ExtSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ma::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    VLOG(2) << "ExtSIInt4ToInt8Pattern: matching\n"
            << DumpToString(op.getOperation());
    auto input_type = cast<RankedTensorType>(op.getIn().getType());
    auto packed_type = converter_.convertType(input_type);
    if (input_type == packed_type) {
      return r.notifyMatchFailure(op, "no conversion needed");
    }

    // Make a new i8 tensor with the shape that is half of the int4 tensor.
    auto loc = op.getLoc();

    Value shift4_const =
        r.create<ma::ConstantOp>(loc, r.getIntegerAttr(r.getI8Type(), 4));
    Value shift4 = r.create<mt::SplatOp>(loc, packed_type, shift4_const);
    Value shifted_lo =
        r.create<ma::ShLIOp>(loc, packed_type, adaptor.getIn(), shift4);
    Value lo = r.create<ma::ShRSIOp>(loc, packed_type, shifted_lo, shift4);
    Value hi = r.create<ma::ShRSIOp>(loc, packed_type, adaptor.getIn(), shift4);
    Value hi_lo = r.create<mt::JoinOp>(loc, lo, hi);
    if (converter_.packed_dimension() + 1 != input_type.getRank()) {
      auto trans_attr = r.getDenseI32ArrayAttr({0, 2, 1});
      hi_lo = r.create<mt::TransOp>(loc, hi_lo, trans_attr);
    }
    auto unpacked_type = input_type.clone(r.getI8Type());
    r.replaceOpWithNewOp<mt::ReshapeOp>(op, unpacked_type, hi_lo,
                                        /*allow_reorder=*/false);
    return success();
  }

 private:
  const I4ToI8Converter &converter_;
};

// Traverses the operands of the op passing though the forOp and returns the
// list of ops that belong to the same argument.
std::vector<Operation *> TraverseUpwards(Operation *op) {
  std::vector<Operation *> result;
  while (op != nullptr) {
    VLOG(2) << "op: \n" << DumpToString(op);
    result.push_back(op);
    // Handle the argN of the forOp.
    if (auto arg = dyn_cast<BlockArgument>(op->getOperand(0))) {
      // Add the other users of the argN except the op itself. Usually the argN
      // is the arg of a ForOp, op is the LoadOp and the other user is the
      // AdvanceOp.
      for (auto user : arg.getUsers()) {
        if (user != op) {
          result.push_back(user);
        }
      }
      // Translate the argN of the forOp to the corresponding op that was passed
      // as the init arg.
      if (auto forOp =
              dyn_cast<scf::ForOp>(arg.getParentBlock()->getParentOp())) {
        auto arg_number = arg.getArgNumber();
        op = forOp.getInitArgs()[arg_number - 1].getDefiningOp();
        continue;
      }
    }

    op = op->getOperand(0).getDefiningOp();
  }
  return result;
}

// Finds all the ExtSIOp that require the type conversion.
std::vector<Operation *> FindInt4ExtSIOp(const ModuleOp &module) {
  // It does not matter which packed dimension idx we use here, because use the
  // converter to detect that the conversion is needed.
  I4ToI8Converter converter(/*packed_dimension=*/0);
  std::vector<Operation *> result;
  module->walk([&](Operation *op) {
    if (auto extSI = dyn_cast<arith::ExtSIOp>(op)) {
      VLOG(2) << "found ExtSI: " << DumpToString(op);
      auto input_type = extSI.getIn().getType();
      if (input_type != converter.convertType(input_type)) {
        result.push_back(op);
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// Finds the packed dimension from the MakeTensorPtrOp.
// The tensor is packed along the minor dimension. Minor dimension is the one
// that has a stride of 1 but a shape that is not 1. For a shape dimension of 1
// the stride can be any value.
int GetPackedDimension(MLIRContext *ctx, const std::vector<Operation *> &ops) {
  for (auto *op : ops) {
    auto make_tensor_ptr = dyn_cast<MakeTensorPtrOp>(op);
    if (!make_tensor_ptr) {
      continue;
    }
    // The order attribute is ignored in Triton, check for default order here.
    CHECK(absl::c_is_sorted(make_tensor_ptr.getOrder(), std::greater<int>()))
        << "Not default order: " << DumpToString(op);
    auto shape = make_tensor_ptr.getShape();
    auto strides = make_tensor_ptr.getStrides();
    for (auto dim : make_tensor_ptr.getOrder()) {
      if (GetConstValue(strides[dim]).value_or(1) == 1 &&
          GetConstValue(shape[dim]).value_or(0) != 1) {
        return dim;
      }
    }
  }
  LOG(FATAL) << "No MakeTensorPtrOp found";
}

struct PlainInt4ToPackedInt4RewritePass
    : public impl::LoadInt4RewritePassBase<PlainInt4ToPackedInt4RewritePass> {
  // The pass converts the types like tensor<AxBxi4> to tensor<A/2xBxi8> in the
  // Triton dialect and replaces the ExtSIOp with the unpack sequence that
  // accepts twice smaller i8 tensor and converts it to the twice bigger i8
  // tensor where every i4 element uses i8 space. At the end the module accepts
  // the tt.ptr<i8> to the packed i4 tensor, and unpacks it to the i8 tensor for
  // further processing. It gets the packed dimension from the MakeTensorPtrOp
  // attribute.
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();

    auto ext_ops = FindInt4ExtSIOp(module);
    int packed_dimension = 0;
    // TODO(b/383255324): Support the case when both sides of the dot are packed
    // differently.
    for (auto *op : ext_ops) {
      VLOG(2) << "ext_op: " << DumpToString(op);
      auto ops = TraverseUpwards(op);
      packed_dimension = GetPackedDimension(ctx, ops);
    }

    ConversionTarget target(*ctx);
    I4ToI8Converter converter(packed_dimension);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto func_op = dyn_cast<FuncOp>(op)) {
        VLOG(2) << "check funcOp: " << DumpToString(func_op);
        if (func_op.getFunctionType() !=
            converter.convertType(func_op.getFunctionType())) {
          VLOG(2) << "funcOp not legal: " << DumpToString(func_op);
          return false;
        }
      }
      bool is_legal = converter.isLegal(op);
      VLOG(2) << "is_legal: " << is_legal << " for " << DumpToString(op);
      return is_legal;
    });
    RewritePatternSet patterns(ctx);
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    patterns.add<ExtSIInt4ToInt8Pattern>(converter, ctx);
    patterns.add<OpTypeConversionPattern<LoadOp>>(converter, ctx);
    patterns.add<AdvanceOpConversionPattern>(converter, ctx);
    patterns.add<AddPtrOpConversionPattern>(converter, ctx);
    patterns.add<MakeTensorPtrOpConversionPattern>(converter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             converter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      VLOG(2) << "failed to apply partial conversion";
      signalPassFailure();
    }
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
