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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

using ::xla::llvm_ir::DumpToString;

namespace mt = ::mlir::triton;
namespace ma = ::mlir::arith;
namespace mtx = ::mlir::triton::xla;

#define GEN_PASS_DEF_LOADINT4REWRITEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

class I4ToI8Converter : public TypeConverter {
 public:
  Type convertIntegerType(IntegerType type) const {
    VLOG(5) << "I4ToI8Converter: converting IntegerType for "
            << DumpToString(type);
    if (type.getWidth() == 4) {
      auto new_type = IntegerType::get(type.getContext(), 8);
      VLOG(5) << "  ->  I4ToI8Converter: IntegerType converted to "
              << DumpToString(new_type);
      return new_type;
    }
    return type;
  }

  Type convertRankedTensorType(RankedTensorType type) const {
    VLOG(5) << "I4ToI8Converter: RankedTensorType for " << DumpToString(type);
    if (!type.getElementType().isInteger(4)) {
      return type;
    }

    auto shape = type.getShape();
    // Only handle static shapes for simplicity.
    if (mlir::ShapedType::isDynamicShape(shape)) {
      return type;
    }

    std::vector<int64_t> new_shape = shape;
    if (!shape.empty()) {
      new_shape[packed_dimension()] /= 2;
    }

    auto new_type = RankedTensorType::get(
        new_shape, IntegerType::get(type.getContext(), 8));
    VLOG(5) << "  ->  I4ToI8Converter: RankedTensorType converted to "
            << DumpToString(new_type);
    return new_type;
  }

  PointerType convertPointerType(PointerType ptr_type) const {
    VLOG(5) << "I4ToI8Converter: converting PointerType for "
            << DumpToString(ptr_type);
    auto pointee_type = ptr_type.getPointeeType();
    auto new_pointee_type = convertType(pointee_type);
    auto new_ptr_type =
        PointerType::get(new_pointee_type, ptr_type.getAddressSpace());
    VLOG(5) << "  ->  I4ToI8Converter: converted PointerType to "
            << DumpToString(new_ptr_type);
    return new_ptr_type;
  }

  Type convertFunctionType(FunctionType func_type) const {
    VLOG(5) << "I4ToI8Converter: converting FunctionType "
            << DumpToString(func_type);

    SmallVector<Type> inputs;
    if (failed(convertTypes(func_type.getInputs(), inputs))) {
      return func_type;
    }

    SmallVector<Type> results;
    if (failed(convertTypes(func_type.getResults(), results))) {
      return func_type;
    }

    auto new_func_type =
        FunctionType::get(func_type.getContext(), inputs, results);
    VLOG(5) << "  ->  I4ToI8Converter: converted FunctionType to "
            << DumpToString(new_func_type);
    return new_func_type;
  }

  explicit I4ToI8Converter(int packed_dimension)
      : packed_dimension_(packed_dimension) {
    // Passthrough for other types.
    addConversion([](Type type) {
      VLOG(5) << "I4ToI8Converter: passthrough for " << DumpToString(type);
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
  return r.template create<ma::DivSIOp>(value.getLoc(), value, const_op);
}

// Divides a value by an integer constant.
Value ceilDiv(ConversionPatternRewriter &r, Value value, int64_t constant) {
  auto const_attr = r.getIntegerAttr(value.getType(), constant);
  auto const_op = r.template create<ma::ConstantOp>(value.getLoc(), const_attr);
  return r.template create<ma::CeilDivSIOp>(value.getLoc(), value, const_op);
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

class TritonXlaExtractOpConversionPattern
    : public OpConversionPattern<mtx::ExtractOp> {
 public:
  using OpConversionPattern<mtx::ExtractOp>::OpConversionPattern;

  TritonXlaExtractOpConversionPattern(const I4ToI8Converter &converter,
                                      MLIRContext *context)
      : OpConversionPattern<mtx::ExtractOp>(converter, context),
        converter_(converter) {}

  LogicalResult matchAndRewrite(
      mtx::ExtractOp op, OpConversionPattern<mtx::ExtractOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const override {
    // Convert the tensor type using the TypeConverter
    auto new_result_type = mlir::cast<mlir::RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    ImplicitLocOpBuilder builder(op.getLoc(), r);
    // We can safely assume these are static because they were checked in
    // GetPackedDimension.
    SmallVector<int64_t, 2> tile_strides(adaptor.getStaticStrides());

    // The stride of the i8 tensor is half of the i4 tensor but at least 1.
    SmallVector<Value, 2> tile_strides_values;
    for (auto stride : tile_strides) {
      tile_strides_values.push_back(builder.create<ma::ConstantOp>(
          builder.getIndexAttr(ceil(stride / 2.0))));
    }

    // We update the offset of the packed dimension to be half of the original
    // offset.
    SmallVector<Value, 2> tile_offsets_values = op.getOffsetsAsValues(builder);
    tile_offsets_values[converter_.packed_dimension()] =
        div(r, tile_offsets_values[converter_.packed_dimension()], 2);

    SmallVector<int64_t> shape = llvm::to_vector(adaptor.getSrcShape());
    shape[converter_.packed_dimension()] =
        (shape[converter_.packed_dimension()] + 1) / 2;

    r.replaceOpWithNewOp<mtx::ExtractOp>(
        op, new_result_type, adaptor.getSrc(), tile_offsets_values,
        tile_strides_values, shape, adaptor.getSrcLayout());
    return success();
  }

 private:
  const I4ToI8Converter &converter_;
};

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
    VLOG(5) << "AvanceOpConversionPattern: matching\n"
            << DumpToString(op.getOperation());
    // Convert the tensor type using the TypeConverter
    auto new_type = converter_.convertType(op.getType());
    if (op.getType() == new_type) {
      VLOG(5) << "AdvanceOpConversionPattern: no conversion needed for "
              << DumpToString(op.getType());
      return r.notifyMatchFailure(op, "no conversion needed");
    }
    SmallVector<Value, 2> offsets = adaptor.getOffsets();
    int packed_dimension = converter_.packed_dimension();
    offsets[packed_dimension] = div(r, offsets[packed_dimension], 2);
    auto new_op = r.replaceOpWithNewOp<AdvanceOp>(op, new_type,
                                                  adaptor.getPtr(), offsets);
    VLOG(5) << "AdvanceOpConversionPattern: replaced "
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
    VLOG(5) << "OpTypeConversionPattern: matching\n"
            << DumpToString(static_cast<Operation *>(op.getOperation()));
    // Convert the tensor type using the TypeConverter
    auto new_type = converter_.convertType(op.getType());
    if (op.getType() == new_type) {
      VLOG(5) << "OpTypeConversionPattern: no conversion needed for "
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
  ExtSIInt4ToInt8Pattern(const I4ToI8Converter &converter, MLIRContext *context,
                         bool bf16x2_enabled)
      : OpConversionPattern<ma::ExtSIOp>(converter, context),
        converter_(converter),
        bf16x2_enabled_(bf16x2_enabled) {}

  using OpConversionPattern<ma::ExtSIOp>::OpConversionPattern;

  LogicalResult RewriteI4ToI8(
      ma::ExtSIOp ext_si_op,
      OpConversionPattern<ma::ExtSIOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const {
    auto loc = ext_si_op.getLoc();
    auto input_type = cast<RankedTensorType>(ext_si_op.getIn().getType());
    auto packed_type = converter_.convertType(input_type);
    VLOG(5) << "ExtSIInt4ToInt8Pattern: Regular int4 to int8 conversion";
    Value shift4_const =
        r.create<ma::ConstantOp>(loc, r.getIntegerAttr(r.getI8Type(), 4));
    Value shift4 = r.create<mt::SplatOp>(loc, packed_type, shift4_const);
    Value shifted_lo =
        r.create<ma::ShLIOp>(loc, packed_type, adaptor.getIn(), shift4);
    Value lo = r.create<ma::ShRSIOp>(loc, packed_type, shifted_lo, shift4);
    Value hi = r.create<ma::ShRSIOp>(loc, packed_type, adaptor.getIn(), shift4);
    Value hi_lo = r.create<mt::JoinOp>(loc, lo, hi);
    if (converter_.packed_dimension() + 1 != input_type.getRank()) {
      // Move the minor (joined) dimension to just after the packed dimension.
      SmallVector<int32_t> trans_order(input_type.getRank() + 1);
      absl::c_iota(trans_order, 0);
      std::rotate(trans_order.begin() + converter_.packed_dimension() + 1,
                  std::prev(trans_order.end()), trans_order.end());
      auto trans_attr = r.getDenseI32ArrayAttr(trans_order);
      hi_lo = r.create<mt::TransOp>(loc, hi_lo, trans_attr);
    }
    auto unpacked_type = input_type.clone(r.getI8Type());
    r.replaceOpWithNewOp<mt::ReshapeOp>(ext_si_op, unpacked_type, hi_lo,
                                        /*allow_reorder=*/false);

    return success();
  }

  LogicalResult RewriteI4ToBf16(
      ma::ExtSIOp ext_si_op, ma::SIToFPOp si_to_fp_op,
      OpConversionPattern<ma::ExtSIOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &r) const {
    VLOG(5) << "RewriteI4ToBf16: Using inline asm i4 to bf16 conversion";
    auto loc = ext_si_op.getLoc();
    auto input_type = cast<RankedTensorType>(ext_si_op.getIn().getType());
    auto packed_type = converter_.convertType(input_type);
    Type bf16_type = input_type.clone(r.getBF16Type());
    Type bf16_packed_type =
        dyn_cast<RankedTensorType>(packed_type).clone(r.getBF16Type());
    constexpr absl::string_view kInt4ToBF16Asm = R"(
      {
        .reg .s32 src_shifted;
        .reg .b32 bias;

        mov.b32 bias, 0x43084308;

        shr.s32 src_shifted, $4, 4;

        // vectorized interleaved ordering:
        prmt.b32 $0, $4, src_shifted, 0xF4F0;
        prmt.b32 $1, $4, src_shifted, 0xF6F2;
        prmt.b32 $2, $4, src_shifted, 0xF5F1;
        prmt.b32 $3, $4, src_shifted, 0xF7F3;

        lop3.b32 $0, $0, 0x000F000F, bias, 0x6a;
        lop3.b32 $1, $1, 0x000F000F, bias, 0x6a;
        lop3.b32 $2, $2, 0x000F000F, bias, 0x6a;
        lop3.b32 $3, $3, 0x000F000F, bias, 0x6a;

        sub.bf16x2 $0, $0, bias;
        sub.bf16x2 $1, $1, bias;
        sub.bf16x2 $2, $2, bias;
        sub.bf16x2 $3, $3, bias;
      }
    )";
    auto elementwise_op = r.create<ElementwiseInlineAsmOp>(
        loc, std::vector<Type>{bf16_packed_type, bf16_packed_type},
        kInt4ToBF16Asm, "=r,=r,=r,=r,r",
        /*pure=*/true, /*pack_result=*/4, adaptor.getOperands());
    Value lo = elementwise_op->getResult(0);
    Value hi = elementwise_op->getResult(1);
    Value hi_lo = r.create<mt::JoinOp>(loc, lo, hi);
    if (converter_.packed_dimension() + 1 != input_type.getRank()) {
      // Move the minor (joined) dimension to just after the packed dimension.
      SmallVector<int32_t> trans_order(input_type.getRank() + 1);
      absl::c_iota(trans_order, 0);
      std::rotate(trans_order.begin() + converter_.packed_dimension() + 1,
                  std::prev(trans_order.end()), trans_order.end());
      auto trans_attr = r.getDenseI32ArrayAttr(trans_order);
      hi_lo = r.create<mt::TransOp>(loc, hi_lo, trans_attr);
    }

    r.replaceOpWithNewOp<mt::ReshapeOp>(si_to_fp_op, bf16_type, hi_lo,
                                        /*allow_reorder=*/false);
    return success();
  }

  LogicalResult matchAndRewrite(ma::ExtSIOp ext_si_op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    VLOG(5) << "ExtSIInt4ToInt8Pattern: matching\n"
            << DumpToString(ext_si_op.getOperation());
    auto input_type = dyn_cast<RankedTensorType>(ext_si_op.getIn().getType());
    if (!input_type) {
      return r.notifyMatchFailure(ext_si_op, "not a ranked tensor");
    }
    auto packed_type = converter_.convertType(input_type);
    if (input_type == packed_type) {
      return r.notifyMatchFailure(ext_si_op, "no conversion needed");
    }

    if (bf16x2_enabled_) {
      // The RewriteI4ToBf16 inline asm uses bf16x2 instructions, so, if bf16x2
      // is not enabled, we use the RewriteI4ToI8 function to convert it.

      // Here we are looking for the sitofp op users that could be directly
      // converted from i4 to bf16. When we find sitofp op, we check if the
      // result type is bf16, if so, we use the RewriteI4ToBf16 function to
      // convert it. Otherwise, we use the RewriteI4ToI8 function to convert it.
      // If there are many such sitofp ops, we convert them separately with the
      // hope that the duplicate instructions will be merged.
      for (auto user : ext_si_op->getUsers()) {
        if (auto si_to_fp_op = dyn_cast<ma::SIToFPOp>(user)) {
          auto result_type = si_to_fp_op->getResultTypes()[0];
          auto tensor_type = dyn_cast<RankedTensorType>(result_type);
          if (!tensor_type || !tensor_type.getElementType().isBF16()) {
            VLOG(5) << "ExtSIInt4ToInt8Pattern: no conversion needed for "
                    << DumpToString(static_cast<Operation *>(si_to_fp_op));
            continue;
          }
          if (RewriteI4ToBf16(ext_si_op, si_to_fp_op, adaptor, r).failed()) {
            continue;
          }
        }
      }
    }
    return RewriteI4ToI8(ext_si_op, adaptor, r);
  }

 private:
  const I4ToI8Converter &converter_;
  const bool bf16x2_enabled_;
};

// Traverses the operands of the op passing though the forOp and returns the
// list of ops that belong to the same argument.
std::vector<Operation *> TraverseUpwards(Operation *op) {
  std::vector<Operation *> result;
  while (op != nullptr) {
    VLOG(5) << "op: \n" << DumpToString(op);
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
    if (auto extSI = dyn_cast<ma::ExtSIOp>(op)) {
      VLOG(5) << "found ExtSI: " << DumpToString(op);
      auto input_type = extSI.getIn().getType();
      if (input_type != converter.convertType(input_type)) {
        result.push_back(op);
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// Finds the packed dimension from MakeTensorPtrOp or mtx::ExtractOp.
// The packed dimension is the most minor dimension that has a unit stride and a
// shape that is > 1.
// TODO(b/393299275): MakeTensorPtrOp will not be emitted by the generic Triton
// emitter. Remove this once the legacy Triton emitter is deprecated.
absl::StatusOr<int> GetPackedDimension(MLIRContext *ctx,
                                       const std::vector<Operation *> &ops) {
  for (auto *op : ops) {
    auto make_tensor_ptr_op = dyn_cast<MakeTensorPtrOp>(op);
    auto extract_op = dyn_cast<mtx::ExtractOp>(op);
    if (!make_tensor_ptr_op && !extract_op) {
      continue;
    }

    if (make_tensor_ptr_op) {
      // The order attribute is ignored in Triton, check for default order here.
      CHECK(
          absl::c_is_sorted(make_tensor_ptr_op.getOrder(), std::greater<int>()))
          << "Not default order: " << DumpToString(op);
      auto shape = make_tensor_ptr_op.getShape();
      auto strides = make_tensor_ptr_op.getStrides();
      for (auto dim : make_tensor_ptr_op.getOrder()) {
        if (GetConstValue(strides[dim]).value_or(1) == 1 &&
            GetConstValue(shape[dim]).value_or(0) != 1) {
          return dim;
        }
      }
    }

    if (extract_op) {
      // Make sure the packed dimension is not dynamic and has a stride of 1.
      auto tile_strides = extract_op.getStaticStrides();
      auto tile_sizes = extract_op.getStaticSizes();
      auto original_shape = extract_op.getSrcShape();

      if (mlir::ShapedType::isDynamicShape(tile_strides) ||
          mlir::ShapedType::isDynamicShape(tile_sizes) ||
          mlir::ShapedType::isDynamicShape(original_shape)) {
        return absl::InvalidArgumentError(
            "dynamic shapes, tile strides, and tile sizes not supported");
      }

      for (auto dim : extract_op.getSrcLayout()) {
        if (tile_strides[dim] == 1 && tile_sizes[dim] > 1 &&
            original_shape[dim] > 1) {
          return dim;
        }
      }

      return absl::InvalidArgumentError("Failed to find a packed dimension.");
    }
  }
  std::string not_found_message =
      "No MakeTensorPtrOp or mlir::triton::xla::ExtractOp found";
  LOG(FATAL) << not_found_message;
  return absl::InvalidArgumentError(not_found_message);
}

LogicalResult SitofpInt4ToInt8Rewrite(ma::SIToFPOp op, PatternRewriter &r) {
  if (!getElementTypeOrSelf(op.getIn().getType()).isInteger(4)) {
    return r.notifyMatchFailure(op, "not an i4 argument");
  }
  Type type = r.getI8Type();
  if (auto tensor_type = dyn_cast<RankedTensorType>(op.getType())) {
    type = tensor_type.clone(type);
  }
  auto ext_si_op = r.create<ma::ExtSIOp>(op.getLoc(), type, op.getIn());
  r.replaceOpWithNewOp<ma::SIToFPOp>(op, op.getType(), ext_si_op);
  return success();
}

LogicalResult TruncfSitofpToSitofpRewrite(ma::TruncFOp trunc_op,
                                          PatternRewriter &rewriter) {
  auto sitofp_op = trunc_op.getIn().getDefiningOp<ma::SIToFPOp>();
  if (!sitofp_op) {
    return rewriter.notifyMatchFailure(trunc_op, "not preceded by sitofp");
  }
  rewriter.replaceOpWithNewOp<ma::SIToFPOp>(trunc_op, trunc_op.getType(),
                                            sitofp_op.getIn());
  return success();
}

bool elementTypeIs(Type type, Type element_type) {
  auto type_ranked = dyn_cast<RankedTensorType>(type);
  return type_ranked && type_ranked.getElementType() == element_type;
}

template <typename OpType>
bool opInputElementTypeIs(mlir::Value value, Type element_type) {
  auto op = dyn_cast<OpType>(value.getDefiningOp());
  if (!op) {
    return false;
  }
  return elementTypeIs(op.getIn().getType(), element_type);
}

// The pattern converts the Sitofp(i4): Fp32 to ExtFOp(Sitofp(i4): bf16): Fp32.
LogicalResult SitofpToExtFpSitofpRewrite(ma::SIToFPOp sitofp_op,
                                         PatternRewriter &rewriter) {
  auto input = sitofp_op.getIn();
  if (!opInputElementTypeIs<ma::ExtSIOp>(input, rewriter.getIntegerType(4))) {
    return rewriter.notifyMatchFailure(
        sitofp_op,
        "Ignore sitofp op that does not have ExtSIOp(tensor<i4>) as input");
  }
  if (!opInputElementTypeIs<ma::SIToFPOp>(sitofp_op,
                                          rewriter.getIntegerType(8))) {
    return rewriter.notifyMatchFailure(
        sitofp_op, "Ignore sitofp op that does not have i8 tensor as input");
  }
  Type type = sitofp_op.getType();
  if (!elementTypeIs(type, rewriter.getF32Type())) {
    return rewriter.notifyMatchFailure(
        sitofp_op, "Ignore sitofp op that does not have F32 type as output");
  }
  auto type_ranked = dyn_cast<RankedTensorType>(type);
  VLOG(5) << "SitofpToExtFpSitofpRewrite: SiToFp(i4): Fp32 -> "
             "ExtFOp(SiToFp(i4): bf16): Fp32: type:"
          << DumpToString(type_ranked);
  auto loc = sitofp_op.getLoc();
  auto sitofp_bf16_op = rewriter.create<ma::SIToFPOp>(
      loc, type_ranked.clone(rewriter.getBF16Type()), sitofp_op.getIn());
  rewriter.replaceOpWithNewOp<ma::ExtFOp>(sitofp_op, type, sitofp_bf16_op,
                                          ma::FastMathFlagsAttr{});
  return success();
}

class PlainInt4ToPackedInt4RewritePass
    : public impl::LoadInt4RewritePassBase<PlainInt4ToPackedInt4RewritePass> {
 public:
  using Base::Base;
  PlainInt4ToPackedInt4RewritePass(
      const PlainInt4ToPackedInt4RewritePass &other) = default;
  explicit PlainInt4ToPackedInt4RewritePass(
      const stream_executor::DeviceDescription &device_description)
      : bf16x2_enabled_(IsBF16x2Enabled(device_description)) {}

 private:
  static bool IsBF16x2Enabled(
      const stream_executor::DeviceDescription &device_description) {
    bool is_cuda =
        std::holds_alternative<stream_executor::CudaComputeCapability>(
            device_description.gpu_compute_capability());
    return is_cuda &&
           device_description.cuda_compute_capability().IsAtLeastHopper();
  }
  // The pass converts the types like tensor<AxBxi4> to tensor<AxB/2xi8>
  // (assuming B is the packed dimension) in the Triton dialect and replaces
  // the ExtSIOp with the unpack sequence that accepts twice smaller i8 tensor
  // and converts it to the twice bigger i8 tensor where every i4 element uses
  // i8 space. At the end the module accepts the tt.ptr<i8> to the packed i4
  // tensor, and unpacks it to the i8 tensor for further processing. It gets the
  // packed dimension from MakeTensorPtrOp or mtx::ExtractOp.
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();

    RewritePatternSet normalize_patterns(ctx);
    normalize_patterns.add(SitofpInt4ToInt8Rewrite);
    normalize_patterns.add(TruncfSitofpToSitofpRewrite);
    normalize_patterns.add(SitofpToExtFpSitofpRewrite);
    if (failed(applyPatternsGreedily(module, std::move(normalize_patterns)))) {
      VLOG(5) << "failed to apply patterns";
      signalPassFailure();
    }

    auto ext_ops = FindInt4ExtSIOp(module);
    int packed_dimension = 0;
    // TODO(b/383255324): Support the case when both sides of the dot are packed
    // differently.
    for (auto *op : ext_ops) {
      VLOG(5) << "ext_op: " << DumpToString(op);
      auto ops = TraverseUpwards(op);
      auto packed_dimension_result = GetPackedDimension(ctx, ops);
      if (!packed_dimension_result.ok()) {
        VLOG(5) << "failed to get packed dimension: "
                << packed_dimension_result.status();
        signalPassFailure();
      };
      packed_dimension = packed_dimension_result.value();
    }

    ConversionTarget target(*ctx);
    I4ToI8Converter converter(packed_dimension);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto func_op = dyn_cast<mlir::FunctionOpInterface>(op)) {
        VLOG(5) << "check funcOp: " << DumpToString(func_op);
        if (func_op.getFunctionType() !=
            converter.convertType(func_op.getFunctionType())) {
          VLOG(5) << "funcOp not legal: " << DumpToString(func_op);
          return false;
        }
      }
      bool is_legal = converter.isLegal(op);
      VLOG(5) << "is_legal: " << is_legal << " for " << DumpToString(op);
      return is_legal;
    });
    RewritePatternSet patterns(ctx);
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    patterns.add<ExtSIInt4ToInt8Pattern>(converter, ctx, bf16x2_enabled_);

    // TODO(b/393299275): LoadOp, AdvanceOp, AddPtrOp, and MakeTensorPtrOp will
    // not be emitted by the generic Triton emitter. Remove these once the
    // legacy Triton emitter is deprecated.
    patterns.add<OpTypeConversionPattern<LoadOp>>(converter, ctx);
    patterns.add<AdvanceOpConversionPattern>(converter, ctx);
    patterns.add<AddPtrOpConversionPattern>(converter, ctx);
    patterns.add<MakeTensorPtrOpConversionPattern>(converter, ctx);

    patterns.add<TritonXlaExtractOpConversionPattern>(converter, ctx);

    // TODO(b/393299275): Remove mt::FuncOp once the legacy Triton emitter is
    // deprecated.
    populateFunctionOpInterfaceTypeConversionPattern<mt::FuncOp>(patterns,
                                                                 converter);

    populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      VLOG(5) << "failed to apply partial conversion";
      signalPassFailure();
    }
  }
  // The default value is true, which means that bf16x2 instructions are used
  // when the device supports them. We need this for the mlir lit tests to pass.
  const bool bf16x2_enabled_ = true;
};

std::unique_ptr<mlir::Pass> CreateInt4ToPackedInt4RewritePass() {
  return std::make_unique<PlainInt4ToPackedInt4RewritePass>();
}

std::unique_ptr<Pass> CreateInt4ToPackedInt4RewritePass(
    const stream_executor::DeviceDescription &device_description) {
  return std::make_unique<PlainInt4ToPackedInt4RewritePass>(device_description);
}

}  // namespace mlir::triton::xla
