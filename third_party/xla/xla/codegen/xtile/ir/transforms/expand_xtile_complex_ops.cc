/* Copyright 2026 The OpenXLA Authors.

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

#include <complex>
#include <cstdint>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla {
namespace xtile {

#define GEN_PASS_DEF_EXPANDXTILECOMPLEXOPSPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

namespace {

using mlir::ArrayRef;
using mlir::ComplexType;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::Type;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::ConstantIndexOp;
using mlir::stablehlo::AddOp;
using mlir::stablehlo::SubtractOp;
using xtile::EntryFuncOp;
using xtile::ExtractTileOp;
using xtile::InsertTileOp;

namespace shlo = ::mlir::stablehlo;
namespace ma = ::mlir::arith;

// Appends 2 to the shape of the given shaped type.
SmallVector<int64_t> ExpandShape(llvm::ArrayRef<int64_t> shape) {
  SmallVector<int64_t> new_shape = llvm::to_vector(shape);
  new_shape.push_back(2);
  return new_shape;
}

Type GetExpandedType(Type type) {
  auto shaped_type = mlir::dyn_cast<ShapedType>(type);
  if (!shaped_type) {
    return type;
  }
  auto complex_type = mlir::dyn_cast<ComplexType>(shaped_type.getElementType());
  if (!complex_type) {
    return type;
  }
  if (auto tensor_type = mlir::dyn_cast<RankedTensorType>(type)) {
    return RankedTensorType::get(ExpandShape(shaped_type.getShape()),
                                 complex_type.getElementType());
  }
  if (auto memref_type = mlir::dyn_cast<MemRefType>(type)) {
    return MemRefType::get(ExpandShape(shaped_type.getShape()),
                           complex_type.getElementType(),
                           /*layout=*/mlir::MemRefLayoutAttrInterface(),
                           memref_type.getMemorySpace());
  }
  return type;
}

bool HasComplex(Type type) {
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(type)) {
    return mlir::isa<ComplexType>(shaped_type.getElementType());
  }
  return mlir::isa<ComplexType>(type);
}

Value UnwrapCast(Value value, PatternRewriter& rewriter) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    return cast.getOperand(0);
  }
  if (!HasComplex(value.getType())) {
    return value;
  }
  return UnrealizedConversionCastOp::create(
             rewriter, value.getLoc(), GetExpandedType(value.getType()), value)
      .getResult(0);
}

Value ExtractFloatingPointTensor(Value complex_val, Location loc,
                                 PatternRewriter& rewriter,
                                 bool is_real = true) {
  auto complex_tensor_type =
      mlir::cast<RankedTensorType>(complex_val.getType());
  auto shape = complex_tensor_type.getShape();
  int64_t rank = complex_tensor_type.getRank();
  auto elem_type = complex_tensor_type.getElementType();

  SmallVector<int64_t> slice_shape(shape.begin(), shape.end() - 1);
  slice_shape.push_back(1);
  auto slice_type = RankedTensorType::get(slice_shape, elem_type);

  SmallVector<int64_t> orig_shape(shape.begin(), shape.end() - 1);
  auto orig_type = RankedTensorType::get(orig_shape, elem_type);

  SmallVector<int64_t> start_indices(rank, 0);
  start_indices.back() = is_real ? 0 : 1;
  SmallVector<int64_t> limit_indices(shape.begin(), shape.end());
  limit_indices.back() = is_real ? 1 : 2;
  SmallVector<int64_t> strides(rank, 1);

  auto slice = shlo::SliceOp::create(rewriter, loc, slice_type, complex_val,
                                     start_indices, limit_indices, strides);
  return shlo::ReshapeOp::create(rewriter, loc, orig_type, slice.getResult());
}

Value ConcatRealAndImag(Value real, Value imag, Location loc,
                        PatternRewriter& rewriter) {
  auto real_type = mlir::cast<RankedTensorType>(real.getType());
  ArrayRef<int64_t> shape = real_type.getShape();
  Type elem_type = real_type.getElementType();

  SmallVector<int64_t> slice_shape(shape.begin(), shape.end());
  slice_shape.push_back(1);
  auto slice_type = RankedTensorType::get(slice_shape, elem_type);

  auto expanded_type = RankedTensorType::get(ExpandShape(shape), elem_type);

  auto real_reshaped = shlo::ReshapeOp::create(rewriter, loc, slice_type, real);
  auto imag_reshaped = shlo::ReshapeOp::create(rewriter, loc, slice_type, imag);

  return shlo::ConcatenateOp::create(
      rewriter, loc, expanded_type,
      ValueRange{real_reshaped.getResult(), imag_reshaped.getResult()},
      /*dimension=*/real_type.getRank());
}

struct RewriteFunctionSignatures : OpRewritePattern<EntryFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(EntryFuncOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumResults() != 0) {
      return rewriter.notifyMatchFailure(
          op, "function with non-zero results not supported");
    }
    auto input_types = op.getFunctionType().getInputs();
    if (llvm::none_of(input_types, HasComplex)) {
      return rewriter.notifyMatchFailure(op, "nothing to expand");
    }
    mlir::Location loc = op.getLoc();
    if (op.getBody().empty()) {
      return rewriter.notifyMatchFailure(op, "empty function body");
    }
    mlir::Block* entry_block = &op.getBody().front();

    // Cast all function arguments to the original type.
    SmallVector<Type> new_operand_types(input_types);
    rewriter.setInsertionPointToStart(entry_block);
    for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
      if (!HasComplex(operand_type)) {
        continue;
      }
      mlir::BlockArgument func_argument = op.getArgument(index);
      auto cast_to_orig_type = UnrealizedConversionCastOp::create(
          rewriter, loc, operand_type, func_argument);
      func_argument.replaceAllUsesExcept(cast_to_orig_type.getResult(0),
                                         cast_to_orig_type);
      operand_type = GetExpandedType(operand_type);
    }
    // Replace the function arguments with the new types.
    for (auto [arg, arg_type] :
         llvm::zip(entry_block->getArguments(), new_operand_types)) {
      arg.setType(arg_type);
    }
    // Update function signature.
    op.setType(rewriter.getFunctionType(new_operand_types, {}));
    return mlir::success();
  }
};

struct RewriteExtractTileOp : OpRewritePattern<ExtractTileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractTileOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }

    mlir::Location loc = op.getLoc();
    auto new_tensor_type = GetExpandedType(tensor_type);
    Value source = UnwrapCast(op.getSource(), rewriter);

    SmallVector<Value> offsets(op.getOffsets().begin(), op.getOffsets().end());
    offsets.push_back(ConstantIndexOp::create(rewriter, loc, 0));

    SmallVector<int64_t> tile_strides = llvm::to_vector(op.getStrides());
    tile_strides.push_back(1);

    auto new_extract =
        ExtractTileOp::create(rewriter, loc, new_tensor_type, source, offsets,
                              ExpandShape(op.getFullTileShape()), tile_strides);

    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, loc, tensor_type, new_extract.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteInsertTileOp : OpRewritePattern<InsertTileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertTileOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type =
        mlir::dyn_cast<RankedTensorType>(op.getSource().getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }

    auto loc = op.getLoc();
    Value source = UnwrapCast(op.getSource(), rewriter);
    Value destination = UnwrapCast(op.getDestination(), rewriter);

    SmallVector<Value> offsets(op.getOffsets().begin(), op.getOffsets().end());
    offsets.push_back(ConstantIndexOp::create(rewriter, loc, 0));

    SmallVector<int64_t> tile_strides = llvm::to_vector(op.getStrides());
    tile_strides.push_back(1);

    InsertTileOp::create(rewriter, loc, source, destination, offsets,
                         ExpandShape(op.getFullTileShape()), tile_strides);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RewriteAddOp : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }
    auto loc = op.getLoc();
    auto new_type = GetExpandedType(tensor_type);
    Value lhs = UnwrapCast(op.getLhs(), rewriter);
    Value rhs = UnwrapCast(op.getRhs(), rewriter);
    auto new_op = AddOp::create(rewriter, loc, new_type, lhs, rhs);
    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, loc, tensor_type, new_op.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteSubtractOp : OpRewritePattern<SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }

    auto loc = op.getLoc();
    auto new_type = GetExpandedType(tensor_type);
    Value lhs = UnwrapCast(op.getLhs(), rewriter);
    Value rhs = UnwrapCast(op.getRhs(), rewriter);

    auto new_op = SubtractOp::create(rewriter, loc, new_type, lhs, rhs);
    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, loc, tensor_type, new_op.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteRealOp : OpRewritePattern<shlo::RealOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::RealOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type =
        mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }
    Value complex_val = UnwrapCast(op.getOperand(), rewriter);
    Value real = ExtractFloatingPointTensor(complex_val, op.getLoc(), rewriter);
    rewriter.replaceOp(op, real);
    return mlir::success();
  }
};

struct RewriteImagOp : OpRewritePattern<shlo::ImagOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::ImagOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type =
        mlir::dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }
    Value complex_val = UnwrapCast(op.getOperand(), rewriter);
    Value imag = ExtractFloatingPointTensor(complex_val, op.getLoc(), rewriter,
                                            /*is_real=*/false);
    rewriter.replaceOp(op, imag);
    return mlir::success();
  }
};

struct RewriteComplexOp : OpRewritePattern<shlo::ComplexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::ComplexOp op,
                                PatternRewriter& rewriter) const override {
    auto result_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!result_type || !mlir::isa<ComplexType>(result_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }
    Value combined =
        ConcatRealAndImag(op.getLhs(), op.getRhs(), op.getLoc(), rewriter);
    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), result_type, combined);
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

template <typename T>
mlir::DenseElementsAttr ExpandComplexDenseAttr(
    mlir::RankedTensorType new_type, mlir::DenseElementsAttr dense_attr) {
  SmallVector<T> values;
  values.reserve(dense_attr.getNumElements() * 2);
  for (const auto& val : dense_attr.getValues<std::complex<T>>()) {
    values.push_back(val.real());
    values.push_back(val.imag());
  }
  return mlir::DenseElementsAttr::get(new_type, llvm::ArrayRef(values));
}

struct RewriteConstantOp : OpRewritePattern<ma::ConstantOp> {
  using OpRewritePattern<ma::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ma::ConstantOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }
    auto complex_type = mlir::cast<ComplexType>(tensor_type.getElementType());
    Type elem_type = complex_type.getElementType();
    auto new_type =
        RankedTensorType::get(ExpandShape(tensor_type.getShape()), elem_type);

    auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!dense_attr) {
      return rewriter.notifyMatchFailure(op, "not a dense elements attr");
    }

    mlir::Location loc = op.getLoc();
    mlir::DenseElementsAttr new_attr;
    if (elem_type.isF32()) {
      new_attr = ExpandComplexDenseAttr<float>(new_type, dense_attr);
    } else if (elem_type.isF64()) {
      new_attr = ExpandComplexDenseAttr<double>(new_type, dense_attr);
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported complex element type");
    }
    auto new_op = ma::ConstantOp::create(rewriter, loc, new_type, new_attr);
    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, loc, tensor_type, new_op.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

struct RewriteBroadcastInDimOp : OpRewritePattern<shlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!tensor_type || !mlir::isa<ComplexType>(tensor_type.getElementType())) {
      return rewriter.notifyMatchFailure(op, "not a complex tensor");
    }

    auto loc = op.getLoc();
    auto new_type = GetExpandedType(tensor_type);
    Value operand = UnwrapCast(op.getOperand(), rewriter);

    SmallVector<int64_t> new_broadcast_dimensions(
        op.getBroadcastDimensions().begin(), op.getBroadcastDimensions().end());
    new_broadcast_dimensions.push_back(tensor_type.getRank());

    auto new_op = shlo::BroadcastInDimOp::create(
        rewriter, loc, new_type, operand,
        rewriter.getDenseI64ArrayAttr(new_broadcast_dimensions));
    auto cast_to_orig_type = UnrealizedConversionCastOp::create(
        rewriter, loc, tensor_type, new_op.getResult());
    rewriter.replaceOp(op, cast_to_orig_type.getResult(0));
    return mlir::success();
  }
};

class ExpandXtileComplexOpsPass
    : public impl::ExpandXtileComplexOpsPassBase<ExpandXtileComplexOpsPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);

    // clang-format off
    patterns.add<
        RewriteAddOp,
        RewriteBroadcastInDimOp,
        RewriteComplexOp,
        RewriteConstantOp,
        RewriteExtractTileOp,
        RewriteFunctionSignatures,
        RewriteImagOp,
        RewriteInsertTileOp,
        RewriteRealOp,
        RewriteSubtractOp
    >(mlir_context);
    // clang-format on

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
    // Check if there are no unrealized_conversion_casts.
    bool module_has_casts = module
                                .walk([](UnrealizedConversionCastOp op) {
                                  return mlir::WalkResult::interrupt();
                                })
                                .wasInterrupted();
    if (module_has_casts) {
      llvm::outs() << "ExpandXtileComplexOpsPass failed to converge";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace xtile
}  // namespace xla
