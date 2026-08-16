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

using mlir::ComplexType;
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
using mlir::arith::ConstantIndexOp;
using mlir::stablehlo::AddOp;
using mlir::stablehlo::SubtractOp;
using xtile::EntryFuncOp;
using xtile::ExtractTileOp;
using xtile::InsertTileOp;

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
        RewriteExtractTileOp,
        RewriteFunctionSignatures,
        RewriteInsertTileOp,
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
