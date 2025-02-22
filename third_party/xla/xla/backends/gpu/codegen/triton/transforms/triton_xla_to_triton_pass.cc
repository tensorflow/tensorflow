/* Copyright 2025 The OpenXLA Authors.

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

#include <stdbool.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Dialect.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

namespace mt = ::mlir::triton;
namespace xgt = ::xla::gpu::triton;

namespace {

#define GEN_PASS_DEF_TRITONXLATOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

SmallVector<Value> GetValueRangeI64(::xla::EmitterLocOpBuilder& builder,
                                    llvm::ArrayRef<int64_t> array_ref) {
  SmallVector<mlir::Value> values;
  for (int64_t value : array_ref) {
    values.push_back(
        builder.create<arith::ConstantIntOp>(value, builder.getI64Type()));
  }
  return values;
}

SmallVector<Value> GetValueRangeI32(::xla::EmitterLocOpBuilder& builder,
                                    llvm::ArrayRef<int32_t> array_ref) {
  SmallVector<mlir::Value> values;
  for (int32_t value : array_ref) {
    values.push_back(
        builder.create<arith::ConstantIntOp>(value, builder.getI32Type()));
  }
  return values;
}

mt::PointerType GetTensorPtrType(::xla::EmitterLocOpBuilder& builder,
                                 TensorType tensor_type) {
  return mt::PointerType::get(xgt::StorageType(builder, tensor_type),
                              mlir::NVVM::kGlobalMemorySpace);
}

mt::PointerType GetTensorPtrType(::xla::EmitterLocOpBuilder& builder,
                                 Type element_type) {
  return mt::PointerType::get(xgt::StorageType(builder, element_type),
                              mlir::NVVM::kGlobalMemorySpace);
}

bool AreInputsTensorOrTensorPtrTypes(ArrayRef<Type> types) {
  for (auto type : types) {
    if (!mlir::isa<TensorType>(type) && !mlir::isa<mt::PointerType>(type)) {
      return false;
    }
  }
  return true;
}

bool AreAllInputsTensorPtrType(ArrayRef<Type> types) {
  for (auto type : types) {
    if (!mlir::isa<mt::PointerType>(type)) {
      return false;
    }
  }
  return true;
}

bool AreAllInputsTensorType(ArrayRef<Type> types) {
  for (auto type : types) {
    if (!mlir::isa<TensorType>(type)) {
      return false;
    }
  }
  return true;
}

bool AreInputsSameType(ArrayRef<Type> types) {
  return AreAllInputsTensorPtrType(types) || AreAllInputsTensorType(types);
}

struct RewriteFuncOp : mlir::OpRewritePattern<mt::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  // Rewrite tensors<> to !tt.ptr<tensor>
  // Remove any returns. i.e. tt.return with no operands.
  mlir::LogicalResult matchAndRewrite(
      mt::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    ::xla::EmitterLocOpBuilder builder(op.getLoc(), rewriter);

    auto input_types = op.getFunctionType().getInputs();

    if (!AreInputsTensorOrTensorPtrTypes(input_types)) {
      return rewriter.notifyMatchFailure(
          op, "Unexpected input types. Expected Tensor or TensorPtr types.");
    }

    if (!AreInputsSameType(input_types)) {
      return rewriter.notifyMatchFailure(
          op,
          "Unexpected mixture of types in inputs. Expecting all to be Tensor "
          "Type or all to be TensorPtr Type.");
    }

    if (!input_types.empty() && mlir::isa<mt::PointerType>(input_types[0])) {
      return rewriter.notifyMatchFailure(op, "Already in TensorPtr type.");
    }

    mlir::Block* entry_block = &op.getBody().front();
    SmallVector<Type> new_result_types;
    SmallVector<Value> new_results;

    // tt.return should have no operands after rewriting since we materialize
    // all tensors.
    entry_block->getTerminator()->eraseOperands(
        0, entry_block->getTerminator()->getNumOperands());

    SmallVector<Type> new_operand_types(input_types);
    rewriter.setInsertionPointToStart(entry_block);
    for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
      mlir::BlockArgument func_arg = op.getArgument(index);

      // !tt.ptr<> -> tensor
      auto cast_to_orig_type = builder.create<mlir::UnrealizedConversionCastOp>(
          operand_type, func_arg);
      func_arg.replaceAllUsesExcept(cast_to_orig_type.getResult(0),
                                    cast_to_orig_type);
      operand_type = GetTensorPtrType(
          builder, mlir::cast<TensorType>(operand_type).getElementType());
    }

    // Replace the function arguments with the new types.
    for (auto [arg, arg_type] :
         llvm::zip(entry_block->getArguments(), new_operand_types)) {
      arg.setType(arg_type);
    }

    // Update the function signature.
    op.setType(rewriter.getFunctionType(new_operand_types, new_result_types));

    return mlir::success();
  }
};

struct RewriteTile : mlir::OpRewritePattern<mlir::triton::xla::TileOp> {
  using OpRewritePattern::OpRewritePattern;

  // Rewriting TileOp as tt.make_tensor_ptr.
  mlir::LogicalResult matchAndRewrite(
      mlir::triton::xla::TileOp op,
      mlir::PatternRewriter& rewriter) const override {
    ::xla::EmitterLocOpBuilder builder(op.getLoc(), rewriter);

    // Order is 0, 1, ..., rank - 1.
    std::vector<int32_t> dim_order(op.getSizes().size());
    std::iota(dim_order.begin(), dim_order.end(), 0);

    // tensor -> !tt.ptr<>
    auto cast_to_tensor_ptr_type =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                GetTensorPtrType(builder,
                                 op.getTensor().getType().getElementType()),
                op.getTensor())
            .getResult(0);

    auto tensor_ptr =
        builder
            .create<mt::MakeTensorPtrOp>(
                cast_to_tensor_ptr_type,
                GetValueRangeI64(builder, op.getTensor().getType().getShape()),
                GetValueRangeI64(builder, op.getStrides()),
                GetValueRangeI32(builder, op.getOffsets()), op.getSizes(),
                dim_order)
            .getResult();

    // !tt.ptr<tensor> -> tiled_tensor
    auto cast_to_tiled_tensor_type =
        builder.create<mlir::UnrealizedConversionCastOp>(
            op.getTiledTensor().getType(), tensor_ptr);

    rewriter.replaceOp(op, cast_to_tiled_tensor_type);
    return mlir::success();
  }
};

struct RewriteExtract : mlir::OpRewritePattern<mlir::triton::xla::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  // Rewriting ExtractOp as tt.advance + tt.load.
  mlir::LogicalResult matchAndRewrite(
      mlir::triton::xla::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    ::xla::EmitterLocOpBuilder builder(op.getLoc(), rewriter);

    // tiled_tensor -> !tt.ptr<tensor>
    auto cast_to_tensor_ptr_type =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                GetTensorPtrType(builder,
                                 RankedTensorType::get(
                                     op.getSrc().getType().getTileShape(),
                                     op.getSrc().getType().getElementType())),
                op.getSrc())
            .getResult(0);

    auto advance =
        builder.create<mt::AdvanceOp>(cast_to_tensor_ptr_type.getType(),
                                      cast_to_tensor_ptr_type, op.getOffsets());

    // TODO(manany): Actually provide boundary info here. For now, we assume
    // perfect tiling.
    std::vector<int32_t> boundary_checks;
    std::optional<mt::PaddingOption> padding;
    auto load = builder
                    .create<mt::LoadOp>(advance, boundary_checks, padding,
                                        mt::CacheModifier::NONE,
                                        mt::EvictionPolicy::NORMAL,
                                        /*isVolatile=*/false)
                    .getResult();

    rewriter.replaceOp(op, load);
    return mlir::success();
  }
};

struct RewriteInsert : mlir::OpRewritePattern<mlir::triton::xla::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  // Rewriting InsertOp as tt.advance + tt.store.
  mlir::LogicalResult matchAndRewrite(
      mlir::triton::xla::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    ::xla::EmitterLocOpBuilder builder(op.getLoc(), rewriter);

    // tiled_tensor -> !tt.ptr<tensor>
    auto cast_dst_to_tensor_ptr_type =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                GetTensorPtrType(builder,
                                 RankedTensorType::get(
                                     op.getDst().getType().getTileShape(),
                                     op.getDst().getType().getElementType())),
                op.getDst())
            .getResult(0);

    auto advance = builder.create<mt::AdvanceOp>(
        cast_dst_to_tensor_ptr_type.getType(), cast_dst_to_tensor_ptr_type,
        op.getOffsets());

    // TODO(manany): Actually provide boundary info here. For now, we assume
    // perfect tiling.
    std::vector<int32_t> boundary_checks;
    rewriter.create<mt::StoreOp>(op->getLoc(), advance, op.getSrc(),
                                 boundary_checks, mt::CacheModifier::NONE,
                                 mt::EvictionPolicy::NORMAL);

    // InsertOp has a result, so we propagate it to the users.
    op->replaceAllUsesWith(ValueRange(op.getDst()));

    return mlir::success();
  }
};

struct TritonXLAToTritonPass
    : public impl::TritonXLAToTritonPassBase<TritonXLAToTritonPass> {
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<RewriteFuncOp, RewriteTile, RewriteExtract, RewriteInsert>(
        mlir_context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAToTritonPass() {
  return std::make_unique<TritonXLAToTritonPass>();
}

}  // namespace mlir::triton::xla
