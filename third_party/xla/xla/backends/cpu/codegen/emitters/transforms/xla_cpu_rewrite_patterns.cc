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

#include "xla/backends/cpu/codegen/emitters/transforms/xla_cpu_rewrite_patterns.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"

namespace xla::cpu {
namespace {

static mlir::LLVM::LLVMStructType KernelDim3Type(mlir::MLIRContext* ctx) {
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return mlir::LLVM::LLVMStructType::getNewIdentified(ctx, "kernel_dim3",
                                                      {i64, i64, i64});
}

static mlir::LLVM::LLVMStructType KernelArgType(mlir::MLIRContext* ctx) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return mlir::LLVM::LLVMStructType::getNewIdentified(ctx, "XLA_CPU_KernelArg",
                                                      {ptr, i64});
}

static mlir::LLVM::LLVMStructType KernelCallFrameType(mlir::MLIRContext* ctx) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return mlir::LLVM::LLVMStructType::getNewIdentified(
      ctx, "XLA_CPU_KernelCallFrame", {ptr, ptr, i64, ptr});
}

struct LowerLoadOp : public mlir::OpRewritePattern<LoadOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      cpu::LoadOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto ptr = b.getType<mlir::LLVM::LLVMPointerType>();
    auto kernel_call_frame = KernelCallFrameType(b.getContext());
    auto kernel_arg = KernelArgType(b.getContext());

    // Get a pointer to the first `KernelArg` struct.
    auto cast = b.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), ptr,
                                                           op.getCallFrame())
                    .getResult(0);
    auto args_gep = b.create<mlir::LLVM::GEPOp>(
        ptr, kernel_call_frame, cast,
        llvm::SmallVector<mlir::LLVM::GEPArg, 2>{mlir::LLVM::GEPArg(0),
                                                 mlir::LLVM::GEPArg(3)},
        /*inbounds=*/true);
    auto args_ptr = b.create<mlir::LLVM::LoadOp>(ptr, args_gep);
    args_ptr.setInvariant(true);

    // Get a pointer to the `KernelArg` at the given index.
    auto arg_gep = b.create<mlir::LLVM::GEPOp>(
        ptr, kernel_arg, args_ptr,
        llvm::SmallVector<mlir::LLVM::GEPArg, 2>{
            mlir::LLVM::GEPArg(op.getIndex()), mlir::LLVM::GEPArg(0)},
        /*inbounds=*/true);
    auto arg_ptr = b.create<mlir::LLVM::LoadOp>(ptr, arg_gep);
    arg_ptr.setInvariant(true);
    arg_ptr->setAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                     b.getIndexAttr(32));

    auto arg_ptr_cast = b.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), op->getResult(0).getType(), arg_ptr.getResult());
    rewriter.replaceOp(op, arg_ptr_cast.getResult(0));
    return mlir::success();
  }
};

struct LowerThreadId : public mlir::OpRewritePattern<ThreadIdOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      cpu::ThreadIdOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto ptr = b.getType<mlir::LLVM::LLVMPointerType>();
    auto kernel_call_frame = KernelCallFrameType(b.getContext());
    auto kernel_dim = KernelDim3Type(b.getContext());
    auto i64_ty = b.getIntegerType(
        mlir::DataLayout::closest(b.getInsertionBlock()->getParentOp())
            .getTypeSizeInBits(b.getI64Type()));

    // Get a pointer to the `KernelThread` struct.
    auto cast = b.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), ptr,
                                                           op.getCallFrame())
                    .getResult(0);
    auto tid_gep = b.create<mlir::LLVM::GEPOp>(
        ptr, kernel_call_frame, cast, mlir::LLVM::GEPArg(1), /*inbounds=*/true);
    auto tid_ptr = b.create<mlir::LLVM::LoadOp>(ptr, tid_gep);

    // Load 'x'.
    auto thread_x_get = b.create<mlir::LLVM::GEPOp>(
        ptr, kernel_dim, tid_ptr, mlir::LLVM::GEPArg(0), /*inbounds=*/true);
    auto thread_id = b.create<mlir::LLVM::LoadOp>(i64_ty, thread_x_get);

    mlir::Value tix = thread_id.getResult();
    auto index_ty = b.getIntegerType(
        mlir::DataLayout::closest(b.getInsertionBlock()->getParentOp())
            .getTypeSizeInBits(op.getType()));
    if (index_ty != i64_ty) {
      tix = b.create<mlir::LLVM::TruncOp>(index_ty, tix);
    }
    auto thread_id_cast = b.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), op->getResult(0).getType(), tix);

    rewriter.replaceOp(op, thread_id_cast.getResult(0));
    return mlir::success();
  }
};

struct LowerStoreOp : public mlir::OpRewritePattern<StoreOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      StoreOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerSuccessOp : public mlir::OpRewritePattern<SuccessOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SuccessOp op, mlir::PatternRewriter& rewriter) const override {
    auto elementPtrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, elementPtrType);
    return mlir::success();
  }
};

struct RewriteFunctionSignatures : mlir::OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    auto func_type = op.getFunctionType();
    if (func_type.getNumInputs() != 1 || func_type.getNumResults() != 1 ||
        !mlir::isa<CallFrameType>(func_type.getInput(0)) ||
        !mlir::isa<ErrorType>(func_type.getResult(0))) {
      return rewriter.notifyMatchFailure(
          op, "the function signature does not match the XLA_CPU_Kernel type.");
    }

    auto ptr = rewriter.getType<mlir::LLVM::LLVMPointerType>();
    llvm::SmallVector<mlir::Type> new_operands{ptr};
    rewriter.setInsertionPointToStart(&op.getBody().front());

    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), func_type.getInput(0), op.getArgument(0));
    op.getArgument(0).replaceAllUsesExcept(cast.getResult(0), cast);
    op.setFunctionType(rewriter.getFunctionType(new_operands, {ptr}));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }
    return mlir::success();
  }
};

}  // namespace

void PopulateXlaCpuConversionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<LowerLoadOp, LowerStoreOp, LowerThreadId, LowerSuccessOp,
               RewriteFunctionSignatures>(patterns.getContext());
}

}  // namespace xla::cpu
