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

#include "xla/backends/cpu/codegen/transforms/xla_cpu_rewrite_patterns.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/ir/xla_cpu_types.h"

namespace xla::cpu {

void PopulateXlaCpuTypeConversionAndLegality(mlir::TypeConverter& converter,
                                             mlir::ConversionTarget& target) {
  converter.addConversion([](CallFrameType call_frame) {
    return mlir::LLVM::LLVMPointerType::get(call_frame.getContext());
  });

  target.addIllegalDialect<XlaCpuDialect>();
}

namespace {
struct LowerLoadOp : public mlir::OpConversionPattern<LoadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      LoadOp op, LoadOp::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override;
};
}  // namespace

// LLVM structs corresponds to `XLA_CPU_KernelCallFrame` struct that defines
// XLA:CPU host kernel ABI contract.

static mlir::LLVM::LLVMStructType KernelDim3Type(mlir::MLIRContext* ctx) {
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return mlir::LLVM::LLVMStructType::getNewIdentified(ctx, "kernel_dim3",
                                                      {i64, i64, i64});
}

static mlir::LLVM::LLVMStructType KernelArgType(mlir::MLIRContext* ctx) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return mlir::LLVM::LLVMStructType::getNewIdentified(ctx, "kernel_arg",
                                                      {ptr, i64});
}

static mlir::LLVM::LLVMStructType KernelCallFrameType(mlir::MLIRContext* ctx) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return mlir::LLVM::LLVMStructType::getNewIdentified(ctx, "kernel_call_frame",
                                                      {ptr, ptr, i64, ptr});
}

mlir::LogicalResult LowerLoadOp::matchAndRewrite(
    LoadOp op, LoadOp::Adaptor adaptor,
    mlir::ConversionPatternRewriter& rewriter) const {
  mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  auto ptr = b.getType<mlir::LLVM::LLVMPointerType>();
  auto kernel_call_frame = KernelCallFrameType(b.getContext());
  auto kernel_arg = KernelArgType(b.getContext());

  // Get a pointer to the first `KernelArg` struct.
  auto args_gep = b.create<mlir::LLVM::GEPOp>(
      ptr, kernel_call_frame, adaptor.getCallFrame(), mlir::LLVM::GEPArg(3));
  auto args_ptr = b.create<mlir::LLVM::LoadOp>(ptr, args_gep);

  // Get a pointer to the `KernelArg` at the given index.
  auto arg_gep = b.create<mlir::LLVM::GEPOp>(ptr, kernel_arg, args_ptr,
                                             mlir::LLVM::GEPArg(op.getIndex()));
  auto arg_ptr = b.create<mlir::LLVM::LoadOp>(ptr, arg_gep);

  rewriter.replaceOp(op, arg_ptr);
  return mlir::success();
}

void PopulateXlaCpuConversionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<LowerLoadOp>(patterns.getContext());
}

}  // namespace xla::cpu
