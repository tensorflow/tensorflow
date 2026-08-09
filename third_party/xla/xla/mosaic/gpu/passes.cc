/* Copyright 2024 The JAX Authors.

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

#include "xla/mosaic/gpu/passes.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/mosaic/pass_boilerplate.h"

namespace mosaic {
namespace gpu {

namespace {

// Upstream MLIR does not implement an LLVM lowering pattern for this op.
struct ConvertExtractStridedSlicePattern final
    : public mlir::OpConversionPattern<mlir::vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::vector::ExtractStridedSliceOp op, OpAdaptor subst,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto vty = op.getSourceVectorType();
    if (vty.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "only 1-D vectors are supported");
    }
    int64_t size =
        (*op.getSizes().getAsRange<mlir::IntegerAttr>().begin()).getInt();
    if (size < 0) {
      return rewriter.notifyMatchFailure(op, "size is negative");
    }
    int64_t start =
        (*op.getOffsets().getAsRange<mlir::IntegerAttr>().begin()).getInt();
    int64_t stride =
        (*op.getStrides().getAsRange<mlir::IntegerAttr>().begin()).getInt();
    if (stride != 1) {
      return rewriter.notifyMatchFailure(op, "only stride 1 is supported");
    }
    if (start < 0 || start + size > vty.getShape()[0]) {
      return rewriter.notifyMatchFailure(op, "slice is out of bounds");
    }
    mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(
        op.getLoc(), op.getResult().getType());
    for (int64_t i = 0; i < size; ++i) {
      result = rewriter.create<mlir::LLVM::InsertElementOp>(
          op.getLoc(), result,
          rewriter.create<mlir::LLVM::ExtractElementOp>(
              op.getLoc(), subst.getVector(),
              rewriter.create<mlir::LLVM::ConstantOp>(
                  op.getLoc(), rewriter.getI32IntegerAttr(i + start))),
          rewriter.create<mlir::LLVM::ConstantOp>(
              op.getLoc(), rewriter.getI32IntegerAttr(i)));
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertGpuToLLVMPass
    : public jaxlib::mlir::Pass<ConvertGpuToLLVMPass, mlir::ModuleOp> {
 public:
  using jaxlib::mlir::Pass<ConvertGpuToLLVMPass, mlir::ModuleOp>::Pass;
  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-convert-gpu-to-llvm";
  static constexpr llvm::StringLiteral kPassName = "ConvertGpuToLLVMPass";

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::LLVMTypeConverter converter(ctx);
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::gpu::GPUModuleOp>();
    target.addDynamicallyLegalOp<mlir::gpu::LaunchFuncOp>(
        [&](mlir::gpu::LaunchFuncOp op) -> bool {
          return converter.isLegal(op->getOperandTypes()) &&
                 converter.isLegal(op->getResultTypes());
        });
    auto symtab = mlir::SymbolTable(getOperation());
    mlir::populateGpuToLLVMConversionPatterns(converter, patterns, false);
    patterns.insert<ConvertExtractStridedSlicePattern>(&getContext());
    if (mlir::applyPartialConversion(getOperation(), target,
                                     std::move(patterns))
            .failed()) {
      signalPassFailure();
    }
  }
};

// Convert all array parameters to GPU kernels into byval pointers.
// NVVM backend converts them into arrays in the .param memory space.
// We only use arrays to pass in TMA descriptors, which is why we also
// require 64-byte alignment.
class ByvalInsertionPass
    : public jaxlib::mlir::Pass<ByvalInsertionPass, mlir::gpu::GPUModuleOp> {
 public:
  using jaxlib::mlir::Pass<ByvalInsertionPass, mlir::gpu::GPUModuleOp>::Pass;
  static constexpr llvm::StringLiteral kArgumentName = "mosaic-byval-insertion";
  static constexpr llvm::StringLiteral kPassName = "ByvalInsertionPass";

  void runOnOperation() override {
    auto result = getOperation().walk([](mlir::LLVM::LLVMFuncOp op) {
      // TODO(apaszke): op.isDeclaration() always returns false...
      if (op.getFunctionBody().empty()) {  // Skip over declarations.
        return mlir::WalkResult::advance();
      }
      auto ptr_ty = mlir::LLVM::LLVMPointerType::get(op.getContext());
      mlir::LLVM::LLVMFunctionType func_ty = op.getFunctionType();
      std::vector<mlir::Type> new_arg_types = func_ty.getParams().vec();
      for (unsigned i = 0; i < op.getNumArguments(); ++i) {
        mlir::BlockArgument arg = op.getArgument(i);
        if (!mlir::isa<mlir::LLVM::LLVMArrayType>(arg.getType())) {
          continue;
        }
        if (op.getArgAttrDict(i)) {
          op->emitOpError(
              "!llvm.array argument already has some argument attributes");
          return mlir::WalkResult::interrupt();
        }
        // It would be a lot simpler to use op.insertArgument, but the
        // impl of FunctionOpInterface for llvm.func is _completely_ broken
        new_arg_types[i] = ptr_ty;
        op.setArgAttr(i, "llvm.byval", mlir::TypeAttr::get(arg.getType()));
        op.setArgAttr(i, "nvvm.grid_constant",
                      mlir::UnitAttr::get(op.getContext()));
        op.setArgAttr(i, "llvm.align",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(op.getContext(), 32), 64));
        arg.setType(ptr_ty);
      }
      op.setFunctionType(mlir::LLVM::LLVMFunctionType::get(
          func_ty.getReturnType(), new_arg_types, func_ty.isVarArg()));
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

void registerConvertGpuToLLVMPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<ConvertGpuToLLVMPass>();
  });
}

void registerByvalInsertionPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<ByvalInsertionPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
