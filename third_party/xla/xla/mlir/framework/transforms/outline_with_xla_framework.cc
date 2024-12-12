/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/mlir/framework/ir/xla_framework.h"
#include "xla/mlir/framework/transforms/passes.h"

namespace mlir {
namespace xla_framework {
namespace {

// Given a FuncOp with only memref args/outputs, create a new function that
// wraps/unwraps xla_framework.buffer types and then calls the original
// function.
//
// For example:
//   func @func_to_outline(%arg0: memref<?xf32>) -> memref<?xf32>
//
// Will generate:
//   func @func_to_outline_xla_framework(%arg0: !xla_framework.buffer)
//     -> !xla_framework.buffer attributes {xla_entry = true} {
//    %0 = xla_framework.buffer_to_mem %arg0 : memref<?xf32>
//    %1 = call @func_to_outline(%0) : (memref<?xf32>) -> memref<?xf32>
//    %2 = xla_framework.mem_to_buffer %1 : memref<?xf32>
//    return %2 : !xla_framework.buffer
//   }
struct OutlineXLAFunc : public RewritePattern {
  explicit OutlineXLAFunc(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(func::FuncOp::getOperationName(), benefit, context) {}

  static void filterFuncAttributes(func::FuncOp func, bool argAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {
    for (const auto &attr : func->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == func.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (argAttrs && attr.getName() == func.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto func = dyn_cast<func::FuncOp>(op);
    auto ctx = rewriter.getContext();
    auto loc = func.getLoc();
    SmallVector<Location> locs(func.getFunctionType().getNumInputs(), loc);

    // Functions should only be outlined once and should only use memrefs
    if (!func) return failure();
    if (func.getSymName() != "main") return failure();
    if (llvm::any_of(op->getOperandTypes(),
                     [](Type t) { return !mlir::isa<MemRefType>(t); }) ||
        op->getNumResults() != 0)
      return failure();
    if (func->hasAttr("outlined")) return failure();
    func->setAttr("outlined", BoolAttr::get(ctx, true));

    // Prepare new func attribute information
    func.setSymNameAttr(mlir::StringAttr::get(ctx, func.getName()));
    SmallVector<Type> operands(func.getFunctionType().getNumInputs(),
                               ::mlir::xla_framework::BufferType::get(ctx));
    SmallVector<Type> result_array(func.getFunctionType().getNumResults(),
                                   ::mlir::xla_framework::BufferType::get(ctx));
    auto func_type = FunctionType::get(ctx, operands, result_array);
    SmallVector<NamedAttribute> attrs;
    filterFuncAttributes(func, true, attrs);
    SmallVector<DictionaryAttr> arg_attrs;
    func.getAllArgAttrs(arg_attrs);

    // The wrapper function will have the same name but with _xla_framework
    // appended and will be annotated with the attribute "xla_entry".
    auto outline_func = rewriter.create<func::FuncOp>(
        loc, func.getSymName().str() + "_xla_framework", func_type, attrs,
        arg_attrs);
    outline_func->setAttr("outlined", BoolAttr::get(ctx, true));
    outline_func->setAttr("xla_entry", BoolAttr::get(ctx, true));
    auto *b = rewriter.createBlock(&outline_func.getBody(), {},
                                   func_type.getInputs(), locs);

    // Unwrap arguments
    SmallVector<Value> args;
    for (const auto &t : llvm::enumerate(func.getFunctionType().getInputs())) {
      args.push_back(rewriter.create<xla_framework::XLABufferToMemOp>(
          loc, t.value(), b->getArgument(t.index())));
    }

    auto call = rewriter.create<func::CallOp>(
        loc, func.getSymName(), func.getFunctionType().getResults(), args);
    // Wrap results
    SmallVector<Value> results;
    for (auto t : call.getResults()) {
      results.push_back(rewriter.create<xla_framework::MemToXLABufferOp>(
          loc, ::mlir::xla_framework::BufferType::get(ctx), t));
    }

    rewriter.create<func::ReturnOp>(loc, results);

    // Finally, mark the called function as private to prevent users from
    // accidentally trying to use it.
    Attribute linkage = mlir::LLVM::LinkageAttr::get(
        rewriter.getContext(), mlir::LLVM::Linkage::Internal);
    func->setAttr("llvm.linkage", linkage);
    func.setPrivate();

    return success();
  }
};

#define GEN_PASS_DEF_OUTLINEWITHXLAFRAMEWORK
#include "xla/mlir/framework/transforms/passes.h.inc"

class OutlineWithXLAFrameworkPass
    : public impl::OutlineWithXLAFrameworkBase<OutlineWithXLAFrameworkPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<xla_framework::XLAFrameworkDialect, mlir::LLVM::LLVMDialect,
                    mlir::BuiltinDialect>();
  }

 public:
  explicit OutlineWithXLAFrameworkPass() {}

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext *ctx = m.getContext();

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<OutlineXLAFunc>(ctx);
    //  Set target.

    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
      signalPassFailure();
    }
    m->walk([](func::FuncOp f) {
      if (f->hasAttr("outlined")) f->removeAttr("outlined");
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateOutlineWithXLAFrameworkPass() {
  return std::make_unique<OutlineWithXLAFrameworkPass>();
}

}  // namespace xla_framework
}  // namespace mlir
