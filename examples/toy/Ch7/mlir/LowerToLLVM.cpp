//====- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM ------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LowerAffine.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<MemRefType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule, llvmDialect);
    Value *formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule,
        llvmDialect);
    Value *newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule, llvmDialect);

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value *, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<loop::ForOp>(loc, lowerBound, upperBound, step);
      loop.getBody()->clear();
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToStart(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      rewriter.create<loop::TerminatorOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<toy::PrintOp>(op);
    auto elementLoad = rewriter.create<LoadOp>(loc, printOp.input(), loopIvs);
    rewriter.create<CallOp>(
        loc, printfRef, rewriter.getIntegerType(32),
        ArrayRef<Value *>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return matchSuccess();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get("printf", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy,
                                                    /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get("printf", context);
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value *getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                        StringRef name, StringRef value,
                                        ModuleOp module,
                                        LLVM::LLVMDialect *llvmDialect) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
      global = builder.create<LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, name, builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value *globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value *cst0 = builder.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
        ArrayRef<Value *>({cst0, cst0}));
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ToyToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToLLVMLoweringPass : public ModulePass<ToyToLLVMLoweringPass> {
  void runOnModule() final;
};
} // end anonymous namespace

void ToyToLLVMLoweringPass::runOnModule() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. Do perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns, &typeConverter)))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
