//===- TestPatterns.cpp - Test dialect pattern driver ---------------------===//
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

#include "TestDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;

// Native function for testing NativeCodeCall
static Value *chooseOperand(Value *input1, Value *input2, BoolAttr choice) {
  return choice.getValue() ? input1 : input2;
}

namespace {
#include "TestPatterns.inc"
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Canonicalizer Driver.
//===----------------------------------------------------------------------===//

namespace {
struct TestPatternDriver : public FunctionPass<TestPatternDriver> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);

    // Verify named pattern is generated with expected name.
    RewriteListBuilder<TestNamedPatternRule>::build(patterns, &getContext());

    applyPatternsGreedily(getFunction(), std::move(patterns));
  }
};
} // end anonymous namespace

static mlir::PassRegistration<TestPatternDriver>
    pass("test-patterns", "Run test dialect patterns");

//===----------------------------------------------------------------------===//
// Legalization Driver.
//===----------------------------------------------------------------------===//

namespace {
/// This pattern is a simple pattern that inlines the first region of a given
/// operation into the parent region.
struct TestRegionRewriteBlockMovement : public ConversionPattern {
  TestRegionRewriteBlockMovement(MLIRContext *ctx)
      : ConversionPattern("test.region", 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const final {
    // Inline this region into the parent region.
    auto &parentRegion = *op->getContainingRegion();
    rewriter.inlineRegionBefore(op->getRegion(0), parentRegion,
                                parentRegion.end());

    // Drop this operation.
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};
/// This pattern simply erases the given operation.
struct TestDropOp : public ConversionPattern {
  TestDropOp(MLIRContext *ctx) : ConversionPattern("test.drop_op", 1, ctx) {}
  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};
/// This pattern simply updates the operands of the given operation.
struct TestPassthroughInvalidOp : public ConversionPattern {
  TestPassthroughInvalidOp(MLIRContext *ctx)
      : ConversionPattern("test.invalid", 1, ctx) {}
  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TestValidOp>(op, llvm::None, operands,
                                             llvm::None);
    return matchSuccess();
  }
};
/// This pattern handles the case of a split return value.
struct TestSplitReturnType : public ConversionPattern {
  TestSplitReturnType(MLIRContext *ctx)
      : ConversionPattern("test.return", 1, ctx) {}
  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const final {
    // Check for a return of F32.
    if (op->getNumOperands() != 1 || !op->getOperand(0)->getType().isF32())
      return matchFailure();

    // Check if the first operation is a cast operation, if it is we use the
    // results directly.
    auto *defOp = operands[0]->getDefiningOp();
    if (auto packerOp = llvm::dyn_cast_or_null<TestCastOp>(defOp)) {
      SmallVector<Value *, 2> returnOperands(packerOp.getOperands());
      rewriter.replaceOpWithNewOp<TestReturnOp>(op, returnOperands);
      return matchSuccess();
    }

    // Otherwise, fail to match.
    return matchFailure();
  }
};
} // namespace

namespace {
struct TestTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) override {
    // Drop I16 types.
    if (t.isInteger(16))
      return success();

    // Convert I64 to F64.
    if (t.isInteger(64)) {
      results.push_back(FloatType::getF64(t.getContext()));
      return success();
    }

    // Split F32 into F16,F16.
    if (t.isF32()) {
      results.assign(2, FloatType::getF16(t.getContext()));
      return success();
    }

    // Otherwise, convert the type directly.
    results.push_back(t);
    return success();
  }

  /// Override the hook to materialize a conversion. This is necessary because
  /// we generate 1->N type mappings.
  Operation *materializeConversion(PatternRewriter &rewriter, Type resultType,
                                   ArrayRef<Value *> inputs,
                                   Location loc) override {
    return rewriter.create<TestCastOp>(loc, resultType, inputs);
  }
};

struct TestConversionTarget : public ConversionTarget {
  TestConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalOp<LegalOpA, TestValidOp>();
    addDynamicallyLegalOp<TestReturnOp>();
  }
  bool isDynamicallyLegal(Operation *op) const final {
    // Don't allow F32 operands.
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isF32(); });
  }
};

struct TestLegalizePatternDriver
    : public ModulePass<TestLegalizePatternDriver> {
  void runOnModule() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    RewriteListBuilder<TestRegionRewriteBlockMovement, TestDropOp,
                       TestPassthroughInvalidOp,
                       TestSplitReturnType>::build(patterns, &getContext());

    TestTypeConverter converter;
    TestConversionTarget target(getContext());
    if (failed(applyPartialConversion(getModule(), target, converter,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

static mlir::PassRegistration<TestLegalizePatternDriver>
    legalizer_pass("test-legalize-patterns",
                   "Run test dialect legalization patterns");
