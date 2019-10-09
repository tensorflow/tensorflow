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
    patterns.insert<TestNamedPatternRule>(&getContext());

    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace

static mlir::PassRegistration<TestPatternDriver>
    pass("test-patterns", "Run test dialect patterns");

//===----------------------------------------------------------------------===//
// ReturnType Driver.
//===----------------------------------------------------------------------===//

struct ReturnTypeOpMatch : public RewritePattern {
  ReturnTypeOpMatch(MLIRContext *ctx)
      : RewritePattern(OpWithInferTypeInterfaceOp::getOperationName(), 1, ctx) {
  }

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const final {
    if (auto retTypeFn = dyn_cast<InferTypeOpInterface>(op)) {
      SmallVector<Value *, 4> values;
      values.reserve(op->getNumOperands());
      for (auto &operand : op->getOpOperands())
        values.push_back(operand.get());
      (void)retTypeFn.inferReturnTypes(op->getLoc(), values, op->getAttrs(),
                                       op->getRegions());
    }
    return matchFailure();
  }
};

namespace {
struct TestReturnTypeDriver : public FunctionPass<TestReturnTypeDriver> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    patterns.insert<ReturnTypeOpMatch>(&getContext());
    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace

static mlir::PassRegistration<TestReturnTypeDriver>
    rt_pass("test-return-type", "Run return type functions");

//===----------------------------------------------------------------------===//
// Legalization Driver.
//===----------------------------------------------------------------------===//

namespace {
//===----------------------------------------------------------------------===//
// Region-Block Rewrite Testing

/// This pattern is a simple pattern that inlines the first region of a given
/// operation into the parent region.
struct TestRegionRewriteBlockMovement : public ConversionPattern {
  TestRegionRewriteBlockMovement(MLIRContext *ctx)
      : ConversionPattern("test.region", 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Inline this region into the parent region.
    auto &parentRegion = *op->getParentRegion();
    if (op->getAttr("legalizer.should_clone"))
      rewriter.cloneRegionBefore(op->getRegion(0), parentRegion,
                                 parentRegion.end());
    else
      rewriter.inlineRegionBefore(op->getRegion(0), parentRegion,
                                  parentRegion.end());

    // Drop this operation.
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};
/// This pattern is a simple pattern that generates a region containing an
/// illegal operation.
struct TestRegionRewriteUndo : public RewritePattern {
  TestRegionRewriteUndo(MLIRContext *ctx)
      : RewritePattern("test.region_builder", 1, ctx) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const final {
    // Create the region operation with an entry block containing arguments.
    OperationState newRegion(op->getLoc(), "test.region");
    newRegion.addRegion();
    auto *regionOp = rewriter.createOperation(newRegion);
    auto *entryBlock = rewriter.createBlock(&regionOp->getRegion(0));
    entryBlock->addArgument(rewriter.getIntegerType(64));

    // Add an explicitly illegal operation to ensure the conversion fails.
    rewriter.create<ILLegalOpF>(op->getLoc(), rewriter.getIntegerType(32));
    rewriter.create<TestValidOp>(op->getLoc(), ArrayRef<Value *>());

    // Drop this operation.
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Type-Conversion Rewrite Testing

/// This pattern simply erases the given operation.
struct TestDropOp : public ConversionPattern {
  TestDropOp(MLIRContext *ctx) : ConversionPattern("test.drop_op", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};
/// This pattern simply updates the operands of the given operation.
struct TestPassthroughInvalidOp : public ConversionPattern {
  TestPassthroughInvalidOp(MLIRContext *ctx)
      : ConversionPattern("test.invalid", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TestValidOp>(op, llvm::None, operands,
                                             llvm::None);
    return matchSuccess();
  }
};
/// This pattern handles the case of a split return value.
struct TestSplitReturnType : public ConversionPattern {
  TestSplitReturnType(MLIRContext *ctx)
      : ConversionPattern("test.return", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
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

//===----------------------------------------------------------------------===//
// Multi-Level Type-Conversion Rewrite Testing
struct TestChangeProducerTypeI32ToF32 : public ConversionPattern {
  TestChangeProducerTypeI32ToF32(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the type is I32, change the type to F32.
    if (!(*op->result_type_begin()).isInteger(32))
      return matchFailure();
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getF32Type());
    return matchSuccess();
  }
};
struct TestChangeProducerTypeF32ToF64 : public ConversionPattern {
  TestChangeProducerTypeF32ToF64(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the type is F32, change the type to F64.
    if (!(*op->result_type_begin()).isF32())
      return matchFailure();
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getF64Type());
    return matchSuccess();
  }
};
struct TestChangeProducerTypeF32ToInvalid : public ConversionPattern {
  TestChangeProducerTypeF32ToInvalid(MLIRContext *ctx)
      : ConversionPattern("test.type_producer", 10, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Always convert to B16, even though it is not a legal type. This tests
    // that values are unmapped correctly.
    rewriter.replaceOpWithNewOp<TestTypeProducerOp>(op, rewriter.getBF16Type());
    return matchSuccess();
  }
};
struct TestUpdateConsumerType : public ConversionPattern {
  TestUpdateConsumerType(MLIRContext *ctx)
      : ConversionPattern("test.type_consumer", 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Verify that the the incoming operand has been successfully remapped to
    // F64.
    if (!operands[0]->getType().isF64())
      return matchFailure();
    rewriter.replaceOpWithNewOp<TestTypeConsumerOp>(op, operands[0]);
    return matchSuccess();
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

struct TestLegalizePatternDriver
    : public ModulePass<TestLegalizePatternDriver> {
  /// The mode of conversion to use with the driver.
  enum class ConversionMode { Analysis, Full, Partial };

  TestLegalizePatternDriver(ConversionMode mode) : mode(mode) {}

  void runOnModule() override {
    TestTypeConverter converter;
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    patterns
        .insert<TestRegionRewriteBlockMovement, TestRegionRewriteUndo,
                TestDropOp, TestPassthroughInvalidOp, TestSplitReturnType,
                TestChangeProducerTypeI32ToF32, TestChangeProducerTypeF32ToF64,
                TestChangeProducerTypeF32ToInvalid, TestUpdateConsumerType>(
            &getContext());
    mlir::populateFuncOpTypeConversionPattern(patterns, &getContext(),
                                              converter);

    // Define the conversion target used for the test.
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    target.addLegalOp<LegalOpA, TestCastOp, TestValidOp>();
    target.addIllegalOp<ILLegalOpF, TestRegionBuilderOp>();
    target.addDynamicallyLegalOp<TestReturnOp>([](TestReturnOp op) {
      // Don't allow F32 operands.
      return llvm::none_of(op.getOperandTypes(),
                           [](Type type) { return type.isF32(); });
    });
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    // Expect the type_producer/type_consumer operations to only operate on f64.
    target.addDynamicallyLegalOp<TestTypeProducerOp>(
        [](TestTypeProducerOp op) { return op.getType().isF64(); });
    target.addDynamicallyLegalOp<TestTypeConsumerOp>([](TestTypeConsumerOp op) {
      return op.getOperand()->getType().isF64();
    });

    // Handle a partial conversion.
    if (mode == ConversionMode::Partial) {
      (void)applyPartialConversion(getModule(), target, patterns, &converter);
      return;
    }

    // Handle a full conversion.
    if (mode == ConversionMode::Full) {
      (void)applyFullConversion(getModule(), target, patterns, &converter);
      return;
    }

    // Otherwise, handle an analysis conversion.
    assert(mode == ConversionMode::Analysis);

    // Analyze the convertible operations.
    DenseSet<Operation *> legalizedOps;
    if (failed(applyAnalysisConversion(getModule(), target, patterns,
                                       legalizedOps, &converter)))
      return signalPassFailure();

    // Emit remarks for each legalizable operation.
    for (auto *op : legalizedOps)
      op->emitRemark() << "op '" << op->getName() << "' is legalizable";
  }

  /// The mode of conversion to use.
  ConversionMode mode;
};
} // end anonymous namespace

static llvm::cl::opt<TestLegalizePatternDriver::ConversionMode>
    legalizerConversionMode(
        "test-legalize-mode",
        llvm::cl::desc("The legalization mode to use with the test driver"),
        llvm::cl::init(TestLegalizePatternDriver::ConversionMode::Partial),
        llvm::cl::values(
            clEnumValN(TestLegalizePatternDriver::ConversionMode::Analysis,
                       "analysis", "Perform an analysis conversion"),
            clEnumValN(TestLegalizePatternDriver::ConversionMode::Full, "full",
                       "Perform a full conversion"),
            clEnumValN(TestLegalizePatternDriver::ConversionMode::Partial,
                       "partial", "Perform a partial conversion")));

static mlir::PassRegistration<TestLegalizePatternDriver>
    legalizer_pass("test-legalize-patterns",
                   "Run test dialect legalization patterns", [] {
                     return std::make_unique<TestLegalizePatternDriver>(
                         legalizerConversionMode);
                   });
