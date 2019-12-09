//===- TestDialect.cpp - MLIR Dialect for Testing -------------------------===//
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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TestDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

// Test support for interacting with the AsmPrinter.
struct TestOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const final {
    if (auto asmOp = dyn_cast<AsmDialectInterfaceOp>(op))
      setNameFn(asmOp, "result");
  }
};

struct TestOpFolderDialectInterface : public OpFolderDialectInterface {
  using OpFolderDialectInterface::OpFolderDialectInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // If this is a one region operation, then insert into it.
    return isa<OneRegionOp>(region->getParentOp());
  }
};

/// This class defines the interface for handling inlining with standard
/// operations.
struct TestInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  bool isLegalToInline(Region *, Region *, BlockAndValueMapping &) const final {
    // Inlining into test dialect regions is legal.
    return true;
  }
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const final {
    // Analyze recursively if this is not a functional region operation, it
    // froms a separate functional scope.
    return !isa<FunctionalRegionOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value *> valuesToRepl) const final {
    // Only handle "test.return" here.
    auto returnOp = dyn_cast<TestReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()]->replaceAllUsesWith(it.value());
  }

  /// Attempt to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value *input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    // Only allow conversion for i16/i32 types.
    if (!(resultType.isInteger(16) || resultType.isInteger(32)) ||
        !(input->getType().isInteger(16) || input->getType().isInteger(32)))
      return nullptr;
    return builder.create<TestCastOp>(conversionLoc, resultType, input);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

TestDialect::TestDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
  addInterfaces<TestOpAsmInterface, TestOpFolderDialectInterface,
                TestInlinerInterface>();
  allowUnknownOperations();
}

LogicalResult TestDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute namedAttr) {
  if (namedAttr.first == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

LogicalResult TestDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute namedAttr) {
  if (namedAttr.first == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

LogicalResult
TestDialect::verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                         unsigned resultIndex,
                                         NamedAttribute namedAttr) {
  if (namedAttr.first == "test.invalid_attr")
    return op->emitError() << "invalid to use 'test.invalid_attr'";
  return success();
}

//===----------------------------------------------------------------------===//
// Test IsolatedRegionOp - parse passthrough region arguments.
//===----------------------------------------------------------------------===//

static ParseResult parseIsolatedRegionOp(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::OperandType argInfo;
  Type argType = parser.getBuilder().getIndexType();

  // Parse the input operand.
  if (parser.parseOperand(argInfo) ||
      parser.resolveOperand(argInfo, argType, result.operands))
    return failure();

  // Parse the body region, and reuse the operand info as the argument info.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, argInfo, argType,
                            /*enableNameShadowing=*/true);
}

static void print(OpAsmPrinter &p, IsolatedRegionOp op) {
  p << "test.isolated_region ";
  p.printOperand(op.getOperand());
  p.shadowRegionArgs(op.region(), op.getOperand());
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// Test parser.
//===----------------------------------------------------------------------===//

static ParseResult parseWrappedKeywordOp(OpAsmParser &parser,
                                         OperationState &result) {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure();
  result.addAttribute("keyword", parser.getBuilder().getStringAttr(keyword));
  return success();
}

static void print(OpAsmPrinter &p, WrappedKeywordOp op) {
  p << WrappedKeywordOp::getOperationName() << " " << op.keyword();
}

//===----------------------------------------------------------------------===//
// Test WrapRegionOp - wrapping op exercising `parseGenericOperation()`.

static ParseResult parseWrappingRegionOp(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseKeyword("wraps"))
    return failure();

  // Parse the wrapped op in a region
  Region &body = *result.addRegion();
  body.push_back(new Block);
  Block &block = body.back();
  Operation *wrapped_op = parser.parseGenericOperation(&block, block.begin());
  if (!wrapped_op)
    return failure();

  // Create a return terminator in the inner region, pass as operand to the
  // terminator the returned values from the wrapped operation.
  SmallVector<Value *, 8> return_operands(wrapped_op->getResults());
  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<TestReturnOp>(wrapped_op->getLoc(), return_operands);

  // Get the results type for the wrapping op from the terminator operands.
  Operation &return_op = body.back().back();
  result.types.append(return_op.operand_type_begin(),
                      return_op.operand_type_end());

  // Use the location of the wrapped op for the "test.wrapping_region" op.
  result.location = wrapped_op->getLoc();

  return success();
}

static void print(OpAsmPrinter &p, WrappingRegionOp op) {
  p << op.getOperationName() << " wraps ";
  p.printGenericOp(&op.region().front().front());
}

//===----------------------------------------------------------------------===//
// Test PolyForOp - parse list of region arguments.
//===----------------------------------------------------------------------===//

static ParseResult parsePolyForOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> ivsInfo;
  // Parse list of region arguments without a delimiter.
  if (parser.parseRegionArgumentList(ivsInfo))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  auto &builder = parser.getBuilder();
  SmallVector<Type, 4> argTypes(ivsInfo.size(), builder.getIndexType());
  return parser.parseRegion(*body, ivsInfo, argTypes);
}

//===----------------------------------------------------------------------===//
// Test removing op with inner ops.
//===----------------------------------------------------------------------===//

namespace {
struct TestRemoveOpWithInnerOps
    : public OpRewritePattern<TestOpWithRegionPattern> {
  using OpRewritePattern<TestOpWithRegionPattern>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TestOpWithRegionPattern op,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};
} // end anonymous namespace

void TestOpWithRegionPattern::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TestRemoveOpWithInnerOps>(context);
}

OpFoldResult TestOpWithRegionFold::fold(ArrayRef<Attribute> operands) {
  return operand();
}

LogicalResult TestOpWithVariadicResultsAndFolder::fold(
    ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results) {
  for (Value *input : this->operands()) {
    results.push_back(input);
  }
  return success();
}

LogicalResult mlir::OpWithInferTypeInterfaceOp::inferReturnTypes(
    llvm::Optional<Location> location, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, ArrayRef<Region> regions,
    SmallVectorImpl<Type> &inferedReturnTypes) {
  if (operands[0]->getType() != operands[1]->getType()) {
    return emitOptionalError(location, "operand type mismatch ",
                             operands[0]->getType(), " vs ",
                             operands[1]->getType());
  }
  inferedReturnTypes.assign({operands[0]->getType()});
  return success();
}

// Static initialization for Test dialect registration.
static mlir::DialectRegistration<mlir::TestDialect> testDialect;

#include "TestOpEnums.cpp.inc"

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
