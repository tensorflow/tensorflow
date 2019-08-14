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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

TestDialect::TestDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Test PolyForOp - parse list of region arguments.
//===----------------------------------------------------------------------===//
ParseResult parsePolyForOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 4> ivsInfo;
  // Parse list of region arguments without a delimiter.
  if (parser->parseRegionArgumentList(ivsInfo))
    return failure();

  // Parse the body region.
  Region *body = result->addRegion();
  auto &builder = parser->getBuilder();
  SmallVector<Type, 4> argTypes(ivsInfo.size(), builder.getIndexType());
  if (parser->parseRegion(*body, ivsInfo, argTypes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Test removing op with inner ops.
//===----------------------------------------------------------------------===//

namespace {
struct TestRemoveOpWithInnerOps : public OpRewritePattern<TestOpWithRegion> {
  using OpRewritePattern<TestOpWithRegion>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TestOpWithRegion op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};
} // end anonymous namespace

void TestOpWithRegion::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TestRemoveOpWithInnerOps>(context);
}

// Static initialization for Test dialect registration.
static mlir::DialectRegistration<mlir::TestDialect> testDialect;

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
