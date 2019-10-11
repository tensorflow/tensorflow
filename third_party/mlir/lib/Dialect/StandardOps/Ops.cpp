//===- Ops.cpp - Standard MLIR Operations ---------------------------------===//
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

#include "mlir/Dialect/StandardOps/Ops.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// StandardOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
struct StdOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. The desired
  /// name should be streamed into 'os'.
  void getOpResultName(Operation *op, raw_ostream &os) const final {
    if (ConstantOp constant = dyn_cast<ConstantOp>(op))
      return getConstantOpResultName(constant, os);
  }

  /// Get a special name to use when printing the given constant.
  static void getConstantOpResultName(ConstantOp op, raw_ostream &os) {
    Type type = op.getType();
    Attribute value = op.getValue();
    if (auto intCst = value.dyn_cast<IntegerAttr>()) {
      if (type.isIndex()) {
        os << 'c' << intCst.getInt();
      } else if (type.cast<IntegerType>().isInteger(1)) {
        // i1 constants get special names.
        os << (intCst.getInt() ? "true" : "false");
      } else {
        os << 'c' << intCst.getInt() << '_' << type;
      }
    } else if (type.isa<FunctionType>()) {
      os << 'f';
    } else {
      os << "cst";
    }
  }
};

/// This class defines the interface for handling inlining with standard
/// operations.
struct StdInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = dyn_cast<ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<BranchOp>(op->getLoc(), newDest,
                             llvm::to_vector<4>(returnOp.getOperands()));
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value *> valuesToRepl) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()]->replaceAllUsesWith(it.value());
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// StandardOpsDialect
//===----------------------------------------------------------------------===//

/// A custom unary operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op should have one operand");
  assert(op->getNumResults() == 1 && "unary op should have one result");

  const int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << *op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0)->getType();
}

/// A custom binary operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op should have two operands");
  assert(op->getNumResults() == 1 && "binary op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0)->getType();
  if (op->getOperand(0)->getType() != resultType ||
      op->getOperand(1)->getType() != resultType) {
    p.printGenericOp(op);
    return;
  }

  const int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << *op->getOperand(0) << ", " << *op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0)->getType();
}

/// A custom cast operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardCastOp(Operation *op, OpAsmPrinter &p) {
  const int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << *op->getOperand(0) << " : " << op->getOperand(0)->getType() << " to "
    << op->getResult(0)->getType();
}

/// A custom cast operation verifier.
template <typename T> static LogicalResult verifyCastOp(T op) {
  auto opType = op.getOperand()->getType();
  auto resType = op.getType();
  if (!T::areCastCompatible(opType, resType))
    return op.emitError("operand type ") << opType << " and result type "
                                         << resType << " are cast incompatible";

  return success();
}

StandardOpsDialect::StandardOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<DmaStartOp, DmaWaitOp,
#define GET_OP_LIST
#include "mlir/Dialect/StandardOps/Ops.cpp.inc"
                >();
  addInterfaces<StdInlinerInterface, StdOpAsmInterface>();
}

void mlir::printDimAndSymbolList(Operation::operand_iterator begin,
                                 Operation::operand_iterator end,
                                 unsigned numDims, OpAsmPrinter &p) {
  p << '(';
  p.printOperands(begin, begin + numDims);
  p << ')';

  if (begin + numDims != end) {
    p << '[';
    p.printOperands(begin + numDims, end);
    p << ']';
  }
}

// Parses dimension and symbol list, and sets 'numDims' to the number of
// dimension operands parsed.
// Returns 'false' on success and 'true' on error.
ParseResult mlir::parseDimAndSymbolList(OpAsmParser &parser,
                                        SmallVector<Value *, 4> &operands,
                                        unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser.parseOperandList(opInfos, OpAsmParser::Delimiter::Paren))
    return failure();
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto indexTy = parser.getBuilder().getIndexType();
  if (parser.parseOperandList(opInfos,
                              OpAsmParser::Delimiter::OptionalSquare) ||
      parser.resolveOperands(opInfos, indexTy, operands))
    return failure();
  return success();
}

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses m_Constant
/// and checks the operation for an index type.
static detail::op_matcher<ConstantIndexOp> m_ConstantIndex() {
  return detail::op_matcher<ConstantIndexOp>();
}

//===----------------------------------------------------------------------===//
// Common canonicalization pattern support logic
//===----------------------------------------------------------------------===//

namespace {
/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
struct MemRefCastFolder : public RewritePattern {
  /// The rootOpName is the name of the root operation to match against.
  MemRefCastFolder(StringRef rootOpName, MLIRContext *context)
      : RewritePattern(rootOpName, 1, context) {}

  PatternMatchResult match(Operation *op) const override {
    for (auto *operand : op->getOperands())
      if (matchPattern(operand, m_Op<MemRefCastOp>()))
        return matchSuccess();

    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      if (auto *memref = op->getOperand(i)->getDefiningOp())
        if (auto cast = dyn_cast<MemRefCastOp>(memref))
          op->setOperand(i, cast.getOperand());
    rewriter.updatedRootInPlace(op);
  }
};

/// Performs const folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                            const CalculationT &calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<AttrElementT>()) {
    auto rhs = operands[1].dyn_cast_or_null<AttrElementT>();
    if (!rhs || lhs.getType() != rhs.getType())
      return {};

    return AttrElementT::get(lhs.getType(),
                             calculate(lhs.getValue(), rhs.getValue()));
  } else if (auto lhs = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType())
      return {};

    auto elementResult = constFoldBinaryOp<AttrElementT>(
        {lhs.getSplatValue(), rhs.getSplatValue()}, calculate);
    if (!elementResult)
      return {};

    return DenseElementsAttr::get(lhs.getType(), elementResult);
  }
  return {};
}
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

OpFoldResult AddFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult AddIOp::fold(ArrayRef<Attribute> operands) {
  /// addi(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, AllocOp op) {
  p << "alloc";

  // Print dynamic dimension operands.
  MemRefType type = op.getType();
  printDimAndSymbolList(op.operand_begin(), op.operand_end(),
                        type.getNumDynamicDims(), p);
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"map"});
  p << " : " << type;
}

static ParseResult parseAllocOp(OpAsmParser &parser, OperationState &result) {
  MemRefType type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, result.operands, numDimOperands) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  // Check numDynamicDims against number of question marks in memref type.
  // Note: this check remains here (instead of in verify()), because the
  // partition between dim operands and symbol operands is lost after parsing.
  // Verification still checks that the total number of operands matches
  // the number of symbols in the affine map, plus the number of dynamic
  // dimensions in the memref.
  if (numDimOperands != type.getNumDynamicDims())
    return parser.emitError(parser.getNameLoc())
           << "dimension operand count does not equal memref dynamic dimension "
              "count";
  result.types.push_back(type);
  return success();
}

static LogicalResult verify(AllocOp op) {
  auto memRefType = op.getResult()->getType().dyn_cast<MemRefType>();
  if (!memRefType)
    return op.emitOpError("result must be a memref");

  unsigned numSymbols = 0;
  if (!memRefType.getAffineMaps().empty()) {
    // Store number of symbols used in affine map (used in subsequent check).
    AffineMap affineMap = memRefType.getAffineMaps()[0];
    numSymbols = affineMap.getNumSymbols();
  }

  // Check that the total number of operands matches the number of symbols in
  // the affine map, plus the number of dynamic dimensions specified in the
  // memref type.
  unsigned numDynamicDims = memRefType.getNumDynamicDims();
  if (op.getNumOperands() != numDynamicDims + numSymbols)
    return op.emitOpError(
        "operand count does not equal dimension plus symbol operand count");

  // Verify that all operands are of type Index.
  for (auto operandType : op.getOperandTypes())
    if (!operandType.isIndex())
      return op.emitOpError("requires operands to be of type Index");
  return success();
}

namespace {
/// Fold constant dimensions into an alloc operation.
struct SimplifyAllocConst : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AllocOp alloc,
                                     PatternRewriter &rewriter) const override {
    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    if (llvm::none_of(alloc.getOperands(), [](Value *operand) {
          return matchPattern(operand, m_ConstantIndex());
        }))
      return matchFailure();

    auto memrefType = alloc.getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());
    SmallVector<Value *, 4> newOperands;
    SmallVector<Value *, 4> droppedOperands;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType.getRank(); dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (dimSize != -1) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = alloc.getOperand(dynamicDimPos)->getDefiningOp();
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.getValue());
        // Record to check for zero uses later below.
        droppedOperands.push_back(constantIndexOp);
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(-1);
        newOperands.push_back(alloc.getOperand(dynamicDimPos));
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    auto newMemRefType = MemRefType::get(
        newShapeConstants, memrefType.getElementType(),
        memrefType.getAffineMaps(), memrefType.getMemorySpace());
    assert(static_cast<int64_t>(newOperands.size()) ==
           newMemRefType.getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc =
        rewriter.create<AllocOp>(alloc.getLoc(), newMemRefType, newOperands);
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast = rewriter.create<MemRefCastOp>(alloc.getLoc(), newAlloc,
                                                    alloc.getType());

    rewriter.replaceOp(alloc, {resultCast}, droppedOperands);
    return matchSuccess();
  }
};

/// Fold alloc operations with no uses. Alloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadAlloc : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AllocOp alloc,
                                     PatternRewriter &rewriter) const override {
    // Check if the alloc'ed value has any uses.
    if (!alloc.use_empty())
      return matchFailure();

    // If it doesn't, we can eliminate it.
    alloc.erase();
    return matchSuccess();
  }
};
} // end anonymous namespace.

void AllocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<SimplifyAllocConst, SimplifyDeadAlloc>(context);
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState &result) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  result.addSuccessor(dest, destOperands);
  return success();
}

static void print(OpAsmPrinter &p, BranchOp op) {
  p << "br ";
  p.printSuccessorAndUseList(op.getOperation(), 0);
}

Block *BranchOp::getDest() { return getSuccessor(0); }

void BranchOp::setDest(Block *block) { return setSuccessor(block, 0); }

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
  SymbolRefAttr calleeAttr;
  FunctionType calleeType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto calleeLoc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, "callee", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.addTypesToList(calleeType.getResults(), result.types) ||
      parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                             result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, CallOp op) {
  p << "call " << op.getAttr("callee") << '(';
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : ";
  p.printType(op.getCalleeType());
}

static LogicalResult verify(CallOp op) {
  // Check that the callee attribute was specified.
  auto fnAttr = op.getAttrOfType<SymbolRefAttr>("callee");
  if (!fnAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");
  auto fn =
      op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
  if (!fn)
    return op.emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != op.getNumOperands())
    return op.emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (op.getOperand(i)->getType() != fnType.getInput(i))
      return op.emitOpError("operand type mismatch");

  if (fnType.getNumResults() != op.getNumResults())
    return op.emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (op.getResult(i)->getType() != fnType.getResult(i))
      return op.emitOpError("result type mismatch");

  return success();
}

FunctionType CallOp::getCalleeType() {
  SmallVector<Type, 4> resultTypes(getResultTypes());
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, resultTypes, getContext());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//
namespace {
/// Fold indirect calls that have a constant function as the callee operand.
struct SimplifyIndirectCallWithKnownCallee
    : public OpRewritePattern<CallIndirectOp> {
  using OpRewritePattern<CallIndirectOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CallIndirectOp indirectCall,
                                     PatternRewriter &rewriter) const override {
    // Check that the callee is a constant callee.
    SymbolRefAttr calledFn;
    if (!matchPattern(indirectCall.getCallee(), m_Constant(&calledFn)))
      return matchFailure();

    // Replace with a direct call.
    SmallVector<Type, 8> callResults(indirectCall.getResultTypes());
    SmallVector<Value *, 8> callOperands(indirectCall.getArgOperands());
    rewriter.replaceOpWithNewOp<CallOp>(indirectCall, calledFn.getValue(),
                                        callResults, callOperands);
    return matchSuccess();
  }
};
} // end anonymous namespace.

static ParseResult parseCallIndirectOp(OpAsmParser &parser,
                                       OperationState &result) {
  FunctionType calleeType;
  OpAsmParser::OperandType callee;
  llvm::SMLoc operandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  return failure(
      parser.parseOperand(callee) || parser.getCurrentLocation(&operandsLoc) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.resolveOperand(callee, calleeType, result.operands) ||
      parser.resolveOperands(operands, calleeType.getInputs(), operandsLoc,
                             result.operands) ||
      parser.addTypesToList(calleeType.getResults(), result.types));
}

static void print(OpAsmPrinter &p, CallIndirectOp op) {
  p << "call_indirect ";
  p.printOperand(op.getCallee());
  p << '(';
  p.printOperands(op.getArgOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : " << op.getCallee()->getType();
}

static LogicalResult verify(CallIndirectOp op) {
  // The callee must be a function.
  auto fnType = op.getCallee()->getType().dyn_cast<FunctionType>();
  if (!fnType)
    return op.emitOpError("callee must have function type");

  // Verify that the operand and result types match the callee.
  if (fnType.getNumInputs() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (op.getOperand(i + 1)->getType() != fnType.getInput(i))
      return op.emitOpError("operand type mismatch");

  if (fnType.getNumResults() != op.getNumResults())
    return op.emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (op.getResult(i)->getType() != fnType.getResult(i))
      return op.emitOpError("result type mismatch");

  return success();
}

void CallIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyIndirectCallWithKnownCallee>(context);
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getCheckedI1SameShape(Builder *build, Type type) {
  auto i1Type = build->getI1Type();
  if (type.isIntOrIndexOrFloat())
    return i1Type;
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return build->getTensorType(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return build->getTensorType(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return build->getVectorType(vectorType.getShape(), i1Type);
  return Type();
}

static Type getI1SameShape(Builder *build, Type type) {
  Type res = getCheckedI1SameShape(build, type);
  assert(res && "expected type with valid i1 shape");
  return res;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

// Returns an array of mnemonics for CmpIPredicates indexed by values thereof.
static inline const char *const *getCmpIPredicateNames() {
  static const char *predicateNames[]{
      /*EQ*/ "eq",
      /*NE*/ "ne",
      /*SLT*/ "slt",
      /*SLE*/ "sle",
      /*SGT*/ "sgt",
      /*SGE*/ "sge",
      /*ULT*/ "ult",
      /*ULE*/ "ule",
      /*UGT*/ "ugt",
      /*UGE*/ "uge",
  };
  static_assert(std::extent<decltype(predicateNames)>::value ==
                    (size_t)CmpIPredicate::NumPredicates,
                "wrong number of predicate names");
  return predicateNames;
}

// Returns a value of the predicate corresponding to the given mnemonic.
// Returns NumPredicates (one-past-end) if there is no such mnemonic.
CmpIPredicate CmpIOp::getPredicateByName(StringRef name) {
  return llvm::StringSwitch<CmpIPredicate>(name)
      .Case("eq", CmpIPredicate::EQ)
      .Case("ne", CmpIPredicate::NE)
      .Case("slt", CmpIPredicate::SLT)
      .Case("sle", CmpIPredicate::SLE)
      .Case("sgt", CmpIPredicate::SGT)
      .Case("sge", CmpIPredicate::SGE)
      .Case("ult", CmpIPredicate::ULT)
      .Case("ule", CmpIPredicate::ULE)
      .Case("ugt", CmpIPredicate::UGT)
      .Case("uge", CmpIPredicate::UGE)
      .Default(CmpIPredicate::NumPredicates);
}

static void buildCmpIOp(Builder *build, OperationState &result,
                        CmpIPredicate predicate, Value *lhs, Value *rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(build, lhs->getType()));
  result.addAttribute(
      CmpIOp::getPredicateAttrName(),
      build->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static ParseResult parseCmpIOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<NamedAttribute, 4> attrs;
  Attribute predicateNameAttr;
  Type type;
  if (parser.parseAttribute(predicateNameAttr, CmpIOp::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttributeDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return failure();

  if (!predicateNameAttr.isa<StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  StringRef predicateName = predicateNameAttr.cast<StringAttr>().getValue();
  auto predicate = CmpIOp::getPredicateByName(predicateName);
  if (predicate == CmpIPredicate::NumPredicates)
    return parser.emitError(parser.getNameLoc())
           << "unknown comparison predicate \"" << predicateName << "\"";

  auto builder = parser.getBuilder();
  Type i1Type = getCheckedI1SameShape(&builder, type);
  if (!i1Type)
    return parser.emitError(parser.getNameLoc(),
                            "expected type with valid i1 shape");

  attrs[0].second = builder.getI64IntegerAttr(static_cast<int64_t>(predicate));
  result.attributes = attrs;

  result.addTypes({i1Type});
  return success();
}

static void print(OpAsmPrinter &p, CmpIOp op) {
  p << "cmpi ";

  auto predicateValue =
      op.getAttrOfType<IntegerAttr>(CmpIOp::getPredicateAttrName()).getInt();
  assert(predicateValue >= static_cast<int>(CmpIPredicate::FirstValidValue) &&
         predicateValue < static_cast<int>(CmpIPredicate::NumPredicates) &&
         "unknown predicate index");
  Builder b(op.getContext());
  auto predicateStringAttr =
      b.getStringAttr(getCmpIPredicateNames()[predicateValue]);
  p.printAttribute(predicateStringAttr);

  p << ", ";
  p.printOperand(op.lhs());
  p << ", ";
  p.printOperand(op.rhs());
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{CmpIOp::getPredicateAttrName()});
  p << " : " << op.lhs()->getType();
}

static LogicalResult verify(CmpIOp op) {
  auto predicateAttr =
      op.getAttrOfType<IntegerAttr>(CmpIOp::getPredicateAttrName());
  if (!predicateAttr)
    return op.emitOpError("requires an integer attribute named 'predicate'");
  auto predicate = predicateAttr.getInt();
  if (predicate < (int64_t)CmpIPredicate::FirstValidValue ||
      predicate >= (int64_t)CmpIPredicate::NumPredicates)
    return op.emitOpError("'predicate' attribute value out of range");

  return success();
}

// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
// comparison predicates.
static bool applyCmpPredicate(CmpIPredicate predicate, const APInt &lhs,
                              const APInt &rhs) {
  switch (predicate) {
  case CmpIPredicate::EQ:
    return lhs.eq(rhs);
  case CmpIPredicate::NE:
    return lhs.ne(rhs);
  case CmpIPredicate::SLT:
    return lhs.slt(rhs);
  case CmpIPredicate::SLE:
    return lhs.sle(rhs);
  case CmpIPredicate::SGT:
    return lhs.sgt(rhs);
  case CmpIPredicate::SGE:
    return lhs.sge(rhs);
  case CmpIPredicate::ULT:
    return lhs.ult(rhs);
  case CmpIPredicate::ULE:
    return lhs.ule(rhs);
  case CmpIPredicate::UGT:
    return lhs.ugt(rhs);
  case CmpIPredicate::UGE:
    return lhs.uge(rhs);
  default:
    llvm_unreachable("unknown comparison predicate");
  }
}

// Constant folding hook for comparisons.
OpFoldResult CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two arguments");

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return IntegerAttr::get(IntegerType::get(1, getContext()), APInt(1, val));
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

// Returns an array of mnemonics for CmpFPredicates indexed by values thereof.
static inline const char *const *getCmpFPredicateNames() {
  static const char *predicateNames[] = {
      /*AlwaysFalse*/ "false",
      /*OEQ*/ "oeq",
      /*OGT*/ "ogt",
      /*OGE*/ "oge",
      /*OLT*/ "olt",
      /*OLE*/ "ole",
      /*ONE*/ "one",
      /*ORD*/ "ord",
      /*UEQ*/ "ueq",
      /*UGT*/ "ugt",
      /*UGE*/ "uge",
      /*ULT*/ "ult",
      /*ULE*/ "ule",
      /*UNE*/ "une",
      /*UNO*/ "uno",
      /*AlwaysTrue*/ "true",
  };
  static_assert(std::extent<decltype(predicateNames)>::value ==
                    (size_t)CmpFPredicate::NumPredicates,
                "wrong number of predicate names");
  return predicateNames;
}

// Returns a value of the predicate corresponding to the given mnemonic.
// Returns NumPredicates (one-past-end) if there is no such mnemonic.
CmpFPredicate CmpFOp::getPredicateByName(StringRef name) {
  return llvm::StringSwitch<CmpFPredicate>(name)
      .Case("false", CmpFPredicate::AlwaysFalse)
      .Case("oeq", CmpFPredicate::OEQ)
      .Case("ogt", CmpFPredicate::OGT)
      .Case("oge", CmpFPredicate::OGE)
      .Case("olt", CmpFPredicate::OLT)
      .Case("ole", CmpFPredicate::OLE)
      .Case("one", CmpFPredicate::ONE)
      .Case("ord", CmpFPredicate::ORD)
      .Case("ueq", CmpFPredicate::UEQ)
      .Case("ugt", CmpFPredicate::UGT)
      .Case("uge", CmpFPredicate::UGE)
      .Case("ult", CmpFPredicate::ULT)
      .Case("ule", CmpFPredicate::ULE)
      .Case("une", CmpFPredicate::UNE)
      .Case("uno", CmpFPredicate::UNO)
      .Case("true", CmpFPredicate::AlwaysTrue)
      .Default(CmpFPredicate::NumPredicates);
}

static void buildCmpFOp(Builder *build, OperationState &result,
                        CmpFPredicate predicate, Value *lhs, Value *rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(build, lhs->getType()));
  result.addAttribute(
      CmpFOp::getPredicateAttrName(),
      build->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static ParseResult parseCmpFOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<NamedAttribute, 4> attrs;
  Attribute predicateNameAttr;
  Type type;
  if (parser.parseAttribute(predicateNameAttr, CmpFOp::getPredicateAttrName(),
                            attrs) ||
      parser.parseComma() || parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttributeDict(attrs) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
    return failure();

  if (!predicateNameAttr.isa<StringAttr>())
    return parser.emitError(parser.getNameLoc(),
                            "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  StringRef predicateName = predicateNameAttr.cast<StringAttr>().getValue();
  auto predicate = CmpFOp::getPredicateByName(predicateName);
  if (predicate == CmpFPredicate::NumPredicates)
    return parser.emitError(parser.getNameLoc(),
                            "unknown comparison predicate \"" + predicateName +
                                "\"");

  auto builder = parser.getBuilder();
  Type i1Type = getCheckedI1SameShape(&builder, type);
  if (!i1Type)
    return parser.emitError(parser.getNameLoc(),
                            "expected type with valid i1 shape");

  attrs[0].second = builder.getI64IntegerAttr(static_cast<int64_t>(predicate));
  result.attributes = attrs;

  result.addTypes({i1Type});
  return success();
}

static void print(OpAsmPrinter &p, CmpFOp op) {
  p << "cmpf ";

  auto predicateValue =
      op.getAttrOfType<IntegerAttr>(CmpFOp::getPredicateAttrName()).getInt();
  assert(predicateValue >= static_cast<int>(CmpFPredicate::FirstValidValue) &&
         predicateValue < static_cast<int>(CmpFPredicate::NumPredicates) &&
         "unknown predicate index");
  Builder b(op.getContext());
  auto predicateStringAttr =
      b.getStringAttr(getCmpFPredicateNames()[predicateValue]);
  p.printAttribute(predicateStringAttr);

  p << ", ";
  p.printOperand(op.lhs());
  p << ", ";
  p.printOperand(op.rhs());
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{CmpFOp::getPredicateAttrName()});
  p << " : " << op.lhs()->getType();
}

static LogicalResult verify(CmpFOp op) {
  auto predicateAttr =
      op.getAttrOfType<IntegerAttr>(CmpFOp::getPredicateAttrName());
  if (!predicateAttr)
    return op.emitOpError("requires an integer attribute named 'predicate'");
  auto predicate = predicateAttr.getInt();
  if (predicate < (int64_t)CmpFPredicate::FirstValidValue ||
      predicate >= (int64_t)CmpFPredicate::NumPredicates)
    return op.emitOpError("'predicate' attribute value out of range");

  return success();
}

// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
// comparison predicates.
static bool applyCmpPredicate(CmpFPredicate predicate, const APFloat &lhs,
                              const APFloat &rhs) {
  auto cmpResult = lhs.compare(rhs);
  switch (predicate) {
  case CmpFPredicate::AlwaysFalse:
    return false;
  case CmpFPredicate::OEQ:
    return cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::OGT:
    return cmpResult == APFloat::cmpGreaterThan;
  case CmpFPredicate::OGE:
    return cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::OLT:
    return cmpResult == APFloat::cmpLessThan;
  case CmpFPredicate::OLE:
    return cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::ONE:
    return cmpResult != APFloat::cmpUnordered && cmpResult != APFloat::cmpEqual;
  case CmpFPredicate::ORD:
    return cmpResult != APFloat::cmpUnordered;
  case CmpFPredicate::UEQ:
    return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::UGT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan;
  case CmpFPredicate::UGE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::ULT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan;
  case CmpFPredicate::ULE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::UNE:
    return cmpResult != APFloat::cmpEqual;
  case CmpFPredicate::UNO:
    return cmpResult == APFloat::cmpUnordered;
  case CmpFPredicate::AlwaysTrue:
    return true;
  default:
    llvm_unreachable("unknown comparison predicate");
  }
}

// Constant folding hook for comparisons.
OpFoldResult CmpFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpf takes two arguments");

  auto lhs = operands.front().dyn_cast_or_null<FloatAttr>();
  auto rhs = operands.back().dyn_cast_or_null<FloatAttr>();

  // TODO(gcmn) We could actually do some intelligent things if we know only one
  // of the operands, but it's inf or nan.
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return IntegerAttr::get(IntegerType::get(1, getContext()), APInt(1, val));
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

namespace {
/// cond_br true, ^bb1, ^bb2 -> br ^bb1
/// cond_br false, ^bb1, ^bb2 -> br ^bb2
///
struct SimplifyConstCondBranchPred : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CondBranchOp condbr,
                                     PatternRewriter &rewriter) const override {
    // Check that the condition is a constant.
    if (!matchPattern(condbr.getCondition(), m_Op<ConstantOp>()))
      return matchFailure();

    Block *foldedDest;
    SmallVector<Value *, 4> branchArgs;

    // If the condition is known to evaluate to false we fold to a branch to the
    // false destination. Otherwise, we fold to a branch to the true
    // destination.
    if (matchPattern(condbr.getCondition(), m_Zero())) {
      foldedDest = condbr.getFalseDest();
      branchArgs.assign(condbr.false_operand_begin(),
                        condbr.false_operand_end());
    } else {
      foldedDest = condbr.getTrueDest();
      branchArgs.assign(condbr.true_operand_begin(), condbr.true_operand_end());
    }

    rewriter.replaceOpWithNewOp<BranchOp>(condbr, foldedDest, branchArgs);
    return matchSuccess();
  }
};
} // end anonymous namespace.

static ParseResult parseCondBranchOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Value *, 4> destOperands;
  Block *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the condition.
  Type int1Ty = parser.getBuilder().getI1Type();
  if (parser.parseOperand(condInfo) || parser.parseComma() ||
      parser.resolveOperand(condInfo, int1Ty, result.operands)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected condition type was boolean (i1)");
  }

  // Parse the true successor.
  if (parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  result.addSuccessor(dest, destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  result.addSuccessor(dest, destOperands);

  return success();
}

static void print(OpAsmPrinter &p, CondBranchOp op) {
  p << "cond_br ";
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
}

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyConstCondBranchPred>(context);
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ConstantOp &op) {
  p << "constant ";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

  if (op.getAttrs().size() > 1)
    p << ' ';
  p.printAttribute(op.getValue());

  // If the value is a symbol reference, print a trailing type.
  if (op.getValue().isa<SymbolRefAttr>())
    p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
static LogicalResult verify(ConstantOp &op) {
  auto value = op.getValue();
  if (!value)
    return op.emitOpError("requires a 'value' attribute");

  auto type = op.getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";

  if (type.isa<IndexType>() || value.isa<BoolAttr>())
    return success();

  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    // If the type has a known bitwidth we verify that the value can be
    // represented with the given bitwidth.
    auto bitwidth = type.cast<IntegerType>().getWidth();
    auto intVal = intAttr.getValue();
    if (!intVal.isSignedIntN(bitwidth) && !intVal.isIntN(bitwidth))
      return op.emitOpError("requires 'value' to be an integer within the "
                            "range of the integer result type");
    return success();
  }

  if (type.isa<FloatType>()) {
    if (!value.isa<FloatAttr>())
      return op.emitOpError("requires 'value' to be a floating point constant");
    return success();
  }

  if (type.isa<ShapedType>()) {
    if (!value.isa<ElementsAttr>())
      return op.emitOpError("requires 'value' to be a shaped constant");
    return success();
  }

  if (type.isa<FunctionType>()) {
    auto fnAttr = value.dyn_cast<SymbolRefAttr>();
    if (!fnAttr)
      return op.emitOpError("requires 'value' to be a function reference");

    // Try to find the referenced function.
    auto fn =
        op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
    if (!fn)
      return op.emitOpError("reference to undefined function 'bar'");

    // Check that the referenced function has the correct type.
    if (fn.getType() != type)
      return op.emitOpError("reference to function with mismatched type");

    return success();
  }

  if (type.isa<NoneType>() && value.isa<UnitAttr>())
    return success();

  return op.emitOpError("unsupported 'value' attribute: ") << value;
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

/// Returns true if a constant operation can be built with the given value and
/// result type.
bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  // SymbolRefAttr can only be used with a function type.
  if (value.isa<SymbolRefAttr>())
    return type.isa<FunctionType>();
  // Otherwise, the attribute must have the same type as 'type'.
  if (value.getType() != type)
    return false;
  // Finally, check that the attribute kind is handled.
  return value.isa<BoolAttr>() || value.isa<IntegerAttr>() ||
         value.isa<FloatAttr>() || value.isa<ElementsAttr>() ||
         value.isa<UnitAttr>();
}

void ConstantFloatOp::build(Builder *builder, OperationState &result,
                            const APFloat &value, FloatType type) {
  ConstantOp::build(builder, result, type, builder->getFloatAttr(type, value));
}

bool ConstantFloatOp::classof(Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0)->getType().isa<FloatType>();
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::classof(Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0)->getType().isa<IntegerType>();
}

void ConstantIntOp::build(Builder *builder, OperationState &result,
                          int64_t value, unsigned width) {
  Type type = builder->getIntegerType(width);
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

/// Build a constant int op producing an integer with the specified type,
/// which must be an integer type.
void ConstantIntOp::build(Builder *builder, OperationState &result,
                          int64_t value, Type type) {
  assert(type.isa<IntegerType>() && "ConstantIntOp can only have integer type");
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

/// ConstantIndexOp only matches values whose result type is Index.
bool ConstantIndexOp::classof(Operation *op) {
  return ConstantOp::classof(op) && op->getResult(0)->getType().isIndex();
}

void ConstantIndexOp::build(Builder *builder, OperationState &result,
                            int64_t value) {
  Type type = builder->getIndexType();
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//
namespace {
/// Fold Dealloc operations that are deallocating an AllocOp that is only used
/// by other Dealloc operations.
struct SimplifyDeadDealloc : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(DeallocOp dealloc,
                                     PatternRewriter &rewriter) const override {
    // Check that the memref operand's defining operation is an AllocOp.
    Value *memref = dealloc.memref();
    if (!isa_and_nonnull<AllocOp>(memref->getDefiningOp()))
      return matchFailure();

    // Check that all of the uses of the AllocOp are other DeallocOps.
    for (auto *user : memref->getUsers())
      if (!isa<DeallocOp>(user))
        return matchFailure();

    // Erase the dealloc operation.
    rewriter.replaceOp(dealloc, llvm::None);
    return matchSuccess();
  }
};
} // end anonymous namespace.

static void print(OpAsmPrinter &p, DeallocOp op) {
  p << "dealloc " << *op.memref() << " : " << op.memref()->getType();
}

static ParseResult parseDeallocOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefInfo;
  MemRefType type;

  return failure(parser.parseOperand(memrefInfo) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperand(memrefInfo, type, result.operands));
}

static LogicalResult verify(DeallocOp op) {
  if (!op.memref()->getType().isa<MemRefType>())
    return op.emitOpError("operand must be a memref");
  return success();
}

void DeallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  /// dealloc(memrefcast) -> dealloc
  results.insert<MemRefCastFolder>(getOperationName(), context);
  results.insert<SimplifyDeadDealloc>(context);
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, DimOp op) {
  p << "dim " << *op.getOperand() << ", " << op.getIndex();
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"index"});
  p << " : " << op.getOperand()->getType();
}

static ParseResult parseDimOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  Type type;
  Type indexType = parser.getBuilder().getIndexType();

  return failure(
      parser.parseOperand(operandInfo) || parser.parseComma() ||
      parser.parseAttribute(indexAttr, indexType, "index", result.attributes) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(operandInfo, type, result.operands) ||
      parser.addTypeToList(indexType, result.types));
}

static LogicalResult verify(DimOp op) {
  // Check that we have an integer index operand.
  auto indexAttr = op.getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return op.emitOpError("requires an integer attribute named 'index'");
  int64_t index = indexAttr.getValue().getSExtValue();

  auto type = op.getOperand()->getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    if (index >= tensorType.getRank())
      return op.emitOpError("index is out of range");
  } else if (auto memrefType = type.dyn_cast<MemRefType>()) {
    if (index >= memrefType.getRank())
      return op.emitOpError("index is out of range");

  } else if (type.isa<UnrankedTensorType>()) {
    // ok, assumed to be in-range.
  } else {
    return op.emitOpError("requires an operand with tensor or memref type");
  }

  return success();
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold dim when the size along the index referred to is a constant.
  auto opType = getOperand()->getType();
  int64_t indexSize = -1;
  if (auto tensorType = opType.dyn_cast<RankedTensorType>())
    indexSize = tensorType.getShape()[getIndex()];
  else if (auto memrefType = opType.dyn_cast<MemRefType>())
    indexSize = memrefType.getShape()[getIndex()];

  if (indexSize >= 0)
    return IntegerAttr::get(IndexType::get(getContext()), indexSize);

  return {};
}

void DimOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  /// dim(memrefcast) -> dim
  results.insert<MemRefCastFolder>(getOperationName(), context);
}

//===----------------------------------------------------------------------===//
// DivISOp
//===----------------------------------------------------------------------===//

OpFoldResult DivISOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  // Don't fold if it requires division by zero.
  if (rhs.getValue().isNullValue())
    return {};

  // Don't fold if it would overflow.
  bool overflow;
  auto result = lhs.getValue().sdiv_ov(rhs.getValue(), overflow);
  return overflow ? IntegerAttr() : IntegerAttr::get(lhs.getType(), result);
}

//===----------------------------------------------------------------------===//
// DivIUOp
//===----------------------------------------------------------------------===//

OpFoldResult DivIUOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  // Don't fold if it requires division by zero.
  auto rhsValue = rhs.getValue();
  if (rhsValue.isNullValue())
    return {};

  return IntegerAttr::get(lhs.getType(), lhs.getValue().udiv(rhsValue));
}

// ---------------------------------------------------------------------------
// DmaStartOp
// ---------------------------------------------------------------------------

void DmaStartOp::build(Builder *builder, OperationState &result,
                       Value *srcMemRef, ArrayRef<Value *> srcIndices,
                       Value *destMemRef, ArrayRef<Value *> destIndices,
                       Value *numElements, Value *tagMemRef,
                       ArrayRef<Value *> tagIndices, Value *stride,
                       Value *elementsPerStride) {
  result.addOperands(srcMemRef);
  result.addOperands(srcIndices);
  result.addOperands(destMemRef);
  result.addOperands(destIndices);
  result.addOperands({numElements, tagMemRef});
  result.addOperands(tagIndices);
  if (stride)
    result.addOperands({stride, elementsPerStride});
}

void DmaStartOp::print(OpAsmPrinter &p) {
  p << "dma_start " << *getSrcMemRef() << '[';
  p.printOperands(getSrcIndices());
  p << "], " << *getDstMemRef() << '[';
  p.printOperands(getDstIndices());
  p << "], " << *getNumElements();
  p << ", " << *getTagMemRef() << '[';
  p.printOperands(getTagIndices());
  p << ']';
  if (isStrided()) {
    p << ", " << *getStride();
    p << ", " << *getNumElementsPerStride();
  }
  p.printOptionalAttrDict(getAttrs());
  p << " : " << getSrcMemRef()->getType();
  p << ", " << getDstMemRef()->getType();
  p << ", " << getTagMemRef()->getType();
}

// Parse DmaStartOp.
// Ex:
//   %dma_id = dma_start %src[%i, %j], %dst[%k, %l], %size,
//                       %tag[%index], %stride, %num_elt_per_stride :
//                     : memref<3076 x f32, 0>,
//                       memref<1024 x f32, 2>,
//                       memref<1 x i32>
//
ParseResult DmaStartOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> srcIndexInfos;
  OpAsmParser::OperandType dstMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> dstIndexInfos;
  OpAsmParser::OperandType numElementsInfo;
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> tagIndexInfos;
  SmallVector<OpAsmParser::OperandType, 2> strideInfo;

  SmallVector<Type, 3> types;
  auto indexType = parser.getBuilder().getIndexType();

  // Parse and resolve the following list of operands:
  // *) source memref followed by its indices (in square brackets).
  // *) destination memref followed by its indices (in square brackets).
  // *) dma size in KiB.
  if (parser.parseOperand(srcMemRefInfo) ||
      parser.parseOperandList(srcIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(dstMemRefInfo) ||
      parser.parseOperandList(dstIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(numElementsInfo) ||
      parser.parseComma() || parser.parseOperand(tagMemrefInfo) ||
      parser.parseOperandList(tagIndexInfos, OpAsmParser::Delimiter::Square))
    return failure();

  // Parse optional stride and elements per stride.
  if (parser.parseTrailingOperandList(strideInfo))
    return failure();

  bool isStrided = strideInfo.size() == 2;
  if (!strideInfo.empty() && !isStrided) {
    return parser.emitError(parser.getNameLoc(),
                            "expected two stride related operands");
  }

  if (parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 3)
    return parser.emitError(parser.getNameLoc(), "fewer/more types expected");

  if (parser.resolveOperand(srcMemRefInfo, types[0], result.operands) ||
      parser.resolveOperands(srcIndexInfos, indexType, result.operands) ||
      parser.resolveOperand(dstMemRefInfo, types[1], result.operands) ||
      parser.resolveOperands(dstIndexInfos, indexType, result.operands) ||
      // size should be an index.
      parser.resolveOperand(numElementsInfo, indexType, result.operands) ||
      parser.resolveOperand(tagMemrefInfo, types[2], result.operands) ||
      // tag indices should be index.
      parser.resolveOperands(tagIndexInfos, indexType, result.operands))
    return failure();

  auto memrefType0 = types[0].dyn_cast<MemRefType>();
  if (!memrefType0)
    return parser.emitError(parser.getNameLoc(),
                            "expected source to be of memref type");

  auto memrefType1 = types[1].dyn_cast<MemRefType>();
  if (!memrefType1)
    return parser.emitError(parser.getNameLoc(),
                            "expected destination to be of memref type");

  auto memrefType2 = types[2].dyn_cast<MemRefType>();
  if (!memrefType2)
    return parser.emitError(parser.getNameLoc(),
                            "expected tag to be of memref type");

  if (isStrided) {
    if (parser.resolveOperands(strideInfo, indexType, result.operands))
      return failure();
  }

  // Check that source/destination index list size matches associated rank.
  if (static_cast<int64_t>(srcIndexInfos.size()) != memrefType0.getRank() ||
      static_cast<int64_t>(dstIndexInfos.size()) != memrefType1.getRank())
    return parser.emitError(parser.getNameLoc(),
                            "memref rank not equal to indices count");
  if (static_cast<int64_t>(tagIndexInfos.size()) != memrefType2.getRank())
    return parser.emitError(parser.getNameLoc(),
                            "tag memref rank not equal to indices count");

  return success();
}

LogicalResult DmaStartOp::verify() {
  // DMAs from different memory spaces supported.
  if (getSrcMemorySpace() == getDstMemorySpace())
    return emitOpError("DMA should be between different memory spaces");

  if (getNumOperands() != getTagMemRefRank() + getSrcMemRefRank() +
                              getDstMemRefRank() + 3 + 1 &&
      getNumOperands() != getTagMemRefRank() + getSrcMemRefRank() +
                              getDstMemRefRank() + 3 + 1 + 2) {
    return emitOpError("incorrect number of operands");
  }
  return success();
}

void DmaStartOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  /// dma_start(memrefcast) -> dma_start
  results.insert<MemRefCastFolder>(getOperationName(), context);
}

// ---------------------------------------------------------------------------
// DmaWaitOp
// ---------------------------------------------------------------------------

void DmaWaitOp::build(Builder *builder, OperationState &result,
                      Value *tagMemRef, ArrayRef<Value *> tagIndices,
                      Value *numElements) {
  result.addOperands(tagMemRef);
  result.addOperands(tagIndices);
  result.addOperands(numElements);
}

void DmaWaitOp::print(OpAsmPrinter &p) {
  p << "dma_wait ";
  p.printOperand(getTagMemRef());
  p << '[';
  p.printOperands(getTagIndices());
  p << "], ";
  p.printOperand(getNumElements());
  p.printOptionalAttrDict(getAttrs());
  p << " : " << getTagMemRef()->getType();
}

// Parse DmaWaitOp.
// Eg:
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 4>
//
ParseResult DmaWaitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 2> tagIndexInfos;
  Type type;
  auto indexType = parser.getBuilder().getIndexType();
  OpAsmParser::OperandType numElementsInfo;

  // Parse tag memref, its indices, and dma size.
  if (parser.parseOperand(tagMemrefInfo) ||
      parser.parseOperandList(tagIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(numElementsInfo) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(tagMemrefInfo, type, result.operands) ||
      parser.resolveOperands(tagIndexInfos, indexType, result.operands) ||
      parser.resolveOperand(numElementsInfo, indexType, result.operands))
    return failure();

  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType)
    return parser.emitError(parser.getNameLoc(),
                            "expected tag to be of memref type");

  if (static_cast<int64_t>(tagIndexInfos.size()) != memrefType.getRank())
    return parser.emitError(parser.getNameLoc(),
                            "tag memref rank not equal to indices count");

  return success();
}

void DmaWaitOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  /// dma_wait(memrefcast) -> dma_wait
  results.insert<MemRefCastFolder>(getOperationName(), context);
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ExtractElementOp op) {
  p << "extract_element " << *op.getAggregate() << '[';
  p.printOperands(op.getIndices());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getAggregate()->getType();
}

static ParseResult parseExtractElementOp(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::OperandType aggregateInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  ShapedType type;

  auto indexTy = parser.getBuilder().getIndexType();
  return failure(
      parser.parseOperand(aggregateInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(aggregateInfo, type, result.operands) ||
      parser.resolveOperands(indexInfo, indexTy, result.operands) ||
      parser.addTypeToList(type.getElementType(), result.types));
}

static LogicalResult verify(ExtractElementOp op) {
  auto aggregateType = op.getAggregate()->getType().cast<ShapedType>();

  // This should be possible with tablegen type constraints
  if (op.getType() != aggregateType.getElementType())
    return op.emitOpError("result type must match element type of aggregate");

  // Verify the # indices match if we have a ranked type.
  if (aggregateType.hasRank() &&
      aggregateType.getRank() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of indices for extract_element");

  return success();
}

OpFoldResult ExtractElementOp::fold(ArrayRef<Attribute> operands) {
  assert(!operands.empty() && "extract_element takes at least one operand");

  // The aggregate operand must be a known constant.
  Attribute aggregate = operands.front();
  if (!aggregate)
    return {};

  // If this is a splat elements attribute, simply return the value. All of the
  // elements of a splat attribute are the same.
  if (auto splatAggregate = aggregate.dyn_cast<SplatElementsAttr>())
    return splatAggregate.getSplatValue();

  // Otherwise, collect the constant indices into the aggregate.
  SmallVector<uint64_t, 8> indices;
  for (Attribute indice : llvm::drop_begin(operands, 1)) {
    if (!indice || !indice.isa<IntegerAttr>())
      return {};
    indices.push_back(indice.cast<IntegerAttr>().getInt());
  }

  // If this is an elements attribute, query the value at the given indices.
  auto elementsAttr = aggregate.dyn_cast<ElementsAttr>();
  if (elementsAttr && elementsAttr.isValidIndex(indices))
    return elementsAttr.getValue(indices);
  return {};
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast is applicable from index to integer and backwards.
bool IndexCastOp::areCastCompatible(Type a, Type b) {
  return (a.isIndex() && b.isa<IntegerType>()) ||
         (a.isa<IntegerType>() && b.isIndex());
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, LoadOp op) {
  p << "load " << *op.getMemRef() << '[';
  p.printOperands(op.getIndices());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getMemRefType();
}

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType type;

  auto indexTy = parser.getBuilder().getIndexType();
  return failure(
      parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(memrefInfo, type, result.operands) ||
      parser.resolveOperands(indexInfo, indexTy, result.operands) ||
      parser.addTypeToList(type.getElementType(), result.types));
}

static LogicalResult verify(LoadOp op) {
  if (op.getType() != op.getMemRefType().getElementType())
    return op.emitOpError("result type must match element type of memref");

  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("incorrect number of indices for load");

  return success();
}

void LoadOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  /// load(memrefcast) -> load
  results.insert<MemRefCastFolder>(getOperationName(), context);
}

//===----------------------------------------------------------------------===//
// MemRefCastOp
//===----------------------------------------------------------------------===//

bool MemRefCastOp::areCastCompatible(Type a, Type b) {
  auto aT = a.dyn_cast<MemRefType>();
  auto bT = b.dyn_cast<MemRefType>();

  if (!aT || !bT)
    return false;
  if (aT.getElementType() != bT.getElementType())
    return false;
  if (aT.getAffineMaps() != bT.getAffineMaps())
    return false;
  if (aT.getMemorySpace() != bT.getMemorySpace())
    return false;

  // They must have the same rank, and any specified dimensions must match.
  if (aT.getRank() != bT.getRank())
    return false;

  for (unsigned i = 0, e = aT.getRank(); i != e; ++i) {
    int64_t aDim = aT.getDimSize(i), bDim = bT.getDimSize(i);
    if (aDim != -1 && bDim != -1 && aDim != bDim)
      return false;
  }

  return true;
}

OpFoldResult MemRefCastOp::fold(ArrayRef<Attribute> operands) {
  return impl::foldCastOp(*this);
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

OpFoldResult MulFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

OpFoldResult MulIOp::fold(ArrayRef<Attribute> operands) {
  /// muli(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// muli(x, 1) -> x
  if (matchPattern(rhs(), m_One()))
    return getOperand(0);

  // TODO: Handle the overflow case.
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, RankOp op) {
  p << "rank " << *op.getOperand() << " : " << op.getOperand()->getType();
}

static ParseResult parseRankOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  Type type;
  Type indexType = parser.getBuilder().getIndexType();
  return failure(parser.parseOperand(operandInfo) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperand(operandInfo, type, result.operands) ||
                 parser.addTypeToList(indexType, result.types));
}

OpFoldResult RankOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold rank when the rank of the tensor is known.
  auto type = getOperand()->getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return IntegerAttr::get(IndexType::get(getContext()), tensorType.getRank());
  return IntegerAttr();
}

//===----------------------------------------------------------------------===//
// RemISOp
//===----------------------------------------------------------------------===//

OpFoldResult RemISOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "remis takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().srem(rhsValue));
}

//===----------------------------------------------------------------------===//
// RemIUOp
//===----------------------------------------------------------------------===//

OpFoldResult RemIUOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "remiu takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().urem(rhsValue));
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

static void print(OpAsmPrinter &p, ReturnOp op) {
  p << "return";
  if (op.getNumOperands() != 0) {
    p << ' ';
    p.printOperands(op.getOperands());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

static LogicalResult verify(ReturnOp op) {
  auto function = cast<FuncOp>(op.getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i)->getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i)->getType()
             << ") doesn't match function result type (" << results[i] << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

// sitofp is applicable from integer types to float types.
bool SIToFPOp::areCastCompatible(Type a, Type b) {
  return a.isa<IntegerType>() && b.isa<FloatType>();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

static ParseResult parseSelectOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<NamedAttribute, 4> attrs;
  Type type;
  if (parser.parseOperandList(ops, 3) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  auto i1Type = getCheckedI1SameShape(&parser.getBuilder(), type);
  if (!i1Type)
    return parser.emitError(parser.getNameLoc(),
                            "expected type with valid i1 shape");

  SmallVector<Type, 3> types = {i1Type, type, type};
  return failure(parser.resolveOperands(ops, types, parser.getNameLoc(),
                                        result.operands) ||
                 parser.addTypeToList(type, result.types));
}

static void print(OpAsmPrinter &p, SelectOp op) {
  p << "select ";
  p.printOperands(op.getOperands());
  p << " : " << op.getTrueValue()->getType();
  p.printOptionalAttrDict(op.getAttrs());
}

static LogicalResult verify(SelectOp op) {
  auto trueType = op.getTrueValue()->getType();
  auto falseType = op.getFalseValue()->getType();

  if (trueType != falseType)
    return op.emitOpError(
        "requires 'true' and 'false' arguments to be of the same type");

  return success();
}

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  auto *condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(condition, m_One()))
    return getTrueValue();

  // select false, %0, %1 => %1
  if (matchPattern(condition, m_Zero()))
    return getFalseValue();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SignExtendIOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(SignExtendIOp op) {
  // Get the scalar type (which is either directly the type of the operand
  // or the vector's/tensor's element type.
  auto srcType = getElementTypeOrSelf(op.getOperand()->getType());
  auto dstType = getElementTypeOrSelf(op.getType());

  // For now, index is forbidden for the source and the destination type.
  if (srcType.isa<IndexType>())
    return op.emitError() << srcType << " is not a valid operand type";
  if (dstType.isa<IndexType>())
    return op.emitError() << dstType << " is not a valid result type";

  if (srcType.cast<IntegerType>().getWidth() >=
      dstType.cast<IntegerType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, SplatOp op) {
  p << "splat " << *op.getOperand();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getType();
}

static ParseResult parseSplatOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType splatValueInfo;
  ShapedType shapedType;

  return failure(parser.parseOperand(splatValueInfo) ||
                 parser.parseOptionalAttributeDict(result.attributes) ||
                 parser.parseColonType(shapedType) ||
                 parser.resolveOperand(splatValueInfo,
                                       shapedType.getElementType(),
                                       result.operands) ||
                 parser.addTypeToList(shapedType, result.types));
}

static LogicalResult verify(SplatOp op) {
  // TODO: we could replace this by a trait.
  if (op.getOperand()->getType() !=
      op.getType().cast<ShapedType>().getElementType())
    return op.emitError("operand should be of elemental type of result type");

  return success();
}

// Constant folding hook for SplatOp.
OpFoldResult SplatOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "splat takes one operand");

  auto constOperand = operands.front();
  if (!constOperand ||
      (!constOperand.isa<IntegerAttr>() && !constOperand.isa<FloatAttr>()))
    return {};

  auto shapedType = getType().cast<ShapedType>();
  assert(shapedType.getElementType() == constOperand.getType() &&
         "incorrect input attribute type for folding");

  // SplatElementsAttr::get treats single value for second arg as being a splat.
  return SplatElementsAttr::get(shapedType, {constOperand});
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, StoreOp op) {
  p << "store " << *op.getValueToStore();
  p << ", " << *op.getMemRef() << '[';
  p.printOperands(op.getIndices());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getMemRefType();
}

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType memrefType;

  auto indexTy = parser.getBuilder().getIndexType();
  return failure(
      parser.parseOperand(storeValueInfo) || parser.parseComma() ||
      parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(memrefType) ||
      parser.resolveOperand(storeValueInfo, memrefType.getElementType(),
                            result.operands) ||
      parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexTy, result.operands));
}

static LogicalResult verify(StoreOp op) {
  // First operand must have same type as memref element type.
  if (op.getValueToStore()->getType() != op.getMemRefType().getElementType())
    return op.emitOpError(
        "first operand must have same type memref element type");

  if (op.getNumOperands() != 2 + op.getMemRefType().getRank())
    return op.emitOpError("store index operand count not equal to memref rank");

  return success();
}

void StoreOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  /// store(memrefcast) -> store
  results.insert<MemRefCastFolder>(getOperationName(), context);
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

OpFoldResult SubFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

OpFoldResult SubIOp::fold(ArrayRef<Attribute> operands) {
  // subi(x,x) -> 0
  if (getOperand(0) == getOperand(1))
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  /// and(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// and(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  /// or(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// or(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

OpFoldResult XOrOp::fold(ArrayRef<Attribute> operands) {
  /// xor(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// xor(x,x) -> 0
  if (lhs() == rhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

//===----------------------------------------------------------------------===//
// TensorCastOp
//===----------------------------------------------------------------------===//

bool TensorCastOp::areCastCompatible(Type a, Type b) {
  auto aT = a.dyn_cast<TensorType>();
  auto bT = b.dyn_cast<TensorType>();
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return succeeded(verifyCompatibleShape(aT, bT));
}

OpFoldResult TensorCastOp::fold(ArrayRef<Attribute> operands) {
  return impl::foldCastOp(*this);
}

//===----------------------------------------------------------------------===//
// Helpers for Tensor[Load|Store]Op
//===----------------------------------------------------------------------===//

static Type getTensorTypeFromMemRefType(Builder &b, Type type) {
  if (auto memref = type.dyn_cast<MemRefType>())
    return b.getTensorType(memref.getShape(), memref.getElementType());
  return b.getNoneType();
}

//===----------------------------------------------------------------------===//
// TensorLoadOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, TensorLoadOp op) {
  p << "tensor_load " << *op.getOperand();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperand()->getType();
}

static ParseResult parseTensorLoadOp(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::OperandType op;
  Type type;
  return failure(parser.parseOperand(op) ||
                 parser.parseOptionalAttributeDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperand(op, type, result.operands) ||
                 parser.addTypeToList(
                     getTensorTypeFromMemRefType(parser.getBuilder(), type),
                     result.types));
}

//===----------------------------------------------------------------------===//
// TensorStoreOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, TensorStoreOp op) {
  p << "tensor_store " << *op.tensor() << ", " << *op.memref();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.memref()->getType();
}

static ParseResult parseTensorStoreOp(OpAsmParser &parser,
                                      OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(
      parser.parseOperandList(ops, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperands(
          ops, {getTensorTypeFromMemRefType(parser.getBuilder(), type), type},
          loc, result.operands));
}

//===----------------------------------------------------------------------===//
// TruncateIOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(TruncateIOp op) {
  auto srcType = getElementTypeOrSelf(op.getOperand()->getType());
  auto dstType = getElementTypeOrSelf(op.getType());

  if (srcType.isa<IndexType>())
    return op.emitError() << srcType << " is not a valid operand type";
  if (dstType.isa<IndexType>())
    return op.emitError() << dstType << " is not a valid result type";

  if (srcType.cast<IntegerType>().getWidth() <=
      dstType.cast<IntegerType>().getWidth())
    return op.emitError("operand type ")
           << srcType << " must be wider than result type " << dstType;

  return success();
}

//===----------------------------------------------------------------------===//
// ZeroExtendIOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ZeroExtendIOp op) {
  auto srcType = getElementTypeOrSelf(op.getOperand()->getType());
  auto dstType = getElementTypeOrSelf(op.getType());

  if (srcType.isa<IndexType>())
    return op.emitError() << srcType << " is not a valid operand type";
  if (dstType.isa<IndexType>())
    return op.emitError() << dstType << " is not a valid result type";

  if (srcType.cast<IntegerType>().getWidth() >=
      dstType.cast<IntegerType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// FPExtOp
//===----------------------------------------------------------------------===//

bool FPExtOp::areCastCompatible(Type a, Type b) {
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() < fb.getWidth();
  return false;
}

//===----------------------------------------------------------------------===//
// FPTruncOp
//===----------------------------------------------------------------------===//

bool FPTruncOp::areCastCompatible(Type a, Type b) {
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() > fb.getWidth();
  return false;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/StandardOps/Ops.cpp.inc"
