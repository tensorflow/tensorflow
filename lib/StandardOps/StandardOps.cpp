//===- StandardOps.cpp - Standard MLIR Operations -------------------------===//
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

#include "mlir/StandardOps/StandardOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// StandardOpsDialect
//===----------------------------------------------------------------------===//

StandardOpsDialect::StandardOpsDialect(MLIRContext *context)
    : Dialect(/*namePrefix=*/"", context) {
  addOperations<AllocOp, CallOp, CallIndirectOp, CmpIOp, DeallocOp, DimOp,
                DmaStartOp, DmaWaitOp, ExtractElementOp, LoadOp, MemRefCastOp,
                SelectOp, StoreOp, TensorCastOp,
#define GET_OP_LIST
#include "mlir/StandardOps/standard_ops.inc"
                >();
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

  PatternMatchResult match(OperationInst *op) const override {
    for (auto *operand : op->getOperands())
      if (matchPattern(operand, m_Op<MemRefCastOp>()))
        return matchSuccess();

    return matchFailure();
  }

  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      if (auto *memref = op->getOperand(i)->getDefiningInst())
        if (auto cast = memref->dyn_cast<MemRefCastOp>())
          op->setOperand(i, cast->getOperand());
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
        {lhs.getValue(), rhs.getValue()}, calculate);
    if (!elementResult)
      return {};

    return SplatElementsAttr::get(lhs.getType(), elementResult);
  }
  return {};
}
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

Attribute AddFOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

Attribute AddIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

Value *AddIOp::fold() {
  /// addi(x, 0) -> x
  if (matchPattern(getOperand(1), m_Zero()))
    return getOperand(0);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

void AllocOp::build(Builder *builder, OperationState *result,
                    MemRefType memrefType, ArrayRef<Value *> operands) {
  result->addOperands(operands);
  result->types.push_back(memrefType);
}

void AllocOp::print(OpAsmPrinter *p) const {
  MemRefType type = getType();
  *p << "alloc";
  // Print dynamic dimension operands.
  printDimAndSymbolList(operand_begin(), operand_end(),
                        type.getNumDynamicDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
  *p << " : " << type;
}

bool AllocOp::parse(OpAsmParser *parser, OperationState *result) {
  MemRefType type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, result->operands, numDimOperands) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  // Check numDynamicDims against number of question marks in memref type.
  // Note: this check remains here (instead of in verify()), because the
  // partition between dim operands and symbol operands is lost after parsing.
  // Verification still checks that the total number of operands matches
  // the number of symbols in the affine map, plus the number of dynamic
  // dimensions in the memref.
  if (numDimOperands != type.getNumDynamicDims()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension operand count does not equal memref "
                             "dynamic dimension count");
  }
  result->types.push_back(type);
  return false;
}

bool AllocOp::verify() const {
  auto memRefType = getResult()->getType().dyn_cast<MemRefType>();
  if (!memRefType)
    return emitOpError("result must be a memref");

  unsigned numSymbols = 0;
  if (!memRefType.getAffineMaps().empty()) {
    AffineMap affineMap = memRefType.getAffineMaps()[0];
    // Store number of symbols used in affine map (used in subsequent check).
    numSymbols = affineMap.getNumSymbols();
    // TODO(zinenko): this check does not belong to AllocOp, or any other op but
    // to the type system itself.  It has been partially hoisted to Parser but
    // remains here in case an AllocOp gets constructed programmatically.
    // Remove when we can emit errors directly from *Type::get(...) functions.
    //
    // Verify that the layout affine map matches the rank of the memref.
    if (affineMap.getNumDims() != memRefType.getRank())
      return emitOpError("affine map dimension count must equal memref rank");
  }
  unsigned numDynamicDims = memRefType.getNumDynamicDims();
  // Check that the total number of operands matches the number of symbols in
  // the affine map, plus the number of dynamic dimensions specified in the
  // memref type.
  if (getInstruction()->getNumOperands() != numDynamicDims + numSymbols) {
    return emitOpError(
        "operand count does not equal dimension plus symbol operand count");
  }
  // Verify that all operands are of type Index.
  for (auto *operand : getOperands()) {
    if (!operand->getType().isIndex())
      return emitOpError("requires operands to be of type Index");
  }
  return false;
}

namespace {
/// Fold constant dimensions into an alloc instruction.
struct SimplifyAllocConst : public RewritePattern {
  SimplifyAllocConst(MLIRContext *context)
      : RewritePattern(AllocOp::getOperationName(), 1, context) {}

  PatternMatchResult match(OperationInst *op) const override {
    auto alloc = op->cast<AllocOp>();

    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    for (auto *operand : alloc->getOperands())
      if (matchPattern(operand, m_ConstantIndex()))
        return matchSuccess();
    return matchFailure();
  }

  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    auto allocOp = op->cast<AllocOp>();
    auto memrefType = allocOp->getType();

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
      auto *defOp = allocOp->getOperand(dynamicDimPos)->getDefiningInst();
      OpPointer<ConstantIndexOp> constantIndexOp;
      if (defOp && (constantIndexOp = defOp->dyn_cast<ConstantIndexOp>())) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp->getValue());
        // Record to check for zero uses later below.
        droppedOperands.push_back(constantIndexOp);
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(-1);
        newOperands.push_back(allocOp->getOperand(dynamicDimPos));
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    auto newMemRefType = MemRefType::get(
        newShapeConstants, memrefType.getElementType(),
        memrefType.getAffineMaps(), memrefType.getMemorySpace());
    assert(newOperands.size() == newMemRefType.getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc =
        rewriter.create<AllocOp>(allocOp->getLoc(), newMemRefType, newOperands);
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast = rewriter.create<MemRefCastOp>(allocOp->getLoc(), newAlloc,
                                                    allocOp->getType());

    rewriter.replaceOp(op, {resultCast}, droppedOperands);
  }
};

/// Fold alloc instructions with no uses. Alloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadAlloc : public RewritePattern {
  SimplifyDeadAlloc(MLIRContext *context)
      : RewritePattern(AllocOp::getOperationName(), 1, context) {}

  PatternMatchResult match(OperationInst *op) const override {
    auto alloc = op->cast<AllocOp>();
    // Check if the alloc'ed value has no uses.
    return alloc->use_empty() ? matchSuccess() : matchFailure();
  }

  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    // Erase the alloc operation.
    op->erase();
  }
};
} // end anonymous namespace.

void AllocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyAllocConst>(context));
  results.push_back(std::make_unique<SimplifyDeadAlloc>(context));
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(Builder *builder, OperationState *result, Function *callee,
                   ArrayRef<Value *> operands) {
  result->addOperands(operands);
  result->addAttribute("callee", builder->getFunctionAttr(callee));
  result->addTypes(callee->getType().getResults());
}

bool CallOp::parse(OpAsmParser *parser, OperationState *result) {
  StringRef calleeName;
  llvm::SMLoc calleeLoc;
  FunctionType calleeType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  Function *callee = nullptr;
  if (parser->parseFunctionName(calleeName, calleeLoc) ||
      parser->parseOperandList(operands, /*requiredOperandCount=*/-1,
                               OpAsmParser::Delimiter::Paren) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(calleeType) ||
      parser->resolveFunctionName(calleeName, calleeType, calleeLoc, callee) ||
      parser->addTypesToList(calleeType.getResults(), result->types) ||
      parser->resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                              result->operands))
    return true;

  result->addAttribute("callee", parser->getBuilder().getFunctionAttr(callee));
  return false;
}

void CallOp::print(OpAsmPrinter *p) const {
  *p << "call ";
  p->printFunctionReference(getCallee());
  *p << '(';
  p->printOperands(getOperands());
  *p << ')';
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"callee");
  *p << " : " << getCallee()->getType();
}

bool CallOp::verify() const {
  // Check that the callee attribute was specified.
  auto fnAttr = getAttrOfType<FunctionAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' function attribute");

  // Verify that the operand and result types match the callee.
  auto fnType = fnAttr.getValue()->getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (getOperand(i)->getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch");
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (getResult(i)->getType() != fnType.getResult(i))
      return emitOpError("result type mismatch");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

void CallIndirectOp::build(Builder *builder, OperationState *result,
                           Value *callee, ArrayRef<Value *> operands) {
  auto fnType = callee->getType().cast<FunctionType>();
  result->operands.push_back(callee);
  result->addOperands(operands);
  result->addTypes(fnType.getResults());
}

bool CallIndirectOp::parse(OpAsmParser *parser, OperationState *result) {
  FunctionType calleeType;
  OpAsmParser::OperandType callee;
  llvm::SMLoc operandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  return parser->parseOperand(callee) ||
         parser->getCurrentLocation(&operandsLoc) ||
         parser->parseOperandList(operands, /*requiredOperandCount=*/-1,
                                  OpAsmParser::Delimiter::Paren) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(calleeType) ||
         parser->resolveOperand(callee, calleeType, result->operands) ||
         parser->resolveOperands(operands, calleeType.getInputs(), operandsLoc,
                                 result->operands) ||
         parser->addTypesToList(calleeType.getResults(), result->types);
}

void CallIndirectOp::print(OpAsmPrinter *p) const {
  *p << "call_indirect ";
  p->printOperand(getCallee());
  *p << '(';
  auto operandRange = getOperands();
  p->printOperands(++operandRange.begin(), operandRange.end());
  *p << ')';
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"callee");
  *p << " : " << getCallee()->getType();
}

bool CallIndirectOp::verify() const {
  // The callee must be a function.
  auto fnType = getCallee()->getType().dyn_cast<FunctionType>();
  if (!fnType)
    return emitOpError("callee must have function type");

  // Verify that the operand and result types match the callee.
  if (fnType.getNumInputs() != getNumOperands() - 1)
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (getOperand(i + 1)->getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch");
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (getResult(i)->getType() != fnType.getResult(i))
      return emitOpError("result type mismatch");
  }

  return false;
}

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getCheckedI1SameShape(Builder *build, Type type) {
  auto i1Type = build->getI1Type();
  if (type.isIntOrIndexOrFloat())
    return i1Type;
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return build->getTensorType(tensorType.getShape(), i1Type);
  if (auto tensorType = type.dyn_cast<UnrankedTensorType>())
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

static inline bool isI1(Type type) {
  return type.isa<IntegerType>() && type.cast<IntegerType>().getWidth() == 1;
}

template <typename Ty>
static inline bool implCheckI1SameShape(Ty pattern, Type type) {
  auto specificType = type.dyn_cast<Ty>();
  if (!specificType)
    return true;
  if (specificType.getShape() != pattern.getShape())
    return true;
  return !isI1(specificType.getElementType());
}

// Checks if "type" has the same shape (scalar, vector or tensor) as "pattern"
// and contains i1.
static bool checkI1SameShape(Type pattern, Type type) {
  if (pattern.isIntOrIndexOrFloat())
    return !isI1(type);
  if (auto patternTensorType = pattern.dyn_cast<TensorType>())
    return implCheckI1SameShape(patternTensorType, type);
  if (auto patternVectorType = pattern.dyn_cast<VectorType>())
    return implCheckI1SameShape(patternVectorType, type);

  llvm_unreachable("unsupported type");
}

// Returns an array of mnemonics for CmpIPredicates, indexed by values thereof.
static inline const char *const *getPredicateNames() {
  static const char *predicateNames[(int)CmpIPredicate::NumPredicates]{
      /*EQ*/ "eq",
      /*NE*/ "ne",
      /*SLT*/ "slt",
      /*SLE*/ "sle",
      /*SGT*/ "sgt",
      /*SGE*/ "sge",
      /*ULT*/ "ult",
      /*ULE*/ "ule",
      /*UGT*/ "ugt",
      /*UGE*/ "uge"};
  return predicateNames;
};

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

void CmpIOp::build(Builder *build, OperationState *result,
                   CmpIPredicate predicate, Value *lhs, Value *rhs) {
  result->addOperands({lhs, rhs});
  result->types.push_back(getI1SameShape(build, lhs->getType()));
  result->addAttribute(getPredicateAttrName(),
                       build->getIntegerAttr(build->getIntegerType(64),
                                             static_cast<int64_t>(predicate)));
}

bool CmpIOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<NamedAttribute, 4> attrs;
  Attribute predicateNameAttr;
  Type type;
  if (parser->parseAttribute(predicateNameAttr, getPredicateAttrName().data(),
                             attrs) ||
      parser->parseComma() || parser->parseOperandList(ops, 2) ||
      parser->parseOptionalAttributeDict(attrs) ||
      parser->parseColonType(type) ||
      parser->resolveOperands(ops, type, result->operands))
    return true;

  if (!predicateNameAttr.isa<StringAttr>())
    return parser->emitError(parser->getNameLoc(),
                             "expected string comparison predicate attribute");

  // Rewrite string attribute to an enum value.
  StringRef predicateName = predicateNameAttr.cast<StringAttr>().getValue();
  auto predicate = getPredicateByName(predicateName);
  if (predicate == CmpIPredicate::NumPredicates)
    return parser->emitError(parser->getNameLoc(),
                             "unknown comparison predicate \"" + predicateName +
                                 "\"");

  auto builder = parser->getBuilder();
  Type i1Type = getCheckedI1SameShape(&builder, type);
  if (!i1Type)
    return parser->emitError(parser->getNameLoc(),
                             "expected type with valid i1 shape");

  attrs[0].second = builder.getI64IntegerAttr(static_cast<int64_t>(predicate));
  result->attributes = attrs;

  result->addTypes({i1Type});
  return false;
}

void CmpIOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << " ";

  auto predicateValue =
      getAttrOfType<IntegerAttr>(getPredicateAttrName()).getInt();
  assert(predicateValue >= static_cast<int>(CmpIPredicate::FirstValidValue) &&
         predicateValue < static_cast<int>(CmpIPredicate::NumPredicates) &&
         "unknown predicate index");
  Builder b(getInstruction()->getContext());
  auto predicateStringAttr =
      b.getStringAttr(getPredicateNames()[predicateValue]);
  p->printAttribute(predicateStringAttr);

  *p << ", ";
  p->printOperand(getOperand(0));
  *p << ", ";
  p->printOperand(getOperand(1));
  p->printOptionalAttrDict(getAttrs(),
                           /*elidedAttrs=*/{getPredicateAttrName().data()});
  *p << " : " << getOperand(0)->getType();
}

bool CmpIOp::verify() const {
  auto predicateAttr = getAttrOfType<IntegerAttr>(getPredicateAttrName());
  if (!predicateAttr)
    return emitOpError("requires an integer attribute named 'predicate'");
  auto predicate = predicateAttr.getInt();
  if (predicate < (int64_t)CmpIPredicate::FirstValidValue ||
      predicate >= (int64_t)CmpIPredicate::NumPredicates)
    return emitOpError("'predicate' attribute value out of range");

  return false;
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
Attribute CmpIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  assert(operands.size() == 2 && "cmpi takes two arguments");

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return IntegerAttr::get(IntegerType::get(1, context), APInt(1, val));
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//
namespace {
/// Fold Dealloc instructions that are deallocating an AllocOp that is only used
/// by other Dealloc operations.
struct SimplifyDeadDealloc : public RewritePattern {
  SimplifyDeadDealloc(MLIRContext *context)
      : RewritePattern(DeallocOp::getOperationName(), 1, context) {}

  PatternMatchResult match(OperationInst *op) const override {
    auto dealloc = op->cast<DeallocOp>();

    // Check that the memref operand's defining instruction is an AllocOp.
    Value *memref = dealloc->getMemRef();
    OperationInst *defOp = memref->getDefiningInst();
    if (!defOp || !defOp->isa<AllocOp>())
      return matchFailure();

    // Check that all of the uses of the AllocOp are other DeallocOps.
    for (auto &use : memref->getUses()) {
      auto *user = dyn_cast<OperationInst>(use.getOwner());
      if (!user || !user->isa<DeallocOp>())
        return matchFailure();
    }
    return matchSuccess();
  }

  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    // Erase the dealloc operation.
    op->erase();
  }
};
} // end anonymous namespace.

void DeallocOp::build(Builder *builder, OperationState *result, Value *memref) {
  result->addOperands(memref);
}

void DeallocOp::print(OpAsmPrinter *p) const {
  *p << "dealloc " << *getMemRef() << " : " << getMemRef()->getType();
}

bool DeallocOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  MemRefType type;

  return parser->parseOperand(memrefInfo) || parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands);
}

bool DeallocOp::verify() const {
  if (!getMemRef()->getType().isa<MemRefType>())
    return emitOpError("operand must be a memref");
  return false;
}

void DeallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  /// dealloc(memrefcast) -> dealloc
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
  results.push_back(std::make_unique<SimplifyDeadDealloc>(context));
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::build(Builder *builder, OperationState *result,
                  Value *memrefOrTensor, unsigned index) {
  result->addOperands(memrefOrTensor);
  auto type = builder->getIndexType();
  result->addAttribute("index", builder->getIntegerAttr(type, index));
  result->types.push_back(type);
}

void DimOp::print(OpAsmPrinter *p) const {
  *p << "dim " << *getOperand() << ", " << getIndex();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"index");
  *p << " : " << getOperand()->getType();
}

bool DimOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  Type type;
  Type indexType = parser->getBuilder().getIndexType();

  return parser->parseOperand(operandInfo) || parser->parseComma() ||
         parser->parseAttribute(indexAttr, indexType, "index",
                                result->attributes) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(operandInfo, type, result->operands) ||
         parser->addTypeToList(indexType, result->types);
}

bool DimOp::verify() const {
  // Check that we have an integer index operand.
  auto indexAttr = getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return emitOpError("requires an integer attribute named 'index'");
  uint64_t index = indexAttr.getValue().getZExtValue();

  auto type = getOperand()->getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    if (index >= tensorType.getRank())
      return emitOpError("index is out of range");
  } else if (auto memrefType = type.dyn_cast<MemRefType>()) {
    if (index >= memrefType.getRank())
      return emitOpError("index is out of range");

  } else if (type.isa<UnrankedTensorType>()) {
    // ok, assumed to be in-range.
  } else {
    return emitOpError("requires an operand with tensor or memref type");
  }

  return false;
}

Attribute DimOp::constantFold(ArrayRef<Attribute> operands,
                              MLIRContext *context) const {
  // Constant fold dim when the size along the index referred to is a constant.
  auto opType = getOperand()->getType();
  int64_t indexSize = -1;
  if (auto tensorType = opType.dyn_cast<RankedTensorType>()) {
    indexSize = tensorType.getShape()[getIndex()];
  } else if (auto memrefType = opType.dyn_cast<MemRefType>()) {
    indexSize = memrefType.getShape()[getIndex()];
  }

  if (indexSize >= 0)
    return IntegerAttr::get(Type::getIndex(context), indexSize);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// DivISOp
//===----------------------------------------------------------------------===//

Attribute DivISOp::constantFold(ArrayRef<Attribute> operands,
                                MLIRContext *context) const {
  assert(operands.size() == 2 && "binary operation takes two operands");
  (void)context;

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  // Don't fold if it requires division by zero.
  if (rhs.getValue().isNullValue()) {
    return {};
  }

  // Don't fold if it would overflow.
  bool overflow;
  auto result = lhs.getValue().sdiv_ov(rhs.getValue(), overflow);
  return overflow ? IntegerAttr{} : IntegerAttr::get(lhs.getType(), result);
}

//===----------------------------------------------------------------------===//
// DivIUOp
//===----------------------------------------------------------------------===//

Attribute DivIUOp::constantFold(ArrayRef<Attribute> operands,
                                MLIRContext *context) const {
  assert(operands.size() == 2 && "binary operation takes two operands");
  (void)context;

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  // Don't fold if it requires division by zero.
  if (rhs.getValue().isNullValue()) {
    return {};
  }

  return IntegerAttr::get(lhs.getType(), lhs.getValue().udiv(rhs.getValue()));
}

// ---------------------------------------------------------------------------
// DmaStartOp
// ---------------------------------------------------------------------------

void DmaStartOp::build(Builder *builder, OperationState *result,
                       Value *srcMemRef, ArrayRef<Value *> srcIndices,
                       Value *destMemRef, ArrayRef<Value *> destIndices,
                       Value *numElements, Value *tagMemRef,
                       ArrayRef<Value *> tagIndices, Value *stride,
                       Value *elementsPerStride) {
  result->addOperands(srcMemRef);
  result->addOperands(srcIndices);
  result->addOperands(destMemRef);
  result->addOperands(destIndices);
  result->addOperands(numElements);
  result->addOperands(tagMemRef);
  result->addOperands(tagIndices);
  if (stride) {
    result->addOperands(stride);
    result->addOperands(elementsPerStride);
  }
}

void DmaStartOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << ' ' << *getSrcMemRef() << '[';
  p->printOperands(getSrcIndices());
  *p << "], " << *getDstMemRef() << '[';
  p->printOperands(getDstIndices());
  *p << "], " << *getNumElements();
  *p << ", " << *getTagMemRef() << '[';
  p->printOperands(getTagIndices());
  *p << ']';
  if (isStrided()) {
    *p << ", " << *getStride();
    *p << ", " << *getNumElementsPerStride();
  }
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getSrcMemRef()->getType();
  *p << ", " << getDstMemRef()->getType();
  *p << ", " << getTagMemRef()->getType();
}

// Parse DmaStartOp.
// Ex:
//   %dma_id = dma_start %src[%i, %j], %dst[%k, %l], %size,
//                             %tag[%index] :
//    memref<3076 x f32, 0>,
//    memref<1024 x f32, 2>,
//    memref<1 x i32>
//
bool DmaStartOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType srcMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> srcIndexInfos;
  OpAsmParser::OperandType dstMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> dstIndexInfos;
  OpAsmParser::OperandType numElementsInfo;
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> tagIndexInfos;
  SmallVector<OpAsmParser::OperandType, 2> strideInfo;

  SmallVector<Type, 3> types;
  auto indexType = parser->getBuilder().getIndexType();

  // Parse and resolve the following list of operands:
  // *) source memref followed by its indices (in square brackets).
  // *) destination memref followed by its indices (in square brackets).
  // *) dma size in KiB.
  if (parser->parseOperand(srcMemRefInfo) ||
      parser->parseOperandList(srcIndexInfos, -1,
                               OpAsmParser::Delimiter::Square) ||
      parser->parseComma() || parser->parseOperand(dstMemRefInfo) ||
      parser->parseOperandList(dstIndexInfos, -1,
                               OpAsmParser::Delimiter::Square) ||
      parser->parseComma() || parser->parseOperand(numElementsInfo) ||
      parser->parseComma() || parser->parseOperand(tagMemrefInfo) ||
      parser->parseOperandList(tagIndexInfos, -1,
                               OpAsmParser::Delimiter::Square))
    return true;

  // Parse optional stride and elements per stride.
  if (parser->parseTrailingOperandList(strideInfo)) {
    return true;
  }
  if (!strideInfo.empty() && strideInfo.size() != 2) {
    return parser->emitError(parser->getNameLoc(),
                             "expected two stride related operands");
  }
  bool isStrided = strideInfo.size() == 2;

  if (parser->parseColonTypeList(types))
    return true;

  if (types.size() != 3)
    return parser->emitError(parser->getNameLoc(), "fewer/more types expected");

  if (parser->resolveOperand(srcMemRefInfo, types[0], result->operands) ||
      parser->resolveOperands(srcIndexInfos, indexType, result->operands) ||
      parser->resolveOperand(dstMemRefInfo, types[1], result->operands) ||
      parser->resolveOperands(dstIndexInfos, indexType, result->operands) ||
      // size should be an index.
      parser->resolveOperand(numElementsInfo, indexType, result->operands) ||
      parser->resolveOperand(tagMemrefInfo, types[2], result->operands) ||
      // tag indices should be index.
      parser->resolveOperands(tagIndexInfos, indexType, result->operands))
    return true;

  if (!types[0].isa<MemRefType>())
    return parser->emitError(parser->getNameLoc(),
                             "expected source to be of memref type");

  if (!types[1].isa<MemRefType>())
    return parser->emitError(parser->getNameLoc(),
                             "expected destination to be of memref type");

  if (!types[2].isa<MemRefType>())
    return parser->emitError(parser->getNameLoc(),
                             "expected tag to be of memref type");

  if (isStrided) {
    if (parser->resolveOperand(strideInfo[0], indexType, result->operands) ||
        parser->resolveOperand(strideInfo[1], indexType, result->operands))
      return true;
  }

  // Check that source/destination index list size matches associated rank.
  if (srcIndexInfos.size() != types[0].cast<MemRefType>().getRank() ||
      dstIndexInfos.size() != types[1].cast<MemRefType>().getRank())
    return parser->emitError(parser->getNameLoc(),
                             "memref rank not equal to indices count");

  if (tagIndexInfos.size() != types[2].cast<MemRefType>().getRank())
    return parser->emitError(parser->getNameLoc(),
                             "tag memref rank not equal to indices count");

  return false;
}

bool DmaStartOp::verify() const {
  // DMAs from different memory spaces supported.
  if (getSrcMemorySpace() == getDstMemorySpace()) {
    return emitOpError("DMA should be between different memory spaces");
  }

  if (getNumOperands() != getTagMemRefRank() + getSrcMemRefRank() +
                              getDstMemRefRank() + 3 + 1 &&
      getNumOperands() != getTagMemRefRank() + getSrcMemRefRank() +
                              getDstMemRefRank() + 3 + 1 + 2) {
    return emitOpError("incorrect number of operands");
  }
  return false;
}

void DmaStartOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  /// dma_start(memrefcast) -> dma_start
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

// ---------------------------------------------------------------------------
// DmaWaitOp
// ---------------------------------------------------------------------------

void DmaWaitOp::build(Builder *builder, OperationState *result,
                      Value *tagMemRef, ArrayRef<Value *> tagIndices,
                      Value *numElements) {
  result->addOperands(tagMemRef);
  result->addOperands(tagIndices);
  result->addOperands(numElements);
}

void DmaWaitOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << ' ';
  // Print operands.
  p->printOperand(getTagMemRef());
  *p << '[';
  p->printOperands(getTagIndices());
  *p << "], ";
  p->printOperand(getNumElements());
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getTagMemRef()->getType();
}

// Parse DmaWaitOp.
// Eg:
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 4>
//
bool DmaWaitOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 2> tagIndexInfos;
  Type type;
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType numElementsInfo;

  // Parse tag memref, its indices, and dma size.
  if (parser->parseOperand(tagMemrefInfo) ||
      parser->parseOperandList(tagIndexInfos, -1,
                               OpAsmParser::Delimiter::Square) ||
      parser->parseComma() || parser->parseOperand(numElementsInfo) ||
      parser->parseColonType(type) ||
      parser->resolveOperand(tagMemrefInfo, type, result->operands) ||
      parser->resolveOperands(tagIndexInfos, indexType, result->operands) ||
      parser->resolveOperand(numElementsInfo, indexType, result->operands))
    return true;

  if (!type.isa<MemRefType>())
    return parser->emitError(parser->getNameLoc(),
                             "expected tag to be of memref type");

  if (tagIndexInfos.size() != type.cast<MemRefType>().getRank())
    return parser->emitError(parser->getNameLoc(),
                             "tag memref rank not equal to indices count");

  return false;
}

void DmaWaitOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  /// dma_wait(memrefcast) -> dma_wait
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

void ExtractElementOp::build(Builder *builder, OperationState *result,
                             Value *aggregate, ArrayRef<Value *> indices) {
  auto aggregateType = aggregate->getType().cast<VectorOrTensorType>();
  result->addOperands(aggregate);
  result->addOperands(indices);
  result->types.push_back(aggregateType.getElementType());
}

void ExtractElementOp::print(OpAsmPrinter *p) const {
  *p << "extract_element " << *getAggregate() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getAggregate()->getType();
}

bool ExtractElementOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType aggregateInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  VectorOrTensorType type;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(aggregateInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(aggregateInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type.getElementType(), result->types);
}

bool ExtractElementOp::verify() const {
  if (getNumOperands() == 0)
    return emitOpError("expected an aggregate to index into");

  auto aggregateType = getAggregate()->getType().dyn_cast<VectorOrTensorType>();
  if (!aggregateType)
    return emitOpError("first operand must be a vector or tensor");

  if (getType() != aggregateType.getElementType())
    return emitOpError("result type must match element type of aggregate");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to extract_element must have 'index' type");

  // Verify the # indices match if we have a ranked type.
  auto aggregateRank = aggregateType.getRank();
  if (aggregateRank != -1 && aggregateRank != getNumOperands() - 1)
    return emitOpError("incorrect number of indices for extract_element");

  return false;
}

Attribute ExtractElementOp::constantFold(ArrayRef<Attribute> operands,
                                         MLIRContext *context) const {
  assert(operands.size() > 1 && "extract_element takes atleast one operands");

  // The aggregate operand must be a known constant.
  Attribute aggregate = operands.front();
  if (!aggregate)
    return Attribute();

  // If this is a splat elements attribute, simply return the value. All of the
  // elements of a splat attribute are the same.
  if (auto splatAggregate = aggregate.dyn_cast<SplatElementsAttr>())
    return splatAggregate.getValue();

  // Otherwise, collect the constant indices into the aggregate.
  SmallVector<uint64_t, 8> indices;
  for (Attribute indice : llvm::drop_begin(operands, 1)) {
    if (!indice || !indice.isa<IntegerAttr>())
      return Attribute();
    indices.push_back(indice.cast<IntegerAttr>().getInt());
  }

  // Get the element value of the aggregate attribute with the given constant
  // indices.
  switch (aggregate.getKind()) {
  case Attribute::Kind::DenseFPElements:
  case Attribute::Kind::DenseIntElements:
    return aggregate.cast<DenseElementsAttr>().getValue(indices);
  case Attribute::Kind::SparseElements:
    return aggregate.cast<SparseElementsAttr>().getValue(indices);
  default:
    return Attribute();
  }
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(Builder *builder, OperationState *result, Value *memref,
                   ArrayRef<Value *> indices) {
  auto memrefType = memref->getType().cast<MemRefType>();
  result->addOperands(memref);
  result->addOperands(indices);
  result->types.push_back(memrefType.getElementType());
}

void LoadOp::print(OpAsmPrinter *p) const {
  *p << "load " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getMemRefType();
}

bool LoadOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType type;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type.getElementType(), result->types);
}

bool LoadOp::verify() const {
  if (getNumOperands() == 0)
    return emitOpError("expected a memref to load from");

  auto memRefType = getMemRef()->getType().dyn_cast<MemRefType>();
  if (!memRefType)
    return emitOpError("first operand must be a memref");

  if (getType() != memRefType.getElementType())
    return emitOpError("result type must match element type of memref");

  if (memRefType.getRank() != getNumOperands() - 1)
    return emitOpError("incorrect number of indices for load");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to load must have 'index' type");

  // TODO: Verify we have the right number of indices.

  // TODO: in Function verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return false;
}

void LoadOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  /// load(memrefcast) -> load
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

//===----------------------------------------------------------------------===//
// MemRefCastOp
//===----------------------------------------------------------------------===//

bool MemRefCastOp::verify() const {
  auto opType = getOperand()->getType().dyn_cast<MemRefType>();
  auto resType = getType().dyn_cast<MemRefType>();
  if (!opType || !resType)
    return emitOpError("requires input and result types to be memrefs");

  if (opType == resType)
    return emitOpError("requires the input and result type to be different");

  if (opType.getElementType() != resType.getElementType())
    return emitOpError(
        "requires input and result element types to be the same");

  if (opType.getAffineMaps() != resType.getAffineMaps())
    return emitOpError("requires input and result mappings to be the same");

  if (opType.getMemorySpace() != resType.getMemorySpace())
    return emitOpError(
        "requires input and result memory spaces to be the same");

  // They must have the same rank, and any specified dimensions must match.
  if (opType.getRank() != resType.getRank())
    return emitOpError("requires input and result ranks to match");

  for (unsigned i = 0, e = opType.getRank(); i != e; ++i) {
    int64_t opDim = opType.getDimSize(i), resultDim = resType.getDimSize(i);
    if (opDim != -1 && resultDim != -1 && opDim != resultDim)
      return emitOpError("requires static dimensions to match");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

Attribute MulFOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

Attribute MulIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  // TODO: Handle the overflow case.
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

namespace {
/// muli(x, 0) -> 0
///
struct SimplifyMulX0 : public RewritePattern {
  SimplifyMulX0(MLIRContext *context)
      : RewritePattern(MulIOp::getOperationName(), 1, context) {}

  PatternMatchResult match(OperationInst *op) const override {
    auto muli = op->cast<MulIOp>();

    if (matchPattern(muli->getOperand(1), m_Zero()))
      return matchSuccess();

    return matchFailure();
  }
  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    auto type = op->getOperand(0)->getType();
    auto zeroAttr = rewriter.getZeroAttr(type);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, type, zeroAttr);
  }
};

/// muli(x, 1) -> x
///
struct SimplifyMulX1 : public RewritePattern {
  SimplifyMulX1(MLIRContext *context)
      : RewritePattern(MulIOp::getOperationName(), 1, context) {}

  PatternMatchResult match(OperationInst *op) const override {
    auto muli = op->cast<MulIOp>();

    if (matchPattern(muli->getOperand(1), m_One()))
      return matchSuccess();

    return matchFailure();
  }
  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op->getOperand(0));
  }
};
} // end anonymous namespace.

void MulIOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyMulX0>(context));
  results.push_back(std::make_unique<SimplifyMulX1>(context));
}

//===----------------------------------------------------------------------===//
// RemISOp
//===----------------------------------------------------------------------===//

Attribute RemISOp::constantFold(ArrayRef<Attribute> operands,
                                MLIRContext *context) const {
  assert(operands.size() == 2 && "remis takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};

  // x % 1 = 0
  if (rhs.getValue().isOneValue())
    return IntegerAttr::get(rhs.getType(),
                            APInt(rhs.getValue().getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhs.getValue().isNullValue()) {
    return {};
  }

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};

  return IntegerAttr::get(lhs.getType(), lhs.getValue().srem(rhs.getValue()));
}

//===----------------------------------------------------------------------===//
// RemIUOp
//===----------------------------------------------------------------------===//

Attribute RemIUOp::constantFold(ArrayRef<Attribute> operands,
                                MLIRContext *context) const {
  assert(operands.size() == 2 && "remiu takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};

  // x % 1 = 0
  if (rhs.getValue().isOneValue())
    return IntegerAttr::get(rhs.getType(),
                            APInt(rhs.getValue().getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhs.getValue().isNullValue()) {
    return {};
  }

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};

  return IntegerAttr::get(lhs.getType(), lhs.getValue().urem(rhs.getValue()));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

void SelectOp::build(Builder *builder, OperationState *result, Value *condition,
                     Value *trueValue, Value *falseValue) {
  result->addOperands({condition, trueValue, falseValue});
  result->addTypes(trueValue->getType());
}

bool SelectOp::parse(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<NamedAttribute, 4> attrs;
  Type type;

  if (parser->parseOperandList(ops, 3) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  auto i1Type = getCheckedI1SameShape(&parser->getBuilder(), type);
  if (!i1Type)
    return parser->emitError(parser->getNameLoc(),
                             "expected type with valid i1 shape");

  SmallVector<Type, 3> types = {i1Type, type, type};
  return parser->resolveOperands(ops, types, parser->getNameLoc(),
                                 result->operands) ||
         parser->addTypeToList(type, result->types);
}

void SelectOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << ' ';
  p->printOperands(getInstruction()->getOperands());
  *p << " : " << getTrueValue()->getType();
  p->printOptionalAttrDict(getAttrs());
}

bool SelectOp::verify() const {
  auto conditionType = getCondition()->getType();
  auto trueType = getTrueValue()->getType();
  auto falseType = getFalseValue()->getType();

  if (trueType != falseType)
    return emitOpError(
        "requires 'true' and 'false' arguments to be of the same type");

  if (checkI1SameShape(trueType, conditionType))
    return emitOpError("requires the condition to have the same shape as "
                       "arguments with elemental type i1");

  return false;
}

Attribute SelectOp::constantFold(ArrayRef<Attribute> operands,
                                 MLIRContext *context) const {
  assert(operands.size() == 3 && "select takes three operands");

  // select true, %0, %1 => %0
  // select false, %0, %1 => %1
  auto cond = operands[0].dyn_cast_or_null<IntegerAttr>();
  if (!cond)
    return {};

  if (cond.getValue().isNullValue()) {
    return operands[2];
  } else if (cond.getValue().isOneValue()) {
    return operands[1];
  }

  llvm_unreachable("first argument of select must be i1");
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(Builder *builder, OperationState *result,
                    Value *valueToStore, Value *memref,
                    ArrayRef<Value *> indices) {
  result->addOperands(valueToStore);
  result->addOperands(memref);
  result->addOperands(indices);
}

void StoreOp::print(OpAsmPrinter *p) const {
  *p << "store " << *getValueToStore();
  *p << ", " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << getMemRefType();
}

bool StoreOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType memrefType;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(storeValueInfo) || parser->parseComma() ||
         parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(memrefType) ||
         parser->resolveOperand(storeValueInfo, memrefType.getElementType(),
                                result->operands) ||
         parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands);
}

bool StoreOp::verify() const {
  if (getNumOperands() < 2)
    return emitOpError("expected a value to store and a memref");

  // Second operand is a memref type.
  auto memRefType = getMemRef()->getType().dyn_cast<MemRefType>();
  if (!memRefType)
    return emitOpError("second operand must be a memref");

  // First operand must have same type as memref element type.
  if (getValueToStore()->getType() != memRefType.getElementType())
    return emitOpError("first operand must have same type memref element type");

  if (getNumOperands() != 2 + memRefType.getRank())
    return emitOpError("store index operand count not equal to memref rank");

  for (auto *idx : getIndices())
    if (!idx->getType().isIndex())
      return emitOpError("index to load must have 'index' type");

  // TODO: Verify we have the right number of indices.

  // TODO: in Function verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return false;
}

void StoreOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  /// store(memrefcast) -> store
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

Attribute SubFOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

Attribute SubIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

namespace {
/// subi(x,x) -> 0
///
struct SimplifyXMinusX : public RewritePattern {
  SimplifyXMinusX(MLIRContext *context)
      : RewritePattern(SubIOp::getOperationName(), 1, context) {}

  PatternMatchResult match(OperationInst *op) const override {
    auto subi = op->cast<SubIOp>();
    if (subi->getOperand(0) == subi->getOperand(1))
      return matchSuccess();

    return matchFailure();
  }
  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    auto subi = op->cast<SubIOp>();
    auto result =
        rewriter.create<ConstantIntOp>(op->getLoc(), 0, subi->getType());

    rewriter.replaceOp(op, {result});
  }
};
} // end anonymous namespace.

void SubIOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyXMinusX>(context));
}

//===----------------------------------------------------------------------===//
// TensorCastOp
//===----------------------------------------------------------------------===//

bool TensorCastOp::verify() const {
  auto opType = getOperand()->getType().dyn_cast<TensorType>();
  auto resType = getType().dyn_cast<TensorType>();
  if (!opType || !resType)
    return emitOpError("requires input and result types to be tensors");

  if (opType == resType)
    return emitOpError("requires the input and result type to be different");

  if (opType.getElementType() != resType.getElementType())
    return emitOpError(
        "requires input and result element types to be the same");

  // If the source or destination are unranked, then the cast is valid.
  auto opRType = opType.dyn_cast<RankedTensorType>();
  auto resRType = resType.dyn_cast<RankedTensorType>();
  if (!opRType || !resRType)
    return false;

  // If they are both ranked, they have to have the same rank, and any specified
  // dimensions must match.
  if (opRType.getRank() != resRType.getRank())
    return emitOpError("requires input and result ranks to match");

  for (unsigned i = 0, e = opRType.getRank(); i != e; ++i) {
    int64_t opDim = opRType.getDimSize(i), resultDim = resRType.getDimSize(i);
    if (opDim != -1 && resultDim != -1 && opDim != resultDim)
      return emitOpError("requires static dimensions to match");
  }

  return false;
}

