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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// StandardOpsDialect
//===----------------------------------------------------------------------===//

StandardOpsDialect::StandardOpsDialect(MLIRContext *context)
    : Dialect(/*opPrefix=*/"", context) {
  addOperations<AddFOp, AddIOp, AllocOp, CallOp, CallIndirectOp, DeallocOp,
                DimOp, DmaStartOp, DmaWaitOp, ExtractElementOp, LoadOp,
                MemRefCastOp, MulFOp, MulIOp, StoreOp, SubFOp, SubIOp,
                TensorCastOp>();
}

//===----------------------------------------------------------------------===//
// Common canonicalization pattern support logic
//===----------------------------------------------------------------------===//

namespace {
/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
struct MemRefCastFolder : public Pattern {
  /// The rootOpName is the name of the root operation to match against.
  MemRefCastFolder(StringRef rootOpName, MLIRContext *context)
      : Pattern(rootOpName, context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    for (auto *operand : op->getOperands())
      if (auto *memref = operand->getDefiningOperation())
        if (memref->isa<MemRefCastOp>())
          return matchSuccess();

    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      if (auto *memref = op->getOperand(i)->getDefiningOperation())
        if (auto cast = memref->dyn_cast<MemRefCastOp>())
          op->setOperand(i, cast->getOperand());
    rewriter.updatedRootInPlace(op);
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

Attribute AddFOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  assert(operands.size() == 2 && "addf takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>())
      return FloatAttr::get(lhs.getValue() + rhs.getValue(), context);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

Attribute AddIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  assert(operands.size() == 2 && "addi takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>())
      return IntegerAttr::get(lhs.getValue() + rhs.getValue(), context);
  }

  return nullptr;
}

namespace {
/// addi(x, 0) -> x
///
struct SimplifyAddX0 : public Pattern {
  SimplifyAddX0(MLIRContext *context)
      : Pattern(AddIOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto addi = op->cast<AddIOp>();
    if (auto *operandOp = addi->getOperand(1)->getDefiningOperation())
      // TODO: Support splatted zero as well.  We need a general zero pattern.
      if (auto cst = operandOp->dyn_cast<ConstantIntOp>()) {
        if (cst->getValue() == 0)
          return matchSuccess();
      }

    return matchFailure();
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.replaceSingleResultOp(op, op->getOperand(0));
  }
};
} // end anonymous namespace.

void AddIOp::getCanonicalizationPatterns(OwningPatternList &results,
                                         MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyAddX0>(context));
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

void AllocOp::build(Builder *builder, OperationState *result,
                    MemRefType *memrefType, ArrayRef<SSAValue *> operands) {
  result->addOperands(operands);
  result->types.push_back(memrefType);
}

void AllocOp::print(OpAsmPrinter *p) const {
  MemRefType *type = getType();
  *p << "alloc";
  // Print dynamic dimension operands.
  printDimAndSymbolList(operand_begin(), operand_end(),
                        type->getNumDynamicDims(), p);
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"map");
  *p << " : " << *type;
}

bool AllocOp::parse(OpAsmParser *parser, OperationState *result) {
  MemRefType *type;

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
  if (numDimOperands != type->getNumDynamicDims()) {
    return parser->emitError(parser->getNameLoc(),
                             "dimension operand count does not equal memref "
                             "dynamic dimension count");
  }
  result->types.push_back(type);
  return false;
}

bool AllocOp::verify() const {
  auto *memRefType = dyn_cast<MemRefType>(getResult()->getType());
  if (!memRefType)
    return emitOpError("result must be a memref");

  unsigned numSymbols = 0;
  if (!memRefType->getAffineMaps().empty()) {
    AffineMap affineMap = memRefType->getAffineMaps()[0];
    // Store number of symbols used in affine map (used in subsequent check).
    numSymbols = affineMap.getNumSymbols();
    // TODO(zinenko): this check does not belong to AllocOp, or any other op but
    // to the type system itself.  It has been partially hoisted to Parser but
    // remains here in case an AllocOp gets constructed programmatically.
    // Remove when we can emit errors directly from *Type::get(...) functions.
    //
    // Verify that the layout affine map matches the rank of the memref.
    if (affineMap.getNumDims() != memRefType->getRank())
      return emitOpError("affine map dimension count must equal memref rank");
  }
  unsigned numDynamicDims = memRefType->getNumDynamicDims();
  // Check that the total number of operands matches the number of symbols in
  // the affine map, plus the number of dynamic dimensions specified in the
  // memref type.
  if (getOperation()->getNumOperands() != numDynamicDims + numSymbols) {
    return emitOpError(
        "operand count does not equal dimension plus symbol operand count");
  }
  // Verify that all operands are of type Index.
  for (auto *operand : getOperands()) {
    if (!operand->getType()->isIndex())
      return emitOpError("requires operands to be of type Index");
  }
  return false;
}

namespace {
/// Fold constant dimensions into an alloc instruction.
struct SimplifyAllocConst : public Pattern {
  SimplifyAllocConst(MLIRContext *context)
      : Pattern(AllocOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto alloc = op->cast<AllocOp>();

    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    for (auto *operand : alloc->getOperands())
      if (auto *opOperation = operand->getDefiningOperation())
        if (opOperation->isa<ConstantIndexOp>())
          return matchSuccess();
    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto allocOp = op->cast<AllocOp>();
    auto memrefType = allocOp->getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType->getRank());
    SmallVector<SSAValue *, 4> newOperands;
    SmallVector<SSAValue *, 4> droppedOperands;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType->getRank(); dim < e; ++dim) {
      int dimSize = memrefType->getDimSize(dim);
      // If this is already static dimension, keep it.
      if (dimSize != -1) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = allocOp->getOperand(dynamicDimPos)->getDefiningOperation();
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
    auto *newMemRefType = MemRefType::get(
        newShapeConstants, memrefType->getElementType(),
        memrefType->getAffineMaps(), memrefType->getMemorySpace());
    assert(newOperands.size() == newMemRefType->getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc =
        rewriter.create<AllocOp>(allocOp->getLoc(), newMemRefType, newOperands);
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast = rewriter.create<MemRefCastOp>(allocOp->getLoc(), newAlloc,
                                                    allocOp->getType());

    rewriter.replaceSingleResultOp(op, resultCast, droppedOperands);
  }
};
} // end anonymous namespace.

void AllocOp::getCanonicalizationPatterns(OwningPatternList &results,
                                          MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyAllocConst>(context));
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(Builder *builder, OperationState *result, Function *callee,
                   ArrayRef<SSAValue *> operands) {
  result->addOperands(operands);
  result->addAttribute("callee", builder->getFunctionAttr(callee));
  result->addTypes(callee->getType()->getResults());
}

bool CallOp::parse(OpAsmParser *parser, OperationState *result) {
  StringRef calleeName;
  llvm::SMLoc calleeLoc;
  FunctionType *calleeType = nullptr;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  Function *callee = nullptr;
  if (parser->parseFunctionName(calleeName, calleeLoc) ||
      parser->parseOperandList(operands, /*requiredOperandCount=*/-1,
                               OpAsmParser::Delimiter::Paren) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(calleeType) ||
      parser->resolveFunctionName(calleeName, calleeType, calleeLoc, callee) ||
      parser->addTypesToList(calleeType->getResults(), result->types) ||
      parser->resolveOperands(operands, calleeType->getInputs(), calleeLoc,
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
  *p << " : " << *getCallee()->getType();
}

bool CallOp::verify() const {
  // Check that the callee attribute was specified.
  auto fnAttr = getAttrOfType<FunctionAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' function attribute");

  // Verify that the operand and result types match the callee.
  auto *fnType = fnAttr.getValue()->getType();
  if (fnType->getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType->getNumInputs(); i != e; ++i) {
    if (getOperand(i)->getType() != fnType->getInput(i))
      return emitOpError("operand type mismatch");
  }

  if (fnType->getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType->getNumResults(); i != e; ++i) {
    if (getResult(i)->getType() != fnType->getResult(i))
      return emitOpError("result type mismatch");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

void CallIndirectOp::build(Builder *builder, OperationState *result,
                           SSAValue *callee, ArrayRef<SSAValue *> operands) {
  auto *fnType = cast<FunctionType>(callee->getType());
  result->operands.push_back(callee);
  result->addOperands(operands);
  result->addTypes(fnType->getResults());
}

bool CallIndirectOp::parse(OpAsmParser *parser, OperationState *result) {
  FunctionType *calleeType = nullptr;
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
         parser->resolveOperands(operands, calleeType->getInputs(), operandsLoc,
                                 result->operands) ||
         parser->addTypesToList(calleeType->getResults(), result->types);
}

void CallIndirectOp::print(OpAsmPrinter *p) const {
  *p << "call_indirect ";
  p->printOperand(getCallee());
  *p << '(';
  auto operandRange = getOperands();
  p->printOperands(++operandRange.begin(), operandRange.end());
  *p << ')';
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"callee");
  *p << " : " << *getCallee()->getType();
}

bool CallIndirectOp::verify() const {
  // The callee must be a function.
  auto *fnType = dyn_cast<FunctionType>(getCallee()->getType());
  if (!fnType)
    return emitOpError("callee must have function type");

  // Verify that the operand and result types match the callee.
  if (fnType->getNumInputs() != getNumOperands() - 1)
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType->getNumInputs(); i != e; ++i) {
    if (getOperand(i + 1)->getType() != fnType->getInput(i))
      return emitOpError("operand type mismatch");
  }

  if (fnType->getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType->getNumResults(); i != e; ++i) {
    if (getResult(i)->getType() != fnType->getResult(i))
      return emitOpError("result type mismatch");
  }

  return false;
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

void DeallocOp::build(Builder *builder, OperationState *result,
                      SSAValue *memref) {
  result->addOperands(memref);
}

void DeallocOp::print(OpAsmPrinter *p) const {
  *p << "dealloc " << *getMemRef() << " : " << *getMemRef()->getType();
}

bool DeallocOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  MemRefType *type;

  return parser->parseOperand(memrefInfo) || parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands);
}

bool DeallocOp::verify() const {
  if (!isa<MemRefType>(getMemRef()->getType()))
    return emitOpError("operand must be a memref");
  return false;
}

void DeallocOp::getCanonicalizationPatterns(OwningPatternList &results,
                                            MLIRContext *context) {
  /// dealloc(memrefcast) -> dealloc
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::build(Builder *builder, OperationState *result,
                  SSAValue *memrefOrTensor, unsigned index) {
  result->addOperands(memrefOrTensor);
  result->addAttribute("index", builder->getIntegerAttr(index));
  result->types.push_back(builder->getIndexType());
}

void DimOp::print(OpAsmPrinter *p) const {
  *p << "dim " << *getOperand() << ", " << getIndex();
  p->printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/"index");
  *p << " : " << *getOperand()->getType();
}

bool DimOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  Type *type;

  return parser->parseOperand(operandInfo) || parser->parseComma() ||
         parser->parseAttribute(indexAttr, "index", result->attributes) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(operandInfo, type, result->operands) ||
         parser->addTypeToList(parser->getBuilder().getIndexType(),
                               result->types);
}

bool DimOp::verify() const {
  // Check that we have an integer index operand.
  auto indexAttr = getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return emitOpError("requires an integer attribute named 'index'");
  uint64_t index = (uint64_t)indexAttr.getValue();

  auto *type = getOperand()->getType();
  if (auto *tensorType = dyn_cast<RankedTensorType>(type)) {
    if (index >= tensorType->getRank())
      return emitOpError("index is out of range");
  } else if (auto *memrefType = dyn_cast<MemRefType>(type)) {
    if (index >= memrefType->getRank())
      return emitOpError("index is out of range");

  } else if (isa<UnrankedTensorType>(type)) {
    // ok, assumed to be in-range.
  } else {
    return emitOpError("requires an operand with tensor or memref type");
  }

  return false;
}

Attribute DimOp::constantFold(ArrayRef<Attribute> operands,
                              MLIRContext *context) const {
  // Constant fold dim when the size along the index referred to is a constant.
  auto *opType = getOperand()->getType();
  int indexSize = -1;
  if (auto *tensorType = dyn_cast<RankedTensorType>(opType)) {
    indexSize = tensorType->getShape()[getIndex()];
  } else if (auto *memrefType = dyn_cast<MemRefType>(opType)) {
    indexSize = memrefType->getShape()[getIndex()];
  }

  if (indexSize >= 0)
    return IntegerAttr::get(indexSize, context);

  return nullptr;
}

// ---------------------------------------------------------------------------
// DmaStartOp
// ---------------------------------------------------------------------------

void DmaStartOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << ' ' << *getSrcMemRef() << '[';
  p->printOperands(getSrcIndices());
  *p << "], " << *getDstMemRef() << '[';
  p->printOperands(getDstIndices());
  *p << "], " << *getNumElements();
  *p << ", " << *getTagMemRef() << '[';
  p->printOperands(getTagIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getSrcMemRef()->getType();
  *p << ", " << *getDstMemRef()->getType();
  *p << ", " << *getTagMemRef()->getType();
}

// Parse DmaStartOp.
// Ex:
//   %dma_id = dma_start %src[%i, %j], %dst[%k, %l], %size,
//                             %tag[%index] :
//    memref<3 x vector<8x128xf32>, (d0) -> (d0), 0>,
//    memref<1 x vector<8x128xf32>, (d0) -> (d0), 2>,
//    memref<1 x i32, (d0) -> (d0), 4>
//
bool DmaStartOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType srcMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> srcIndexInfos;
  OpAsmParser::OperandType dstMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> dstIndexInfos;
  OpAsmParser::OperandType numElementsInfo;
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> tagIndexInfos;

  SmallVector<Type *, 3> types;
  auto *indexType = parser->getBuilder().getIndexType();

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
                               OpAsmParser::Delimiter::Square) ||
      parser->parseColonTypeList(types))
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

  // Check that source/destination index list size matches associated rank.
  if (srcIndexInfos.size() != cast<MemRefType>(types[0])->getRank() ||
      dstIndexInfos.size() != cast<MemRefType>(types[1])->getRank())
    return parser->emitError(parser->getNameLoc(),
                             "memref rank not equal to indices count");

  if (tagIndexInfos.size() != cast<MemRefType>(types[2])->getRank())
    return parser->emitError(parser->getNameLoc(),
                             "tag memref rank not equal to indices count");

  return false;
}

void DmaStartOp::getCanonicalizationPatterns(OwningPatternList &results,
                                             MLIRContext *context) {
  /// dma_start(memrefcast) -> dma_start
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

// ---------------------------------------------------------------------------
// DmaWaitOp
// ---------------------------------------------------------------------------

void DmaWaitOp::print(OpAsmPrinter *p) const {
  *p << getOperationName() << ' ';
  // Print operands.
  p->printOperand(getTagMemRef());
  *p << '[';
  p->printOperands(getTagIndices());
  *p << "], ";
  p->printOperand(getNumElements());
  *p << " : " << *getTagMemRef()->getType();
}

// Parse DmaWaitOp.
// Eg:
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 4>
//
bool DmaWaitOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 2> tagIndexInfos;
  Type *type;
  auto *indexType = parser->getBuilder().getIndexType();
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

  if (tagIndexInfos.size() != cast<MemRefType>(type)->getRank())
    return parser->emitError(parser->getNameLoc(),
                             "tag memref rank not equal to indices count");

  return false;
}

void DmaWaitOp::getCanonicalizationPatterns(OwningPatternList &results,
                                            MLIRContext *context) {
  /// dma_wait(memrefcast) -> dma_wait
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

void ExtractElementOp::build(Builder *builder, OperationState *result,
                             SSAValue *aggregate,
                             ArrayRef<SSAValue *> indices) {
  auto *aggregateType = cast<VectorOrTensorType>(aggregate->getType());
  result->addOperands(aggregate);
  result->addOperands(indices);
  result->types.push_back(aggregateType->getElementType());
}

void ExtractElementOp::print(OpAsmPrinter *p) const {
  *p << "extract_element " << *getAggregate() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getAggregate()->getType();
}

bool ExtractElementOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType aggregateInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  VectorOrTensorType *type;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(aggregateInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(aggregateInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type->getElementType(), result->types);
}

bool ExtractElementOp::verify() const {
  if (getNumOperands() == 0)
    return emitOpError("expected an aggregate to index into");

  auto *aggregateType = dyn_cast<VectorOrTensorType>(getAggregate()->getType());
  if (!aggregateType)
    return emitOpError("first operand must be a vector or tensor");

  if (getType() != aggregateType->getElementType())
    return emitOpError("result type must match element type of aggregate");

  for (auto *idx : getIndices())
    if (!idx->getType()->isIndex())
      return emitOpError("index to extract_element must have 'index' type");

  // Verify the # indices match if we have a ranked type.
  auto aggregateRank = aggregateType->getRank();
  if (aggregateRank != -1 && aggregateRank != getNumOperands() - 1)
    return emitOpError("incorrect number of indices for extract_element");

  return false;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(Builder *builder, OperationState *result, SSAValue *memref,
                   ArrayRef<SSAValue *> indices) {
  auto *memrefType = cast<MemRefType>(memref->getType());
  result->addOperands(memref);
  result->addOperands(indices);
  result->types.push_back(memrefType->getElementType());
}

void LoadOp::print(OpAsmPrinter *p) const {
  *p << "load " << *getMemRef() << '[';
  p->printOperands(getIndices());
  *p << ']';
  p->printOptionalAttrDict(getAttrs());
  *p << " : " << *getMemRef()->getType();
}

bool LoadOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *type;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(type) ||
         parser->resolveOperand(memrefInfo, type, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
         parser->addTypeToList(type->getElementType(), result->types);
}

bool LoadOp::verify() const {
  if (getNumOperands() == 0)
    return emitOpError("expected a memref to load from");

  auto *memRefType = dyn_cast<MemRefType>(getMemRef()->getType());
  if (!memRefType)
    return emitOpError("first operand must be a memref");

  if (getType() != memRefType->getElementType())
    return emitOpError("result type must match element type of memref");

  if (memRefType->getRank() != getNumOperands() - 1)
    return emitOpError("incorrect number of indices for load");

  for (auto *idx : getIndices())
    if (!idx->getType()->isIndex())
      return emitOpError("index to load must have 'index' type");

  // TODO: Verify we have the right number of indices.

  // TODO: in MLFunction verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return false;
}

void LoadOp::getCanonicalizationPatterns(OwningPatternList &results,
                                         MLIRContext *context) {
  /// load(memrefcast) -> load
  results.push_back(
      std::make_unique<MemRefCastFolder>(getOperationName(), context));
}

//===----------------------------------------------------------------------===//
// MemRefCastOp
//===----------------------------------------------------------------------===//

bool MemRefCastOp::verify() const {
  auto *opType = dyn_cast<MemRefType>(getOperand()->getType());
  auto *resType = dyn_cast<MemRefType>(getType());
  if (!opType || !resType)
    return emitOpError("requires input and result types to be memrefs");

  if (opType == resType)
    return emitOpError("requires the input and result type to be different");

  if (opType->getElementType() != resType->getElementType())
    return emitOpError(
        "requires input and result element types to be the same");

  if (opType->getAffineMaps() != resType->getAffineMaps())
    return emitOpError("requires input and result mappings to be the same");

  if (opType->getMemorySpace() != resType->getMemorySpace())
    return emitOpError(
        "requires input and result memory spaces to be the same");

  // They must have the same rank, and any specified dimensions must match.
  if (opType->getRank() != resType->getRank())
    return emitOpError("requires input and result ranks to match");

  for (unsigned i = 0, e = opType->getRank(); i != e; ++i) {
    int opDim = opType->getDimSize(i), resultDim = resType->getDimSize(i);
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
  assert(operands.size() == 2 && "mulf takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>())
      return FloatAttr::get(lhs.getValue() * rhs.getValue(), context);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

Attribute MulIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  assert(operands.size() == 2 && "muli takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    // 0*x == 0
    if (lhs.getValue() == 0)
      return lhs;

    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>())
      // TODO: Handle the overflow case.
      return IntegerAttr::get(lhs.getValue() * rhs.getValue(), context);
  }

  // x*0 == 0
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>())
    if (rhs.getValue() == 0)
      return rhs;

  return nullptr;
}

namespace {
/// muli(x, 1) -> x
///
struct SimplifyMulX1 : public Pattern {
  SimplifyMulX1(MLIRContext *context)
      : Pattern(MulIOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto muli = op->cast<MulIOp>();
    if (auto *operandOp = muli->getOperand(1)->getDefiningOperation())
      // TODO: Support splatted one as well.  We need a general one pattern.
      if (auto cst = operandOp->dyn_cast<ConstantIntOp>()) {
        if (cst->getValue() == 1)
          return matchSuccess();
      }

    return matchFailure();
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.replaceSingleResultOp(op, op->getOperand(0));
  }
};
} // end anonymous namespace.

void MulIOp::getCanonicalizationPatterns(OwningPatternList &results,
                                         MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyMulX1>(context));
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(Builder *builder, OperationState *result,
                    SSAValue *valueToStore, SSAValue *memref,
                    ArrayRef<SSAValue *> indices) {
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
  *p << " : " << *getMemRef()->getType();
}

bool StoreOp::parse(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  MemRefType *memrefType;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return parser->parseOperand(storeValueInfo) || parser->parseComma() ||
         parser->parseOperand(memrefInfo) ||
         parser->parseOperandList(indexInfo, -1,
                                  OpAsmParser::Delimiter::Square) ||
         parser->parseOptionalAttributeDict(result->attributes) ||
         parser->parseColonType(memrefType) ||
         parser->resolveOperand(storeValueInfo, memrefType->getElementType(),
                                result->operands) ||
         parser->resolveOperand(memrefInfo, memrefType, result->operands) ||
         parser->resolveOperands(indexInfo, affineIntTy, result->operands);
}

bool StoreOp::verify() const {
  if (getNumOperands() < 2)
    return emitOpError("expected a value to store and a memref");

  // Second operand is a memref type.
  auto *memRefType = dyn_cast<MemRefType>(getMemRef()->getType());
  if (!memRefType)
    return emitOpError("second operand must be a memref");

  // First operand must have same type as memref element type.
  if (getValueToStore()->getType() != memRefType->getElementType())
    return emitOpError("first operand must have same type memref element type");

  if (getNumOperands() != 2 + memRefType->getRank())
    return emitOpError("store index operand count not equal to memref rank");

  for (auto *idx : getIndices())
    if (!idx->getType()->isIndex())
      return emitOpError("index to load must have 'index' type");

  // TODO: Verify we have the right number of indices.

  // TODO: in MLFunction verify that the indices are parameters, IV's, or the
  // result of an affine_apply.
  return false;
}

void StoreOp::getCanonicalizationPatterns(OwningPatternList &results,
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
  assert(operands.size() == 2 && "subf takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>())
      return FloatAttr::get(lhs.getValue() - rhs.getValue(), context);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

Attribute SubIOp::constantFold(ArrayRef<Attribute> operands,
                               MLIRContext *context) const {
  assert(operands.size() == 2 && "subi takes two operands");

  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>())
      return IntegerAttr::get(lhs.getValue() - rhs.getValue(), context);
  }

  return nullptr;
}

namespace {
/// subi(x,x) -> 0
///
struct SimplifyXMinusX : public Pattern {
  SimplifyXMinusX(MLIRContext *context)
      : Pattern(SubIOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto subi = op->cast<SubIOp>();
    if (subi->getOperand(0) == subi->getOperand(1))
      return matchSuccess();

    return matchFailure();
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto subi = op->cast<SubIOp>();
    auto result =
        rewriter.create<ConstantIntOp>(op->getLoc(), 0, subi->getType());

    rewriter.replaceSingleResultOp(op, result);
  }
};
} // end anonymous namespace.

void SubIOp::getCanonicalizationPatterns(OwningPatternList &results,
                                         MLIRContext *context) {
  results.push_back(std::make_unique<SimplifyXMinusX>(context));
}

//===----------------------------------------------------------------------===//
// TensorCastOp
//===----------------------------------------------------------------------===//

bool TensorCastOp::verify() const {
  auto *opType = dyn_cast<TensorType>(getOperand()->getType());
  auto *resType = dyn_cast<TensorType>(getType());
  if (!opType || !resType)
    return emitOpError("requires input and result types to be tensors");

  if (opType == resType)
    return emitOpError("requires the input and result type to be different");

  if (opType->getElementType() != resType->getElementType())
    return emitOpError(
        "requires input and result element types to be the same");

  // If the source or destination are unranked, then the cast is valid.
  auto *opRType = dyn_cast<RankedTensorType>(opType);
  auto *resRType = dyn_cast<RankedTensorType>(resType);
  if (!opRType || !resRType)
    return false;

  // If they are both ranked, they have to have the same rank, and any specified
  // dimensions must match.
  if (opRType->getRank() != resRType->getRank())
    return emitOpError("requires input and result ranks to match");

  for (unsigned i = 0, e = opRType->getRank(); i != e; ++i) {
    int opDim = opRType->getDimSize(i), resultDim = resRType->getDimSize(i);
    if (opDim != -1 && resultDim != -1 && opDim != resultDim)
      return emitOpError("requires static dimensions to match");
  }

  return false;
}
