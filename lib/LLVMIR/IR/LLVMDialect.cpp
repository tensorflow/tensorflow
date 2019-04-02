//===- LLVMDialect.cpp - LLVM IR Ops and Dialect registration -------------===//
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
// This file defines the types and operation details for the LLVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMTypeStorage : public ::mlir::TypeStorage {
  LLVMTypeStorage(llvm::Type *ty) : underlyingType(ty) {}

  // LLVM types are pointer-unique.
  using KeyTy = llvm::Type *;
  bool operator==(const KeyTy &key) const { return key == underlyingType; }

  static LLVMTypeStorage *construct(TypeStorageAllocator &allocator,
                                    llvm::Type *ty) {
    return new (allocator.allocate<LLVMTypeStorage>()) LLVMTypeStorage(ty);
  }

  llvm::Type *underlyingType;
};
} // end namespace detail
} // end namespace LLVM
} // end namespace mlir

LLVMType LLVMType::get(MLIRContext *context, llvm::Type *llvmType) {
  return Base::get(context, FIRST_LLVM_TYPE, llvmType);
}

llvm::Type *LLVMType::getUnderlyingType() const {
  return getImpl()->underlyingType;
}

static void printLLVMBinaryOp(OpAsmPrinter *p, Operation *op) {
  // Fallback to the generic form if the op is not well-formed (may happen
  // during incomplete rewrites, and used for debugging).
  const auto *abstract = op->getAbstractOperation();
  (void)abstract;
  assert(abstract && "pretty printing an unregistered operation");

  auto resultType = op->getResult(0)->getType();
  if (resultType != op->getOperand(0)->getType() ||
      resultType != op->getOperand(1)->getType())
    return p->printGenericOp(op);

  *p << op->getName().getStringRef() << ' ' << *op->getOperand(0) << ", "
     << *op->getOperand(1);
  p->printOptionalAttrDict(op->getAttrs());
  *p << " : " << op->getResult(0)->getType();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ICmpOp.
//===----------------------------------------------------------------------===//

// Return an array of mnemonics for ICmpPredicates indexed by its value.
static const char *const *getICmpPredicateNames() {
  static const char *predicateNames[]{/*EQ*/ "eq",
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

// Returns a value of the ICmp predicate corresponding to the given mnemonic.
// Returns -1 if there is no such mnemonic.
static int getICmpPredicateByName(StringRef name) {
  return llvm::StringSwitch<int>(name)
      .Case("eq", 0)
      .Case("ne", 1)
      .Case("slt", 2)
      .Case("sle", 3)
      .Case("sgt", 4)
      .Case("sge", 5)
      .Case("ult", 6)
      .Case("ule", 7)
      .Case("ugt", 8)
      .Case("uge", 9)
      .Default(-1);
}

static void printICmpOp(OpAsmPrinter *p, ICmpOp &op) {
  *p << op.getOperationName() << " \""
     << getICmpPredicateNames()[op.predicate().getZExtValue()] << "\" "
     << *op.getOperand(0) << ", " << *op.getOperand(1);
  p->printOptionalAttrDict(op.getAttrs(), {"predicate"});
  *p << " : " << op.lhs()->getType();
}

// <operation> ::= `llvm.icmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
static bool parseICmpOp(OpAsmParser *parser, OperationState *result) {
  Builder &builder = parser->getBuilder();

  Attribute predicate;
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType lhs, rhs;
  Type type;
  llvm::SMLoc predicateLoc, trailingTypeLoc;
  if (parser->getCurrentLocation(&predicateLoc) ||
      parser->parseAttribute(predicate, "predicate", attrs) ||
      parser->parseOperand(lhs) || parser->parseComma() ||
      parser->parseOperand(rhs) || parser->parseOptionalAttributeDict(attrs) ||
      parser->parseColon() || parser->getCurrentLocation(&trailingTypeLoc) ||
      parser->parseType(type) ||
      parser->resolveOperand(lhs, type, result->operands) ||
      parser->resolveOperand(rhs, type, result->operands))
    return true;

  // Replace the string attribute `predicate` with an integer attribute.
  auto predicateStr = predicate.dyn_cast<StringAttr>();
  if (!predicateStr)
    return parser->emitError(predicateLoc,
                             "expected 'predicate' attribute of string type");
  int predicateValue = getICmpPredicateByName(predicateStr.getValue());
  if (predicateValue == -1)
    return parser->emitError(
        predicateLoc,
        "'" + Twine(predicateStr.getValue()) +
            "' is an incorrect value of the 'predicate' attribute");

  attrs[0].second = parser->getBuilder().getI64IntegerAttr(predicateValue);

  // The result type is either i1 or a vector type <? x i1> if the inputs are
  // vectors.
  LLVMDialect *dialect = static_cast<LLVMDialect *>(
      builder.getContext()->getRegisteredDialect("llvm"));
  llvm::Type *llvmResultType = llvm::Type::getInt1Ty(dialect->getLLVMContext());
  auto argType = type.dyn_cast<LLVM::LLVMType>();
  if (!argType)
    return parser->emitError(trailingTypeLoc, "expected LLVM IR dialect type");
  if (argType.getUnderlyingType()->isVectorTy())
    llvmResultType = llvm::VectorType::get(
        llvmResultType, argType.getUnderlyingType()->getVectorNumElements());
  auto resultType = builder.getType<LLVM::LLVMType>(llvmResultType);

  result->attributes = attrs;
  result->addTypes({resultType});
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::AllocaOp.
//===----------------------------------------------------------------------===//

static void printAllocaOp(OpAsmPrinter *p, AllocaOp &op) {
  auto *llvmPtrTy = op.getType().cast<LLVM::LLVMType>().getUnderlyingType();
  auto *llvmElemTy = llvm::cast<llvm::PointerType>(llvmPtrTy)->getElementType();
  auto elemTy = LLVM::LLVMType::get(op.getContext(), llvmElemTy);

  auto funcTy = FunctionType::get({op.arraySize()->getType()}, {op.getType()},
                                  op.getContext());

  *p << op.getOperationName() << ' ' << *op.arraySize() << " x " << elemTy;
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << funcTy;
}

// <operation> ::= `llvm.alloca` ssa-use `x` type attribute-dict?
//                 `:` type `,` type
static bool parseAllocaOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType arraySize;
  Type type, elemType;
  llvm::SMLoc trailingTypeLoc;
  if (parser->parseOperand(arraySize) || parser->parseKeyword("x") ||
      parser->parseType(elemType) ||
      parser->parseOptionalAttributeDict(attrs) || parser->parseColon() ||
      parser->getCurrentLocation(&trailingTypeLoc) || parser->parseType(type))
    return true;

  // Extract the result type from the trailing function type.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumInputs() != 1 ||
      funcType.getNumResults() != 1)
    return parser->emitError(
        trailingTypeLoc,
        "expected trailing function type with one argument and one result");

  if (parser->resolveOperand(arraySize, funcType.getInput(0), result->operands))
    return true;

  result->attributes = attrs;
  result->addTypes({funcType.getResult(0)});
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::GEPOp.
//===----------------------------------------------------------------------===//

static void printGEPOp(OpAsmPrinter *p, GEPOp &op) {
  SmallVector<Type, 8> types;
  for (auto *operand : op.getOperands())
    types.push_back(operand->getType());
  auto funcTy =
      FunctionType::get(types, op.getResult()->getType(), op.getContext());

  *p << op.getOperationName() << ' ' << *op.base() << '[';
  p->printOperands(std::next(op.operand_begin()), op.operand_end());
  *p << ']';
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << funcTy;
}

// <operation> ::= `llvm.getelementptr` ssa-use `[` ssa-use-list `]`
//                 attribute-dict? `:` type
static bool parseGEPOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType base;
  SmallVector<OpAsmParser::OperandType, 8> indices;
  Type type;
  llvm::SMLoc trailingTypeLoc;
  if (parser->parseOperand(base) ||
      parser->parseOperandList(indices, /*requiredOperandCount=*/-1,
                               OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(attrs) || parser->parseColon() ||
      parser->getCurrentLocation(&trailingTypeLoc) || parser->parseType(type))
    return true;

  // Deconstruct the trailing function type to extract the types of the base
  // pointer and result (same type) and the types of the indices.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumResults() != 1 ||
      funcType.getNumInputs() == 0)
    return parser->emitError(trailingTypeLoc,
                             "expected trailing function type with at least "
                             "one argument and one result");

  if (parser->resolveOperand(base, funcType.getInput(0), result->operands) ||
      parser->resolveOperands(indices, funcType.getInputs().drop_front(),
                              parser->getNameLoc(), result->operands))
    return true;

  result->attributes = attrs;
  result->addTypes(funcType.getResults());
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::LoadOp.
//===----------------------------------------------------------------------===//

static void printLoadOp(OpAsmPrinter *p, LoadOp &op) {
  *p << op.getOperationName() << ' ' << *op.addr();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.addr()->getType();
}

// Extract the pointee type from the LLVM pointer type wrapped in MLIR.  Return
// the resulting type wrapped in MLIR, or nullptr on error.
static Type getLoadStoreElementType(OpAsmParser *parser, Type type,
                                    llvm::SMLoc trailingTypeLoc) {
  auto llvmTy = type.dyn_cast<LLVM::LLVMType>();
  if (!llvmTy)
    return parser->emitError(trailingTypeLoc, "expected LLVM IR dialect type"),
           nullptr;
  auto *llvmPtrTy = dyn_cast<llvm::PointerType>(llvmTy.getUnderlyingType());
  if (!llvmPtrTy)
    return parser->emitError(trailingTypeLoc, "expected LLVM pointer type"),
           nullptr;
  auto elemTy = LLVM::LLVMType::get(parser->getBuilder().getContext(),
                                    llvmPtrTy->getElementType());
  return elemTy;
}

// <operation> ::= `llvm.load` ssa-use attribute-dict? `:` type
static bool parseLoadOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType addr;
  Type type;
  llvm::SMLoc trailingTypeLoc;

  if (parser->parseOperand(addr) || parser->parseOptionalAttributeDict(attrs) ||
      parser->parseColon() || parser->getCurrentLocation(&trailingTypeLoc) ||
      parser->parseType(type) ||
      parser->resolveOperand(addr, type, result->operands))
    return true;

  Type elemTy = getLoadStoreElementType(parser, type, trailingTypeLoc);

  result->attributes = attrs;
  result->addTypes(elemTy);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::StoreOp.
//===----------------------------------------------------------------------===//

static void printStoreOp(OpAsmPrinter *p, StoreOp &op) {
  *p << op.getOperationName() << ' ' << *op.value() << ", " << *op.addr();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.addr()->getType();
}

// <operation> ::= `llvm.store` ssa-use `,` ssa-use attribute-dict? `:` type
static bool parseStoreOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType addr, value;
  Type type;
  llvm::SMLoc trailingTypeLoc;

  if (parser->parseOperand(value) || parser->parseComma() ||
      parser->parseOperand(addr) || parser->parseOptionalAttributeDict(attrs) ||
      parser->parseColon() || parser->getCurrentLocation(&trailingTypeLoc) ||
      parser->parseType(type))
    return true;

  Type elemTy = getLoadStoreElementType(parser, type, trailingTypeLoc);
  if (!elemTy)
    return true;

  if (parser->resolveOperand(value, elemTy, result->operands) ||
      parser->resolveOperand(addr, type, result->operands))
    return true;

  result->attributes = attrs;
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::BitcastOp.
//===----------------------------------------------------------------------===//

static void printBitcastOp(OpAsmPrinter *p, BitcastOp &op) {
  *p << op.getOperationName() << ' ' << *op.arg();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.arg()->getType() << " to " << op.getType();
}

// <operation> ::= `llvm.bitcast` ssa-use attribute-dict? `:` type `to` type
static bool parseBitcastOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType arg;
  Type sourceType, type;

  if (parser->parseOperand(arg) || parser->parseOptionalAttributeDict(attrs) ||
      parser->parseColonType(sourceType) || parser->parseKeyword("to") ||
      parser->parseType(type) ||
      parser->resolveOperand(arg, sourceType, result->operands))
    return true;

  result->attributes = attrs;
  result->addTypes(type);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::CallOp.
//===----------------------------------------------------------------------===//

static void printCallOp(OpAsmPrinter *p, CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();

  // Print the direct callee if present as a function attribute, or an indirect
  // callee (first operand) otherwise.
  *p << op.getOperationName() << ' ';
  if (isDirect)
    *p << '@' << callee.getValue()->getName().strref();
  else
    *p << *op.getOperand(0);

  *p << '(';
  p->printOperands(std::next(op.operand_begin(), callee.hasValue() ? 0 : 1),
                   op.operand_end());
  *p << ')';

  p->printOptionalAttrDict(op.getAttrs(), {"callee"});

  if (isDirect) {
    *p << " : " << callee.getValue()->getType();
    return;
  }

  // Reconstruct the function MLIR function type from LLVM function type,
  // and print it.
  auto operandType = op.getOperand(0)->getType().cast<LLVM::LLVMType>();
  auto *llvmPtrType =
      dyn_cast<llvm::PointerType>(operandType.getUnderlyingType());
  assert(llvmPtrType &&
         "operand #0 must have LLVM pointer type for indirect calls");
  auto *llvmType = dyn_cast<llvm::FunctionType>(llvmPtrType->getElementType());
  assert(llvmType &&
         "operand #0 must have LLVM Function pointer type for indirect calls");

  auto *llvmResultType = llvmType->getReturnType();
  SmallVector<Type, 1> resultTypes;
  if (!llvmResultType->isVoidTy())
    resultTypes.push_back(LLVM::LLVMType::get(op.getContext(), llvmResultType));

  SmallVector<Type, 8> argTypes;
  argTypes.reserve(llvmType->getNumParams());
  for (int i = 0, e = llvmType->getNumParams(); i < e; ++i)
    argTypes.push_back(
        LLVM::LLVMType::get(op.getContext(), llvmType->getParamType(i)));

  *p << " : " << FunctionType::get(argTypes, resultTypes, op.getContext());
}

// <operation> ::= `llvm.call` (function-id | ssa-use) `(` ssa-use-list `)`
//                 attribute-dict? `:` function-type
static bool parseCallOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  SmallVector<OpAsmParser::OperandType, 8> operands;
  Type type;
  StringRef calleeName;
  llvm::SMLoc calleeLoc, trailingTypeLoc;

  // Parse an operand list that will, in practice, contain 0 or 1 operand.  In
  // case of an indirect call, there will be 1 operand before `(`.  In case of a
  // direct call, there will be no operands and the parser will stop at the
  // function identifier without complaining.
  if (parser->parseOperandList(operands))
    return true;
  bool isDirect = operands.empty();

  // Optionally parse a function identifier.
  if (isDirect)
    if (parser->parseFunctionName(calleeName, calleeLoc))
      return true;

  if (parser->parseOperandList(operands, /*requiredOperandCount=*/-1,
                               OpAsmParser::Delimiter::Paren) ||
      parser->parseOptionalAttributeDict(attrs) || parser->parseColon() ||
      parser->getCurrentLocation(&trailingTypeLoc) || parser->parseType(type))
    return true;

  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType)
    return parser->emitError(trailingTypeLoc, "expected function type");
  if (isDirect) {
    // Add the direct callee as an Op attribute.
    Function *func;
    if (parser->resolveFunctionName(calleeName, funcType, calleeLoc, func))
      return true;
    auto funcAttr = parser->getBuilder().getFunctionAttr(func);
    attrs.push_back(parser->getBuilder().getNamedAttr("callee", funcAttr));

    // Make sure types match.
    if (parser->resolveOperands(operands, funcType.getInputs(),
                                parser->getNameLoc(), result->operands))
      return true;
    result->addTypes(funcType.getResults());
  } else {
    // Construct the LLVM IR Dialect function type that the first operand
    // should match.
    if (funcType.getNumResults() > 1)
      return parser->emitError(trailingTypeLoc,
                               "expected function with 0 or 1 result");

    Builder &builder = parser->getBuilder();
    auto *llvmDialect = static_cast<LLVM::LLVMDialect *>(
        builder.getContext()->getRegisteredDialect("llvm"));
    llvm::Type *llvmResultType;
    Type wrappedResultType;
    if (funcType.getNumResults() == 0) {
      llvmResultType = llvm::Type::getVoidTy(llvmDialect->getLLVMContext());
      wrappedResultType = builder.getType<LLVM::LLVMType>(llvmResultType);
    } else {
      wrappedResultType = funcType.getResult(0);
      auto wrappedLLVMResultType = wrappedResultType.dyn_cast<LLVM::LLVMType>();
      if (!wrappedLLVMResultType)
        return parser->emitError(trailingTypeLoc,
                                 "expected result to have LLVM type");
      llvmResultType = wrappedLLVMResultType.getUnderlyingType();
    }

    SmallVector<llvm::Type *, 8> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    for (int i = 0, e = funcType.getNumInputs(); i < e; ++i) {
      auto argType = funcType.getInput(i).dyn_cast<LLVM::LLVMType>();
      if (!argType)
        return parser->emitError(trailingTypeLoc,
                                 "expected LLVM types as inputs");
      argTypes.push_back(argType.getUnderlyingType());
    }
    auto *llvmFuncType = llvm::FunctionType::get(llvmResultType, argTypes,
                                                 /*isVarArg=*/false);
    auto wrappedFuncType =
        builder.getType<LLVM::LLVMType>(llvmFuncType->getPointerTo());

    auto funcArguments =
        ArrayRef<OpAsmParser::OperandType>(operands).drop_front();

    // Make sure that the first operand (indirect callee) matches the wrapped
    // LLVM IR function type, and that the types of the other call operands
    // match the types of the function arguments.
    if (parser->resolveOperand(operands[0], wrappedFuncType,
                               result->operands) ||
        parser->resolveOperands(funcArguments, funcType.getInputs(),
                                parser->getNameLoc(), result->operands))
      return true;

    result->addTypes(wrappedResultType);
  }

  result->attributes = attrs;
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractValueOp.
//===----------------------------------------------------------------------===//

static void printExtractValueOp(OpAsmPrinter *p, ExtractValueOp &op) {
  *p << op.getOperationName() << ' ' << *op.container() << op.position();
  p->printOptionalAttrDict(op.getAttrs(), {"position"});
  *p << " : " << op.container()->getType();
}

// Extract the type at `position` in the wrapped LLVM IR aggregate type
// `containerType`.  Position is an integer array attribute where each value
// is a zero-based position of the element in the aggregate type.  Return the
// resulting type wrapped in MLIR, or nullptr on error.
static LLVM::LLVMType getInsertExtractValueElementType(OpAsmParser *parser,
                                                       Type containerType,
                                                       Attribute positionAttr,
                                                       llvm::SMLoc attributeLoc,
                                                       llvm::SMLoc typeLoc) {
  auto wrappedContainerType = containerType.dyn_cast<LLVM::LLVMType>();
  if (!wrappedContainerType)
    return parser->emitError(typeLoc, "expected LLVM IR Dialect type"), nullptr;

  auto positionArrayAttr = positionAttr.dyn_cast<ArrayAttr>();
  if (!positionArrayAttr)
    return parser->emitError(attributeLoc, "expected an array attribute"),
           nullptr;

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // stuctures.  Check the position index before accessing, it is supposed to be
  // in bounds.
  llvm::Type *llvmContainerType = wrappedContainerType.getUnderlyingType();
  for (Attribute subAttr : positionArrayAttr) {
    auto positionElementAttr = subAttr.dyn_cast<IntegerAttr>();
    if (!positionElementAttr)
      return parser->emitError(attributeLoc,
                               "expected an array of integer literals"),
             nullptr;
    int position = positionElementAttr.getInt();
    if (llvmContainerType->isArrayTy()) {
      if (position < 0 || position >= llvmContainerType->getArrayNumElements())
        return parser->emitError(attributeLoc, "position out of bounds"),
               nullptr;
      llvmContainerType = llvmContainerType->getArrayElementType();
    } else if (llvmContainerType->isStructTy()) {
      if (position < 0 || position >= llvmContainerType->getStructNumElements())
        return parser->emitError(attributeLoc, "position out of bounds"),
               nullptr;
      llvmContainerType = llvmContainerType->getStructElementType(position);
    } else {
      return parser->emitError(typeLoc,
                               "expected wrapped LLVM IR structure/array type"),
             nullptr;
    }
  }

  Builder &builder = parser->getBuilder();
  return builder.getType<LLVM::LLVMType>(llvmContainerType);
}

// <operation> ::= `llvm.extractvalue` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static bool parseExtractValueOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType container;
  Type containerType;
  Attribute positionAttr;
  llvm::SMLoc attributeLoc, trailingTypeLoc;

  if (parser->parseOperand(container) ||
      parser->getCurrentLocation(&attributeLoc) ||
      parser->parseAttribute(positionAttr, "position", attrs) ||
      parser->parseOptionalAttributeDict(attrs) || parser->parseColon() ||
      parser->getCurrentLocation(&trailingTypeLoc) ||
      parser->parseType(containerType) ||
      parser->resolveOperand(container, containerType, result->operands))
    return true;

  auto elementType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!elementType)
    return true;

  result->attributes = attrs;
  result->addTypes(elementType);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertValueOp.
//===----------------------------------------------------------------------===//

static void printInsertValueOp(OpAsmPrinter *p, InsertValueOp &op) {
  *p << op.getOperationName() << ' ' << *op.value() << ", " << *op.container()
     << op.position();
  p->printOptionalAttrDict(op.getAttrs(), {"position"});
  *p << " : " << op.container()->getType();
}

// <operation> ::= `llvm.insertvaluevalue` ssa-use `,` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static bool parseInsertValueOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType container, value;
  Type containerType;
  Attribute positionAttr;
  llvm::SMLoc attributeLoc, trailingTypeLoc;

  if (parser->parseOperand(value) || parser->parseComma() ||
      parser->parseOperand(container) ||
      parser->getCurrentLocation(&attributeLoc) ||
      parser->parseAttribute(positionAttr, "position", result->attributes) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColon() || parser->getCurrentLocation(&trailingTypeLoc) ||
      parser->parseType(containerType))
    return true;

  auto valueType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!valueType)
    return true;

  if (parser->resolveOperand(container, containerType, result->operands) ||
      parser->resolveOperand(value, valueType, result->operands))
    return true;

  result->addTypes(containerType);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::SelectOp.
//===----------------------------------------------------------------------===//

static void printSelectOp(OpAsmPrinter *p, SelectOp &op) {
  *p << op.getOperationName() << ' ' << *op.condition() << ", "
     << *op.trueValue() << ", " << *op.falseValue();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.condition()->getType() << ", " << op.trueValue()->getType();
}

// <operation> ::= `llvm.select` ssa-use `,` ssa-use `,` ssa-use
//                 attribute-dict? `:` type, type
static bool parseSelectOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType condition, trueValue, falseValue;
  Type conditionType, argType;

  if (parser->parseOperand(condition) || parser->parseComma() ||
      parser->parseOperand(trueValue) || parser->parseComma() ||
      parser->parseOperand(falseValue) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(conditionType) || parser->parseComma() ||
      parser->parseType(argType))
    return true;

  if (parser->resolveOperand(condition, conditionType, result->operands) ||
      parser->resolveOperand(trueValue, argType, result->operands) ||
      parser->resolveOperand(falseValue, argType, result->operands))
    return true;

  result->addTypes(argType);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::BrOp.
//===----------------------------------------------------------------------===//

static void printBrOp(OpAsmPrinter *p, BrOp &op) {
  *p << op.getOperationName() << ' ';
  p->printSuccessorAndUseList(op.getOperation(), 0);
  p->printOptionalAttrDict(op.getAttrs());
}

// <operation> ::= `llvm.br` bb-id (`[` ssa-use-and-type-list `]`)?
// attribute-dict?
static bool parseBrOp(OpAsmParser *parser, OperationState *result) {
  Block *dest;
  SmallVector<Value *, 4> operands;
  if (parser->parseSuccessorAndUseList(dest, operands) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;

  result->addSuccessor(dest, operands);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::CondBrOp.
//===----------------------------------------------------------------------===//

static void printCondBrOp(OpAsmPrinter *p, CondBrOp &op) {
  *p << op.getOperationName() << ' ' << *op.getOperand(0) << ", ";
  p->printSuccessorAndUseList(op.getOperation(), 0);
  *p << ", ";
  p->printSuccessorAndUseList(op.getOperation(), 1);
  p->printOptionalAttrDict(op.getAttrs());
}

// <operation> ::= `llvm.cond_br` ssa-use `,`
//                  bb-id (`[` ssa-use-and-type-list `]`)? `,`
//                  bb-id (`[` ssa-use-and-type-list `]`)? attribute-dict?
static bool parseCondBrOp(OpAsmParser *parser, OperationState *result) {
  Block *trueDest;
  Block *falseDest;
  SmallVector<Value *, 4> trueOperands;
  SmallVector<Value *, 4> falseOperands;
  OpAsmParser::OperandType condition;

  Builder &builder = parser->getBuilder();
  auto *llvmDialect = static_cast<LLVM::LLVMDialect *>(
      builder.getContext()->getRegisteredDialect("llvm"));
  auto i1Type = builder.getType<LLVM::LLVMType>(
      llvm::Type::getInt1Ty(llvmDialect->getLLVMContext()));

  if (parser->parseOperand(condition) || parser->parseComma() ||
      parser->parseSuccessorAndUseList(trueDest, trueOperands) ||
      parser->parseComma() ||
      parser->parseSuccessorAndUseList(falseDest, falseOperands) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->resolveOperand(condition, i1Type, result->operands))
    return true;

  result->addSuccessor(trueDest, trueOperands);
  result->addSuccessor(falseDest, falseOperands);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ReturnOp.
//===----------------------------------------------------------------------===//

static void printReturnOp(OpAsmPrinter *p, ReturnOp &op) {
  *p << op.getOperationName();
  p->printOptionalAttrDict(op.getAttrs());
  assert(op.getNumOperands() <= 1);

  if (op.getNumOperands() == 0)
    return;

  *p << ' ' << *op.getOperand(0) << " : " << op.getOperand(0)->getType();
}

// <operation> ::= `llvm.return` ssa-use-list attribute-dict? `:`
//                 type-list-no-parens
static bool parseReturnOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 1> operands;
  Type type;

  if (parser->parseOperandList(operands) ||
      parser->parseOptionalAttributeDict(result->attributes))
    return true;
  if (operands.empty())
    return false;

  if (parser->parseColonType(type) ||
      parser->resolveOperand(operands[0], type, result->operands))
    return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::UndefOp.
//===----------------------------------------------------------------------===//

static void printUndefOp(OpAsmPrinter *p, UndefOp &op) {
  *p << op.getOperationName();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.res()->getType();
}

// <operation> ::= `llvm.undef` attribute-dict? : type
static bool parseUndefOp(OpAsmParser *parser, OperationState *result) {
  Type type;

  if (parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  result->addTypes(type);
  return false;
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ConstantOp.
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter *p, ConstantOp &op) {
  *p << op.getOperationName() << '(' << op.value();
  // Print attribute types other than i64 and f64 because attribute parsing will
  // assume those in absence of explicit attribute type.
  if (auto intAttr = op.value().dyn_cast<IntegerAttr>()) {
    auto type = intAttr.getType();
    if (!type.isInteger(64))
      *p << " : " << intAttr.getType();
  } else if (auto floatAttr = op.value().dyn_cast<FloatAttr>()) {
    auto type = floatAttr.getType();
    if (!type.isF64())
      *p << " : " << type;
  }
  *p << ')';
  p->printOptionalAttrDict(op.getAttrs(), {"value"});
  *p << " : " << op.res()->getType();
}

// <operation> ::= `llvm.constant` `(` attribute `)` attribute-list? : type
static bool parseConstantOp(OpAsmParser *parser, OperationState *result) {
  Attribute valueAttr;
  Type type;

  if (parser->parseLParen() ||
      parser->parseAttribute(valueAttr, "value", result->attributes) ||
      parser->parseRParen() ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type))
    return true;

  result->addTypes(type);
  return false;
}

//===----------------------------------------------------------------------===//
// LLVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

LLVMDialect::LLVMDialect(MLIRContext *context)
    : Dialect("llvm", context), module("LLVMDialectModule", llvmContext) {
  addTypes<LLVMType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/LLVMIR/LLVMOps.cpp.inc"
      >();

  // Support unknown operations because not all LLVM operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/LLVMIR/LLVMOps.cpp.inc"

/// Parse a type registered to this dialect.
Type LLVMDialect::parseType(StringRef tyData, Location loc) const {
  llvm::SMDiagnostic errorMessage;
  llvm::Type *type = llvm::parseType(tyData, errorMessage, module);
  if (!type)
    return (getContext()->emitError(loc, errorMessage.getMessage()), nullptr);
  return LLVMType::get(getContext(), type);
}

/// Print a type registered to this dialect.
void LLVMDialect::printType(Type type, raw_ostream &os) const {
  auto llvmType = type.dyn_cast<LLVMType>();
  assert(llvmType && "printing wrong type");
  assert(llvmType.getUnderlyingType() && "no underlying LLVM type");
  llvmType.getUnderlyingType()->print(os);
}

/// Verify LLVMIR function argument attributes.
LogicalResult LLVMDialect::verifyFunctionArgAttribute(Function *func,
                                                      unsigned argIdx,
                                                      NamedAttribute argAttr) {
  // Check that llvm.noalias is a boolean attribute.
  if (argAttr.first == "llvm.noalias" && !argAttr.second.isa<BoolAttr>())
    return func->emitError(
        "llvm.noalias argument attribute of non boolean type");
  return success();
}

static DialectRegistration<LLVMDialect> llvmDialect;
