//===- SPIRVOps.cpp - MLIR SPIR-V operations ------------------------------===//
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
// This file defines the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVOps.h"

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/StringExtras.h"

using namespace mlir;

// TODO(antiagainst): generate these strings using ODS.
static constexpr const char kAlignmentAttrName[] = "alignment";
static constexpr const char kBranchWeightAttrName[] = "branch_weights";
static constexpr const char kDefaultValueAttrName[] = "default_value";
static constexpr const char kFnNameAttrName[] = "fn";
static constexpr const char kIndicesAttrName[] = "indices";
static constexpr const char kInitializerAttrName[] = "initializer";
static constexpr const char kInterfaceAttrName[] = "interface";
static constexpr const char kSpecConstAttrName[] = "spec_const";
static constexpr const char kTypeAttrName[] = "type";
static constexpr const char kValueAttrName[] = "value";
static constexpr const char kValuesAttrName[] = "values";
static constexpr const char kVariableAttrName[] = "variable";

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

template <typename Dst, typename Src>
inline Dst bitwiseCast(Src source) noexcept {
  Dst dest;
  static_assert(sizeof(source) == sizeof(dest),
                "bitwiseCast requires same source and destination bitwidth");
  std::memcpy(&dest, &source, sizeof(dest));
  return dest;
}

static LogicalResult extractValueFromConstOp(Operation *op,
                                             int32_t &indexValue) {
  auto constOp = dyn_cast<spirv::ConstantOp>(op);
  if (!constOp) {
    return failure();
  }
  auto valueAttr = constOp.value();
  auto integerValueAttr = valueAttr.dyn_cast<IntegerAttr>();
  if (!integerValueAttr) {
    return failure();
  }
  indexValue = integerValueAttr.getInt();
  return success();
}

static ParseResult parseBinaryLogicalOp(OpAsmParser *parser,
                                        OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  if (parser->parseOperandList(ops, 2) || parser->parseColonType(type) ||
      parser->resolveOperands(ops, type, result->operands)) {
    return failure();
  }
  // Result must be a scalar or vector of boolean type.
  Type resultType = parser->getBuilder().getIntegerType(1);
  if (auto opsType = type.dyn_cast<VectorType>()) {
    resultType = VectorType::get(opsType.getNumElements(), resultType);
  }
  result->addTypes(resultType);
  return success();
}

template <typename EnumClass>
static ParseResult parseEnumAttribute(EnumClass &value, OpAsmParser *parser) {
  Attribute attrVal;
  SmallVector<NamedAttribute, 1> attr;
  auto loc = parser->getCurrentLocation();
  if (parser->parseAttribute(attrVal, parser->getBuilder().getNoneType(),
                             spirv::attributeName<EnumClass>(), attr)) {
    return failure();
  }
  if (!attrVal.isa<StringAttr>()) {
    return parser->emitError(loc, "expected ")
           << spirv::attributeName<EnumClass>()
           << " attribute specified as string";
  }
  auto attrOptional =
      spirv::symbolizeEnum<EnumClass>()(attrVal.cast<StringAttr>().getValue());
  if (!attrOptional) {
    return parser->emitError(loc, "invalid ")
           << spirv::attributeName<EnumClass>()
           << " attribute specification: " << attrVal;
  }
  value = attrOptional.getValue();
  return success();
}

template <typename EnumClass>
static ParseResult parseEnumAttribute(EnumClass &value, OpAsmParser *parser,
                                      OperationState *state) {
  if (parseEnumAttribute(value, parser)) {
    return failure();
  }
  state->addAttribute(
      spirv::attributeName<EnumClass>(),
      parser->getBuilder().getI32IntegerAttr(bitwiseCast<int32_t>(value)));
  return success();
}

static ParseResult parseMemoryAccessAttributes(OpAsmParser *parser,
                                               OperationState *state) {
  // Parse an optional list of attributes staring with '['
  if (parser->parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  if (parseEnumAttribute(memoryAccessAttr, parser, state)) {
    return failure();
  }

  if (memoryAccessAttr == spirv::MemoryAccess::Aligned) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    Type i32Type = parser->getBuilder().getIntegerType(32);
    if (parser->parseComma() ||
        parser->parseAttribute(alignmentAttr, i32Type, kAlignmentAttrName,
                               state->attributes)) {
      return failure();
    }
  }
  return parser->parseRSquare();
}

// Parses an op that has no inputs and no outputs.
static ParseResult parseNoIOOp(OpAsmParser *parser, OperationState *state) {
  if (parser->parseOptionalAttributeDict(state->attributes))
    return failure();
  return success();
}

static void printBinaryLogicalOp(Operation *logicalOp, OpAsmPrinter *printer) {
  *printer << logicalOp->getName() << ' ' << *logicalOp->getOperand(0) << ", "
           << *logicalOp->getOperand(1);
  *printer << " : " << logicalOp->getOperand(0)->getType();
}

template <typename LoadStoreOpTy>
static void
printMemoryAccessAttribute(LoadStoreOpTy loadStoreOp, OpAsmPrinter *printer,
                           SmallVectorImpl<StringRef> &elidedAttrs) {
  // Print optional memory access attribute.
  if (auto memAccess = loadStoreOp.memory_access()) {
    elidedAttrs.push_back(spirv::attributeName<spirv::MemoryAccess>());
    *printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    // Print integer alignment attribute.
    if (auto alignment = loadStoreOp.alignment()) {
      elidedAttrs.push_back(kAlignmentAttrName);
      *printer << ", " << alignment;
    }
    *printer << "]";
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
}

template <typename LoadStoreOpTy>
static LogicalResult verifyMemoryAccessAttribute(LoadStoreOpTy loadStoreOp) {
  // ODS checks for attributes values. Just need to verify that if the
  // memory-access attribute is Aligned, then the alignment attribute must be
  // present.
  auto *op = loadStoreOp.getOperation();
  auto memAccessAttr = op->getAttr(spirv::attributeName<spirv::MemoryAccess>());
  if (!memAccessAttr) {
    // Alignment attribute shouldn't be present if memory access attribute is
    // not present.
    if (op->getAttr(kAlignmentAttrName)) {
      return loadStoreOp.emitOpError(
          "invalid alignment specification without aligned memory access "
          "specification");
    }
    return success();
  }

  auto memAccessVal = memAccessAttr.template cast<IntegerAttr>();
  auto memAccess = spirv::symbolizeMemoryAccess(memAccessVal.getInt());

  if (!memAccess) {
    return loadStoreOp.emitOpError("invalid memory access specifier: ")
           << memAccessVal;
  }

  if (*memAccess == spirv::MemoryAccess::Aligned) {
    if (!op->getAttr(kAlignmentAttrName)) {
      return loadStoreOp.emitOpError("missing alignment value");
    }
  } else {
    if (op->getAttr(kAlignmentAttrName)) {
      return loadStoreOp.emitOpError(
          "invalid alignment specification with non-aligned memory access "
          "specification");
    }
  }
  return success();
}

template <typename LoadStoreOpTy>
static LogicalResult verifyLoadStorePtrAndValTypes(LoadStoreOpTy op, Value *ptr,
                                                   Value *val) {
  // ODS already checks ptr is spirv::PointerType. Just check that the pointee
  // type of the pointer and the type of the value are the same
  //
  // TODO(ravishankarm): Check that the value type satisfies restrictions of
  // SPIR-V OpLoad/OpStore operations
  if (val->getType() !=
      ptr->getType().cast<spirv::PointerType>().getPointeeType()) {
    return op.emitOpError("mismatch in result type and pointer type");
  }
  return success();
}

// Prints an op that has no inputs and no outputs.
static void printNoIOOp(Operation *op, OpAsmPrinter *printer) {
  *printer << op->getName();
  printer->printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseVariableDecorations(OpAsmParser *parser,
                                            OperationState *state) {
  auto builtInName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn));
  if (succeeded(parser->parseOptionalKeyword("bind"))) {
    Attribute set, binding;
    // Parse optional descriptor binding
    auto descriptorSetName = convertToSnakeCase(
        stringifyDecoration(spirv::Decoration::DescriptorSet));
    auto bindingName =
        convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
    Type i32Type = parser->getBuilder().getIntegerType(32);
    if (parser->parseLParen() ||
        parser->parseAttribute(set, i32Type, descriptorSetName,
                               state->attributes) ||
        parser->parseComma() ||
        parser->parseAttribute(binding, i32Type, bindingName,
                               state->attributes) ||
        parser->parseRParen()) {
      return failure();
    }
  } else if (succeeded(parser->parseOptionalKeyword(builtInName.c_str()))) {
    StringAttr builtIn;
    if (parser->parseLParen() ||
        parser->parseAttribute(builtIn, Type(), builtInName,
                               state->attributes) ||
        parser->parseRParen()) {
      return failure();
    }
  }

  // Parse other attributes
  if (parser->parseOptionalAttributeDict(state->attributes))
    return failure();

  return success();
}

static void printVariableDecorations(Operation *op, OpAsmPrinter *printer,
                                     SmallVectorImpl<StringRef> &elidedAttrs) {
  // Print optional descriptor binding
  auto descriptorSetName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
  auto descriptorSet = op->getAttrOfType<IntegerAttr>(descriptorSetName);
  auto binding = op->getAttrOfType<IntegerAttr>(bindingName);
  if (descriptorSet && binding) {
    elidedAttrs.push_back(descriptorSetName);
    elidedAttrs.push_back(bindingName);
    *printer << " bind(" << descriptorSet.getInt() << ", " << binding.getInt()
             << ")";
  }

  // Print BuiltIn attribute if present
  auto builtInName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn));
  if (auto builtin = op->getAttrOfType<StringAttr>(builtInName)) {
    *printer << " " << builtInName << "(\"" << builtin.getValue() << "\")";
    elidedAttrs.push_back(builtInName);
  }

  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

//===----------------------------------------------------------------------===//
// spv.AccessChainOp
//===----------------------------------------------------------------------===//

static Type getElementPtrType(Type type, ArrayRef<Value *> indices,
                              Location baseLoc) {
  if (indices.empty()) {
    emitError(baseLoc, "'spv.AccessChain' op expected at least "
                       "one index ");
    return nullptr;
  }

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    emitError(baseLoc, "'spv.AccessChain' op expected a pointer "
                       "to composite type, but provided ")
        << type;
    return nullptr;
  }

  auto resultType = ptrType.getPointeeType();
  auto resultStorageClass = ptrType.getStorageClass();
  int32_t index = 0;

  for (auto indexSSA : indices) {
    auto cType = resultType.dyn_cast<spirv::CompositeType>();
    if (!cType) {
      emitError(baseLoc,
                "'spv.AccessChain' op cannot extract from non-composite type ")
          << resultType << " with index " << index;
      return nullptr;
    }
    index = 0;
    if (resultType.isa<spirv::StructType>()) {
      Operation *op = indexSSA->getDefiningOp();
      if (!op) {
        emitError(baseLoc, "'spv.AccessChain' op index must be an "
                           "integer spv.constant to access "
                           "element of spv.struct");
        return nullptr;
      }

      // TODO(denis0x0D): this should be relaxed to allow
      // integer literals of other bitwidths.
      if (failed(extractValueFromConstOp(op, index))) {
        emitError(baseLoc,
                  "'spv.AccessChain' index must be an integer spv.constant to "
                  "access element of spv.struct, but provided ")
            << op->getName();
        return nullptr;
      }
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        emitError(baseLoc, "'spv.AccessChain' op index ")
            << index << " out of bounds for " << resultType;
        return nullptr;
      }
    }
    resultType = cType.getElementType(index);
  }
  return spirv::PointerType::get(resultType, resultStorageClass);
}

void spirv::AccessChainOp::build(Builder *builder, OperationState *state,
                                 Value *basePtr, ArrayRef<Value *> indices) {
  auto type = getElementPtrType(basePtr->getType(), indices, state->location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, indices);
}

static ParseResult parseAccessChainOp(OpAsmParser *parser,
                                      OperationState *state) {
  OpAsmParser::OperandType ptrInfo;
  SmallVector<OpAsmParser::OperandType, 4> indicesInfo;
  Type type;
  // TODO(denis0x0D): regarding to the spec an index must be any integer type,
  // figure out how to use resolveOperand with a range of types and do not
  // fail on first attempt.
  Type indicesType = parser->getBuilder().getIntegerType(32);

  if (parser->parseOperand(ptrInfo) ||
      parser->parseOperandList(indicesInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseColonType(type) ||
      parser->resolveOperand(ptrInfo, type, state->operands) ||
      parser->resolveOperands(indicesInfo, indicesType, state->operands)) {
    return failure();
  }

  auto resultType = getElementPtrType(
      type, llvm::makeArrayRef(state->operands).drop_front(), state->location);
  if (!resultType) {
    return failure();
  }

  state->addTypes(resultType);
  return success();
}

static void print(spirv::AccessChainOp op, OpAsmPrinter *printer) {
  *printer << spirv::AccessChainOp::getOperationName() << ' ' << *op.base_ptr()
           << '[';
  printer->printOperands(op.indices());
  *printer << "] : " << op.base_ptr()->getType();
}

static LogicalResult verify(spirv::AccessChainOp accessChainOp) {
  SmallVector<Value *, 4> indices(accessChainOp.indices().begin(),
                                  accessChainOp.indices().end());
  auto resultType = getElementPtrType(accessChainOp.base_ptr()->getType(),
                                      indices, accessChainOp.getLoc());
  if (!resultType) {
    return failure();
  }

  auto providedResultType =
      accessChainOp.getType().dyn_cast<spirv::PointerType>();
  if (!providedResultType) {
    return accessChainOp.emitOpError(
               "result type must be a pointer, but provided")
           << providedResultType;
  }

  if (resultType != providedResultType) {
    return accessChainOp.emitOpError("invalid result type: expected ")
           << resultType << ", but provided " << providedResultType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv._address_of
//===----------------------------------------------------------------------===//

static ParseResult parseAddressOfOp(OpAsmParser *parser,
                                    OperationState *state) {
  SymbolRefAttr varRefAttr;
  Type type;
  if (parser->parseAttribute(varRefAttr, Type(), kVariableAttrName,
                             state->attributes) ||
      parser->parseColonType(type)) {
    return failure();
  }
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return parser->emitError(parser->getCurrentLocation(),
                             "expected spv.ptr type");
  }
  state->addTypes(ptrType);
  return success();
}

static void print(spirv::AddressOfOp addressOfOp, OpAsmPrinter *printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  *printer << spirv::AddressOfOp::getOperationName();

  // Print symbol name.
  *printer << " @" << addressOfOp.variable();

  // Print the type.
  *printer << " : " << addressOfOp.pointer()->getType();
}

static LogicalResult verify(spirv::AddressOfOp addressOfOp) {
  auto moduleOp = addressOfOp.getParentOfType<spirv::ModuleOp>();
  auto varOp =
      moduleOp.lookupSymbol<spirv::GlobalVariableOp>(addressOfOp.variable());
  if (!varOp) {
    return addressOfOp.emitOpError("expected spv.globalVariable symbol");
  }
  if (addressOfOp.pointer()->getType() != varOp.type()) {
    return addressOfOp.emitOpError(
        "result type mismatch with the referenced global variable's type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.BranchOp
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser *parser, OperationState *state) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser->parseSuccessorAndUseList(dest, destOperands))
    return failure();
  state->addSuccessor(dest, destOperands);
  return success();
}

static void print(spirv::BranchOp branchOp, OpAsmPrinter *printer) {
  *printer << spirv::BranchOp::getOperationName() << ' ';
  printer->printSuccessorAndUseList(branchOp.getOperation(), /*index=*/0);
}

static LogicalResult verify(spirv::BranchOp branchOp) {
  auto *op = branchOp.getOperation();
  if (op->getNumSuccessors() != 1)
    branchOp.emitOpError("must have exactly one successor");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.BranchConditionalOp
//===----------------------------------------------------------------------===//

static ParseResult parseBranchConditionalOp(OpAsmParser *parser,
                                            OperationState *state) {
  auto &builder = parser->getBuilder();
  OpAsmParser::OperandType condInfo;
  Block *dest;
  SmallVector<Value *, 4> destOperands;

  // Parse the condition.
  Type boolTy = builder.getI1Type();
  if (parser->parseOperand(condInfo) ||
      parser->resolveOperand(condInfo, boolTy, state->operands))
    return failure();

  // Parse the optional branch weights.
  if (succeeded(parser->parseOptionalLSquare())) {
    IntegerAttr trueWeight, falseWeight;
    SmallVector<NamedAttribute, 2> weights;

    auto i32Type = builder.getIntegerType(32);
    if (parser->parseAttribute(trueWeight, i32Type, "weight", weights) ||
        parser->parseComma() ||
        parser->parseAttribute(falseWeight, i32Type, "weight", weights) ||
        parser->parseRSquare())
      return failure();

    state->addAttribute(kBranchWeightAttrName,
                        builder.getArrayAttr({trueWeight, falseWeight}));
  }

  // Parse the true branch.
  if (parser->parseComma() ||
      parser->parseSuccessorAndUseList(dest, destOperands))
    return failure();
  state->addSuccessor(dest, destOperands);

  // Parse the false branch.
  destOperands.clear();
  if (parser->parseComma() ||
      parser->parseSuccessorAndUseList(dest, destOperands))
    return failure();
  state->addSuccessor(dest, destOperands);

  return success();
}

static void print(spirv::BranchConditionalOp branchOp, OpAsmPrinter *printer) {
  *printer << spirv::BranchConditionalOp::getOperationName() << ' ';
  printer->printOperand(branchOp.condition());

  if (auto weights = branchOp.branch_weights()) {
    *printer << " [";
    mlir::interleaveComma(
        weights->getValue(), printer->getStream(),
        [&](Attribute a) { *printer << a.cast<IntegerAttr>().getInt(); });
    *printer << "]";
  }

  *printer << ", ";
  printer->printSuccessorAndUseList(branchOp.getOperation(),
                                    spirv::BranchConditionalOp::kTrueIndex);
  *printer << ", ";
  printer->printSuccessorAndUseList(branchOp.getOperation(),
                                    spirv::BranchConditionalOp::kFalseIndex);
}

static LogicalResult verify(spirv::BranchConditionalOp branchOp) {
  auto *op = branchOp.getOperation();
  if (op->getNumSuccessors() != 2)
    return branchOp.emitOpError("must have exactly two successors");

  if (auto weights = branchOp.branch_weights()) {
    if (weights->getValue().size() != 2) {
      return branchOp.emitOpError("must have exactly two branch weights");
    }
    if (llvm::all_of(*weights, [](Attribute attr) {
          return attr.cast<IntegerAttr>().getValue().isNullValue();
        }))
      return branchOp.emitOpError("branch weights cannot both be zero");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CompositeExtractOp
//===----------------------------------------------------------------------===//

static ParseResult parseCompositeExtractOp(OpAsmParser *parser,
                                           OperationState *state) {
  OpAsmParser::OperandType compositeInfo;
  Attribute indicesAttr;
  Type compositeType;
  llvm::SMLoc attrLocation;
  int32_t index;

  if (parser->parseOperand(compositeInfo) ||
      parser->getCurrentLocation(&attrLocation) ||
      parser->parseAttribute(indicesAttr, kIndicesAttrName,
                             state->attributes) ||
      parser->parseColonType(compositeType) ||
      parser->resolveOperand(compositeInfo, compositeType, state->operands)) {
    return failure();
  }

  auto indicesArrayAttr = indicesAttr.dyn_cast<ArrayAttr>();
  if (!indicesArrayAttr) {
    return parser->emitError(
        attrLocation,
        "expected an 32-bit integer array attribute for 'indices'");
  }

  if (!indicesArrayAttr.size()) {
    return parser->emitError(
        attrLocation, "expected at least one index for spv.CompositeExtract");
  }

  Type resultType = compositeType;
  for (auto indexAttr : indicesArrayAttr) {
    if (auto indexIntAttr = indexAttr.dyn_cast<IntegerAttr>()) {
      index = indexIntAttr.getInt();
    } else {
      return parser->emitError(
                 attrLocation,
                 "expexted an 32-bit integer for index, but found '")
             << indexAttr << "'";
    }

    if (auto cType = resultType.dyn_cast<spirv::CompositeType>()) {
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        return parser->emitError(attrLocation, "index ")
               << index << " out of bounds for " << resultType;
      }
      resultType = cType.getElementType(index);
    } else {
      return parser->emitError(attrLocation,
                               "cannot extract from non-composite type ")
             << resultType << " with index " << index;
    }
  }

  state->addTypes(resultType);
  return success();
}

static void print(spirv::CompositeExtractOp compositeExtractOp,
                  OpAsmPrinter *printer) {
  *printer << spirv::CompositeExtractOp::getOperationName() << ' '
           << *compositeExtractOp.composite() << compositeExtractOp.indices()
           << " : " << compositeExtractOp.composite()->getType();
}

static LogicalResult verify(spirv::CompositeExtractOp compExOp) {
  auto resultType = compExOp.composite()->getType();
  auto indicesArrayAttr = compExOp.indices().dyn_cast<ArrayAttr>();

  if (!indicesArrayAttr.size()) {
    return compExOp.emitOpError(
        "expexted at least one index for spv.CompositeExtractOp");
  }

  int32_t index;
  for (auto indexAttr : indicesArrayAttr) {
    index = indexAttr.dyn_cast<IntegerAttr>().getInt();
    if (auto cType = resultType.dyn_cast<spirv::CompositeType>()) {
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        return compExOp.emitOpError("index ")
               << index << " out of bounds for " << resultType;
      }
      resultType = cType.getElementType(index);
    } else {
      return compExOp.emitError("cannot extract from non-composite type ")
             << resultType << " with index " << index;
    }
  }

  if (resultType != compExOp.getType()) {
    return compExOp.emitOpError("invalid result type: expected ")
           << resultType << " but provided " << compExOp.getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

static ParseResult parseConstantOp(OpAsmParser *parser, OperationState *state) {
  Attribute value;
  if (parser->parseAttribute(value, kValueAttrName, state->attributes))
    return failure();

  Type type;
  if (value.getType().isa<NoneType>()) {
    if (parser->parseColonType(type))
      return failure();
  } else {
    type = value.getType();
  }

  return parser->addTypeToList(type, state->types);
}

static void print(spirv::ConstantOp constOp, OpAsmPrinter *printer) {
  *printer << spirv::ConstantOp::getOperationName() << ' ' << constOp.value();
  if (constOp.getType().isa<spirv::ArrayType>()) {
    *printer << " : " << constOp.getType();
  }
}

static LogicalResult verify(spirv::ConstantOp constOp) {
  auto opType = constOp.getType();
  auto value = constOp.value();
  auto valueType = value.getType();

  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  switch (value.getKind()) {
  case StandardAttributes::Bool:
  case StandardAttributes::Integer:
  case StandardAttributes::Float:
  case StandardAttributes::DenseElements:
  case StandardAttributes::SparseElements: {
    if (valueType != opType)
      return constOp.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  } break;
  case StandardAttributes::Array: {
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    if (!arrayType)
      return constOp.emitOpError(
          "must have spv.array result type for array value");
    auto elemType = arrayType.getElementType();
    for (auto element : value.cast<ArrayAttr>().getValue()) {
      if (element.getType() != elemType)
        return constOp.emitOpError("has array element whose type (")
               << element.getType()
               << ") does not match the result element type (" << elemType
               << ')';
    }
  } break;
  default:
    return constOp.emitOpError("cannot have value of type ") << valueType;
  }

  return success();
}

OpFoldResult spirv::ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

bool spirv::ConstantOp::isBuildableWith(Type type) {
  // Must be valid SPIR-V type first.
  if (!SPIRVDialect::isValidType(type))
    return false;

  if (type.getKind() >= Type::FIRST_SPIRV_TYPE &&
      type.getKind() <= spirv::TypeKind::LAST_SPIRV_TYPE) {
    // TODO(antiagainst): support contant struct
    return type.isa<spirv::ArrayType>();
  }

  return true;
}

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

static ParseResult parseEntryPointOp(OpAsmParser *parser,
                                     OperationState *state) {
  spirv::ExecutionModel execModel;
  SmallVector<OpAsmParser::OperandType, 0> identifiers;
  SmallVector<Type, 0> idTypes;

  SymbolRefAttr fn;
  if (parseEnumAttribute(execModel, parser, state) ||
      parser->parseAttribute(fn, Type(), kFnNameAttrName, state->attributes)) {
    return failure();
  }

  if (!parser->parseOptionalComma()) {
    // Parse the interface variables
    SmallVector<Attribute, 4> interfaceVars;
    do {
      // The name of the interface variable attribute isnt important
      auto attrName = "var_symbol";
      SymbolRefAttr var;
      SmallVector<NamedAttribute, 1> attrs;
      if (parser->parseAttribute(var, Type(), attrName, attrs)) {
        return failure();
      }
      interfaceVars.push_back(var);
    } while (!parser->parseOptionalComma());
    state->addAttribute(kInterfaceAttrName,
                        parser->getBuilder().getArrayAttr(interfaceVars));
  }
  return success();
}

static void print(spirv::EntryPointOp entryPointOp, OpAsmPrinter *printer) {
  *printer << spirv::EntryPointOp::getOperationName() << " \""
           << stringifyExecutionModel(entryPointOp.execution_model()) << "\" @"
           << entryPointOp.fn();
  if (auto interface = entryPointOp.interface()) {
    *printer << ", ";
    mlir::interleaveComma(interface.getValue().getValue(), printer->getStream(),
                          [&](Attribute a) { printer->printAttribute(a); });
  }
}

static LogicalResult verify(spirv::EntryPointOp entryPointOp) {
  // Checks for fn and interface symbol reference are done in spirv::ModuleOp
  // verification.
  return success();
}

//===----------------------------------------------------------------------===//
// spv.ExecutionMode
//===----------------------------------------------------------------------===//

static ParseResult parseExecutionModeOp(OpAsmParser *parser,
                                        OperationState *state) {
  spirv::ExecutionMode execMode;
  Attribute fn;
  if (parser->parseAttribute(fn, kFnNameAttrName, state->attributes) ||
      parseEnumAttribute(execMode, parser, state)) {
    return failure();
  }

  SmallVector<int32_t, 4> values;
  Type i32Type = parser->getBuilder().getIntegerType(32);
  while (!parser->parseOptionalComma()) {
    SmallVector<NamedAttribute, 1> attr;
    Attribute value;
    if (parser->parseAttribute(value, i32Type, "value", attr)) {
      return failure();
    }
    values.push_back(value.cast<IntegerAttr>().getInt());
  }
  state->addAttribute(kValuesAttrName,
                      parser->getBuilder().getI32ArrayAttr(values));
  return success();
}

static void print(spirv::ExecutionModeOp execModeOp, OpAsmPrinter *printer) {
  *printer << spirv::ExecutionModeOp::getOperationName() << " @"
           << execModeOp.fn() << " \""
           << stringifyExecutionMode(execModeOp.execution_mode()) << "\"";
  auto values = execModeOp.values();
  if (!values) {
    return;
  }
  *printer << ", ";
  mlir::interleaveComma(
      values.getValue().cast<ArrayAttr>(), printer->getStream(),
      [&](Attribute a) { *printer << a.cast<IntegerAttr>().getInt(); });
}

//===----------------------------------------------------------------------===//
// spv.globalVariable
//===----------------------------------------------------------------------===//

static ParseResult parseGlobalVariableOp(OpAsmParser *parser,
                                         OperationState *state) {
  // Parse variable name.
  StringAttr nameAttr;
  if (parser->parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                              state->attributes)) {
    return failure();
  }

  // Parse optional initializer
  if (succeeded(parser->parseOptionalKeyword(kInitializerAttrName))) {
    SymbolRefAttr initSymbol;
    if (parser->parseLParen() ||
        parser->parseAttribute(initSymbol, Type(), kInitializerAttrName,
                               state->attributes) ||
        parser->parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, state)) {
    return failure();
  }

  Type type;
  auto loc = parser->getCurrentLocation();
  if (parser->parseColonType(type)) {
    return failure();
  }
  if (!type.isa<spirv::PointerType>()) {
    return parser->emitError(loc, "expected spv.ptr type");
  }
  state->addAttribute(kTypeAttrName, parser->getBuilder().getTypeAttr(type));

  return success();
}

static void print(spirv::GlobalVariableOp varOp, OpAsmPrinter *printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  *printer << spirv::GlobalVariableOp::getOperationName();

  // Print variable name.
  *printer << " @" << varOp.sym_name();
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());

  // Print optional initializer
  if (auto initializer = varOp.initializer()) {
    *printer << " " << kInitializerAttrName << "(@" << initializer.getValue()
             << ")";
    elidedAttrs.push_back(kInitializerAttrName);
  }

  elidedAttrs.push_back(kTypeAttrName);
  printVariableDecorations(op, printer, elidedAttrs);
  *printer << " : " << varOp.type();
}

static LogicalResult verify(spirv::GlobalVariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (varOp.storageClass() == spirv::StorageClass::Generic)
    return varOp.emitOpError("storage class cannot be 'Generic'");

  if (auto init = varOp.getAttrOfType<SymbolRefAttr>(kInitializerAttrName)) {
    auto moduleOp = varOp.getParentOfType<spirv::ModuleOp>();
    auto *initOp = moduleOp.lookupSymbol(init.getValue());
    // TODO: Currently only variable initialization with specialization
    // constants and other variables is supported. They could be normal
    // constants in the module scope as well.
    if (!initOp || !(isa<spirv::GlobalVariableOp>(initOp) ||
                     isa<spirv::SpecConstantOp>(initOp))) {
      return varOp.emitOpError("initializer must be result of a "
                               "spv.specConstant or spv.globalVariable op");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.LoadOp
//===----------------------------------------------------------------------===//

static ParseResult parseLoadOp(OpAsmParser *parser, OperationState *state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::OperandType ptrInfo;
  Type elementType;
  if (parseEnumAttribute(storageClass, parser) ||
      parser->parseOperand(ptrInfo) ||
      parseMemoryAccessAttributes(parser, state) ||
      parser->parseOptionalAttributeDict(state->attributes) ||
      parser->parseColon() || parser->parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser->resolveOperand(ptrInfo, ptrType, state->operands)) {
    return failure();
  }

  state->addTypes(elementType);
  return success();
}

static void print(spirv::LoadOp loadOp, OpAsmPrinter *printer) {
  auto *op = loadOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      loadOp.ptr()->getType().cast<spirv::PointerType>().getStorageClass());
  *printer << spirv::LoadOp::getOperationName() << " \"" << sc << "\" ";
  // Print the pointer operand.
  printer->printOperand(loadOp.ptr());

  printMemoryAccessAttribute(loadOp, printer, elidedAttrs);

  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  *printer << " : " << loadOp.getType();
}

static LogicalResult verify(spirv::LoadOp loadOp) {
  // SPIR-V spec : "Result Type is the type of the loaded object. It must be a
  // type with fixed size; i.e., it cannot be, nor include, any
  // OpTypeRuntimeArray types."
  if (failed(verifyLoadStorePtrAndValTypes(loadOp, loadOp.ptr(),
                                           loadOp.value()))) {
    return failure();
  }
  return verifyMemoryAccessAttribute(loadOp);
}

//===----------------------------------------------------------------------===//
// spv.loop
//===----------------------------------------------------------------------===//

static ParseResult parseLoopOp(OpAsmParser *parser, OperationState *state) {
  // TODO(antiagainst): support loop control properly
  Builder builder = parser->getBuilder();
  state->addAttribute("loop_control",
                      builder.getI32IntegerAttr(
                          static_cast<uint32_t>(spirv::LoopControl::None)));

  return parser->parseRegion(*state->addRegion(), /*arguments=*/{},
                             /*argTypes=*/{});
}

static void print(spirv::LoopOp loopOp, OpAsmPrinter *printer) {
  auto *op = loopOp.getOperation();

  *printer << spirv::LoopOp::getOperationName();
  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/true);
}

/// Returns true if the given `block` only contains one `spv._merge` op.
static inline bool isMergeBlock(Block &block) {
  return std::next(block.begin()) == block.end() &&
         isa<spirv::MergeOp>(block.front());
}

/// Returns true if the given `srcBlock` contains only one `spv.Branch` to the
/// given `dstBlock`.
static inline bool hasOneBranchOpTo(Block &srcBlock, Block &dstBlock) {
  // Check that there is only one op in the `srcBlock`.
  if (std::next(srcBlock.begin()) != srcBlock.end())
    return false;

  auto branchOp = dyn_cast<spirv::BranchOp>(srcBlock.back());
  return branchOp && branchOp.getSuccessor(0) == &dstBlock;
}

static LogicalResult verify(spirv::LoopOp loopOp) {
  auto *op = loopOp.getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +-------------+
  //                     | entry block |
  //                     +-------------+
  //                            |
  //                            v
  //                     +-------------+
  //                     | loop header | <-----+
  //                     +-------------+       |
  //                                           |
  //                           ...             |
  //                          \ | /            |
  //                            v              |
  //                    +---------------+      |
  //                    | loop continue | -----+
  //                    +---------------+
  //
  //                           ...
  //                          \ | /
  //                            v
  //                     +-------------+
  //                     | merge block |
  //                     +-------------+

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerated case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block is the merge block.
  Block &merge = region.back();
  if (!isMergeBlock(merge))
    return loopOp.emitOpError(
        "last block must be the merge block with only one 'spv._merge' op");

  if (std::next(region.begin()) == region.end())
    return loopOp.emitOpError(
        "must have an entry block branching to the loop header block");
  // The first block is the entry block.
  Block &entry = region.front();

  if (std::next(region.begin(), 2) == region.end())
    return loopOp.emitOpError(
        "must have a loop header block branched from the entry block");
  // The second block is the loop header block.
  Block &header = *std::next(region.begin(), 1);

  if (!hasOneBranchOpTo(entry, header))
    return loopOp.emitOpError(
        "entry block must only have one 'spv.Branch' op to the second block");

  if (std::next(region.begin(), 3) == region.end())
    return loopOp.emitOpError(
        "requires a loop continue block branching to the loop header block");
  // The second to last block is the loop continue block.
  Block &cont = *std::prev(region.end(), 2);

  // Make sure that we have a branch from the loop continue block to the loop
  // header block.
  if (llvm::none_of(
          llvm::seq<unsigned>(0, cont.getNumSuccessors()),
          [&](unsigned index) { return cont.getSuccessor(index) == &header; }))
    return loopOp.emitOpError("second to last block must be the loop continue "
                              "block that branches to the loop header block");

  // Make sure that no other blocks (except the entry and loop continue block)
  // branches to the loop header block.
  for (auto &block : llvm::make_range(std::next(region.begin(), 2),
                                      std::prev(region.end(), 2))) {
    for (auto i : llvm::seq<unsigned>(0, block.getNumSuccessors())) {
      if (block.getSuccessor(i) == &header) {
        return loopOp.emitOpError("can only have the entry and loop continue "
                                  "block branching to the loop header block");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv._merge
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::MergeOp mergeOp) {
  Block &parentLastBlock = mergeOp.getParentRegion()->back();
  if (mergeOp.getOperation() != parentLastBlock.getTerminator())
    return mergeOp.emitOpError(
        "can only be used in the last block of 'spv.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

void spirv::ModuleOp::build(Builder *builder, OperationState *state) {
  ensureTerminator(*state->addRegion(), *builder, state->location);
}

void spirv::ModuleOp::build(Builder *builder, OperationState *state,
                            IntegerAttr addressing_model,
                            IntegerAttr memory_model, ArrayAttr capabilities,
                            ArrayAttr extensions,
                            ArrayAttr extended_instruction_sets) {
  state->addAttribute("addressing_model", addressing_model);
  state->addAttribute("memory_model", memory_model);
  if (capabilities)
    state->addAttribute("capabilities", capabilities);
  if (extensions)
    state->addAttribute("extensions", extensions);
  if (extended_instruction_sets)
    state->addAttribute("extended_instruction_sets", extended_instruction_sets);
  ensureTerminator(*state->addRegion(), *builder, state->location);
}

static ParseResult parseModuleOp(OpAsmParser *parser, OperationState *state) {
  Region *body = state->addRegion();

  // Parse attributes
  spirv::AddressingModel addrModel;
  spirv::MemoryModel memoryModel;
  if (parseEnumAttribute(addrModel, parser, state) ||
      parseEnumAttribute(memoryModel, parser, state)) {
    return failure();
  }

  if (parser->parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  if (succeeded(parser->parseOptionalKeyword("attributes"))) {
    if (parser->parseOptionalAttributeDict(state->attributes))
      return failure();
  }

  spirv::ModuleOp::ensureTerminator(*body, parser->getBuilder(),
                                    state->location);
  return success();
}

static void print(spirv::ModuleOp moduleOp, OpAsmPrinter *printer) {
  auto *op = moduleOp.getOperation();

  // Only print out addressing model and memory model in a nicer way if both
  // presents. Otherwise, print them in the general form. This helps debugging
  // ill-formed ModuleOp.
  SmallVector<StringRef, 2> elidedAttrs;
  auto addressingModelAttrName = spirv::attributeName<spirv::AddressingModel>();
  auto memoryModelAttrName = spirv::attributeName<spirv::MemoryModel>();
  if (op->getAttr(addressingModelAttrName) &&
      op->getAttr(memoryModelAttrName)) {
    *printer << spirv::ModuleOp::getOperationName() << " \""
             << spirv::stringifyAddressingModel(moduleOp.addressing_model())
             << "\" \"" << spirv::stringifyMemoryModel(moduleOp.memory_model())
             << '"';
    elidedAttrs.assign({addressingModelAttrName, memoryModelAttrName});
  }

  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/false);

  bool printAttrDict =
      elidedAttrs.size() != 2 ||
      llvm::any_of(op->getAttrs(), [&addressingModelAttrName,
                                    &memoryModelAttrName](NamedAttribute attr) {
        return attr.first != addressingModelAttrName &&
               attr.first != memoryModelAttrName;
      });

  if (printAttrDict) {
    *printer << " attributes";
    printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  }
}

static LogicalResult verify(spirv::ModuleOp moduleOp) {
  auto &op = *moduleOp.getOperation();
  auto *dialect = op.getDialect();
  auto &body = op.getRegion(0).front();
  llvm::DenseMap<std::pair<FuncOp, spirv::ExecutionModel>, spirv::EntryPointOp>
      entryPoints;
  SymbolTable table(moduleOp);

  for (auto &op : body) {
    if (op.getDialect() == dialect) {
      // For EntryPoint op, check that the function and execution model is not
      // duplicated in EntryPointOps. Also verify that the interface specified
      // comes from globalVariables here to make this check cheaper.
      if (auto entryPointOp = dyn_cast<spirv::EntryPointOp>(op)) {
        auto funcOp = table.lookup<FuncOp>(entryPointOp.fn());
        if (!funcOp) {
          return entryPointOp.emitError("function '")
                 << entryPointOp.fn() << "' not found in 'spv.module'";
        }
        if (auto interface = entryPointOp.interface()) {
          for (auto varRef : interface.getValue().getValue()) {
            auto varSymRef = varRef.dyn_cast<SymbolRefAttr>();
            if (!varSymRef) {
              return entryPointOp.emitError(
                         "expected symbol reference for interface "
                         "specification instead of '")
                     << varRef;
            }
            auto variableOp =
                table.lookup<spirv::GlobalVariableOp>(varSymRef.getValue());
            if (!variableOp) {
              return entryPointOp.emitError("expected spv.globalVariable "
                                            "symbol reference instead of'")
                     << varSymRef << "'";
            }
          }
        }

        auto key = std::pair<FuncOp, spirv::ExecutionModel>(
            funcOp, entryPointOp.execution_model());
        auto entryPtIt = entryPoints.find(key);
        if (entryPtIt != entryPoints.end()) {
          return entryPointOp.emitError("duplicate of a previous EntryPointOp");
        }
        entryPoints[key] = entryPointOp;
      }
      continue;
    }

    auto funcOp = dyn_cast<FuncOp>(op);
    if (!funcOp)
      return op.emitError("'spv.module' can only contain func and spv.* ops");

    if (funcOp.isExternal())
      return op.emitError("'spv.module' cannot contain external functions");

    for (auto &block : funcOp)
      for (auto &op : block) {
        if (op.getDialect() == dialect)
          continue;

        if (isa<FuncOp>(op))
          return op.emitError("'spv.module' cannot contain nested functions");

        return op.emitError(
            "functions in 'spv.module' can only contain spv.* ops");
      }
  }

  // Verify capabilities. ODS already guarantees that we have an array of
  // string attributes.
  if (auto caps = moduleOp.getAttrOfType<ArrayAttr>("capabilities")) {
    for (auto cap : caps.getValue()) {
      auto capStr = cap.cast<StringAttr>().getValue();
      if (!spirv::symbolizeCapability(capStr))
        return moduleOp.emitOpError("uses unknown capability: ") << capStr;
    }
  }

  // Verify extensions. ODS already guarantees that we have an array of
  // string attributes.
  if (auto exts = moduleOp.getAttrOfType<ArrayAttr>("extensions")) {
    for (auto ext : exts.getValue()) {
      auto extStr = ext.cast<StringAttr>().getValue();
      if (!spirv::symbolizeExtension(extStr))
        return moduleOp.emitOpError("uses unknown extension: ") << extStr;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv._reference_of
//===----------------------------------------------------------------------===//

static ParseResult parseReferenceOfOp(OpAsmParser *parser,
                                      OperationState *state) {
  SymbolRefAttr constRefAttr;
  Type type;
  if (parser->parseAttribute(constRefAttr, Type(), kSpecConstAttrName,
                             state->attributes) ||
      parser->parseColonType(type)) {
    return failure();
  }
  return parser->addTypeToList(type, state->types);
}

static void print(spirv::ReferenceOfOp referenceOfOp, OpAsmPrinter *printer) {
  *printer << spirv::ReferenceOfOp::getOperationName() << " @"
           << referenceOfOp.spec_const() << " : "
           << referenceOfOp.reference()->getType();
}

static LogicalResult verify(spirv::ReferenceOfOp referenceOfOp) {
  auto moduleOp = referenceOfOp.getParentOfType<spirv::ModuleOp>();
  auto specConstOp =
      moduleOp.lookupSymbol<spirv::SpecConstantOp>(referenceOfOp.spec_const());
  if (!specConstOp) {
    return referenceOfOp.emitOpError("expected spv.specConstant symbol");
  }
  if (referenceOfOp.reference()->getType() !=
      specConstOp.default_value().getType()) {
    return referenceOfOp.emitOpError("result type mismatch with the referenced "
                                     "specialization constant's type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::ReturnOp returnOp) {
  auto funcOp = cast<FuncOp>(returnOp.getParentOp());
  auto numOutputs = funcOp.getType().getNumResults();
  if (numOutputs != 0)
    return returnOp.emitOpError("cannot be used in functions returning value")
           << (numOutputs > 1 ? "s" : "");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

static ParseResult parseReturnValueOp(OpAsmParser *parser,
                                      OperationState *state) {
  OpAsmParser::OperandType retValInfo;
  Type retValType;
  return failure(
      parser->parseOperand(retValInfo) || parser->parseColonType(retValType) ||
      parser->resolveOperand(retValInfo, retValType, state->operands));
}

static void print(spirv::ReturnValueOp retValOp, OpAsmPrinter *printer) {
  *printer << spirv::ReturnValueOp::getOperationName() << ' ';
  printer->printOperand(retValOp.value());
  *printer << " : " << retValOp.value()->getType();
}

static LogicalResult verify(spirv::ReturnValueOp retValOp) {
  auto funcOp = cast<FuncOp>(retValOp.getParentOp());
  auto numFnResults = funcOp.getType().getNumResults();
  if (numFnResults != 1)
    return retValOp.emitOpError(
               "returns 1 value but enclosing function requires ")
           << numFnResults << " results";

  auto operandType = retValOp.value()->getType();
  auto fnResultType = funcOp.getType().getResult(0);
  if (operandType != fnResultType)
    return retValOp.emitOpError(" return value's type (")
           << operandType << ") mismatch with function's result type ("
           << fnResultType << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// spv.Select
//===----------------------------------------------------------------------===//

static ParseResult parseSelectOp(OpAsmParser *parser, OperationState *state) {
  OpAsmParser::OperandType condition;
  SmallVector<OpAsmParser::OperandType, 2> operands;
  SmallVector<Type, 2> types;
  auto loc = parser->getCurrentLocation();
  if (parser->parseOperand(condition) || parser->parseComma() ||
      parser->parseOperandList(operands, 2) ||
      parser->parseColonTypeList(types)) {
    return failure();
  }
  if (types.size() != 2) {
    return parser->emitError(
        loc, "need exactly two trailing types for select condition and object");
  }
  if (parser->resolveOperand(condition, types[0], state->operands) ||
      parser->resolveOperands(operands, types[1], state->operands)) {
    return failure();
  }
  return parser->addTypesToList(types[1], state->types);
}

static void print(spirv::SelectOp op, OpAsmPrinter *printer) {
  *printer << spirv::SelectOp::getOperationName() << " ";

  // Print the operands.
  printer->printOperands(op.getOperands());

  // Print colon and types.
  *printer << " : " << op.condition()->getType() << ", "
           << op.result()->getType();
}

static LogicalResult verify(spirv::SelectOp op) {
  auto resultTy = op.result()->getType();
  if (op.true_value()->getType() != resultTy) {
    return op.emitOpError("result type and true value type must be the same");
  }
  if (op.false_value()->getType() != resultTy) {
    return op.emitOpError("result type and false value type must be the same");
  }
  if (auto conditionTy = op.condition()->getType().dyn_cast<VectorType>()) {
    auto resultVectorTy = resultTy.dyn_cast<VectorType>();
    if (!resultVectorTy) {
      return op.emitOpError("result expected to be of vector type when "
                            "condition is of vector type");
    }
    if (resultVectorTy.getNumElements() != conditionTy.getNumElements()) {
      return op.emitOpError("result should have the same number of elements as "
                            "the condition when condition is of vector type");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.specConstant
//===----------------------------------------------------------------------===//

static ParseResult parseSpecConstantOp(OpAsmParser *parser,
                                       OperationState *state) {
  StringAttr nameAttr;
  Attribute valueAttr;

  if (parser->parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                              state->attributes) ||
      parser->parseEqual() ||
      parser->parseAttribute(valueAttr, kDefaultValueAttrName,
                             state->attributes))
    return failure();

  return success();
}

static void print(spirv::SpecConstantOp constOp, OpAsmPrinter *printer) {
  *printer << spirv::SpecConstantOp::getOperationName() << " @"
           << constOp.sym_name() << " = ";
  printer->printAttribute(constOp.default_value());
}

static LogicalResult verify(spirv::SpecConstantOp constOp) {
  auto value = constOp.default_value();

  switch (value.getKind()) {
  case StandardAttributes::Bool:
  case StandardAttributes::Integer:
  case StandardAttributes::Float: {
    // Make sure bitwidth is allowed.
    if (!spirv::SPIRVDialect::isValidType(value.getType()))
      return constOp.emitOpError("default value bitwidth disallowed");
    return success();
  }
  default:
    return constOp.emitOpError(
        "default value can only be a bool, integer, or float scalar");
  }
}

//===----------------------------------------------------------------------===//
// spv.StoreOp
//===----------------------------------------------------------------------===//

static ParseResult parseStoreOp(OpAsmParser *parser, OperationState *state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  auto loc = parser->getCurrentLocation();
  Type elementType;
  if (parseEnumAttribute(storageClass, parser) ||
      parser->parseOperandList(operandInfo, 2) ||
      parseMemoryAccessAttributes(parser, state) || parser->parseColon() ||
      parser->parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser->resolveOperands(operandInfo, {ptrType, elementType}, loc,
                              state->operands)) {
    return failure();
  }
  return success();
}

static void print(spirv::StoreOp storeOp, OpAsmPrinter *printer) {
  auto *op = storeOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      storeOp.ptr()->getType().cast<spirv::PointerType>().getStorageClass());
  *printer << spirv::StoreOp::getOperationName() << " \"" << sc << "\" ";
  // Print the pointer operand
  printer->printOperand(storeOp.ptr());
  *printer << ", ";
  // Print the value operand
  printer->printOperand(storeOp.value());

  printMemoryAccessAttribute(storeOp, printer, elidedAttrs);

  *printer << " : " << storeOp.value()->getType();

  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

static LogicalResult verify(spirv::StoreOp storeOp) {
  // SPIR-V spec : "Pointer is the pointer to store through. Its type must be an
  // OpTypePointer whose Type operand is the same as the type of Object."
  if (failed(verifyLoadStorePtrAndValTypes(storeOp, storeOp.ptr(),
                                           storeOp.value()))) {
    return failure();
  }
  return verifyMemoryAccessAttribute(storeOp);
}

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

static ParseResult parseVariableOp(OpAsmParser *parser, OperationState *state) {
  // Parse optional initializer
  Optional<OpAsmParser::OperandType> initInfo;
  if (succeeded(parser->parseOptionalKeyword("init"))) {
    initInfo = OpAsmParser::OperandType();
    if (parser->parseLParen() || parser->parseOperand(*initInfo) ||
        parser->parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, state)) {
    return failure();
  }

  // Parse result pointer type
  Type type;
  if (parser->parseColon())
    return failure();
  auto loc = parser->getCurrentLocation();
  if (parser->parseType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser->emitError(loc, "expected spv.ptr type");
  state->addTypes(ptrType);

  // Resolve the initializer operand
  SmallVector<Value *, 1> init;
  if (initInfo) {
    if (parser->resolveOperand(*initInfo, ptrType.getPointeeType(), init))
      return failure();
    state->addOperands(init);
  }

  auto attr = parser->getBuilder().getI32IntegerAttr(
      bitwiseCast<int32_t>(ptrType.getStorageClass()));
  state->addAttribute(spirv::attributeName<spirv::StorageClass>(), attr);

  return success();
}

static void print(spirv::VariableOp varOp, OpAsmPrinter *printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  *printer << spirv::VariableOp::getOperationName();

  // Print optional initializer
  if (op->getNumOperands() > 0) {
    *printer << " init(";
    printer->printOperands(varOp.initializer());
    *printer << ")";
  }

  printVariableDecorations(op, printer, elidedAttrs);

  *printer << " : " << varOp.getType();
}

static LogicalResult verify(spirv::VariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (varOp.storage_class() != spirv::StorageClass::Function) {
    return varOp.emitOpError(
        "can only be used to model function-level variables. Use "
        "spv.globalVariable for module-level variables.");
  }

  auto pointerType = varOp.pointer()->getType().cast<spirv::PointerType>();
  if (varOp.storage_class() != pointerType.getStorageClass())
    return varOp.emitOpError(
        "storage class must match result pointer's storage class");

  if (varOp.getNumOperands() != 0) {
    // SPIR-V spec: "Initializer must be an <id> from a constant instruction or
    // a global (module scope) OpVariable instruction".
    auto *initOp = varOp.getOperand(0)->getDefiningOp();
    if (!initOp || !(isa<spirv::ConstantOp>(initOp) ||    // for normal constant
                     isa<spirv::ReferenceOfOp>(initOp) || // for spec constant
                     isa<spirv::AddressOfOp>(initOp)))
      return varOp.emitOpError("initializer must be the result of a "
                               "constant or spv.globalVariable op");
  }

  // TODO(antiagainst): generate these strings using ODS.
  auto *op = varOp.getOperation();
  auto descriptorSetName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
  auto builtInName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn));

  for (const auto &attr : {descriptorSetName, bindingName, builtInName}) {
    if (op->getAttr(attr))
      return varOp.emitOpError("cannot have '")
             << attr << "' attribute (only allowed in spv.globalVariable)";
  }

  return success();
}

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVOps.cpp.inc"

} // namespace spirv
} // namespace mlir
