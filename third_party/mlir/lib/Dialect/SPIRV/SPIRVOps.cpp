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

#include "mlir/Analysis/CallInterfaces.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/bit.h"

using namespace mlir;

// TODO(antiagainst): generate these strings using ODS.
static constexpr const char kAlignmentAttrName[] = "alignment";
static constexpr const char kBranchWeightAttrName[] = "branch_weights";
static constexpr const char kCallee[] = "callee";
static constexpr const char kDefaultValueAttrName[] = "default_value";
static constexpr const char kExecutionScopeAttrName[] = "execution_scope";
static constexpr const char kEqualSemanticsAttrName[] = "equal_semantics";
static constexpr const char kFnNameAttrName[] = "fn";
static constexpr const char kIndicesAttrName[] = "indices";
static constexpr const char kInitializerAttrName[] = "initializer";
static constexpr const char kInterfaceAttrName[] = "interface";
static constexpr const char kMemoryScopeAttrName[] = "memory_scope";
static constexpr const char kSpecConstAttrName[] = "spec_const";
static constexpr const char kSpecIdAttrName[] = "spec_id";
static constexpr const char kTypeAttrName[] = "type";
static constexpr const char kUnequalSemanticsAttrName[] = "unequal_semantics";
static constexpr const char kValueAttrName[] = "value";
static constexpr const char kValuesAttrName[] = "values";
static constexpr const char kVariableAttrName[] = "variable";

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

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

template <typename EnumClass>
static ParseResult
parseEnumAttribute(EnumClass &value, OpAsmParser &parser,
                   StringRef attrName = spirv::attributeName<EnumClass>()) {
  Attribute attrVal;
  SmallVector<NamedAttribute, 1> attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                            attrName, attr)) {
    return failure();
  }
  if (!attrVal.isa<StringAttr>()) {
    return parser.emitError(loc, "expected ")
           << attrName << " attribute specified as string";
  }
  auto attrOptional =
      spirv::symbolizeEnum<EnumClass>()(attrVal.cast<StringAttr>().getValue());
  if (!attrOptional) {
    return parser.emitError(loc, "invalid ")
           << attrName << " attribute specification: " << attrVal;
  }
  value = attrOptional.getValue();
  return success();
}

template <typename EnumClass>
static ParseResult
parseEnumAttribute(EnumClass &value, OpAsmParser &parser, OperationState &state,
                   StringRef attrName = spirv::attributeName<EnumClass>()) {
  if (parseEnumAttribute(value, parser)) {
    return failure();
  }
  state.addAttribute(attrName, parser.getBuilder().getI32IntegerAttr(
                                   llvm::bit_cast<int32_t>(value)));
  return success();
}

static ParseResult parseMemoryAccessAttributes(OpAsmParser &parser,
                                               OperationState &state) {
  // Parse an optional list of attributes staring with '['
  if (parser.parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  if (parseEnumAttribute(memoryAccessAttr, parser, state)) {
    return failure();
  }

  if (spirv::bitEnumContains(memoryAccessAttr, spirv::MemoryAccess::Aligned)) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseComma() ||
        parser.parseAttribute(alignmentAttr, i32Type, kAlignmentAttrName,
                              state.attributes)) {
      return failure();
    }
  }
  return parser.parseRSquare();
}

template <typename LoadStoreOpTy>
static void
printMemoryAccessAttribute(LoadStoreOpTy loadStoreOp, OpAsmPrinter &printer,
                           SmallVectorImpl<StringRef> &elidedAttrs) {
  // Print optional memory access attribute.
  if (auto memAccess = loadStoreOp.memory_access()) {
    elidedAttrs.push_back(spirv::attributeName<spirv::MemoryAccess>());
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    // Print integer alignment attribute.
    if (auto alignment = loadStoreOp.alignment()) {
      elidedAttrs.push_back(kAlignmentAttrName);
      printer << ", " << alignment;
    }
    printer << "]";
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
}

static LogicalResult verifyCastOp(Operation *op,
                                  bool requireSameBitWidth = true) {
  Type operandType = op->getOperand(0)->getType();
  Type resultType = op->getResult(0)->getType();

  // ODS checks that result type and operand type have the same shape.
  if (auto vectorType = operandType.dyn_cast<VectorType>()) {
    operandType = vectorType.getElementType();
    resultType = resultType.cast<VectorType>().getElementType();
  }

  auto operandTypeBitWidth = operandType.getIntOrFloatBitWidth();
  auto resultTypeBitWidth = resultType.getIntOrFloatBitWidth();
  auto isSameBitWidth = operandTypeBitWidth == resultTypeBitWidth;

  if (requireSameBitWidth) {
    if (!isSameBitWidth) {
      return op->emitOpError(
                 "expected the same bit widths for operand type and result "
                 "type, but provided ")
             << operandType << " and " << resultType;
    }
    return success();
  }

  if (isSameBitWidth) {
    return op->emitOpError(
               "expected the different bit widths for operand type and result "
               "type, but provided ")
           << operandType << " and " << resultType;
  }
  return success();
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

  if (spirv::bitEnumContains(*memAccess, spirv::MemoryAccess::Aligned)) {
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

template <typename BarrierOp>
static LogicalResult verifyMemorySemantics(BarrierOp op) {
  // According to the SPIR-V specification:
  // "Despite being a mask and allowing multiple bits to be combined, it is
  // invalid for more than one of these four bits to be set: Acquire, Release,
  // AcquireRelease, or SequentiallyConsistent. Requesting both Acquire and
  // Release semantics is done by setting the AcquireRelease bit, not by setting
  // two bits."
  auto memorySemantics = op.memory_semantics();
  auto atMostOneInSet = spirv::MemorySemantics::Acquire |
                        spirv::MemorySemantics::Release |
                        spirv::MemorySemantics::AcquireRelease |
                        spirv::MemorySemantics::SequentiallyConsistent;

  auto bitCount = llvm::countPopulation(
      static_cast<uint32_t>(memorySemantics & atMostOneInSet));
  if (bitCount > 1) {
    return op.emitError("expected at most one of these four memory constraints "
                        "to be set: `Acquire`, `Release`,"
                        "`AcquireRelease` or `SequentiallyConsistent`");
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

static ParseResult parseVariableDecorations(OpAsmParser &parser,
                                            OperationState &state) {
  auto builtInName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn));
  if (succeeded(parser.parseOptionalKeyword("bind"))) {
    Attribute set, binding;
    // Parse optional descriptor binding
    auto descriptorSetName = convertToSnakeCase(
        stringifyDecoration(spirv::Decoration::DescriptorSet));
    auto bindingName =
        convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseLParen() ||
        parser.parseAttribute(set, i32Type, descriptorSetName,
                              state.attributes) ||
        parser.parseComma() ||
        parser.parseAttribute(binding, i32Type, bindingName,
                              state.attributes) ||
        parser.parseRParen()) {
      return failure();
    }
  } else if (succeeded(parser.parseOptionalKeyword(builtInName))) {
    StringAttr builtIn;
    if (parser.parseLParen() ||
        parser.parseAttribute(builtIn, builtInName, state.attributes) ||
        parser.parseRParen()) {
      return failure();
    }
  }

  // Parse other attributes
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

static void printVariableDecorations(Operation *op, OpAsmPrinter &printer,
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
    printer << " bind(" << descriptorSet.getInt() << ", " << binding.getInt()
            << ")";
  }

  // Print BuiltIn attribute if present
  auto builtInName =
      convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn));
  if (auto builtin = op->getAttrOfType<StringAttr>(builtInName)) {
    printer << " " << builtInName << "(\"" << builtin.getValue() << "\")";
    elidedAttrs.push_back(builtInName);
  }

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

// Extracts an element from the given `composite` by following the given
// `indices`. Returns a null Attribute if error happens.
static Attribute extractCompositeElement(Attribute composite,
                                         ArrayRef<unsigned> indices) {
  // Check that given composite is a constant.
  if (!composite)
    return {};
  // Return composite itself if we reach the end of the index chain.
  if (indices.empty())
    return composite;

  if (auto vector = composite.dyn_cast<ElementsAttr>()) {
    assert(indices.size() == 1 && "must have exactly one index for a vector");
    return vector.getValue({indices[0]});
  }

  if (auto array = composite.dyn_cast<ArrayAttr>()) {
    assert(!indices.empty() && "must have at least one index for an array");
    return extractCompositeElement(array.getValue()[indices[0]],
                                   indices.drop_front());
  }

  return {};
}

// Get bit width of types.
static unsigned getBitWidth(Type type) {
  if (type.isa<spirv::PointerType>()) {
    // Just return 64 bits for pointer types for now.
    // TODO: Make sure not caller relies on the actual pointer width value.
    return 64;
  }
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth();
  }
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    assert(vectorType.getElementType().isIntOrFloat());
    return vectorType.getNumElements() *
           vectorType.getElementType().getIntOrFloatBitWidth();
  }
  llvm_unreachable("unhandled bit width computation for type");
}

/// Walks the given type hierarchy with the given indices, potentially down
/// to component granularity, to select an element type. Returns null type and
/// emits errors with the given loc on failure.
static Type getElementType(Type type, ArrayAttr indices, Location loc) {
  if (!indices.size()) {
    emitError(loc, "expected at least one index");
    return nullptr;
  }

  int32_t index;
  for (auto indexAttr : indices) {
    index = indexAttr.dyn_cast<IntegerAttr>().getInt();
    if (auto cType = type.dyn_cast<spirv::CompositeType>()) {
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        emitError(loc, "index ") << index << " out of bounds for " << type;
        return nullptr;
      }
      type = cType.getElementType(index);
    } else {
      emitError(loc, "cannot extract from non-composite type ")
          << type << " with index " << index;
      return nullptr;
    }
  }

  return type;
}

/// Returns true if the given `block` only contains one `spv._merge` op.
static inline bool isMergeBlock(Block &block) {
  return !block.empty() && std::next(block.begin()) == block.end() &&
         isa<spirv::MergeOp>(block.front());
}

//===----------------------------------------------------------------------===//
// TableGen'erated canonicalizers
//===----------------------------------------------------------------------===//

namespace {
#include "SPIRVCanonicalization.inc"
}

//===----------------------------------------------------------------------===//
// Common parsers and printers
//===----------------------------------------------------------------------===//

static ParseResult parseBitFieldExtractOp(OpAsmParser &parser,
                                          OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 3> operandInfo;
  Type baseType;
  Type offsetType;
  Type countType;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operandInfo, 3) || parser.parseColon() ||
      parser.parseType(baseType) || parser.parseComma() ||
      parser.parseType(offsetType) || parser.parseComma() ||
      parser.parseType(countType) ||
      parser.resolveOperands(operandInfo, {baseType, offsetType, countType},
                             loc, state.operands)) {
    return failure();
  }
  state.addTypes(baseType);
  return success();
}

static void printBitFieldExtractOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName() << ' ';
  printer.printOperands(op->getOperands());
  printer << " : " << op->getOperand(0)->getType() << ", "
          << op->getOperand(1)->getType() << ", "
          << op->getOperand(2)->getType();
}

static LogicalResult verifyBitFieldExtractOp(Operation *op) {
  if (op->getOperand(0)->getType() != op->getResult(0)->getType()) {
    return op->emitError("expected the same type for the first operand and "
                         "result, but provided ")
           << op->getOperand(0)->getType() << " and "
           << op->getResult(0)->getType();
  }
  return success();
}

// Parses an op that has no inputs and no outputs.
static ParseResult parseNoIOOp(OpAsmParser &parser, OperationState &state) {
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();
  return success();
}

// Prints an op that has no inputs and no outputs.
static void printNoIOOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();
  printer.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseUnaryOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType operandInfo;
  Type type;
  if (parser.parseOperand(operandInfo) || parser.parseColonType(type) ||
      parser.resolveOperands(operandInfo, type, state.operands)) {
    return failure();
  }
  state.addTypes(type);
  return success();
}

static void printUnaryOp(Operation *unaryOp, OpAsmPrinter &printer) {
  printer << unaryOp->getName() << ' ' << *unaryOp->getOperand(0) << " : "
          << unaryOp->getOperand(0)->getType();
}

/// Result of a logical op must be a scalar or vector of boolean type.
static Type getUnaryOpResultType(Builder &builder, Type operandType) {
  Type resultType = builder.getIntegerType(1);
  if (auto vecType = operandType.dyn_cast<VectorType>()) {
    return VectorType::get(vecType.getNumElements(), resultType);
  }
  return resultType;
}

static ParseResult parseLogicalUnaryOp(OpAsmParser &parser,
                                       OperationState &state) {
  OpAsmParser::OperandType operandInfo;
  Type type;
  if (parser.parseOperand(operandInfo) || parser.parseColonType(type) ||
      parser.resolveOperand(operandInfo, type, state.operands)) {
    return failure();
  }
  state.addTypes(getUnaryOpResultType(parser.getBuilder(), type));
  return success();
}

static ParseResult parseLogicalBinaryOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  if (parser.parseOperandList(ops, 2) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands)) {
    return failure();
  }
  result.addTypes(getUnaryOpResultType(parser.getBuilder(), type));
  return success();
}

static void printLogicalOp(Operation *logicalOp, OpAsmPrinter &printer) {
  printer << logicalOp->getName() << ' ';
  printer.printOperands(logicalOp->getOperands());
  printer << " : " << logicalOp->getOperand(0)->getType();
}

static ParseResult parseShiftOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  Type baseType;
  Type shiftType;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operandInfo, 2) || parser.parseColon() ||
      parser.parseType(baseType) || parser.parseComma() ||
      parser.parseType(shiftType) ||
      parser.resolveOperands(operandInfo, {baseType, shiftType}, loc,
                             state.operands)) {
    return failure();
  }
  state.addTypes(baseType);
  return success();
}

static void printShiftOp(Operation *op, OpAsmPrinter &printer) {
  Value *base = op->getOperand(0);
  Value *shift = op->getOperand(1);
  printer << op->getName() << ' ' << *base << ", " << *shift << " : "
          << base->getType() << ", " << shift->getType();
}

static LogicalResult verifyShiftOp(Operation *op) {
  if (op->getOperand(0)->getType() != op->getResult(0)->getType()) {
    return op->emitError("expected the same type for the first operand and "
                         "result, but provided ")
           << op->getOperand(0)->getType() << " and "
           << op->getResult(0)->getType();
  }
  return success();
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

void spirv::AccessChainOp::build(Builder *builder, OperationState &state,
                                 Value *basePtr, ArrayRef<Value *> indices) {
  auto type = getElementPtrType(basePtr->getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, indices);
}

static ParseResult parseAccessChainOp(OpAsmParser &parser,
                                      OperationState &state) {
  OpAsmParser::OperandType ptrInfo;
  SmallVector<OpAsmParser::OperandType, 4> indicesInfo;
  Type type;
  // TODO(denis0x0D): regarding to the spec an index must be any integer type,
  // figure out how to use resolveOperand with a range of types and do not
  // fail on first attempt.
  Type indicesType = parser.getBuilder().getIntegerType(32);

  if (parser.parseOperand(ptrInfo) ||
      parser.parseOperandList(indicesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptrInfo, type, state.operands) ||
      parser.resolveOperands(indicesInfo, indicesType, state.operands)) {
    return failure();
  }

  auto resultType = getElementPtrType(
      type, llvm::makeArrayRef(state.operands).drop_front(), state.location);
  if (!resultType) {
    return failure();
  }

  state.addTypes(resultType);
  return success();
}

static void print(spirv::AccessChainOp op, OpAsmPrinter &printer) {
  printer << spirv::AccessChainOp::getOperationName() << ' ' << *op.base_ptr()
          << '[';
  printer.printOperands(op.indices());
  printer << "] : " << op.base_ptr()->getType();
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

namespace {

/// Combines chained `spirv::AccessChainOp` operations into one
/// `spirv::AccessChainOp` operation.
struct CombineChainedAccessChain
    : public OpRewritePattern<spirv::AccessChainOp> {
  using OpRewritePattern<spirv::AccessChainOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::AccessChainOp accessChainOp,
                                     PatternRewriter &rewriter) const override {
    auto parentAccessChainOp = dyn_cast_or_null<spirv::AccessChainOp>(
        accessChainOp.base_ptr()->getDefiningOp());

    if (!parentAccessChainOp) {
      return matchFailure();
    }

    // Combine indices.
    SmallVector<Value *, 4> indices(parentAccessChainOp.indices());
    indices.append(accessChainOp.indices().begin(),
                   accessChainOp.indices().end());

    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
        accessChainOp, parentAccessChainOp.base_ptr(), indices);

    return matchSuccess();
  }
};
} // end anonymous namespace

void spirv::AccessChainOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<CombineChainedAccessChain>(context);
}

//===----------------------------------------------------------------------===//
// spv._address_of
//===----------------------------------------------------------------------===//

void spirv::AddressOfOp::build(Builder *builder, OperationState &state,
                               spirv::GlobalVariableOp var) {
  build(builder, state, var.type(), builder->getSymbolRefAttr(var));
}

static ParseResult parseAddressOfOp(OpAsmParser &parser,
                                    OperationState &state) {
  FlatSymbolRefAttr varRefAttr;
  Type type;
  if (parser.parseAttribute(varRefAttr, Type(), kVariableAttrName,
                            state.attributes) ||
      parser.parseColonType(type)) {
    return failure();
  }
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected spv.ptr type");
  }
  state.addTypes(ptrType);
  return success();
}

static void print(spirv::AddressOfOp addressOfOp, OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  printer << spirv::AddressOfOp::getOperationName();

  // Print symbol name.
  printer << ' ';
  printer.printSymbolName(addressOfOp.variable());

  // Print the type.
  printer << " : " << addressOfOp.pointer()->getType();
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
// spv.AtomicCompareExchangeWeak
//===----------------------------------------------------------------------===//

static ParseResult parseAtomicCompareExchangeWeakOp(OpAsmParser &parser,
                                                    OperationState &state) {
  spirv::Scope memoryScope;
  spirv::MemorySemantics equalSemantics, unequalSemantics;
  SmallVector<OpAsmParser::OperandType, 3> operandInfo;
  Type type;
  if (parseEnumAttribute(memoryScope, parser, state, kMemoryScopeAttrName) ||
      parseEnumAttribute(equalSemantics, parser, state,
                         kEqualSemanticsAttrName) ||
      parseEnumAttribute(unequalSemantics, parser, state,
                         kUnequalSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, 3))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  if (parser.resolveOperands(
          operandInfo,
          {ptrType, ptrType.getPointeeType(), ptrType.getPointeeType()},
          parser.getNameLoc(), state.operands))
    return failure();

  return parser.addTypeToList(ptrType.getPointeeType(), state.types);
}

static void print(spirv::AtomicCompareExchangeWeakOp atomOp,
                  OpAsmPrinter &printer) {
  printer << spirv::AtomicCompareExchangeWeakOp::getOperationName() << " \""
          << stringifyScope(atomOp.memory_scope()) << "\" \""
          << stringifyMemorySemantics(atomOp.equal_semantics()) << "\" \""
          << stringifyMemorySemantics(atomOp.unequal_semantics()) << "\" ";
  printer.printOperands(atomOp.getOperands());
  printer << " : " << atomOp.pointer()->getType();
}

static LogicalResult verify(spirv::AtomicCompareExchangeWeakOp atomOp) {
  // According to the spec:
  // "The type of Value must be the same as Result Type. The type of the value
  // pointed to by Pointer must be the same as Result Type. This type must also
  // match the type of Comparator."
  if (atomOp.getType() != atomOp.value()->getType())
    return atomOp.emitOpError("value operand must have the same type as the op "
                              "result, but found ")
           << atomOp.value()->getType() << " vs " << atomOp.getType();

  if (atomOp.getType() != atomOp.comparator()->getType())
    return atomOp.emitOpError(
               "comparator operand must have the same type as the op "
               "result, but found ")
           << atomOp.comparator()->getType() << " vs " << atomOp.getType();

  Type pointeeType =
      atomOp.pointer()->getType().cast<spirv::PointerType>().getPointeeType();
  if (atomOp.getType() != pointeeType)
    return atomOp.emitOpError(
               "pointer operand's pointee type must have the same "
               "as the op result type, but found ")
           << pointeeType << " vs " << atomOp.getType();

  // TODO(antiagainst): Unequal cannot be set to Release or Acquire and Release.
  // In addition, Unequal cannot be set to a stronger memory-order then Equal.

  return success();
}

//===----------------------------------------------------------------------===//
// spv.BitcastOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::BitcastOp bitcastOp) {
  // TODO: The SPIR-V spec validation rules are different for different
  // versions.
  auto operandType = bitcastOp.operand()->getType();
  auto resultType = bitcastOp.result()->getType();
  if (operandType == resultType) {
    return bitcastOp.emitError(
        "result type must be different from operand type");
  }
  if (operandType.isa<spirv::PointerType>() &&
      !resultType.isa<spirv::PointerType>()) {
    return bitcastOp.emitError(
        "unhandled bit cast conversion from pointer type to non-pointer type");
  }
  if (!operandType.isa<spirv::PointerType>() &&
      resultType.isa<spirv::PointerType>()) {
    return bitcastOp.emitError(
        "unhandled bit cast conversion from non-pointer type to pointer type");
  }
  auto operandBitWidth = getBitWidth(operandType);
  auto resultBitWidth = getBitWidth(resultType);
  if (operandBitWidth != resultBitWidth) {
    return bitcastOp.emitOpError("mismatch in result type bitwidth ")
           << resultBitWidth << " and operand type bitwidth "
           << operandBitWidth;
  }
  return success();
}

void spirv::BitcastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ConvertChainedBitcast>(context);
}

//===----------------------------------------------------------------------===//
// spv.BitFieldInsert
//===----------------------------------------------------------------------===//

static ParseResult parseBitFieldInsertOp(OpAsmParser &parser,
                                         OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 4> operandInfo;
  Type baseType;
  Type offsetType;
  Type countType;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operandInfo, 4) || parser.parseColon() ||
      parser.parseType(baseType) || parser.parseComma() ||
      parser.parseType(offsetType) || parser.parseComma() ||
      parser.parseType(countType) ||
      parser.resolveOperands(operandInfo,
                             {baseType, baseType, offsetType, countType}, loc,
                             state.operands)) {
    return failure();
  }
  state.addTypes(baseType);
  return success();
}

static void print(spirv::BitFieldInsertOp bitFieldInsertOp,
                  OpAsmPrinter &printer) {
  printer << spirv::BitFieldInsertOp::getOperationName() << ' ';
  printer.printOperands(bitFieldInsertOp.getOperands());
  printer << " : " << bitFieldInsertOp.base()->getType() << ", "
          << bitFieldInsertOp.offset()->getType() << ", "
          << bitFieldInsertOp.count()->getType();
}

static LogicalResult verify(spirv::BitFieldInsertOp bitFieldOp) {
  auto baseType = bitFieldOp.base()->getType();
  auto insertType = bitFieldOp.insert()->getType();
  auto resultType = bitFieldOp.getResult()->getType();

  if ((baseType != insertType) || (baseType != resultType)) {
    return bitFieldOp.emitError("expected the same type for the base operand, "
                                "insert operand, and "
                                "result, but provided ")
           << baseType << ", " << insertType << " and " << resultType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.BranchOp
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState &state) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  state.addSuccessor(dest, destOperands);
  return success();
}

static void print(spirv::BranchOp branchOp, OpAsmPrinter &printer) {
  printer << spirv::BranchOp::getOperationName() << ' ';
  printer.printSuccessorAndUseList(branchOp.getOperation(), /*index=*/0);
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

static ParseResult parseBranchConditionalOp(OpAsmParser &parser,
                                            OperationState &state) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType condInfo;
  Block *dest;
  SmallVector<Value *, 4> destOperands;

  // Parse the condition.
  Type boolTy = builder.getI1Type();
  if (parser.parseOperand(condInfo) ||
      parser.resolveOperand(condInfo, boolTy, state.operands))
    return failure();

  // Parse the optional branch weights.
  if (succeeded(parser.parseOptionalLSquare())) {
    IntegerAttr trueWeight, falseWeight;
    SmallVector<NamedAttribute, 2> weights;

    auto i32Type = builder.getIntegerType(32);
    if (parser.parseAttribute(trueWeight, i32Type, "weight", weights) ||
        parser.parseComma() ||
        parser.parseAttribute(falseWeight, i32Type, "weight", weights) ||
        parser.parseRSquare())
      return failure();

    state.addAttribute(kBranchWeightAttrName,
                       builder.getArrayAttr({trueWeight, falseWeight}));
  }

  // Parse the true branch.
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  state.addSuccessor(dest, destOperands);

  // Parse the false branch.
  destOperands.clear();
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  state.addSuccessor(dest, destOperands);

  return success();
}

static void print(spirv::BranchConditionalOp branchOp, OpAsmPrinter &printer) {
  printer << spirv::BranchConditionalOp::getOperationName() << ' ';
  printer.printOperand(branchOp.condition());

  if (auto weights = branchOp.branch_weights()) {
    printer << " [";
    interleaveComma(weights->getValue(), printer, [&](Attribute a) {
      printer << a.cast<IntegerAttr>().getInt();
    });
    printer << "]";
  }

  printer << ", ";
  printer.printSuccessorAndUseList(branchOp.getOperation(),
                                   spirv::BranchConditionalOp::kTrueIndex);
  printer << ", ";
  printer.printSuccessorAndUseList(branchOp.getOperation(),
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

static ParseResult parseCompositeExtractOp(OpAsmParser &parser,
                                           OperationState &state) {
  OpAsmParser::OperandType compositeInfo;
  Attribute indicesAttr;
  Type compositeType;
  llvm::SMLoc attrLocation;
  int32_t index;

  if (parser.parseOperand(compositeInfo) ||
      parser.getCurrentLocation(&attrLocation) ||
      parser.parseAttribute(indicesAttr, kIndicesAttrName, state.attributes) ||
      parser.parseColonType(compositeType) ||
      parser.resolveOperand(compositeInfo, compositeType, state.operands)) {
    return failure();
  }

  auto indicesArrayAttr = indicesAttr.dyn_cast<ArrayAttr>();
  if (!indicesArrayAttr) {
    return parser.emitError(
        attrLocation,
        "expected an 32-bit integer array attribute for 'indices'");
  }

  if (!indicesArrayAttr.size()) {
    return parser.emitError(
        attrLocation, "expected at least one index for spv.CompositeExtract");
  }

  Type resultType = compositeType;
  for (auto indexAttr : indicesArrayAttr) {
    if (auto indexIntAttr = indexAttr.dyn_cast<IntegerAttr>()) {
      index = indexIntAttr.getInt();
    } else {
      return parser.emitError(
                 attrLocation,
                 "expected an 32-bit integer for index, but found '")
             << indexAttr << "'";
    }

    if (auto cType = resultType.dyn_cast<spirv::CompositeType>()) {
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        return parser.emitError(attrLocation, "index ")
               << index << " out of bounds for " << resultType;
      }
      resultType = cType.getElementType(index);
    } else {
      return parser.emitError(attrLocation,
                              "cannot extract from non-composite type ")
             << resultType << " with index " << index;
    }
  }

  state.addTypes(resultType);
  return success();
}

static void print(spirv::CompositeExtractOp compositeExtractOp,
                  OpAsmPrinter &printer) {
  printer << spirv::CompositeExtractOp::getOperationName() << ' '
          << *compositeExtractOp.composite() << compositeExtractOp.indices()
          << " : " << compositeExtractOp.composite()->getType();
}

static LogicalResult verify(spirv::CompositeExtractOp compExOp) {
  auto indicesArrayAttr = compExOp.indices().dyn_cast<ArrayAttr>();
  auto resultType = getElementType(compExOp.composite()->getType(),
                                   indicesArrayAttr, compExOp.getLoc());
  if (!resultType)
    return failure();

  if (resultType != compExOp.getType()) {
    return compExOp.emitOpError("invalid result type: expected ")
           << resultType << " but provided " << compExOp.getType();
  }

  return success();
}

OpFoldResult spirv::CompositeExtractOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "spv.CompositeExtract expects one operand");
  auto indexVector = functional::map(
      [](Attribute attr) {
        return static_cast<unsigned>(attr.cast<IntegerAttr>().getInt());
      },
      indices());
  return extractCompositeElement(operands[0], indexVector);
}

//===----------------------------------------------------------------------===//
// spv.CompositeInsert
//===----------------------------------------------------------------------===//

static ParseResult parseCompositeInsertOp(OpAsmParser &parser,
                                          OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  Type objectType, compositeType;
  Attribute indicesAttr;
  auto loc = parser.getCurrentLocation();

  return failure(
      parser.parseOperandList(operands, 2) ||
      parser.parseAttribute(indicesAttr, kIndicesAttrName, state.attributes) ||
      parser.parseColonType(objectType) ||
      parser.parseKeywordType("into", compositeType) ||
      parser.resolveOperands(operands, {objectType, compositeType}, loc,
                             state.operands) ||
      parser.addTypesToList(compositeType, state.types));
}

static LogicalResult verify(spirv::CompositeInsertOp compositeInsertOp) {
  auto indicesArrayAttr = compositeInsertOp.indices().dyn_cast<ArrayAttr>();
  auto objectType =
      getElementType(compositeInsertOp.composite()->getType(), indicesArrayAttr,
                     compositeInsertOp.getLoc());
  if (!objectType)
    return failure();

  if (objectType != compositeInsertOp.object()->getType()) {
    return compositeInsertOp.emitOpError("object operand type should be ")
           << objectType << ", but found "
           << compositeInsertOp.object()->getType();
  }

  if (compositeInsertOp.composite()->getType() != compositeInsertOp.getType()) {
    return compositeInsertOp.emitOpError("result type should be the same as "
                                         "the composite type, but found ")
           << compositeInsertOp.composite()->getType() << " vs "
           << compositeInsertOp.getType();
  }

  return success();
}

static void print(spirv::CompositeInsertOp compositeInsertOp,
                  OpAsmPrinter &printer) {
  printer << spirv::CompositeInsertOp::getOperationName() << " "
          << *compositeInsertOp.object() << ", "
          << *compositeInsertOp.composite() << compositeInsertOp.indices()
          << " : " << compositeInsertOp.object()->getType() << " into "
          << compositeInsertOp.composite()->getType();
}

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

static ParseResult parseConstantOp(OpAsmParser &parser, OperationState &state) {
  Attribute value;
  if (parser.parseAttribute(value, kValueAttrName, state.attributes))
    return failure();

  Type type = value.getType();
  if (type.isa<NoneType>() || type.isa<TensorType>()) {
    if (parser.parseColonType(type))
      return failure();
  }

  return parser.addTypeToList(type, state.types);
}

static void print(spirv::ConstantOp constOp, OpAsmPrinter &printer) {
  printer << spirv::ConstantOp::getOperationName() << ' ' << constOp.value();
  if (constOp.getType().isa<spirv::ArrayType>()) {
    printer << " : " << constOp.getType();
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
  case StandardAttributes::Float: {
    if (valueType != opType)
      return constOp.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  } break;
  case StandardAttributes::DenseElements:
  case StandardAttributes::SparseElements: {
    if (valueType == opType)
      break;
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    auto shapedType = valueType.dyn_cast<ShapedType>();
    if (!arrayType) {
      return constOp.emitOpError(
          "must have spv.array result type for array value");
    }

    int numElements = arrayType.getNumElements();
    auto opElemType = arrayType.getElementType();
    while (auto t = opElemType.dyn_cast<spirv::ArrayType>()) {
      numElements *= t.getNumElements();
      opElemType = t.getElementType();
    }
    if (!opElemType.isIntOrFloat()) {
      return constOp.emitOpError("only support nested array result type");
    }

    auto valueElemType = shapedType.getElementType();
    if (valueElemType != opElemType) {
      return constOp.emitOpError("result element type (")
             << opElemType << ") does not match value element type ("
             << valueElemType << ")";
    }

    if (numElements != shapedType.getNumElements()) {
      return constOp.emitOpError("result number of elements (")
             << numElements << ") does not match value number of elements ("
             << shapedType.getNumElements() << ")";
    }
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
  assert(operands.empty() && "spv.constant has no operands");
  return value();
}

bool spirv::ConstantOp::isBuildableWith(Type type) {
  // Must be valid SPIR-V type first.
  if (!SPIRVDialect::isValidType(type))
    return false;

  if (type.getKind() >= Type::FIRST_SPIRV_TYPE &&
      type.getKind() <= spirv::TypeKind::LAST_SPIRV_TYPE) {
    // TODO(antiagainst): support constant struct
    return type.isa<spirv::ArrayType>();
  }

  return true;
}

spirv::ConstantOp spirv::ConstantOp::getZero(Type type, Location loc,
                                             OpBuilder *builder) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    Attribute val;
    if (width == 1)
      return builder->create<spirv::ConstantOp>(loc, type,
                                                builder->getBoolAttr(false));
    return builder->create<spirv::ConstantOp>(
        loc, type, builder->getIntegerAttr(type, APInt(width, 0)));
  }

  llvm_unreachable("unimplemented types for ConstantOp::getZero()");
}

spirv::ConstantOp spirv::ConstantOp::getOne(Type type, Location loc,
                                            OpBuilder *builder) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    if (width == 1)
      return builder->create<spirv::ConstantOp>(loc, type,
                                                builder->getBoolAttr(true));
    return builder->create<spirv::ConstantOp>(
        loc, type, builder->getIntegerAttr(type, APInt(width, 1)));
  }

  llvm_unreachable("unimplemented types for ConstantOp::getOne()");
}

//===----------------------------------------------------------------------===//
// spv.ControlBarrier
//===----------------------------------------------------------------------===//

static ParseResult parseControlBarrierOp(OpAsmParser &parser,
                                         OperationState &state) {
  spirv::Scope executionScope;
  spirv::Scope memoryScope;
  spirv::MemorySemantics memorySemantics;

  return failure(
      parseEnumAttribute(executionScope, parser, state,
                         kExecutionScopeAttrName) ||
      parser.parseComma() ||
      parseEnumAttribute(memoryScope, parser, state, kMemoryScopeAttrName) ||
      parser.parseComma() ||
      parseEnumAttribute(memorySemantics, parser, state));
}

static void print(spirv::ControlBarrierOp op, OpAsmPrinter &printer) {
  printer << spirv::ControlBarrierOp::getOperationName() << " \""
          << stringifyScope(op.execution_scope()) << "\", \""
          << stringifyScope(op.memory_scope()) << "\", \""
          << stringifyMemorySemantics(op.memory_semantics()) << "\"";
}

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

void spirv::EntryPointOp::build(Builder *builder, OperationState &state,
                                spirv::ExecutionModel executionModel,
                                FuncOp function,
                                ArrayRef<Attribute> interfaceVars) {
  build(builder, state,
        builder->getI32IntegerAttr(static_cast<int32_t>(executionModel)),
        builder->getSymbolRefAttr(function),
        builder->getArrayAttr(interfaceVars));
}

static ParseResult parseEntryPointOp(OpAsmParser &parser,
                                     OperationState &state) {
  spirv::ExecutionModel execModel;
  SmallVector<OpAsmParser::OperandType, 0> identifiers;
  SmallVector<Type, 0> idTypes;
  SmallVector<Attribute, 4> interfaceVars;

  FlatSymbolRefAttr fn;
  if (parseEnumAttribute(execModel, parser, state) ||
      parser.parseAttribute(fn, Type(), kFnNameAttrName, state.attributes)) {
    return failure();
  }

  if (!parser.parseOptionalComma()) {
    // Parse the interface variables
    do {
      // The name of the interface variable attribute isnt important
      auto attrName = "var_symbol";
      FlatSymbolRefAttr var;
      SmallVector<NamedAttribute, 1> attrs;
      if (parser.parseAttribute(var, Type(), attrName, attrs)) {
        return failure();
      }
      interfaceVars.push_back(var);
    } while (!parser.parseOptionalComma());
  }
  state.addAttribute(kInterfaceAttrName,
                     parser.getBuilder().getArrayAttr(interfaceVars));
  return success();
}

static void print(spirv::EntryPointOp entryPointOp, OpAsmPrinter &printer) {
  printer << spirv::EntryPointOp::getOperationName() << " \""
          << stringifyExecutionModel(entryPointOp.execution_model()) << "\" ";
  printer.printSymbolName(entryPointOp.fn());
  auto interfaceVars = entryPointOp.interface().getValue();
  if (!interfaceVars.empty()) {
    printer << ", ";
    interleaveComma(interfaceVars, printer);
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

void spirv::ExecutionModeOp::build(Builder *builder, OperationState &state,
                                   FuncOp function,
                                   spirv::ExecutionMode executionMode,
                                   ArrayRef<int32_t> params) {
  build(builder, state, builder->getSymbolRefAttr(function),
        builder->getI32IntegerAttr(static_cast<int32_t>(executionMode)),
        builder->getI32ArrayAttr(params));
}

static ParseResult parseExecutionModeOp(OpAsmParser &parser,
                                        OperationState &state) {
  spirv::ExecutionMode execMode;
  Attribute fn;
  if (parser.parseAttribute(fn, kFnNameAttrName, state.attributes) ||
      parseEnumAttribute(execMode, parser, state)) {
    return failure();
  }

  SmallVector<int32_t, 4> values;
  Type i32Type = parser.getBuilder().getIntegerType(32);
  while (!parser.parseOptionalComma()) {
    SmallVector<NamedAttribute, 1> attr;
    Attribute value;
    if (parser.parseAttribute(value, i32Type, "value", attr)) {
      return failure();
    }
    values.push_back(value.cast<IntegerAttr>().getInt());
  }
  state.addAttribute(kValuesAttrName,
                     parser.getBuilder().getI32ArrayAttr(values));
  return success();
}

static void print(spirv::ExecutionModeOp execModeOp, OpAsmPrinter &printer) {
  printer << spirv::ExecutionModeOp::getOperationName() << " @"
          << execModeOp.fn() << " \""
          << stringifyExecutionMode(execModeOp.execution_mode()) << "\"";
  auto values = execModeOp.values();
  if (!values.size()) {
    return;
  }
  printer << ", ";
  interleaveComma(values, printer, [&](Attribute a) {
    printer << a.cast<IntegerAttr>().getInt();
  });
}

//===----------------------------------------------------------------------===//
// spv.FunctionCall
//===----------------------------------------------------------------------===//

static ParseResult parseFunctionCallOp(OpAsmParser &parser,
                                       OperationState &state) {
  FlatSymbolRefAttr calleeAttr;
  FunctionType type;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto loc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, kCallee, state.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseColonType(type)) {
    return failure();
  }

  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType) {
    return parser.emitError(loc, "expected function type, but provided ")
           << type;
  }

  if (funcType.getNumResults() > 1) {
    return parser.emitError(loc, "expected callee function to have 0 or 1 "
                                 "result, but provided ")
           << funcType.getNumResults();
  }

  return failure(parser.addTypesToList(funcType.getResults(), state.types) ||
                 parser.resolveOperands(operands, funcType.getInputs(), loc,
                                        state.operands));
}

static void print(spirv::FunctionCallOp functionCallOp, OpAsmPrinter &printer) {
  SmallVector<Type, 4> argTypes(functionCallOp.getOperandTypes());
  SmallVector<Type, 1> resultTypes(functionCallOp.getResultTypes());
  Type functionType =
      FunctionType::get(argTypes, resultTypes, functionCallOp.getContext());

  printer << spirv::FunctionCallOp::getOperationName() << ' '
          << functionCallOp.getAttr(kCallee) << '(';
  printer.printOperands(functionCallOp.arguments());
  printer << ") : " << functionType;
}

static LogicalResult verify(spirv::FunctionCallOp functionCallOp) {
  auto fnName = functionCallOp.callee();

  auto moduleOp = functionCallOp.getParentOfType<spirv::ModuleOp>();
  if (!moduleOp) {
    return functionCallOp.emitOpError(
        "must appear in a function inside 'spv.module'");
  }

  auto funcOp = moduleOp.lookupSymbol<FuncOp>(fnName);
  if (!funcOp) {
    return functionCallOp.emitOpError("callee function '")
           << fnName << "' not found in 'spv.module'";
  }

  auto functionType = funcOp.getType();

  if (functionCallOp.getNumResults() > 1) {
    return functionCallOp.emitOpError(
               "expected callee function to have 0 or 1 result, but provided ")
           << functionCallOp.getNumResults();
  }

  if (functionType.getNumInputs() != functionCallOp.getNumOperands()) {
    return functionCallOp.emitOpError(
               "has incorrect number of operands for callee: expected ")
           << functionType.getNumInputs() << ", but provided "
           << functionCallOp.getNumOperands();
  }

  for (uint32_t i = 0, e = functionType.getNumInputs(); i != e; ++i) {
    if (functionCallOp.getOperand(i)->getType() != functionType.getInput(i)) {
      return functionCallOp.emitOpError(
                 "operand type mismatch: expected operand type ")
             << functionType.getInput(i) << ", but provided "
             << functionCallOp.getOperand(i)->getType()
             << " for operand number " << i;
    }
  }

  if (functionType.getNumResults() != functionCallOp.getNumResults()) {
    return functionCallOp.emitOpError(
               "has incorrect number of results has for callee: expected ")
           << functionType.getNumResults() << ", but provided "
           << functionCallOp.getNumResults();
  }

  if (functionCallOp.getNumResults() &&
      (functionCallOp.getResult(0)->getType() != functionType.getResult(0))) {
    return functionCallOp.emitOpError("result type mismatch: expected ")
           << functionType.getResult(0) << ", but provided "
           << functionCallOp.getResult(0)->getType();
  }

  return success();
}

CallInterfaceCallable spirv::FunctionCallOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>(kCallee);
}

Operation::operand_range spirv::FunctionCallOp::getArgOperands() {
  return arguments();
}

//===----------------------------------------------------------------------===//
// spv.globalVariable
//===----------------------------------------------------------------------===//

void spirv::GlobalVariableOp::build(Builder *builder, OperationState &state,
                                    Type type, StringRef name,
                                    unsigned descriptorSet, unsigned binding) {
  build(builder, state, TypeAttr::get(type), builder->getStringAttr(name),
        nullptr);
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::DescriptorSet),
      builder->getI32IntegerAttr(descriptorSet));
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::Binding),
      builder->getI32IntegerAttr(binding));
}

static ParseResult parseGlobalVariableOp(OpAsmParser &parser,
                                         OperationState &state) {
  // Parse variable name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes)) {
    return failure();
  }

  // Parse optional initializer
  if (succeeded(parser.parseOptionalKeyword(kInitializerAttrName))) {
    FlatSymbolRefAttr initSymbol;
    if (parser.parseLParen() ||
        parser.parseAttribute(initSymbol, Type(), kInitializerAttrName,
                              state.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, state)) {
    return failure();
  }

  Type type;
  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type)) {
    return failure();
  }
  if (!type.isa<spirv::PointerType>()) {
    return parser.emitError(loc, "expected spv.ptr type");
  }
  state.addAttribute(kTypeAttrName, TypeAttr::get(type));

  return success();
}

static void print(spirv::GlobalVariableOp varOp, OpAsmPrinter &printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  printer << spirv::GlobalVariableOp::getOperationName();

  // Print variable name.
  printer << ' ';
  printer.printSymbolName(varOp.sym_name());
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());

  // Print optional initializer
  if (auto initializer = varOp.initializer()) {
    printer << " " << kInitializerAttrName << '(';
    printer.printSymbolName(initializer.getValue());
    printer << ')';
    elidedAttrs.push_back(kInitializerAttrName);
  }

  elidedAttrs.push_back(kTypeAttrName);
  printVariableDecorations(op, printer, elidedAttrs);
  printer << " : " << varOp.type();
}

static LogicalResult verify(spirv::GlobalVariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (varOp.storageClass() == spirv::StorageClass::Generic)
    return varOp.emitOpError("storage class cannot be 'Generic'");

  if (auto init =
          varOp.getAttrOfType<FlatSymbolRefAttr>(kInitializerAttrName)) {
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
// spv.GroupNonUniformBallotOp
//===----------------------------------------------------------------------===//

static ParseResult parseGroupNonUniformBallotOp(OpAsmParser &parser,
                                                OperationState &state) {
  spirv::Scope executionScope;
  OpAsmParser::OperandType operandInfo;
  Type resultType;
  IntegerType i1Type = parser.getBuilder().getI1Type();
  if (parseEnumAttribute(executionScope, parser, state,
                         kExecutionScopeAttrName) ||
      parser.parseOperand(operandInfo) || parser.parseColonType(resultType) ||
      parser.resolveOperand(operandInfo, i1Type, state.operands))
    return failure();

  return parser.addTypeToList(resultType, state.types);
}

static void print(spirv::GroupNonUniformBallotOp ballotOp,
                  OpAsmPrinter &printer) {
  printer << spirv::GroupNonUniformBallotOp::getOperationName() << " \""
          << stringifyScope(ballotOp.execution_scope()) << "\" ";
  printer.printOperand(ballotOp.predicate());
  printer << " : " << ballotOp.getType();
}

static LogicalResult verify(spirv::GroupNonUniformBallotOp ballotOp) {
  // TODO(antiagainst): check the result integer type's signedness bit is 0.

  spirv::Scope scope = ballotOp.execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return ballotOp.emitOpError(
        "execution scope must be 'Workgroup' or 'Subgroup'");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IAddOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "spv.IAdd expects two operands");
  // lhs + 0 = lhs
  if (matchPattern(operand2(), m_Zero()))
    return operand1();

  return nullptr;
}

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IMulOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "spv.IMul expects two operands");
  // lhs * 0 == 0
  if (matchPattern(operand2(), m_Zero()))
    return operand2();
  // lhs * 1 = lhs
  if (matchPattern(operand2(), m_One()))
    return operand1();

  return nullptr;
}

//===----------------------------------------------------------------------===//
// spv.LoadOp
//===----------------------------------------------------------------------===//

void spirv::LoadOp::build(Builder *builder, OperationState &state,
                          Value *basePtr, IntegerAttr memory_access,
                          IntegerAttr alignment) {
  auto ptrType = basePtr->getType().cast<spirv::PointerType>();
  build(builder, state, ptrType.getPointeeType(), basePtr, memory_access,
        alignment);
}

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::OperandType ptrInfo;
  Type elementType;
  if (parseEnumAttribute(storageClass, parser) ||
      parser.parseOperand(ptrInfo) ||
      parseMemoryAccessAttributes(parser, state) ||
      parser.parseOptionalAttrDict(state.attributes) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser.resolveOperand(ptrInfo, ptrType, state.operands)) {
    return failure();
  }

  state.addTypes(elementType);
  return success();
}

static void print(spirv::LoadOp loadOp, OpAsmPrinter &printer) {
  auto *op = loadOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      loadOp.ptr()->getType().cast<spirv::PointerType>().getStorageClass());
  printer << spirv::LoadOp::getOperationName() << " \"" << sc << "\" ";
  // Print the pointer operand.
  printer.printOperand(loadOp.ptr());

  printMemoryAccessAttribute(loadOp, printer, elidedAttrs);

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  printer << " : " << loadOp.getType();
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
// spv.LogicalNot
//===----------------------------------------------------------------------===//

void spirv::LogicalNotOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ConvertLogicalNotOfIEqual, ConvertLogicalNotOfINotEqual,
                 ConvertLogicalNotOfLogicalEqual,
                 ConvertLogicalNotOfLogicalNotEqual>(context);
}

//===----------------------------------------------------------------------===//
// spv.loop
//===----------------------------------------------------------------------===//

void spirv::LoopOp::build(Builder *builder, OperationState &state) {
  state.addAttribute("loop_control",
                     builder->getI32IntegerAttr(
                         static_cast<uint32_t>(spirv::LoopControl::None)));
  state.addRegion();
}

static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &state) {
  // TODO(antiagainst): support loop control properly
  Builder builder = parser.getBuilder();
  state.addAttribute("loop_control",
                     builder.getI32IntegerAttr(
                         static_cast<uint32_t>(spirv::LoopControl::None)));

  return parser.parseRegion(*state.addRegion(), /*arguments=*/{},
                            /*argTypes=*/{});
}

static void print(spirv::LoopOp loopOp, OpAsmPrinter &printer) {
  auto *op = loopOp.getOperation();

  printer << spirv::LoopOp::getOperationName();
  printer.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

/// Returns true if the given `srcBlock` contains only one `spv.Branch` to the
/// given `dstBlock`.
static inline bool hasOneBranchOpTo(Block &srcBlock, Block &dstBlock) {
  // Check that there is only one op in the `srcBlock`.
  if (!has_single_element(srcBlock))
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

Block *spirv::LoopOp::getEntryBlock() {
  assert(!body().empty() && "op region should not be empty!");
  return &body().front();
}

Block *spirv::LoopOp::getHeaderBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The second block is the loop header block.
  return &*std::next(body().begin());
}

Block *spirv::LoopOp::getContinueBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The second to last block is the loop continue block.
  return &*std::prev(body().end(), 2);
}

Block *spirv::LoopOp::getMergeBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The last block is the loop merge block.
  return &body().back();
}

void spirv::LoopOp::addEntryAndMergeBlock() {
  assert(body().empty() && "entry and merge block already exist");
  body().push_back(new Block());
  auto *mergeBlock = new Block();
  body().push_back(mergeBlock);
  OpBuilder builder(mergeBlock);

  // Add a spv._merge op into the merge block.
  builder.create<spirv::MergeOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// spv._merge
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::MergeOp mergeOp) {
  auto *parentOp = mergeOp.getParentOp();
  if (!parentOp ||
      (!isa<spirv::SelectionOp>(parentOp) && !isa<spirv::LoopOp>(parentOp)))
    return mergeOp.emitOpError(
        "expected parent op to be 'spv.selection' or 'spv.loop'");

  Block &parentLastBlock = mergeOp.getParentRegion()->back();
  if (mergeOp.getOperation() != parentLastBlock.getTerminator())
    return mergeOp.emitOpError(
        "can only be used in the last block of 'spv.selection' or 'spv.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// spv.MemoryBarrier
//===----------------------------------------------------------------------===//

static ParseResult parseMemoryBarrierOp(OpAsmParser &parser,
                                        OperationState &state) {
  spirv::Scope memoryScope;
  spirv::MemorySemantics memorySemantics;

  return failure(
      parseEnumAttribute(memoryScope, parser, state, kMemoryScopeAttrName) ||
      parser.parseComma() ||
      parseEnumAttribute(memorySemantics, parser, state));
}

static void print(spirv::MemoryBarrierOp op, OpAsmPrinter &printer) {
  printer << spirv::MemoryBarrierOp::getOperationName() << " \""
          << stringifyScope(op.memory_scope()) << "\", \""
          << stringifyMemorySemantics(op.memory_semantics()) << "\"";
}

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

void spirv::ModuleOp::build(Builder *builder, OperationState &state) {
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

void spirv::ModuleOp::build(Builder *builder, OperationState &state,
                            IntegerAttr addressing_model,
                            IntegerAttr memory_model, ArrayAttr capabilities,
                            ArrayAttr extensions,
                            ArrayAttr extended_instruction_sets) {
  state.addAttribute("addressing_model", addressing_model);
  state.addAttribute("memory_model", memory_model);
  if (capabilities)
    state.addAttribute("capabilities", capabilities);
  if (extensions)
    state.addAttribute("extensions", extensions);
  if (extended_instruction_sets)
    state.addAttribute("extended_instruction_sets", extended_instruction_sets);
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

static ParseResult parseModuleOp(OpAsmParser &parser, OperationState &state) {
  Region *body = state.addRegion();

  // Parse attributes
  spirv::AddressingModel addrModel;
  spirv::MemoryModel memoryModel;
  if (parseEnumAttribute(addrModel, parser, state) ||
      parseEnumAttribute(memoryModel, parser, state)) {
    return failure();
  }

  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  spirv::ModuleOp::ensureTerminator(*body, parser.getBuilder(), state.location);
  return success();
}

static void print(spirv::ModuleOp moduleOp, OpAsmPrinter &printer) {
  auto *op = moduleOp.getOperation();

  // Only print out addressing model and memory model in a nicer way if both
  // presents. Otherwise, print them in the general form. This helps debugging
  // ill-formed ModuleOp.
  SmallVector<StringRef, 2> elidedAttrs;
  auto addressingModelAttrName = spirv::attributeName<spirv::AddressingModel>();
  auto memoryModelAttrName = spirv::attributeName<spirv::MemoryModel>();
  if (op->getAttr(addressingModelAttrName) &&
      op->getAttr(memoryModelAttrName)) {
    printer << spirv::ModuleOp::getOperationName() << " \""
            << spirv::stringifyAddressingModel(moduleOp.addressing_model())
            << "\" \"" << spirv::stringifyMemoryModel(moduleOp.memory_model())
            << '"';
    elidedAttrs.assign({addressingModelAttrName, memoryModelAttrName});
  }

  printer.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
  printer.printOptionalAttrDictWithKeyword(op->getAttrs(), elidedAttrs);
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
          for (Attribute varRef : interface) {
            auto varSymRef = varRef.dyn_cast<FlatSymbolRefAttr>();
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

static ParseResult parseReferenceOfOp(OpAsmParser &parser,
                                      OperationState &state) {
  FlatSymbolRefAttr constRefAttr;
  Type type;
  if (parser.parseAttribute(constRefAttr, Type(), kSpecConstAttrName,
                            state.attributes) ||
      parser.parseColonType(type)) {
    return failure();
  }
  return parser.addTypeToList(type, state.types);
}

static void print(spirv::ReferenceOfOp referenceOfOp, OpAsmPrinter &printer) {
  printer << spirv::ReferenceOfOp::getOperationName() << ' ';
  printer.printSymbolName(referenceOfOp.spec_const());
  printer << " : " << referenceOfOp.reference()->getType();
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
  auto funcOp = returnOp.getParentOfType<FuncOp>();
  auto numOutputs = funcOp.getType().getNumResults();
  if (numOutputs != 0)
    return returnOp.emitOpError("cannot be used in functions returning value")
           << (numOutputs > 1 ? "s" : "");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

static ParseResult parseReturnValueOp(OpAsmParser &parser,
                                      OperationState &state) {
  OpAsmParser::OperandType retValInfo;
  Type retValType;
  return failure(parser.parseOperand(retValInfo) ||
                 parser.parseColonType(retValType) ||
                 parser.resolveOperand(retValInfo, retValType, state.operands));
}

static void print(spirv::ReturnValueOp retValOp, OpAsmPrinter &printer) {
  printer << spirv::ReturnValueOp::getOperationName() << ' ';
  printer.printOperand(retValOp.value());
  printer << " : " << retValOp.value()->getType();
}

static LogicalResult verify(spirv::ReturnValueOp retValOp) {
  auto funcOp = retValOp.getParentOfType<FuncOp>();
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

void spirv::SelectOp::build(Builder *builder, OperationState &state,
                            Value *cond, Value *trueValue, Value *falseValue) {
  build(builder, state, trueValue->getType(), cond, trueValue, falseValue);
}

static ParseResult parseSelectOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType condition;
  SmallVector<OpAsmParser::OperandType, 2> operands;
  SmallVector<Type, 2> types;
  auto loc = parser.getCurrentLocation();
  if (parser.parseOperand(condition) || parser.parseComma() ||
      parser.parseOperandList(operands, 2) ||
      parser.parseColonTypeList(types)) {
    return failure();
  }
  if (types.size() != 2) {
    return parser.emitError(
        loc, "need exactly two trailing types for select condition and object");
  }
  if (parser.resolveOperand(condition, types[0], state.operands) ||
      parser.resolveOperands(operands, types[1], state.operands)) {
    return failure();
  }
  return parser.addTypesToList(types[1], state.types);
}

static void print(spirv::SelectOp op, OpAsmPrinter &printer) {
  printer << spirv::SelectOp::getOperationName() << " ";

  // Print the operands.
  printer.printOperands(op.getOperands());

  // Print colon and types.
  printer << " : " << op.condition()->getType() << ", "
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
// spv.selection
//===----------------------------------------------------------------------===//

static ParseResult parseSelectionOp(OpAsmParser &parser,
                                    OperationState &state) {
  // TODO(antiagainst): support selection control properly
  Builder builder = parser.getBuilder();
  state.addAttribute("selection_control",
                     builder.getI32IntegerAttr(
                         static_cast<uint32_t>(spirv::SelectionControl::None)));

  return parser.parseRegion(*state.addRegion(), /*arguments=*/{},
                            /*argTypes=*/{});
}

static void print(spirv::SelectionOp selectionOp, OpAsmPrinter &printer) {
  auto *op = selectionOp.getOperation();

  printer << spirv::SelectionOp::getOperationName();
  printer.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

static LogicalResult verify(spirv::SelectionOp selectionOp) {
  auto *op = selectionOp.getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +--------------+
  //                     | header block |
  //                     +--------------+
  //                          / | \
  //                           ...
  //
  //
  //         +---------+   +---------+   +---------+
  //         | case #0 |   | case #1 |   | case #2 |  ...
  //         +---------+   +---------+   +---------+
  //
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
  if (!isMergeBlock(region.back()))
    return selectionOp.emitOpError(
        "last block must be the merge block with only one 'spv._merge' op");

  if (std::next(region.begin()) == region.end())
    return selectionOp.emitOpError("must have a selection header block");

  return success();
}

Block *spirv::SelectionOp::getHeaderBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The first block is the loop header block.
  return &body().front();
}

Block *spirv::SelectionOp::getMergeBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The last block is the loop merge block.
  return &body().back();
}

void spirv::SelectionOp::addMergeBlock() {
  assert(body().empty() && "entry and merge block already exist");
  auto *mergeBlock = new Block();
  body().push_back(mergeBlock);
  OpBuilder builder(mergeBlock);

  // Add a spv._merge op into the merge block.
  builder.create<spirv::MergeOp>(getLoc());
}

namespace {
// Blocks from the given `spv.selection` operation must satisfy the following
// layout:
//
//       +-----------------------------------------------+
//       | header block                                  |
//       | spv.BranchConditionalOp %cond, ^case0, ^case1 |
//       +-----------------------------------------------+
//                            /   \
//                             ...
//
//
//   +------------------------+    +------------------------+
//   | case #0                |    | case #1                |
//   | spv.Store %ptr %value0 |    | spv.Store %ptr %value1 |
//   | spv.Branch ^merge      |    | spv.Branch ^merge      |
//   +------------------------+    +------------------------+
//
//
//                             ...
//                            \   /
//                              v
//                       +-------------+
//                       | merge block |
//                       +-------------+
//
struct ConvertSelectionOpToSelect
    : public OpRewritePattern<spirv::SelectionOp> {
  using OpRewritePattern<spirv::SelectionOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::SelectionOp selectionOp,
                                     PatternRewriter &rewriter) const override {
    auto *op = selectionOp.getOperation();
    auto &body = op->getRegion(0);
    // Verifier allows an empty region for `spv.selection`.
    if (body.empty()) {
      return matchFailure();
    }

    // Check that region consists of 4 blocks:
    // header block, `true` block, `false` block and merge block.
    if (std::distance(body.begin(), body.end()) != 4) {
      return matchFailure();
    }

    auto *headerBlock = selectionOp.getHeaderBlock();
    if (!onlyContainsBranchConditionalOp(headerBlock)) {
      return matchFailure();
    }

    auto brConditionalOp =
        cast<spirv::BranchConditionalOp>(headerBlock->front());

    auto *trueBlock = brConditionalOp.getSuccessor(0);
    auto *falseBlock = brConditionalOp.getSuccessor(1);
    auto *mergeBlock = selectionOp.getMergeBlock();

    if (!canCanonicalizeSelection(trueBlock, falseBlock, mergeBlock)) {
      return matchFailure();
    }

    auto *trueValue = getSrcValue(trueBlock);
    auto *falseValue = getSrcValue(falseBlock);
    auto *ptrValue = getDstPtr(trueBlock);
    auto storeOpAttributes =
        cast<spirv::StoreOp>(trueBlock->front()).getOperation()->getAttrs();

    auto selectOp = rewriter.create<spirv::SelectOp>(
        selectionOp.getLoc(), trueValue->getType(), brConditionalOp.condition(),
        trueValue, falseValue);
    rewriter.create<spirv::StoreOp>(selectOp.getLoc(), ptrValue,
                                    selectOp.getResult(), storeOpAttributes);

    // `spv.selection` is not needed anymore.
    rewriter.eraseOp(op);
    return matchSuccess();
  }

private:
  // Checks that given blocks follow the following rules:
  // 1. Each conditional block consists of two operations, the first operation
  //    is a `spv.Store` and the last operation is a `spv.Branch`.
  // 2. Each `spv.Store` uses the same pointer and the same memory attributes.
  // 3. A control flow goes into the given merge block from the given
  //    conditional blocks.
  PatternMatchResult canCanonicalizeSelection(Block *trueBlock,
                                              Block *falseBlock,
                                              Block *mergeBlock) const;

  bool onlyContainsBranchConditionalOp(Block *block) const {
    return std::next(block->begin()) == block->end() &&
           isa<spirv::BranchConditionalOp>(block->front());
  }

  bool isSameAttrList(spirv::StoreOp lhs, spirv::StoreOp rhs) const {
    return lhs.getOperation()->getAttrList().getDictionary() ==
           rhs.getOperation()->getAttrList().getDictionary();
  }

  // Checks that given type is valid for `spv.SelectOp`.
  // According to SPIR-V spec:
  // "Before version 1.4, Result Type must be a pointer, scalar, or vector.
  // Starting with version 1.4, Result Type can additionally be a composite type
  // other than a vector."
  bool isValidType(Type type) const {
    return spirv::SPIRVDialect::isValidScalarType(type) ||
           type.isa<VectorType>();
  }

  // Returns a soruce value for the given block.
  Value *getSrcValue(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.value();
  }

  // Returns a destination value for the given block.
  Value *getDstPtr(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.ptr();
  }
};

PatternMatchResult ConvertSelectionOpToSelect::canCanonicalizeSelection(
    Block *trueBlock, Block *falseBlock, Block *mergeBlock) const {
  // Each block must consists of 2 operations.
  if ((std::distance(trueBlock->begin(), trueBlock->end()) != 2) ||
      (std::distance(falseBlock->begin(), falseBlock->end()) != 2)) {
    return matchFailure();
  }

  auto trueBrStoreOp = dyn_cast<spirv::StoreOp>(trueBlock->front());
  auto trueBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(trueBlock->begin()));
  auto falseBrStoreOp = dyn_cast<spirv::StoreOp>(falseBlock->front());
  auto falseBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(falseBlock->begin()));

  if (!trueBrStoreOp || !trueBrBranchOp || !falseBrStoreOp ||
      !falseBrBranchOp) {
    return matchFailure();
  }

  // Check that each `spv.Store` uses the same pointer, memory access
  // attributes and a valid type of the value.
  if ((trueBrStoreOp.ptr() != falseBrStoreOp.ptr()) ||
      !isSameAttrList(trueBrStoreOp, falseBrStoreOp) ||
      !isValidType(trueBrStoreOp.value()->getType())) {
    return matchFailure();
  }

  if ((trueBrBranchOp.getOperation()->getSuccessor(0) != mergeBlock) ||
      (falseBrBranchOp.getOperation()->getSuccessor(0) != mergeBlock)) {
    return matchFailure();
  }

  return matchSuccess();
}
} // end anonymous namespace

void spirv::SelectionOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ConvertSelectionOpToSelect>(context);
}

//===----------------------------------------------------------------------===//
// spv.specConstant
//===----------------------------------------------------------------------===//

static ParseResult parseSpecConstantOp(OpAsmParser &parser,
                                       OperationState &state) {
  StringAttr nameAttr;
  Attribute valueAttr;

  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse optional spec_id.
  if (succeeded(parser.parseOptionalKeyword(kSpecIdAttrName))) {
    IntegerAttr specIdAttr;
    if (parser.parseLParen() ||
        parser.parseAttribute(specIdAttr, kSpecIdAttrName, state.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (parser.parseEqual() ||
      parser.parseAttribute(valueAttr, kDefaultValueAttrName, state.attributes))
    return failure();

  return success();
}

static void print(spirv::SpecConstantOp constOp, OpAsmPrinter &printer) {
  printer << spirv::SpecConstantOp::getOperationName() << ' ';
  printer.printSymbolName(constOp.sym_name());
  if (auto specID = constOp.getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    printer << ' ' << kSpecIdAttrName << '(' << specID.getInt() << ')';
  printer << " = ";
  printer.printAttribute(constOp.default_value());
}

static LogicalResult verify(spirv::SpecConstantOp constOp) {
  if (auto specID = constOp.getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    if (specID.getValue().isNegative())
      return constOp.emitOpError("SpecId cannot be negative");

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

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  auto loc = parser.getCurrentLocation();
  Type elementType;
  if (parseEnumAttribute(storageClass, parser) ||
      parser.parseOperandList(operandInfo, 2) ||
      parseMemoryAccessAttributes(parser, state) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser.resolveOperands(operandInfo, {ptrType, elementType}, loc,
                             state.operands)) {
    return failure();
  }
  return success();
}

static void print(spirv::StoreOp storeOp, OpAsmPrinter &printer) {
  auto *op = storeOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      storeOp.ptr()->getType().cast<spirv::PointerType>().getStorageClass());
  printer << spirv::StoreOp::getOperationName() << " \"" << sc << "\" ";
  // Print the pointer operand
  printer.printOperand(storeOp.ptr());
  printer << ", ";
  // Print the value operand
  printer.printOperand(storeOp.value());

  printMemoryAccessAttribute(storeOp, printer, elidedAttrs);

  printer << " : " << storeOp.value()->getType();

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
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
// spv.SubgroupBallotKHROp
//===----------------------------------------------------------------------===//

static ParseResult parseSubgroupBallotKHROp(OpAsmParser &parser,
                                            OperationState &state) {
  OpAsmParser::OperandType operandInfo;
  Type resultType;
  IntegerType i1Type = parser.getBuilder().getI1Type();
  if (parser.parseOperand(operandInfo) || parser.parseColonType(resultType) ||
      parser.resolveOperand(operandInfo, i1Type, state.operands))
    return failure();

  return parser.addTypeToList(resultType, state.types);
}

static void print(spirv::SubgroupBallotKHROp ballotOp, OpAsmPrinter &printer) {
  printer << spirv::SubgroupBallotKHROp::getOperationName() << ' ';
  printer.printOperand(ballotOp.predicate());
  printer << " : " << ballotOp.getType();
}

//===----------------------------------------------------------------------===//
// spv.Undef
//===----------------------------------------------------------------------===//

static ParseResult parseUndefOp(OpAsmParser &parser, OperationState &state) {
  Type type;
  if (parser.parseColonType(type)) {
    return failure();
  }
  state.addTypes(type);
  return success();
}

static void print(spirv::UndefOp undefOp, OpAsmPrinter &printer) {
  printer << spirv::UndefOp::getOperationName() << " : " << undefOp.getType();
}

//===----------------------------------------------------------------------===//
// spv.Unreachable
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::UnreachableOp unreachableOp) {
  auto *op = unreachableOp.getOperation();
  auto *block = op->getBlock();
  // Fast track: if this is in entry block, its invalid. Otherwise, if no
  // predecessors, it's valid.
  if (block->isEntryBlock())
    return unreachableOp.emitOpError("cannot be used in reachable block");
  if (block->hasNoPredecessors())
    return success();

  // TODO(antiagainst): further verification needs to analyze reachablility from
  // the entry block.

  return success();
}

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

static ParseResult parseVariableOp(OpAsmParser &parser, OperationState &state) {
  // Parse optional initializer
  Optional<OpAsmParser::OperandType> initInfo;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    initInfo = OpAsmParser::OperandType();
    if (parser.parseLParen() || parser.parseOperand(*initInfo) ||
        parser.parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, state)) {
    return failure();
  }

  // Parse result pointer type
  Type type;
  if (parser.parseColon())
    return failure();
  auto loc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser.emitError(loc, "expected spv.ptr type");
  state.addTypes(ptrType);

  // Resolve the initializer operand
  if (initInfo) {
    if (parser.resolveOperand(*initInfo, ptrType.getPointeeType(),
                              state.operands))
      return failure();
  }

  auto attr = parser.getBuilder().getI32IntegerAttr(
      llvm::bit_cast<int32_t>(ptrType.getStorageClass()));
  state.addAttribute(spirv::attributeName<spirv::StorageClass>(), attr);

  return success();
}

static void print(spirv::VariableOp varOp, OpAsmPrinter &printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  printer << spirv::VariableOp::getOperationName();

  // Print optional initializer
  if (op->getNumOperands() > 0) {
    printer << " init(";
    printer.printOperands(varOp.initializer());
    printer << ")";
  }

  printVariableDecorations(op, printer, elidedAttrs);

  printer << " : " << varOp.getType();
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
