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

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace spirv {
#include "mlir/Dialect/SPIRV/SPIRVOpUtils.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

// TODO(antiagainst): generate these strings using ODS.
static constexpr const char kAlignmentAttrName[] = "alignment";
static constexpr const char kBindingAttrName[] = "binding";
static constexpr const char kDescriptorSetAttrName[] = "descriptor_set";
static constexpr const char kIndicesAttrName[] = "indices";
static constexpr const char kValueAttrName[] = "value";
static constexpr const char kValuesAttrName[] = "values";
static constexpr const char kFnNameAttrName[] = "fn";

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
  auto constOp = llvm::dyn_cast<spirv::ConstantOp>(op);
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
static ParseResult parseEnumAttribute(EnumClass &value, OpAsmParser *parser,
                                      OperationState *state) {
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

//===----------------------------------------------------------------------===//
// spv.AccessChainOp
//===----------------------------------------------------------------------===//

static Type getElementPtrType(Type type, ArrayRef<Value *> indices,
                              Location baseLoc) {
  if (!indices.size()) {
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

  Location baseLoc = state->operands.front()->getLoc();
  auto resultType = getElementPtrType(
      type, llvm::makeArrayRef(state->operands).drop_front(), baseLoc);
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
  *printer << spirv::ConstantOp::getOperationName() << " " << constOp.value();
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
        return constOp.emitOpError(
            "has array element that are not of result array element type");
    }
  } break;
  default:
    return constOp.emitOpError("cannot have value of type ") << valueType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

static ParseResult parseEntryPointOp(OpAsmParser *parser,
                                     OperationState *state) {
  spirv::ExecutionModel execModel;
  SmallVector<OpAsmParser::OperandType, 0> identifiers;
  SmallVector<Type, 0> idTypes;

  Attribute fn;
  auto loc = parser->getCurrentLocation();

  if (parseEnumAttribute(execModel, parser, state) ||
      parser->parseAttribute(fn, kFnNameAttrName, state->attributes) ||
      parser->parseTrailingOperandList(identifiers) ||
      parser->parseOptionalColonTypeList(idTypes) ||
      parser->resolveOperands(identifiers, idTypes, loc, state->operands)) {
    return failure();
  }
  if (!fn.isa<SymbolRefAttr>()) {
    return parser->emitError(loc, "expected symbol reference attribute");
  }
  return success();
}

static void print(spirv::EntryPointOp entryPointOp, OpAsmPrinter *printer) {
  *printer << spirv::EntryPointOp::getOperationName() << " \""
           << stringifyExecutionModel(entryPointOp.execution_model()) << "\" @"
           << entryPointOp.fn();
  if (!entryPointOp.getNumOperands()) {
    return;
  }
  *printer << ", ";
  mlir::interleaveComma(entryPointOp.getOperands(), printer->getStream(),
                        [&](Value *a) { printer->printOperand(a); });
  *printer << " : ";
  mlir::interleaveComma(entryPointOp.getOperands(), printer->getStream(),
                        [&](const Value *a) { *printer << a->getType(); });
}

static LogicalResult verify(spirv::EntryPointOp entryPointOp) {
  // Verify that all the interface ops are created from VariableOp
  for (auto interface : entryPointOp.interface()) {
    if (!llvm::isa_and_nonnull<spirv::VariableOp>(interface->getDefiningOp())) {
      return entryPointOp.emitOpError("interface operands to entry point must "
                                      "be generated from a variable op");
    }
    // Before version 1.4 the variables can only have storage_class of Input or
    // Output.
    // TODO: Add versioning so that this can be avoided for 1.4
    auto storageClass =
        interface->getType().cast<spirv::PointerType>().getStorageClass();
    switch (storageClass) {
    case spirv::StorageClass::Input:
    case spirv::StorageClass::Output:
      break;
    default:
      return entryPointOp.emitOpError("invalid storage class '")
             << stringifyStorageClass(storageClass)
             << "' for interface variables";
    }
  }
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
// spv.LoadOp
//===----------------------------------------------------------------------===//

static ParseResult parseLoadOp(OpAsmParser *parser, OperationState *state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::OperandType ptrInfo;
  Type elementType;
  if (parseEnumAttribute(storageClass, parser, state) ||
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
// spv.module
//===----------------------------------------------------------------------===//

static void ensureModuleEnd(Region *region, Builder builder, Location loc) {
  impl::ensureRegionTerminator<spirv::ModuleEndOp>(*region, builder, loc);
}

void spirv::ModuleOp::build(Builder *builder, OperationState *state) {
  ensureModuleEnd(state->addRegion(), *builder, state->location);
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

  ensureModuleEnd(body, parser->getBuilder(), state->location);

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
  llvm::StringMap<FuncOp> funcNames;
  llvm::DenseMap<std::pair<FuncOp, spirv::ExecutionModel>, spirv::EntryPointOp>
      entryPoints;

  for (auto &op : body) {
    if (op.getDialect() == dialect) {
      // For EntryPoint op, check that the function name is one of the specified
      // func ops already specified, and that the function and execution model
      // is not duplicated in EntryPointOps
      if (auto entryPointOp = llvm::dyn_cast<spirv::EntryPointOp>(op)) {
        auto it = funcNames.find(entryPointOp.fn());
        if (it == funcNames.end()) {
          return entryPointOp.emitError("function '")
                 << entryPointOp.fn() << "' not found in 'spv.module'";
        }
        auto funcOp = it->second;
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

    auto funcOp = llvm::dyn_cast<FuncOp>(op);
    if (!funcOp)
      return op.emitError("'spv.module' can only contain func and spv.* ops");

    funcNames[funcOp.getName()] = funcOp;

    if (funcOp.isExternal())
      return op.emitError("'spv.module' cannot contain external functions");

    for (auto &block : funcOp)
      for (auto &op : block) {
        if (op.getDialect() == dialect)
          continue;

        if (llvm::isa<FuncOp>(op))
          return op.emitError("'spv.module' cannot contain nested functions");

        return op.emitError(
            "functions in 'spv.module' can only contain spv.* ops");
      }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

static LogicalResult verifyReturn(spirv::ReturnOp returnOp) {
  auto funcOp = llvm::dyn_cast<FuncOp>(returnOp.getOperation()->getParentOp());
  if (!funcOp)
    return returnOp.emitOpError("must appear in a 'func' op");

  auto numOutputs = funcOp.getType().getNumResults();
  if (numOutputs != 0)
    return returnOp.emitOpError("cannot be used in functions returning value")
           << (numOutputs > 1 ? "s" : "");

  return success();
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
  if (parseEnumAttribute(storageClass, parser, state) ||
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

  // Parse optional descriptor binding
  Attribute set, binding;
  if (succeeded(parser->parseOptionalKeyword("bind"))) {
    Type i32Type = parser->getBuilder().getIntegerType(32);
    if (parser->parseLParen() ||
        parser->parseAttribute(set, i32Type, kDescriptorSetAttrName,
                               state->attributes) ||
        parser->parseComma() ||
        parser->parseAttribute(binding, i32Type, kBindingAttrName,
                               state->attributes) ||
        parser->parseRParen())
      return failure();
  }

  // Parse other attributes
  if (parser->parseOptionalAttributeDict(state->attributes))
    return failure();

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

  // Print optional descriptor binding
  auto set = varOp.getAttrOfType<IntegerAttr>(kDescriptorSetAttrName);
  auto binding = varOp.getAttrOfType<IntegerAttr>(kBindingAttrName);
  if (set && binding) {
    elidedAttrs.push_back(kDescriptorSetAttrName);
    elidedAttrs.push_back(kBindingAttrName);
    *printer << " bind(" << set.getInt() << ", " << binding.getInt() << ")";
  }

  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  *printer << " : " << varOp.getType();
}

static LogicalResult verify(spirv::VariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (varOp.storage_class() == spirv::StorageClass::Generic)
    return varOp.emitOpError("storage class cannot be 'Generic'");

  auto pointerType = varOp.pointer()->getType().cast<spirv::PointerType>();
  if (varOp.storage_class() != pointerType.getStorageClass())
    return varOp.emitOpError(
        "storage class must match result pointer's storage class");

  if (varOp.getNumOperands() != 0) {
    // SPIR-V spec: "Initializer must be an <id> from a constant instruction or
    // a global (module scope) OpVariable instruction".
    bool valid = false;
    if (auto *initOp = varOp.getOperand(0)->getDefiningOp()) {
      if (llvm::isa<spirv::ConstantOp>(initOp)) {
        valid = true;
      } else if (llvm::isa<spirv::VariableOp>(initOp)) {
        valid = llvm::isa_and_nonnull<spirv::ModuleOp>(initOp->getParentOp());
      }
    }
    if (!valid)
      return varOp.emitOpError("initializer must be the result of a "
                               "spv.Constant or module-level spv.Variable op");
  }

  return success();
}

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVOps.cpp.inc"

} // namespace spirv
} // namespace mlir
