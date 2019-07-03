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

#include "mlir/SPIRV/SPIRVOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/SPIRV/SPIRVTypes.h"

using namespace mlir;

// TODO(antiagainst): generate these strings using ODS.
static constexpr const char kAddressingModelAttrName[] = "addressing_model";
static constexpr const char kBindingAttrName[] = "binding";
static constexpr const char kDescriptorSetAttrName[] = "descriptor_set";
static constexpr const char kMemoryModelAttrName[] = "memory_model";
static constexpr const char kStorageClassAttrName[] = "storage_class";
static constexpr const char kValueAttrName[] = "value";

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

static ParseResult parseStorageClassAttribute(spirv::StorageClass &storageClass,
                                              OpAsmParser *parser,
                                              OperationState *state) {
  Attribute storageClassAttr;
  SmallVector<NamedAttribute, 1> storageAttr;
  auto loc = parser->getCurrentLocation();
  if (parser->parseAttribute(storageClassAttr,
                             parser->getBuilder().getNoneType(),
                             "storage_class", storageAttr)) {
    return failure();
  }
  if (!storageClassAttr.isa<StringAttr>()) {
    return parser->emitError(loc, "expected a string storage class specifier");
  }
  auto storageClassOptional = spirv::symbolizeStorageClass(
      storageClassAttr.cast<StringAttr>().getValue());
  if (!storageClassOptional) {
    return parser->emitError(loc, "invalid storage class specifier: ")
           << storageClassAttr;
  }
  storageClass = storageClassOptional.getValue();
  return success();
}

template <typename LoadStoreOpTy>
static ParseResult parseMemoryAccessAttributes(OpAsmParser *parser,
                                               OperationState *state) {
  // Parse an optional list of attributes staring with '['
  if (parser->parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  StringRef memAccessAttrName = LoadStoreOpTy::getMemoryAccessAttrName();
  Attribute memAccessAttr;
  SmallVector<NamedAttribute, 1> attrs;
  auto loc = parser->getCurrentLocation();

  if (parser->parseAttribute(memAccessAttr, memAccessAttrName, attrs))
    return failure();
  // Check that this is a memory attribute
  if (!memAccessAttr.isa<StringAttr>()) {
    return parser->emitError(loc, "expected a string memory access specifier");
  }
  auto memAccessOptional =
      spirv::symbolizeMemoryAccess(memAccessAttr.cast<StringAttr>().getValue());
  if (!memAccessOptional) {
    return parser->emitError(loc, "invalid memory access specifier: ")
           << memAccessAttr;
  }
  state->addAttribute(memAccessAttrName,
                      parser->getBuilder().getI32IntegerAttr(
                          bitwiseCast<int32_t>(*memAccessOptional)));

  if (auto memAccess =
          memAccessOptional.getValue() == spirv::MemoryAccess::Aligned) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    Type i32Type = parser->getBuilder().getIntegerType(32);
    if (parser->parseComma() ||
        parser->parseAttribute(alignmentAttr, i32Type,
                               LoadStoreOpTy::getAlignmentAttrName(),
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
    elidedAttrs.push_back(LoadStoreOpTy::getMemoryAccessAttrName());
    *printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    // Print integer alignment attribute.
    if (auto alignment = loadStoreOp.alignment()) {
      elidedAttrs.push_back(LoadStoreOpTy::getAlignmentAttrName());
      *printer << ", " << alignment;
    }
    *printer << "]";
  }
}

template <typename LoadStoreOpTy>
static LogicalResult verifyMemoryAccessAttribute(LoadStoreOpTy loadStoreOp) {
  // ODS checks for attributes values. Just need to verify that if the
  // memory-access attribute is Aligned, then the alignment attribute must be
  // present.
  auto *op = loadStoreOp.getOperation();
  auto memAccessAttr = op->getAttr(LoadStoreOpTy::getMemoryAccessAttrName());
  if (!memAccessAttr) {
    // Alignment attribute shouldn't be present if memory access attribute is
    // not present.
    if (op->getAttr(LoadStoreOpTy::getAlignmentAttrName())) {
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
    if (!op->getAttr(LoadStoreOpTy::getAlignmentAttrName())) {
      return loadStoreOp.emitOpError("missing alignment value");
    }
  } else {
    if (op->getAttr(LoadStoreOpTy::getAlignmentAttrName())) {
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

// Verifies that the given op can only be placed in a `spv.module`.
static LogicalResult verifyModuleOnly(Operation *op) {
  if (!llvm::isa_and_nonnull<spirv::ModuleOp>(op->getParentOp()))
    return op->emitOpError("can only be used in a 'spv.module' block");

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
  *printer << spirv::ConstantOp::getOperationName() << " " << constOp.value()
           << " : " << constOp.getType();
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
// spv.LoadOp
//===----------------------------------------------------------------------===//

static ParseResult parseLoadOp(OpAsmParser *parser, OperationState *state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::OperandType ptrInfo;
  Type elementType;
  if (parseStorageClassAttribute(storageClass, parser, state) ||
      parser->parseOperand(ptrInfo) ||
      parseMemoryAccessAttributes<spirv::LoadOp>(parser, state) ||
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
  Builder builder = parser->getBuilder();
  Region *body = state->addRegion();

  Attribute addressingModel, memoryModel;
  SmallVector<NamedAttribute, 2> attrs;

  // Parse addressing model
  auto loc = parser->getCurrentLocation();
  if (parser->parseAttribute(addressingModel, kAddressingModelAttrName, attrs))
    return failure();
  if (!addressingModel.isa<StringAttr>()) {
    return parser->emitError(loc,
                             "requires string for addressing model but found '")
           << addressingModel << "'";
  }
  auto addrModel = spirv::symbolizeAddressingModel(
      addressingModel.cast<StringAttr>().getValue());
  if (!addrModel) {
    return parser->emitError(loc, "unknown addressing model: ")
           << addressingModel;
  }
  state->addAttribute(
      kAddressingModelAttrName,
      builder.getI32IntegerAttr(bitwiseCast<int32_t>(*addrModel)));

  // Parse memory model
  loc = parser->getCurrentLocation();
  if (parser->parseAttribute(memoryModel, kMemoryModelAttrName, attrs))
    return failure();
  if (!memoryModel.isa<StringAttr>()) {
    return parser->emitError(loc,
                             "requires string for memory model but found '")
           << memoryModel << "'";
  }
  auto memModel =
      spirv::symbolizeMemoryModel(memoryModel.cast<StringAttr>().getValue());
  if (!memModel) {
    return parser->emitError(loc, "unknown memory model: ") << memoryModel;
  }
  state->addAttribute(
      kMemoryModelAttrName,
      builder.getI32IntegerAttr(bitwiseCast<int32_t>(*memModel)));

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
  if (op->getAttr(kAddressingModelAttrName) &&
      op->getAttr(kMemoryModelAttrName)) {
    *printer << spirv::ModuleOp::getOperationName() << " \""
             << spirv::stringifyAddressingModel(moduleOp.addressing_model())
             << "\" \"" << spirv::stringifyMemoryModel(moduleOp.memory_model())
             << '"';
    elidedAttrs.assign({kAddressingModelAttrName, kMemoryModelAttrName});
  }

  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/false);

  bool printAttrDict = elidedAttrs.size() != 2 ||
                       llvm::any_of(op->getAttrs(), [](NamedAttribute attr) {
                         return attr.first != kAddressingModelAttrName &&
                                attr.first != kMemoryModelAttrName;
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

  for (auto &op : body) {
    if (op.getDialect() == dialect)
      continue;

    auto funcOp = llvm::dyn_cast<FuncOp>(op);
    if (!funcOp)
      return op.emitError("'spv.module' can only contain func and spv.* ops");

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
  auto funcOp =
      returnOp.getOperation()->getContainingRegion()->getContainingFunction();
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
  if (parseStorageClassAttribute(storageClass, parser, state) ||
      parser->parseOperandList(operandInfo, 2) ||
      parseMemoryAccessAttributes<spirv::StoreOp>(parser, state) ||
      parser->parseColon() || parser->parseType(elementType)) {
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
  state->addAttribute(kStorageClassAttrName, attr);

  return success();
}

static void print(spirv::VariableOp varOp, OpAsmPrinter *printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{kStorageClassAttrName};
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
#include "mlir/SPIRV/SPIRVOps.cpp.inc"

} // namespace spirv
} // namespace mlir
