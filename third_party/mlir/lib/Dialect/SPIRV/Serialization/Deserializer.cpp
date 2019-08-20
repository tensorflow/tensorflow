//===- Deserializer.cpp - MLIR SPIR-V Deserialization ---------------------===//
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
// This file defines the SPIR-V binary to MLIR SPIR-V module deseralization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Serialization.h"

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"

using namespace mlir;

// Decodes a string literal in `words` starting at `wordIndex`. Update the
// latter to point to the position in words after the string literal.
static inline StringRef decodeStringLiteral(ArrayRef<uint32_t> words,
                                            unsigned &wordIndex) {
  StringRef str(reinterpret_cast<const char *>(words.data() + wordIndex));
  wordIndex += str.size() / 4 + 1;
  return str;
}

namespace {
/// A SPIR-V module serializer.
///
/// A SPIR-V binary module is a single linear stream of instructions; each
/// instruction is composed of 32-bit words. The first word of an instruction
/// records the total number of words of that instruction using the 16
/// higher-order bits. So this deserializer uses that to get instruction
/// boundary and parse instructions and build a SPIR-V ModuleOp gradually.
///
// TODO(antiagainst): clean up created ops on errors
class Deserializer {
public:
  /// Creates a deserializer for the given SPIR-V `binary` module.
  /// The SPIR-V ModuleOp will be created into `context.
  explicit Deserializer(ArrayRef<uint32_t> binary, MLIRContext *context);

  /// Deserializes the remembered SPIR-V binary module.
  LogicalResult deserialize();

  /// Collects the final SPIR-V ModuleOp.
  Optional<spirv::ModuleOp> collect();

private:
  //===--------------------------------------------------------------------===//
  // Module structure
  //===--------------------------------------------------------------------===//

  /// Initializes the `module` ModuleOp in this deserializer instance.
  spirv::ModuleOp createModuleOp();

  /// Processes SPIR-V module header in `binary`.
  LogicalResult processHeader();

  /// Processes the SPIR-V OpMemoryModel with `operands` and updates `module`.
  LogicalResult processMemoryModel(ArrayRef<uint32_t> operands);

  /// Process SPIR-V OpName with `operands`
  LogicalResult processName(ArrayRef<uint32_t> operands);

  /// Method to process an OpDecorate instruction.
  LogicalResult processDecoration(ArrayRef<uint32_t> words);

  /// Processes the SPIR-V function at the current `offset` into `binary`.
  /// The operands to the OpFunction instruction is passed in as ``operands`.
  /// This method processes each instruction inside the function and dispatches
  /// them to their handler method accordingly.
  LogicalResult processFunction(ArrayRef<uint32_t> operands);

  /// Process the OpVariable instructions at current `offset` into `binary`. It
  /// is expected that this method is used for variables that are to be defined
  /// at module scope and will be deserialized into a spv.globalVariable
  /// instruction.
  LogicalResult processGlobalVariable(ArrayRef<uint32_t> operands);

  /// Get the FuncOp associated with a result <id> of OpFunction.
  FuncOp getFunction(uint32_t id) { return funcMap.lookup(id); }

  /// Get the global variable associated with a result <id> of OpVariable
  spirv::GlobalVariableOp getVariable(uint32_t id) {
    return globalVariableMap.lookup(id);
  }

  //===--------------------------------------------------------------------===//
  // Type
  //===--------------------------------------------------------------------===//

  /// Gets type for a given result <id>.
  Type getType(uint32_t id) { return typeMap.lookup(id); }

  /// Returns true if the given `type` is for SPIR-V void type.
  bool isVoidType(Type type) const { return type.isa<NoneType>(); }

  /// Processes a SPIR-V type instruction with given `opcode` and `operands` and
  /// registers the type into `module`.
  LogicalResult processType(spirv::Opcode opcode, ArrayRef<uint32_t> operands);

  LogicalResult processArrayType(ArrayRef<uint32_t> operands);

  LogicalResult processFunctionType(ArrayRef<uint32_t> operands);

  //===--------------------------------------------------------------------===//
  // Constant
  //===--------------------------------------------------------------------===//

  /// Processes a SPIR-V Op{|Spec}Constant instruction with the given
  /// `operands`. `isSpec` indicates whether this is a specialization constant.
  LogicalResult processConstant(ArrayRef<uint32_t> operands, bool isSpec);

  /// Processes a SPIR-V Op{|Spec}Constant{True|False} instruction with the
  /// given `operands`. `isSpec` indicates whether this is a specialization
  /// constant.
  LogicalResult processConstantBool(bool isTrue, ArrayRef<uint32_t> operands,
                                    bool isSpec);

  /// Processes a SPIR-V Op{|Spec}ConstantComposite instruction with the given
  /// `operands`. `isSpec` indicates whether this is a specialization constant.
  LogicalResult processConstantComposite(ArrayRef<uint32_t> operands,
                                         bool isSpec);

  /// Processes a SPIR-V OpConstantNull instruction with the given `operands`.
  LogicalResult processConstantNull(ArrayRef<uint32_t> operands);

  //===--------------------------------------------------------------------===//
  // Instruction
  //===--------------------------------------------------------------------===//

  /// Get the Value associated with a result <id>.
  Value *getValue(uint32_t id) {
    if (auto varOp = getVariable(id)) {
      auto addressOfOp = opBuilder.create<spirv::AddressOfOp>(
          unknownLoc, varOp.type(),
          opBuilder.getSymbolRefAttr(varOp.getOperation()));
      return addressOfOp.pointer();
    }
    return valueMap.lookup(id);
  }

  /// Slices the first instruction out of `binary` and returns its opcode and
  /// operands via `opcode` and `operands` respectively. Returns failure if
  /// there is no more remaining instructions (`expectedOpcode` will be used to
  /// compose the error message) or the next instruction is malformed.
  LogicalResult
  sliceInstruction(spirv::Opcode &opcode, ArrayRef<uint32_t> &operands,
                   Optional<spirv::Opcode> expectedOpcode = llvm::None);

  /// Processes a SPIR-V instruction with the given `opcode` and `operands`.
  /// This method is the main entrance for handling SPIR-V instruction; it
  /// checks the instruction opcode and dispatches to the corresponding handler.
  /// Processing of Some instructions (like OpEntryPoint and OpExecutionMode)
  /// might need to be defered, since they contain forward references to <id>s
  /// in the deserialized binary, but module in SPIR-V dialect expects these to
  /// be ssa-uses.
  LogicalResult processInstruction(spirv::Opcode opcode,
                                   ArrayRef<uint32_t> operands,
                                   bool deferInstructions = true);

  /// Method to dispatch to the specialized deserialization function for an
  /// operation in SPIR-V dialect that is a mirror of an instruction in the
  /// SPIR-V spec. This is auto-generated from ODS. Dispatch is handled for
  /// all operations in SPIR-V dialect that have hasOpcode == 1.
  LogicalResult dispatchToAutogenDeserialization(spirv::Opcode opcode,
                                                 ArrayRef<uint32_t> words);

  /// Method to deserialize an operation in the SPIR-V dialect that is a mirror
  /// of an instruction in the SPIR-V spec. This is auto generated if hasOpcode
  /// == 1 and autogenSerialization == 1 in ODS.
  template <typename OpTy> LogicalResult processOp(ArrayRef<uint32_t> words) {
    return emitError(unknownLoc, "unsupported deserialization for ")
           << OpTy::getOperationName() << " op";
  }

private:
  /// The SPIR-V binary module.
  ArrayRef<uint32_t> binary;

  /// The current word offset into the binary module.
  unsigned curOffset = 0;

  /// MLIRContext to create SPIR-V ModuleOp into.
  MLIRContext *context;

  // TODO(antiagainst): create Location subclass for binary blob
  Location unknownLoc;

  /// The SPIR-V ModuleOp.
  Optional<spirv::ModuleOp> module;

  OpBuilder opBuilder;

  // Result <id> to type mapping.
  DenseMap<uint32_t, Type> typeMap;

  // Result <id> to function mapping.
  DenseMap<uint32_t, FuncOp> funcMap;

  // Result <id> to variable mapping;
  DenseMap<uint32_t, spirv::GlobalVariableOp> globalVariableMap;

  // Result <id> to value mapping.
  DenseMap<uint32_t, Value *> valueMap;

  // Result <id> to name mapping.
  DenseMap<uint32_t, StringRef> nameMap;

  // Result <id> to decorations mapping.
  DenseMap<uint32_t, NamedAttributeList> decorations;

  // Result <id> to type decorations.
  DenseMap<uint32_t, uint32_t> typeDecorations;

  // List of instructions that are processed in a defered fashion (after an
  // initial processing of the entire binary). Some operations like
  // OpEntryPoint, and OpExecutionMode use forward references to function
  // <id>s. In SPIR-V dialect the corresponding operations (spv.EntryPoint and
  // spv.ExecutionMode) need these references resolved. So these instructions
  // are deserialized and stored for processing once the entire binary is
  // processed.
  SmallVector<std::pair<spirv::Opcode, ArrayRef<uint32_t>>, 4>
      deferedInstructions;
};
} // namespace

Deserializer::Deserializer(ArrayRef<uint32_t> binary, MLIRContext *context)
    : binary(binary), context(context), unknownLoc(UnknownLoc::get(context)),
      module(createModuleOp()),
      opBuilder(module->getOperation()->getRegion(0)) {}

LogicalResult Deserializer::deserialize() {
  if (failed(processHeader()))
    return failure();

  spirv::Opcode opcode = spirv::Opcode::OpNop;
  ArrayRef<uint32_t> operands;
  auto binarySize = binary.size();
  while (curOffset < binarySize) {
    // Slice the next instruction out and populate `opcode` and `operands`.
    // Interally this also updates `curOffset`.
    if (failed(sliceInstruction(opcode, operands)))
      return failure();

    if (failed(processInstruction(opcode, operands)))
      return failure();
  }

  assert(curOffset == binarySize &&
         "deserializer should never index beyond the binary end");

  for (auto &defered : deferedInstructions) {
    if (failed(processInstruction(defered.first, defered.second, false))) {
      return failure();
    }
  }

  return success();
}

Optional<spirv::ModuleOp> Deserializer::collect() { return module; }

//===----------------------------------------------------------------------===//
// Module structure
//===----------------------------------------------------------------------===//

spirv::ModuleOp Deserializer::createModuleOp() {
  Builder builder(context);
  OperationState state(unknownLoc, spirv::ModuleOp::getOperationName());
  // TODO(antiagainst): use target environment to select the version
  state.addAttribute("major_version", builder.getI32IntegerAttr(1));
  state.addAttribute("minor_version", builder.getI32IntegerAttr(0));
  spirv::ModuleOp::build(&builder, &state);
  return cast<spirv::ModuleOp>(Operation::create(state));
}

LogicalResult Deserializer::processHeader() {
  if (binary.size() < spirv::kHeaderWordCount)
    return emitError(unknownLoc,
                     "SPIR-V binary module must have a 5-word header");

  if (binary[0] != spirv::kMagicNumber)
    return emitError(unknownLoc, "incorrect magic number");

  // TODO(antiagainst): generator number, bound, schema
  curOffset = spirv::kHeaderWordCount;
  return success();
}

LogicalResult Deserializer::processMemoryModel(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2)
    return emitError(unknownLoc, "OpMemoryModel must have two operands");

  module->setAttr(
      "addressing_model",
      opBuilder.getI32IntegerAttr(llvm::bit_cast<int32_t>(operands.front())));
  module->setAttr(
      "memory_model",
      opBuilder.getI32IntegerAttr(llvm::bit_cast<int32_t>(operands.back())));

  return success();
}

LogicalResult Deserializer::processDecoration(ArrayRef<uint32_t> words) {
  // TODO : This function should also be auto-generated. For now, since only a
  // few decorations are processed/handled in a meaningful manner, going with a
  // manual implementation.
  if (words.size() < 2) {
    return emitError(
        unknownLoc, "OpDecorate must have at least result <id> and Decoration");
  }
  auto decorationName =
      stringifyDecoration(static_cast<spirv::Decoration>(words[1]));
  if (decorationName.empty()) {
    return emitError(unknownLoc, "invalid Decoration code : ") << words[1];
  }
  auto attrName = convertToSnakeCase(decorationName);
  switch (static_cast<spirv::Decoration>(words[1])) {
  case spirv::Decoration::DescriptorSet:
  case spirv::Decoration::Binding:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecorate with ")
             << decorationName << " needs a single integer literal";
    }
    decorations[words[0]].set(
        opBuilder.getIdentifier(attrName),
        opBuilder.getI32IntegerAttr(static_cast<int32_t>(words[2])));
    break;
  case spirv::Decoration::BuiltIn:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecorate with ")
             << decorationName << " needs a single integer literal";
    }
    decorations[words[0]].set(opBuilder.getIdentifier(attrName),
                              opBuilder.getStringAttr(stringifyBuiltIn(
                                  static_cast<spirv::BuiltIn>(words[2]))));
    break;
  case spirv::Decoration::ArrayStride:
    if (words.size() != 3) {
      return emitError(unknownLoc, "OpDecorate with ")
             << decorationName << " needs a single integer literal";
    }
    typeDecorations[words[0]] = static_cast<uint32_t>(words[2]);
    break;
  default:
    return emitError(unknownLoc, "unhandled Decoration : '") << decorationName;
  }
  return success();
}

LogicalResult Deserializer::processFunction(ArrayRef<uint32_t> operands) {
  // Get the result type
  if (operands.size() != 4) {
    return emitError(unknownLoc, "OpFunction must have 4 parameters");
  }
  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }
  if (funcMap.count(operands[1])) {
    return emitError(unknownLoc, "duplicate function definition/declaration");
  }
  auto functionControl = spirv::symbolizeFunctionControl(operands[2]);
  if (!functionControl) {
    return emitError(unknownLoc, "unknown Function Control: ") << operands[2];
  }
  if (functionControl.getValue() != spirv::FunctionControl::None) {
    /// TODO : Handle different function controls
    return emitError(unknownLoc, "unhandled Function Control: '")
           << spirv::stringifyFunctionControl(functionControl.getValue())
           << "'";
  }
  Type fnType = getType(operands[3]);
  if (!fnType || !fnType.isa<FunctionType>()) {
    return emitError(unknownLoc, "unknown function type from <id> ")
           << operands[3];
  }
  auto functionType = fnType.cast<FunctionType>();
  if ((isVoidType(resultType) && functionType.getNumResults() != 0) ||
      (functionType.getNumResults() == 1 &&
       functionType.getResult(0) != resultType)) {
    return emitError(unknownLoc, "mismatch in function type ")
           << functionType << " and return type " << resultType << " specified";
  }

  std::string fnName = nameMap.lookup(operands[1]).str();
  if (fnName.empty()) {
    fnName = "spirv_fn_" + std::to_string(operands[2]);
  }
  auto funcOp = opBuilder.create<FuncOp>(unknownLoc, fnName, functionType,
                                         ArrayRef<NamedAttribute>());
  funcMap[operands[1]] = funcOp;
  funcOp.addEntryBlock();

  // Parse the op argument instructions
  if (functionType.getNumInputs()) {
    for (size_t i = 0, e = functionType.getNumInputs(); i != e; ++i) {
      auto argType = functionType.getInput(i);
      spirv::Opcode opcode = spirv::Opcode::OpNop;
      ArrayRef<uint32_t> operands;
      if (failed(sliceInstruction(opcode, operands,
                                  spirv::Opcode::OpFunctionParameter))) {
        return failure();
      }
      if (opcode != spirv::Opcode::OpFunctionParameter) {
        return emitError(
                   unknownLoc,
                   "missing OpFunctionParameter instruction for argument ")
               << i;
      }
      if (operands.size() != 2) {
        return emitError(
            unknownLoc,
            "expected result type and result <id> for OpFunctionParameter");
      }
      auto argDefinedType = getType(operands[0]);
      if (!argDefinedType || argDefinedType != argType) {
        return emitError(unknownLoc,
                         "mismatch in argument type between function type "
                         "definition ")
               << functionType << " and argument type definition "
               << argDefinedType << " at argument " << i;
      }
      if (getValue(operands[1])) {
        return emitError(unknownLoc, "duplicate definition of result <id> '")
               << operands[1];
      }
      auto argValue = funcOp.getArgument(i);
      valueMap[operands[1]] = argValue;
    }
  }

  // Create a new builder for building the body
  OpBuilder funcBody(funcOp.getBody());
  std::swap(funcBody, opBuilder);

  spirv::Opcode opcode = spirv::Opcode::OpNop;
  ArrayRef<uint32_t> instOperands;
  while (succeeded(sliceInstruction(opcode, instOperands,
                                    spirv::Opcode::OpFunctionEnd)) &&
         opcode != spirv::Opcode::OpFunctionEnd) {
    if (failed(processInstruction(opcode, instOperands))) {
      return failure();
    }
  }
  if (opcode != spirv::Opcode::OpFunctionEnd) {
    return failure();
  }
  if (!instOperands.empty()) {
    return emitError(unknownLoc, "unexpected operands for OpFunctionEnd");
  }
  std::swap(funcBody, opBuilder);
  return success();
}

LogicalResult Deserializer::processGlobalVariable(ArrayRef<uint32_t> operands) {
  unsigned wordIndex = 0;
  if (operands.size() < 3) {
    return emitError(
        unknownLoc,
        "OpVariable needs at least 3 operands, type, <id> and storage class");
  }

  // Result Type.
  auto type = getType(operands[wordIndex]);
  if (!type) {
    return emitError(unknownLoc, "unknown result type <id> : ")
           << operands[wordIndex];
  }
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return emitError(unknownLoc,
                     "expected a result type <id> to be a spv.ptr, found : ")
           << type;
  }
  wordIndex++;

  // Result <id>.
  auto variableID = operands[wordIndex];
  auto variableName = nameMap.lookup(variableID).str();
  if (variableName.empty()) {
    variableName = "spirv_var_" + std::to_string(variableID);
  }
  wordIndex++;

  // Storage class.
  auto storageClass = static_cast<spirv::StorageClass>(operands[wordIndex]);
  if (ptrType.getStorageClass() != storageClass) {
    return emitError(unknownLoc, "mismatch in storage class of pointer type ")
           << type << " and that specified in OpVariable instruction  : "
           << stringifyStorageClass(storageClass);
  }
  wordIndex++;

  // Initializer.
  SymbolRefAttr initializer = nullptr;
  if (wordIndex < operands.size()) {
    auto initializerOp = getVariable(operands[wordIndex]);
    if (!initializerOp) {
      return emitError(unknownLoc, "unknown <id> ")
             << operands[wordIndex] << "used as initializer";
    }
    wordIndex++;
    initializer = opBuilder.getSymbolRefAttr(initializerOp.getOperation());
  }
  if (wordIndex != operands.size()) {
    return emitError(unknownLoc,
                     "found more operands than expected when deserializing "
                     "OpVariable instruction, only ")
           << wordIndex << " of " << operands.size() << " processed";
  }
  auto varOp = opBuilder.create<spirv::GlobalVariableOp>(
      unknownLoc, opBuilder.getTypeAttr(type),
      opBuilder.getStringAttr(variableName), initializer);

  // Decorations.
  if (decorations.count(variableID)) {
    for (auto attr : decorations[variableID].getAttrs()) {
      varOp.setAttr(attr.first, attr.second);
    }
  }
  globalVariableMap[variableID] = varOp;
  return success();
}

LogicalResult Deserializer::processName(ArrayRef<uint32_t> operands) {
  if (operands.size() < 2) {
    return emitError(unknownLoc, "OpName needs at least 2 operands");
  }
  if (!nameMap.lookup(operands[0]).empty()) {
    return emitError(unknownLoc, "duplicate name found for result <id> ")
           << operands[0];
  }
  unsigned wordIndex = 1;
  StringRef name = decodeStringLiteral(operands, wordIndex);
  if (wordIndex != operands.size()) {
    return emitError(unknownLoc,
                     "unexpected trailing words in OpName instruction");
  }
  nameMap[operands[0]] = name;
  return success();
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

LogicalResult Deserializer::processType(spirv::Opcode opcode,
                                        ArrayRef<uint32_t> operands) {
  if (operands.empty()) {
    return emitError(unknownLoc, "type instruction with opcode ")
           << spirv::stringifyOpcode(opcode) << " needs at least one <id>";
  }

  /// TODO: Types might be forward declared in some instructions and need to be
  /// handled appropriately.
  if (typeMap.count(operands[0])) {
    return emitError(unknownLoc, "duplicate definition for result <id> ")
           << operands[0];
  }

  switch (opcode) {
  case spirv::Opcode::OpTypeVoid:
    if (operands.size() != 1) {
      return emitError(unknownLoc, "OpTypeVoid must have no parameters");
    }
    typeMap[operands[0]] = opBuilder.getNoneType();
    break;
  case spirv::Opcode::OpTypeBool:
    if (operands.size() != 1) {
      return emitError(unknownLoc, "OpTypeBool must have no parameters");
    }
    typeMap[operands[0]] = opBuilder.getI1Type();
    break;
  case spirv::Opcode::OpTypeInt:
    if (operands.size() != 3) {
      return emitError(
          unknownLoc, "OpTypeInt must have bitwidth and signedness parameters");
    }
    if (operands[2] == 0) {
      return emitError(unknownLoc, "unhandled unsigned OpTypeInt");
    }
    typeMap[operands[0]] = opBuilder.getIntegerType(operands[1]);
    break;
  case spirv::Opcode::OpTypeFloat: {
    if (operands.size() != 2) {
      return emitError(unknownLoc, "OpTypeFloat must have bitwidth parameter");
    }
    Type floatTy;
    switch (operands[1]) {
    case 16:
      floatTy = opBuilder.getF16Type();
      break;
    case 32:
      floatTy = opBuilder.getF32Type();
      break;
    case 64:
      floatTy = opBuilder.getF64Type();
      break;
    default:
      return emitError(unknownLoc, "unsupported OpTypeFloat bitwdith: ")
             << operands[1];
    }
    typeMap[operands[0]] = floatTy;
  } break;
  case spirv::Opcode::OpTypeVector: {
    if (operands.size() != 3) {
      return emitError(
          unknownLoc,
          "OpTypeVector must have element type and count parameters");
    }
    Type elementTy = getType(operands[1]);
    if (!elementTy) {
      return emitError(unknownLoc, "OpTypeVector references undefined <id> ")
             << operands[1];
    }
    typeMap[operands[0]] = opBuilder.getVectorType({operands[2]}, elementTy);
  } break;
  case spirv::Opcode::OpTypePointer: {
    if (operands.size() != 3) {
      return emitError(unknownLoc, "OpTypePointer must have two parameters");
    }
    auto pointeeType = getType(operands[2]);
    if (!pointeeType) {
      return emitError(unknownLoc, "unknown OpTypePointer pointee type <id> ")
             << operands[2];
    }
    auto storageClass = static_cast<spirv::StorageClass>(operands[1]);
    typeMap[operands[0]] = spirv::PointerType::get(pointeeType, storageClass);
  } break;
  case spirv::Opcode::OpTypeArray:
    return processArrayType(operands);
  case spirv::Opcode::OpTypeFunction:
    return processFunctionType(operands);
  default:
    return emitError(unknownLoc, "unhandled type instruction");
  }
  return success();
}

LogicalResult Deserializer::processArrayType(ArrayRef<uint32_t> operands) {
  if (operands.size() != 3) {
    return emitError(unknownLoc,
                     "OpTypeArray must have element type and count parameters");
  }

  Type elementTy = getType(operands[1]);
  if (!elementTy) {
    return emitError(unknownLoc, "OpTypeArray references undefined <id> ")
           << operands[1];
  }

  unsigned count = 0;
  auto *countValue = getValue(operands[2]);
  if (!countValue) {
    return emitError(unknownLoc, "OpTypeArray references undefined <id> ")
           << operands[2];
  }

  auto *defOp = countValue->getDefiningOp();
  if (auto constOp = dyn_cast<spirv::ConstantOp>(defOp)) {
    if (auto intVal = constOp.value().dyn_cast<IntegerAttr>()) {
      count = intVal.getInt();
    } else {
      return emitError(unknownLoc, "OpTypeArray count must come from a "
                                   "scalar integer constant instruction");
    }
  } else {
    return emitError(unknownLoc,
                     "unsupported OpTypeArray count generated from ")
           << defOp->getName();
  }

  typeMap[operands[0]] = spirv::ArrayType::get(
      elementTy, count, typeDecorations.lookup(operands[0]));
  return success();
}

LogicalResult Deserializer::processFunctionType(ArrayRef<uint32_t> operands) {
  assert(!operands.empty() && "No operands for processing function type");
  if (operands.size() == 1) {
    return emitError(unknownLoc, "missing return type for OpTypeFunction");
  }
  auto returnType = getType(operands[1]);
  if (!returnType) {
    return emitError(unknownLoc, "unknown return type in OpTypeFunction");
  }
  SmallVector<Type, 1> argTypes;
  for (size_t i = 2, e = operands.size(); i < e; ++i) {
    auto ty = getType(operands[i]);
    if (!ty) {
      return emitError(unknownLoc, "unknown argument type in OpTypeFunction");
    }
    argTypes.push_back(ty);
  }
  ArrayRef<Type> returnTypes;
  if (!isVoidType(returnType)) {
    returnTypes = llvm::makeArrayRef(returnType);
  }
  typeMap[operands[0]] = FunctionType::get(argTypes, returnTypes, context);
  return success();
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

LogicalResult Deserializer::processConstant(ArrayRef<uint32_t> operands,
                                            bool isSpec) {
  StringRef opname = isSpec ? "OpSpecConstant" : "OpConstant";

  if (operands.size() < 2) {
    return emitError(unknownLoc)
           << opname << " must have type <id> and result <id>";
  }
  if (operands.size() < 3) {
    return emitError(unknownLoc)
           << opname << " must have at least 1 more parameter";
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  auto checkOperandSizeForBitwidth = [&](unsigned bitwidth) -> LogicalResult {
    if (bitwidth == 64) {
      if (operands.size() == 4) {
        return success();
      }
      return emitError(unknownLoc)
             << opname << " should have 2 parameters for 64-bit values";
    }
    if (bitwidth <= 32) {
      if (operands.size() == 3) {
        return success();
      }

      return emitError(unknownLoc)
             << opname
             << " should have 1 parameter for values with no more than 32 bits";
    }
    return emitError(unknownLoc, "unsupported OpConstant bitwidth: ")
           << bitwidth;
  };

  spirv::ConstantOp op;
  UnitAttr isSpecConst = isSpec ? opBuilder.getUnitAttr() : UnitAttr();
  if (auto intType = resultType.dyn_cast<IntegerType>()) {
    auto bitwidth = intType.getWidth();
    if (failed(checkOperandSizeForBitwidth(bitwidth))) {
      return failure();
    }

    APInt value;
    if (bitwidth == 64) {
      // 64-bit integers are represented with two SPIR-V words. According to
      // SPIR-V spec: "When the type’s bit width is larger than one word, the
      // literal’s low-order words appear first."
      struct DoubleWord {
        uint32_t word1;
        uint32_t word2;
      } words = {operands[2], operands[3]};
      value = APInt(64, llvm::bit_cast<uint64_t>(words), /*isSigned=*/true);
    } else if (bitwidth <= 32) {
      value = APInt(bitwidth, operands[2], /*isSigned=*/true);
    }

    auto attr = opBuilder.getIntegerAttr(intType, value);
    op = opBuilder.create<spirv::ConstantOp>(unknownLoc, intType, attr,
                                             isSpecConst);
  } else if (auto floatType = resultType.dyn_cast<FloatType>()) {
    auto bitwidth = floatType.getWidth();
    if (failed(checkOperandSizeForBitwidth(bitwidth))) {
      return failure();
    }

    APFloat value(0.f);
    if (floatType.isF64()) {
      // Double values are represented with two SPIR-V words. According to
      // SPIR-V spec: "When the type’s bit width is larger than one word, the
      // literal’s low-order words appear first."
      struct DoubleWord {
        uint32_t word1;
        uint32_t word2;
      } words = {operands[2], operands[3]};
      value = APFloat(llvm::bit_cast<double>(words));
    } else if (floatType.isF32()) {
      value = APFloat(llvm::bit_cast<float>(operands[2]));
    } else if (floatType.isF16()) {
      APInt data(16, operands[2]);
      value = APFloat(APFloat::IEEEhalf(), data);
    }

    auto attr = opBuilder.getFloatAttr(floatType, value);
    op = opBuilder.create<spirv::ConstantOp>(unknownLoc, floatType, attr,
                                             isSpecConst);
  } else {
    return emitError(unknownLoc, "OpConstant can only generate values of "
                                 "scalar integer or floating-point type");
  }

  valueMap[operands[1]] = op.getResult();
  return success();
}

LogicalResult Deserializer::processConstantBool(bool isTrue,
                                                ArrayRef<uint32_t> operands,
                                                bool isSpec) {
  if (operands.size() != 2) {
    return emitError(unknownLoc, "Op")
           << (isSpec ? "Spec" : "") << "Constant"
           << (isTrue ? "True" : "False")
           << " must have type <id> and result <id>";
  }

  auto attr = opBuilder.getBoolAttr(isTrue);
  UnitAttr isSpecConst = isSpec ? opBuilder.getUnitAttr() : UnitAttr();
  auto op = opBuilder.create<spirv::ConstantOp>(
      unknownLoc, opBuilder.getI1Type(), attr, isSpecConst);

  valueMap[operands[1]] = op.getResult();
  return success();
}

LogicalResult
Deserializer::processConstantComposite(ArrayRef<uint32_t> operands,
                                       bool isSpec) {
  if (operands.size() < 2) {
    return emitError(unknownLoc,
                     "OpConstantComposite must have type <id> and result <id>");
  }
  if (operands.size() < 3) {
    return emitError(unknownLoc,
                     "OpConstantComposite must have at least 1 parameter");
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  SmallVector<Attribute, 4> elements;
  elements.reserve(operands.size() - 2);
  for (unsigned i = 2, e = operands.size(); i < e; ++i) {
    Value *value = getValue(operands[i]);
    if (!value) {
      return emitError(unknownLoc,
                       "OpConstantComposite references undefined <id> ")
             << operands[i];
    }
    auto *defOp = value->getDefiningOp();
    if (auto elementOp = dyn_cast<spirv::ConstantOp>(defOp)) {
      elements.push_back(elementOp.value());
    } else {
      return emitError(
                 unknownLoc,
                 "unsupported OpConstantComposite component generated from ")
             << defOp->getName();
    }
  }

  spirv::ConstantOp op;
  UnitAttr isSpecConst = isSpec ? opBuilder.getUnitAttr() : UnitAttr();
  if (auto vectorType = resultType.dyn_cast<VectorType>()) {
    auto attr = opBuilder.getDenseElementsAttr(vectorType, elements);
    op = opBuilder.create<spirv::ConstantOp>(unknownLoc, resultType, attr,
                                             isSpecConst);
  } else if (auto arrayType = resultType.dyn_cast<spirv::ArrayType>()) {
    auto attr = opBuilder.getArrayAttr(elements);
    op = opBuilder.create<spirv::ConstantOp>(unknownLoc, resultType, attr,
                                             isSpecConst);
  } else {
    return emitError(unknownLoc, "unsupported OpConstantComposite type: ")
           << resultType;
  }

  valueMap[operands[1]] = op.getResult();
  return success();
}

LogicalResult Deserializer::processConstantNull(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2) {
    return emitError(unknownLoc,
                     "OpConstantNull must have type <id> and result <id>");
  }

  Type resultType = getType(operands[0]);
  if (!resultType) {
    return emitError(unknownLoc, "undefined result type from <id> ")
           << operands[0];
  }

  spirv::ConstantOp op;
  if (resultType.isa<IntegerType>() || resultType.isa<FloatType>() ||
      resultType.isa<VectorType>()) {
    auto attr = opBuilder.getZeroAttr(resultType);
    UnitAttr isSpecConst;
    op = opBuilder.create<spirv::ConstantOp>(unknownLoc, resultType, attr,
                                             isSpecConst);
  } else {
    return emitError(unknownLoc, "unsupported OpConstantNull type: ")
           << resultType;
  }

  valueMap[operands[1]] = op.getResult();
  return success();
}

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

LogicalResult
Deserializer::sliceInstruction(spirv::Opcode &opcode,
                               ArrayRef<uint32_t> &operands,
                               Optional<spirv::Opcode> expectedOpcode) {
  auto binarySize = binary.size();
  if (curOffset >= binarySize) {
    return emitError(unknownLoc, "expected ")
           << (expectedOpcode ? spirv::stringifyOpcode(*expectedOpcode)
                              : "more")
           << " instruction";
  }

  // For each instruction, get its word count from the first word to slice it
  // from the stream properly, and then dispatch to the instruction handler.

  uint32_t wordCount = binary[curOffset] >> 16;

  if (wordCount == 0)
    return emitError(unknownLoc, "word count cannot be zero");

  uint32_t nextOffset = curOffset + wordCount;
  if (nextOffset > binarySize)
    return emitError(unknownLoc, "insufficient words for the last instruction");

  opcode = static_cast<spirv::Opcode>(binary[curOffset] & 0xffff);
  operands = binary.slice(curOffset + 1, wordCount - 1);
  curOffset = nextOffset;
  return success();
}

LogicalResult Deserializer::processInstruction(spirv::Opcode opcode,
                                               ArrayRef<uint32_t> operands,
                                               bool deferInstructions) {
  // First dispatch all the instructions whose opcode does not correspond to
  // those that have a direct mirror in the SPIR-V dialect
  switch (opcode) {
  case spirv::Opcode::OpMemoryModel:
    return processMemoryModel(operands);
  case spirv::Opcode::OpEntryPoint:
  case spirv::Opcode::OpExecutionMode:
    if (deferInstructions) {
      deferedInstructions.emplace_back(opcode, operands);
      return success();
    }
    break;
  case spirv::Opcode::OpVariable:
    if (isa<spirv::ModuleOp>(opBuilder.getBlock()->getParentOp())) {
      return processGlobalVariable(operands);
    }
    break;
  case spirv::Opcode::OpName:
    return processName(operands);
  case spirv::Opcode::OpTypeVoid:
  case spirv::Opcode::OpTypeBool:
  case spirv::Opcode::OpTypeInt:
  case spirv::Opcode::OpTypeFloat:
  case spirv::Opcode::OpTypeVector:
  case spirv::Opcode::OpTypeArray:
  case spirv::Opcode::OpTypeFunction:
  case spirv::Opcode::OpTypePointer:
    return processType(opcode, operands);
  case spirv::Opcode::OpConstant:
    return processConstant(operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstant:
    return processConstant(operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantComposite:
    return processConstantComposite(operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstantComposite:
    return processConstantComposite(operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantTrue:
    return processConstantBool(/*isTrue=*/true, operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstantTrue:
    return processConstantBool(/*isTrue=*/true, operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantFalse:
    return processConstantBool(/*isTrue=*/false, operands, /*isSpec=*/false);
  case spirv::Opcode::OpSpecConstantFalse:
    return processConstantBool(/*isTrue=*/false, operands, /*isSpec=*/true);
  case spirv::Opcode::OpConstantNull:
    return processConstantNull(operands);
  case spirv::Opcode::OpDecorate:
    return processDecoration(operands);
  case spirv::Opcode::OpFunction:
    return processFunction(operands);
  default:
    break;
  }
  return dispatchToAutogenDeserialization(opcode, operands);
}

namespace {

template <>
LogicalResult
Deserializer::processOp<spirv::EntryPointOp>(ArrayRef<uint32_t> words) {
  unsigned wordIndex = 0;
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc,
                     "missing Execution Model specification in OpEntryPoint");
  }
  auto exec_model = opBuilder.getI32IntegerAttr(words[wordIndex++]);
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc, "missing <id> in OpEntryPoint");
  }
  // Get the function <id>
  auto fnID = words[wordIndex++];
  // Get the function name
  auto fnName = decodeStringLiteral(words, wordIndex);
  // Verify that the function <id> matches the fnName
  auto parsedFunc = getFunction(fnID);
  if (!parsedFunc) {
    return emitError(unknownLoc, "no function matching <id> ") << fnID;
  }
  if (parsedFunc.getName() != fnName) {
    return emitError(unknownLoc, "function name mismatch between OpEntryPoint "
                                 "and OpFunction with <id> ")
           << fnID << ": " << fnName << " vs. " << parsedFunc.getName();
  }
  SmallVector<Attribute, 4> interface;
  while (wordIndex < words.size()) {
    auto arg = getVariable(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "undefined result <id> ")
             << words[wordIndex] << " while decoding OpEntryPoint";
    }
    interface.push_back(opBuilder.getSymbolRefAttr(arg.getOperation()));
    wordIndex++;
  }
  opBuilder.create<spirv::EntryPointOp>(unknownLoc, exec_model,
                                        opBuilder.getSymbolRefAttr(fnName),
                                        opBuilder.getArrayAttr(interface));
  return success();
}

template <>
LogicalResult
Deserializer::processOp<spirv::ExecutionModeOp>(ArrayRef<uint32_t> words) {
  unsigned wordIndex = 0;
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc,
                     "missing function result <id> in OpExecutionMode");
  }
  // Get the function <id> to get the name of the function
  auto fnID = words[wordIndex++];
  auto fn = getFunction(fnID);
  if (!fn) {
    return emitError(unknownLoc, "no function matching <id> ") << fnID;
  }
  // Get the Execution mode
  if (wordIndex >= words.size()) {
    return emitError(unknownLoc, "missing Execution Mode in OpExecutionMode");
  }
  auto execMode = opBuilder.getI32IntegerAttr(words[wordIndex++]);

  // Get the values
  SmallVector<Attribute, 4> attrListElems;
  while (wordIndex < words.size()) {
    attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
  }
  auto values = opBuilder.getArrayAttr(attrListElems);
  opBuilder.create<spirv::ExecutionModeOp>(
      unknownLoc, opBuilder.getSymbolRefAttr(fn.getName()), execMode, values);
  return success();
}

// Pull in auto-generated Deserializer::dispatchToAutogenDeserialization() and
// various Deserializer::processOp<...>() specializations.
#define GET_DESERIALIZATION_FNS
#include "mlir/Dialect/SPIRV/SPIRVSerialization.inc"
} // namespace

Optional<spirv::ModuleOp> spirv::deserialize(ArrayRef<uint32_t> binary,
                                             MLIRContext *context) {
  Deserializer deserializer(binary, context);

  if (failed(deserializer.deserialize()))
    return llvm::None;

  return deserializer.collect();
}
