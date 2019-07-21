//===- Serializer.cpp - MLIR SPIR-V Serialization -------------------------===//
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
// This file defines the MLIR SPIR-V module to SPIR-V binary seralization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Serialization.h"

#include "SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

static inline uint32_t getPrefixedOpcode(uint32_t wordCount,
                                         spirv::Opcode opcode) {
  assert(((wordCount >> 16) == 0) && "word count out of range!");
  return (wordCount << 16) | static_cast<uint32_t>(opcode);
}

static inline void buildInstruction(spirv::Opcode op,
                                    ArrayRef<uint32_t> operands,
                                    SmallVectorImpl<uint32_t> &binary) {
  uint32_t wordCount = 1 + operands.size();
  binary.push_back(getPrefixedOpcode(wordCount, op));
  if (!operands.empty()) {
    binary.append(operands.begin(), operands.end());
  }
}

static inline void encodeStringLiteral(StringRef literal,
                                       SmallVectorImpl<uint32_t> &buffer) {
  // Encoding is the literal + null termination
  auto encodingSize = literal.size() / 4 + 1;
  auto bufferStartSize = buffer.size();
  buffer.resize(bufferStartSize + encodingSize, 0);
  std::memcpy(buffer.data() + bufferStartSize, literal.data(), literal.size());
}

namespace {

/// A SPIR-V module serializer.
///
/// A SPIR-V binary module is a single linear stream of instructions; each
/// instruction is composed of 32-bit words with the layout:
///
///   | <word-count>|<opcode> |  <operand>   |  <operand>   | ... |
///   | <------ word -------> | <-- word --> | <-- word --> | ... |
///
/// For the first word, the 16 high-order bits are the word count of the
/// instruction, the 16 low-order bits are the opcode enumerant. The
/// instructions then belong to different sections, which must be laid out in
/// the particular order as specified in "2.4 Logical Layout of a Module" of
/// the SPIR-V spec.
class Serializer {
public:
  /// Creates a serializer for the given SPIR-V `module`.
  explicit Serializer(spirv::ModuleOp module) : module(module) {}

  /// Serializes the remembered SPIR-V module.
  LogicalResult serialize();

  /// Collects the final SPIR-V `binary`.
  void collect(SmallVectorImpl<uint32_t> &binary);

private:
  uint32_t getNextID() { return nextID++; }

  //===--------------------------------------------------------------------===//
  // Module structure
  //===--------------------------------------------------------------------===//

  /// Creates SPIR-V module header in the given `header`.
  LogicalResult processHeader();

  LogicalResult processMemoryModel();

  // It is illegal to use <id> 0 for SSA value in SPIR-V serialization. The
  // method uses that to check if the function is defined in the serialized
  // binary or not.
  uint32_t findFunctionID(StringRef fnName) const {
    return funcIDMap.lookup(fnName);
  }

  /// Processes a SPIR-V function op.
  LogicalResult processFuncOp(FuncOp op);

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  // It is illegal to use <id> 0 for SSA value in SPIR-V serialization. The
  // method uses that to check if the type is defined in the serialized binary
  // or not.
  uint32_t findTypeID(Type type) const { return typeIDMap.lookup(type); }

  Type voidType() { return mlir::NoneType::get(module.getContext()); }

  bool isVoidType(Type type) const { return type.isa<NoneType>(); }

  /// Main dispatch method for serializing a type. The result <id> of the
  /// serialized type will be returned as `typeID`.
  LogicalResult processType(Location loc, Type type, uint32_t &typeID);

  /// Method for preparing basic SPIR-V type serialization. Returns the type's
  /// opcode and operands for the instruction via `typeEnum` and `operands`.
  LogicalResult processBasicType(Location loc, Type type,
                                 spirv::Opcode &typeEnum,
                                 SmallVectorImpl<uint32_t> &operands);

  LogicalResult processFunctionType(Location loc, FunctionType type,
                                    spirv::Opcode &typeEnum,
                                    SmallVectorImpl<uint32_t> &operands);

  //===--------------------------------------------------------------------===//
  // Operations
  //===--------------------------------------------------------------------===//

  // It is illegal to use <id> 0 for SSA value in SPIR-V serialization. The
  // method uses that to check if `val` has a corresponding <id>
  uint32_t findValueID(Value *val) const { return valueIDMap.lookup(val); }

  /// Main dispatch method for serializing an operation.
  LogicalResult processOperation(Operation *op);

  /// Method to dispatch to the serialization function for an operation in
  /// SPIR-V dialect that is a mirror of an instruction in the SPIR-V spec.
  /// This is auto-generated from ODS. Dispatch is handled for all operations
  /// in SPIR-V dialect that have hasOpcode == 1.
  LogicalResult dispatchToAutogenSerialization(Operation *op);

  /// Method to serialize an operation in the SPIR-V dialect that is a mirror of
  /// an instruction in the SPIR-V spec. This is auto generated if hasOpcode ==
  /// 1 and autogenSerialization == 1 in ODS.
  template <typename OpTy> LogicalResult processOp(OpTy op) {
    return processOpImpl(op);
  }

  template <typename OpTy> LogicalResult processOpImpl(OpTy op) {
    return op.emitError("unsupported op serialization");
  }

private:
  /// The SPIR-V module to be serialized.
  spirv::ModuleOp module;

  /// The next available result <id>.
  uint32_t nextID = 1;

  // The following are for different SPIR-V instruction sections. They follow
  // the logical layout of a SPIR-V module.

  SmallVector<uint32_t, spirv::kHeaderWordCount> header;
  SmallVector<uint32_t, 4> capabilities;
  SmallVector<uint32_t, 0> extensions;
  SmallVector<uint32_t, 0> extendedSets;
  SmallVector<uint32_t, 3> memoryModel;
  SmallVector<uint32_t, 0> entryPoints;
  SmallVector<uint32_t, 4> executionModes;
  // TODO(antiagainst): debug instructions
  SmallVector<uint32_t, 0> names;
  SmallVector<uint32_t, 0> decorations;
  SmallVector<uint32_t, 0> typesGlobalValues;
  SmallVector<uint32_t, 0> functions;

  // Map from type used in SPIR-V module to their <id>s
  DenseMap<Type, uint32_t> typeIDMap;

  // Map from FuncOps name to <id>s.
  llvm::StringMap<uint32_t> funcIDMap;

  // Map from Value to Ids.
  DenseMap<Value *, uint32_t> valueIDMap;
};
} // namespace

LogicalResult Serializer::serialize() {
  if (failed(module.verify()))
    return failure();

  // TODO(antiagainst): handle the other sections
  processMemoryModel();

  // Iterate over the module body to serialze it. Assumptions are that there is
  // only one basic block in the moduleOp
  for (auto &op : module.getBlock()) {
    if (failed(processOperation(&op))) {
      return failure();
    }
  }
  return success();
}

void Serializer::collect(SmallVectorImpl<uint32_t> &binary) {
  auto moduleSize = header.size() + capabilities.size() + extensions.size() +
                    extendedSets.size() + memoryModel.size() +
                    entryPoints.size() + executionModes.size() +
                    decorations.size() + typesGlobalValues.size() +
                    functions.size();

  binary.clear();
  binary.reserve(moduleSize);

  processHeader();
  binary.append(header.begin(), header.end());
  binary.append(capabilities.begin(), capabilities.end());
  binary.append(extensions.begin(), extensions.end());
  binary.append(extendedSets.begin(), extendedSets.end());
  binary.append(memoryModel.begin(), memoryModel.end());
  binary.append(entryPoints.begin(), entryPoints.end());
  binary.append(executionModes.begin(), executionModes.end());
  binary.append(names.begin(), names.end());
  binary.append(decorations.begin(), decorations.end());
  binary.append(typesGlobalValues.begin(), typesGlobalValues.end());
  binary.append(functions.begin(), functions.end());
}
//===----------------------------------------------------------------------===//
// Module structure
//===----------------------------------------------------------------------===//

LogicalResult Serializer::processHeader() {
  // The serializer tool ID registered to the Khronos Group
  constexpr uint32_t kGeneratorNumber = 22;
  // The major and minor version number for the generated SPIR-V binary.
  // TODO(antiagainst): use target environment to select the version
  constexpr uint8_t kMajorVersion = 1;
  constexpr uint8_t kMinorVersion = 0;

  // See "2.3. Physical Layout of a SPIR-V Module and Instruction" in the SPIR-V
  // spec for the definition of the binary module header.
  //
  // The first five words of a SPIR-V module must be:
  // +-------------------------------------------------------------------------+
  // | Magic number                                                            |
  // +-------------------------------------------------------------------------+
  // | Version number (bytes: 0 | major number | minor number | 0)             |
  // +-------------------------------------------------------------------------+
  // | Generator magic number                                                  |
  // +-------------------------------------------------------------------------+
  // | Bound (all result <id>s in the module guaranteed to be less than it)    |
  // +-------------------------------------------------------------------------+
  // | 0 (reserved for instruction schema)                                     |
  // +-------------------------------------------------------------------------+
  header.push_back(spirv::kMagicNumber);
  header.push_back((kMajorVersion << 16) | (kMinorVersion << 8));
  header.push_back(kGeneratorNumber);
  header.push_back(nextID); // <id> bound
  header.push_back(0);      // Schema (reserved word)
  return success();
}

LogicalResult Serializer::processMemoryModel() {
  uint32_t mm = module.getAttrOfType<IntegerAttr>("memory_model").getInt();
  uint32_t am = module.getAttrOfType<IntegerAttr>("addressing_model").getInt();

  buildInstruction(spirv::Opcode::OpMemoryModel, {am, mm}, memoryModel);
  return success();
}

LogicalResult Serializer::processFuncOp(FuncOp op) {
  uint32_t fnTypeID = 0;
  // Generate type of the function.
  processType(op.getLoc(), op.getType(), fnTypeID);

  // Add the function definition.
  SmallVector<uint32_t, 4> operands;
  uint32_t resTypeID = 0;
  auto resultTypes = op.getType().getResults();
  if (resultTypes.size() > 1) {
    return emitError(op.getLoc(),
                     "cannot serialize function with multiple return types");
  }
  if (failed(processType(op.getLoc(),
                         (resultTypes.empty() ? voidType() : resultTypes[0]),
                         resTypeID))) {
    return failure();
  }
  operands.push_back(resTypeID);
  auto funcID = getNextID();
  funcIDMap[op.getName()] = funcID;
  operands.push_back(funcID);
  // TODO : Support other function control options.
  operands.push_back(static_cast<uint32_t>(spirv::FunctionControl::None));
  operands.push_back(fnTypeID);
  buildInstruction(spirv::Opcode::OpFunction, operands, functions);

  // Add function name.
  SmallVector<uint32_t, 4> nameOperands;
  nameOperands.push_back(funcID);
  encodeStringLiteral(op.getName(), nameOperands);
  buildInstruction(spirv::Opcode::OpName, nameOperands, names);

  // Declare the parameters.
  for (auto arg : op.getArguments()) {
    uint32_t argTypeID = 0;
    if (failed(processType(op.getLoc(), arg->getType(), argTypeID))) {
      return failure();
    }
    auto argValueID = getNextID();
    valueIDMap[arg] = argValueID;
    buildInstruction(spirv::Opcode::OpFunctionParameter,
                     {argTypeID, argValueID}, functions);
  }

  // Process the body.
  if (op.isExternal()) {
    return emitError(op.getLoc(), "external function is unhandled");
  }

  for (auto &b : op) {
    for (auto &op : b) {
      if (failed(processOperation(&op))) {
        return failure();
      }
    }
  }

  // Insert Function End.
  buildInstruction(spirv::Opcode::OpFunctionEnd, {}, functions);

  return success();
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

LogicalResult Serializer::processType(Location loc, Type type,
                                      uint32_t &typeID) {
  typeID = findTypeID(type);
  if (typeID) {
    return success();
  }
  typeID = getNextID();
  SmallVector<uint32_t, 4> operands;
  operands.push_back(typeID);
  auto typeEnum = spirv::Opcode::OpTypeVoid;
  if ((type.isa<FunctionType>() &&
       succeeded(processFunctionType(loc, type.cast<FunctionType>(), typeEnum,
                                     operands))) ||
      succeeded(processBasicType(loc, type, typeEnum, operands))) {
    buildInstruction(typeEnum, operands, typesGlobalValues);
    typeIDMap[type] = typeID;
    return success();
  }
  return failure();
}

LogicalResult
Serializer::processBasicType(Location loc, Type type, spirv::Opcode &typeEnum,
                             SmallVectorImpl<uint32_t> &operands) {
  if (isVoidType(type)) {
    typeEnum = spirv::Opcode::OpTypeVoid;
    return success();
  } else if (type.isa<FloatType>()) {
    typeEnum = spirv::Opcode::OpTypeFloat;
    operands.push_back(type.cast<FloatType>().getWidth());
    return success();
  } else if (type.isa<spirv::PointerType>()) {
    auto ptrType = type.cast<spirv::PointerType>();
    uint32_t pointeeTypeID = 0;
    if (failed(processType(loc, ptrType.getPointeeType(), pointeeTypeID))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypePointer;
    operands.push_back(static_cast<uint32_t>(ptrType.getStorageClass()));
    operands.push_back(pointeeTypeID);
    return success();
  }
  // TODO(ravishankarm) : Handle other types.
  return emitError(loc, "unhandled type in serialization : ") << type;
}

LogicalResult
Serializer::processFunctionType(Location loc, FunctionType type,
                                spirv::Opcode &typeEnum,
                                SmallVectorImpl<uint32_t> &operands) {
  typeEnum = spirv::Opcode::OpTypeFunction;
  assert(type.getNumResults() <= 1 &&
         "Serialization supports only a single return value");
  uint32_t resultID = 0;
  if (failed(processType(
          loc, type.getNumResults() == 1 ? type.getResult(0) : voidType(),
          resultID))) {
    return failure();
  }
  operands.push_back(resultID);
  for (auto &res : type.getInputs()) {
    uint32_t argTypeID = 0;
    if (failed(processType(loc, res, argTypeID))) {
      return failure();
    }
    operands.push_back(argTypeID);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

LogicalResult Serializer::processOperation(Operation *op) {
  // First dispatch the methods that do not directly mirror an operation from
  // the SPIR-V spec
  if (isa<FuncOp>(op)) {
    return processFuncOp(cast<FuncOp>(op));
  } else if (isa<spirv::ModuleEndOp>(op)) {
    return success();
  }
  return dispatchToAutogenSerialization(op);
}

namespace {
template <>
LogicalResult
Serializer::processOp<spirv::EntryPointOp>(spirv::EntryPointOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the ExectionModel.
  operands.push_back(static_cast<uint32_t>(op.execution_model()));
  // Add the function <id>.
  auto funcID = findFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be defined before spv.EntryPoint is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the name of the function.
  encodeStringLiteral(op.fn(), operands);

  // Add the interface values.
  for (auto val : op.interface()) {
    auto id = findValueID(val);
    if (!id) {
      return op.emitError("referencing unintialized variable <id>. "
                          "spv.EntryPoint is at the end of spv.module. All "
                          "referenced variables should already be defined");
    }
    operands.push_back(id);
  }
  buildInstruction(spirv::Opcode::OpEntryPoint, operands, entryPoints);
  return success();
}

template <>
LogicalResult
Serializer::processOp<spirv::ExecutionModeOp>(spirv::ExecutionModeOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the function <id>.
  auto funcID = findFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be serialized before ExecutionModeOp is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the ExecutionMode.
  operands.push_back(static_cast<uint32_t>(op.execution_mode()));

  // Serialize values if any.
  auto values = op.values();
  if (values) {
    for (auto &intVal : values.getValue()) {
      operands.push_back(static_cast<uint32_t>(
          intVal.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  buildInstruction(spirv::Opcode::OpExecutionMode, operands, executionModes);
  return success();
}

// Pull in auto-generated Serializer::dispatchToAutogenSerialization() and
// various processOpImpl specializations.
#define GET_SERIALIZATION_FNS
#include "mlir/Dialect/SPIRV/SPIRVSerialization.inc"
} // namespace

LogicalResult spirv::serialize(spirv::ModuleOp module,
                               SmallVectorImpl<uint32_t> &binary) {
  Serializer serializer(module);

  if (failed(serializer.serialize()))
    return failure();

  serializer.collect(binary);
  return success();
}
