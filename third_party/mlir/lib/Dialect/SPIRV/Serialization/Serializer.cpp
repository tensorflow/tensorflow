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
// This file defines the MLIR SPIR-V module to SPIR-V binary serialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Serialization.h"

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "spirv-serialization"

using namespace mlir;

/// Encodes an SPIR-V instruction with the given `opcode` and `operands` into
/// the given `binary` vector.
static LogicalResult encodeInstructionInto(SmallVectorImpl<uint32_t> &binary,
                                           spirv::Opcode op,
                                           ArrayRef<uint32_t> operands) {
  uint32_t wordCount = 1 + operands.size();
  binary.push_back(spirv::getPrefixedOpcode(wordCount, op));
    binary.append(operands.begin(), operands.end());
  return success();
}

/// A pre-order depth-first visitor function for processing basic blocks.
///
/// Visits the basic blocks starting from the given `headerBlock` in pre-order
/// depth-first manner and calls `blockHandler` on each block. Skips handling
/// blocks in the `skipBlocks` list. If `skipHeader` is true, `blockHandler`
/// will not be invoked in `headerBlock` but still handles all `headerBlock`'s
/// successors.
///
/// SPIR-V spec "2.16.1. Universal Validation Rules" requires that "the order
/// of blocks in a function must satisfy the rule that blocks appear before
/// all blocks they dominate." This can be achieved by a pre-order CFG
/// traversal algorithm. To make the serialization output more logical and
/// readable to human, we perform depth-first CFG traversal and delay the
/// serialization of the merge block and the continue block, if exists, until
/// after all other blocks have been processed.
static LogicalResult visitInPrettyBlockOrder(
    Block *headerBlock, llvm::function_ref<LogicalResult(Block *)> blockHandler,
    bool skipHeader = false, ArrayRef<Block *> skipBlocks = {}) {
  llvm::df_iterator_default_set<Block *, 4> doneBlocks;
  doneBlocks.insert(skipBlocks.begin(), skipBlocks.end());

  for (Block *block : llvm::depth_first_ext(headerBlock, doneBlocks)) {
    if (skipHeader && block == headerBlock)
      continue;
    if (failed(blockHandler(block)))
      return failure();
  }
  return success();
}

/// Returns the last structured control flow op's merge block if the given
/// `block` contains any structured control flow op. Otherwise returns nullptr.
static Block *getLastStructuredControlFlowOpMergeBlock(Block *block) {
  for (Operation &op : llvm::reverse(block->getOperations())) {
    if (auto selectionOp = dyn_cast<spirv::SelectionOp>(op))
      return selectionOp.getMergeBlock();
    if (auto loopOp = dyn_cast<spirv::LoopOp>(op))
      return loopOp.getMergeBlock();
  }
  return nullptr;
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
  explicit Serializer(spirv::ModuleOp module);

  /// Serializes the remembered SPIR-V module.
  LogicalResult serialize();

  /// Collects the final SPIR-V `binary`.
  void collect(SmallVectorImpl<uint32_t> &binary);

  /// (For debugging) prints each value and its corresponding result <id>.
  void printValueIDMap(raw_ostream &os);

private:
  // Note that there are two main categories of methods in this class:
  // * process*() methods are meant to fully serialize a SPIR-V module entity
  //   (header, type, op, etc.). They update internal vectors containing
  //   different binary sections. They are not meant to be called except the
  //   top-level serialization loop.
  // * prepare*() methods are meant to be helpers that prepare for serializing
  //   certain entity. They may or may not update internal vectors containing
  //   different binary sections. They are meant to be called among themselves
  //   or by other process*() methods for subtasks.

  //===--------------------------------------------------------------------===//
  // <id>
  //===--------------------------------------------------------------------===//

  // Note that it is illegal to use id <0> in SPIR-V binary module. Various
  // methods in this class, if using SPIR-V word (uint32_t) as interface,
  // check or return id <0> to indicate error in processing.

  /// Consumes the next unused <id>. This method will never return 0.
  uint32_t getNextID() { return nextID++; }

  //===--------------------------------------------------------------------===//
  // Module structure
  //===--------------------------------------------------------------------===//

  uint32_t getSpecConstID(StringRef constName) const {
    return specConstIDMap.lookup(constName);
  }

  uint32_t getVariableID(StringRef varName) const {
    return globalVarIDMap.lookup(varName);
  }

  uint32_t getFunctionID(StringRef fnName) const {
    return funcIDMap.lookup(fnName);
  }

  /// Gets the <id> for the function with the given name. Assigns the next
  /// available <id> if the function haven't been deserialized.
  uint32_t getOrCreateFunctionID(StringRef fnName);

  void processCapability();

  void processExtension();

  void processMemoryModel();

  LogicalResult processConstantOp(spirv::ConstantOp op);

  LogicalResult processSpecConstantOp(spirv::SpecConstantOp op);

  /// SPIR-V dialect supports OpUndef using spv.UndefOp that produces a SSA
  /// value to use with other operations. The SPIR-V spec recommends that
  /// OpUndef be generated at module level. The serialization generates an
  /// OpUndef for each type needed at module level.
  LogicalResult processUndefOp(spirv::UndefOp op);

  /// Emit OpName for the given `resultID`.
  LogicalResult processName(uint32_t resultID, StringRef name);

  /// Processes a SPIR-V function op.
  LogicalResult processFuncOp(FuncOp op);

  LogicalResult processVariableOp(spirv::VariableOp op);

  /// Process a SPIR-V GlobalVariableOp
  LogicalResult processGlobalVariableOp(spirv::GlobalVariableOp varOp);

  /// Process attributes that translate to decorations on the result <id>
  LogicalResult processDecoration(Location loc, uint32_t resultID,
                                  NamedAttribute attr);

  template <typename DType>
  LogicalResult processTypeDecoration(Location loc, DType type,
                                      uint32_t resultId) {
    return emitError(loc, "unhandled decoration for type:") << type;
  }

  /// Process member decoration
  LogicalResult processMemberDecoration(uint32_t structID, uint32_t memberIndex,
                                        spirv::Decoration decorationType,
                                        ArrayRef<uint32_t> values = {});

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  uint32_t getTypeID(Type type) const { return typeIDMap.lookup(type); }

  Type getVoidType() { return mlirBuilder.getNoneType(); }

  bool isVoidType(Type type) const { return type.isa<NoneType>(); }

  /// Returns true if the given type is a pointer type to a struct in Uniform or
  /// StorageBuffer storage class.
  bool isInterfaceStructPtrType(Type type) const;

  /// Main dispatch method for serializing a type. The result <id> of the
  /// serialized type will be returned as `typeID`.
  LogicalResult processType(Location loc, Type type, uint32_t &typeID);

  /// Method for preparing basic SPIR-V type serialization. Returns the type's
  /// opcode and operands for the instruction via `typeEnum` and `operands`.
  LogicalResult prepareBasicType(Location loc, Type type, uint32_t resultID,
                                 spirv::Opcode &typeEnum,
                                 SmallVectorImpl<uint32_t> &operands);

  LogicalResult prepareFunctionType(Location loc, FunctionType type,
                                    spirv::Opcode &typeEnum,
                                    SmallVectorImpl<uint32_t> &operands);

  //===--------------------------------------------------------------------===//
  // Constant
  //===--------------------------------------------------------------------===//

  uint32_t getConstantID(Attribute value) const {
    return constIDMap.lookup(value);
  }

  /// Main dispatch method for processing a constant with the given `constType`
  /// and `valueAttr`. `constType` is needed here because we can interpret the
  /// `valueAttr` as a different type than the type of `valueAttr` itself; for
  /// example, ArrayAttr, whose type is NoneType, is used for spirv::ArrayType
  /// constants.
  uint32_t prepareConstant(Location loc, Type constType, Attribute valueAttr);

  /// Prepares array attribute serialization. This method emits corresponding
  /// OpConstant* and returns the result <id> associated with it. Returns 0 if
  /// failed.
  uint32_t prepareArrayConstant(Location loc, Type constType, ArrayAttr attr);

  /// Prepares bool/int/float DenseElementsAttr serialization. This method
  /// iterates the DenseElementsAttr to construct the constant array, and
  /// returns the result <id>  associated with it. Returns 0 if failed. Note
  /// that the size of `index` must match the rank.
  /// TODO(hanchung): Consider to enhance splat elements cases. For splat cases,
  /// we don't need to loop over all elements, especially when the splat value
  /// is zero. We can use OpConstantNull when the value is zero.
  uint32_t prepareDenseElementsConstant(Location loc, Type constType,
                                        DenseElementsAttr valueAttr, int dim,
                                        MutableArrayRef<uint64_t> index);

  /// Prepares scalar attribute serialization. This method emits corresponding
  /// OpConstant* and returns the result <id> associated with it. Returns 0 if
  /// the attribute is not for a scalar bool/integer/float value. If `isSpec` is
  /// true, then the constant will be serialized as a specialization constant.
  uint32_t prepareConstantScalar(Location loc, Attribute valueAttr,
                                 bool isSpec = false);

  uint32_t prepareConstantBool(Location loc, BoolAttr boolAttr,
                               bool isSpec = false);

  uint32_t prepareConstantInt(Location loc, IntegerAttr intAttr,
                              bool isSpec = false);

  uint32_t prepareConstantFp(Location loc, FloatAttr floatAttr,
                             bool isSpec = false);

  //===--------------------------------------------------------------------===//
  // Control flow
  //===--------------------------------------------------------------------===//

  /// Returns the result <id> for the given block.
  uint32_t getBlockID(Block *block) const { return blockIDMap.lookup(block); }

  /// Returns the result <id> for the given block. If no <id> has been assigned,
  /// assigns the next available <id>
  uint32_t getOrCreateBlockID(Block *block);

  /// Processes the given `block` and emits SPIR-V instructions for all ops
  /// inside. Does not emit OpLabel for this block if `omitLabel` is true.
  /// `actionBeforeTerminator` is a callback that will be invoked before
  /// handling the terminator op. It can be used to inject the Op*Merge
  /// instruction if this is a SPIR-V selection/loop header block.
  LogicalResult
  processBlock(Block *block, bool omitLabel = false,
               llvm::function_ref<void()> actionBeforeTerminator = nullptr);

  /// Emits OpPhi instructions for the given block if it has block arguments.
  LogicalResult emitPhiForBlockArguments(Block *block);

  LogicalResult processSelectionOp(spirv::SelectionOp selectionOp);

  LogicalResult processLoopOp(spirv::LoopOp loopOp);

  LogicalResult processBranchConditionalOp(spirv::BranchConditionalOp);

  LogicalResult processBranchOp(spirv::BranchOp branchOp);

  //===--------------------------------------------------------------------===//
  // Operations
  //===--------------------------------------------------------------------===//

  LogicalResult encodeExtensionInstruction(Operation *op,
                                           StringRef extensionSetName,
                                           uint32_t opcode,
                                           ArrayRef<uint32_t> operands);

  uint32_t getValueID(Value *val) const { return valueIDMap.lookup(val); }

  LogicalResult processAddressOfOp(spirv::AddressOfOp addressOfOp);

  LogicalResult processReferenceOfOp(spirv::ReferenceOfOp referenceOfOp);

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
    return op.emitError("unsupported op serialization");
  }

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Emits an OpDecorate instruction to decorate the given `target` with the
  /// given `decoration`.
  LogicalResult emitDecoration(uint32_t target, spirv::Decoration decoration,
                               ArrayRef<uint32_t> params = {});

private:
  /// The SPIR-V module to be serialized.
  spirv::ModuleOp module;

  /// An MLIR builder for getting MLIR constructs.
  mlir::Builder mlirBuilder;

  /// The next available result <id>.
  uint32_t nextID = 1;

  // The following are for different SPIR-V instruction sections. They follow
  // the logical layout of a SPIR-V module.

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

  /// `functionHeader` contains all the instructions that must be in the first
  /// block in the function, and `functionBody` contains the rest. After
  /// processing FuncOp, the encoded instructions of a function are appended to
  /// `functions`. An example of instructions in `functionHeader` in order:
  /// OpFunction ...
  /// OpFunctionParameter ...
  /// OpFunctionParameter ...
  /// OpLabel ...
  /// OpVariable ...
  /// OpVariable ...
  SmallVector<uint32_t, 0> functionHeader;
  SmallVector<uint32_t, 0> functionBody;

  /// Map from type used in SPIR-V module to their <id>s.
  DenseMap<Type, uint32_t> typeIDMap;

  /// Map from constant values to their <id>s.
  DenseMap<Attribute, uint32_t> constIDMap;

  /// Map from specialization constant names to their <id>s.
  llvm::StringMap<uint32_t> specConstIDMap;

  /// Map from GlobalVariableOps name to <id>s.
  llvm::StringMap<uint32_t> globalVarIDMap;

  /// Map from FuncOps name to <id>s.
  llvm::StringMap<uint32_t> funcIDMap;

  /// Map from blocks to their <id>s.
  DenseMap<Block *, uint32_t> blockIDMap;

  /// Map from the Type to the <id> that represents undef value of that type.
  DenseMap<Type, uint32_t> undefValIDMap;

  /// Map from results of normal operations to their <id>s.
  DenseMap<Value *, uint32_t> valueIDMap;

  /// Map from extended instruction set name to <id>s.
  llvm::StringMap<uint32_t> extendedInstSetIDMap;

  /// Map from values used in OpPhi instructions to their offset in the
  /// `functions` section.
  ///
  /// When processing a block with arguments, we need to emit OpPhi
  /// instructions to record the predecessor block <id>s and the values they
  /// send to the block in question. But it's not guaranteed all values are
  /// visited and thus assigned result <id>s. So we need this list to capture
  /// the offsets into `functions` where a value is used so that we can fix it
  /// up later after processing all the blocks in a function.
  ///
  /// More concretely, say if we are visiting the following blocks:
  ///
  /// ```mlir
  /// ^phi(%arg0: i32):
  ///   ...
  /// ^parent1:
  ///   ...
  ///   spv.Branch ^phi(%val0: i32)
  /// ^parent2:
  ///   ...
  ///   spv.Branch ^phi(%val1: i32)
  /// ```
  ///
  /// When we are serializing the `^phi` block, we need to emit at the beginning
  /// of the block OpPhi instructions which has the following parameters:
  ///
  /// OpPhi id-for-i32 id-for-%arg0 id-for-%val0 id-for-^parent1
  ///                               id-for-%val1 id-for-^parent2
  ///
  /// But we don't know the <id> for %val0 and %val1 yet. One way is to visit
  /// all the blocks twice and use the first visit to assign an <id> to each
  /// value. But it's paying the overheads just for OpPhi emission. Instead,
  /// we still visit the blocks once for emission. When we emit the OpPhi
  /// instructions, we use 0 as a placeholder for the <id>s for %val0 and %val1.
  /// At the same time, we record their offsets in the emitted binary (which is
  /// placed inside `functions`) here. And then after emitting all blocks, we
  /// replace the dummy <id> 0 with the real result <id> by overwriting
  /// `functions[offset]`.
  DenseMap<Value *, llvm::SmallVector<size_t, 1>> deferredPhiValues;
};
} // namespace

Serializer::Serializer(spirv::ModuleOp module)
    : module(module), mlirBuilder(module.getContext()) {}

LogicalResult Serializer::serialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting serialization +++\n");

  if (failed(module.verify()))
    return failure();

  // TODO(antiagainst): handle the other sections
  processCapability();
  processExtension();
  processMemoryModel();

  // Iterate over the module body to serialize it. Assumptions are that there is
  // only one basic block in the moduleOp
  for (auto &op : module.getBlock()) {
    if (failed(processOperation(&op))) {
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "+++ completed serialization +++\n");
  return success();
}

void Serializer::collect(SmallVectorImpl<uint32_t> &binary) {
  auto moduleSize = spirv::kHeaderWordCount + capabilities.size() +
                    extensions.size() + extendedSets.size() +
                    memoryModel.size() + entryPoints.size() +
                    executionModes.size() + decorations.size() +
                    typesGlobalValues.size() + functions.size();

  binary.clear();
  binary.reserve(moduleSize);

  spirv::appendModuleHeader(binary, nextID);
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

void Serializer::printValueIDMap(raw_ostream &os) {
  os << "\n= Value <id> Map =\n\n";
  for (auto valueIDPair : valueIDMap) {
    Value *val = valueIDPair.first;
    os << "  " << val << " "
       << "id = " << valueIDPair.second << ' ';
    if (auto *op = val->getDefiningOp()) {
      os << "from op '" << op->getName() << "'";
    } else if (auto *arg = dyn_cast<BlockArgument>(val)) {
      Block *block = arg->getOwner();
      os << "from argument of block " << block << ' ';
      os << " in op '" << block->getParentOp()->getName() << "'";
    }
    os << '\n';
  }
}

//===----------------------------------------------------------------------===//
// Module structure
//===----------------------------------------------------------------------===//

uint32_t Serializer::getOrCreateFunctionID(StringRef fnName) {
  auto funcID = funcIDMap.lookup(fnName);
  if (!funcID) {
    funcID = getNextID();
    funcIDMap[fnName] = funcID;
  }
  return funcID;
}

void Serializer::processCapability() {
  auto caps = module.getAttrOfType<ArrayAttr>("capabilities");
  if (!caps)
    return;

  for (auto cap : caps.getValue()) {
    auto capStr = cap.cast<StringAttr>().getValue();
    auto capVal = spirv::symbolizeCapability(capStr);
    encodeInstructionInto(capabilities, spirv::Opcode::OpCapability,
                          {static_cast<uint32_t>(*capVal)});
  }
}

void Serializer::processExtension() {
  auto exts = module.getAttrOfType<ArrayAttr>("extensions");
  if (!exts)
    return;

  SmallVector<uint32_t, 16> extName;
  for (auto ext : exts.getValue()) {
    auto extStr = ext.cast<StringAttr>().getValue();
    extName.clear();
    spirv::encodeStringLiteralInto(extName, extStr);
    encodeInstructionInto(extensions, spirv::Opcode::OpExtension, extName);
  }
}

void Serializer::processMemoryModel() {
  uint32_t mm = module.getAttrOfType<IntegerAttr>("memory_model").getInt();
  uint32_t am = module.getAttrOfType<IntegerAttr>("addressing_model").getInt();

  encodeInstructionInto(memoryModel, spirv::Opcode::OpMemoryModel, {am, mm});
}

LogicalResult Serializer::processConstantOp(spirv::ConstantOp op) {
  if (auto resultID = prepareConstant(op.getLoc(), op.getType(), op.value())) {
    valueIDMap[op.getResult()] = resultID;
    return success();
  }
  return failure();
}

LogicalResult Serializer::processSpecConstantOp(spirv::SpecConstantOp op) {
  if (auto resultID = prepareConstantScalar(op.getLoc(), op.default_value(),
                                            /*isSpec=*/true)) {
    // Emit the OpDecorate instruction for SpecId.
    if (auto specID = op.getAttrOfType<IntegerAttr>("spec_id")) {
      auto val = static_cast<uint32_t>(specID.getInt());
      emitDecoration(resultID, spirv::Decoration::SpecId, {val});
    }

    specConstIDMap[op.sym_name()] = resultID;
    return processName(resultID, op.sym_name());
  }
  return failure();
}

LogicalResult Serializer::processUndefOp(spirv::UndefOp op) {
  auto undefType = op.getType();
  auto &id = undefValIDMap[undefType];
  if (!id) {
    id = getNextID();
    uint32_t typeID = 0;
    if (failed(processType(op.getLoc(), undefType, typeID)) ||
        failed(encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpUndef,
                                     {typeID, id}))) {
      return failure();
    }
  }
  valueIDMap[op.getResult()] = id;
  return success();
}

LogicalResult Serializer::processDecoration(Location loc, uint32_t resultID,
                                            NamedAttribute attr) {
  auto attrName = attr.first.strref();
  auto decorationName = mlir::convertToCamelCase(attrName, true);
  auto decoration = spirv::symbolizeDecoration(decorationName);
  if (!decoration) {
    return emitError(
               loc, "non-argument attributes expected to have snake-case-ified "
                    "decoration name, unhandled attribute with name : ")
           << attrName;
  }
  SmallVector<uint32_t, 1> args;
  switch (decoration.getValue()) {
  case spirv::Decoration::DescriptorSet:
  case spirv::Decoration::Binding:
    if (auto intAttr = attr.second.dyn_cast<IntegerAttr>()) {
      args.push_back(intAttr.getValue().getZExtValue());
      break;
    }
    return emitError(loc, "expected integer attribute for ") << attrName;
  case spirv::Decoration::BuiltIn:
    if (auto strAttr = attr.second.dyn_cast<StringAttr>()) {
      auto enumVal = spirv::symbolizeBuiltIn(strAttr.getValue());
      if (enumVal) {
        args.push_back(static_cast<uint32_t>(enumVal.getValue()));
        break;
      }
      return emitError(loc, "invalid ")
             << attrName << " attribute " << strAttr.getValue();
    }
    return emitError(loc, "expected string attribute for ") << attrName;
  default:
    return emitError(loc, "unhandled decoration ") << decorationName;
  }
  return emitDecoration(resultID, decoration.getValue(), args);
}

LogicalResult Serializer::processName(uint32_t resultID, StringRef name) {
  assert(!name.empty() && "unexpected empty string for OpName");

  SmallVector<uint32_t, 4> nameOperands;
  nameOperands.push_back(resultID);
  if (failed(spirv::encodeStringLiteralInto(nameOperands, name))) {
    return failure();
  }
  return encodeInstructionInto(names, spirv::Opcode::OpName, nameOperands);
}

namespace {
template <>
LogicalResult Serializer::processTypeDecoration<spirv::ArrayType>(
    Location loc, spirv::ArrayType type, uint32_t resultID) {
  if (type.hasLayout()) {
    // OpDecorate %arrayTypeSSA ArrayStride strideLiteral
    return emitDecoration(resultID, spirv::Decoration::ArrayStride,
                          {static_cast<uint32_t>(type.getArrayStride())});
  }
  return success();
}

LogicalResult
Serializer::processMemberDecoration(uint32_t structID, uint32_t memberIndex,
                                    spirv::Decoration decorationType,
                                    ArrayRef<uint32_t> values) {
  SmallVector<uint32_t, 4> args(
      {structID, memberIndex, static_cast<uint32_t>(decorationType)});
  if (!values.empty()) {
    args.append(values.begin(), values.end());
  }
  return encodeInstructionInto(decorations, spirv::Opcode::OpMemberDecorate,
                               args);
}
} // namespace

LogicalResult Serializer::processFuncOp(FuncOp op) {
  LLVM_DEBUG(llvm::dbgs() << "-- start function '" << op.getName() << "' --\n");
  assert(functionHeader.empty() && functionBody.empty());

  uint32_t fnTypeID = 0;
  // Generate type of the function.
  processType(op.getLoc(), op.getType(), fnTypeID);

  // Add the function definition.
  SmallVector<uint32_t, 4> operands;
  uint32_t resTypeID = 0;
  auto resultTypes = op.getType().getResults();
  if (resultTypes.size() > 1) {
    return op.emitError("cannot serialize function with multiple return types");
  }
  if (failed(processType(op.getLoc(),
                         (resultTypes.empty() ? getVoidType() : resultTypes[0]),
                         resTypeID))) {
    return failure();
  }
  operands.push_back(resTypeID);
  auto funcID = getOrCreateFunctionID(op.getName());
  operands.push_back(funcID);
  // TODO : Support other function control options.
  operands.push_back(static_cast<uint32_t>(spirv::FunctionControl::None));
  operands.push_back(fnTypeID);
  encodeInstructionInto(functionHeader, spirv::Opcode::OpFunction, operands);

  // Add function name.
  if (failed(processName(funcID, op.getName()))) {
    return failure();
  }

  // Declare the parameters.
  for (auto arg : op.getArguments()) {
    uint32_t argTypeID = 0;
    if (failed(processType(op.getLoc(), arg->getType(), argTypeID))) {
      return failure();
    }
    auto argValueID = getNextID();
    valueIDMap[arg] = argValueID;
    encodeInstructionInto(functionHeader, spirv::Opcode::OpFunctionParameter,
                          {argTypeID, argValueID});
  }

  // Process the body.
  if (op.isExternal()) {
    return op.emitError("external function is unhandled");
  }

  // Some instructions (e.g., OpVariable) in a function must be in the first
  // block in the function. These instructions will be put in functionHeader.
  // Thus, we put the label in functionHeader first, and omit it from the first
  // block.
  encodeInstructionInto(functionHeader, spirv::Opcode::OpLabel,
                        {getOrCreateBlockID(&op.front())});
  processBlock(&op.front(), /*omitLabel=*/true);
  if (failed(visitInPrettyBlockOrder(
          &op.front(), [&](Block *block) { return processBlock(block); },
          /*skipHeader=*/true))) {
    return failure();
  }

  // There might be OpPhi instructions who have value references needing to fix.
  for (auto deferredValue : deferredPhiValues) {
    Value *value = deferredValue.first;
    uint32_t id = getValueID(value);
    LLVM_DEBUG(llvm::dbgs() << "[phi] fix reference of value " << value
                            << " to id = " << id << '\n');
    assert(id && "OpPhi references undefined value!");
    for (size_t offset : deferredValue.second)
      functionBody[offset] = id;
  }
  deferredPhiValues.clear();

  LLVM_DEBUG(llvm::dbgs() << "-- completed function '" << op.getName()
                          << "' --\n");
  // Insert OpFunctionEnd.
  if (failed(encodeInstructionInto(functionBody, spirv::Opcode::OpFunctionEnd,
                                   {}))) {
    return failure();
  }

  functions.append(functionHeader.begin(), functionHeader.end());
  functions.append(functionBody.begin(), functionBody.end());
  functionHeader.clear();
  functionBody.clear();

  return success();
}

LogicalResult Serializer::processVariableOp(spirv::VariableOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  auto attr = op.getAttr(spirv::attributeName<spirv::StorageClass>());
  if (attr) {
    operands.push_back(static_cast<uint32_t>(
        attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
  for (auto arg : op.getODSOperands(0)) {
    auto argID = getValueID(arg);
    if (!argID) {
      return emitError(op.getLoc(), "operand 0 has a use before def");
    }
    operands.push_back(argID);
  }
  encodeInstructionInto(functionHeader, spirv::getOpcode<spirv::VariableOp>(),
                        operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs,
                     [&](StringRef elided) { return attr.first.is(elided); })) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
Serializer::processGlobalVariableOp(spirv::GlobalVariableOp varOp) {
  // Get TypeID.
  uint32_t resultTypeID = 0;
  SmallVector<StringRef, 4> elidedAttrs;
  if (failed(processType(varOp.getLoc(), varOp.type(), resultTypeID))) {
    return failure();
  }

  if (isInterfaceStructPtrType(varOp.type())) {
    auto structType = varOp.type()
                          .cast<spirv::PointerType>()
                          .getPointeeType()
                          .cast<spirv::StructType>();
    if (failed(
            emitDecoration(getTypeID(structType), spirv::Decoration::Block))) {
      return varOp.emitError("cannot decorate ")
             << structType << " with Block decoration";
    }
  }

  elidedAttrs.push_back("type");
  SmallVector<uint32_t, 4> operands;
  operands.push_back(resultTypeID);
  auto resultID = getNextID();

  // Encode the name.
  auto varName = varOp.sym_name();
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());
  if (failed(processName(resultID, varName))) {
    return failure();
  }
  globalVarIDMap[varName] = resultID;
  operands.push_back(resultID);

  // Encode StorageClass.
  operands.push_back(static_cast<uint32_t>(varOp.storageClass()));

  // Encode initialization.
  if (auto initializer = varOp.initializer()) {
    auto initializerID = getVariableID(initializer.getValue());
    if (!initializerID) {
      return emitError(varOp.getLoc(),
                       "invalid usage of undefined variable as initializer");
    }
    operands.push_back(initializerID);
    elidedAttrs.push_back("initializer");
  }

  if (failed(encodeInstructionInto(typesGlobalValues, spirv::Opcode::OpVariable,
                                   operands))) {
    elidedAttrs.push_back("initializer");
    return failure();
  }

  // Encode decorations.
  for (auto attr : varOp.getAttrs()) {
    if (llvm::any_of(elidedAttrs,
                     [&](StringRef elided) { return attr.first.is(elided); })) {
      continue;
    }
    if (failed(processDecoration(varOp.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

bool Serializer::isInterfaceStructPtrType(Type type) const {
  if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    auto storageClass = ptrType.getStorageClass();
    if (storageClass == spirv::StorageClass::Uniform ||
        storageClass == spirv::StorageClass::StorageBuffer) {
      return ptrType.getPointeeType().isa<spirv::StructType>();
    }
  }
  return false;
}

LogicalResult Serializer::processType(Location loc, Type type,
                                      uint32_t &typeID) {
  typeID = getTypeID(type);
  if (typeID) {
    return success();
  }
  typeID = getNextID();
  SmallVector<uint32_t, 4> operands;
  operands.push_back(typeID);
  auto typeEnum = spirv::Opcode::OpTypeVoid;
  if ((type.isa<FunctionType>() &&
       succeeded(prepareFunctionType(loc, type.cast<FunctionType>(), typeEnum,
                                     operands))) ||
      succeeded(prepareBasicType(loc, type, typeID, typeEnum, operands))) {
    typeIDMap[type] = typeID;
    return encodeInstructionInto(typesGlobalValues, typeEnum, operands);
  }
  return failure();
}

LogicalResult
Serializer::prepareBasicType(Location loc, Type type, uint32_t resultID,
                             spirv::Opcode &typeEnum,
                             SmallVectorImpl<uint32_t> &operands) {
  if (isVoidType(type)) {
    typeEnum = spirv::Opcode::OpTypeVoid;
    return success();
  }

  if (auto intType = type.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      typeEnum = spirv::Opcode::OpTypeBool;
      return success();
    }

    typeEnum = spirv::Opcode::OpTypeInt;
    operands.push_back(intType.getWidth());
    // TODO(antiagainst): support unsigned integers
    operands.push_back(1);
    return success();
  }

  if (auto floatType = type.dyn_cast<FloatType>()) {
    typeEnum = spirv::Opcode::OpTypeFloat;
    operands.push_back(floatType.getWidth());
    return success();
  }

  if (auto vectorType = type.dyn_cast<VectorType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processType(loc, vectorType.getElementType(), elementTypeID))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypeVector;
    operands.push_back(elementTypeID);
    operands.push_back(vectorType.getNumElements());
    return success();
  }

  if (auto arrayType = type.dyn_cast<spirv::ArrayType>()) {
    typeEnum = spirv::Opcode::OpTypeArray;
    uint32_t elementTypeID = 0;
    if (failed(processType(loc, arrayType.getElementType(), elementTypeID))) {
      return failure();
    }
    operands.push_back(elementTypeID);
    if (auto elementCountID = prepareConstantInt(
            loc, mlirBuilder.getI32IntegerAttr(arrayType.getNumElements()))) {
      operands.push_back(elementCountID);
    }
    return processTypeDecoration(loc, arrayType, resultID);
  }

  if (auto ptrType = type.dyn_cast<spirv::PointerType>()) {
    uint32_t pointeeTypeID = 0;
    if (failed(processType(loc, ptrType.getPointeeType(), pointeeTypeID))) {
      return failure();
    }
    typeEnum = spirv::Opcode::OpTypePointer;
    operands.push_back(static_cast<uint32_t>(ptrType.getStorageClass()));
    operands.push_back(pointeeTypeID);
    return success();
  }

  if (auto runtimeArrayType = type.dyn_cast<spirv::RuntimeArrayType>()) {
    uint32_t elementTypeID = 0;
    if (failed(processType(loc, runtimeArrayType.getElementType(),
                           elementTypeID))) {
      return failure();
    }
    operands.push_back(elementTypeID);
    typeEnum = spirv::Opcode::OpTypeRuntimeArray;
    return success();
  }

  if (auto structType = type.dyn_cast<spirv::StructType>()) {
    bool hasLayout = structType.hasLayout();
    for (auto elementIndex :
         llvm::seq<uint32_t>(0, structType.getNumElements())) {
      uint32_t elementTypeID = 0;
      if (failed(processType(loc, structType.getElementType(elementIndex),
                             elementTypeID))) {
        return failure();
      }
      operands.push_back(elementTypeID);
      if (hasLayout) {
        // Decorate each struct member with an offset
        if (failed(processMemberDecoration(
                resultID, elementIndex, spirv::Decoration::Offset,
                static_cast<uint32_t>(structType.getOffset(elementIndex))))) {
          return emitError(loc, "cannot decorate ")
                 << elementIndex << "-th member of " << structType
                 << " with its offset";
        }
      }
    }
    SmallVector<spirv::StructType::MemberDecorationInfo, 4> memberDecorations;
    structType.getMemberDecorations(memberDecorations);
    for (auto &memberDecoration : memberDecorations) {
      if (failed(processMemberDecoration(resultID, memberDecoration.first,
                                         memberDecoration.second))) {
        return emitError(loc, "cannot decorate ")
               << memberDecoration.first << "-th member of " << structType
               << " with " << stringifyDecoration(memberDecoration.second);
      }
    }
    typeEnum = spirv::Opcode::OpTypeStruct;
    return success();
  }

  // TODO(ravishankarm) : Handle other types.
  return emitError(loc, "unhandled type in serialization: ") << type;
}

LogicalResult
Serializer::prepareFunctionType(Location loc, FunctionType type,
                                spirv::Opcode &typeEnum,
                                SmallVectorImpl<uint32_t> &operands) {
  typeEnum = spirv::Opcode::OpTypeFunction;
  assert(type.getNumResults() <= 1 &&
         "serialization supports only a single return value");
  uint32_t resultID = 0;
  if (failed(processType(
          loc, type.getNumResults() == 1 ? type.getResult(0) : getVoidType(),
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
// Constant
//===----------------------------------------------------------------------===//

uint32_t Serializer::prepareConstant(Location loc, Type constType,
                                     Attribute valueAttr) {
  if (auto id = prepareConstantScalar(loc, valueAttr)) {
    return id;
  }

  // This is a composite literal. We need to handle each component separately
  // and then emit an OpConstantComposite for the whole.

  if (auto id = getConstantID(valueAttr)) {
    return id;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  uint32_t resultID = 0;
  if (auto attr = valueAttr.dyn_cast<DenseElementsAttr>()) {
    int rank = attr.getType().dyn_cast<ShapedType>().getRank();
    SmallVector<uint64_t, 4> index(rank);
    resultID = prepareDenseElementsConstant(loc, constType, attr,
                                            /*dim=*/0, index);
  } else if (auto arrayAttr = valueAttr.dyn_cast<ArrayAttr>()) {
    resultID = prepareArrayConstant(loc, constType, arrayAttr);
  }

  if (resultID == 0) {
    emitError(loc, "cannot serialize attribute: ") << valueAttr;
    return 0;
  }

  constIDMap[valueAttr] = resultID;
  return resultID;
}

uint32_t Serializer::prepareArrayConstant(Location loc, Type constType,
                                          ArrayAttr attr) {
  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  uint32_t resultID = getNextID();
  SmallVector<uint32_t, 4> operands = {typeID, resultID};
  operands.reserve(attr.size() + 2);
  auto elementType = constType.cast<spirv::ArrayType>().getElementType();
  for (Attribute elementAttr : attr) {
    if (auto elementID = prepareConstant(loc, elementType, elementAttr)) {
      operands.push_back(elementID);
    } else {
      return 0;
    }
  }
  spirv::Opcode opcode = spirv::Opcode::OpConstantComposite;
  encodeInstructionInto(typesGlobalValues, opcode, operands);

  return resultID;
}

// TODO(hanchung): Turn the below function into iterative function, instead of
// recursive function.
uint32_t
Serializer::prepareDenseElementsConstant(Location loc, Type constType,
                                         DenseElementsAttr valueAttr, int dim,
                                         MutableArrayRef<uint64_t> index) {
  auto shapedType = valueAttr.getType().dyn_cast<ShapedType>();
  assert(dim <= shapedType.getRank());
  if (shapedType.getRank() == dim) {
    if (auto attr = valueAttr.dyn_cast<DenseIntElementsAttr>()) {
      return attr.getType().getElementType().isInteger(1)
                 ? prepareConstantBool(loc, attr.getValue<BoolAttr>(index))
                 : prepareConstantInt(loc, attr.getValue<IntegerAttr>(index));
    }
    if (auto attr = valueAttr.dyn_cast<DenseFPElementsAttr>()) {
      return prepareConstantFp(loc, attr.getValue<FloatAttr>(index));
    }
    return 0;
  }

  uint32_t typeID = 0;
  if (failed(processType(loc, constType, typeID))) {
    return 0;
  }

  uint32_t resultID = getNextID();
  SmallVector<uint32_t, 4> operands = {typeID, resultID};
  operands.reserve(shapedType.getDimSize(dim) + 2);
  auto elementType = constType.cast<spirv::CompositeType>().getElementType(0);
  for (int i = 0; i < shapedType.getDimSize(dim); ++i) {
    index[dim] = i;
    if (auto elementID = prepareDenseElementsConstant(
            loc, elementType, valueAttr, dim + 1, index)) {
      operands.push_back(elementID);
    } else {
      return 0;
    }
  }
  spirv::Opcode opcode = spirv::Opcode::OpConstantComposite;
  encodeInstructionInto(typesGlobalValues, opcode, operands);

  return resultID;
}

uint32_t Serializer::prepareConstantScalar(Location loc, Attribute valueAttr,
                                           bool isSpec) {
  if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>()) {
    return prepareConstantFp(loc, floatAttr, isSpec);
  }
  if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
    return prepareConstantInt(loc, intAttr, isSpec);
  }
  if (auto boolAttr = valueAttr.dyn_cast<BoolAttr>()) {
    return prepareConstantBool(loc, boolAttr, isSpec);
  }

  return 0;
}

uint32_t Serializer::prepareConstantBool(Location loc, BoolAttr boolAttr,
                                         bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate normal constants, but not specialization constants.
    if (auto id = getConstantID(boolAttr)) {
      return id;
    }
  }

  // Process the type for this bool literal
  uint32_t typeID = 0;
  if (failed(processType(loc, boolAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  auto opcode = boolAttr.getValue()
                    ? (isSpec ? spirv::Opcode::OpSpecConstantTrue
                              : spirv::Opcode::OpConstantTrue)
                    : (isSpec ? spirv::Opcode::OpSpecConstantFalse
                              : spirv::Opcode::OpConstantFalse);
  encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID});

  if (!isSpec) {
    constIDMap[boolAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareConstantInt(Location loc, IntegerAttr intAttr,
                                        bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate normal constants, but not specialization constants.
    if (auto id = getConstantID(intAttr)) {
      return id;
    }
  }

  // Process the type for this integer literal
  uint32_t typeID = 0;
  if (failed(processType(loc, intAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  APInt value = intAttr.getValue();
  unsigned bitwidth = value.getBitWidth();
  bool isSigned = value.isSignedIntN(bitwidth);

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  // According to SPIR-V spec, "When the type's bit width is less than 32-bits,
  // the literal's value appears in the low-order bits of the word, and the
  // high-order bits must be 0 for a floating-point type, or 0 for an integer
  // type with Signedness of 0, or sign extended when Signedness is 1."
  if (bitwidth == 32 || bitwidth == 16) {
    uint32_t word = 0;
    if (isSigned) {
      word = static_cast<int32_t>(value.getSExtValue());
    } else {
      word = static_cast<uint32_t>(value.getZExtValue());
    }
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  }
  // According to SPIR-V spec: "When the type's bit width is larger than one
  // word, the literal’s low-order words appear first."
  else if (bitwidth == 64) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words;
    if (isSigned) {
      words = llvm::bit_cast<DoubleWord>(value.getSExtValue());
    } else {
      words = llvm::bit_cast<DoubleWord>(value.getZExtValue());
    }
    encodeInstructionInto(typesGlobalValues, opcode,
                          {typeID, resultID, words.word1, words.word2});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss, /*isSigned=*/false);

    emitError(loc, "cannot serialize ")
        << bitwidth << "-bit integer literal: " << rss.str();
    return 0;
  }

  if (!isSpec) {
    constIDMap[intAttr] = resultID;
  }
  return resultID;
}

uint32_t Serializer::prepareConstantFp(Location loc, FloatAttr floatAttr,
                                       bool isSpec) {
  if (!isSpec) {
    // We can de-duplicate normal constants, but not specialization constants.
    if (auto id = getConstantID(floatAttr)) {
      return id;
    }
  }

  // Process the type for this float literal
  uint32_t typeID = 0;
  if (failed(processType(loc, floatAttr.getType(), typeID))) {
    return 0;
  }

  auto resultID = getNextID();
  APFloat value = floatAttr.getValue();
  APInt intValue = value.bitcastToAPInt();

  auto opcode =
      isSpec ? spirv::Opcode::OpSpecConstant : spirv::Opcode::OpConstant;

  if (&value.getSemantics() == &APFloat::IEEEsingle()) {
    uint32_t word = llvm::bit_cast<uint32_t>(value.convertToFloat());
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } else if (&value.getSemantics() == &APFloat::IEEEdouble()) {
    struct DoubleWord {
      uint32_t word1;
      uint32_t word2;
    } words = llvm::bit_cast<DoubleWord>(value.convertToDouble());
    encodeInstructionInto(typesGlobalValues, opcode,
                          {typeID, resultID, words.word1, words.word2});
  } else if (&value.getSemantics() == &APFloat::IEEEhalf()) {
    uint32_t word =
        static_cast<uint32_t>(value.bitcastToAPInt().getZExtValue());
    encodeInstructionInto(typesGlobalValues, opcode, {typeID, resultID, word});
  } else {
    std::string valueStr;
    llvm::raw_string_ostream rss(valueStr);
    value.print(rss);

    emitError(loc, "cannot serialize ")
        << floatAttr.getType() << "-typed float literal: " << rss.str();
    return 0;
  }

  if (!isSpec) {
    constIDMap[floatAttr] = resultID;
  }
  return resultID;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

uint32_t Serializer::getOrCreateBlockID(Block *block) {
  if (uint32_t id = getBlockID(block))
    return id;
  return blockIDMap[block] = getNextID();
}

LogicalResult
Serializer::processBlock(Block *block, bool omitLabel,
                         llvm::function_ref<void()> actionBeforeTerminator) {
  LLVM_DEBUG(llvm::dbgs() << "processing block " << block << ":\n");
  LLVM_DEBUG(block->print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');
  if (!omitLabel) {
    uint32_t blockID = getOrCreateBlockID(block);
    LLVM_DEBUG(llvm::dbgs()
               << "[block] " << block << " (id = " << blockID << ")\n");

    // Emit OpLabel for this block.
    encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {blockID});
  }

  // Emit OpPhi instructions for block arguments, if any.
  if (failed(emitPhiForBlockArguments(block)))
    return failure();

  // Process each op in this block except the terminator.
  for (auto &op : llvm::make_range(block->begin(), std::prev(block->end()))) {
    if (failed(processOperation(&op)))
      return failure();
  }

  // Process the terminator.
  if (actionBeforeTerminator)
    actionBeforeTerminator();
  if (failed(processOperation(&block->back())))
    return failure();

  return success();
}

LogicalResult Serializer::emitPhiForBlockArguments(Block *block) {
  // Nothing to do if this block has no arguments or it's the entry block, which
  // always has the same arguments as the function signature.
  if (block->args_empty() || block->isEntryBlock())
    return success();

  // If the block has arguments, we need to create SPIR-V OpPhi instructions.
  // A SPIR-V OpPhi instruction is of the syntax:
  //   OpPhi | result type | result <id> | (value <id>, parent block <id>) pair
  // So we need to collect all predecessor blocks and the arguments they send
  // to this block.
  SmallVector<std::pair<Block *, Operation::operand_iterator>, 4> predecessors;
  for (Block *predecessor : block->getPredecessors()) {
    auto *terminator = predecessor->getTerminator();
    // Check whether this predecessor block contains a structured control flow
    // op. If so, the structured control flow op will be serialized to multiple
    // SPIR-V blocks. The branch op jumping to the OpPhi's block then resides in
    // the last structured control flow op's merge block.
    if (auto *merge = getLastStructuredControlFlowOpMergeBlock(predecessor))
      predecessor = merge;
    if (auto branchOp = dyn_cast<spirv::BranchOp>(terminator)) {
      predecessors.emplace_back(predecessor, branchOp.operand_begin());
    } else {
      return terminator->emitError("unimplemented terminator for Phi creation");
    }
  }

  // Then create OpPhi instruction for each of the block argument.
  for (auto argIndex : llvm::seq<unsigned>(0, block->getNumArguments())) {
    BlockArgument *arg = block->getArgument(argIndex);

    // Get the type <id> and result <id> for this OpPhi instruction.
    uint32_t phiTypeID = 0;
    if (failed(processType(arg->getLoc(), arg->getType(), phiTypeID)))
      return failure();
    uint32_t phiID = getNextID();

    LLVM_DEBUG(llvm::dbgs() << "[phi] for block argument #" << argIndex << ' '
                            << arg << " (id = " << phiID << ")\n");

    SmallVector<uint32_t, 8> phiArgs;
    phiArgs.push_back(phiTypeID);
    phiArgs.push_back(phiID);

    for (auto predIndex : llvm::seq<unsigned>(0, predecessors.size())) {
      Value *value = *(predecessors[predIndex].second + argIndex);
      uint32_t predBlockId = getOrCreateBlockID(predecessors[predIndex].first);
      LLVM_DEBUG(llvm::dbgs() << "[phi] use predecessor (id = " << predBlockId
                              << ") value " << value << ' ');
      // Each pair is a value <id> ...
      uint32_t valueId = getValueID(value);
      if (valueId == 0) {
        // The op generating this value hasn't been visited yet so we don't have
        // an <id> assigned yet. Record this to fix up later.
        LLVM_DEBUG(llvm::dbgs() << "(need to fix)\n");
        deferredPhiValues[value].push_back(functionBody.size() + 1 +
                                           phiArgs.size());
      } else {
        LLVM_DEBUG(llvm::dbgs() << "(id = " << valueId << ")\n");
      }
      phiArgs.push_back(valueId);
      // ... and a parent block <id>.
      phiArgs.push_back(predBlockId);
    }

    encodeInstructionInto(functionBody, spirv::Opcode::OpPhi, phiArgs);
    valueIDMap[arg] = phiID;
  }

  return success();
}

LogicalResult Serializer::processSelectionOp(spirv::SelectionOp selectionOp) {
  // Assign <id>s to all blocks so that branches inside the SelectionOp can
  // resolve properly.
  auto &body = selectionOp.body();
  for (Block &block : body)
    getOrCreateBlockID(&block);

  auto *headerBlock = selectionOp.getHeaderBlock();
  auto *mergeBlock = selectionOp.getMergeBlock();
  auto mergeID = getBlockID(mergeBlock);

  // Emit the selection header block, which dominates all other blocks, first.
  // We need to emit an OpSelectionMerge instruction before the loop header
  // block's terminator.
  auto emitSelectionMerge = [&]() {
    // TODO(antiagainst): properly support loop control here
    encodeInstructionInto(
        functionBody, spirv::Opcode::OpSelectionMerge,
        {mergeID, static_cast<uint32_t>(spirv::LoopControl::None)});
  };
  // For structured selection, we cannot have blocks in the selection construct
  // branching to the selection header block. Entering the selection (and
  // reaching the selection header) must be from the block containing the
  // spv.selection op. If there are ops ahead of the spv.selection op in the
  // block, we can "merge" them into the selection header. So here we don't need
  // to emit a separate block; just continue with the existing block.
  if (failed(processBlock(headerBlock, /*omitLabel=*/true, emitSelectionMerge)))
    return failure();

  // Process all blocks with a depth-first visitor starting from the header
  // block. The selection header block and merge block are skipped by this
  // visitor.
  if (failed(visitInPrettyBlockOrder(
          headerBlock, [&](Block *block) { return processBlock(block); },
          /*skipHeader=*/true, /*skipBlocks=*/{mergeBlock})))
    return failure();

  // There is nothing to do for the merge block in the selection, which just
  // contains a spv._merge op, itself. But we need to have an OpLabel
  // instruction to start a new SPIR-V block for ops following this SelectionOp.
  // The block should use the <id> for the merge block.
  return encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {mergeID});
}

LogicalResult Serializer::processLoopOp(spirv::LoopOp loopOp) {
  // Assign <id>s to all blocks so that branches inside the LoopOp can resolve
  // properly. We don't need to assign for the entry block, which is just for
  // satisfying MLIR region's structural requirement.
  auto &body = loopOp.body();
  for (Block &block :
       llvm::make_range(std::next(body.begin(), 1), body.end())) {
    getOrCreateBlockID(&block);
  }
  auto *headerBlock = loopOp.getHeaderBlock();
  auto *continueBlock = loopOp.getContinueBlock();
  auto *mergeBlock = loopOp.getMergeBlock();
  auto headerID = getBlockID(headerBlock);
  auto continueID = getBlockID(continueBlock);
  auto mergeID = getBlockID(mergeBlock);

  // This LoopOp is in some MLIR block with preceding and following ops. In the
  // binary format, it should reside in separate SPIR-V blocks from its
  // preceding and following ops. So we need to emit unconditional branches to
  // jump to this LoopOp's SPIR-V blocks and jumping back to the normal flow
  // afterwards.
  encodeInstructionInto(functionBody, spirv::Opcode::OpBranch, {headerID});

  // We omit the LoopOp's entry block and start serialization from the loop
  // header block. The entry block should not contain any additional ops other
  // than a single spv.Branch that jumps to the loop header block. However,
  // the spv.Branch can contain additional block arguments. Those block
  // arguments must come from out of the loop using implicit capture. We will
  // need to query the <id> for the value sent and the <id> for the incoming
  // parent block. For the latter, we need to make sure this block is
  // registered. The value sent should come from the block this loop resides in.
  blockIDMap[loopOp.getEntryBlock()] =
      getBlockID(loopOp.getOperation()->getBlock());

  // Emit the loop header block, which dominates all other blocks, first. We
  // need to emit an OpLoopMerge instruction before the loop header block's
  // terminator.
  auto emitLoopMerge = [&]() {
    // TODO(antiagainst): properly support loop control here
    encodeInstructionInto(
        functionBody, spirv::Opcode::OpLoopMerge,
        {mergeID, continueID, static_cast<uint32_t>(spirv::LoopControl::None)});
  };
  if (failed(processBlock(headerBlock, /*omitLabel=*/false, emitLoopMerge)))
    return failure();

  // Process all blocks with a depth-first visitor starting from the header
  // block. The loop header block, loop continue block, and loop merge block are
  // skipped by this visitor and handled later in this function.
  if (failed(visitInPrettyBlockOrder(
          headerBlock, [&](Block *block) { return processBlock(block); },
          /*skipHeader=*/true, /*skipBlocks=*/{continueBlock, mergeBlock})))
    return failure();

  // We have handled all other blocks. Now get to the loop continue block.
  if (failed(processBlock(continueBlock)))
    return failure();

  // There is nothing to do for the merge block in the loop, which just contains
  // a spv._merge op, itself. But we need to have an OpLabel instruction to
  // start a new SPIR-V block for ops following this LoopOp. The block should
  // use the <id> for the merge block.
  return encodeInstructionInto(functionBody, spirv::Opcode::OpLabel, {mergeID});
}

LogicalResult Serializer::processBranchConditionalOp(
    spirv::BranchConditionalOp condBranchOp) {
  auto conditionID = getValueID(condBranchOp.condition());
  auto trueLabelID = getOrCreateBlockID(condBranchOp.getTrueBlock());
  auto falseLabelID = getOrCreateBlockID(condBranchOp.getFalseBlock());
  SmallVector<uint32_t, 5> arguments{conditionID, trueLabelID, falseLabelID};

  if (auto weights = condBranchOp.branch_weights()) {
    for (auto val : weights->getValue())
      arguments.push_back(val.cast<IntegerAttr>().getInt());
  }

  return encodeInstructionInto(functionBody, spirv::Opcode::OpBranchConditional,
                               arguments);
}

LogicalResult Serializer::processBranchOp(spirv::BranchOp branchOp) {
  return encodeInstructionInto(functionBody, spirv::Opcode::OpBranch,
                               {getOrCreateBlockID(branchOp.getTarget())});
}

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

LogicalResult Serializer::encodeExtensionInstruction(
    Operation *op, StringRef extensionSetName, uint32_t extensionOpcode,
    ArrayRef<uint32_t> operands) {
  // Check if the extension has been imported.
  auto &setID = extendedInstSetIDMap[extensionSetName];
  if (!setID) {
    setID = getNextID();
    SmallVector<uint32_t, 16> importOperands;
    importOperands.push_back(setID);
    if (failed(
            spirv::encodeStringLiteralInto(importOperands, extensionSetName)) ||
        failed(encodeInstructionInto(
            extendedSets, spirv::Opcode::OpExtInstImport, importOperands))) {
      return failure();
    }
  }

  // The first two operands are the result type <id> and result <id>. The set
  // <id> and the opcode need to be insert after this.
  if (operands.size() < 2) {
    return op->emitError("extended instructions must have a result encoding");
  }
  SmallVector<uint32_t, 8> extInstOperands;
  extInstOperands.reserve(operands.size() + 2);
  extInstOperands.append(operands.begin(), std::next(operands.begin(), 2));
  extInstOperands.push_back(setID);
  extInstOperands.push_back(extensionOpcode);
  extInstOperands.append(std::next(operands.begin(), 2), operands.end());
  return encodeInstructionInto(functionBody, spirv::Opcode::OpExtInst,
                               extInstOperands);
}

LogicalResult Serializer::processAddressOfOp(spirv::AddressOfOp addressOfOp) {
  auto varName = addressOfOp.variable();
  auto variableID = getVariableID(varName);
  if (!variableID) {
    return addressOfOp.emitError("unknown result <id> for variable ")
           << varName;
  }
  valueIDMap[addressOfOp.pointer()] = variableID;
  return success();
}

LogicalResult
Serializer::processReferenceOfOp(spirv::ReferenceOfOp referenceOfOp) {
  auto constName = referenceOfOp.spec_const();
  auto constID = getSpecConstID(constName);
  if (!constID) {
    return referenceOfOp.emitError(
               "unknown result <id> for specialization constant ")
           << constName;
  }
  valueIDMap[referenceOfOp.reference()] = constID;
  return success();
}

LogicalResult Serializer::processOperation(Operation *opInst) {
  LLVM_DEBUG(llvm::dbgs() << "[op] '" << opInst->getName() << "'\n");

  // First dispatch the ops that do not directly mirror an instruction from
  // the SPIR-V spec.
  return TypeSwitch<Operation *, LogicalResult>(opInst)
      .Case([&](spirv::AddressOfOp op) { return processAddressOfOp(op); })
      .Case([&](spirv::BranchOp op) { return processBranchOp(op); })
      .Case([&](spirv::BranchConditionalOp op) {
        return processBranchConditionalOp(op);
      })
      .Case([&](spirv::ConstantOp op) { return processConstantOp(op); })
      .Case([&](FuncOp op) { return processFuncOp(op); })
      .Case([&](spirv::GlobalVariableOp op) {
        return processGlobalVariableOp(op);
      })
      .Case([&](spirv::LoopOp op) { return processLoopOp(op); })
      .Case([&](spirv::ModuleEndOp) { return success(); })
      .Case([&](spirv::ReferenceOfOp op) { return processReferenceOfOp(op); })
      .Case([&](spirv::SelectionOp op) { return processSelectionOp(op); })
      .Case([&](spirv::SpecConstantOp op) { return processSpecConstantOp(op); })
      .Case([&](spirv::UndefOp op) { return processUndefOp(op); })
      .Case([&](spirv::VariableOp op) { return processVariableOp(op); })

      // Then handle all the ops that directly mirror SPIR-V instructions with
      // auto-generated methods.
      .Default([&](auto *op) { return dispatchToAutogenSerialization(op); });
}

namespace {
template <>
LogicalResult
Serializer::processOp<spirv::EntryPointOp>(spirv::EntryPointOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the ExecutionModel.
  operands.push_back(static_cast<uint32_t>(op.execution_model()));
  // Add the function <id>.
  auto funcID = getFunctionID(op.fn());
  if (!funcID) {
    return op.emitError("missing <id> for function ")
           << op.fn()
           << "; function needs to be defined before spv.EntryPoint is "
              "serialized";
  }
  operands.push_back(funcID);
  // Add the name of the function.
  spirv::encodeStringLiteralInto(operands, op.fn());

  // Add the interface values.
  if (auto interface = op.interface()) {
    for (auto var : interface.getValue()) {
      auto id = getVariableID(var.cast<FlatSymbolRefAttr>().getValue());
      if (!id) {
        return op.emitError("referencing undefined global variable."
                            "spv.EntryPoint is at the end of spv.module. All "
                            "referenced variables should already be defined");
      }
      operands.push_back(id);
    }
  }
  return encodeInstructionInto(entryPoints, spirv::Opcode::OpEntryPoint,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::ControlBarrierOp>(spirv::ControlBarrierOp op) {
  StringRef argNames[] = {"execution_scope", "memory_scope",
                          "memory_semantics"};
  SmallVector<uint32_t, 3> operands;

  for (auto argName : argNames) {
    auto argIntAttr = op.getAttrOfType<IntegerAttr>(argName);
    auto operand = prepareConstantInt(op.getLoc(), argIntAttr);
    if (!operand) {
      return failure();
    }
    operands.push_back(operand);
  }

  return encodeInstructionInto(functionBody, spirv::Opcode::OpControlBarrier,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::ExecutionModeOp>(spirv::ExecutionModeOp op) {
  SmallVector<uint32_t, 4> operands;
  // Add the function <id>.
  auto funcID = getFunctionID(op.fn());
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
  return encodeInstructionInto(executionModes, spirv::Opcode::OpExecutionMode,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::MemoryBarrierOp>(spirv::MemoryBarrierOp op) {
  StringRef argNames[] = {"memory_scope", "memory_semantics"};
  SmallVector<uint32_t, 2> operands;

  for (auto argName : argNames) {
    auto argIntAttr = op.getAttrOfType<IntegerAttr>(argName);
    auto operand = prepareConstantInt(op.getLoc(), argIntAttr);
    if (!operand) {
      return failure();
    }
    operands.push_back(operand);
  }

  return encodeInstructionInto(functionBody, spirv::Opcode::OpMemoryBarrier,
                               operands);
}

template <>
LogicalResult
Serializer::processOp<spirv::FunctionCallOp>(spirv::FunctionCallOp op) {
  auto funcName = op.callee();
  uint32_t resTypeID = 0;

  llvm::SmallVector<Type, 1> resultTypes(op.getResultTypes());
  if (failed(processType(op.getLoc(),
                         (resultTypes.empty() ? getVoidType() : resultTypes[0]),
                         resTypeID))) {
    return failure();
  }

  auto funcID = getOrCreateFunctionID(funcName);
  auto funcCallID = getNextID();
  SmallVector<uint32_t, 8> operands{resTypeID, funcCallID, funcID};

  for (auto *value : op.arguments()) {
    auto valueID = getValueID(value);
    assert(valueID && "cannot find a value for spv.FunctionCall");
    operands.push_back(valueID);
  }

  if (!resultTypes.empty()) {
    valueIDMap[op.getResult(0)] = funcCallID;
  }

  return encodeInstructionInto(functionBody, spirv::Opcode::OpFunctionCall,
                               operands);
}

// Pull in auto-generated Serializer::dispatchToAutogenSerialization() and
// various Serializer::processOp<...>() specializations.
#define GET_SERIALIZATION_FNS
#include "mlir/Dialect/SPIRV/SPIRVSerialization.inc"
} // namespace

LogicalResult Serializer::emitDecoration(uint32_t target,
                                         spirv::Decoration decoration,
                                         ArrayRef<uint32_t> params) {
  uint32_t wordCount = 3 + params.size();
  decorations.push_back(
      spirv::getPrefixedOpcode(wordCount, spirv::Opcode::OpDecorate));
  decorations.push_back(target);
  decorations.push_back(static_cast<uint32_t>(decoration));
  decorations.append(params.begin(), params.end());
  return success();
}

LogicalResult spirv::serialize(spirv::ModuleOp module,
                               SmallVectorImpl<uint32_t> &binary) {
  Serializer serializer(module);

  if (failed(serializer.serialize()))
    return failure();

  LLVM_DEBUG(serializer.printValueIDMap(llvm::dbgs()));

  serializer.collect(binary);
  return success();
}
