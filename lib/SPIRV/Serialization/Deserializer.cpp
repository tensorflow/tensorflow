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

#include "mlir/SPIRV/Serialization.h"

#include "SPIRVBinaryUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/SPIRV/SPIRVOps.h"
#include "mlir/SPIRV/SPIRVTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

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
  /// Processes SPIR-V module header.
  LogicalResult processHeader();

  /// Processes a SPIR-V instruction with the given `opcode` and `operands`.
  LogicalResult processInstruction(uint32_t opcode,
                                   ArrayRef<uint32_t> operands);

  LogicalResult processMemoryModel(ArrayRef<uint32_t> operands);

  /// Initializes the `module` ModuleOp in this deserializer instance.
  spirv::ModuleOp createModuleOp();

private:
  /// The SPIR-V binary module.
  ArrayRef<uint32_t> binary;

  /// The current word offset into the binary module.
  unsigned curOffset = 0;

  /// MLIRContext to create SPIR-V ModuleOp into.
  MLIRContext *context;

  // TODO(antiagainst): create Location subclass for binary blob
  UnknownLoc unknownLoc;

  /// The SPIR-V ModuleOp.
  Optional<spirv::ModuleOp> module;

  OpBuilder opBuilder;
};
} // namespace

Deserializer::Deserializer(ArrayRef<uint32_t> binary, MLIRContext *context)
    : binary(binary), context(context), unknownLoc(UnknownLoc::get(context)),
      module(createModuleOp()),
      opBuilder(module->getOperation()->getRegion(0)) {}

LogicalResult Deserializer::deserialize() {
  if (failed(processHeader()))
    return failure();

  auto binarySize = binary.size();
  curOffset = spirv::kHeaderWordCount;

  while (curOffset < binarySize) {
    // For each instruction, get its word count from the first word to slice it
    // from the stream properly, and then dispatch to the instruction handler.

    uint32_t wordCount = binary[curOffset] >> 16;
    uint32_t opcode = binary[curOffset] & 0xffff;

    if (wordCount == 0)
      return context->emitError(unknownLoc, "word count cannot be zero");

    uint32_t nextOffset = curOffset + wordCount;
    if (nextOffset > binarySize)
      return context->emitError(unknownLoc,
                                "insufficient words for the last instruction");

    auto operands = binary.slice(curOffset + 1, wordCount - 1);
    if (failed(processInstruction(opcode, operands)))
      return failure();

    curOffset = nextOffset;
  }

  return success();
}

Optional<spirv::ModuleOp> Deserializer::collect() { return module; }

LogicalResult Deserializer::processHeader() {
  if (binary.size() < spirv::kHeaderWordCount)
    return context->emitError(unknownLoc,
                              "SPIR-V binary module must have a 5-word header");

  if (binary[0] != spirv::kMagicNumber)
    return context->emitError(unknownLoc, "incorrect magic number");

  // TODO(antiagainst): generator number, bound, schema
  return success();
}

LogicalResult Deserializer::processInstruction(uint32_t opcode,
                                               ArrayRef<uint32_t> operands) {
  switch (opcode) {
  case spirv::kOpMemoryModelOpcode:
    return processMemoryModel(operands);
  default:
    break;
  }
  return context->emitError(unknownLoc, "NYI: opcode ") << opcode;
}

LogicalResult Deserializer::processMemoryModel(ArrayRef<uint32_t> operands) {
  if (operands.size() != 2)
    return context->emitError(unknownLoc,
                              "OpMemoryModel must have two operands");

  // TODO(antiagainst): use IntegerAttr-backed enum attributes to avoid the
  // excessive string conversions here.

  auto am = spirv::symbolizeAddressingModel(operands.front());
  if (!am)
    return context->emitError(unknownLoc,
                              "unknown addressing model for OpMemoryModel");

  auto mm = spirv::symbolizeMemoryModel(operands.back());
  if (!mm)
    return context->emitError(unknownLoc,
                              "unknown memory model for OpMemoryModel");

  module->setAttr(
      "addressing_model",
      opBuilder.getStringAttr(spirv::stringifyAddressingModel(*am)));
  module->setAttr("memory_model",
                  opBuilder.getStringAttr(spirv::stringifyMemoryModel(*mm)));

  return success();
}

spirv::ModuleOp Deserializer::createModuleOp() {
  Builder builder(context);
  OperationState state(unknownLoc, spirv::ModuleOp::getOperationName());
  // TODO(antiagainst): use target environment to select the version
  state.addAttribute("major_version", builder.getI32IntegerAttr(1));
  state.addAttribute("minor_version", builder.getI32IntegerAttr(0));
  spirv::ModuleOp::build(&builder, &state);
  return llvm::cast<spirv::ModuleOp>(Operation::create(state));
}

Optional<spirv::ModuleOp> spirv::deserialize(ArrayRef<uint32_t> binary,
                                             MLIRContext *context) {
  Deserializer deserializer(binary, context);

  if (failed(deserializer.deserialize()))
    return llvm::None;

  return deserializer.collect();
}
