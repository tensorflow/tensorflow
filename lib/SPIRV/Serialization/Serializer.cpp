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

#include "mlir/SPIRV/Serialization.h"

#include "SPIRVBinaryUtils.h"
#include "mlir/SPIRV/SPIRVOps.h"
#include "mlir/SPIRV/SPIRVTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

static inline uint32_t getPrefixedOpcode(uint32_t wordCount, uint32_t opcode) {
  assert(((wordCount >> 16) == 0) && "word count out of range!");
  return (wordCount << 16) | opcode;
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
  /// Creates SPIR-V module header in the given `header`.
  void processHeader(SmallVectorImpl<uint32_t> &header);

  void processMemoryModel();

private:
  /// The SPIR-V module to be serialized.
  spirv::ModuleOp module;

  /// The next available result <id>.
  uint32_t nextID = 0;

  // The following are for different SPIR-V instruction sections. They follow
  // the logical layout of a SPIR-V module.

  SmallVector<uint32_t, 4> capabilities;
  SmallVector<uint32_t, 0> extensions;
  SmallVector<uint32_t, 0> extendedSets;
  SmallVector<uint32_t, 3> memoryModel;
  SmallVector<uint32_t, 0> entryPoints;
  SmallVector<uint32_t, 4> executionModes;
  // TODO(antiagainst): debug instructions
  SmallVector<uint32_t, 0> decorations;
  SmallVector<uint32_t, 0> typesGlobalValues;
  SmallVector<uint32_t, 0> functions;
};
} // namespace

namespace {
#include "mlir/SPIRV/SPIRVSerialization.inc"
}

LogicalResult Serializer::serialize() {
  if (failed(module.verify()))
    return failure();

  // TODO(antiagainst): handle the other sections
  processMemoryModel();

  return success();
}

void Serializer::collect(SmallVectorImpl<uint32_t> &binary) {
  // The number of words in the SPIR-V module header

  auto moduleSize = spirv::kHeaderWordCount + capabilities.size() +
                    extensions.size() + extendedSets.size() +
                    memoryModel.size() + entryPoints.size() +
                    executionModes.size() + decorations.size() +
                    typesGlobalValues.size() + functions.size();

  binary.clear();
  binary.reserve(moduleSize);

  processHeader(binary);
  binary.append(capabilities.begin(), capabilities.end());
  binary.append(extensions.begin(), extensions.end());
  binary.append(extendedSets.begin(), extendedSets.end());
  binary.append(memoryModel.begin(), memoryModel.end());
  binary.append(entryPoints.begin(), entryPoints.end());
  binary.append(executionModes.begin(), executionModes.end());
  binary.append(decorations.begin(), decorations.end());
  binary.append(typesGlobalValues.begin(), typesGlobalValues.end());
  binary.append(functions.begin(), functions.end());
}

void Serializer::processHeader(SmallVectorImpl<uint32_t> &header) {
  // The serializer tool ID registered to the Khronos Group
  constexpr uint32_t kGeneratorNumber = 22;
  // The major and minor version number for the generated SPIR-V binary.
  // TODO(antiagainst): use target environment to select the version
  constexpr uint8_t kMajorVersion = 1;
  constexpr uint8_t kMinorVersion = 0;

  header.push_back(spirv::kMagicNumber);
  header.push_back((kMajorVersion << 16) | (kMinorVersion << 8));
  header.push_back(kGeneratorNumber);
  header.push_back(nextID); // ID bound
  header.push_back(0);      // Schema (reserved word)
}

void Serializer::processMemoryModel() {
  // TODO(antiagainst): use IntegerAttr-backed enum attributes to avoid the
  // excessive string conversions here.
  auto mm = static_cast<uint32_t>(*spirv::symbolizeMemoryModel(
      module.getAttrOfType<StringAttr>("memory_model").getValue()));
  auto am = static_cast<uint32_t>(*spirv::symbolizeAddressingModel(
      module.getAttrOfType<StringAttr>("addressing_model").getValue()));

  constexpr uint32_t kNumWords = 3;

  memoryModel.reserve(kNumWords);
  memoryModel.assign(
      {getPrefixedOpcode(kNumWords, spirv::kOpMemoryModelOpcode), am, mm});
}

bool spirv::serialize(spirv::ModuleOp module,
                      SmallVectorImpl<uint32_t> &binary) {
  Serializer serializer(module);

  if (failed(serializer.serialize()))
    return false;

  serializer.collect(binary);
  return true;
}
