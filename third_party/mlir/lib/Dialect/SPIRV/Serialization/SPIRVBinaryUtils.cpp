//===- SPIRVBinaryUtils.cpp - MLIR SPIR-V Binary Module Utilities ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for SPIR-V binary module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"

using namespace mlir;

void spirv::appendModuleHeader(SmallVectorImpl<uint32_t> &header,
                               uint32_t idBound) {
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
  header.push_back(idBound); // <id> bound
  header.push_back(0);       // Schema (reserved word)
}

/// Returns the word-count-prefixed opcode for an SPIR-V instruction.
uint32_t spirv::getPrefixedOpcode(uint32_t wordCount, spirv::Opcode opcode) {
  assert(((wordCount >> 16) == 0) && "word count out of range!");
  return (wordCount << 16) | static_cast<uint32_t>(opcode);
}

LogicalResult spirv::encodeStringLiteralInto(SmallVectorImpl<uint32_t> &binary,
                                             StringRef literal) {
  // We need to encode the literal and the null termination.
  auto encodingSize = literal.size() / 4 + 1;
  auto bufferStartSize = binary.size();
  binary.resize(bufferStartSize + encodingSize, 0);
  std::memcpy(binary.data() + bufferStartSize, literal.data(), literal.size());
  return success();
}
