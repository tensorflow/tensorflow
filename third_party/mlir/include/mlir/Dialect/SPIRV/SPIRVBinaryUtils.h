//===- SPIRVBinaryUtils.cpp - SPIR-V Binary Module Utils --------*- C++ -*-===//
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
// This file declares common utilities for SPIR-V binary module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRV_BINARY_UTILS_H_
#define MLIR_DIALECT_SPIRV_SPIRV_BINARY_UTILS_H_

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace spirv {

/// SPIR-V binary header word count
constexpr unsigned kHeaderWordCount = 5;

/// SPIR-V magic number
constexpr uint32_t kMagicNumber = 0x07230203;

/// The serializer tool ID registered to the Khronos Group
constexpr uint32_t kGeneratorNumber = 22;

/// Auto-generated getOpcode<*Op>() specializations
#define GET_SPIRV_SERIALIZATION_UTILS
#include "mlir/Dialect/SPIRV/SPIRVSerialization.inc"

/// Appends a SPRI-V module header to `header` with the given `idBound`.
void appendModuleHeader(SmallVectorImpl<uint32_t> &header, uint32_t idBound);

/// Returns the word-count-prefixed opcode for an SPIR-V instruction.
uint32_t getPrefixedOpcode(uint32_t wordCount, spirv::Opcode opcode);

/// Encodes an SPIR-V `literal` string into the given `binary` vector.
LogicalResult encodeStringLiteralInto(SmallVectorImpl<uint32_t> &binary,
                                      StringRef literal);
} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRV_BINARY_UTILS_H_
