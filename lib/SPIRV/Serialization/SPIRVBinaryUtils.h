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
// This file defines common utilities for SPIR-V binary module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SPIRV_SERIALIZATION_SPIRV_BINARY_UTILS_H_
#define MLIR_SPIRV_SERIALIZATION_SPIRV_BINARY_UTILS_H_

#include <cstdint>

namespace mlir {
namespace spirv {

/// SPIR-V binary header word count
constexpr unsigned kHeaderWordCount = 5;

/// SPIR-V magic number
constexpr uint32_t kMagicNumber = 0x07230203;

/// Opcode for SPIR-V OpMemoryModel
constexpr uint32_t kOpMemoryModelOpcode = 14;

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_SPIRV_SERIALIZATION_SPIRV_BINARY_UTILS_H_
