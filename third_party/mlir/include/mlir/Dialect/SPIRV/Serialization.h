//===- Serialization.h - MLIR SPIR-V (De)serialization ----------*- C++ -*-===//
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
// This file declares the entry points for serialize and deserialize SPIR-V
// binary modules.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SERIALIZATION_H_
#define MLIR_DIALECT_SPIRV_SERIALIZATION_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class MLIRContext;

namespace spirv {
class ModuleOp;

/// Serializes the given SPIR-V `module` and writes to `binary`. On failure,
/// reports errors to the error handler registered with the MLIR context for
/// `module`.
LogicalResult serialize(ModuleOp module, SmallVectorImpl<uint32_t> &binary);

/// Deserializes the given SPIR-V `binary` module and creates a MLIR ModuleOp
/// in the given `context`. Returns the ModuleOp on success; otherwise, reports
/// errors to the error handler registered with `context` and returns
/// llvm::None.
Optional<ModuleOp> deserialize(ArrayRef<uint32_t> binary, MLIRContext *context);

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SERIALIZATION_H_
