//===- Serialization.h - MLIR SPIR-V (De)serialization ----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
