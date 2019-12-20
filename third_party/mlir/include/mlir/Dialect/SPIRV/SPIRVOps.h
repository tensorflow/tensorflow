//===- SPIRVOps.h - MLIR SPIR-V operations ----------------------*- C++ -*-===//
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
// This file declares the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVOPS_H_
#define MLIR_DIALECT_SPIRV_SPIRVOPS_H_

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Function.h"

namespace mlir {
class OpBuilder;

namespace spirv {

#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVOps.h.inc"

/// Following methods are auto-generated.
///
/// Get the name used in the Op to refer to an enum value of the given
/// `EnumClass`.
/// template <typename EnumClass> StringRef attributeName();
///
/// Get the function that can be used to symbolize an enum value.
/// template <typename EnumClass>
/// Optional<EnumClass> (*)(StringRef) symbolizeEnum();
#include "mlir/Dialect/SPIRV/SPIRVOpUtils.inc"

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVOPS_H_
