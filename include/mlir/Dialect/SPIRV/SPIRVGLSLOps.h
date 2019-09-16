//===- SPIRVGLSLOps.h - MLIR SPIR-V extended ops for GLSL --------*- C++-*-===//
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
// This file declares the extended operations for GLSL in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVGLSLOPS_H_
#define MLIR_DIALECT_SPIRV_SPIRVGLSLOPS_H_

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/SPIRVGLSLOps.h.inc"

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVGLSLOPS_H_
