//===- Lowering.h - Lexer for the Toy language ----------------------------===//
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
// This file exposes the interface to the lowering for Toy. It is divided in
// two parts:  an *early lowering* that emits operations in the `Linalg`
// dialects for a subset of the Toy IR, and a *late lowering* that materializes
// buffers and converts all operations and type to the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXAMPLES_TOY_LOWERING_H_
#define MLIR_EXAMPLES_TOY_LOWERING_H_

#include <memory>

namespace mlir {
class Pass;
class DialectConversion;
} // namespace mlir

namespace toy {
/// Create a pass for lowering operations in the `Linalg` dialects, for a subset
/// of the Toy IR (matmul).
mlir::Pass *createEarlyLoweringPass();

/// Create a pass for the late lowering toward LLVM dialect.
mlir::Pass *createLateLoweringPass();

} // namespace toy

#endif // MLIR_EXAMPLES_TOY_LOWERING_H_
