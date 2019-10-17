//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
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
// This file implements the IR Dialect for the Toy language.
// See g3doc/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "toy/ShapeInferenceInterface.h"

namespace mlir {
namespace toy {

/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "toy"; }
};

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

} // end namespace toy
} // end namespace mlir

#endif // MLIR_TUTORIAL_TOY_DIALECT_H_
