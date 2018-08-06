//===- MLIRContext.h - MLIR Global Context Class ----------------*- C++ -*-===//
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

#ifndef MLIR_IR_MLIRCONTEXT_H
#define MLIR_IR_MLIRCONTEXT_H

#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>

namespace mlir {
class MLIRContextImpl;
class Attribute;

/// MLIRContext is the top-level object for a collection of MLIR modules.  It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
class MLIRContext {
  const std::unique_ptr<MLIRContextImpl> impl;
  MLIRContext(const MLIRContext&) = delete;
  void operator=(const MLIRContext&) = delete;
public:
  explicit MLIRContext();
  ~MLIRContext();

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() const { return *impl.get(); }

  // This is the interpretation of a diagnostic that is emitted to the
  // diagnostic handler below.
  enum class DiagnosticKind { Note, Warning, Error };

  // Diagnostic handler registration and use.  MLIR supports the ability for the
  // IR to carry arbitrary metadata about operation location information.  If an
  // problem is detected by the compiler, it can invoke the emitError /
  // emitWarning / emitNote method on an Operation and have it get reported
  // through this interface.
  //
  // Tools using MLIR are encouraged to register error handlers and define a
  // schema for their location information.  If they don't, then warnings and
  // notes will be dropped and errors will terminate the process with exit(1).

  using DiagnosticHandlerTy = std::function<void(
      Attribute *location, StringRef message, DiagnosticKind kind)>;

  /// Register a diagnostic handler with this LLVM context.  The handler is
  /// passed location information if present (nullptr if not) along with a
  /// message and a boolean that indicates whether this is an error or warning.
  void registerDiagnosticHandler(const DiagnosticHandlerTy &handler);

  /// This emits an diagnostic using the registered issue handle if present, or
  /// with the default behavior if not.  The MLIR compiler should not generally
  /// interact with this, it should use methods on Operation instead.
  void emitDiagnostic(Attribute *location, const Twine &message,
                      DiagnosticKind kind) const;
};
} // end namespace mlir

#endif  // MLIR_IR_MLIRCONTEXT_H
