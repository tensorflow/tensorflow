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
#include <vector>

namespace mlir {
class AbstractOperation;
class MLIRContextImpl;
class Location;
class Dialect;

namespace detail {
class TypeUniquer;
}

/// MLIRContext is the top-level object for a collection of MLIR modules.  It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
class MLIRContext {
public:
  explicit MLIRContext();
  ~MLIRContext();

  /// Return information about all registered IR dialects.
  std::vector<Dialect *> getRegisteredDialects() const;

  /// Get a registered IR dialect with the given namespace. If an exact match is
  /// not found, then return nullptr.
  Dialect *getRegisteredDialect(StringRef name) const;

  /// Return information about all registered operations.  This isn't very
  /// efficient: typically you should ask the operations about their properties
  /// directly.
  std::vector<AbstractOperation *> getRegisteredOperations() const;

  /// This is the interpretation of a diagnostic that is emitted to the
  /// diagnostic handler below.
  enum class DiagnosticKind { Note, Warning, Error };

  // Diagnostic handler registration and use.  MLIR supports the ability for the
  // IR to carry arbitrary metadata about operation location information.  If an
  // problem is detected by the compiler, it can invoke the emitError /
  // emitWarning / emitNote method on an Instruction and have it get reported
  // through this interface.
  //
  // Tools using MLIR are encouraged to register error handlers and define a
  // schema for their location information.  If they don't, then warnings and
  // notes will be dropped and errors will terminate the process with exit(1).

  using DiagnosticHandlerTy = std::function<void(
      Location location, StringRef message, DiagnosticKind kind)>;

  /// Register a diagnostic handler with this LLVM context.  The handler is
  /// passed location information if present (nullptr if not) along with a
  /// message and a boolean that indicates whether this is an error or warning.
  void registerDiagnosticHandler(const DiagnosticHandlerTy &handler);

  /// Return the current diagnostic handler, or null if none is present.
  DiagnosticHandlerTy getDiagnosticHandler() const;

  /// Emit a diagnostic using the registered issue handle if present, or with
  /// the default behavior if not.  The MLIR compiler should not generally
  /// interact with this, it should use methods on Instruction instead.
  void emitDiagnostic(Location location, const Twine &message,
                      DiagnosticKind kind) const;

  /// Emit an error message using the registered issue handle if present, or to
  /// the standard error stream otherwise and return true.
  bool emitError(Location location, const Twine &message) const;

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() const { return *impl.get(); }

  /// Get the type uniquer for this context.
  detail::TypeUniquer &getTypeUniquer() const;

private:
  const std::unique_ptr<MLIRContextImpl> impl;

  MLIRContext(const MLIRContext &) = delete;
  void operator=(const MLIRContext &) = delete;
};
} // end namespace mlir

#endif // MLIR_IR_MLIRCONTEXT_H
