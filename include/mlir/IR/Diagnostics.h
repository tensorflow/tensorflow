//===- Diagnostics.h - MLIR Diagnostics -------------------------*- C++ -*-===//
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
// This file defines utilities for emitting diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIAGNOSTICS_H
#define MLIR_IR_DIAGNOSTICS_H

#include "mlir/Support/LLVM.h"
#include <functional>

namespace mlir {
class Location;

namespace detail {
struct DiagnosticEngineImpl;
} // end namespace detail

/// Defines the different supported severity of a diagnostic.
enum class DiagnosticSeverity {
  Note,
  Warning,
  Error,
  Remark,
};

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

/// This class is the main interface for diagnostics. The DiagnosticEngine
/// manages the registration of diagnostic handlers as well as the core API for
/// diagnostic emission. This class should not be constructed directly, but
/// instead interfaced with via an MLIRContext instance.
class DiagnosticEngine {
public:
  ~DiagnosticEngine();

  // Diagnostic handler registration and use.  MLIR supports the ability for the
  // IR to carry arbitrary metadata about operation location information.  If a
  // problem is detected by the compiler, it can invoke the emitError /
  // emitWarning / emitNote method on an Operation and have it get reported
  // through this interface.
  //
  // Tools using MLIR are encouraged to register error handlers and define a
  // schema for their location information.  If they don't, then warnings and
  // notes will be dropped and errors will terminate the process with exit(1).

  using HandlerTy =
      std::function<void(Location, StringRef, DiagnosticSeverity)>;

  /// Set the diagnostic handler for this engine.  The handler is passed
  /// location information if present (nullptr if not) along with a message and
  /// a severity that indicates whether this is an error, warning, etc. Note
  /// that this replaces any existing handler.
  void setHandler(const HandlerTy &handler);

  /// Return the current diagnostic handler, or null if none is present.
  HandlerTy getHandler();

  /// Emit a diagnostic using the registered issue handle if present, or with
  /// the default behavior if not.  The MLIR compiler should not generally
  /// interact with this, it should use methods on Operation instead.
  void emit(Location loc, const Twine &msg, DiagnosticSeverity severity);

private:
  friend class MLIRContextImpl;
  DiagnosticEngine();

  /// The internal implementation of the DiagnosticEngine.
  std::unique_ptr<detail::DiagnosticEngineImpl> impl;
};
} // namespace mlir

#endif
