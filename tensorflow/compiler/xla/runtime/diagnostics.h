/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_DIAGNOSTICS_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_DIAGNOSTICS_H_

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace runtime {

// Forward declare.
class DiagnosticEngine;

// XLA runtime diagnostic engine enables XLA runtime custom calls and compiled
// programs to pass diagnostic information (e.g. detailed run time error
// information attached to the absl::Status) to the caller via the side channel,
// because the API (and ABI) of the compiled executable is very simple, and
// doesn't allow to pass complex C++ tyopes.
//
// XLA runtime diagnostics borrows a lot of ideas from the MLIR compile time
// diagnostics (which is largely based on the Swift compiler diagnostics),
// however in contrast to MLIR compilation pipelines we need to emit diagnostics
// for the run time events (vs compile time) and correlate them back to the
// location in the input module.
//
// See MLIR Diagnostics documentation: https://mlir.llvm.org/docs/Diagnostics.

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Add location tracking to diagnostic, so that we can
// correlate emitted diagnostic to the location in the input module, and from
// there rely on the MLIR location to correlate events back to the user program
// (e.g. original JAX program written in Python).
class Diagnostic {
 public:
  explicit Diagnostic(absl::Status status) : status_(std::move(status)) {}

  Diagnostic(Diagnostic &&) = default;
  Diagnostic &operator=(Diagnostic &&) = default;

  absl::Status status() const { return status_; }

 private:
  Diagnostic(const Diagnostic &rhs) = delete;
  Diagnostic &operator=(const Diagnostic &rhs) = delete;

  absl::Status status_;
};

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

// RAII wrapper around constructed, but but not yet emitted diagnostic. In
// flight diagnostic gives an opportunity to build a diagnostic before reporting
// it to the engine, similar to the builder pattern.
class InFlightDiagnostic {
 public:
  InFlightDiagnostic(InFlightDiagnostic &&other)
      : engine_(other.engine_), diagnostic_(std::move(other.diagnostic_)) {
    other.diagnostic_.reset();
    other.Abandon();
  }

  ~InFlightDiagnostic() {
    if (IsInFlight()) Report();
  }

  void Report();
  void Abandon();

  // Allow a diagnostic to be converted to 'failure'.
  //
  // Example:
  //
  //   LogicalResult call(DiagnosticEngine diag, ...) {
  //     if (<check failed>) return diag.EmitError(InternalError("oops"));
  //     ...
  //   }
  //
  operator LogicalResult() const { return failure(); }  // NOLINT

 private:
  friend class DiagnosticEngine;

  InFlightDiagnostic(const DiagnosticEngine *engine, Diagnostic diagnostic)
      : engine_(engine), diagnostic_(std::move(diagnostic)) {}

  InFlightDiagnostic &operator=(const InFlightDiagnostic &) = delete;
  InFlightDiagnostic &operator=(InFlightDiagnostic &&) = delete;

  bool IsActive() const { return diagnostic_.has_value(); }
  bool IsInFlight() const { return engine_ != nullptr; }

  // Diagnostic engine that will report this diagnostic once its ready.
  const DiagnosticEngine *engine_ = nullptr;
  std::optional<Diagnostic> diagnostic_;
};

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

// Diagnostic engine is responsible for passing diagnostics to the user.
//
// XLA runtime users must set up diagnostic engine to report errors back to the
// caller, e.g. the handler can collect all of the emitted diagnostics into the
// string message, and pass it to the caller as the async error.
//
// Unhandled error diagnostics will be logged with the warning level.
class DiagnosticEngine {
 public:
  // Diagnostic handler must return success if it consumed the diagnostic, and
  // failure if the engine should pass it to the next registered handler.
  using HandlerTy = std::function<LogicalResult(Diagnostic &)>;

  // Returns the default instance of the diagnostic engine.
  static const DiagnosticEngine *DefaultDiagnosticEngine();

  InFlightDiagnostic EmitError(absl::Status status) const {
    return InFlightDiagnostic(this, Diagnostic(std::move(status)));
  }

  void AddHandler(HandlerTy handler) {
    handlers_.push_back(std::move(handler));
  }

  void Emit(Diagnostic diagnostic) const {
    for (auto &handler : llvm::reverse(handlers_)) {
      if (succeeded(handler(diagnostic))) return;
    }

    // Log unhandled errors to the warning log.
    LOG(WARNING) << "XLA runtime error: " << diagnostic.status();
  }

 private:
  llvm::SmallVector<HandlerTy> handlers_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_DIAGNOSTICS_H_
