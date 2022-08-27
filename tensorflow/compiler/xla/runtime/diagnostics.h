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

#ifndef XLA_RUNTIME_DIAGNOSTICS_H_
#define XLA_RUNTIME_DIAGNOSTICS_H_

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"

namespace xla {
namespace runtime {

// Forward declare.
class DiagnosticEngine;

// XLA runtime diagnostics borrows a lot of ideas from the MLIR compile time
// diagnostics (which is largely based on the Swift compiler diagnostics),
// however in contrast to MLIR compilation pipelines we need to emit diagnostics
// for the run time events (vs compile time) and correlate them back to the
// location in the input module.
//
// See MLIR Diagnostics documentation: https://mlir.llvm.org/docs/Diagnostics.
//
// TODO(ezhulenev): Add location tracking, so that we can correlate emitted
// diagnostics to the location in the input module, and from there rely on the
// MLIR location to correlate events back to the user program (e.g. original
// JAX program written in Python).
//
// TODO(ezhulenev): In contrast to MLIR we don't have notes. Add them if needed.

enum class DiagnosticSeverity { kWarning, kError, kRemark };

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

class Diagnostic {
 public:
  explicit Diagnostic(DiagnosticSeverity severity) : severity_(severity) {}

  Diagnostic(Diagnostic &&) = default;
  Diagnostic &operator=(Diagnostic &&) = default;

  // TODO(ezhulenev): Instead of relying on `<<` implementation pass diagnostic
  // arguments explicitly, similar to MLIR?

  template <typename Arg>
  Diagnostic &operator<<(Arg &&arg) {
    llvm::raw_string_ostream(message_) << std::forward<Arg>(arg);
    return *this;
  }

  template <typename Arg>
  Diagnostic &append(Arg &&arg) {
    *this << std::forward<Arg>(arg);
    return *this;
  }

  template <typename T>
  Diagnostic &appendRange(const T &c, const char *delim = ", ") {
    llvm::interleave(
        c, [this](const auto &a) { *this << a; }, [&]() { *this << delim; });
    return *this;
  }

  DiagnosticSeverity severity() const { return severity_; }

  std::string str() const { return message_; }

 private:
  Diagnostic(const Diagnostic &rhs) = delete;
  Diagnostic &operator=(const Diagnostic &rhs) = delete;

  DiagnosticSeverity severity_;
  std::string message_;
};

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

// In flight diagnostic gives an opportunity to build a diagnostic before
// reporting it to the engine, similar to the builder pattern.
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

  template <typename Arg>
  InFlightDiagnostic &operator<<(Arg &&arg) & {
    return append(std::forward<Arg>(arg));
  }
  template <typename Arg>
  InFlightDiagnostic &&operator<<(Arg &&arg) && {
    return std::move(append(std::forward<Arg>(arg)));
  }

  template <typename Arg>
  InFlightDiagnostic &append(Arg &&arg) & {
    assert(IsActive() && "diagnostic not active");
    if (IsInFlight()) diagnostic_->append(std::forward<Arg>(arg));
    return *this;
  }

  template <typename Arg>
  InFlightDiagnostic &&append(Arg &&arg) && {
    return std::move(append(std::forward<Arg>(arg)));
  }

  void Report();
  void Abandon();

  // Allow a diagnostic to be converted to 'failure'.
  //
  // Example:
  //
  //   LogicalResult call(DiagnosticEngine diag, ...) {
  //     if (<check failed>) return diag.EmitError() << "Oops";
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
// Unhandled error diagnostics will be dumped to the llvm::errs() stream.
class DiagnosticEngine {
 public:
  // Diagnostic handler must return success if it consumed the diagnostic, and
  // failure if the engine should pass it to the next registered handler.
  using HandlerTy = std::function<LogicalResult(Diagnostic &)>;

  // Returns the default instance of the diagnostic engine.
  static const DiagnosticEngine *DefaultDiagnosticEngine();

  InFlightDiagnostic Emit(DiagnosticSeverity severity) const {
    return InFlightDiagnostic(this, Diagnostic(severity));
  }

  InFlightDiagnostic EmitError() const {
    return Emit(DiagnosticSeverity::kError);
  }

  void AddHandler(HandlerTy handler) {
    handlers_.push_back(std::move(handler));
  }

  void Emit(Diagnostic diagnostic) const {
    for (auto &handler : llvm::reverse(handlers_)) {
      if (succeeded(handler(diagnostic))) return;
    }

    // Dump unhandled errors to llvm::errs() stream.
    if (diagnostic.severity() == DiagnosticSeverity::kError)
      llvm::errs() << "Error: " << diagnostic.str() << "\n";
  }

 private:
  llvm::SmallVector<HandlerTy> handlers_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_DIAGNOSTICS_H_
