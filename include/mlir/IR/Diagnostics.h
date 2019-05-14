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

#include "mlir/IR/Location.h"
#include "mlir/Support/STLExtras.h"
#include <functional>

namespace llvm {
class MemoryBuffer;
class SMLoc;
class SourceMgr;
} // end namespace llvm

namespace mlir {
class DiagnosticEngine;
class Identifier;
struct LogicalResult;
class MLIRContext;
class Type;

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
// DiagnosticArgument
//===----------------------------------------------------------------------===//

/// A variant type that holds a single argument for a diagnostic.
class DiagnosticArgument {
public:
  /// Enum that represents the different kinds of diagnostic arguments
  /// supported.
  enum class DiagnosticArgumentKind {
    Attribute,
    Double,
    Integer,
    String,
    Type,
    Unsigned,
  };

  /// Outputs this argument to a stream.
  void print(raw_ostream &os) const;

  /// Returns the kind of this argument.
  DiagnosticArgumentKind getKind() const { return kind; }

  /// Returns this argument as an Attribute.
  Attribute getAsAttribute() const;

  /// Returns this argument as a double.
  double getAsDouble() const {
    assert(getKind() == DiagnosticArgumentKind::Double);
    return doubleVal;
  }

  /// Returns this argument as a signed integer.
  int64_t getAsInteger() const {
    assert(getKind() == DiagnosticArgumentKind::Integer);
    return static_cast<int64_t>(opaqueVal);
  }

  /// Returns this argument as a string.
  StringRef getAsString() const {
    assert(getKind() == DiagnosticArgumentKind::String);
    return stringVal;
  }

  /// Returns this argument as a Type.
  Type getAsType() const;

  /// Returns this argument as an unsigned integer.
  uint64_t getAsUnsigned() const {
    assert(getKind() == DiagnosticArgumentKind::Unsigned);
    return static_cast<uint64_t>(opaqueVal);
  }

private:
  friend class Diagnostic;

  // Construct from an Attribute.
  explicit DiagnosticArgument(Attribute attr);

  // Construct from a floating point number.
  explicit DiagnosticArgument(double val)
      : kind(DiagnosticArgumentKind::Double), doubleVal(val) {}
  explicit DiagnosticArgument(float val) : DiagnosticArgument(double(val)) {}

  // Construct from a signed integer.
  template <typename T>
  explicit DiagnosticArgument(
      T val, typename std::enable_if<std::is_signed<T>::value &&
                                     std::numeric_limits<T>::is_integer &&
                                     sizeof(T) <= sizeof(int64_t)>::type * = 0)
      : kind(DiagnosticArgumentKind::Integer), opaqueVal(int64_t(val)) {}

  // Construct from an unsigned integer.
  template <typename T>
  explicit DiagnosticArgument(
      T val, typename std::enable_if<std::is_unsigned<T>::value &&
                                     std::numeric_limits<T>::is_integer &&
                                     sizeof(T) <= sizeof(uint64_t)>::type * = 0)
      : kind(DiagnosticArgumentKind::Unsigned), opaqueVal(uint64_t(val)) {}

  // Construct from a string reference.
  explicit DiagnosticArgument(StringRef val)
      : kind(DiagnosticArgumentKind::String), stringVal(val) {}

  // Construct from a Type.
  explicit DiagnosticArgument(Type val);

  /// The kind of this argument.
  DiagnosticArgumentKind kind;

  /// The value of this argument.
  union {
    double doubleVal;
    intptr_t opaqueVal;
    StringRef stringVal;
  };
};

inline raw_ostream &operator<<(raw_ostream &os, const DiagnosticArgument &arg) {
  arg.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

/// This class contains all of the information necessary to report a diagnostic
/// to the DiagnosticEngine. It should generally not be constructed directly,
/// and instead used transitively via InFlightDiagnostic.
class Diagnostic {
  using NoteVector = std::vector<std::unique_ptr<Diagnostic>>;

  /// This class implements a wrapper iterator around NoteVector::iterator to
  /// implicitly dereference the unique_ptr.
  template <typename IteratorTy, typename NotePtrTy = decltype(*IteratorTy()),
            typename ResultTy = decltype(**IteratorTy())>
  class NoteIteratorImpl
      : public llvm::mapped_iterator<IteratorTy, ResultTy (*)(NotePtrTy)> {
    static ResultTy &unwrap(NotePtrTy note) { return *note; }

  public:
    NoteIteratorImpl(IteratorTy it)
        : llvm::mapped_iterator<IteratorTy, ResultTy (*)(NotePtrTy)>(it,
                                                                     &unwrap) {}
  };

public:
  Diagnostic(Location loc, DiagnosticSeverity severity)
      : loc(loc), severity(severity) {}
  Diagnostic(Diagnostic &&) = default;
  Diagnostic &operator=(Diagnostic &&) = default;

  /// Returns the severity of this diagnostic.
  DiagnosticSeverity getSeverity() const { return severity; }

  /// Returns the source location for this diagnostic.
  Location getLocation() const { return loc; }

  /// Returns the current list of diagnostic arguments.
  MutableArrayRef<DiagnosticArgument> getArguments() { return arguments; }
  ArrayRef<DiagnosticArgument> getArguments() const { return arguments; }

  /// Stream operator for inserting new diagnostic arguments.
  template <typename Arg>
  typename std::enable_if<!std::is_convertible<Arg, StringRef>::value,
                          Diagnostic &>::type
  operator<<(Arg &&val) {
    arguments.push_back(DiagnosticArgument(std::forward<Arg>(val)));
    return *this;
  }

  /// Stream in a string literal.
  Diagnostic &operator<<(const char *val) {
    arguments.push_back(DiagnosticArgument(val));
    return *this;
  }

  /// Stream in a Twine argument.
  Diagnostic &operator<<(char val);
  Diagnostic &operator<<(const Twine &val);
  Diagnostic &operator<<(Twine &&val);

  /// Stream in an Identifier.
  Diagnostic &operator<<(Identifier val);

  /// Stream in a range.
  template <typename T> Diagnostic &operator<<(llvm::iterator_range<T> range) {
    return appendRange(range);
  }
  template <typename T> Diagnostic &operator<<(llvm::ArrayRef<T> range) {
    return appendRange(range);
  }

  /// Append a range to the diagnostic. The default delimiter between elements
  /// is ','.
  template <typename T, template <typename> class Container>
  Diagnostic &appendRange(const Container<T> &c, const char *delim = ", ") {
    interleave(
        c, [&](T a) { *this << a; }, [&]() { *this << delim; });
    return *this;
  }

  /// Append arguments to the diagnostic.
  template <typename Arg1, typename Arg2, typename... Args>
  Diagnostic &append(Arg1 &&arg1, Arg2 &&arg2, Args &&... args) {
    append(std::forward<Arg1>(arg1));
    return append(std::forward<Arg2>(arg2), std::forward<Args>(args)...);
  }
  /// Append one argument to the diagnostic.
  template <typename Arg> Diagnostic &append(Arg &&arg) {
    *this << std::forward<Arg>(arg);
    return *this;
  }

  /// Outputs this diagnostic to a stream.
  void print(raw_ostream &os) const;

  /// Converts the diagnostic to a string.
  std::string str() const;

  /// Attaches a note to this diagnostic. A new location may be optionally
  /// provided, if not, then the location defaults to the one specified for this
  /// diagnostic. Notes may not be attached to other notes.
  Diagnostic &attachNote(llvm::Optional<Location> noteLoc = llvm::None);

  using note_iterator = NoteIteratorImpl<NoteVector::iterator>;
  using const_note_iterator = NoteIteratorImpl<NoteVector::const_iterator>;

  /// Returns the notes held by this diagnostic.
  llvm::iterator_range<note_iterator> getNotes() {
    return {notes.begin(), notes.end()};
  }
  llvm::iterator_range<const_note_iterator> getNotes() const {
    return {notes.begin(), notes.end()};
  }

private:
  Diagnostic(const Diagnostic &rhs) = delete;
  Diagnostic &operator=(const Diagnostic &rhs) = delete;

  /// The source location.
  Location loc;

  /// The severity of this diagnostic.
  DiagnosticSeverity severity;

  /// The current list of arguments.
  SmallVector<DiagnosticArgument, 4> arguments;

  /// A list of string values used as arguments. This is used to guarantee the
  /// liveness of non-constant strings used in diagnostics.
  std::vector<std::unique_ptr<char[]>> strings;

  /// A list of attached notes.
  NoteVector notes;
};

inline raw_ostream &operator<<(raw_ostream &os, const Diagnostic &diag) {
  diag.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

/// This class represents a diagnostic that is inflight and set to be reported.
/// This allows for last minute modifications of the diagnostic before it is
/// emitted by a DiagnosticEngine.
class InFlightDiagnostic {
public:
  InFlightDiagnostic() = default;
  InFlightDiagnostic(InFlightDiagnostic &&rhs)
      : owner(rhs.owner), impl(std::move(rhs.impl)) {
    // Reset the rhs diagnostic.
    rhs.impl.reset();
    rhs.abandon();
  }
  ~InFlightDiagnostic() {
    if (isInFlight())
      report();
  }

  /// Stream operator for new diagnostic arguments.
  template <typename Arg> InFlightDiagnostic &operator<<(Arg &&arg) & {
    return append(std::forward<Arg>(arg));
  }
  template <typename Arg> InFlightDiagnostic &&operator<<(Arg &&arg) && {
    return std::move(append(std::forward<Arg>(arg)));
  }

  /// Append arguments to the diagnostic.
  template <typename... Args> InFlightDiagnostic &append(Args &&... args) & {
    assert(isActive() && "diagnostic not active");
    if (isInFlight())
      impl->append(std::forward<Args>(args)...);
    return *this;
  }
  template <typename... Args> InFlightDiagnostic &&append(Args &&... args) && {
    return std::move(append(std::forward<Args>(args)...));
  }

  /// Attaches a note to this diagnostic.
  Diagnostic &attachNote(llvm::Optional<Location> noteLoc = llvm::None) {
    assert(isActive() && "diagnostic not active");
    return impl->attachNote(noteLoc);
  }

  /// Reports the diagnostic to the engine.
  void report();

  /// Abandons this diagnostic so that it will no longer be reported.
  void abandon();

  /// Allow an inflight diagnostic to be converted to 'failure', otherwise
  /// 'success' if this is an empty diagnostic.
  operator LogicalResult() const;

private:
  InFlightDiagnostic &operator=(const InFlightDiagnostic &) = delete;
  InFlightDiagnostic &operator=(InFlightDiagnostic &&) = delete;
  InFlightDiagnostic(DiagnosticEngine *owner, Diagnostic &&rhs)
      : owner(owner), impl(std::move(rhs)) {}

  /// Returns if the diagnostic is still active, i.e. it has a live diagnostic.
  bool isActive() const { return impl.hasValue(); }

  /// Returns if the diagnostic is still in flight to be reported.
  bool isInFlight() const { return owner; }

  // Allow access to the constructor.
  friend DiagnosticEngine;

  /// The engine that this diagnostic is to report to.
  DiagnosticEngine *owner;

  /// The raw diagnostic that is inflight to be reported.
  llvm::Optional<Diagnostic> impl;
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
  // emitWarning / emitRemark method on an Operation and have it get reported
  // through this interface.
  //
  // Tools using MLIR are encouraged to register error handlers and define a
  // schema for their location information.  If they don't, then warnings and
  // notes will be dropped and errors will be emitted to errs.

  using HandlerTy = std::function<void(Diagnostic)>;

  /// Set the diagnostic handler for this engine. Note that this replaces any
  /// existing handler.
  void setHandler(const HandlerTy &handler);

  /// Return the current diagnostic handler, or null if none is present.
  HandlerTy getHandler();

  /// Create a new inflight diagnostic with the given location and severity.
  InFlightDiagnostic emit(Location loc, DiagnosticSeverity severity) {
    assert(severity != DiagnosticSeverity::Note &&
           "notes should not be emitted directly");
    return InFlightDiagnostic(this, Diagnostic(loc, severity));
  }

  /// Emit a diagnostic using the registered issue handler if present, or with
  /// the default behavior if not.
  void emit(Diagnostic diag);

private:
  friend class MLIRContextImpl;
  DiagnosticEngine();

  /// The internal implementation of the DiagnosticEngine.
  std::unique_ptr<detail::DiagnosticEngineImpl> impl;
};

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticHandler
//===----------------------------------------------------------------------===//

namespace detail {
struct SourceMgrDiagnosticHandlerImpl;
} // end namespace detail

/// This class is a utility diagnostic handler for use with llvm::SourceMgr.
class SourceMgrDiagnosticHandler {
public:
  SourceMgrDiagnosticHandler(llvm::SourceMgr &mgr, MLIRContext *ctx);
  ~SourceMgrDiagnosticHandler();

  /// Emit the given diagnostic information with the held source manager.
  void emitDiagnostic(Location loc, Twine message, DiagnosticSeverity kind);

protected:
  /// Emit the given diagnostic with the held source manager.
  void emitDiagnostic(Diagnostic &diag);

  /// Get a memory buffer for the given file, or nullptr if no file is
  /// available.
  const llvm::MemoryBuffer *getBufferForFile(StringRef filename);

  /// The source manager that we are wrapping.
  llvm::SourceMgr &mgr;

private:
  /// Convert a location into the given memory buffer into an SMLoc.
  llvm::SMLoc convertLocToSMLoc(FileLineColLoc loc);

  /// The maximum depth that a call stack will be printed.
  /// TODO(riverriddle) This should be a tunable flag.
  unsigned callStackLimit = 10;

  std::unique_ptr<detail::SourceMgrDiagnosticHandlerImpl> impl;
};

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticVerifierHandler
//===----------------------------------------------------------------------===//

namespace detail {
struct SourceMgrDiagnosticVerifierHandlerImpl;
} // end namespace detail

/// This class is a utility diagnostic handler for use with llvm::SourceMgr that
/// verifies that emitted diagnostics match 'expected-*' lines on the
/// corresponding line of the source file.
class SourceMgrDiagnosticVerifierHandler : public SourceMgrDiagnosticHandler {
public:
  SourceMgrDiagnosticVerifierHandler(llvm::SourceMgr &srcMgr, MLIRContext *ctx);
  ~SourceMgrDiagnosticVerifierHandler();

  /// Returns the status of the handler and verifies that all expected
  /// diagnostics were emitted. This return success if all diagnostics were
  /// verified correctly, failure otherwise.
  LogicalResult verify();

private:
  /// Process a single diagnostic.
  void process(Diagnostic &diag);

  /// Process a FileLineColLoc diagnostic.
  void process(FileLineColLoc loc, StringRef msg, DiagnosticSeverity kind);

  std::unique_ptr<detail::SourceMgrDiagnosticVerifierHandlerImpl> impl;
};

} // namespace mlir

#endif
