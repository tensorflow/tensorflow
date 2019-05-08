//===- Diagnostics.cpp - MLIR Diagnostics ---------------------------------===//
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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// DiagnosticArgument
//===----------------------------------------------------------------------===//

// Construct from a Type.
DiagnosticArgument::DiagnosticArgument(Type val)
    : kind(DiagnosticArgumentKind::Type),
      opaqueVal(reinterpret_cast<intptr_t>(val.getAsOpaquePointer())) {}

/// Returns this argument as a Type.
Type DiagnosticArgument::getAsType() const {
  assert(getKind() == DiagnosticArgumentKind::Type);
  return Type::getFromOpaquePointer(reinterpret_cast<const void *>(opaqueVal));
}

/// Outputs this argument to a stream.
void DiagnosticArgument::print(raw_ostream &os) const {
  switch (kind) {
  case DiagnosticArgumentKind::Double:
    os << getAsDouble();
    break;
  case DiagnosticArgumentKind::Integer:
    os << getAsInteger();
    break;
  case DiagnosticArgumentKind::String:
    os << getAsString();
    break;
  case DiagnosticArgumentKind::Type:
    os << getAsType();
    break;
  case DiagnosticArgumentKind::Unsigned:
    os << getAsUnsigned();
    break;
  }
}

//===----------------------------------------------------------------------===//
// Diagnostic
//===----------------------------------------------------------------------===//

/// Stream in an Identifier.
Diagnostic &Diagnostic::operator<<(Identifier val) {
  // An identifier is stored in the context, so we don't need to worry about the
  // lifetime of its data.
  arguments.push_back(DiagnosticArgument(val.strref()));
  return *this;
}

/// Outputs this diagnostic to a stream.
void Diagnostic::print(raw_ostream &os) const {
  for (auto &arg : getArguments())
    arg.print(os);
}

/// Convert the diagnostic to a string.
std::string Diagnostic::str() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  print(os);
  return os.str();
}

/// Attaches a note to this diagnostic. A new location may be optionally
/// provided, if not, then the location defaults to the one specified for this
/// diagnostic. Notes may not be attached to other notes.
Diagnostic &Diagnostic::attachNote(llvm::Optional<Location> noteLoc) {
  // We don't allow attaching notes to notes.
  assert(severity != DiagnosticSeverity::Note &&
         "cannot attach a note to a note");

  // If a location wasn't provided then reuse our location.
  if (!noteLoc)
    noteLoc = loc;

  /// Append and return a new note.
  notes.push_back(
      llvm::make_unique<Diagnostic>(*noteLoc, DiagnosticSeverity::Note));
  return *notes.back();
}

//===----------------------------------------------------------------------===//
// InFlightDiagnostic
//===----------------------------------------------------------------------===//

/// Allow an inflight diagnostic to be converted to 'failure', otherwise
/// 'success' if this is an empty diagnostic.
InFlightDiagnostic::operator LogicalResult() const {
  return failure(isActive());
}

/// Reports the diagnostic to the engine.
void InFlightDiagnostic::report() {
  // If this diagnostic is still inflight and it hasn't been abandoned, then
  // report it.
  if (isInFlight()) {
    owner->emit(*impl);
    owner = nullptr;
  }
  impl.reset();
}

/// Abandons this diagnostic.
void InFlightDiagnostic::abandon() { owner = nullptr; }

//===----------------------------------------------------------------------===//
// DiagnosticEngineImpl
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
struct DiagnosticEngineImpl {
  /// Emit a diagnostic using the registered issue handle if present, or with
  /// the default behavior if not.
  void emit(Location loc, StringRef msg, DiagnosticSeverity severity);

  /// A mutex to ensure that diagnostics emission is thread-safe.
  llvm::sys::SmartMutex<true> mutex;

  /// This is the handler to use to report diagnostics, or null if not
  /// registered.
  DiagnosticEngine::HandlerTy handler;
};
} // namespace detail
} // namespace mlir

/// Emit a diagnostic using the registered issue handle if present, or with
/// the default behavior if not.
void DiagnosticEngineImpl::emit(Location loc, StringRef msg,
                                DiagnosticSeverity severity) {
  // If we had a handler registered, emit the diagnostic using it.
  if (handler) {
    // TODO(b/131756158) FusedLoc should be handled by the diagnostic handler
    // instead of here.
    // Check to see if we are emitting a diagnostic on a fused location.
    if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      auto fusedLocs = fusedLoc->getLocations();

      // Emit the original diagnostic with the first location in the fused list.
      emit(fusedLocs.front(), msg, severity);

      // Emit the rest of the locations as notes.
      for (Location subLoc : fusedLocs.drop_front())
        emit(subLoc, "fused from here", DiagnosticSeverity::Note);
      return;
    }

    return handler(loc, msg, severity);
  }

  // Otherwise, if this is an error we emit it to stderr.
  if (severity != DiagnosticSeverity::Error)
    return;

  auto &os = llvm::errs();
  if (!loc.isa<UnknownLoc>())
    os << loc << ": ";
  os << "error: ";

  // The default behavior for errors is to emit them to stderr.
  os << msg << '\n';
  os.flush();
}

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

DiagnosticEngine::DiagnosticEngine() : impl(new DiagnosticEngineImpl()) {}
DiagnosticEngine::~DiagnosticEngine() {}

/// Set the diagnostic handler for this engine.  The handler is passed
/// location information if present (nullptr if not) along with a message and
/// a severity that indicates whether this is an error, warning, etc. Note
/// that this replaces any existing handler.
void DiagnosticEngine::setHandler(const HandlerTy &handler) {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  impl->handler = handler;
}

/// Return the current diagnostic handler, or null if none is present.
auto DiagnosticEngine::getHandler() -> HandlerTy {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  return impl->handler;
}

/// Emit a diagnostic using the registered issue handler if present, or with
/// the default behavior if not.
void DiagnosticEngine::emit(const Diagnostic &diag) {
  assert(diag.getSeverity() != DiagnosticSeverity::Note &&
         "notes should not be emitted directly");
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  impl->emit(diag.getLocation(), diag.str(), diag.getSeverity());

  // Emit any notes that were attached to this diagnostic.
  for (auto &note : diag.getNotes())
    impl->emit(note.getLocation(), note.str(), note.getSeverity());
}
