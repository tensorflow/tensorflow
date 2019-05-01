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
#include "mlir/IR/Location.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

namespace mlir {
namespace detail {
struct DiagnosticEngineImpl {
  /// A mutex to ensure that diagnostics emission is thread-safe.
  llvm::sys::SmartMutex<true> mutex;

  /// This is the handler to use to report diagnostics, or null if not
  /// registered.
  DiagnosticEngine::HandlerTy handler;
};
} // namespace detail
} // namespace mlir

//===----------------------------------------------------------------------===//
// DiagnosticEngine
//===----------------------------------------------------------------------===//

DiagnosticEngine::DiagnosticEngine() : impl(new DiagnosticEngineImpl()) {}
DiagnosticEngine::~DiagnosticEngine() {}

/// Register a diagnostic handler with this engine.  The handler is
/// passed location information if present (nullptr if not) along with a
/// message and a severity that indicates whether this is an error, warning,
/// etc.
void DiagnosticEngine::setHandler(const HandlerTy &handler) {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  impl->handler = handler;
}

/// Return the current diagnostic handler, or null if none is present.
auto DiagnosticEngine::getHandler() -> HandlerTy {
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);
  return impl->handler;
}

/// Emit a diagnostic using the registered issue handle if present, or with
/// the default behavior if not.  The MLIR compiler should not generally
/// interact with this, it should use methods on Operation instead.
void DiagnosticEngine::emit(Location loc, const Twine &msg,
                            DiagnosticSeverity severity) {
  /// Lock access to the diagnostic engine.
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);

  // If we had a handler registered, emit the diagnostic using it.
  if (impl->handler) {
    // TODO(b/131756158) FusedLoc should be handled by the diagnostic handler
    // instead of here.
    // Check to see if we are emitting a diagnostic on a fused
    // location.
    if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      auto fusedLocs = fusedLoc->getLocations();

      // Emit the original diagnostic with the first location in the fused list.
      emit(fusedLocs.front(), msg, severity);

      // Emit the rest of the locations as notes.
      for (Location subLoc : fusedLocs.drop_front())
        emit(subLoc, "fused from here", DiagnosticSeverity::Note);
      return;
    }

    return impl->handler(loc, msg.str(), severity);
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
