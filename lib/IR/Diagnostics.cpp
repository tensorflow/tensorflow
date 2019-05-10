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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// DiagnosticArgument
//===----------------------------------------------------------------------===//

// Construct from an Attribute.
DiagnosticArgument::DiagnosticArgument(Attribute attr)
    : kind(DiagnosticArgumentKind::Attribute),
      opaqueVal(reinterpret_cast<intptr_t>(attr.getAsOpaquePointer())) {}

// Construct from a Type.
DiagnosticArgument::DiagnosticArgument(Type val)
    : kind(DiagnosticArgumentKind::Type),
      opaqueVal(reinterpret_cast<intptr_t>(val.getAsOpaquePointer())) {}

/// Returns this argument as an Attribute.
Attribute DiagnosticArgument::getAsAttribute() const {
  assert(getKind() == DiagnosticArgumentKind::Attribute);
  return Attribute::getFromOpaquePointer(
      reinterpret_cast<const void *>(opaqueVal));
}

/// Returns this argument as a Type.
Type DiagnosticArgument::getAsType() const {
  assert(getKind() == DiagnosticArgumentKind::Type);
  return Type::getFromOpaquePointer(reinterpret_cast<const void *>(opaqueVal));
}

/// Outputs this argument to a stream.
void DiagnosticArgument::print(raw_ostream &os) const {
  switch (kind) {
  case DiagnosticArgumentKind::Attribute:
    os << getAsAttribute();
    break;
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

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticHandler
//===----------------------------------------------------------------------===//

/// Return a processable FileLineColLoc from the given location.
static llvm::Optional<FileLineColLoc> getFileLineColLoc(Location loc) {
  // Process a FileLineColLoc.
  if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
    return fileLoc;

  // Process a CallSiteLoc.
  if (auto callLoc = loc.dyn_cast<CallSiteLoc>()) {
    // Check to see if the caller is a FileLineColLoc.
    if (auto fileLoc = callLoc->getCaller().dyn_cast<FileLineColLoc>())
      return fileLoc;

    // Otherwise, if the caller is another CallSiteLoc, check if its callee is a
    // FileLineColLoc.
    if (auto callerLoc = callLoc->getCaller().dyn_cast<CallSiteLoc>())
      return callerLoc->getCallee().dyn_cast<FileLineColLoc>();
  }
  return llvm::None;
}

/// Given a diagnostic kind, returns the LLVM DiagKind.
static llvm::SourceMgr::DiagKind getDiagKind(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return llvm::SourceMgr::DK_Note;
  case DiagnosticSeverity::Warning:
    return llvm::SourceMgr::DK_Warning;
  case DiagnosticSeverity::Error:
    return llvm::SourceMgr::DK_Error;
  case DiagnosticSeverity::Remark:
    return llvm::SourceMgr::DK_Remark;
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

SourceMgrDiagnosticHandler::SourceMgrDiagnosticHandler(llvm::SourceMgr &mgr,
                                                       MLIRContext *ctx)
    : mgr(mgr) {
  // Register a simple diagnostic handler.
  ctx->getDiagEngine().setHandler(
      [this](Location loc, StringRef message, DiagnosticSeverity kind) {
        emitDiagnostic(loc, message, kind);
      });
}

void SourceMgrDiagnosticHandler::emitDiagnostic(Location loc, Twine message,
                                                DiagnosticSeverity kind) {
  mgr.PrintMessage(convertLocToSMLoc(loc), getDiagKind(kind), message);
}

/// Get a memory buffer for the given file, or nullptr if one is not found.
const llvm::MemoryBuffer *
SourceMgrDiagnosticHandler::getBufferForFile(StringRef filename) {
  // Check for an existing mapping to the buffer id for this file.
  auto bufferIt = filenameToBuf.find(filename);
  if (bufferIt != filenameToBuf.end())
    return bufferIt->second;

  // Look for a buffer in the manager that has this filename.
  for (unsigned i = 1, e = mgr.getNumBuffers() + 1; i != e; ++i) {
    auto *buf = mgr.getMemoryBuffer(i);
    if (buf->getBufferIdentifier() == filename)
      return filenameToBuf[filename] = buf;
  }

  // Otherwise, default to the main file buffer, if it exists.
  const llvm::MemoryBuffer *newBuf = nullptr;
  if (mgr.getNumBuffers() != 0)
    newBuf = mgr.getMemoryBuffer(mgr.getMainFileID());
  return filenameToBuf[filename] = newBuf;
}

/// Get a memory buffer for the given file, or the main file of the source
/// manager if one doesn't exist. This always returns non-null.
llvm::SMLoc SourceMgrDiagnosticHandler::convertLocToSMLoc(Location loc) {
  // Process a FileLineColLoc.
  auto fileLoc = getFileLineColLoc(loc);
  if (!fileLoc)
    return llvm::SMLoc();

  // Get the buffer for this filename.
  auto *membuf = getBufferForFile(fileLoc->getFilename());
  if (!membuf)
    return llvm::SMLoc();

  // TODO: This should really be upstreamed to be a method on llvm::SourceMgr.
  // Doing so would allow it to use the offset cache that is already maintained
  // by SrcBuffer, making this more efficient.
  unsigned lineNo = fileLoc->getLine();
  unsigned columnNo = fileLoc->getColumn();

  // Scan for the correct line number.
  const char *position = membuf->getBufferStart();
  const char *end = membuf->getBufferEnd();

  // We start counting line and column numbers from 1.
  if (lineNo != 0)
    --lineNo;
  if (columnNo != 0)
    --columnNo;

  while (position < end && lineNo) {
    auto curChar = *position++;

    // Scan for newlines.  If this isn't one, ignore it.
    if (curChar != '\r' && curChar != '\n')
      continue;

    // We saw a line break, decrement our counter.
    --lineNo;

    // Check for \r\n and \n\r and treat it as a single escape.  We know that
    // looking past one character is safe because MemoryBuffer's are always nul
    // terminated.
    if (*position != curChar && (*position == '\r' || *position == '\n'))
      ++position;
  }

  // If the line/column counter was invalid, return a pointer to the start of
  // the buffer.
  if (lineNo || position + columnNo > end)
    return llvm::SMLoc::getFromPointer(membuf->getBufferStart());

  // Otherwise return the right pointer.
  return llvm::SMLoc::getFromPointer(position + columnNo);
}

//===----------------------------------------------------------------------===//
// SourceMgrDiagnosticVerifierHandler
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
// Record the expected diagnostic's position, substring and whether it was
// seen.
struct ExpectedDiag {
  DiagnosticSeverity kind;
  unsigned lineNo;
  StringRef substring;
  llvm::SMLoc fileLoc;
  bool matched;
};

struct SourceMgrDiagnosticVerifierHandlerImpl {
  SourceMgrDiagnosticVerifierHandlerImpl() : status(success()) {}

  /// Returns the expected diagnostics for the given source file.
  llvm::Optional<MutableArrayRef<ExpectedDiag>>
  getExpectedDiags(StringRef bufName);

  /// Computes the expected diagnostics for the given source buffer.
  MutableArrayRef<ExpectedDiag>
  computeExpectedDiags(const llvm::MemoryBuffer *buf);

  /// The current status of the verifier.
  LogicalResult status;

  /// A list of expected diagnostics for each buffer of the source manager.
  llvm::StringMap<SmallVector<ExpectedDiag, 2>> expectedDiagsPerFile;

  /// Regex to match the expected diagnostics format.
  llvm::Regex expected = llvm::Regex(
      "expected-(error|note|remark|warning) *(@[+-][0-9]+)? *{{(.*)}}");
};
} // end namespace detail
} // end namespace mlir

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagKindStr(DiagnosticSeverity kind) {
  switch (kind) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

/// Returns the expected diagnostics for the given source file.
llvm::Optional<MutableArrayRef<ExpectedDiag>>
SourceMgrDiagnosticVerifierHandlerImpl::getExpectedDiags(StringRef bufName) {
  auto expectedDiags = expectedDiagsPerFile.find(bufName);
  if (expectedDiags != expectedDiagsPerFile.end())
    return MutableArrayRef<ExpectedDiag>(expectedDiags->second);
  return llvm::None;
}

/// Computes the expected diagnostics for the given source buffer.
MutableArrayRef<ExpectedDiag>
SourceMgrDiagnosticVerifierHandlerImpl::computeExpectedDiags(
    const llvm::MemoryBuffer *buf) {
  // If the buffer is invalid, return an empty list.
  if (!buf)
    return llvm::None;
  auto &expectedDiags = expectedDiagsPerFile[buf->getBufferIdentifier()];

  // Scan the file for expected-* designators.
  SmallVector<StringRef, 100> lines;
  buf->getBuffer().split(lines, '\n');
  for (unsigned lineNo = 0, e = lines.size(); lineNo < e; ++lineNo) {
    SmallVector<StringRef, 3> matches;
    if (!expected.match(lines[lineNo], &matches))
      continue;
    // Point to the start of expected-*.
    auto expectedStart = llvm::SMLoc::getFromPointer(matches[0].data());

    DiagnosticSeverity kind;
    if (matches[1] == "error")
      kind = DiagnosticSeverity::Error;
    else if (matches[1] == "warning")
      kind = DiagnosticSeverity::Warning;
    else if (matches[1] == "remark")
      kind = DiagnosticSeverity::Remark;
    else {
      assert(matches[1] == "note");
      kind = DiagnosticSeverity::Note;
    }

    ExpectedDiag record{kind, lineNo + 1, matches[3], expectedStart, false};
    auto offsetMatch = matches[2];
    if (!offsetMatch.empty()) {
      int offset;
      // Get the integer value without the @ and +/- prefix.
      if (!offsetMatch.drop_front(2).getAsInteger(0, offset)) {
        if (offsetMatch[1] == '+')
          record.lineNo += offset;
        else
          record.lineNo -= offset;
      }
    }
    expectedDiags.push_back(record);
  }
  return expectedDiags;
}

SourceMgrDiagnosticVerifierHandler::SourceMgrDiagnosticVerifierHandler(
    llvm::SourceMgr &srcMgr, MLIRContext *ctx)
    : SourceMgrDiagnosticHandler(srcMgr, ctx),
      impl(new SourceMgrDiagnosticVerifierHandlerImpl()) {
  // Compute the expected diagnostics for each of the current files in the
  // source manager.
  for (unsigned i = 0, e = mgr.getNumBuffers(); i != e; ++i)
    (void)impl->computeExpectedDiags(mgr.getMemoryBuffer(i + 1));

  // Register a handler to verfy the diagnostics.
  ctx->getDiagEngine().setHandler(
      [&](Location loc, StringRef msg, DiagnosticSeverity kind) {
        // Process a FileLineColLoc.
        if (auto fileLoc = getFileLineColLoc(loc))
          return process(*fileLoc, msg, kind);

        emitDiagnostic(loc, "unexpected " + getDiagKindStr(kind) + ": " + msg,
                       DiagnosticSeverity::Error);
        impl->status = failure();
      });
}

SourceMgrDiagnosticVerifierHandler::~SourceMgrDiagnosticVerifierHandler() {
  // Ensure that all expected diagnosics were handled.
  (void)verify();
}

/// Returns the status of the verifier and verifies that all expected
/// diagnostics were emitted. This return success if all diagnostics were
/// verified correctly, failure otherwise.
LogicalResult SourceMgrDiagnosticVerifierHandler::verify() {
  // Verify that all expected errors were seen.
  for (auto &expectedDiagsPair : impl->expectedDiagsPerFile) {
    for (auto &err : expectedDiagsPair.second) {
      if (err.matched)
        continue;
      llvm::SMRange range(err.fileLoc,
                          llvm::SMLoc::getFromPointer(err.fileLoc.getPointer() +
                                                      err.substring.size()));
      mgr.PrintMessage(err.fileLoc, llvm::SourceMgr::DK_Error,
                       "expected " + getDiagKindStr(err.kind) + " \"" +
                           err.substring + "\" was not produced",
                       range);
      impl->status = failure();
    }
  }
  impl->expectedDiagsPerFile.clear();
  return impl->status;
}

/// Process a FileLineColLoc diagnostic.
void SourceMgrDiagnosticVerifierHandler::process(FileLineColLoc loc,
                                                 StringRef msg,
                                                 DiagnosticSeverity kind) {
  // Get the expected diagnostics for this file.
  auto diags = impl->getExpectedDiags(loc.getFilename());
  if (!diags)
    diags = impl->computeExpectedDiags(getBufferForFile(loc.getFilename()));

  // Search for a matching expected diagnostic.
  // If we find something that is close then emit a more specific error.
  ExpectedDiag *nearMiss = nullptr;

  // If this was an expected error, remember that we saw it and return.
  unsigned line = loc.getLine();
  for (auto &e : *diags) {
    if (line == e.lineNo && msg.contains(e.substring)) {
      if (e.kind == kind) {
        e.matched = true;
        return;
      }

      // If this only differs based on the diagnostic kind, then consider it
      // to be a near miss.
      nearMiss = &e;
    }
  }

  // Otherwise, emit an error for the near miss.
  if (nearMiss)
    mgr.PrintMessage(nearMiss->fileLoc, llvm::SourceMgr::DK_Error,
                     "'" + getDiagKindStr(kind) +
                         "' diagnostic emitted when expecting a '" +
                         getDiagKindStr(nearMiss->kind) + "'");
  else
    emitDiagnostic(loc, "unexpected " + getDiagKindStr(kind) + ": " + msg,
                   DiagnosticSeverity::Error);
  impl->status = failure();
}
