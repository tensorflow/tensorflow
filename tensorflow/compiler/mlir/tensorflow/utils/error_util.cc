/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

#include "tensorflow/core/lib/core/errors.h"

namespace mlir {

StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler(
    MLIRContext* context, bool propagate)
    : ScopedDiagnosticHandler(context), propagate_(propagate) {
  context->getDiagEngine().setHandler(
      [this](Diagnostic diag) { this->handler(std::move(diag)); });
}

StatusScopedDiagnosticHandler::~StatusScopedDiagnosticHandler() {
  // Verify errors were consumed and re-register old handler.
  bool all_errors_produced_were_consumed = ok();
  DCHECK(all_errors_produced_were_consumed) << "Error status not consumed:\n"
                                            << instr_str_;
}

bool StatusScopedDiagnosticHandler::ok() const { return instr_str_.empty(); }

Status StatusScopedDiagnosticHandler::ConsumeStatus() {
  if (ok()) return Status::OK();

  Status s = tensorflow::errors::Unknown(instr_str_);
  instr_str_.clear();
  return s;
}

Status StatusScopedDiagnosticHandler::Combine(Status status) {
  if (status.ok()) return ConsumeStatus();

  // status is not-OK here, so if there was no diagnostics reported
  // additionally then return this error.
  if (ok()) return status;

  // Append the diagnostics reported to the status. This repeats the behavior of
  // TensorFlow's AppendToMessage without the additional formatting inserted
  // there.
  status = ::tensorflow::Status(
      status.code(), absl::StrCat(status.error_message(), instr_str_));
  instr_str_.clear();
  return status;
}

void StatusScopedDiagnosticHandler::handler(Diagnostic diag) {
  // Skip notes and warnings.
  if (diag.getSeverity() != DiagnosticSeverity::Error) {
#ifndef NDEBUG
    VLOG(1) << "Non-error diagnostic: " << diag.str();
    for (auto& note : diag.getNotes())
      VLOG(1) << "Non-error diagnostic: " << note.str();
#endif
    return;
  }

  // Indent the diagnostic message to effectively show the diagnostics reported
  // as nested under the returned Status's message.
  llvm::raw_string_ostream os(instr_str_);
  os.indent(2);
  if (auto fileLoc = diag.getLocation().dyn_cast<FileLineColLoc>())
    os << fileLoc.getFilename() << ':' << fileLoc.getLine() << ':'
       << fileLoc.getColumn() << ": ";
  os << "error: " << diag << '\n';

  // Propagate error if needed.
  if (propagate_) {
    propagateDiagnostic(std::move(diag));
  }
}

}  // namespace mlir
