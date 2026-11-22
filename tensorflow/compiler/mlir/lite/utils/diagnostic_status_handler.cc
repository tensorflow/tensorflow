/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/diagnostic_status_handler.h"

#include <cstddef>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::TFL {

// The base class only keeps references to `source_mgr_` and `diag_stream_`;
// it does not use them before they are constructed.
DiagnosticStatusHandler::DiagnosticStatusHandler(MLIRContext* context,
                                                 bool propagate)
    : SourceMgrDiagnosticHandler(source_mgr_, context, diag_stream_),
      diag_stream_(diag_str_),
      propagate_(propagate) {
  setHandler([this](Diagnostic& diag) { return Handle(diag); });
}

DiagnosticStatusHandler::~DiagnosticStatusHandler() {
  DCHECK(ok()) << "Error status not consumed:\n" << diag_str_;
}

bool DiagnosticStatusHandler::ok() const { return diag_str_.empty(); }

absl::Status DiagnosticStatusHandler::ConsumeStatus() {
  if (ok()) return absl::OkStatus();
  absl::Status status = absl::UnknownError(diag_str_);
  diag_str_.clear();
  return status;
}

LogicalResult DiagnosticStatusHandler::Handle(Diagnostic& diag) {
  const size_t previous_size = diag_str_.size();
  emitDiagnostic(diag);
  diag_stream_.flush();

  // Only errors go into the status.
  if (diag.getSeverity() != DiagnosticSeverity::Error) {
    VLOG(1) << diag_str_.substr(previous_size);
    diag_str_.resize(previous_size);
  }

  // Failure tells the engine to pass the diagnostic on to the previous
  // handler.
  return failure(propagate_);
}

}  // namespace mlir::TFL
