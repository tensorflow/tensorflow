/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

#include <cassert>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/status/status.h"
#include "tsl/platform/logging.h"

namespace tsl {

StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler(
    mlir::MLIRContext* context)
    : mlir::SourceMgrDiagnosticHandler(source_mgr_, context, diag_stream_),
      diag_stream_(diag_str_) {
  setHandler([&](mlir::Diagnostic& diag) { return handleDiagnostic(diag); });
}

StatusScopedDiagnosticHandler::~StatusScopedDiagnosticHandler() {
  assert(consumed_ && "Status must be consumed_ before destruction");
}

absl::Status StatusScopedDiagnosticHandler::consumeStatus() {
  consumed_ = true;
  return status_;
}

absl::Status StatusScopedDiagnosticHandler::consumeStatus(
    mlir::LogicalResult result) {
  consumed_ = true;
  if (failed(result) && status_.ok()) {
    return absl::UnknownError("Unknown MLIR failure");
  }
  return status_;
}

mlir::LogicalResult StatusScopedDiagnosticHandler::handleDiagnostic(
    mlir::Diagnostic& diag) {
  diag_str_.clear();
  emitDiagnostic(diag);
  diag_stream_.flush();

  // Emit non-errors to VLOG instead of the internal status.
  if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) {
    VLOG(1) << diag_str_;
    return mlir::success();
  }

  status_.Update(absl::UnknownError(diag_str_));

  // Return success to show that we `consumed_` the diagnostic.
  return mlir::success();
}

}  // namespace tsl
