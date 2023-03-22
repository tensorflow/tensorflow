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

#include "tensorflow/compiler/xla/mlir/utils/error_util.h"

#include <string>
#include <string_view>

#include "tensorflow/tsl/platform/errors.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project

namespace mlir {
BaseScopedDiagnosticHandler::BaseScopedDiagnosticHandler(MLIRContext* context,
                                                         bool propagate,
                                                         bool filter_stack)
    : SourceMgrDiagnosticHandler(source_mgr_, context, diag_stream_),
      diag_stream_(diag_str_),
      propagate_(propagate) {}

BaseScopedDiagnosticHandler::~BaseScopedDiagnosticHandler() {
  // Verify errors were consumed and re-register old handler.
  bool all_errors_produced_were_consumed = ok();
  DCHECK(all_errors_produced_were_consumed) << "Error status not consumed:\n"
                                            << diag_str_;
}

bool BaseScopedDiagnosticHandler::ok() const { return diag_str_.empty(); }

absl::Status BaseScopedDiagnosticHandler::ConsumeStatus() {
  if (ok()) return absl::OkStatus();

  // TODO(jpienaar) This should be combining status with one previously built
  // up.
  absl::Status s = absl::UnknownError(diag_str_);
  diag_str_.clear();
  return s;
}

absl::Status BaseScopedDiagnosticHandler::Combine(absl::Status status) {
  if (status.ok()) return ConsumeStatus();

  // status is not-OK here, so if there was no diagnostics reported
  // additionally then return this error.
  if (ok()) return status;

  // Append the diagnostics reported to the status. This repeats the behavior of
  // TensorFlow's AppendToMessage without the additional formatting inserted
  // there.
  std::string str_status_message(status.message());
  status = absl::Status(status.code(), str_status_message + diag_str_);
  diag_str_.clear();

  return status;
}

LogicalResult BaseScopedDiagnosticHandler::handler(Diagnostic* diag) {
  size_t current_diag_str_size_ = diag_str_.size();

  // Emit the diagnostic and flush the stream.
  emitDiagnostic(*diag);
  diag_stream_.flush();

  // Emit non-errors to VLOG instead of the internal status.
  if (diag->getSeverity() != DiagnosticSeverity::Error) {
    VLOG(1) << diag_str_.substr(current_diag_str_size_);
    diag_str_.resize(current_diag_str_size_);
  }

  // Return failure to signal propagation if necessary.
  return failure(propagate_);
}

}  // namespace mlir
