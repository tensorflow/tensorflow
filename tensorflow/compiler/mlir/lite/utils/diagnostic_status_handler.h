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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_DIAGNOSTIC_STATUS_HANDLER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_DIAGNOSTIC_STATUS_HANDLER_H_

#include <string>

#include "absl/status/status.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::TFL {

// Diagnostic handler that collects the error diagnostics reported while it is
// in scope and turns them into an absl::Status. Non-error diagnostics are
// logged at VLOG(1) instead.
//
// Callers must consume a non-OK status (`ConsumeStatus`) before the handler
// goes out of scope.
class DiagnosticStatusHandler : public SourceMgrDiagnosticHandler {
 public:
  // If `propagate` is true, diagnostics are also passed on to the previously
  // registered handler.
  explicit DiagnosticStatusHandler(MLIRContext* context,
                                   bool propagate = false);
  ~DiagnosticStatusHandler();

  DiagnosticStatusHandler(const DiagnosticStatusHandler&) = delete;
  DiagnosticStatusHandler& operator=(const DiagnosticStatusHandler&) = delete;

  // Returns whether no errors were reported since the last `ConsumeStatus`.
  bool ok() const;

  // Returns an UNKNOWN status holding the error diagnostics reported so far,
  // or OK if there were none, and clears them.
  absl::Status ConsumeStatus();

 private:
  LogicalResult Handle(Diagnostic& diag);

  // Collected error messages. Declared before `diag_stream_`, which writes
  // to it.
  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;
  llvm::SourceMgr source_mgr_;
  bool propagate_;
};

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_DIAGNOSTIC_STATUS_HANDLER_H_
