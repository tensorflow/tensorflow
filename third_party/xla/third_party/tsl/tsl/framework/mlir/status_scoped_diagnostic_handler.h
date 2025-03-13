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

#ifndef TENSORFLOW_TSL_FRAMEWORK_MLIR_STATUS_SCOPED_DIAGNOSTIC_HANDLER_H_
#define TENSORFLOW_TSL_FRAMEWORK_MLIR_STATUS_SCOPED_DIAGNOSTIC_HANDLER_H_

#include <string>

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/status/status.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace tsl {

// Diagnostic handler that collects all of the diagnostics reported and produces
// an absl::Status to return to callers.
class StatusScopedDiagnosticHandler : public mlir::SourceMgrDiagnosticHandler {
 public:
  explicit StatusScopedDiagnosticHandler(mlir::MLIRContext* context);

  // Destruction CHECK-fails if ConsumeStatus has not been called.
  ~StatusScopedDiagnosticHandler();

  // Returns the aggregate status.
  absl::Status consumeStatus();

  // Returns the aggregate status, if it is non-OK, or an error, if `result` is
  // mlir::failed. If the aggregate status is OK and mlir::succeeded(result),
  // returns OK.
  absl::Status consumeStatus(mlir::LogicalResult result);

 private:
  mlir::LogicalResult handleDiagnostic(mlir::Diagnostic& diag);

  std::string diag_str_;
  llvm::raw_string_ostream diag_stream_;
  llvm::SourceMgr source_mgr_;
  absl::Status status_;
  bool consumed_ = false;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_MLIR_STATUS_SCOPED_DIAGNOSTIC_HANDLER_H_
