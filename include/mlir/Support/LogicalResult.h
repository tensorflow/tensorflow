//===- LogicalResult.h - Utilities for handling success/failure -*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_LOGICAL_RESULT_H
#define MLIR_SUPPORT_LOGICAL_RESULT_H

#include "mlir/Support/LLVM.h"

namespace mlir {

// Values that can be used to signal success/failure. This should be used in
// conjunction with the 'succeeded' and 'failed' functions below.
struct LogicalResult {
  enum ResultEnum { Success, Failure } value;
  LogicalResult(ResultEnum v) : value(v) {}

  /// Utility method to generate a success result.
  static LogicalResult success() { return Success; }

  /// Utility method to generate a failure result.
  static LogicalResult failure() { return Failure; }
};

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a success value.
inline bool succeeded(LogicalResult result) {
  return result.value == LogicalResult::Success;
}

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a failure value.
inline bool failed(LogicalResult result) {
  return result.value == LogicalResult::Failure;
}

} // namespace mlir

#endif // MLIR_SUPPORT_LOGICAL_RESULT_H
