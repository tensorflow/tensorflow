//===- Status.h - Utilities for handling success/failure --------*- C++ -*-===//
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

#ifndef MLIR_SUPPORT_STATUS_H
#define MLIR_SUPPORT_STATUS_H

#include "mlir/Support/LLVM.h"

namespace mlir {

// Values that can be used to signal success/failure. This should be used in
// conjunction with the 'succeeded' and 'failed' functions below.
struct Status {
  enum ResultEnum { Success, Failure } value;
  Status(ResultEnum v) : value(v) {}

  /// Utility method to generate a success result.
  static Status success() { return Success; }

  /// Utility method to generate a failure result.
  static Status failure() { return Failure; }
};

/// Utility function that returns true if the provided Status corresponds
/// to a success value.
inline bool succeeded(Status result) { return result.value == Status::Success; }

/// Utility function that returns true if the provided Status corresponds
/// to a failure value.
inline bool failed(Status result) { return result.value == Status::Failure; }

} // namespace mlir

#endif // MLIR_SUPPORT_STATUS_H
