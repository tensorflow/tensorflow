/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_MLIR_FINGERPRINT_UTILS_H_
#define XLA_PYTHON_IFRT_MLIR_FINGERPRINT_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"

namespace xla {
namespace ifrt {

// Returns a fingerprint of the given MLIR module. Two MLIR modules are
// equivalent if their fingerprints are the same. May ignore debug info.
absl::StatusOr<uint64_t> FingerprintModuleOp(mlir::ModuleOp module_op);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MLIR_FINGERPRINT_UTILS_H_
