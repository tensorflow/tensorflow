/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_UTILS_CONSTRAINTS_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_UTILS_CONSTRAINTS_H_

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/FunctionInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/xla/runtime/constraints.h"

namespace xla {
namespace runtime {
// Returns arguments constraints inferred from the function signature.
absl::StatusOr<llvm::SmallVector<ArgumentConstraint>> GetArgumentsConstraints(
    mlir::FunctionOpInterface func);

// Resolves argument constraint based on the argument type, if constraint is
// fully satisfied by the type, returns `kResolved`.
absl::StatusOr<ArgumentConstraint> ResolveArgumentConstraint(
    ArgumentConstraint constraint, mlir::Type type);

// Returns true iff the value of given type can be sunk into the function body
// at run time via value specialization.
inline bool SupportsValueSpecialization(mlir::Type type) {
  // TODO(ezhulenev): Add support for sinking `memref` values once the value
  // specialization will support it.
  mlir::TensorType tensor = type.dyn_cast<mlir::TensorType>();
  return tensor && (tensor.getRank() == 0 || tensor.getRank() == 1) &&
         (tensor.getElementType().isInteger(32) ||
          tensor.getElementType().isInteger(64));
}

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_UTILS_CONSTRAINTS_H_
