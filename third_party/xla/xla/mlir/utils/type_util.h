/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_MLIR_UTILS_TYPE_UTIL_H_
#define XLA_MLIR_UTILS_TYPE_UTIL_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "xla/xla_data.pb.h"

// Type utilities to match MLIR types to XLA primitive types and vice versa.
namespace xla {
// Converts an XLA primitive type to the corresponding MLIR type.
// Signed XLA primitive types are converted to signless MLIR types;
// unsigned XLA primitive types are converted to unsigned MLIR types.
absl::StatusOr<mlir::Type> ConvertPrimitiveTypeToMlirType(
    xla::PrimitiveType type, mlir::Builder b);

// Returns an XLA xla::PrimitiveType equivalent of an MLIR Type that represents
// a primitive type (e.g., i8, f32), else returns PRIMITIVE_TYPE_INVALID.
// Signless MLIR types are converted to signed XLA primitive types.
xla::PrimitiveType ConvertMlirTypeToPrimitiveType(mlir::Type type);
}  // namespace xla

#endif  // XLA_MLIR_UTILS_TYPE_UTIL_H_
