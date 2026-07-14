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
#ifndef XLA_CODEGEN_EMITTERS_TYPE_UTIL_H_
#define XLA_CODEGEN_EMITTERS_TYPE_UTIL_H_

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace emitters {

inline constexpr absl::string_view kHasNoCompute = "no_compute";

// Converts an XLA tensor to an MLIR ranked tensor. The layout is stored in the
// encoding attribute, if it is not the default layout. `shape` must be an
// array.
mlir::Type TensorShapeToMlirType(const Shape& shape, mlir::OpBuilder& b);

// Converts an XLA primitive type to an MLIR type. All integers are converted to
// signless integers.
mlir::Type PrimitiveTypeToMlirType(PrimitiveType type, mlir::OpBuilder& b);
// Converts an XLA primitive type to an MLIR type, preserving the sign.
mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType type,
                                           mlir::OpBuilder& b);

// If `shape` is a tuple, returns the converted tuple shapes. Otherwise returns
// just the converted shape. Nested tuples are not supported.
llvm::SmallVector<mlir::Type> ShapeToMlirTypes(const Shape& shape,
                                               mlir::OpBuilder& b);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TYPE_UTIL_H_
