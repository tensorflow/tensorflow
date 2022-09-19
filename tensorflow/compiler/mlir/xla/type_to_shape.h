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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TYPE_TO_SHAPE_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TYPE_TO_SHAPE_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Returns a XLA Shape equivalent of a MLIR Type, else returns empty shape.
Shape TypeToShape(mlir::Type type);

// Returns a XLA PrimitiveType equivalent of a MLIR Type that represents a
// primitive type (e.g., i8, f32), else returns PRIMITIVE_TYPE_INVALID.
PrimitiveType TypeToPrimitiveType(mlir::Type type);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TYPE_TO_SHAPE_H_
