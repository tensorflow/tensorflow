/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_DIALECT_H_
#define TENSORFLOW_CORE_IR_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/core/ir/types/dialect.h"

namespace mlir {
namespace tfg {
// Include all the TensorFlow types directly in the TFG namespace.
using namespace mlir::tf_type;  // NOLINT
}  // namespace tfg
}  // namespace mlir
// Dialect main class is defined in ODS, we include it here.
#include "tensorflow/core/ir/dialect.h.inc"

#endif  // TENSORFLOW_CORE_IR_DIALECT_H_
