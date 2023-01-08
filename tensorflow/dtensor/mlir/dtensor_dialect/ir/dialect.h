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

#ifndef TENSORFLOW_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DIALECT_H_
#define TENSORFLOW_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project

// Dialect main class is defined in ODS, we include it here. The
// constructor and the printing/parsing of dialect types are manually
// implemented (see ops.cpp).
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h.inc"

namespace mlir {
namespace dtensor {

//===----------------------------------------------------------------------===//
// DTENSOR dialect types.
//===----------------------------------------------------------------------===//

}  // namespace dtensor
}  // namespace mlir

#endif  // TENSORFLOW_DTENSOR_MLIR_DTENSOR_DIALECT_IR_DIALECT_H_
