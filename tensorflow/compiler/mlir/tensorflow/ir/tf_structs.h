/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_STRUCTS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_STRUCTS_H_

#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project

namespace mlir {

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h.inc"

}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_STRUCTS_H_
