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

//===----------------------------------------------------------------------===//
//
// This file defines the dialect for TensorFlow.js
//
//===----------------------------------------------------------------------===//

#ifndef TENSORFLOW_COMPILER_MLIR_TFJS_IR_TFJS_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TFJS_IR_TFJS_OPS_H_

#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace tfjs {

#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_dialect.h.inc"

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.h.inc"

}  // namespace tfjs
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TFJS_IR_TFJS_OPS_H_
