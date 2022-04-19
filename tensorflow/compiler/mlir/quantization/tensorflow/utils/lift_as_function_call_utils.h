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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_OPS_LIFTING_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_OPS_LIFTING_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

// This header file defines common utils used by TF-Quant transformation
// passes to lift op compositions to a function.
namespace mlir {
namespace quant {

// Checks if the op is inside a lifted function.
bool IsInLiftedFunc(Operation *op);

// Creates a function to wrap the section between arguments and results.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results,
    const llvm::SmallVector<Attribute> &attributes);

// Same as above but with empty attributes.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results);

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_OPS_LIFTING_UTILS_H_
