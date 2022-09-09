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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_LIFT_AS_FUNCTION_CALL_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_LIFT_AS_FUNCTION_CALL_UTILS_H_

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

// This header file defines common utils used by TF-Quant transformation
// passes to lift op compositions to a function.
namespace mlir {
namespace quant {

inline constexpr absl::string_view kAttrMapAttribute = "attr_map";
// This attribute will be set for functions created by this pass.
inline constexpr absl::string_view kFusedFunctionAttr =
    "tf_quant.composite_function";
// The keyword to detect if this is a `NullAttribute`.
inline constexpr absl::string_view kNullAttributeValue = "N/A";

// Checks if the op is inside a lifted function.
bool IsInLiftedFunc(Operation *op);

// Creates a function to wrap the section between arguments and results.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results,
    const llvm::SmallVector<NamedAttribute> &attributes);

// Same as above but with empty attributes.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results);

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_UTILS_LIFT_AS_FUNCTION_CALL_UTILS_H_
