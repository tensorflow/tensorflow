/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_LIFT_AS_FUNCTION_CALL_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_LIFT_AS_FUNCTION_CALL_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::quant {

// This attribute will be set for functions created by this pass.
// Presence of this attribute will mark the function as quantization target.
inline constexpr StringRef kFusedFunctionAttr = "tf_quant.composite_function";
// The keyword to detect if this is a `NullAttribute`.
inline constexpr StringRef kNullAttributeValue = "N/A";

// The attribute will be used for TF::XlaCallModuleOp to restore the original
// function name when loading it back.
inline constexpr StringRef kOriginalStablehloEntryFunctionAttrName =
    "_original_entry_function";

// Name of the string attribute attached to `XlaCallModuleOp`, which is the
// textproto representation of `Method`.
inline constexpr StringRef kQuantizationMethodAttr = "_quantization_method";

// FunctionCallOpType to be generated as the function call operator when
// function lifting will happen.
enum FunctionCallOpType { TFPartitionedCallOp = 0, TFXlaCallModuleOp = 1 };

// Checks if the op is inside a lifted function.
bool IsInLiftedFunc(Operation &op);

// Checks if the given einsum op is supported for XlaDotV2 quantization.
bool IsEinsumSupportedByXlaDotV2(mlir::StringAttr equation_attr);

// Creates a function to wrap the section between arguments and results.
// The generated function call op type will be decided by the given call_op_type
// argument. Currently, it supports TF::XlaCallModuleOp and
// TF::PartitionedCallOp function call op generations.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, FunctionCallOpType call_op_type,
    StringRef func_name, const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results,
    const llvm::SmallVector<NamedAttribute> &attributes);

// Same as above but with empty attributes.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, FunctionCallOpType call_op_type,
    StringRef func_name, const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results);

// Add the second argument to the first argument, which is expected to be an
// argument list.
// Used to attach bias to einsum argument list.
llvm::SmallVector<Value> AppendToVector(
    const llvm::SmallVector<Value> &arguments, Value append);

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_LIFT_AS_FUNCTION_CALL_H_
