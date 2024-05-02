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

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir::quant {

// This attribute will be set for functions created by this pass.
// Presence of this attribute will mark the function as quantization target.
inline constexpr StringRef kFusedFunctionAttr = "tf_quant.composite_function";
// The keyword to detect if this is a `NullAttribute`.
inline constexpr StringRef kNullAttributeValue = "N/A";

// Prefixes attached to lifted functions.
constexpr StringRef kQuantizedFuncPrefix = "quantized_";
constexpr StringRef kCompositeFuncPrefix = "composite_";

// The attribute will be used for TF::XlaCallModuleOp to restore the original
// function name when loading it back.
inline constexpr StringRef kOriginalStablehloEntryFunctionAttrName =
    "_original_entry_function";

// FunctionCallOpType to be generated as the function call operator when
// function lifting will happen.
enum FunctionCallOpType { TFPartitionedCallOp = 0, TFXlaCallModuleOp = 1 };

// Checks if an op is inside a lifted function.
// If the given op pointer is a nullptr, returns false.
bool IsInLiftedFunc(Operation* op);

// Checks if the op is inside a StableHLO op with region.
// If the given op pointer is a nullptr, returns false.
bool IsInStableHloOpRegion(Operation* op);

// Checks if a given einsum op is supported for XlaDotV2 quantization.
bool IsEinsumSupportedByXlaDotV2(StringAttr equation_attr);

// Gets the quantization method from `op`. It is retrieved from the
// `kQuantizationMethodAttr` string attribute. Returns
// `absl::InvalidArgumentError` when the attribute doesn't exist. Returns
// `absl::InternalError` when parsing the attribute to `Method` failed.
// `op` must be non-null.
absl::StatusOr<::stablehlo::quantization::Method> GetQuantizationMethod(
    absl::Nonnull<Operation*> op);

// Gets the quantization method from `op`. It is retrieved from the
// `kQuantizationMethodAttr` string attribute. Returns a default instance of
// `Method` iff the attribute doesn't exist or the attribute contains an invalid
// textproto for `Method`. `op` must be non-null.
::stablehlo::quantization::Method GetQuantizationMethodOrDefault(
    absl::Nonnull<Operation*> op);

// Creates a function to wrap the section between arguments and results.
// The generated function call op type will be decided by the given call_op_type
// argument. Currently, it supports TF::XlaCallModuleOp and
// TF::PartitionedCallOp function call op generations.
SmallVector<Value, 4> LiftAsFunctionCall(OpBuilder& builder, Location location,
                                         FunctionCallOpType call_op_type,
                                         StringRef func_name,
                                         ArrayRef<Value> arguments,
                                         ArrayRef<Value> results,
                                         ArrayRef<NamedAttribute> attributes);

// Same as above but with empty attributes.
SmallVector<Value, 4> LiftAsFunctionCall(OpBuilder& builder, Location location,
                                         FunctionCallOpType call_op_type,
                                         StringRef func_name,
                                         ArrayRef<Value> arguments,
                                         ArrayRef<Value> results);

// Add the second argument to the first argument, which is expected to be an
// argument list.
// Used to attach bias to einsum argument list.
SmallVector<Value> AppendToVector(ArrayRef<Value> arguments, Value append);

// Checks if the `Method` attatched to the given `tf.XlaCallModule` op has
// `WeightOnlyPtq`.
bool HasWeightOnlyPtqMethod(TF::XlaCallModuleOp xla_call_module_op);

// Checks if an op is a `tf.XlaCallModule` op, contains 'conv' or 'dot_general'
// in its name and has `Method` with `WeightOnlyPtq`.
bool IsWeightOnlyQuantizableOp(const Operation& op);

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_LIFT_AS_FUNCTION_CALL_H_
