/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_QUANT_ATTRS_AND_CONSTRAINTS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_QUANT_ATTRS_AND_CONSTRAINTS_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

// Attribute and constraint helpers used by the StableHLO quantization passes.
// These have no dependency on the TensorFlow dialect.

namespace mlir::odml {

// Returns true if `value` has a ranked shaped type with a static shape.
bool HasStaticShape(Value value);

// Clones `old_type` (which must be a `ShapedType`) with a new element type.
// Returns a null type if `old_type` is not a `ShapedType`.
Type CloneTypeWithNewElementType(Type old_type, Type element_type);

// Tries casting `op` with a concrete op type `T`. If the cast fails or `op` is
// a `nullptr`, returns `failure` and prints a debugging message identifying
// the cast attempt as `name`.
template <typename T>
FailureOr<T> TryCast(Operation* op, const StringRef name) {
  auto cast_op = dyn_cast_or_null<T>(op);
  if (cast_op) return cast_op;
  DEBUG_WITH_TYPE("mlir-odml-quant-attrs-and-constraints",
                  llvm::dbgs() << "Failed to match " << name << " ("
                               << T::getOperationName() << ").\n");
  return failure();
}

// Casts `value` to int32. Returns failure if it is out of range of int32.
FailureOr<int32_t> CastI64ToI32(int64_t value);

// Casts an array of int64 to int32. Returns failure if any element is out of
// range of int32.
FailureOr<SmallVector<int32_t>> CastI64ArrayToI32(
    ArrayRef<int64_t> int64_array);

// Returns the first user of `op` of type `T`, or nullptr if there is none.
template <typename T = Operation*>
Operation* FindUserOfType(Operation* op) {
  for (Operation* user : op->getUsers()) {
    if (isa<T>(user)) return user;
  }
  return nullptr;
}

// Returns the first operand of `op` that is defined by an op of type `T`, or
// nullptr if there is none.
template <typename T = Operation*>
Operation* FindOperandOfType(Operation* op) {
  for (Value operand_value : op->getOperands()) {
    if (isa<T>(operand_value.getDefiningOp())) {
      return operand_value.getDefiningOp();
    }
  }
  return nullptr;
}

// Returns true if `dot_general_op` can be lowered to a fully connected op.
absl::StatusOr<bool> IsDotGeneralFullyConnected(
    stablehlo::DotGeneralOp dot_general_op);

}  // namespace mlir::odml

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_QUANT_ATTRS_AND_CONSTRAINTS_H_
