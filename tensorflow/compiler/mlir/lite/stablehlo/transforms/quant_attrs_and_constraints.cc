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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/quant_attrs_and_constraints.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::odml {

bool HasStaticShape(Value value) {
  auto shaped_type = mlir::dyn_cast<ShapedType>(value.getType());
  if (!shaped_type) return false;
  return shaped_type.hasStaticShape();
}

Type CloneTypeWithNewElementType(Type old_type, Type element_type) {
  if (!mlir::isa<ShapedType>(old_type)) return {};
  return mlir::cast<ShapedType>(old_type).clone(element_type);
}

FailureOr<int32_t> CastI64ToI32(const int64_t value) {
  if (!llvm::isInt<32>(value)) {
    DEBUG_WITH_TYPE("mlir-odml-quant-attrs-and-constraints",
                    llvm::dbgs()
                        << "Tried to cast " << value
                        << "from int64 to int32, but lies out of range of "
                           "int32.\n");
    return failure();
  }
  return static_cast<int32_t>(value);
}

FailureOr<SmallVector<int32_t>> CastI64ArrayToI32(
    const ArrayRef<int64_t> int64_array) {
  SmallVector<int32_t> int32_array{};
  int32_array.reserve(int64_array.size());

  for (const int64_t i64 : int64_array) {
    FailureOr<int32_t> cast_i32 = CastI64ToI32(i64);
    if (failed(cast_i32)) return failure();
    int32_array.push_back(*cast_i32);
  }
  return int32_array;
}

absl::StatusOr<bool> IsDotGeneralFullyConnected(
    stablehlo::DotGeneralOp dot_general_op) {
  if (dot_general_op == nullptr) {
    return absl::InvalidArgumentError(
        "Given dot_general op cannot be null when checking "
        "`IsDotGeneralFullyConnected`.");
  }
  const stablehlo::DotDimensionNumbersAttr dot_dimension_numbers =
      dot_general_op.getDotDimensionNumbers();
  const ArrayRef<int64_t> lhs_contracting_dims =
      dot_dimension_numbers.getLhsContractingDimensions();
  const ArrayRef<int64_t> rhs_contracting_dims =
      dot_dimension_numbers.getRhsContractingDimensions();
  const int64_t input_rank =
      mlir::dyn_cast<ShapedType>(dot_general_op.getOperand(0).getType())
          .getRank();
  const int64_t filter_rank =
      mlir::dyn_cast<ShapedType>(dot_general_op.getOperand(1).getType())
          .getRank();
  // The following conditions are such requirements:
  //   - rank(lhs) is 1 or 2
  //   - rank(rhs) = 2
  //   - size(lhs_contracting_dimensions) = 1
  //   - size(rhs_contracting_dimensions) = 1
  //   - lhs_contracting_dimension = last dimension of lhs.
  //   - `stablehlo.dot_general` should not have `lhs_batching_dim`.
  //   - quantization_dimension(rhs) should not be in
  //     `rhs_contracting_dimensions`.
  // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
  const bool has_proper_rank =
      (input_rank == 1 || input_rank == 2) && filter_rank == 2;
  const bool has_proper_contracting_dim =
      lhs_contracting_dims.size() == 1 && rhs_contracting_dims.size() == 1 &&
      lhs_contracting_dims[0] == input_rank - 1;
  const bool is_not_batch_op =
      dot_dimension_numbers.getLhsBatchingDimensions().empty();
  const bool has_proper_quantization_dimension =
      absl::c_find(rhs_contracting_dims, filter_rank) ==
      rhs_contracting_dims.end();
  return has_proper_rank && has_proper_contracting_dim && is_not_batch_op &&
         has_proper_quantization_dimension;
}

}  // namespace mlir::odml
