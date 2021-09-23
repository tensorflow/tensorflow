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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_LHLO_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_LHLO_UTILS_H_

#include "llvm/ADT/SmallSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace lmhlo {

// Verifies replica groups attached to collective communication operations.
// If the attribute is not empty, it must be a rank 2 tensor, and each replica
// should appear exactly once. If `is_uniform_sized` is true, then we also check
// that each group is of the same size. If the operation has
// `use_global_device_ids` set, then replica group cannot be empty.
template <typename OpT>
LogicalResult VerifyReplicaGroups(OpT op, bool is_uniform_sized) {
  DenseIntElementsAttr attr = op.replica_groups();
  auto replica_group_type = attr.getType().dyn_cast<RankedTensorType>();
  if (!replica_group_type || replica_group_type.getRank() != 2 ||
      !replica_group_type.getElementType().isInteger(/*width=*/64))
    return op.emitOpError(
        "replica groups should be a rank 2 tensor of 64 bit integers");

  if (replica_group_type.getShape().equals(ArrayRef<int64_t>{0, 0})) {
    if (op.use_global_device_ids()) {
      return op.emitOpError(
          "if `use_global_device_ids` is set, the replica groups cannot be "
          "empty");
    }
    return success();
  }

  int64_t max_replica_id_seen = 0;
  llvm::SmallSet<int64_t, 8> replica_seen;
  for (int64_t id : attr.getValues<int64_t>()) {
    // Replica groups are stored in a 2D tensor. If the op supports non-uniform
    // groups, null replica IDs are stored as -1.
    if (id == -1) {
      if (is_uniform_sized) {
        return op.emitOpError("Invalid replica id -1");
      }
      continue;
    }

    if (!replica_seen.insert(id).second) {
      return op.emitOpError("replica id #") << id << " seen more than once";
    }
    max_replica_id_seen = std::max(max_replica_id_seen, id);
  }

  for (int64_t id = 0; id <= max_replica_id_seen; id++) {
    if (!replica_seen.contains(id)) {
      return op.emitOpError("replica id #")
             << id << " not seen in replica groups";
    }
  }
  return success();
}

template <typename OpT>
static LogicalResult VerifyAllReduce(OpT op) {
  if (failed(VerifyReplicaGroups(op, /*is_uniform_sized=*/false)))
    return failure();

  // AllReduce has variadic operands and results that have the same size.
  // Each member of the operand should have the same type as the corresponding
  // member of the result.
  for (auto it : llvm::enumerate(
           llvm::zip(op.operands().getTypes(), op.results().getTypes()))) {
    Type operandType = std::get<0>(it.value());
    Type resultType = std::get<1>(it.value());
    if (operandType != resultType)
      return op.emitOpError("requires operand #")
             << it.index() << " (type: " << operandType << ") and result #"
             << it.index() << " (type: " << resultType << ") to have same type";
  }
  return success();
}

}  // namespace lmhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_LHLO_UTILS_H_
