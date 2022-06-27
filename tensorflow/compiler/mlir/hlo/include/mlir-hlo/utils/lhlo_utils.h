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

#ifndef MLIR_HLO_UTILS_LHLO_UTILS_H
#define MLIR_HLO_UTILS_LHLO_UTILS_H

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace lmhlo {

// TODO(b/236017415): remove when mhlo uses prefix accessor.
namespace accessor_dispatch {
template <typename OpT>
auto getOutputs(OpT op, int) -> decltype(op.getOutputs(), ValueRange{}) {
  return op.getOutputs();
}
template <typename OpT>
auto getOutputs(OpT op, char) -> decltype(op.results(), ValueRange{}) {
  return op.results();
}

template <typename OpT>
auto getInputs(OpT op, int) -> decltype(op.getInputs(), ValueRange{}) {
  return op.getInputs();
}
template <typename OpT>
auto getInputs(OpT op, char) -> decltype(op.operands(), ValueRange{}) {
  return op.operands();
}
}  // namespace accessor_dispatch

template <typename OpT>
static LogicalResult VerifyAllReduce(OpT op) {
  if (failed(mlir::hlo::VerifyReplicaGroups(op, /*is_uniform_sized=*/false)))
    return failure();

  // AllReduce has variadic operands and results that have the same size.
  // Each member of the operand should have the same type as the corresponding
  // member of the result.
  for (auto it : llvm::enumerate(
           llvm::zip(accessor_dispatch::getInputs(op, 0).getTypes(),
                     accessor_dispatch::getOutputs(op, 0).getTypes()))) {
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

#endif  // MLIR_HLO_UTILS_LHLO_UTILS_H
