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

#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"  // IWYU pragma: keep

#include "mlir/IR/Builders.h"  // from @llvm-project  // IWYU pragma: keep

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

using llvm::Optional;

//===----------------------------------------------------------------------===//
// TraceOp
//===----------------------------------------------------------------------===//

void TraceOp::getSuccessorRegions(Optional<unsigned> index,
                                  ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the TraceOp, branch into the body.
  if (!index) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

LogicalResult TraceOp::verify() {
  if (getRegion().front().getNumArguments() > 0)
    return emitOpError("region cannot have any arguments");
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

MutableOperandRange YieldOp::getMutableSuccessorOperands(
    Optional<unsigned> index) {
  return operandsMutable();
}

}  // namespace runtime
}  // namespace xla

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.cc.inc"
