/* Copyright 2023 The JAX Authors.

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

#include <vector>

#include "absl/log/check.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir {
namespace tpu {

namespace {

MemRefType updateMemorySpace(MemRefType ty, Attribute memory_space) {
  return MemRefType::get(ty.getShape(), ty.getElementType(), ty.getLayout(),
                         memory_space);
}

MemRefType updateMemorySpace(MemRefType ty, MemorySpace memory_space) {
  return updateMemorySpace(ty,
                           MemorySpaceAttr::get(ty.getContext(), memory_space));
}

}  // namespace

LogicalResult specializeMemorySpace(TypedValue<MemRefType> value,
                                    MemorySpace memory_space) {
  MemorySpaceAttr attr =
      dyn_cast_if_present<MemorySpaceAttr>(value.getType().getMemorySpace());
  if (!attr) {
    return failure();
  }
  MemorySpace current_memory_space = attr.getValue();
  if (current_memory_space == memory_space) {
    return success();  // Nothing to do here.
  } else if (current_memory_space != MemorySpace::kAny) {
    return failure();  // Memory space mismatch!
  }
  value.setType(updateMemorySpace(value.getType(), memory_space));
  std::vector<Operation*> to_update(value.getUsers().begin(),
                                    value.getUsers().end());
  auto updateResultFrom = [&](Operation* op, MemRefType ty) {
    Attribute source_memory_space = ty.getMemorySpace();
    CHECK_EQ(op->getNumResults(), 1);
    Value result = op->getResult(0);
    MemRefType result_type = cast<MemRefType>(result.getType());
    if (result_type.getMemorySpace() != source_memory_space) {
      result.setType(updateMemorySpace(result_type, source_memory_space));
      to_update.insert(to_update.end(), result.getUsers().begin(),
                       result.getUsers().end());
    }
  };
  while (!to_update.empty()) {
    Operation* some_op = to_update.back();
    to_update.pop_back();
    // Here we only have to handle the operations allowed on refs with
    // unspecified memory space.
    if (auto op = dyn_cast<tpu::ReinterpretCastOp>(some_op)) {
      updateResultFrom(op, op.getInput().getType());
      continue;
    }
    if (auto op = dyn_cast<tpu::MemRefSliceOp>(some_op)) {
      updateResultFrom(op, op.getMemRef().getType());
      continue;
    }
    if (auto op = dyn_cast<tpu::MemRefSqueezeOp>(some_op)) {
      updateResultFrom(op, op.getInput().getType());
      continue;
    }
    if (auto op = dyn_cast<tpu::MemRefBitcastOp>(some_op)) {
      updateResultFrom(op, op.getInput().getType());
      continue;
    }
    if (auto op = dyn_cast<tpu::MemRefReshapeOp>(some_op)) {
      updateResultFrom(op, op.getInput().getType());
      continue;
    }
    if (auto op = dyn_cast<tpu::EraseLayoutOp>(some_op)) {
      updateResultFrom(op, op.getOperand().getType());
      continue;
    }
    if (auto op = dyn_cast<tpu::EnqueueDMAOp>(some_op)) {
      continue;  // Nothing to do.
    }
    if (auto op = dyn_cast<tpu::WaitDMAOp>(some_op)) {
      continue;  // Nothing to do.
    }
    if (auto op = dyn_cast<tpu::WaitDMA2Op>(some_op)) {
      continue;  // Nothing to do.
    }
    some_op->emitOpError(
        "Failed to propagate memory space update through this operation");
    return failure();
  }
  return success();
}

}  // namespace tpu
}  // namespace mlir
