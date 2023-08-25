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

#include "deallocation/IR/deallocation_ops.h"
#include "tools/mlir_interpreter/framework/registration.h"
#include "tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue getBuffer(InterpreterState&, deallocation::GetBufferOp,
                           const InterpreterValue& memref) {
  if (!memref.buffer()) return {intptr_t{0}};
  return {reinterpret_cast<intptr_t>(memref.buffer()->at(0, 0))};
}

InterpreterValue null(InterpreterState&, deallocation::NullOp) {
  TensorOrMemref<uint8_t> r;
  r.buffer = nullptr;
  r.view.sizes = {0};
  return {r};
}

InterpreterValue own(InterpreterState&, deallocation::OwnOp,
                     const InterpreterValue& alloc) {
  return alloc;
}

void freeAlloc(InterpreterState& state, deallocation::FreeOp op,
               const InterpreterValue& alloc) {
  if (auto* stats = state.getOptions().stats) {
    stats->heapSize -= alloc.buffer()->getByteSize();
    ++stats->numDeallocations;
  }
  alloc.buffer()->deallocate(op);
}

SmallVector<InterpreterValue> retain(InterpreterState& state,
                                     deallocation::RetainOp op,
                                     ArrayRef<InterpreterValue> values,
                                     ArrayRef<InterpreterValue> owned) {
  SmallVector<InterpreterValue> result(values.size(), null(state, {}));

  llvm::SmallBitVector used(owned.size());
  for (auto [index, v] : llvm::enumerate(values)) {
    for (auto [ownedIndex, o] : llvm::enumerate(owned)) {
      if (used[ownedIndex] || o.buffer() == nullptr) continue;
      if (v.buffer() == o.buffer()) {
        used[ownedIndex] = true;
        result[index] = o;
      }
    }
  }

  for (int64_t i = 0; i < owned.size(); ++i) {
    if (!used[i] && owned[i].buffer()) {
      if (auto* stats = state.getOptions().stats) {
        stats->heapSize -= owned[i].buffer()->getByteSize();
        ++stats->numDeallocations;
      }
      owned[i].buffer()->deallocate(op);
    }
  }
  return result;
}

REGISTER_MLIR_INTERPRETER_OP(getBuffer);
REGISTER_MLIR_INTERPRETER_OP(own);
REGISTER_MLIR_INTERPRETER_OP(null);
REGISTER_MLIR_INTERPRETER_OP(retain);
REGISTER_MLIR_INTERPRETER_OP(freeAlloc);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
