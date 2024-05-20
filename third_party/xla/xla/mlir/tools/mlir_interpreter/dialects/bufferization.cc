/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project

// clang-format erroneously puts the Bufferization header above.
#include <algorithm>  // NOLINT
#include <cstdint>    // NOLINT
#include <optional>   // NOLINT

#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue ToTensor(InterpreterState&, bufferization::ToTensorOp,
                          const InterpreterValue& in) {
  return in.Clone();
}

InterpreterValue ToMemref(InterpreterState&, bufferization::ToMemrefOp,
                          const InterpreterValue& in) {
  return in;
}

InterpreterValue AllocTensor(
    InterpreterState&, bufferization::AllocTensorOp alloc,
    ArrayRef<int64_t> dynamic_sizes, std::optional<InterpreterValue> copy,
    const std::optional<InterpreterValue>& /*sizeHint*/) {
  auto ty = alloc->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = ReplaceDynamicVals(ty.getShape(), dynamic_sizes);

  if (copy) {
    return copy->Clone();
  }
  return InterpreterValue::MakeTensor(ty.getElementType(), shape);
}

InterpreterValue Clone(InterpreterState& state, bufferization::CloneOp,
                       const InterpreterValue& in) {
  if (auto* stats = state.GetOptions().stats) {
    stats->heap_size += in.GetBuffer()->GetByteSize();
    stats->peak_heap_size = std::max(stats->peak_heap_size, stats->heap_size);
    ++stats->num_allocations;
  }
  return in.Clone();
}

REGISTER_MLIR_INTERPRETER_OP(AllocTensor);
REGISTER_MLIR_INTERPRETER_OP(Clone);
REGISTER_MLIR_INTERPRETER_OP(ToMemref);
REGISTER_MLIR_INTERPRETER_OP(ToTensor);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
