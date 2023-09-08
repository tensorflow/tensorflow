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

#include "thlo/IR/thlo_ops.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> scatter(InterpreterState& state,
                                            thlo::ScatterOp scatter,
                                            const InterpreterValue& indices,
                                            const InterpreterValue& updates,
                                            InterpreterValue init) {
  auto out = scatter.getNumResults() == 0 ? init : init.clone();

  const auto& inputView = init.view();
  const auto& updatesView = updates.view();
  int64_t operandRank = inputView.rank();

  for (const auto& updateIndices : updatesView.indices()) {
    llvm::SmallVector<int64_t> inputIndices(operandRank);
    llvm::SmallVector<int64_t> maxIndices(operandRank);
    llvm::SmallVector<int64_t> minIndices(operandRank);

    for (int64_t dim = 0; dim < operandRank; ++dim) {
      int64_t scatterIndex =
          indices.view().sizes[1] > dim
              ? indices.extractElement({updateIndices[0], dim}).asInt()
              : 0;
      inputIndices[dim] = updateIndices[dim + 1] + scatterIndex;
      minIndices[dim] = scatterIndex;
      maxIndices[dim] = updatesView.sizes[dim + 1] - 1 + scatterIndex;
    }

    if (!inputView.inBounds(minIndices)) continue;
    if (!inputView.inBounds(maxIndices)) continue;

    auto currentValue = out.extractElement(inputIndices);
    auto update = updates.extractElement(updateIndices);

    auto result = interpret(state, scatter.getUpdateComputation(),
                            {update, currentValue});
    if (state.hasFailure()) {
      break;
    }
    out.insertElement(inputIndices, result.front());
  }

  if (scatter.getNumResults() == 0) return {};
  return {out};
}

llvm::SmallVector<InterpreterValue> concatenate(
    InterpreterState&, thlo::ConcatenateOp op,
    ArrayRef<InterpreterValue> values, InterpreterValue init) {
  if (op.getNumResults() > 0) {
    init = init.clone();
  }
  int64_t dim = op.getDimension().getSExtValue();
  int64_t offset = 0;
  for (const auto& value : values) {
    for (auto index : value.view().indices()) {
      auto val = value.extractElement(index);
      index[dim] += offset;
      init.insertElement(index, val);
    }
    offset += value.view().sizes[dim];
  }
  if (op.getNumResults() > 0) return {init};
  return {};
}

llvm::SmallVector<InterpreterValue> reverse(InterpreterState& state,
                                            thlo::ReverseOp op,
                                            const InterpreterValue& in,
                                            InterpreterValue init) {
  // No bufferized thlo.reverse currently exists. This if guards against
  // possible future changes to that.
  if (op->getNumResults() != 1) {
    state.addFailure("bufferized thlo.reverse unsupported");
    return {};
  }

  init = init.clone();
  init.fill([&](ArrayRef<int64_t> indices) {
    auto srcIndex = llvm::to_vector(indices);
    for (int64_t dim : op.getReverseDimensions()) {
      srcIndex[dim] = in.view().sizes[dim] - 1 - srcIndex[dim];
    }
    return in.extractElement(srcIndex);
  });

  if (op->getNumResults() > 0) return {init};
  return {};
}

REGISTER_MLIR_INTERPRETER_OP("thlo.yield", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(concatenate);
REGISTER_MLIR_INTERPRETER_OP(scatter);
REGISTER_MLIR_INTERPRETER_OP(reverse);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
