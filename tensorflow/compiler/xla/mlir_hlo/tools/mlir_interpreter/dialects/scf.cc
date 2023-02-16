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

#include "mlir/Dialect/SCF/IR/SCF.h"

#include <iterator>  // NOLINT
#include <memory>    // NOLINT
#include <utility>   // NOLINT

#include "llvm/ADT/SmallVector.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

class ParallelSideChannel : public InterpreterSideChannel {
 public:
  ParallelSideChannel(
      llvm::SmallVector<InterpreterValue>& results,
      const llvm::DenseMap<scf::ReduceOp, int64_t>& reduceOpIndices)
      : results(results), reduceOpIndices(reduceOpIndices) {}

  InterpreterValue& result(scf::ReduceOp op) const {
    return results[reduceOpIndices.find(op)->second];
  }

 private:
  SmallVector<InterpreterValue>& results;
  const llvm::DenseMap<scf::ReduceOp, int64_t>& reduceOpIndices;
};

llvm::SmallVector<InterpreterValue> scfFor(InterpreterState& state,
                                           scf::ForOp op, int64_t lb,
                                           int64_t ub, int64_t step,
                                           ArrayRef<InterpreterValue> inits) {
  llvm::SmallVector<InterpreterValue> results;
  for (int64_t i = 0; i < inits.size(); ++i) {
    results.push_back(getInitOperand(op.getInitArgs(), i, inits));
  }

  auto& region = op->getRegion(0);
  for (; lb < ub; lb += step) {
    SmallVector<InterpreterValue> inputs{{lb}};
    llvm::copy(results, std::back_inserter(inputs));
    results = interpret(state, region, inputs);
    if (state.hasFailure()) break;
  }
  return results;
}

llvm::SmallVector<InterpreterValue> scfIf(InterpreterState& state, scf::IfOp op,
                                          bool condition) {
  if (condition) {
    return interpret(state, op.getThenRegion(), {});
  }
  if (op.getElseRegion().hasOneBlock()) {
    return interpret(state, op.getElseRegion(), {});
  }
  return {};
}

llvm::SmallVector<InterpreterValue> parallel(InterpreterState& state,
                                             scf::ParallelOp parallel,
                                             ArrayRef<int64_t> lbs,
                                             ArrayRef<int64_t> ubs,
                                             ArrayRef<int64_t> steps,
                                             ArrayRef<InterpreterValue> inits) {
  llvm::SmallVector<InterpreterValue> results;
  for (int64_t i = 0; i < parallel.getNumReductions(); ++i) {
    results.push_back(getInitOperand(parallel.getInitVals(), i, inits));
  }

  BufferView iter;
  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    iter.sizes.push_back((ub - lb + (step - 1)) / step);
  }

  llvm::DenseMap<scf::ReduceOp, int64_t> reduceOps;
  for (auto& subOp : parallel.getBody()->getOperations()) {
    if (auto reduce = llvm::dyn_cast<scf::ReduceOp>(subOp)) {
      int64_t index = reduceOps.size();
      reduceOps[reduce] = index;
    }
  }

  assert(reduceOps.size() == results.size() &&
         "expected equal number of reduce ops and results");

  // Make the results available to reduce ops.
  state.getTopScope()->setSideChannel(
      std::make_shared<ParallelSideChannel>(results, reduceOps));
  for (const auto& indices : iter.indices()) {
    SmallVector<InterpreterValue> iterArgs;
    for (auto [i, lb, step] : llvm::zip(indices, lbs, steps)) {
      iterArgs.push_back(InterpreterValue{i * step + lb});
    }

    // Execute the region. It has no results.
    interpret(state, parallel.getRegion(), iterArgs);
  }

  return results;
}

void reduce(InterpreterState& state, scf::ReduceOp reduce,
            const InterpreterValue& operand) {
  auto& accumulator =
      state.getTopScope()->getSideChannel<ParallelSideChannel>()->result(
          reduce);
  // TODO(jreiffers): Is this the correct order?
  auto results = interpret(state, reduce.getRegion(), {accumulator, operand});
  if (!state.hasFailure()) {
    accumulator = results.front();
  }
}

llvm::SmallVector<InterpreterValue> scfWhile(
    InterpreterState& state, scf::WhileOp op,
    MutableArrayRef<InterpreterValue> inits) {
  auto loopVars = interpret(state, op.getBefore(), inits);
  while (!state.hasFailure() && std::get<bool>(loopVars.front().storage)) {
    loopVars = interpret(state, op.getAfter(),
                         ArrayRef<InterpreterValue>(loopVars).drop_front());
    if (state.hasFailure()) break;
    loopVars = interpret(state, op.getBefore(), loopVars);
  }
  if (state.hasFailure()) return {};
  loopVars.erase(loopVars.begin());
  return loopVars;
}

REGISTER_MLIR_INTERPRETER_OP("scf.condition", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("scf.reduce.return", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("scf.yield", noOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(parallel);
REGISTER_MLIR_INTERPRETER_OP(reduce);
REGISTER_MLIR_INTERPRETER_OP(scfFor);
REGISTER_MLIR_INTERPRETER_OP(scfIf);
REGISTER_MLIR_INTERPRETER_OP(scfWhile);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
