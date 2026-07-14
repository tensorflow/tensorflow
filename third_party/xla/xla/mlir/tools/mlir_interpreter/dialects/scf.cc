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

#include "mlir/Dialect/SCF/IR/SCF.h"

#include <cassert>   // NOLINT
#include <cstdint>   // NOLINT
#include <iterator>  // NOLINT
#include <memory>    // NOLINT
#include <utility>   // NOLINT

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

class ParallelSideChannel : public InterpreterSideChannel {
 public:
  explicit ParallelSideChannel(llvm::SmallVector<InterpreterValue>& results)
      : results_(results) {}

  InterpreterValue& result(int index) const { return results_[index]; }

 private:
  SmallVector<InterpreterValue>& results_;
};

llvm::SmallVector<InterpreterValue> For(InterpreterState& state, scf::ForOp op,
                                        int64_t lb, int64_t ub, int64_t step,
                                        ArrayRef<InterpreterValue> inits) {
  llvm::SmallVector<InterpreterValue> results;
  for (int64_t i = 0; i < inits.size(); ++i) {
    results.push_back(GetInitOperand(op.getInitArgs(), i, inits));
  }

  auto& region = op->getRegion(0);
  for (; lb < ub; lb += step) {
    SmallVector<InterpreterValue> inputs;
    DispatchScalarType(op.getLowerBound().getType(), [&](auto dummy) {
      inputs.push_back(InterpreterValue{static_cast<decltype(dummy)>(lb)});
    });
    llvm::copy(results, std::back_inserter(inputs));
    results = Interpret(state, region, inputs);
    if (state.HasFailure()) {
      break;
    }
  }
  return results;
}

llvm::SmallVector<InterpreterValue> ForAll(
    InterpreterState& state, scf::ForallOp op,
    ArrayRef<InterpreterValue> dynamic_lower_bounds,
    ArrayRef<InterpreterValue> dynamic_upper_bounds,
    ArrayRef<InterpreterValue> dynamic_steps,
    ArrayRef<InterpreterValue> inits) {
  bool is_bufferized = op.getNumResults() == 0;

  // Clone any tensors that are passed in `shared_outs`.
  SmallVector<InterpreterValue> outs = llvm::to_vector(inits);
  for (auto [type, out] : llvm::zip(op.getOutputs().getTypes(), outs)) {
    if (isa<TensorType>(type)) {
      out = out.Clone();
    }
  }

  auto lbs = ReplaceDynamicVals(op.getStaticLowerBound(), dynamic_lower_bounds);
  auto ubs = ReplaceDynamicVals(op.getStaticUpperBound(), dynamic_upper_bounds);
  auto steps = ReplaceDynamicVals(op.getStaticStep(), dynamic_steps);

  SmallVector<int64_t> iter_sizes;
  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    if (step == 0) {
      state.AddFailure("invalid step");
      return {};
    }
    iter_sizes.push_back((ub - lb + (step - 1)) / step);
  }

  // Make a fake buffer view to abuse its index iterator.
  BufferView view{0, std::move(iter_sizes), {}};
  for (const auto& indices : view.Indices()) {
    SmallVector<InterpreterValue> args;
    for (auto [i, lb, step] : llvm::zip(indices, lbs, steps)) {
      args.push_back(InterpreterValue{i * step + lb});
    }
    llvm::copy(outs, std::back_inserter(args));

    auto yielded = Interpret(state, op->getRegion(0), args);
    if (state.HasFailure()) {
      break;
    }
    assert(yielded.empty() && "forall loop shouldn't have yielded anything");
  }

  if (is_bufferized) {
    return {};
  }
  return outs;
}

void InParallel(InterpreterState& state, scf::InParallelOp op) {
  Interpret(state, op.getRegion(), {});
}

llvm::SmallVector<InterpreterValue> If(InterpreterState& state, scf::IfOp op,
                                       bool condition) {
  if (condition) {
    return Interpret(state, op.getThenRegion(), {});
  }
  if (op.getElseRegion().hasOneBlock()) {
    return Interpret(state, op.getElseRegion(), {});
  }
  return {};
}

llvm::SmallVector<InterpreterValue> Parallel(InterpreterState& state,
                                             scf::ParallelOp parallel,
                                             ArrayRef<int64_t> lbs,
                                             ArrayRef<int64_t> ubs,
                                             ArrayRef<int64_t> steps,
                                             ArrayRef<InterpreterValue> inits) {
  llvm::SmallVector<InterpreterValue> results;
  for (int64_t i = 0; i < parallel.getNumReductions(); ++i) {
    results.push_back(GetInitOperand(parallel.getInitVals(), i, inits));
  }

  BufferView iter;
  for (auto [lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    iter.sizes.push_back((ub - lb + (step - 1)) / step);
  }

  // Make the results available to reduce ops.
  state.GetTopScope()->SetSideChannel(
      std::make_shared<ParallelSideChannel>(results));
  for (const auto& indices : iter.Indices()) {
    SmallVector<InterpreterValue> iter_args;
    for (auto [i, lb, step] : llvm::zip(indices, lbs, steps)) {
      iter_args.push_back(InterpreterValue{i * step + lb});
    }

    // Execute the region. It has no results.
    Interpret(state, parallel.getRegion(), iter_args);
  }

  return results;
}

void Reduce(InterpreterState& state, scf::ReduceOp reduce,
            ArrayRef<InterpreterValue> operands) {
  if (operands.size() != reduce.getNumRegions()) {
    state.AddFailure("reduce op has wrong number of operands");
    return;
  }
  for (int i = 0; i < reduce.getNumRegions(); ++i) {
    auto& accumulator =
        state.GetTopScope()->GetSideChannel<ParallelSideChannel>()->result(i);
    auto results =
        Interpret(state, reduce.getRegion(i), {accumulator, operands[i]});
    if (state.HasFailure()) {
      return;
    }
    accumulator = results.front();
  }
}

llvm::SmallVector<InterpreterValue> While(
    InterpreterState& state, scf::WhileOp op,
    MutableArrayRef<InterpreterValue> inits) {
  auto loop_vars = Interpret(state, op.getBefore(), inits);
  while (!state.HasFailure() && std::get<bool>(loop_vars.front().storage)) {
    loop_vars = Interpret(state, op.getAfter(),
                          ArrayRef<InterpreterValue>(loop_vars).drop_front());
    if (state.HasFailure()) {
      break;
    }
    loop_vars = Interpret(state, op.getBefore(), loop_vars);
  }
  if (state.HasFailure()) {
    return {};
  }
  loop_vars.erase(loop_vars.begin());
  return loop_vars;
}

REGISTER_MLIR_INTERPRETER_OP("scf.condition", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("scf.reduce.return", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP("scf.yield", NoOpTerminator);
REGISTER_MLIR_INTERPRETER_OP(ForAll);
REGISTER_MLIR_INTERPRETER_OP(InParallel);
REGISTER_MLIR_INTERPRETER_OP(Parallel);
REGISTER_MLIR_INTERPRETER_OP(Reduce);
REGISTER_MLIR_INTERPRETER_OP(For);
REGISTER_MLIR_INTERPRETER_OP(If);
REGISTER_MLIR_INTERPRETER_OP(While);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
