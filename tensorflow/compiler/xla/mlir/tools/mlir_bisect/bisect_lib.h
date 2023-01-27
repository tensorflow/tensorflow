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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TOOLS_MLIR_BISECT_BISECT_LIB_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_TOOLS_MLIR_BISECT_BISECT_LIB_H_

#include <functional>
#include <tuple>
#include <utility>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "tensorflow/compiler/xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"

#define REGISTER_MLIR_REDUCE_STRATEGY(name)                      \
  static int name##_init = []() {                                \
    ::mlir::bisect::detail::RegisterReduceStrategy(#name, name); \
    return 1;                                                    \
  }();

namespace mlir {
namespace bisect {

class BisectState {
 public:
  void SetTrace(mlir::interpreter::ExecutionTrace trace) {
    trace_ = std::move(trace);
  }

  // Returns all executions of the given op.
  llvm::SmallVector<const interpreter::InstructionTrace*> GetExecutions(
      mlir::Operation* op) const {
    return interpreter::FindOpExecutionsInTrace(trace_, op);
  }

 private:
  mlir::interpreter::ExecutionTrace trace_;
};

std::pair<OwningOpRef<ModuleOp>, Operation*> CloneModuleFor(Operation* op);
Operation* FindInClone(Operation* op, ModuleOp clone);

template <typename Op>
std::pair<OwningOpRef<ModuleOp>, Op> CloneModuleFor(Op op) {
  auto [module, op_clone] = CloneModuleFor(op.getOperation());
  return {std::move(module), llvm::cast<Op>(op_clone)};
}

namespace detail {

using CandidateVector = SmallVector<OwningOpRef<ModuleOp>>;

CandidateVector GetCandidates(
    const std::function<CandidateVector(BisectState&, Operation*)>& strategy,
    BisectState& state, ModuleOp op);

DenseMap<StringRef, std::function<CandidateVector(BisectState&, Operation*)>>&
GetStrategies();

// Registers a strategy that applies to all ops.
void RegisterReduceStrategy(
    StringRef name,
    std::function<CandidateVector(BisectState&, Operation*)> fn);

// Registers a strategy that applies to specific ops.
template <typename Op>
void RegisterReduceStrategy(StringRef name,
                            CandidateVector (*fn)(BisectState&, Op)) {
  RegisterReduceStrategy(
      name, [fn](BisectState& state, Operation* op) -> CandidateVector {
        if (auto cast = llvm::dyn_cast<Op>(op)) {
          return fn(state, cast);
        }
        return {};
      });
}

}  // namespace detail

}  // namespace bisect
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TOOLS_MLIR_BISECT_BISECT_LIB_H_
