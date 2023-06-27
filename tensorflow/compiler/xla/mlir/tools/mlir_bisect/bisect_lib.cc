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

#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"

#include <functional>
#include <iterator>
#include <tuple>
#include <utility>

#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace bisect {

Operation* FindInClone(Operation* op, ModuleOp clone) {
  if (llvm::isa<ModuleOp>(op)) {
    return clone;
  }

  auto* parent_clone = FindInClone(op->getParentOp(), clone);
  auto cloned_ops =
      parent_clone->getRegions()[op->getParentRegion()->getRegionNumber()]
          .getOps();
  for (auto [original_op, cloned_op] :
       llvm::zip(op->getParentRegion()->getOps(), cloned_ops)) {
    if (&original_op == op) {
      return &cloned_op;
    }
  }

  llvm_unreachable("Op not found in clone.");
}

std::pair<OwningOpRef<ModuleOp>, Operation*> CloneModuleFor(Operation* op) {
  auto module = op->getParentOfType<ModuleOp>().clone();
  return {OwningOpRef<ModuleOp>{module}, FindInClone(op, module)};
}

namespace detail {

DenseMap<StringRef, std::function<CandidateVector(BisectState&, Operation*)>>&
GetStrategies() {
  static auto* strategies =
      new DenseMap<StringRef,
                   std::function<CandidateVector(BisectState&, Operation*)>>();
  return *strategies;
}

void RegisterReduceStrategy(
    StringRef name,
    std::function<CandidateVector(BisectState&, Operation*)> fn) {
  GetStrategies()[name] = fn;
}

CandidateVector GetCandidates(
    const std::function<CandidateVector(BisectState&, Operation*)>& strategy,
    BisectState& state, ModuleOp op) {
  assert(strategy && "GetCandidates was passed a null strategy");
  CandidateVector result;
  op.lookupSymbol("main")->walk([&](Operation* subOp) {
    llvm::move(strategy(state, subOp), std::back_inserter(result));
  });
  return result;
}

}  // namespace detail
}  // namespace bisect
}  // namespace mlir
