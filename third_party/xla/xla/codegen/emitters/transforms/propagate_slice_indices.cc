/* Copyright 2024 The OpenXLA Authors.

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
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/transforms/passes.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_PROPAGATESLICEINDICESPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

class PropagateSliceIndicesPass
    : public impl::PropagateSliceIndicesPassBase<PropagateSliceIndicesPass> {
 public:
  void runOnOperation() override;
};

void PropagateSliceIndicesPass::runOnOperation() {
  mlir::func::FuncOp entry;
  for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func->getAttr("xla.entry")) {
      entry = func;
      break;
    }
  }

  if (!entry) {
    getOperation()->emitOpError("No entry function found.");
    signalPassFailure();
    return;
  }

  for (auto func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getNumArguments() == 0 || func == entry) {
      continue;
    }

    for (int i = 0; i < func.getNumArguments(); ++i) {
      if (mlir::isa<mlir::RankedTensorType>(func.getArgument(i).getType())) {
        if (auto index = entry.getArgAttr(i, "xla.slice_index")) {
          func.setArgAttr(i, "xla.slice_index", index);
        }
        if (auto invariant = entry.getArgAttr(i, "xla.invariant")) {
          func.setArgAttr(i, "xla.invariant", invariant);
        }
      } else {
        break;
      }
    }
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> CreatePropagateSliceIndicesPass() {
  return std::make_unique<PropagateSliceIndicesPass>();
}

}  // namespace emitters
}  // namespace xla
