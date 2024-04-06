/* Copyright 2022 The OpenXLA Authors.

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

#include <algorithm>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_ops.h"
#include "xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_ORDINALASSIGNMENT
#include "xla/mlir/runtime/transforms/passes.h.inc"

class OrdinalAssignmentPass
    : public impl::OrdinalAssignmentBase<OrdinalAssignmentPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//

void OrdinalAssignmentPass::runOnOperation() {
  llvm::SmallVector<ExportOp> assigned;    // ops with assigned ordinal
  llvm::SmallVector<ExportOp> unassigned;  // ops with unassigned ordinal

  ModuleOp module = getOperation();
  OpBuilder b(module);

  for (ExportOp op : module.getOps<ExportOp>()) {
    // Collect export ops without assigned ordinals.
    if (!op.ordinal()) {
      unassigned.push_back(op);
      continue;
    }

    unsigned ordinal = *op.ordinal();
    if (ordinal >= assigned.size()) assigned.resize(ordinal + 1);

    // Check that we do not have any duplicate exports.
    if (assigned[ordinal]) {
      op.emitError("duplicate exported function with ordinal ") << ordinal;
      return signalPassFailure();
    }

    assigned[ordinal] = op;
  }

  // Check that we have enough unassigned export ops to fill all ordinals.
  size_t num_holes = llvm::count_if(assigned, [](ExportOp op) { return !op; });

  if (unassigned.size() < num_holes) {
    module.emitError("can't fill all ordinals with exported functions");
    return signalPassFailure();
  }

  // Ordinals that must be filled first.
  llvm::SmallVector<unsigned> unassigned_ordinals;
  for (unsigned ordinal = 0; ordinal < assigned.size(); ++ordinal)
    if (!assigned[ordinal]) unassigned_ordinals.push_back(ordinal);

  // Reverse order of unassigned ordinals and operations to assign ordinals to
  // operations according to their order in the module.
  std::reverse(unassigned.begin(), unassigned.end());
  std::reverse(unassigned_ordinals.begin(), unassigned_ordinals.end());

  // Fill unassigned ordinals first.
  while (!unassigned_ordinals.empty()) {
    unsigned ordinal = unassigned_ordinals.pop_back_val();
    assigned[ordinal] = unassigned.pop_back_val();
    assigned[ordinal].setOrdinalAttr(b.getI32IntegerAttr(ordinal));
  }

  // Then assign new ordinals for the remaining export operations.
  while (!unassigned.empty()) {
    unsigned ordinal = assigned.size();
    assigned.emplace_back(unassigned.pop_back_val())
        .setOrdinalAttr(b.getI32IntegerAttr(ordinal));
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateOrdinalAssignmentPass() {
  return std::make_unique<OrdinalAssignmentPass>();
}

}  // namespace runtime
}  // namespace xla
