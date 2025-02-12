/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/cf_sink/pass.h"

#include <memory>

#include "absl/log/log.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/ControlFlowSinkUtils.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_CONTROLFLOWSINK
#include "tensorflow/core/transforms/passes.h.inc"

class ControlFlowSinkPass
    : public impl::ControlFlowSinkBase<ControlFlowSinkPass> {
 public:
  // Initialize the pass by getting a cached identifier to the name attribute.
  LogicalResult initialize(MLIRContext *context) override {
    name_id_ =
        context->getOrLoadDialect<TFGraphDialect>()->getNameAttrIdentifier();
    return success();
  }

  // Move the operation to the start of the entry block. Rename it if necessary.
  void moveAndRename(Operation *op, Region *region);

  void runOnOperation() override;

 private:
  // Cached name identifier.
  StringAttr name_id_;
};
}  // namespace

static bool IsStateless(Operation *op) {
  if (auto registry = dyn_cast<TensorFlowRegistryInterface>(op))
    return !registry.isStateful();
  return false;
}

// Don't sink TPU-specific ops and ops with regions.
static bool IsExcluded(Operation *op) {
  // TODO(b/228618345) Ops with `i32` operands cannot be safely sunk due to a
  // potential placement bug on GPU.

  // Don't sink ops with regions as it can create nested regions so deep that
  // the verifier is stack overflowed.
  if (op->getNumRegions()) {
    return true;
  }

  // TPU ops cannot be moved, even though they are marked as stateless.
  // TODO(jeffniu): TPU ops should be marked in some other way.
  StringRef op_name = op->getName().stripDialect();
  return op_name == "TPUReplicateMetadata" || op_name == "TPUReplicatedInput" ||
         op_name == "TPUReplicatedOutput" ||
         op_name == "TPUCompilationResult" || op_name == "_TPUReplicate";
}

void ControlFlowSinkPass::moveAndRename(Operation *op, Region *region) {
  op->moveBefore(&region->front(), region->front().begin());
  auto name = op->getAttrOfType<StringAttr>(name_id_);
  auto parent_name = region->getParentOp()->getAttrOfType<StringAttr>(name_id_);
  if (!name || !parent_name) return;
  op->setAttr(name_id_, StringAttr::get(op->getContext(),
                                        name.getValue() + "_tfg_cf_sunk_" +
                                            parent_name.getValue()));
}

void ControlFlowSinkPass::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();
  getOperation()->walk([&](RegionBranchOpInterface branch) {
    SmallVector<Region *> regions;
    getSinglyExecutedRegionsToSink(branch, regions);
    num_sunk += controlFlowSink(
        regions, domInfo,
        /*shouldMoveIntoRegion=*/
        [&](Operation *op, Region *) {
          return IsStateless(op) && !IsExcluded(op);
        },
        /*moveIntoRegion=*/
        [&](Operation *op, Region *region) { moveAndRename(op, region); });
  });
  VLOG(1) << "tfg-cf-sink num-sunk: " << num_sunk;
}

std::unique_ptr<Pass> CreateControlFlowSinkPass() {
  return std::make_unique<ControlFlowSinkPass>();
}

}  // namespace tfg
}  // namespace mlir
