/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {

namespace {

// This pass removes Identity/IdentityN ops from the TPU computation and
// reachable functions.
// TODO(lyandy): Remove this pass once resource op lifting is migrated to use
// resource alias analysis and support region based control flow. Removing
// Identity ops may remove `_XlaSharding` annotation attribute if Identity ops
// are used to propagate such information.

struct TPUIdentityPruning
    : public TF::TPUIdentityPruningPassBase<TPUIdentityPruning> {
  void runOnOperation() override;
};

// Collects all reachable functions (via call ops) from a given region.
SmallVector<FuncOp, 4> CollectReachableFunctions(Region& region) {
  llvm::SmallPtrSet<FuncOp, 4> reachable_funcs;

  auto collect_reachable_funcs =
      [&reachable_funcs](Region& src, SmallVectorImpl<FuncOp>& funcs_to_visit) {
        src.walk([&reachable_funcs, &funcs_to_visit](CallOpInterface call_op) {
          auto func = dyn_cast_or_null<FuncOp>(call_op.resolveCallable());
          if (func && reachable_funcs.insert(func).second)
            funcs_to_visit.push_back(func);
        });
      };

  SmallVector<FuncOp, 4> funcs_to_visit;
  collect_reachable_funcs(region, funcs_to_visit);

  while (!funcs_to_visit.empty()) {
    SmallVector<FuncOp, 4> new_funcs_to_visit;
    for (FuncOp func_to_visit : funcs_to_visit) {
      if (!func_to_visit.getCallableRegion()) continue;
      collect_reachable_funcs(*func_to_visit.getCallableRegion(),
                              new_funcs_to_visit);
    }
    funcs_to_visit.swap(new_funcs_to_visit);
  }

  return llvm::to_vector<4>(reachable_funcs);
}

// Removes Identity/IdentityN ops from a region and forwards its operands to its
// results.
void RemoveIdentityFromRegion(Region& region) {
  region.walk([](Operation* op) {
    if (isa<TF::IdentityOp, TF::IdentityNOp>(op)) {
      op->replaceAllUsesWith(op->getOperands());
      op->erase();
    }
  });
}

void TPUIdentityPruning::runOnOperation() {
  SmallVector<tf_device::ClusterOp, 4> clusters;
  getOperation().walk(
      [&](tf_device::ClusterOp cluster) { clusters.push_back(cluster); });

  for (tf_device::ClusterOp cluster : clusters) {
    RemoveIdentityFromRegion(cluster.body());
    auto reachable_funcs = CollectReachableFunctions(cluster.body());
    for (FuncOp reachable_func : reachable_funcs)
      RemoveIdentityFromRegion(*reachable_func.getCallableRegion());
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUIdentityPruningPass() {
  return std::make_unique<TPUIdentityPruning>();
}

}  // namespace TFTPU
}  // namespace mlir
