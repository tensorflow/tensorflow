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

#include <cstdint>
#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kReplicationAttr[] = "mhlo.is_same_data_across_replicas";
constexpr char kMirroredVariableIndicesAttr[] = "_mirrored_variable_indices";

#define GEN_PASS_DEF_ANNOTATEPARAMETERREPLICATIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

// Analyzes the inputs to ClusterFuncOps in the module, and annotates their
// invoked functions whether each input has the same data across replicas.
struct AnnotateParameterReplicationPass
    : public impl::AnnotateParameterReplicationPassBase<
          AnnotateParameterReplicationPass> {
  void runOnOperation() override;
};

// Returns the first value in the chain of operands, which is not defined by a
// tf.IdentityOp or a tf.ReadVariableOp.
Value SkipIdentityAndReadVariable(Value v) {
  while (auto op = v.getDefiningOp()) {
    if (!isa<TF::IdentityOp, TF::ReadVariableOp>(op)) break;
    v = op->getOperand(0);
  }
  return v;
}

void AnnotateParameterReplicationPass::runOnOperation() {
  ModuleOp m = getOperation();
  OpBuilder builder(m.getContext());
  m.walk([&](tf_device::ClusterFuncOp cluster_func) {
    auto replicate = cluster_func->getParentOfType<tf_device::ReplicateOp>();
    if (!replicate) return;
    auto mirrored_variable_indices_attr =
        replicate->getAttrOfType<ArrayAttr>(kMirroredVariableIndicesAttr);
    llvm::SmallDenseSet<int64_t, 8> mirrored_replicate_args;
    if (mirrored_variable_indices_attr) {
      for (const auto& mirrored_index : mirrored_variable_indices_attr) {
        mirrored_replicate_args.insert(
            mlir::cast<IntegerAttr>(mirrored_index).getInt());
      }
    }
    auto func =
        llvm::cast<func::FuncOp>(m.lookupSymbol(cluster_func.getFunc()));
    for (auto entry : llvm::enumerate(cluster_func.getOperands())) {
      auto operand = SkipIdentityAndReadVariable(entry.value());
      auto block_arg = mlir::dyn_cast<BlockArgument>(operand);
      if (block_arg && block_arg.getOwner() == &replicate.GetBody()) {
        // Only mirrored args of ReplicateOp can be annotated.
        if (mirrored_replicate_args.count(block_arg.getArgNumber()) == 0) {
          continue;
        }
      } else if (!operand.getParentRegion()->isProperAncestor(
                     &replicate.getBody())) {
        // Not a replication-invariant operand.
        continue;
      }
      func.setArgAttr(entry.index(), kReplicationAttr,
                      builder.getBoolAttr(true));
    }
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateAnnotateParameterReplicationPass() {
  return std::make_unique<AnnotateParameterReplicationPass>();
}

}  // namespace TFDevice
}  // namespace mlir
