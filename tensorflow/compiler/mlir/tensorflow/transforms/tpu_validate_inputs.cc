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
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_TPUVALIDATEINPUTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

typedef std::unordered_map<std::string, TF::TPUReplicateMetadataOp> MetadataMap;

struct TPUValidateInputsPass
    : public impl::TPUValidateInputsPassBase<TPUValidateInputsPass> {
  void runOnOperation() override;
};
bool IsTpuRegularOp(Operation* op) {
  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 32>* ops_set =
        new llvm::SmallDenseSet<mlir::TypeID, 32>{
            TypeID::get<mlir::ModuleOp>(),
            TypeID::get<mlir::tf_executor::GraphOp>(),
            TypeID::get<mlir::func::ReturnOp>(),
            TypeID::get<mlir::func::FuncOp>(),
            TypeID::get<mlir::tf_executor::YieldOp>(),
            TypeID::get<mlir::tf_executor::IslandOp>(),
            TypeID::get<TF::StatefulPartitionedCallOp>(),
            TypeID::get<TF::PartitionedCallOp>(),
            TypeID::get<TF::TPUPartitionedCallOp>(),
            TypeID::get<TF::TPUReplicatedInputOp>(),
            TypeID::get<TF::TPUReplicatedOutputOp>(),
            TypeID::get<TF::TPUPartitionedInputOp>(),
            TypeID::get<TF::TPUPartitionedInputV2Op>(),
            TypeID::get<TF::TPUPartitionedOutputOp>(),
            TypeID::get<TF::TPUPartitionedOutputV2Op>(),
            TypeID::get<TF::TPUReplicateMetadataOp>(),
            TypeID::get<mlir::tf_executor::FetchOp>(),
        };
    return ops_set;
  }();
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return true;
  return ops->count(abstractOp->getTypeID()) == 0;
}

bool IsPartitionedCall(Operation* op) {
  return (isa<TF::PartitionedCallOp>(op) || isa<TF::TPUPartitionedCallOp>(op) ||
          isa<TF::StatefulPartitionedCallOp>(op));
}
// Gets the successors of an op wrapped in a tf_executor.island.
llvm::SmallVector<Operation*> GetSuccessors(Operation* op) {
  llvm::SmallVector<Operation*> successors;
  for (auto result : op->getParentOp()->getOpResults()) {
    for (auto& use : result.getUses()) {
      auto succ = use.getOwner();
      successors.push_back(succ);
    }
  }
  return successors;
}
// Gets the predecessors of an op wrapped in tf_executor.island.
llvm::SmallVector<Operation*> GetPredecessors(Operation* op) {
  llvm::SmallVector<Operation*> predecessors;
  for (auto operand : op->getOperands()) {
    if (Operation* pred = operand.getDefiningOp()) {
      pred->walk([&](mlir::Operation* opinexecutor) {
        predecessors.push_back(opinexecutor);
      });
    }
  }
  return predecessors;
}
bool CheckTpuReplicateAttr(Operation* op, StringAttr attr,
                           std::function<std::string()> errormsg) {
  if (!op->hasAttr("_tpu_replicate")) {
    op->emitOpError("TF2XLA TPU bridge input check: " + errormsg() +
                    "missing _tpu_replicate attr");
    return false;
  }
  auto opattr = op->getAttr(TF::kTpuReplicateAttr);
  if (opattr != attr) {
    op->emitOpError("TF2XLA TPU bridge input check: " + errormsg() +
                    "invalid _tpu_replicate attr.")
        << " Expected attr: " << attr << ", Actual attr: " << opattr;
    return false;
  }
  return true;
}

bool ValidateReplicatedInput(TF::TPUReplicatedInputOp rep, int num_replicas,
                             StringAttr attr) {
  int arity = rep.getInputs().size();
  if (rep.getIsPacked() && arity != 1) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: packed with number of inputs not 1.")
        << " num_replicas=" << num_replicas << " no. of inputs=" << arity;
    return false;
  } else if (!rep.getIsPacked() && arity != num_replicas) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: number of inputs inconsistent.")
        << " num_replicas=" << num_replicas << " no. of inputs=" << arity;
    return false;
  }
  for (auto& succ : GetSuccessors(rep)) {
    if (!IsTpuRegularOp(succ)) continue;
    auto errormsg = [&]() -> std::string {
      return rep->getName().getStringRef().str() + " op has successor op " +
             succ->getName().getStringRef().str() + " with error: ";
    };
    if (!CheckTpuReplicateAttr(succ, attr, errormsg)) return false;
  }
  return true;
}
bool ValidateReplicatedOutput(TF::TPUReplicatedOutputOp rep, int num_replicas,
                              StringAttr attr) {
  int arity = rep.getOutputs().size();
  if (arity != num_replicas) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: number of outputs inconsistent.")
        << " num_replicas=" << num_replicas << " no. of outputs=" << arity;
    return false;
  }
  for (auto& pred : GetPredecessors(rep)) {
    if (!IsTpuRegularOp(pred)) continue;
    auto errormsg = [&]() -> std::string {
      return rep->getName().getStringRef().str() + " op has predecessor op " +
             pred->getName().getStringRef().str() + " with error: ";
    };
    if (!CheckTpuReplicateAttr(pred, attr, errormsg)) return false;
  }
  return true;
}
bool ValidatePartitionedInput(TF::TPUPartitionedInputOp rep,
                              int num_cores_per_replica) {
  int arity = rep.getInputs().size();
  if (arity != num_cores_per_replica) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: number of inputs inconsistent.")
        << " num_cores_per_replica=" << num_cores_per_replica
        << " no. of inputs=" << arity;
    return false;
  }
  return true;
}
bool ValidatePartitionedInputV2(TF::TPUPartitionedInputV2Op rep,
                                int num_cores_per_replica) {
  int arity = rep.getInputs().size();
  if (rep.getIsPacked() && arity != 1) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: packed with number of inputs not 1.")
        << " num_cores_per_replicas=" << num_cores_per_replica
        << " no. of inputs=" << arity;
    return false;
  } else if (!rep.getIsPacked() && arity != num_cores_per_replica) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: number of inputs inconsistent.")
        << " num_cores_per_replica=" << num_cores_per_replica
        << " no. of inputs=" << arity;
    return false;
  }
  return true;
}
template <typename T>
bool ValidatePartitionedOutput(T rep, int num_cores_per_replica) {
  int arity = rep.getOutput().size();
  if (arity != num_cores_per_replica) {
    rep.emitOpError(
        "TF2XLA TPU bridge input check: number of outputs inconsistent.")
        << " num_cores_per_replica=" << num_cores_per_replica
        << " no. of outputs=" << arity;
    return false;
  }
  return true;
}

bool CheckReplicatedOp(Operation* op, std::string cluster,
                       MetadataMap& metadata_map, Operation* parent) {
  if (metadata_map.find(cluster) == metadata_map.end()) {
    op->emitOpError(
        "TF2XLA TPU bridge input check: no tf.TPUReplicateMetadata op")
        << " with cluster = " << cluster
        << " The Parent op = " << parent->getName();
    return false;
  }
  auto metadata = metadata_map[cluster];
  int num_replicas = metadata.getNumReplicas();
  int num_cores_per_replica = metadata.getNumCoresPerReplica();
  StringAttr tpu_replicate_attr =
      metadata->getAttrOfType<StringAttr>(TF::kTpuReplicateAttr);
  if (auto repinput = dyn_cast<TF::TPUReplicatedInputOp>(op)) {
    if (!ValidateReplicatedInput(repinput, num_replicas, tpu_replicate_attr))
      return false;
  }
  if (auto repoutput = dyn_cast<TF::TPUReplicatedOutputOp>(op)) {
    if (!ValidateReplicatedOutput(repoutput, num_replicas, tpu_replicate_attr))
      return false;
  }
  if (auto partinput = dyn_cast<TF::TPUPartitionedInputOp>(op)) {
    if (!ValidatePartitionedInput(partinput, num_cores_per_replica))
      return false;
  }
  if (auto partinput = dyn_cast<TF::TPUPartitionedInputV2Op>(op)) {
    if (!ValidatePartitionedInputV2(partinput, num_cores_per_replica))
      return false;
  }
  if (auto partoutput = dyn_cast<TF::TPUPartitionedOutputOp>(op)) {
    if (!ValidatePartitionedOutput(partoutput, num_cores_per_replica))
      return false;
  }
  if (auto partoutput = dyn_cast<TF::TPUPartitionedOutputV2Op>(op)) {
    if (!ValidatePartitionedOutput(partoutput, num_cores_per_replica))
      return false;
  }
  return true;
}
// Checking op which is successor to a cluster op.
bool CheckClusterSuccessors(Operation* op, std::string cluster,
                            Operation* parent, MetadataMap& metadata_map) {
  if (!IsTpuRegularOp(op)) {
    return CheckReplicatedOp(op, cluster, metadata_map, parent);
  }
  std::string cluster_succ = "";
  if (op->hasAttr(TF::kTpuReplicateAttr)) {
    cluster_succ = op->getAttrOfType<StringAttr>(TF::kTpuReplicateAttr).str();
  }
  if (cluster_succ.empty()) {
    // TODO (b/269195256#comment16): Change to error after resolving issue
    // with test. Will fix it after the upstream code is fixed.
    op->emitWarning("TF2XLA TPU bridge input check: cluster op = ")
        << parent->getName() << " with cluster = " << cluster
        << " has successor as non cluster op " << op->getName();
    return true;
  }
  if (cluster != cluster_succ) {
    op->emitOpError(
        "TF2XLA TPU bridge input check: mismatch clusters tpu_replicate "
        "attr. Parent op ")
        << parent->getName() << " with cluster = " << cluster
        << " has successor cluster op " << op->getName()
        << "with cluster = " << cluster_succ;
    return false;
  }
  return true;
}
// Checking op which is a predecessor to a cluster op.
bool CheckClusterPredecessors(Operation* op, std::string cluster,
                              Operation* parent, MetadataMap& metadata_map) {
  if (!IsTpuRegularOp(op)) {
    return CheckReplicatedOp(op, cluster, metadata_map, parent);
  }
  return true;
}
// Checking op which is a predecessor to a non-cluster op.
bool CheckNonClusterSuccessors(Operation* op, Operation* parent,
                               MetadataMap& metadata_map) {
  if (!IsTpuRegularOp(op)) {
    if (isa<TF::TPUReplicatedOutputOp>(op)) {
      op->emitOpError("TF2XLA TPU bridge input check: non-cluster op = ")
          << parent->getName()
          << " has invalid successor op = " << op->getName();
      return false;
    } else {
      return true;
    }
  }
  return true;
}
// Checking op which is a successor to a non-cluster op.
bool CheckNonClusterPredecessors(Operation* op, Operation* parent,
                                 MetadataMap& metadata_map) {
  if (!IsTpuRegularOp(op)) {
    if (isa<TF::TPUReplicatedInputOp>(op)) {
      op->emitOpError("TF2XLA TPU bridge input check: non-cluster op = ")
          << parent->getName()
          << " has invalid predecessor op = " << op->getName();
      return false;
    } else {
      return true;
    }
  }
  return true;
}

bool CheckOpsClusterIO(Operation* op, MetadataMap& metadata_map) {
  if (op->hasAttr("_tpu_replicate")) {
    std::string cluster =
        op->getAttrOfType<StringAttr>(TF::kTpuReplicateAttr).str();
    if (cluster.empty()) {
      op->emitOpError("TF2XLA TPU bridge input check: empty _tpu_replicate")
          << " attr for op = " << op->getName();
      return false;
    }
    for (auto pred : GetPredecessors(op)) {
      if (!CheckClusterPredecessors(pred, cluster, op, metadata_map))
        return false;
    }
    for (auto succ : GetSuccessors(op)) {
      if (!CheckClusterSuccessors(succ, cluster, op, metadata_map))
        return false;
    }
  } else {
    for (auto pred : GetPredecessors(op)) {
      if (!CheckNonClusterPredecessors(pred, op, metadata_map)) return false;
    }
    for (auto succ : GetSuccessors(op)) {
      if (!CheckNonClusterSuccessors(succ, op, metadata_map)) return false;
    }
  }
  return true;
}

void TPUValidateInputsPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool success = true;
  int num_metadata = 0;
  TF::TPUReplicateMetadataOp metadata;
  MetadataMap metadata_map;
  module.walk([&](TF::TPUReplicateMetadataOp meta) {
    ++num_metadata;
    metadata = meta;
    metadata_map[meta->getAttrOfType<StringAttr>(TF::kTpuReplicateAttr).str()] =
        meta;
  });
  bool found_partitioned_call = false;
  getOperation().walk([&](mlir::Operation* op) {
    if (IsPartitionedCall(op)) found_partitioned_call = true;
  });
  if (found_partitioned_call) return;

  getOperation().walk([&](mlir::Operation* op) {
    if (IsTpuRegularOp(op)) {
      success &= CheckOpsClusterIO(op, metadata_map);
    }
  });
  if (!success) {
    signalPassFailure();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUValidateInputsPass() {
  return std::make_unique<TPUValidateInputsPass>();
}

}  // namespace TFTPU
}  // namespace mlir
