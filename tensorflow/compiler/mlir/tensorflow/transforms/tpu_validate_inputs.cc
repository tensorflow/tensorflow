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
#include <memory>
#include <string>

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

struct TPUValidateInputsPass
    : public impl::TPUValidateInputsPassBase<TPUValidateInputsPass> {
  void runOnOperation() override;
};
bool IsTpuReplicateAttrOps(Operation* op) {
  return !(isa<mlir::tf_executor::YieldOp>(op) ||
           isa<mlir::tf_executor::IslandOp>(op) ||
           isa<TF::TPUReplicatedInputOp>(op) ||
           isa<TF::TPUReplicatedOutputOp>(op) ||
           isa<TF::TPUPartitionedInputOp>(op) ||
           isa<TF::TPUPartitionedInputV2Op>(op) ||
           isa<TF::TPUPartitionedOutputOp>(op) ||
           isa<TF::TPUPartitionedOutputV2Op>(op) ||
           isa<mlir::tf_executor::FetchOp>(op));
}
bool CheckTpuReplicateAttr(Operation* op, StringAttr attr,
                           std::string errormsg) {
  if (!op->hasAttr(TF::kTpuReplicateAttr)) {
    op->emitOpError("TF/XLA TPU bridge input check: " + errormsg +
                    "missing _tpu_replicate attr");
    return false;
  }
  auto opattr = op->getAttr(TF::kTpuReplicateAttr);
  if (opattr != attr) {
    op->emitOpError("TF/XLA TPU bridge input check: " + errormsg +
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
        "TF/XLA TPU bridge input check: packed with number of inputs not 1.")
        << " num_replicas=" << num_replicas << " no. of inputs=" << arity;
    return false;
  } else if (!rep.getIsPacked() && arity != num_replicas) {
    rep.emitOpError(
        "TF/XLA TPU bridge input check: number of inputs inconsistent.")
        << " num_replicas=" << num_replicas << " no. of inputs=" << arity;
    return false;
  }
  auto repparent = rep->getParentOp();
  for (auto& use : repparent->getOpResults().getUses()) {
    auto op = use.getOwner();
    if (!IsTpuReplicateAttrOps(op)) continue;
    std::string errormsg = rep->getName().getStringRef().str() +
                           " op has successor op " +
                           op->getName().getStringRef().str() + " with error: ";
    if (!CheckTpuReplicateAttr(op, attr, errormsg)) return false;
  }
  return true;
}
bool ValidateReplicatedOutput(TF::TPUReplicatedOutputOp rep, int num_replicas,
                              StringAttr attr) {
  int arity = rep.getOutputs().size();
  if (arity != num_replicas) {
    rep.emitOpError(
        "TF/XLA TPU bridge input check: number of outputs inconsistent.")
        << " num_replicas=" << num_replicas << " no. of outputs=" << arity;
    return false;
  }
  auto opparent = rep.getInput().getDefiningOp();
  WalkResult result = opparent->walk([&](mlir::Operation* op) {
    if (!IsTpuReplicateAttrOps(op)) return WalkResult::advance();
    std::string errormsg = rep->getName().getStringRef().str() +
                           " op has predecessor op " +
                           op->getName().getStringRef().str() + " with error: ";
    if (!CheckTpuReplicateAttr(op, attr, errormsg))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return false;
  return true;
}
bool ValidatePartitionedInput(TF::TPUPartitionedInputOp rep,
                              int num_cores_per_replica) {
  int arity = rep.getInputs().size();
  if (arity != num_cores_per_replica) {
    rep.emitOpError(
        "TF/XLA TPU bridge input check: number of inputs inconsistent.")
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
        "TF/XLA TPU bridge input check: packed with number of inputs not 1.")
        << " num_cores_per_replicas=" << num_cores_per_replica
        << " no. of inputs=" << arity;
    return false;
  } else if (!rep.getIsPacked() && arity != num_cores_per_replica) {
    rep.emitOpError(
        "TF/XLA TPU bridge input check: number of inputs inconsistent.")
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
        "TF/XLA TPU bridge input check: number of outputs inconsistent.")
        << " num_cores_per_replica=" << num_cores_per_replica
        << " no. of outputs=" << arity;
    return false;
  }
  return true;
}
void TPUValidateInputsPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool success = true;
  int num_metadata = 0;
  TF::TPUReplicateMetadataOp metadata;
  module.walk([&](TF::TPUReplicateMetadataOp meta) {
    ++num_metadata;
    metadata = meta;
  });
  // TODO(b/269195256): support multi-TPUReplicateMetadata case.
  // Currently handling case with one metadata op / cluster. Further CLs will
  // address cases with multi-TPUReplicatedMetadata.
  if (num_metadata == 1) {
    int num_replicas = metadata.getNumReplicas();
    int num_cores_per_replica = metadata.getNumCoresPerReplica();
    StringAttr tpu_replicate_attr =
        metadata->getAttrOfType<StringAttr>(TF::kTpuReplicateAttr);
    module.walk([&](mlir::Operation* op) {
      if (auto repinput = dyn_cast<TF::TPUReplicatedInputOp>(op)) {
        success &=
            ValidateReplicatedInput(repinput, num_replicas, tpu_replicate_attr);
      }
      if (auto repoutput = dyn_cast<TF::TPUReplicatedOutputOp>(op)) {
        success &= ValidateReplicatedOutput(repoutput, num_replicas,
                                            tpu_replicate_attr);
      }
      if (auto partinput = dyn_cast<TF::TPUPartitionedInputOp>(op)) {
        success &= ValidatePartitionedInput(partinput, num_cores_per_replica);
      }
      if (auto partinput = dyn_cast<TF::TPUPartitionedInputV2Op>(op)) {
        success &= ValidatePartitionedInputV2(partinput, num_cores_per_replica);
      }
      if (auto partoutput = dyn_cast<TF::TPUPartitionedOutputOp>(op)) {
        success &= ValidatePartitionedOutput(partoutput, num_cores_per_replica);
      }
      if (auto partoutput = dyn_cast<TF::TPUPartitionedOutputV2Op>(op)) {
        success &= ValidatePartitionedOutput(partoutput, num_cores_per_replica);
      }
    });
  }
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
