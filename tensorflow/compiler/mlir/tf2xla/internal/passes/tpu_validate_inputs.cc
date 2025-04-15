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
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/tpu_validate_inputs_utils.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

#define GEN_PASS_DEF_TPUVALIDATEINPUTSPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

constexpr char kXLAShardingAttr[] = "_XlaSharding";
constexpr char kShardingAttr[] = "sharding";

using mlir::dyn_cast;
using mlir::isa;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OperationPass;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::Type;
using mlir::TypeID;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::TF::AssertOp;
using mlir::TF::ConstOp;
using mlir::TF::kCompileDeviceTypeAttr;
using mlir::TF::kTpuReplicateAttr;
using mlir::TF::OutfeedEnqueueTupleOp;
using mlir::TF::PartitionedCallOp;
using mlir::TF::StatefulPartitionedCallOp;
using mlir::TF::TPUPartitionedCallOp;
using mlir::TF::TPUPartitionedInputOp;
using mlir::TF::TPUPartitionedInputV2Op;
using mlir::TF::TPUPartitionedOutputOp;
using mlir::TF::TPUPartitionedOutputV2Op;
using mlir::TF::TPUReplicatedInputOp;
using mlir::TF::TPUReplicatedOutputOp;
using mlir::TF::TPUReplicateMetadataOp;
using mlir::TF::WhileOp;
using mlir::TF::XlaSetDynamicDimensionSizeOp;
using mlir::tf_executor::FetchOp;
using mlir::tf_executor::GraphOp;
using mlir::tf_executor::IslandOp;
using mlir::tf_executor::YieldOp;

typedef std::unordered_map<std::string, TPUReplicateMetadataOp> MetadataMap;

struct TPUValidateInputsPass
    : public impl::TPUValidateInputsPassBase<TPUValidateInputsPass> {
  void runOnOperation() override;
};
bool IsTpuRegularOp(Operation* op) {
  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 32>* ops_set =
        new llvm::SmallDenseSet<mlir::TypeID, 32>{
            TypeID::get<ModuleOp>(),
            TypeID::get<GraphOp>(),
            TypeID::get<ReturnOp>(),
            TypeID::get<FuncOp>(),
            TypeID::get<YieldOp>(),
            TypeID::get<IslandOp>(),
            TypeID::get<TPUReplicatedInputOp>(),
            TypeID::get<TPUReplicatedOutputOp>(),
            TypeID::get<TPUPartitionedInputOp>(),
            TypeID::get<TPUPartitionedInputV2Op>(),
            TypeID::get<TPUPartitionedOutputOp>(),
            TypeID::get<TPUPartitionedOutputV2Op>(),
            TypeID::get<TPUReplicateMetadataOp>(),
            TypeID::get<FetchOp>(),
            TypeID::get<OutfeedEnqueueTupleOp>(),
        };
    return ops_set;
  }();
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return true;
  return ops->count(abstractOp->getTypeID()) == 0;
}

bool IsIntersectionXlaNonXlaOps(Operation* op) {
  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 32>* ops_set =
        new llvm::SmallDenseSet<mlir::TypeID, 32>{
            TypeID::get<ConstOp>(),
            TypeID::get<WhileOp>(),
            TypeID::get<AssertOp>(),
            TypeID::get<XlaSetDynamicDimensionSizeOp>(),
        };
    return ops_set;
  }();
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return true;
  return ops->count(abstractOp->getTypeID()) == 0;
}

bool IsPartitionedOp(Operation* op) {
  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 32>* ops_set =
        new llvm::SmallDenseSet<mlir::TypeID, 32>{
            TypeID::get<StatefulPartitionedCallOp>(),
            TypeID::get<PartitionedCallOp>(),
            TypeID::get<TPUPartitionedCallOp>(),
        };
    return ops_set;
  }();
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;
  return ops->count(abstractOp->getTypeID()) != 0;
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
  if (!op->hasAttr(kTpuReplicateAttr)) {
    op->emitOpError("TF2XLA TPU bridge input check: " + errormsg() +
                    "missing _tpu_replicate attr");
    return false;
  }
  auto opattr = op->getAttr(kTpuReplicateAttr);
  if (opattr != attr) {
    op->emitOpError("TF2XLA TPU bridge input check: " + errormsg() +
                    "invalid _tpu_replicate attr.")
        << " Expected attr: " << attr << ", Actual attr: " << opattr;
    return false;
  }
  return true;
}

bool ValidateReplicatedInput(TPUReplicatedInputOp rep, int num_replicas,
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
bool ValidateReplicatedOutput(TPUReplicatedOutputOp rep, int num_replicas,
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
bool ValidatePartitionedInput(TPUPartitionedInputOp rep,
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
bool ValidatePartitionedInputV2(TPUPartitionedInputV2Op rep,
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

bool CheckReplicatedIOOp(Operation* op, TPUReplicateMetadataOp metadata,
                         Operation* parent) {
  int num_replicas = metadata.getNumReplicas();
  int num_cores_per_replica = metadata.getNumCoresPerReplica();
  StringAttr tpu_replicate_attr =
      metadata->getAttrOfType<StringAttr>(kTpuReplicateAttr);
  if (auto repinput = dyn_cast<TPUReplicatedInputOp>(op)) {
    if (!ValidateReplicatedInput(repinput, num_replicas, tpu_replicate_attr))
      return false;
  }
  if (auto repoutput = dyn_cast<TPUReplicatedOutputOp>(op)) {
    if (!ValidateReplicatedOutput(repoutput, num_replicas, tpu_replicate_attr))
      return false;
  }
  if (auto partinput = dyn_cast<TPUPartitionedInputOp>(op)) {
    if (!ValidatePartitionedInput(partinput, num_cores_per_replica))
      return false;
  }
  if (auto partinput = dyn_cast<TPUPartitionedInputV2Op>(op)) {
    if (!ValidatePartitionedInputV2(partinput, num_cores_per_replica))
      return false;
  }
  if (auto partoutput = dyn_cast<TPUPartitionedOutputOp>(op)) {
    if (!ValidatePartitionedOutput(partoutput, num_cores_per_replica))
      return false;
  }
  if (auto partoutput = dyn_cast<TPUPartitionedOutputV2Op>(op)) {
    if (!ValidatePartitionedOutput(partoutput, num_cores_per_replica))
      return false;
  }
  return true;
}
// Checking op which is successor to a cluster op.
bool CheckClusterSuccessors(Operation* op, std::string cluster,
                            Operation* parent, MetadataMap& metadata_map) {
  std::string cluster_succ = "";
  if (op->hasAttr(kTpuReplicateAttr)) {
    cluster_succ = op->getAttrOfType<StringAttr>(kTpuReplicateAttr).str();
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
        << " with cluster = " << cluster_succ;
    return false;
  }
  return true;
}

// Checking op which is a predecessor to a non-cluster op.
bool CheckNonClusterSuccessors(Operation* op, Operation* parent,
                               MetadataMap& metadata_map) {
  if (!IsTpuRegularOp(op)) {
    if (isa<TPUReplicatedOutputOp>(op)) {
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
    if (isa<TPUReplicatedInputOp>(op)) {
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
  bool is_cluster_op = false;
  std::string cluster = "";
  if (op->hasAttr(kTpuReplicateAttr)) {
    cluster = op->getAttrOfType<StringAttr>(kTpuReplicateAttr).str();
    if (cluster.empty()) {
      op->emitOpError("TF2XLA TPU bridge input check: empty _tpu_replicate")
          << " attr for op = " << op->getName();
      return false;
    }
    is_cluster_op = true;
  }
  bool has_cluster_metadata =
      (metadata_map.find(cluster) != metadata_map.end());

  for (auto pred : GetPredecessors(op)) {
    if (is_cluster_op && !IsTpuRegularOp(pred) && has_cluster_metadata) {
      if (!CheckReplicatedIOOp(pred, metadata_map[cluster], op)) return false;
    }
    if (!is_cluster_op) {
      if (!CheckNonClusterPredecessors(pred, op, metadata_map)) return false;
    }
  }

  for (auto succ : GetSuccessors(op)) {
    if (is_cluster_op && !IsTpuRegularOp(succ) && has_cluster_metadata) {
      if (!CheckReplicatedIOOp(succ, metadata_map[cluster], op)) return false;
    }
    if (is_cluster_op && IsTpuRegularOp(succ)) {
      if (!CheckClusterSuccessors(succ, cluster, op, metadata_map))
        return false;
    }
    if (!is_cluster_op) {
      if (!CheckNonClusterSuccessors(succ, op, metadata_map)) return false;
    }
  }
  return true;
}

bool TypeMustBeNonXLA(const Type& type) {
  const Type elem = getElementTypeOrSelf(type);
  return !mlir::isa<mlir::TF::ResourceType>(elem) &&
         !tensorflow::TypeValidForXLA(type);
}

// Check if the op cannot be XLA compiled. If the op does not satisfy this
// criteria, then it is possible for the op to be XLA and non-XLA. But this
// function specifically checks if the op must be non-xla.
bool IsMustNotBeXlaOp(Operation* op) {
  for (auto& input : op->getOpOperands()) {
    if (TypeMustBeNonXLA(input.get().getType())) return true;
  }
  for (auto output_types : op->getResultTypes()) {
    if (TypeMustBeNonXLA(output_types)) return true;
  }
  return false;
}

// Check if the op must be compiled with XLA. If the op does not satisfy this
// critiria for "must be xla" then it is still possible for this op to be xla
// and non-xla as well. But below function specifically checks for the op to be
// only XLA op.
bool IsMustBeXlaOp(Operation* op, MetadataMap metadata_map) {
  // All PartitionedCall are inlined-out before XLA.
  // So MustBeXLA should return false
  if (IsPartitionedOp(op)) return false;
  if (!op->hasAttr(kTpuReplicateAttr)) return false;
  auto cluster = op->getAttrOfType<StringAttr>(kTpuReplicateAttr).str();
  if (metadata_map.find(cluster) == metadata_map.end()) return false;
  auto metadata = metadata_map[cluster];
  if (!metadata.getAllowSoftPlacement() &&
      !op->hasAttr(mlir::TF::kXlaOutsideCompilationAttr))
    return true;
  std::string device = "";
  if (op->hasAttr(mlir::TF::kDeviceAttr))
    device = op->getAttrOfType<StringAttr>(mlir::TF::kDeviceAttr).str();
  else
    return false;
  if (absl::StrContains(device, mlir::TF::kTpuDevice)) return true;
  return false;
}
bool ValidateIntersectionXlaNonXlaOps(Operation* op, MetadataMap metadata_map) {
  if (isa<TPUReplicateMetadataOp>(op) || isa<TPUReplicatedInputOp>(op) ||
      isa<TPUReplicatedOutputOp>(op) || isa<TPUPartitionedInputOp>(op) ||
      isa<TPUPartitionedInputV2Op>(op) || isa<TPUPartitionedOutputOp>(op) ||
      isa<TPUPartitionedOutputV2Op>(op))
    return true;
  if (IsMustBeXlaOp(op, metadata_map) && IsMustNotBeXlaOp(op)) {
    // TODO(b/269195256#comment19) change the warning for Identity op to error
    // when issue with input graph is resolved. Possible issue with python layer
    // inserting Identity op incorrectly.
    if (isa<mlir::TF::IdentityOp>(op)) {
      op->emitWarning("TF/XLA TPU bridge input check: found invalid op. ")
          << op->getName() << " can't be both xla and non-xla";
      return true;
    }
    op->emitOpError("TF/XLA TPU bridge input check: found invalid op. ")
        << "Can't be both xla and non-xla";
    return false;
  }
  return true;
}
void GetFlattenedShardings(llvm::SmallVector<xla::OpSharding>& shardings_result,
                           std::string shard_string) {
  xla::OpSharding sharding;
  if (shard_string.empty()) return;
  if (!sharding.ParseFromString(shard_string)) return;
  std::queue<xla::OpSharding> shardings;
  shardings.push(sharding);
  while (!shardings.empty()) {
    auto sharding_next = shardings.front();
    shardings.pop();
    if (sharding_next.type() == xla::OpSharding::TUPLE) {
      for (auto& child_sharding : sharding_next.tuple_shardings()) {
        shardings.push(child_sharding);
      }
    } else {
      shardings_result.push_back(sharding_next);
    }
  }
}

bool IsValidShardingTupleForArity(Operation* op) {
  if (!op->hasAttr(kXLAShardingAttr) && !op->hasAttr(kShardingAttr)) {
    return true;
  }
  std::string shard_string;
  if (op->hasAttr(kXLAShardingAttr)) {
    shard_string =
        op->getAttrOfType<StringAttr>(kXLAShardingAttr).strref().str();
  } else {
    shard_string = op->getAttrOfType<StringAttr>(kShardingAttr).strref().str();
  }
  xla::OpSharding sharding;
  if (!shard_string.empty() && sharding.ParseFromString(shard_string)) {
    // Only checking op with TUPLE sharding
    if (sharding.type() != xla::OpSharding::TUPLE) {
      return true;
    }
    // Each output is expected to have a corresponding sharding given by
    // tuple shardings. So, the no. of outputs (arity) should be same as
    // number of tuple shardings.
    if (sharding.tuple_shardings().size() != op->getNumResults()) {
      op->emitOpError(
          "TF2XLA TPU bridge input check: invalid no. of tuple shardings ")
          << sharding.tuple_shardings().size()
          << " for arity = " << op->getNumResults() << "\n The sharding is "
          << sharding.DebugString() << "\n";
      return false;
    }
  }
  return true;
}

bool IsValidMAXIMALSharding(Operation* op, MetadataMap& metadata_map) {
  if (!op->hasAttr(kTpuReplicateAttr)) return true;
  if (!op->hasAttr(kXLAShardingAttr) && !op->hasAttr(kShardingAttr)) {
    return true;
  }

  int num_cores_per_replica;
  // Assuming that the op is a IsTpuRegularOp and has a cluster metadata
  // for it. These checks are already performed in CheckOpsClusterIO.
  // Also assuming that if there is sharding, then there must be
  // cluster and the metadata corresponding to it.
  auto cluster = op->getAttrOfType<StringAttr>(kTpuReplicateAttr).str();
  if (cluster.empty()) {
    return true;
  }
  if (metadata_map.find(cluster) == metadata_map.end()) {
    return true;
  }
  num_cores_per_replica = metadata_map[cluster].getNumCoresPerReplica();
  std::optional<StringRef> shard_string;
  if (op->hasAttr(kXLAShardingAttr)) {
    shard_string = op->getAttrOfType<StringAttr>(kXLAShardingAttr).strref();
  } else {
    shard_string = op->getAttrOfType<StringAttr>(kShardingAttr).strref();
  }
  llvm::SmallVector<xla::OpSharding> shardings;
  GetFlattenedShardings(shardings, shard_string.value().str());

  for (auto& sharding : shardings) {
    if (sharding.type() != xla::OpSharding::MAXIMAL) {
      continue;
    }
    if (sharding.tile_assignment_devices_size() != 1) {
      op->emitOpError("TF/XLA TPU bridge input check: There must be ")
          << "exactly 1 device for MAXIMAL sharding."
          << " Number of devices assigned are "
          << sharding.tile_assignment_devices_size() << "\n";
      return false;
    } else {
      int sharding_device = sharding.tile_assignment_devices(0);
      if (sharding_device >= num_cores_per_replica || sharding_device < 0) {
        op->emitOpError(
            "TF2XLA TPU bridge input check: invalid sharding device ")
            << sharding_device
            << " for num_cores_per_replica = " << num_cores_per_replica
            << "\n The sharding is " << sharding.DebugString() << "\n";
        return false;
      }
    }
  }
  return true;
}

bool HasSingleCoreTpu(Operation* op) {
  if (auto compilation_attr =
          op->getAttrOfType<mlir::StringAttr>(kCompileDeviceTypeAttr)) {
    if (compilation_attr.getValue().str() == mlir::TF::kTpuDevice) {
      op->emitOpError(
          "TF2XLA TPU bridge input check: found a single-core TPU graph");
      return true;
    }
  }
  return false;
}

void TPUValidateInputsPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool success = true;
  int num_metadata = 0;
  TPUReplicateMetadataOp metadata;
  MetadataMap metadata_map;
  module.walk([&](TPUReplicateMetadataOp meta) {
    ++num_metadata;
    metadata = meta;
    metadata_map[meta->getAttrOfType<StringAttr>(kTpuReplicateAttr).str()] =
        meta;
  });

  getOperation().walk([&](mlir::Operation* op) {
    if (IsPotentialUnsupportedOp(op)) {
      LOG(WARNING) << "Potential unsupported op: "
                   << op->getName().getStringRef().str()
                   << ". TF2XLA MLIR bridge does not guarantee to support it.";
    }

    if (IsTpuRegularOp(op)) {
      success &= CheckOpsClusterIO(op, metadata_map);
    }
    if (IsIntersectionXlaNonXlaOps(op)) {
      success &= ValidateIntersectionXlaNonXlaOps(op, metadata_map);
    }
    if (IsTpuRegularOp(op)) {
      success &= IsValidMAXIMALSharding(op, metadata_map);
      success &= IsValidShardingTupleForArity(op);
    }
    success &= !HasSingleCoreTpu(op);

    if (!success) {
      signalPassFailure();
    }
  });

  module.walk([&](GraphOp graph) {
    if (HasV1ControlFlow(graph)) {
      LOG(WARNING) << "TF2XLA MLIR bridge does not support v1 control flow."
                   << " Use at your own risk.";
    }
    if (!success) {
      signalPassFailure();
    }
  });
}

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUValidateInputsPass() {
  return std::make_unique<TPUValidateInputsPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
