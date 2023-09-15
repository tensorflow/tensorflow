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

#ifndef XLA_SERVICE_GPU_NCCL_P2P_THUNK_COMMON_H_
#define XLA_SERVICE_GPU_NCCL_P2P_THUNK_COMMON_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/nccl_collective_thunk.h"

namespace xla {
namespace gpu {

// Records the information for implementing CollectivePermute, Send and Recv.
struct NcclP2PConfig {
  // Record the target ID for sending a data and the source ID from which to
  // receive a data. Either target or source can be optional.
  struct SourceTargetMapEntry {
    std::optional<int64_t> source;
    std::optional<int64_t> target;
  };

  using IdToSourceTargetMap =
      absl::flat_hash_map<int64_t, SourceTargetMapEntry>;

  // Returns the source and target ID corresponding to the given ID (these IDs
  // are replica_ids for cross replica permute or partition_ids for cross
  // partition permute). The source ID is the id which will send data to this
  // ID and the target ID is the id to which this ID will send its data. Either
  // can be optional.
  static SourceTargetMapEntry GetSourceTarget(
      const IdToSourceTargetMap& id_to_source_target, int64_t id) {
    auto it = id_to_source_target.find(id);
    if (it != id_to_source_target.end()) return it->second;
    return SourceTargetMapEntry{};
  }

  NcclCollectiveConfig config;
  IdToSourceTargetMap id_to_source_target;
};

// Extracts source/target pairs for send/recv from frontend attributes.
StatusOr<std::vector<std::pair<int64_t, int64_t>>> GetSourceTargetPairs(
    mlir::DictionaryAttr frontend_attributes);

// Returns the GroupMode for Send and Recv.
template <typename OpT>
std::enable_if_t<std::is_same_v<OpT, mlir::lmhlo::SendOp> ||
                     std::is_same_v<OpT, mlir::lmhlo::RecvOp>,
                 CollectiveOpGroupMode>
GetGroupModeForSendRecv(OpT op) {
  return GetCollectiveOpGroupMode(op.getChannelHandle().getHandle() > 1,
                                  std::nullopt)
      .value();
}

// Constructs the NcclP2PConfig for Send and Recv.
template <typename OpT>
std::enable_if_t<std::is_same_v<OpT, mlir::lmhlo::SendOp> ||
                     std::is_same_v<OpT, mlir::lmhlo::RecvOp>,
                 NcclP2PConfig>
GetNcclP2PConfigForSendRecv(OpT op, int64_t replica_count,
                            int64_t partition_count) {
  NcclP2PConfig p2p_config;
  auto& config = p2p_config.config;

  config.operand_count = 1;
  const Shape shape = GetShape(op.getOperand(0));
  config.operand_element_type.push_back(shape.element_type());

  const int64_t channel_id = op.getChannelHandle().getHandle();
  config.group_mode = GetGroupModeForSendRecv(op);
  // Emulate SetCollectiveOpKindAndID.
  // Send and Recv ops have a non-optional channel id while collective-permute
  // has an optional channel id. We use 0 to encode the send/recv transformed
  // from collective-permute without a channel id.
  if (channel_id >= 1) {
    config.collective_op_kind = RendezvousKey::kCrossModule;
    config.op_id = channel_id;
  } else {
    config.collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    config.op_id = static_cast<int64_t>(unique_id.getInt());
  }

  // All execution instances of a send/recv together form a replica group.
  const int64_t num_participants =
      config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? replica_count
          : partition_count;
  config.replica_groups.emplace_back();
  ReplicaGroup& replica_group = config.replica_groups.front();
  for (int i = 0; i < num_participants; ++i) {
    replica_group.add_replica_ids(i);
  }

  auto source_target_pairs = GetSourceTargetPairs(op.getFrontendAttributes());
  TF_CHECK_OK(source_target_pairs.status());
  for (const std::pair<int64_t, int64_t>& source_target :
       *source_target_pairs) {
    int64_t source = source_target.first;
    int64_t target = source_target.second;

    p2p_config.id_to_source_target.insert({target, {}}).first->second.source =
        source;
    p2p_config.id_to_source_target.insert({source, {}}).first->second.target =
        target;
  }

  return p2p_config;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NCCL_P2P_THUNK_COMMON_H_
