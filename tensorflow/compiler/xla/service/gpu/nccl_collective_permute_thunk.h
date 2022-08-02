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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_PERMUTE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_PERMUTE_THUNK_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct NcclCollectivePermuteConfig {
  // During a collective permute, every node optionally sends its data to
  // another node (including possibly itself) and received data from another
  // node. For each node, remember who it receives data from (source) and who
  // it send data to (target). Either are optional.
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

// Thunk that performs a NCCL-based collective permute.
class NcclCollectivePermuteThunk : public NcclCollectiveThunk {
 public:
  static NcclCollectivePermuteConfig GetNcclCollectivePermuteConfig(
      mlir::lmhlo::CollectivePermuteOp op, int64_t replica_count,
      int64_t partition_count);

  NcclCollectivePermuteThunk(ThunkInfo thunk_info,
                             mlir::lmhlo::CollectivePermuteOp op,
                             int64_t replica_count, int64_t partition_count,
                             const Buffer& buffer);

  // Returns whether the given instruction can be lowered to a nccl collective
  // permute thunk.
  static bool CanImplement(mlir::lmhlo::CollectivePermuteOp op);

  static const char* GetName() { return "CollectivePermute"; }
  static bool IsDegenerate(mlir::lmhlo::CollectivePermuteOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo::CollectivePermuteOp op) {
    return GetCollectiveOpGroupMode(op.getChannelId().has_value(), std::nullopt)
        .ValueOrDie();
  }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

  const NcclCollectiveConfig& config() const override { return config_.config; }

 private:
  const NcclCollectivePermuteConfig config_;
  const Buffer buffer_;
};

Status RunCollectivePermute(
    NcclCollectivePermuteConfig::SourceTargetMapEntry source_target,
    DeviceBufferPair& buffer, se::Stream& stream, ncclComm_t comm,
    absl::string_view device_string, int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_PERMUTE_THUNK_H_
