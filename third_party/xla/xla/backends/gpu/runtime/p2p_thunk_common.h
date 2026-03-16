/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_P2P_THUNK_COMMON_H_
#define XLA_BACKENDS_GPU_RUNTIME_P2P_THUNK_COMMON_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Records the information for implementing CollectivePermute, Send and Recv.
struct P2PConfig {
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
    if (it != id_to_source_target.end()) {
      return it->second;
    }
    return SourceTargetMapEntry{};
  }

  CollectiveConfig config;
  IdToSourceTargetMap id_to_source_target;
};

// Extracts source/target pairs for send/recv from frontend attributes.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> GetSourceTargetPairs(
    mlir::DictionaryAttr frontend_attributes);

// Constructs the P2PConfig for an HLO Send or Recv instruction.
P2PConfig GetP2PConfigForSendRecv(const HloSendRecvInstruction* instr,
                                  const Shape& shape, int64_t replica_count,
                                  int64_t partition_count);

absl::StatusOr<const int64_t> GetCollectiveCurrentId(
    CollectiveParams* collective_params, const P2PConfig& config);

P2PConfigProto P2PConfigToProto(const P2PConfig& config);
absl::StatusOr<P2PConfig> P2PConfigFromProto(const P2PConfigProto& proto);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_P2P_THUNK_COMMON_H_
