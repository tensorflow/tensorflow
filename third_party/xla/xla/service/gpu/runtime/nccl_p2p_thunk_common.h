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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_P2P_THUNK_COMMON_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_P2P_THUNK_COMMON_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

// Count the number of times a Send or Recv instruction executed on a device.
class ExecutionCounters {
 public:
  absl::Status Initialize(se::StreamExecutor* executor, RunId run_id);
  absl::StatusOr<int64_t*> GetCounter(se::StreamExecutor* executor,
                                      RunId run_id);

 private:
  using CounterKey = std::pair<se::StreamExecutor*, RunId>;
  absl::Mutex mu_;
  // TODO(b/338288906): may need to clean up the counters for finished runs.
  absl::flat_hash_map<CounterKey, int64_t> counters_ ABSL_GUARDED_BY(mu_);
};

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

  enum class ValidationKind { kValid = 0, kInvalid = 1, kConditional = 2 };

  using SourceTargetToBounds = absl::flat_hash_map<std::pair<int64_t, int64_t>,
                                                   std::pair<int64_t, int64_t>>;

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
  ValidationKind validation_kind = ValidationKind::kValid;
  // When a Send or Recv has validation_kind = ValidationKind::kConditional,
  // record the valid execution numbers as a pair of [lower-bound, upper-bound]
  // for each source and target pair.
  SourceTargetToBounds source_target_to_bounds;
};

// Extracts source/target pairs for send/recv from frontend attributes.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> GetSourceTargetPairs(
    mlir::DictionaryAttr frontend_attributes);

// Constructs the NcclP2PConfig for an HLO Send or Recv instruction.
NcclP2PConfig GetNcclP2PConfigForSendRecv(const HloSendRecvInstruction* instr,
                                          const Shape& shape,
                                          int64_t replica_count,
                                          int64_t partition_count);
// Returns the stream kind for the asynchronous stream used to execute an HLO
// Send or Recv instruction, by inspecting the frontend attributes of the
// instruction.
AsyncStreamKind GetStreamKindForSendRecv(const HloSendRecvInstruction* instr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_P2P_THUNK_COMMON_H_
