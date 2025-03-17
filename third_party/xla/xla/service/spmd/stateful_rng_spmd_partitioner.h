/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_STATEFUL_RNG_SPMD_PARTITIONER_H_
#define XLA_SERVICE_SPMD_STATEFUL_RNG_SPMD_PARTITIONER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace xla {
namespace spmd {

class StatefulRngSpmdPartitioningVisitor
    : public spmd::SpmdPartitioningVisitor {
 public:
  StatefulRngSpmdPartitioningVisitor(
      HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
      const spmd::SPMDCollectiveOpsCreator& collective_ops_creator,
      int64_t* next_channel_id, spmd::SpmdLogger* logger,
      spmd::SpmdPartitionerOptions options, spmd::SpmdPartitioner* partitioner,
      const CallGraph& call_graph)
      : spmd::SpmdPartitioningVisitor(computation, num_partitions, num_replicas,
                                      collective_ops_creator, next_channel_id,
                                      logger, std::move(options), partitioner,
                                      call_graph) {}
  absl::Status HandleRngGetAndUpdateState(HloInstruction* hlo) override;
};

class StatefulRngSpmdPartitioner : public spmd::SpmdPartitioner {
 public:
  StatefulRngSpmdPartitioner(
      int64_t num_partitions, int64_t num_replicas,
      int64_t threshold_for_windowed_einsum_mib = 100000,
      bool windowed_einsum_use_multiple_streams = false,
      bool skip_checking_windowed_einsum_users = false,
      bool disable_ag_rewrite_for_multiple_consumers = false,
      std::optional<int64_t> total_bytes_windowed_einsum_threshold =
          std::nullopt)
      : spmd::SpmdPartitioner(
            num_partitions, num_replicas,
            GetSpmdPartitionerOptions(threshold_for_windowed_einsum_mib,
                                      windowed_einsum_use_multiple_streams,
                                      skip_checking_windowed_einsum_users,
                                      disable_ag_rewrite_for_multiple_consumers,
                                      total_bytes_windowed_einsum_threshold)) {}

 protected:
  std::unique_ptr<spmd::SpmdPartitioningVisitor> CreateVisitor(
      HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
      const spmd::SPMDCollectiveOpsCreator& collective_ops_creator,
      int64_t* next_channel_id, spmd::SpmdLogger* logger,
      spmd::SpmdPartitionerOptions options,
      const CallGraph& call_graph) override;

  absl::Status PreprocessSharding(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // This adds an unsafe attribute labelling the while loop as a pipelined
  // while loop. This attribute lets the rest of the passes ignore the
  // computations in the pipeline bubble.
  absl::Status HandleRotateRightWhilePreprocessing(
      HloComputation* computation) override;
  bool CanSideEffectingHaveReplicatedSharding(
      const HloInstruction* hlo) override;

 private:
  static spmd::SpmdPartitionerOptions GetSpmdPartitionerOptions(
      int64_t threshold_for_windowed_einsum_mib,
      bool windowed_einsum_use_multiple_streams = false,
      bool skip_checking_windowed_einsum_users = false,
      bool disable_ag_rewrite_for_multiple_consumers = false,
      std::optional<int64_t> total_bytes_windowed_einsum_threshold =
          std::nullopt) {
    spmd::SpmdPartitionerOptions options;
    options.allow_module_signature_change = true;
    options.threshold_for_windowed_einsum_mib =
        threshold_for_windowed_einsum_mib;
    options.unroll_windowed_einsum = windowed_einsum_use_multiple_streams;
    options.skip_checking_windowed_einsum_users =
        skip_checking_windowed_einsum_users;
    options.disable_ag_rewrite_for_multiple_consumers =
        disable_ag_rewrite_for_multiple_consumers;
    options.total_bytes_windowed_einsum_threshold =
        total_bytes_windowed_einsum_threshold;
    return options;
  }
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_STATEFUL_RNG_SPMD_PARTITIONER_H_
