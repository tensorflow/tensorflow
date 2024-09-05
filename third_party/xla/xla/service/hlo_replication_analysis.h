/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_REPLICATION_ANALYSIS_H_
#define XLA_SERVICE_HLO_REPLICATION_ANALYSIS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla_data.pb.h"

namespace xla {

// An HLO pass that determines whether each instruction in the module outputs
// the same value across replicas or across partitions (depending on the value
// `cross_partition_spmd`). It propagates sources of replicated values to
// the rest of the module, where sources include cross-replica-sum, annotated
// entry parameters, and constants.
class HloReplicationAnalysis {
 public:
  // Runs the analysis on module and returns the result or an error.
  static absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>> Run(
      const HloModule* module, bool cross_partition_spmd);

  // Same as above, but the caller can provide additional annotations: a set of
  // while loops that are known to have the same iteration counts across
  // replicas or partitions.
  static absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>> Run(
      const HloModule* module, bool cross_partition_spmd,
      const absl::flat_hash_set<const HloInstruction*>*
          loops_known_with_same_iterations);

  // Same as above but supports finding partially replicated HLOs.
  static absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
  RunWithPartialReplication(const HloModule* module, bool cross_partition_spmd);

  // Returns if the HLO instruction outputs the same value (i.e., replicated) at
  // the given index across all replicas or partitions.
  bool HloInstructionIsReplicatedAt(const HloInstruction* inst,
                                    const ShapeIndex& index) const;

  bool HloInstructionIsReplicatedAt(
      const HloInstruction* inst, const ShapeIndex& index,
      absl::Span<const ReplicaGroup> replica_groups) const;

 private:
  // A data structure that represents how an HLO is replicated among a set of
  // devices. Device ID could be either partition ID or replica ID.
  // We represent partial replication by grouping devices that have the same
  // value into the same set.
  class HloReplication {
   public:
    static HloReplication ReplicatedOnAllDevices();
    static HloReplication UniqueOnAllDevices();
    static HloReplication PartiallyReplicated(
        absl::Span<const absl::Span<const int64_t>> device_sets);
    HloReplication();
    HloReplication(const HloReplication& other) = default;
    HloReplication(HloReplication&& other) = default;
    HloReplication& operator=(HloReplication&& other) = default;
    HloReplication Merge(const HloReplication& other) const;
    bool Equal(const HloReplication& other) const;
    bool IsReplicatedOnAllDevices() const;
    bool IsUniqueOnAllDevices() const;
    bool IsReplicatedWithinSubgroup(absl::Span<const int64_t> device_ids) const;
    std::string ToString() const;

   private:
    enum class State {
      kReplicatedOnAllDevices = 0,
      kUniqueOnAllDevices = 1,
      kPartiallyReplicated = 2,
    };
    explicit HloReplication(State state,
                            absl::Span<const int64_t> device_set_root);
    State state_;
    // Empty if state_ is kReplicatedOnAllDevices or kUniqueOnAllDevices.
    // Otherwise, its size equals to the number of devices (either partitions
    // or replications). Maps each device ID to the smallest device ID in the
    // set.
    std::vector<int64_t> device_set_root_;
  };

  static HloReplication DetermineHloInstructionIsReplicated(
      const HloInstruction* hlo, const ShapeIndex& index,
      bool cross_partition_spmd,
      const absl::flat_hash_map<const HloInstruction*,
                                ShapeTree<HloReplication>>& hlo_replication,
      bool support_partial_replication);

  HloReplicationAnalysis(const HloModule* module, bool cross_partition_spmd,
                         const absl::flat_hash_set<const HloInstruction*>*
                             loops_known_with_same_iterations,
                         bool support_partial_replication)
      : module_(module),
        cross_partition_spmd_(cross_partition_spmd),
        loops_known_with_same_iterations_(*loops_known_with_same_iterations),
        support_partial_replication_(support_partial_replication) {}

  // Computes hlo_replication_.
  absl::Status ComputeHloReplication();

  // A helper function to recursively compute hlo_replication on a computation.
  // Returns whether hlo_replication_ is changed.
  bool ComputeHloReplicationOnComputation(const HloComputation* computation,
                                          bool mark_everything_not_replicated);

  const HloModule* module_;

  // If true, run this replication analysis for replicated values across
  // partitions (not across replicas) on an SPMD partitioned module. This means
  // that HloInstructionIsReplicatedAt() returns true if the value is identical
  // across partitions for each replica. The module-level parameter and root
  // instructions may have HloSharding attributes that indicate whether values
  // are identical across partitions.
  //
  // If false, HloReplicationAnalysis runs across replicas.
  bool cross_partition_spmd_;

  // A set of while loops that are known to have the same iteration counts
  // across replicas or partitions. This is provided by the caller as additional
  // annotations.
  const absl::flat_hash_set<const HloInstruction*>&
      loops_known_with_same_iterations_;

  const bool support_partial_replication_;

  // A map from each analyzed HLO instruction to a shape tree that represents
  // whether the instruction outputs the same value across replicas or
  // partitions at each shape index.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<HloReplication>>
      hlo_replication_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_REPLICATION_ANALYSIS_H_
