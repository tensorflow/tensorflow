/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/spmd/partition_assignment.h"

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/xla.pb.h"

namespace xla {

PartitioningAlgorithm::PartitioningAlgorithm(AlgorithmKind kind,
                                             int64_t num_partitions) {
  kind_ = kind;
  CHECK_GT(num_partitions, 1) << "Number of partitions must be at least two.";
  num_partitions_ = num_partitions;
}

absl::string_view PartitioningAlgorithm::name() const {
  switch (kind_) {
    case AlgorithmKind::kNoop:
    default:
      return "Noop";
  }
}

const PartitioningAlgorithm::AlgorithmKind& PartitioningAlgorithm::kind()
    const {
  return kind_;
}

int64_t PartitioningAlgorithm::num_partitions() const {
  return num_partitions_;
}

/* static */ std::unique_ptr<PartitioningAlgorithm>
PartitioningAlgorithm::CreateNoopPartitioning(int64_t num_partitions) {
  return std::make_unique<NoopPartitioning>(num_partitions);
}

NoopPartitioning::NoopPartitioning(int64_t num_partitions)
    : PartitioningAlgorithm(AlgorithmKind::kNoop, num_partitions) {
  VLOG(2) << "Created a no-op algorithm with the number of partitions: "
          << num_partitions;
}

absl::StatusOr<bool> NoopPartitioning::Run(HloModule* module) const {
  VLOG(2) << "No-op algorithm was called to partition module: "
          << module->name();
  return false;
}

PartitionAssignment::PartitionAssignment(int64_t num_partitions) {
  CHECK_GT(num_partitions, 1) << "Number of partitions must be at least two.";
  num_partitions_ = num_partitions;
}

absl::string_view PartitionAssignment::name() const {
  return "partitioning-assignment";
}

const PartitioningAlgorithm* PartitionAssignment::algorithm() {
  return algorithm_.get();
}

int64_t PartitionAssignment::num_partitions() const { return num_partitions_; }

std::unique_ptr<PartitioningAlgorithm>
PartitionAssignment::ChoosePartitioningAlgorithm(
    const HloModule& module) const {
  // Create the partitioning algorithm based on the flag. In the future we can
  // decide on the algorithm by analyzing the module.
  auto algo = module.config().debug_options().xla_partitioning_algorithm();
  CHECK_EQ(algo, DebugOptions::PARTITIONING_ALGORITHM_NOOP);
  return PartitioningAlgorithm::CreateNoopPartitioning(num_partitions());
}

absl::StatusOr<bool> PartitionAssignment::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Running partition assignment on module " << module->name();

  // Choose partitioning algorithm.
  algorithm_ = ChoosePartitioningAlgorithm(*module);

  // Run the algorithm.
  return algorithm()->Run(module);
}

}  // namespace xla
