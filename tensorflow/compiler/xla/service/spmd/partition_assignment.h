/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_PARTITION_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_PARTITION_ASSIGNMENT_H_

#include <cstdint>
#include <memory>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Base class for the partitioning algorithm. The derived classes will implement
// different partitioning algorithms using various heuristics and cost models.
// The aim is to create HLO shardings with small costs.
class PartitioningAlgorithm {
 public:
  // The kind/type/name of the (derived) algorithm.
  enum class AlgorithmKind {
    kNoop,
    kExp0,
    kExp1,
    kExp2,
  };

  // Constructors and destructor.
  PartitioningAlgorithm() = delete;
  PartitioningAlgorithm(const PartitioningAlgorithm&) = delete;
  PartitioningAlgorithm& operator=(const PartitioningAlgorithm&) = delete;
  virtual ~PartitioningAlgorithm() = default;

  // Factory method to create a Noop partitioning algorithm.
  static std::unique_ptr<PartitioningAlgorithm> CreateNoopPartitioning(
      int64_t num_partitions);

  // Returns the kind of this algorithm.
  const AlgorithmKind& kind() const;

  // Returns the name of this algorithm.
  absl::string_view name() const;

  // Returns the number of shards/partitions.
  int64_t num_partitions() const;

  // Assigns shardings to the given module.
  virtual StatusOr<bool> Run(HloModule* module) const = 0;

 protected:
  // Internal constructor for a given algorithm kind. Other fields must be
  // filled by factory methods.
  explicit PartitioningAlgorithm(AlgorithmKind kind, int64_t num_partitions);

 private:
  // Kind for this algorithm.
  AlgorithmKind kind_ = AlgorithmKind::kNoop;

  // Number of requested shards (parts), i.e., number of available devices.
  int64_t num_partitions_;
};

// Noop algorithm is essentially 'algorithm 0'.
class NoopPartitioning : public PartitioningAlgorithm {
 public:
  explicit NoopPartitioning(int64_t num_partitions);

  // Assigns shardings to the given module.
  StatusOr<bool> Run(HloModule* module) const override;
};

// PartitionAssignment assigns sharding annotations to some HLOs in the given
// module. The HLOs to target are more important/costly than the others in terms
// of certain metrics. The plan is to find and assign good sharding annotations
// to those HLOs in this pass and let the sharding propagation pass propagate
// those to the remaining HLOs. The current assumption is that the module does
// not have any sharding annotations yet.
class PartitionAssignment : public HloModulePass {
 public:
  explicit PartitionAssignment(int64_t num_partitions);

  // Returns the name of the pass.
  absl::string_view name() const override;

  // Returns the PartitioningAlgorithm to be used by PartitionAssignment.
  virtual std::unique_ptr<PartitioningAlgorithm> ChoosePartitioningAlgorithm(
      const HloModule& module) const;

  // Runs the pass.
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Returns the algorithm being used.
  const PartitioningAlgorithm* algorithm();

  // Returns the number of partitions.
  int64_t num_partitions() const;

 private:
  // The partitioning algorithm to be used. For now, it is determined by a flag.
  std::unique_ptr<PartitioningAlgorithm> algorithm_ = nullptr;

  // The number of partitions (shards) being requested.
  int64_t num_partitions_;
};

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_PARTITION_ASSIGNMENT_H_
