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

#ifndef XLA_SERVICE_PROFILE_GUIDED_LATENCY_ESTIMATOR_H_
#define XLA_SERVICE_PROFILE_GUIDED_LATENCY_ESTIMATOR_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {

// Helper class enabling gathering of statistics (such as missing instruction
// from the profile) for PGLE.
class ProfileStatisticsAggregator {
 public:
  struct Statistics {
    int found_instructions_count;
    absl::flat_hash_set<const HloInstruction*>& missing_instructions;
  };

  virtual ~ProfileStatisticsAggregator() = default;

  // Handler for the missing instruction cost.
  virtual void HandleMissingInstructionCost(
      const HloInstruction& instruction) = 0;

  // Handler for found instruction cost.
  virtual void HandleFoundInstructionCost(
      const HloInstruction& instruction) = 0;

  // Handler for the missing latency info between `from` and `to`.
  virtual void HandleMissingInstructionLatency(const HloInstruction& from,
                                               const HloInstruction& to) = 0;

  // Handler for found latency info between `from` and `to`.
  virtual void HandleFoundInstructionLatency(const HloInstruction& from,
                                             const HloInstruction& to) = 0;

  // Returns gathered statistics summary.
  Statistics GetStats();

 protected:
  absl::flat_hash_set<const HloInstruction*> missing_instructions_;
  int found_instructions_count_ = 0;
};

// Implementation of LatencyEstimator using a profile to estimate HLO cost and
// latencies between instructions. If a cost is not known, it will forward to
// an underlying estimator.
class ProfileGuidedLatencyEstimator : public LatencyEstimator {
 public:
  ProfileGuidedLatencyEstimator(
      const SchedulerConfig& config,
      std::unique_ptr<LatencyEstimator> latency_estimator,
      const tensorflow::profiler::ProfiledInstructionsProto& proto,
      std::unique_ptr<ProfileStatisticsAggregator> aggregator = nullptr);

  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override;
  TimeCost NodeCost(const HloInstruction* instr) const override;
  int CyclesPerMicrosecond() const override {
    return latency_estimator_->CyclesPerMicrosecond();
  }

  // Checks whether `module` has all the respective instructions present in the
  // profile grabbed by this object.
  //
  // Returns absl::OkStatus if accuracy check passes,
  // `absl::InvalidArgumentError` does not pass and
  // `absl::FailedPreconditionError` if `aggregator_` is not provided.
  absl::Status CheckAccuracy(const HloModule& module);

 private:
  const SchedulerConfig config_;
  std::unique_ptr<LatencyEstimator> latency_estimator_;

  // Profile info pertaining to a single instruction.
  struct ProfileInfo {
    std::optional<TimeCost> cost;
    // Latencies to other instruction with this instruction as source.
    absl::flat_hash_map<std::string, TimeCost> latencies;
  };
  absl::flat_hash_map<std::string, ProfileInfo> instr_map_;
  // Aggregator gathering data about missed/found instructions.
  std::unique_ptr<ProfileStatisticsAggregator> aggregator_;
};

}  // namespace xla

#endif  // XLA_SERVICE_PROFILE_GUIDED_LATENCY_ESTIMATOR_H_
