/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_H_
#define XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/interpolator.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class CollectiveInterpolator {
 public:
  struct FallbackInterpolatorKey {
    HloOpcode opcode;
    GPUCommunicationType communication_type;

    template <typename H>
    friend H AbslHashValue(H h, const FallbackInterpolatorKey& key) {
      return H::combine(std::move(h), key.opcode, key.communication_type);
    }

    bool operator==(const FallbackInterpolatorKey& other) const {
      return opcode == other.opcode &&
             communication_type == other.communication_type;
    }
  };

  struct ExactInterpolatorKey {
    HloOpcode opcode;
    CollectiveDeviceList device_list;
    std::optional<PrimitiveType> data_type;

    template <typename H>
    friend H AbslHashValue(H h, const ExactInterpolatorKey& key) {
      return H::combine(
          std::move(h), key.opcode,
          key.device_list.ToString(/*print_full_replica_group_list=*/true),
          key.data_type);
    }

    bool operator==(const ExactInterpolatorKey& other) const {
      return opcode == other.opcode &&
             device_list.ToString(/*print_full_replica_group_list=*/true) ==
                 other.device_list.ToString(
                     /*print_full_replica_group_list=*/true) &&
             data_type == other.data_type;
    }
  };

  using FallbackInterpolatorMap = std::unique_ptr<absl::flat_hash_map<
      FallbackInterpolatorKey, std::unique_ptr<InterpolatorBase<int64_t, 2>>>>;

  using ExactInterpolatorMap = std::unique_ptr<absl::flat_hash_map<
      ExactInterpolatorKey, std::unique_ptr<InterpolatorBase<int64_t, 1>>>>;

  static absl::StatusOr<std::unique_ptr<CollectiveInterpolator>> Create(
      int num_devices_per_host, const HloInstructionProfileList& profiles,
      const se::DeviceDescription& device_info,
      const GpuHloCostAnalysis* analysis = nullptr);

  static absl::StatusOr<std::unique_ptr<CollectiveInterpolator>> Create(
      int num_devices_per_host, const se::DeviceDescription& device_info,
      const GpuHloCostAnalysis* analysis = nullptr);

  // Constructs the semantically correct module from the profile.
  // Usually the root instruction of the entry computation is of interest and is
  // directly related to the `profile`d information.
  static std::unique_ptr<HloModule> ConstructModule(
      const HloInstructionProfile& profile);

  // Returns the estimated runtime for a supported `collective`.
  std::optional<absl::Duration> EstimatedRuntime(
      const HloCollectiveInstruction& instr) const;

 private:
  explicit CollectiveInterpolator(
      ExactInterpolatorMap exact_interpolators,
      FallbackInterpolatorMap fallback_interpolators,
      const se::DeviceDescription& device_info, int num_devices_per_host,
      const GpuHloCostAnalysis* analysis)
      : exact_interpolators_(std::move(exact_interpolators)),
        fallback_interpolators_(std::move(fallback_interpolators)),
        device_info_(device_info),
        num_devices_per_host_(num_devices_per_host),
        analysis_(analysis) {}

  ExactInterpolatorMap exact_interpolators_;
  FallbackInterpolatorMap fallback_interpolators_;
  const se::DeviceDescription& device_info_;
  int num_devices_per_host_;
  const GpuHloCostAnalysis* analysis_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_H_
