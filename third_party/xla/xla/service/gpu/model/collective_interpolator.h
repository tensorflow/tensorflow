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
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/interpolator.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

class CollectiveInterpolator {
 public:
  enum class CommunicationType {
    UNDEFINED = 0,
    RAIL_ALIGNED = 1,
    NON_RAIL_ALIGNED = 2,
    SINGLE_HOST = 3
  };

  struct InterpolatorKey {
    HloOpcode opcode;
    CommunicationType communication_type;

    template <typename H>
    friend H AbslHashValue(H h, const InterpolatorKey& key) {
      return H::combine(std::move(h), key.opcode, key.communication_type);
    }

    bool operator==(const InterpolatorKey& other) const {
      return opcode == other.opcode &&
             communication_type == other.communication_type;
    }
  };

  using InterpolatorMap =
      absl::flat_hash_map<InterpolatorKey, EuclideanNNInterpolator<int64_t, 2>>;

  static absl::StatusOr<std::unique_ptr<CollectiveInterpolator>> Create(
      HloInstructionProfileList profiles,
      const se::DeviceDescription& device_info);

  // Constructs the semantically correct module from the profile.
  // Usually the root instruction of the entry computation is of interest and is
  // directly related to the `profile`d information.
  static std::unique_ptr<HloModule> ConstructModule(
      const HloInstructionProfile& profile);

  // Returns the estimated runtime for a supported `collective`.
  std::optional<absl::Duration> EstimatedRuntime(
      HloCollectiveInstruction& instr);

 private:
  // Uses `EuclideanNNInterpolator` to figure get the closest neighbour from
  // profiles.
  explicit CollectiveInterpolator(HloInstructionProfileList profiles,
                                  InterpolatorMap interpolators,
                                  const se::DeviceDescription& device_info)
      : profiles_(profiles),
        interpolators_(interpolators),
        device_info_(device_info) {}

  HloInstructionProfileList profiles_;
  InterpolatorMap interpolators_;

  const se::DeviceDescription& device_info_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_H_
