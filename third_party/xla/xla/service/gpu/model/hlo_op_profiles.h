/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_H_
#define XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class HloOpProfiles {
 public:
  using HloOpProfile =
      absl::flat_hash_map<std::pair<HloOpcode, PrimitiveType>, int64_t>;
  using ProfilesNestedMap =
      absl::flat_hash_map<std::string,  // compute capability.
                          HloOpProfile>;

  // Returns singleton with profiler data.
  static const HloOpProfiles& Singleton();

  // Returns profile name for the given device.
  // For CUDA, the format is "sm_XX".
  // Returns "<unknown>" for unknown devices.
  static std::string GetProfileName(const se::DeviceDescription& device_info);

  // Loads profiles from the given text proto data.
  static std::unique_ptr<HloOpProfiles> Load(
      std::string_view profiles_text_proto,
      std::string_view default_profile_name);

  const HloOpProfile& GetProfile(
      const se::DeviceDescription& device_info) const;

  const HloOpProfile& GetDefaultProfile() const { return default_profile_; }

 private:
  HloOpProfiles(ProfilesNestedMap profiles,
                std::string_view default_profile_name)
      : profiles_(std::move(profiles)),
        default_profile_(profiles_.at(default_profile_name)) {}

  ProfilesNestedMap profiles_;
  const HloOpProfile& default_profile_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_HLO_OP_PROFILES_H_
