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

#include "xla/service/gpu/model/hlo_op_profiles.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles_data.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {

/*static*/ const HloOpProfiles& HloOpProfiles::Singleton() {
  static const auto* hlo_op_profiles =
      HloOpProfiles::Load(kDeviceHloOpProfiles,
                          /*default_profile_name=*/"sm_86")
          .release();
  return *hlo_op_profiles;
}

/*static*/ std::string HloOpProfiles::GetProfileName(
    const se::DeviceDescription& device_info) {
  if (auto* ptr = std::get_if<stream_executor::CudaComputeCapability>(
          &device_info.gpu_compute_capability())) {
    return absl::StrCat("sm_", ptr->major, ptr->minor);
  }
  return "<unknown>";
}

/*static*/ std::unique_ptr<HloOpProfiles> HloOpProfiles::Load(
    absl::string_view profiles_text_proto,
    absl::string_view default_profile_name) {
  ProfilesNestedMap profiles_map;
  DeviceHloInstructionProfiles all_device_profiles;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      std::string(profiles_text_proto), &all_device_profiles));
  for (const auto& device_profile : all_device_profiles.entries()) {
    for (const auto& entry : device_profile.second.entries()) {
      auto op_code = StringToHloOpcode(entry.instruction().opcode()).value();
      auto element_type = entry.instruction().shape().element_type();

      profiles_map[device_profile.first][std::make_pair(
          op_code, element_type)] = entry.clock_cycles();
    }
  }
  return absl::WrapUnique(
      new HloOpProfiles(std::move(profiles_map), default_profile_name));
}

const HloOpProfiles::HloOpProfile& HloOpProfiles::GetProfile(
    const se::DeviceDescription& device_info) const {
  auto it = profiles_.find(GetProfileName(device_info));
  if (it != profiles_.end()) return it->second;
  return default_profile_;
}

}  // namespace gpu
}  // namespace xla
