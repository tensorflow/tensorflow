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

#include "xla/pjrt/plugin/xla_cpu/cpu_device_description.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"

namespace xla {

namespace {

constexpr char kCpuPlatformName[] = "cpu";

}

CpuDeviceDescription::CpuDeviceDescription(int process_id, int local_device_id)
    : id_(PackCpuDeviceId(process_id, local_device_id)),
      process_index_(process_id),
      local_hardware_id_(local_device_id) {
  debug_string_ = absl::StrCat("TFRT_CPU_", id_.value());
  to_string_ = absl::StrCat("CpuDevice(id=", id_.value(), ")");
}

absl::string_view CpuDeviceDescription::device_kind() const {
  return kCpuPlatformName;
}

absl::string_view CpuDeviceDescription::DebugString() const {
  return debug_string_;
}

absl::string_view CpuDeviceDescription::ToString() const { return to_string_; }

}  // namespace xla
