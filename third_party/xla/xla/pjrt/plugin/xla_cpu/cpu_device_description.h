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

#ifndef XLA_PJRT_PLUGIN_XLA_CPU_CPU_DEVICE_DESCRIPTION_H_
#define XLA_PJRT_PLUGIN_XLA_CPU_CPU_DEVICE_DESCRIPTION_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla {

class CpuDeviceDescription final : public PjRtDeviceDescription {
 public:
  explicit CpuDeviceDescription(int process_id, int local_device_id);

  int id() const override { return id_.value(); }

  int process_index() const override { return process_index_; }

  int local_hardware_id() const { return local_hardware_id_; }

  absl::string_view device_kind() const override;

  absl::string_view DebugString() const override;

  absl::string_view ToString() const override;

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  PjRtGlobalDeviceId id_;
  int process_index_;
  int local_hardware_id_;
  std::string debug_string_;
  std::string to_string_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_CPU_CPU_DEVICE_DESCRIPTION_H_
