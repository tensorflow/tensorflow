// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/device.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"

namespace xla {
namespace ifrt {
namespace proxy {

ifrt::Client* Device::client() const { return client_; }

DeviceId Device::Id() const { return DeviceId(description_.id()); }

bool Device::IsAddressable() const { return is_addressable_; }

absl::string_view Device::Kind() const { return description_.device_kind(); }
absl::string_view Device::ToString() const { return description_.ToString(); }

absl::string_view Device::DebugString() const {
  return description_.DebugString();
}

absl::Span<ifrt::Memory* const> Device::Memories() const { return memories_; }

absl::StatusOr<ifrt::Memory*> Device::DefaultMemory() const {
  if (default_memory_ == nullptr) {
    return absl::UnimplementedError("Device does not support default_memory");
  }
  return default_memory_;
}

int Device::ProcessIndex() const { return description_.process_index(); }

const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
Device::Attributes() const {
  return description_.Attributes();
}

char Device::ID = 0;  // NOLINT

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
