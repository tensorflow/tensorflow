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

#include "xla/service/gpu/nccl_clique_key.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/service/global_device_id.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NcclCliqueKey
//===----------------------------------------------------------------------===//

NcclCliqueKey::NcclCliqueKey(std::vector<GlobalDeviceId> devices,
                             int64_t stream_id)
    : devices_(std::move(devices)), stream_id_(stream_id) {}

absl::Span<const GlobalDeviceId> NcclCliqueKey::devices() const {
  return devices_;
}

std::optional<int64_t> NcclCliqueKey::rank(GlobalDeviceId id) const {
  if (auto it = absl::c_find(devices_, id); it != devices_.end()) {
    return it - devices_.begin();
  }
  return std::nullopt;
}

std::string NcclCliqueKey::ToString() const {
  return absl::StrCat("devices=", GlobalDeviceIdsToString(devices_),
                      "; stream=", stream_id_);
}

bool operator==(const NcclCliqueKey& a, const NcclCliqueKey& b) {
  return a.devices_ == b.devices_ && a.stream_id_ == b.stream_id_;
}

bool operator<(const NcclCliqueKey& a, const NcclCliqueKey& b) {
  if (a.stream_id_ < b.stream_id_) return true;
  if (b.stream_id_ < a.stream_id_) return false;

  return a.devices_ < b.devices_;
}

//===----------------------------------------------------------------------===//
// NcclCliqueId
//===----------------------------------------------------------------------===//

NcclCliqueId::NcclCliqueId() { std::fill(data_.begin(), data_.end(), 0); }

NcclCliqueId::NcclCliqueId(char bytes[kSize]) {
  std::copy(bytes, bytes + kSize, data_.data());
}

absl::StatusOr<NcclCliqueId> NcclCliqueId::FromString(std::string_view str) {
  if (str.size() != kSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid NCCL clique id size: ", str.size(), ", expected ",
                     kSize, " bytes"));
  }
  char bytes[kSize];
  std::copy(str.data(), str.data() + kSize, bytes);
  return NcclCliqueId(bytes);
}

absl::Span<const char> NcclCliqueId::data() const { return data_; }

std::string NcclCliqueId::ToString() const {
  return std::string(data_.data(), data_.size());
}

}  // namespace xla::gpu
