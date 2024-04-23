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

#include "xla/pjrt/host_memory_spaces.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/pjrt/pjrt_client.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

UnpinnedHostMemorySpace::UnpinnedHostMemorySpace(int id, PjRtDevice* device)
    : id_(id), device_(device) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  auto* client = device_->client();
  debug_string_ = absl::StrFormat(
      "UnpinnedHostMemorySpace(id=%i, process_index=%i, client=%s)", id_,
      client->process_index(), client->platform_name());
  to_string_ = absl::StrFormat("UNPINNED_HOST_%i", id_);
}

const int UnpinnedHostMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(UnpinnedHostMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

PinnedHostMemorySpace::PinnedHostMemorySpace(int id, PjRtDevice* device)
    : id_(id), device_(device) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  auto* client = device_->client();
  debug_string_ =
      absl::StrFormat("PinnedHostMemory(id=%i, process_index=%i, client=%s)",
                      id_, client->process_index(), client->platform_name());
  to_string_ = absl::StrFormat("PINNED_HOST_%i", id_);
}

const int PinnedHostMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(PinnedHostMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

}  // namespace xla
