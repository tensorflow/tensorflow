/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_payload_deduplicator.h"

#include <cstdint>
#include <deque>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/backend_config.h"

namespace xla {

HloPayloadDeduplicator::HloPayloadDeduplicator(int64_t base_offset)
    : offset_(base_offset) {}

int64_t HloPayloadDeduplicator::Deduplicate(
    const BackendConfigWrapper* wrapper) {
  auto it = pointer_map_.find(wrapper);
  if (it != pointer_map_.end()) {
    return it->second;
  }

  // Fall back to string deduplication.
  int64_t id = Deduplicate(wrapper->GetRawString());
  pointer_map_.emplace(wrapper, id);
  return id;
}

int64_t HloPayloadDeduplicator::Deduplicate(absl::string_view value) {
  auto it = string_map_.find(value);
  if (it != string_map_.end()) {
    return it->second;
  }
  int64_t id = offset_ + payloads_.size();
  payloads_.emplace_back(value);
  string_map_.emplace(payloads_.back(), id);
  return id;
}

std::deque<std::string> HloPayloadDeduplicator::TakePayloads() {
  std::deque<std::string> result = std::move(payloads_);
  payloads_.clear();
  string_map_.clear();
  pointer_map_.clear();
  return result;
}

}  // namespace xla
