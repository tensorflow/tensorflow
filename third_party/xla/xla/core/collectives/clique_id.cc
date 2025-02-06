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

#include "xla/core/collectives/clique_id.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/crc/crc32c.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla {

CliqueId::CliqueId(absl::string_view data) : data_(data.begin(), data.end()) {}

absl::Span<const char> CliqueId::data() const { return data_; }

std::string CliqueId::ToString() const {
  return std::string(data_.data(), data_.size());
}

uint32_t CliqueId::fingerprint() const {
  absl::string_view data_view(data_.data(), data_.size());
  return static_cast<uint32_t>(absl::ComputeCrc32c(data_view));
}

size_t CliqueId::size() const { return data_.size(); }

CliqueIds::CliqueIds(const CliqueId& id) { Add(id); }

void CliqueIds::Add(const CliqueId& id) { ids_.push_back(id); }

absl::Span<const CliqueId> CliqueIds::data() const { return ids_; }

const CliqueId& CliqueIds::at(size_t index) const { return ids_[index]; }

uint32_t CliqueIds::fingerprint() const {
  absl::crc32c_t crc(0);
  for (const auto& clique_id : ids_) {
    crc = absl::ExtendCrc32c(crc, absl::string_view(clique_id.data().data(),
                                                    clique_id.data().size()));
  }
  return static_cast<uint32_t>(crc);
}

}  // namespace xla
