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

#include "xla/compact_string.h"

#include "tsl/platform/coding.h"

namespace xla {

void CompactString::set(absl::string_view s) {
  if (s.size() == 0) {
    rep_.reset(nullptr);
  } else {
    size_t bytes_needed = s.size() + tsl::core::VarintLength(s.size());
    rep_.reset(new char[bytes_needed]);
    char* p = rep_.get();
    p = tsl::core::EncodeVarint64(p, s.size());
    memcpy(p, s.data(), s.size());
  }
}

size_t CompactString::size() const {
  if (rep_ == nullptr) {
    return 0;
  } else {
    const char* p = rep_.get();
    uint64_t len;
    p = tsl::core::GetVarint64Ptr(p, p + tsl::core::kMaxVarint64Bytes, &len);
    return len;
  }
}

absl::string_view CompactString::view() const {
  if (rep_ == nullptr) {
    return absl::string_view();
  } else {
    const char* p = rep_.get();
    uint64_t len;
    p = tsl::core::GetVarint64Ptr(p, p + tsl::core::kMaxVarint64Bytes, &len);
    return absl::string_view(p, len);
  }
}

}  // namespace xla
