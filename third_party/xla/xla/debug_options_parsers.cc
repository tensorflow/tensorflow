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

#include "xla/debug_options_parsers.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/xla.pb.h"

namespace xla {
namespace details {

bool ParseIntRangeInclusive(absl::string_view string_value,
                            IntRangeInclusive& range) {
  std::vector<absl::string_view> parts = absl::StrSplit(string_value, ':');

  if (parts.size() == 1) {
    // A single integer x is a valid [x, x] range.
    int64_t first;
    if (!absl::SimpleAtoi(parts[0], &first)) {
      return false;
    }
    range.set_first(first);
    range.set_last(first);
    return true;
  }

  if (parts.size() == 2) {
    if (parts[0].empty() && parts[1].empty()) {
      // ":" is not a valid range.
      return false;
    }

    // Allow semi-open ranges (e.g. "1:", ":100").
    int64_t first = std::numeric_limits<int64_t>::min();
    int64_t last = std::numeric_limits<int64_t>::max();
    if (!parts[0].empty() && !absl::SimpleAtoi(parts[0], &first)) {
      return false;
    }
    if (!parts[1].empty() && !absl::SimpleAtoi(parts[1], &last)) {
      return false;
    }
    if (first > last) {
      return false;
    }
    range.set_first(first);
    range.set_last(last);
    return true;
  }

  return false;
}

}  // namespace details
}  // namespace xla
