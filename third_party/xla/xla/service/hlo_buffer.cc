/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_buffer.h"

#include <algorithm>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {

bool HloBuffer::operator==(const HloBuffer& other) const {
  bool equal = id() == other.id();
  if (equal) {
    // DCHECK because these comparisons are expensive (linear time).
    DCHECK(values_ == other.values_);
  }
  return equal;
}

std::vector<HloPosition> HloBuffer::ComputePositions() const {
  std::vector<HloPosition> positions;
  for (const HloValue* value : values_) {
    positions.insert(positions.end(), value->positions().begin(),
                     value->positions().end());
  }
  // Remove duplicates and sort positions.
  absl::c_sort(positions);
  positions.erase(std::unique(positions.begin(), positions.end()),
                  positions.end());
  return positions;
}

std::string HloBuffer::ToString() const {
  return absl::StrCat(
      "HloBuffer ", id_, ", values: ",
      absl::StrJoin(values_, ", ",
                    [](std::string* result, const HloValue* value) {
                      result->append(value->ToShortString());
                    }));
}

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer) {
  out << buffer.ToString();
  return out;
}

}  // namespace xla
