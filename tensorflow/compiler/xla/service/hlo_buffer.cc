/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_buffer.h"

#include <algorithm>
#include <ostream>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrCat;

void HloBuffer::AddValue(const HloValue& value) {
  values_.push_back(&value);
  // Sort vector and remove duplicates.
  std::sort(values_.begin(), values_.end(), HloValue::IdLessThan);
  values_.erase(std::unique(values_.begin(), values_.end(), HloValue::IdEqual),
                values_.end());
}

void HloBuffer::RemoveValue(const HloValue& value) {
  // The values are sorted, so finding the value could be done in log(n) time
  // with a binary search.
  auto it = std::find(values_.begin(), values_.end(), &value);
  CHECK(it != values_.end());
  values_.erase(it);
}

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
  std::sort(positions.begin(), positions.end());
  positions.erase(std::unique(positions.begin(), positions.end()),
                  positions.end());
  return positions;
}

string HloBuffer::ToString() const {
  return StrCat("HloBuffer ", id_, ", values: ",
                Join(values_, ", ", [](string* result, const HloValue* value) {
                  result->append(value->ToShortString());
                }));
}

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer) {
  out << buffer.ToString();
  return out;
}

}  // namespace xla
