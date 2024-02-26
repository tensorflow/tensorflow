/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_VALUE_RANGE_H_
#define XLA_SERVICE_VALUE_RANGE_H_

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/service/constant_value.h"

namespace xla {

// Class keeping track of the range of an HLO value.
class Range {
 public:
  Range()
      : min_(ConstantValue::GetZero(/*bitwidth=*/64, /*is_signed=*/false)),
        max_(ConstantValue::GetZero(/*bitwidth=*/64, /*is_signed=*/false)),
        empty_(true),
        is_linear_(false) {}
  Range(const ConstantValue& min, const ConstantValue& max, bool is_linear)
      : min_(min), max_(max), empty_(false), is_linear_(is_linear) {}
  // Minimum value of the range.
  const ConstantValue& min() const { return min_; }
  // Maximum value of the range.
  const ConstantValue& max() const { return max_; }
  // Returns if the range is empty (no value in set).
  bool IsEmpty() const { return empty_; }
  // Only one value in set. This means the range is a constant.
  bool IsSingleValue() const { return !IsEmpty() && min_ == max_; }
  // This is a way to track in some way recurring values that change in a
  // monotonic way. This true means that the variables driving the range change
  // in a monotonic way and that the way they are composed together is linear
  // causing the final value represented by the range in a monotonic way during
  // loop recursion.
  bool IsLinear() const { return is_linear_; }
  // If this range represents a single value return that signed value.
  std::optional<int64_t> GetSingleSignedValue() const;
  // If this range represents a single value return that unsigned value.
  std::optional<int64_t> GetSingleUnsignedValue() const;

  std::string ToString() const;

 private:
  ConstantValue min_;
  ConstantValue max_;
  bool empty_;
  bool is_linear_;
};

// Constructs a Range object from a HloInstruction. Gets a "predefined_ranges"
// object as input that returns known ranges for some variables for which we
// already know the range. The final range is composed from operations over
// these predetermined ranges.
// The input HLO needs to be of scalar type and integer.
Range RecursivelyIdentifyRange(
    const HloInstruction* instr,
    const absl::flat_hash_map<const HloInstruction*, Range>& predefined_ranges);

}  // namespace xla

#endif  // XLA_SERVICE_VALUE_RANGE_H_
