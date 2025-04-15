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

#ifndef XLA_SERVICE_HLO_BUFFER_H_
#define XLA_SERVICE_HLO_BUFFER_H_

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "xla/service/hlo_value.h"
#include "xla/shape_tree.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A container which can hold one or more HloValues. An HLO buffer abstractly
// represents the allocation which HLO instructions write into and read
// from. Generally there is a one-to-one correspondence between HloBuffers and
// HloValue where each HloValue in the module is held in a unique HloBuffer. An
// exception is the while instruction which updates the loop state in-place. In
// this case, we have a single HloBuffer for each HloPosition in the loop state,
// but multiple HloValues. For example:
//
//   %init = ...
//   %while = While(%init, body, condition)
//
//  body:
//   %body_param = Param(0)
//     ...
//   %body_root = ...
//
//  condition:
//   %cond_param = Param(0)
//     ...
//
// For simplicity, assume that %while is array-shaped. In this case, we have a
// single HloBuffer which holds the following HloValues: HloValue{%init},
// HloValue{%while}, HloValue{%body_param}, HloValue{%body_root}, and
// HloValue{%cond_param}.
//
// HloBuffers may appear at different HloPositions in the module mirroring the
// same property of HloValues. For example:
//
//   %sub = Sub(...)
//   %add = Add(...)
//   %tuple = Tuple(%add, %sub)
//   %gte = GetTupleElement(%tuple, 0)
//
// In this case, the HloBuffer containing %add appears at the following
// positions: HloPosition{%add, {}}, HloPosition{%tuple, {0}}, and
// HloPosition{%gte, {}}.
//
// Different HloPositions which share the same HloBuffer indicate mandatory
// aliasing in the HLO module. These positions must share the same memory
// allocation for correctness (the backends rely on this property). This differs
// from incidental aliasing introduced by memory reuse in BufferAssignment where
// different instructions may happen to get the same allocation.
class HloBuffer {
 public:
  using Id = int64_t;

  // Predicate comparing HloBuffers by increasing id, useful for std::sort.
  static bool IdLessThan(const HloBuffer* a, const HloBuffer* b) {
    return a->id() < b->id();
  }

  // Predicate comparing HloBuffers by equal id, useful for std::unique.
  static bool IdEqual(const HloBuffer* a, const HloBuffer* b) {
    return a->id() == b->id();
  }

  HloBuffer(Id id, std::vector<const HloValue*> values)
      : id_(id), values_(std::move(values)) {}

  // Return the unique identifier for this HloBuffer.
  Id id() const { return id_; }

  // Return all values contained in this buffer.
  const std::vector<const HloValue*>& values() const { return values_; }

  // Memory space color. Used to indicate the memory space that the hlo buffer
  // needs to live in.
  BufferValue::Color color() const {
    // Invariant: All values in the buffer should have the same color.
    BufferValue::Color result = values()[0]->color();
    for (const HloValue* value : values()) {
      DCHECK_EQ(result, value->color());
    }
    return result;
  }

  // Return the unique HLO value in the buffer. CHECK fails if the buffer does
  // not contain exactly one value.
  const HloValue& GetUniqueValue() const {
    CHECK_EQ(values_.size(), 1);
    return *values_[0];
  }

  std::vector<HloPosition> ComputePositions() const;

  std::string ToString() const;

  bool operator==(const HloBuffer& other) const;
  bool operator!=(const HloBuffer& other) const { return !(*this == other); }

 private:
  // Unique identifier for this HloBuffer.
  Id id_;

  // The set of values contained in this buffer. Vector contains no duplicates
  // and is sorted stably by HloValue::Id.
  std::vector<const HloValue*> values_;
};

std::ostream& operator<<(std::ostream& out, const HloBuffer& buffer);

}  // namespace xla

#endif  // XLA_SERVICE_HLO_BUFFER_H_
