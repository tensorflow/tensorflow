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

#ifndef XLA_SERVICE_IO_PARAM_H_
#define XLA_SERVICE_IO_PARAM_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// A class for capturing an input or output to an HLO op.
class IOParam {
 public:
  explicit IOParam(HloPosition position) : value_(position) {}
  explicit IOParam(HloUse use) : value_(use) {}
  IOParam(const IOParam& other) = default;
  IOParam& operator=(const IOParam& other) = default;
  IOParam(IOParam&& other) = default;
  IOParam& operator=(IOParam&& other) = default;

  bool operator==(const IOParam& other) const { return value_ == other.value_; }
  bool operator<(const IOParam& other) const;

  template <typename H>
  friend H AbslHashValue(H h, const IOParam& param) {
    return H::combine(std::move(h), param.ToTuple());
  }

  friend std::ostream& operator<<(std::ostream& out, const IOParam& param);

  bool IsInput() const { return std::holds_alternative<HloUse>(value_); }
  bool IsOutput() const { return std::holds_alternative<HloPosition>(value_); }
  const HloUse& AsInput() const;
  const HloPosition& AsOutput() const;

  // Returns the call site of the input or output.
  HloInstruction* GetCallSite() const;

  const HloValue& GetHloValue(const HloAliasAnalysis& alias_analysis) const;
  const HloBuffer& GetHloBuffer(const HloAliasAnalysis& alias_analysis) const;

  // Returns the immediate/direct source of the `IOParam`.
  HloPosition GetSourcePosition() const;

  // Returns the shape of the param.
  const Shape& GetShape() const;

  // Find the non-trivial uses of the given position.
  //
  // Get-tuple-element, bitcast, and tuple uses are considered trivial.
  absl::StatusOr<std::vector<IOParam>> GetNonTrivialUses() const;

  // Find the non-trivial source position of the given use.
  //
  // Get-tuple-element, bitcast, and tuple positions are considered trivial.
  absl::StatusOr<IOParam> GetNonTrivialSourcePosition() const;

  std::string ToString() const;

 private:
  using Tuple = std::tuple<int64_t, bool, int, const ShapeIndex&>;

  IOParam() = default;

  Tuple ToTuple() const;

  std::variant<HloPosition, HloUse> value_;
};

}  // namespace xla

#endif  // XLA_SERVICE_IO_PARAM_H_
