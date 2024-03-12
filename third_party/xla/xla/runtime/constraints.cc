/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/runtime/constraints.h"

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xla {
namespace runtime {

using absl::InvalidArgumentError;
using absl::StatusOr;
using absl::StrCat;

StatusOr<ArgumentConstraint> ParseArgumentConstraint(std::string_view str) {
  if (str == "rank") return ArgumentConstraint::kRank;
  if (str == "shape") return ArgumentConstraint::kShape;
  if (str == "value") return ArgumentConstraint::kValue;
  return InvalidArgumentError(StrCat("unknown operand constraint: ", str));
}

std::string ArgumentConstraintToString(ArgumentConstraint constraint) {
  switch (constraint) {
    case ArgumentConstraint::kResolved:
      return "resolved";
    case ArgumentConstraint::kRank:
      return "rank";
    case ArgumentConstraint::kShape:
      return "shape";
    case ArgumentConstraint::kValue:
      return "value";
    default:
      llvm_unreachable("unknown operand constraint");
  }
}

}  // namespace runtime
}  // namespace xla
