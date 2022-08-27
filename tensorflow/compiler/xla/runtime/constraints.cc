/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/runtime/constraints.h"

#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {
namespace runtime {

using absl::InvalidArgumentError;
using absl::Span;
using absl::StatusOr;
using absl::StrCat;

using llvm::raw_ostream;

raw_ostream& operator<<(raw_ostream& os, const ArgumentConstraint& constraint) {
  auto str = [](ArgumentConstraint constraint) {
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
  };

  os << str(constraint);
  return os;
}

raw_ostream& operator<<(raw_ostream& os,
                        Span<const ArgumentConstraint> constraints) {
  os << "[";
  llvm::interleaveComma(constraints, os);
  os << "]";
  return os;
}

StatusOr<ArgumentConstraint> ParseArgumentConstraint(std::string_view str) {
  if (str == "rank") return ArgumentConstraint::kRank;
  if (str == "shape") return ArgumentConstraint::kShape;
  if (str == "value") return ArgumentConstraint::kValue;
  return InvalidArgumentError(StrCat("unknown operand constraint: ", str));
}

}  // namespace runtime
}  // namespace xla
