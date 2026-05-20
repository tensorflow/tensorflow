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

#ifndef XLA_CORE_COLLECTIVES_REDUCTION_KIND_H_
#define XLA_CORE_COLLECTIVES_REDUCTION_KIND_H_

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/core/collectives/reduction_kind.pb.h"
#include "xla/util.h"

namespace xla {

enum class ReductionKind { SUM, PRODUCT, MIN, MAX };

ReductionKindProto ToReductionKindProto(ReductionKind kind);

absl::StatusOr<ReductionKind> FromReductionKindProto(
    const ReductionKindProto& proto);

template <typename Sink>
void AbslStringify(Sink& sink, ReductionKind reduction_kind) {
  absl::Format(&sink, "%s", [&] {
    switch (reduction_kind) {
      case ReductionKind::SUM:
        return "sum";
      case ReductionKind::PRODUCT:
        return "prod";
      case ReductionKind::MIN:
        return "min";
      case ReductionKind::MAX:
        return "max";
    }
  }());
}

inline absl::StatusOr<ReductionKind> ParseReductionKind(
    absl::string_view reduction_kind) {
  if (reduction_kind == "sum") {
    return ReductionKind::SUM;
  }
  if (reduction_kind == "prod") {
    return ReductionKind::PRODUCT;
  }
  if (reduction_kind == "min") {
    return ReductionKind::MIN;
  }
  if (reduction_kind == "max") {
    return ReductionKind::MAX;
  }
  return InvalidArgument("Invalid reduction kind: %s", reduction_kind);
}

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_REDUCTION_KIND_H_
