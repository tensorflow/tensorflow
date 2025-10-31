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

#ifndef XLA_HLO_IR_NAMED_SHARDING_H_
#define XLA_HLO_IR_NAMED_SHARDING_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/xla_data.pb.h"

namespace xla {

// C++ representation for corresponding `OpSharding::NamedSharding` proto so
// same documentation applies.
class NamedSharding {
 public:
  class DimensionSharding {
   public:
    bool operator==(const DimensionSharding& other) const {
      return axes_ == other.axes_ && is_closed_ == other.is_closed_;
    }

    explicit DimensionSharding(std::vector<AxisRef> axes, bool is_closed)
        : axes_(std::move(axes)), is_closed_(is_closed) {}

   private:
    std::vector<AxisRef> axes_;
    bool is_closed_;
  };

  // Shardings using mesh with similar device assignment should compare equal
  bool operator==(const NamedSharding& other) const {
    return mesh_.DeviceAssignmentEquals(other.mesh_) &&
           dim_shardings_ == other.dim_shardings_ &&
           replicated_axes_ == other.replicated_axes_ &&
           unreduced_axes_ == other.unreduced_axes_;
  }

  bool operator!=(const NamedSharding& other) const {
    return !(*this == other);
  }

  // TODO(b/456212087): Add some validation checks
  explicit NamedSharding(Mesh mesh,
                         absl::Span<const DimensionSharding> dim_shardings = {},
                         absl::Span<const AxisRef> replicated_axes = {},
                         absl::Span<const AxisRef> unreduced_axes = {},
                         absl::Span<const OpMetadata> metadata = {})
      : mesh_(std::move(mesh)),
        dim_shardings_(dim_shardings.begin(), dim_shardings.end()),
        replicated_axes_(replicated_axes.begin(), replicated_axes.end()),
        unreduced_axes_(unreduced_axes.begin(), unreduced_axes.end()),
        metadata_(metadata.begin(), metadata.end()) {}

 private:
  Mesh mesh_;
  std::vector<DimensionSharding> dim_shardings_;
  std::vector<AxisRef> replicated_axes_;
  std::vector<AxisRef> unreduced_axes_;
  std::vector<OpMetadata> metadata_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_NAMED_SHARDING_H_
