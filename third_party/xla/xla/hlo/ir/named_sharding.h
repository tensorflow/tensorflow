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

#include <vector>

#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/xla_data.pb.h"

namespace xla {

// C++ representation for corresponding `OpSharding::NamedSharding` proto so
// same documentation applies.
class NamedSharding {
  struct DimensionSharding {
    std::vector<AxisRef> axes;
    bool is_closed;
  };

  Mesh mesh_;
  std::vector<DimensionSharding> dim_shardings_;
  std::vector<AxisRef> replicated_axes_;
  std::vector<AxisRef> unreduced_axes_;
  std::vector<OpMetadata> metadata_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_NAMED_SHARDING_H_
