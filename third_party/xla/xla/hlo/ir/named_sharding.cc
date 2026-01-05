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

#include "xla/hlo/ir/named_sharding.h"

#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/mesh_and_axis.h"

namespace xla {

std::optional<NamedSharding::DimensionSharding>
NamedSharding::DimensionSharding::Slice(const Mesh& mesh, int64_t slice_size) {
  if (slice_size == 1) {
    return DimensionSharding({}, is_closed_);
  }
  if (getShardedSize(mesh) % slice_size != 0) {
    return std::nullopt;
  }

  int64_t axis_index = 0;
  std::vector<AxisRef> sliced_axes, remaining_axes;

  for (; axis_index < axes().size(); ++axis_index) {
    const AxisRef& curr_axis = axes()[axis_index];
    int64_t curr_axis_size = curr_axis.size(mesh);

    if (slice_size == curr_axis_size) {
      sliced_axes =
          std::vector<AxisRef>(axes().begin(), axes().begin() + axis_index + 1);
      slice_size = 1;
      break;
    }
    if (slice_size % curr_axis_size == 0) {
      slice_size /= curr_axis_size;
    } else if (curr_axis_size % slice_size == 0) {
      sliced_axes =
          std::vector<AxisRef>(axes().begin(), axes().begin() + axis_index);
      int64_t sliced_axis_pre_size =
          curr_axis.sub_axis_info() ? curr_axis.sub_axis_info()->pre_size : 1;
      sliced_axes.push_back(AxisRef(curr_axis.mesh_axis_index(),
                                    {sliced_axis_pre_size, slice_size}));
      remaining_axes.push_back(AxisRef(
          curr_axis.mesh_axis_index(),
          {sliced_axis_pre_size * slice_size, curr_axis_size / slice_size}));
      slice_size = 1;
      break;
    } else {
      return std::nullopt;
    }
  }

  if (slice_size != 1) {
    return std::nullopt;
  }

  remaining_axes.insert(remaining_axes.end(), axes().begin() + axis_index + 1,
                        axes().end());
  axes_ = std::move(remaining_axes);
  return NamedSharding::DimensionSharding(sliced_axes, is_closed_);
}

int64_t NamedSharding::DimensionSharding::getShardedSize(
    const Mesh& mesh) const {
  return std::accumulate(axes_.begin(), axes_.end(), 1,
                         [&mesh](int64_t cur, const AxisRef& axis) {
                           return cur * axis.size(mesh);
                         });
}

namespace test_utils {
// Construct sharding with given mesh. 'dim_shardings', 'replicated_axes',
// 'unreduced_axes' refer to axis names in the mesh.
// This is a test only helper function.
NamedSharding FromAxisNames(
    Mesh mesh, absl::Span<const std::vector<std::string>> dim_shardings,
    absl::Span<const std::string> replicated_axes,
    absl::Span<const std::string> unreduced_axes,
    absl::Span<const OpMetadata> metadata) {
  std::map<std::string, int64_t> mesh_axis_to_index;
  for (int64_t i = 0; i < mesh.axis_names().size(); ++i) {
    mesh_axis_to_index[mesh.axis_names()[i]] = i;
  }

  std::vector<NamedSharding::DimensionSharding> dim_shardings_;
  dim_shardings_.reserve(dim_shardings.size());
  for (const auto& axes_for_dim : dim_shardings) {
    std::vector<AxisRef> axis_refs;
    axis_refs.reserve(axes_for_dim.size());
    for (const std::string& axis_name : axes_for_dim) {
      auto it = mesh_axis_to_index.find(axis_name);
      CHECK(it != mesh_axis_to_index.end())
          << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
      axis_refs.push_back(AxisRef(it->second));
    }
    dim_shardings_.push_back(NamedSharding::DimensionSharding(
        std::move(axis_refs), /*is_closed=*/true));
  }

  std::vector<AxisRef> replicated_axes_;
  replicated_axes_.reserve(replicated_axes.size());
  for (const std::string& axis_name : replicated_axes) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK(it != mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    replicated_axes_.push_back(AxisRef(it->second));
  }

  std::vector<AxisRef> unreduced_axes_;
  unreduced_axes_.reserve(unreduced_axes.size());
  for (const std::string& axis_name : unreduced_axes) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK(it != mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    unreduced_axes_.push_back(AxisRef(it->second));
  }

  return NamedSharding(mesh, dim_shardings_, replicated_axes_, unreduced_axes_,
                       metadata);
}
}  // namespace test_utils
}  // namespace xla
