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
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/mesh_and_axis.h"

namespace xla {

void NamedSharding::DimensionSharding::append(
    const NamedSharding::DimensionSharding& other) {
  for (const AxisRef& axis : other.axes()) {
    // TODO: Not sure if sort + merge is required, because if axis are already
    // sorted + merged then removing some parts of it won't require merging ?
    axes_.push_back(axis);
  }
}

NamedSharding::DimensionSharding NamedSharding::DimensionSharding::split(
    const Mesh& mesh, int64_t split_size) {
  CHECK_GT(split_size, 1);
  CHECK_EQ(getShardedSize(mesh) % split_size, 0);

  int64_t axis_index = 0;
  AxisRef curr_axis = axes()[axis_index];
  int64_t curr_axis_size = curr_axis.size(mesh);

  std::vector<AxisRef> splitted_axes, remaining_axes;

  while (axis_index < axes().size() && split_size > 1) {
    curr_axis = axes()[axis_index];
    curr_axis_size = curr_axis.size(mesh);

    int64_t gcd = std::gcd(curr_axis_size, split_size);
    if (gcd == 1) {
      remaining_axes.push_back(curr_axis);
      ++axis_index;
      continue;
    }

    split_size /= gcd;
    if (gcd == curr_axis_size) {
      splitted_axes.push_back(curr_axis);
    } else {
      int64_t split_axis_pre_size =
          curr_axis.sub_axis_info() ? curr_axis.sub_axis_info()->pre_size : 1;
      splitted_axes.push_back(
          AxisRef(curr_axis.mesh_axis_index(), {split_axis_pre_size, gcd}));
      remaining_axes.push_back(
          AxisRef(curr_axis.mesh_axis_index(),
                  {split_axis_pre_size * gcd, curr_axis_size / gcd}));
      // TODO: Do we want merging of axes here ?
    }

    ++axis_index;
  }

  for (; axis_index < axes().size(); ++axis_index) {
    remaining_axes.push_back(axes()[axis_index]);
  }

  axes_ = std::move(remaining_axes);
  return NamedSharding::DimensionSharding(splitted_axes, is_closed_);
}

int64_t NamedSharding::DimensionSharding::getShardedSize(
    const Mesh& mesh) const {
  return std::accumulate(axes_.begin(), axes_.end(), 1,
                         [&mesh](int64_t cur, const AxisRef& axis) {
                           return cur * axis.size(mesh);
                         });
}

std::string NamedSharding::DimensionSharding::ToString(const Mesh& mesh) const {
  if (axes_.empty()) {
    return is_closed_ ? "{}" : "{?}";
  }

  std::string result = "{";
  absl::StrAppend(&result,
                  absl::StrJoin(axes_, ", ",
                                [&mesh](std::string* out, const AxisRef& axis) {
                                  absl::StrAppend(out, axis.ToString(mesh));
                                }));
  absl::StrAppend(&result, (is_closed_ ? "" : ", ?"));
  absl::StrAppend(&result, "}");
  return result;
}

std::string NamedSharding::ToString(bool include_metadata) const {
  std::string result = "{";
  absl::StrAppend(&result, mesh_.ToString());

  auto print_metadata = [&] {
    if (include_metadata && !metadata_.empty()) {
      absl::StrAppend(&result, ", metadata={");
      absl::StrAppend(
          &result, absl::StrJoin(metadata_, ", ",
                                 [&](std::string* out, const auto& netadata) {
                                   absl::StrAppend(
                                       out, OpMetadataToString(netadata));
                                 }));
      absl::StrAppend(&result, "}");
    }
  };

  // TODO: Different printing for replicated, maximal sharding
  if (IsReplicated()) {
    absl::StrAppend(&result, ", replicated={}");
    print_metadata();
    absl::StrAppend(&result, "}");
    // TODO: In this case would replicated axes be empty ?
    return result;
  }

  if (IsMaximal()) {
    absl::StrAppend(&result, ", maximal={}");
    print_metadata();
    absl::StrAppend(&result, "}");
    return result;
  }

  // Dimension sharding
  absl::StrAppend(&result, ", [");
  absl::StrAppend(
      &result,
      absl::StrJoin(dim_shardings_, ", ",
                    [&](std::string* out, const DimensionSharding& ds) {
                      absl::StrAppend(out, ds.ToString(mesh_));
                    }));
  absl::StrAppend(&result, "]");

  if (!replicated_axes_.empty()) {
    absl::StrAppend(&result, ", replicated={");
    absl::StrAppend(&result,
                    absl::StrJoin(replicated_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  if (!unreduced_axes_.empty()) {
    absl::StrAppend(&result, ", unreduced={");
    absl::StrAppend(&result,
                    absl::StrJoin(unreduced_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  print_metadata();
  absl::StrAppend(&result, "}");

  return result;
}

// std::ostream& operator<<(std::ostream& out,
//                          const NamedSharding::DimensionSharding& sharding) {
//   // TODO: Implement DimeSharding printer not taking mesh
//   return out << sharding.ToString();
// }

std::ostream& operator<<(std::ostream& out, const NamedSharding& sharding) {
  return out << sharding.ToString();
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
