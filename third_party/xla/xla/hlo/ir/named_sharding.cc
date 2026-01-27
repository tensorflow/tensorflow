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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

using DimensionSharding = NamedSharding::DimensionSharding;

NamedSharding::NamedSharding(Mesh mesh,
                             absl::Span<const DimensionSharding> dim_shardings,
                             absl::Span<const AxisRef> replicated_axes,
                             absl::Span<const AxisRef> unreduced_axes,
                             absl::Span<const AxisRef> manual_axes,
                             absl::Span<const OpMetadata> metadata)
    : mesh_(std::move(mesh)),
      dim_shardings_(CanonicalizedDimShardings(dim_shardings)),
      replicated_axes_(replicated_axes.begin(), replicated_axes.end()),
      unreduced_axes_(unreduced_axes.begin(), unreduced_axes.end()),
      manual_axes_(manual_axes.begin(), manual_axes.end()),
      metadata_(metadata.begin(), metadata.end()) {
  CHECK_OK(VerifyNamedSharding(*this));
  InitShardedSizes();
}

void DimensionSharding::Append(const DimensionSharding& other,
                               const Mesh& mesh) {
  if (other.axes_.empty()) {
    return;
  }
  if (axes_.empty()) {
    axes_ = other.axes_;
    return;
  }

  // Merge last element of `axes_` with first element of `other.axes_`
  if (!axes_.back().Merge(other.axes_.front(), mesh)) {
    axes_.push_back(other.axes_.front());
  }

  axes_.insert(axes_.end(), other.axes_.begin() + 1, other.axes_.end());
}

std::optional<DimensionSharding> DimensionSharding::Slice(const Mesh& mesh,
                                                          int64_t slice_size) {
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
  return DimensionSharding(sliced_axes, is_closed_);
}

int64_t DimensionSharding::getShardedSize(const Mesh& mesh) const {
  return std::accumulate(axes_.begin(), axes_.end(), 1,
                         [&mesh](int64_t cur, const AxisRef& axis) {
                           return cur * axis.size(mesh);
                         });
}

std::string DimensionSharding::ToString(const Mesh* mesh) const {
  std::string result = "{";
  absl::StrAppend(
      &result,
      absl::StrJoin(axes_, ", ", [mesh](std::string* out, const AxisRef& axis) {
        absl::StrAppend(out, axis.ToString(mesh));
      }));

  if (!is_closed_) {
    if (axes_.empty()) {
      absl::StrAppend(&result, "?");
    } else {
      absl::StrAppend(&result, ", ?");
    }
  }

  absl::StrAppend(&result, "}");
  return result;
}

NamedShardingProto::DimensionSharding DimensionSharding::ToProto() const {
  NamedShardingProto::DimensionSharding proto;
  for (const AxisRef& axis : axes_) {
    *proto.add_axes() = axis.ToProto();
  }
  proto.set_is_closed(is_closed_);
  return proto;
}

DimensionSharding DimensionSharding::FromProto(
    const NamedShardingProto::DimensionSharding& proto) {
  DimensionSharding dim_sharding;
  dim_sharding.is_closed_ = proto.is_closed();
  dim_sharding.axes_.reserve(proto.axes_size());
  for (const AxisRefProto& axis_proto : proto.axes()) {
    dim_sharding.axes_.push_back(AxisRef::FromProto(axis_proto));
  }
  return dim_sharding;
}

std::string NamedSharding::ToString(bool include_metadata) const {
  std::string result = "{";

  std::string metadata_str;
  if (include_metadata && !metadata_.empty()) {
    metadata_str = ", metadata={";
    absl::StrAppend(
        &metadata_str,
        absl::StrJoin(
            metadata_, ", ", [&](std::string* out, const auto& metadata) {
              absl::StrAppend(out, "{", OpMetadataToString(metadata), "}");
            }));
    absl::StrAppend(&metadata_str, "}");
  }

  // Special cases.
  if (IsReplicated() && replicated_axes_.empty()) {
    absl::StrAppend(&result, "replicated");
    absl::StrAppend(&result, metadata_str);
    absl::StrAppend(&result, "}");
    return result;
  }

  if (IsMaximal()) {
    absl::StrAppend(&result, "maximal device=");
    absl::StrAppend(&result, *mesh_.device_assignment().array().begin());
    absl::StrAppend(&result, metadata_str);
    absl::StrAppend(&result, "}");
    return result;
  }

  absl::StrAppend(&result, mesh_.ToString());

  // Dimension sharding.
  absl::StrAppend(&result, ", [");
  absl::StrAppend(
      &result,
      absl::StrJoin(dim_shardings_, ", ",
                    [&](std::string* out, const DimensionSharding& ds) {
                      absl::StrAppend(out, ds.ToString(&mesh_));
                    }));
  absl::StrAppend(&result, "]");

  if (!replicated_axes_.empty()) {
    absl::StrAppend(&result, ", replicated={");
    absl::StrAppend(&result,
                    absl::StrJoin(replicated_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  if (!unreduced_axes_.empty()) {
    absl::StrAppend(&result, ", unreduced={");
    absl::StrAppend(&result,
                    absl::StrJoin(unreduced_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  if (!manual_axes_.empty()) {
    absl::StrAppend(&result, ", manual={");
    absl::StrAppend(&result,
                    absl::StrJoin(manual_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  absl::StrAppend(&result, metadata_str);
  absl::StrAppend(&result, "}");

  return result;
}

NamedShardingProto NamedSharding::ToProto() const {
  NamedShardingProto proto;
  *proto.mutable_mesh() = mesh_.ToProto();
  for (const DimensionSharding& dim_sharding : dim_shardings_) {
    *proto.add_dim_shardings() = dim_sharding.ToProto();
  }
  for (const AxisRef& axis : replicated_axes_) {
    *proto.add_replicated_axes() = axis.ToProto();
  }
  for (const AxisRef& axis : unreduced_axes_) {
    *proto.add_unreduced_axes() = axis.ToProto();
  }
  for (const AxisRef& axis : manual_axes_) {
    *proto.add_manual_axes() = axis.ToProto();
  }
  for (const OpMetadata& metadata : metadata_) {
    *proto.add_metadata() = metadata;
  }
  return proto;
}

NamedSharding NamedSharding::FromProto(const NamedShardingProto& proto) {
  std::vector<DimensionSharding> dim_shardings;
  dim_shardings.reserve(proto.dim_shardings_size());
  for (const auto& dim_sharding_proto : proto.dim_shardings()) {
    dim_shardings.push_back(DimensionSharding::FromProto(dim_sharding_proto));
  }

  std::vector<AxisRef> replicated_axes;
  replicated_axes.reserve(proto.replicated_axes_size());
  for (const auto& axis_proto : proto.replicated_axes()) {
    replicated_axes.push_back(AxisRef::FromProto(axis_proto));
  }

  std::vector<AxisRef> unreduced_axes;
  unreduced_axes.reserve(proto.unreduced_axes_size());
  for (const auto& axis_proto : proto.unreduced_axes()) {
    unreduced_axes.push_back(AxisRef::FromProto(axis_proto));
  }

  std::vector<AxisRef> manual_axes;
  manual_axes.reserve(proto.manual_axes_size());
  for (const auto& axis_proto : proto.manual_axes()) {
    manual_axes.push_back(AxisRef::FromProto(axis_proto));
  }

  std::vector<OpMetadata> metadata(proto.metadata().begin(),
                                   proto.metadata().end());

  return NamedSharding(Mesh::FromProto(proto.mesh()), dim_shardings,
                       replicated_axes, unreduced_axes, manual_axes, metadata);
}

std::ostream& operator<<(std::ostream& out, const DimensionSharding& sharding) {
  return out << sharding.ToString();
}

std::ostream& operator<<(std::ostream& out, const NamedSharding& sharding) {
  return out << sharding.ToString();
}

namespace test_utils {

namespace {

AxisRef ParseAxisRef(
    const Mesh& mesh, absl::string_view axis_str,
    const absl::flat_hash_map<std::string, int64_t>& mesh_axis_to_index) {
  size_t colon_pos = axis_str.rfind(":(");
  if (colon_pos == absl::string_view::npos) {
    auto it = mesh_axis_to_index.find(axis_str);
    CHECK_NE(it, mesh_axis_to_index.end())
        << "Axis " << axis_str << " not found in mesh " << mesh.ToString();
    return AxisRef(it->second);
  }

  absl::string_view name = axis_str.substr(0, colon_pos);
  absl::string_view suffix = axis_str.substr(colon_pos + 2);  // skip ":("

  size_t paren_pos = suffix.find(')');
  CHECK_NE(paren_pos, absl::string_view::npos)
      << "Invalid sub-axis format: " << axis_str;

  absl::string_view pre_size_str = suffix.substr(0, paren_pos);
  absl::string_view size_str = suffix.substr(paren_pos + 1);

  int64_t pre_size;
  int64_t size;

  CHECK(absl::SimpleAtoi(pre_size_str, &pre_size))
      << "Invalid pre-size: " << pre_size_str;
  CHECK(absl::SimpleAtoi(size_str, &size)) << "Invalid size: " << size_str;

  auto it = mesh_axis_to_index.find(name);
  CHECK_NE(it, mesh_axis_to_index.end())
      << "Axis " << name << " not found in mesh " << mesh.ToString();
  return AxisRef(it->second, {pre_size, size});
}

}  // namespace

// Construct sharding with given mesh. 'dim_shardings', 'replicated_axes',
// 'unreduced_axes' refer to axis names in the mesh.
// This is a test only helper function.
NamedSharding FromAxisNames(
    Mesh mesh, absl::Span<const std::vector<std::string>> dim_shardings,
    absl::Span<const std::string> replicated_axes,
    absl::Span<const std::string> unreduced_axes,
    absl::Span<const std::string> manual_axes,
    absl::Span<const OpMetadata> metadata) {
  absl::flat_hash_map<std::string, int64_t> mesh_axis_to_index;
  for (int64_t i = 0; i < mesh.axis_names().size(); ++i) {
    mesh_axis_to_index[mesh.axis_names()[i]] = i;
  }

  std::vector<NamedSharding::DimensionSharding> dim_shardings_vec;
  dim_shardings_vec.reserve(dim_shardings.size());
  for (const std::vector<std::string>& axes_for_dim : dim_shardings) {
    std::vector<AxisRef> axis_refs;
    axis_refs.reserve(axes_for_dim.size());
    bool is_closed = true;
    for (const std::string& axis_name : axes_for_dim) {
      if (axis_name == "?") {
        is_closed = false;
        continue;
      }
      axis_refs.push_back(ParseAxisRef(mesh, axis_name, mesh_axis_to_index));
    }
    dim_shardings_vec.push_back(
        NamedSharding::DimensionSharding(std::move(axis_refs), is_closed));
  }

  std::vector<AxisRef> replicated_axes_vec;
  replicated_axes_vec.reserve(replicated_axes.size());
  for (const std::string& axis_name : replicated_axes) {
    replicated_axes_vec.push_back(
        ParseAxisRef(mesh, axis_name, mesh_axis_to_index));
  }

  std::vector<AxisRef> unreduced_axes_vec;
  unreduced_axes_vec.reserve(unreduced_axes.size());
  for (const std::string& axis_name : unreduced_axes) {
    unreduced_axes_vec.push_back(
        ParseAxisRef(mesh, axis_name, mesh_axis_to_index));
  }

  std::vector<AxisRef> manual_axes_vec;
  manual_axes_vec.reserve(manual_axes.size());
  for (const std::string& axis_name : manual_axes) {
    manual_axes_vec.push_back(
        ParseAxisRef(mesh, axis_name, mesh_axis_to_index));
  }

  return NamedSharding(mesh, dim_shardings_vec, replicated_axes_vec,
                       unreduced_axes_vec, manual_axes_vec, metadata);
}

}  // namespace test_utils

namespace {

bool CompareAxisRefs(const AxisRef& a, const AxisRef& b, const Mesh& mesh) {
  if (a.mesh_axis_index() != b.mesh_axis_index()) {
    return a.mesh_axis_index() < b.mesh_axis_index();
  }
  int64_t a_pre = a.sub_axis_info() ? a.sub_axis_info()->pre_size : 1;
  int64_t b_pre = b.sub_axis_info() ? b.sub_axis_info()->pre_size : 1;

  return a_pre != b_pre ? a_pre < b_pre : a.size(mesh) < b.size(mesh);
}

absl::Status VerifyAndTrack(
    const AxisRef& axis, const Mesh& mesh,
    absl::flat_hash_map<int64_t, std::vector<AxisRef>>& seen_axes) {
  TF_RETURN_IF_ERROR(axis.Validate(mesh));
  auto& axes_on_dim = seen_axes[axis.mesh_axis_index()];
  for (const AxisRef& other : axes_on_dim) {
    if (!axis.CanCoexistWithoutOverlap(other)) {
      return absl::InvalidArgumentError("Axes cannot coexist or axes overlap.");
    }
  }
  axes_on_dim.push_back(axis);
  return absl::OkStatus();
}

absl::Status VerifySortedAxes(
    absl::Span<const AxisRef> axes, absl::string_view name, const Mesh& mesh,
    absl::flat_hash_map<int64_t, std::vector<AxisRef>>& seen_axes) {
  if (!absl::c_is_sorted(axes, [&](const AxisRef& a, const AxisRef& b) {
        return CompareAxisRefs(a, b, mesh);
      })) {
    return absl::InvalidArgumentError(
        absl::StrCat(name,
                     " axes must be sorted by mesh axis index and "
                     "sub-axis pre-size."));
  }
  for (auto it = axes.begin(); it != axes.end(); ++it) {
    TF_RETURN_IF_ERROR(VerifyAndTrack(*it, mesh, seen_axes));
    if (it != axes.begin() && std::prev(it)->CanMerge(*it)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Adjacent axes in ", name, " axes can be merged: ",
          std::prev(it)->ToString(&mesh), ", ", it->ToString(&mesh)));
    }
  }
  return absl::OkStatus();
}

absl::Status VerifyDimShardings(
    absl::Span<const NamedSharding::DimensionSharding> dim_shardings,
    const Mesh& mesh,
    absl::flat_hash_map<int64_t, std::vector<AxisRef>>& seen_axes) {
  for (const NamedSharding::DimensionSharding& ds : dim_shardings) {
    if (ds.axes().empty()) {
      continue;
    }
    for (auto it = ds.axes().begin(); it != ds.axes().end(); ++it) {
      TF_RETURN_IF_ERROR(VerifyAndTrack(*it, mesh, seen_axes));
      if (it != ds.axes().begin() && std::prev(it)->CanMerge(*it)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Adjacent axes in dimension sharding can be merged: ",
            std::prev(it)->ToString(&mesh), ", ", it->ToString(&mesh)));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status VerifyNamedSharding(const NamedSharding& named_sharding) {
  absl::flat_hash_map<int64_t, std::vector<AxisRef>> seen_axes;
  const Mesh& mesh = named_sharding.mesh();
  TF_RETURN_IF_ERROR(
      VerifyDimShardings(named_sharding.dim_shardings(), mesh, seen_axes));
  TF_RETURN_IF_ERROR(VerifySortedAxes(named_sharding.replicated_axes(),
                                      "Replicated", mesh, seen_axes));
  TF_RETURN_IF_ERROR(VerifySortedAxes(named_sharding.unreduced_axes(),
                                      "Unreduced", mesh, seen_axes));
  TF_RETURN_IF_ERROR(VerifySortedAxes(named_sharding.manual_axes(), "Manual",
                                      mesh, seen_axes));
  return absl::OkStatus();
}

}  // namespace xla
