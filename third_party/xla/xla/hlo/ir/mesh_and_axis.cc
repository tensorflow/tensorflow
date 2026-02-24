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

#include "xla/hlo/ir/mesh_and_axis.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::Status Mesh::Validate() {
  if (device_assignment_.num_dimensions() == 0) {
    // Empty mesh or maximal mesh.
    if (device_assignment_.num_elements() <= 1) {
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Non-maximal mesh must have exactly 1 device id. Number of "
        "device ids: ",
        device_assignment_.num_elements()));
  }

  if (device_assignment_.num_dimensions() != axes_names_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of axes names must match number of dimensions in the device "
        "assignment. Number of axes names: ",
        axes_names_.size(),
        ", Number of dimensions: ", device_assignment_.dimensions().size()));
  }

  absl::flat_hash_set<std::string> seen_axis_names;
  for (const std::string& axis_name : axes_names_) {
    if (!seen_axis_names.insert(axis_name).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mesh has duplicate axis names. Duplicate axis name: ", axis_name));
    }
    int64_t value;
    if (absl::SimpleAtoi(axis_name, &value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Mesh axis name cannot be an integer to avoid confusion "
                       "with axis indices: ",
                       axis_name));
    }
  }

  // Validate device ids are permutation of iota in non-iota cases.
  if (device_assignment_.iota().has_value()) {
    return absl::OkStatus();
  }
  std::vector<int64_t> device_ids(device_assignment_.array().begin(),
                                  device_assignment_.array().end());
  for (int64_t device_id : device_ids) {
    if (device_id < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mesh device ids must be non-negative. Device id: ", device_id));
    }
  }
  std::vector<int64_t> iota(device_ids.size());
  std::iota(iota.begin(), iota.end(), 0);

  // For non-iota cases the device ids should be a non-identity permutation
  // of iota.
  if (device_ids == iota) {
    return absl::InvalidArgumentError(
        "Non-iota device assignment has iota device id list [0,1,2,3...].");
  }
  absl::c_sort(device_ids);
  if (device_ids != iota) {
    return absl::InvalidArgumentError(
        "Device ids must be a permutation of [0,1,2,3...].");
  }
  return absl::OkStatus();
}

Mesh::Mesh(TileAssignment device_assignment,
           absl::Span<const absl::string_view> axes_names)
    : device_assignment_(std::move(device_assignment)),
      axes_names_(axes_names.begin(), axes_names.end()) {
  CHECK_OK(Validate());
}

std::string Mesh::ToString() const {
  if (IsMaximal()) {
    return absl::StrCat(
        "maximal_mesh[device_id=", device_assignment_.array()(0), "]");
  }

  std::string mesh_str = "mesh";
  // Add the mesh axes names and sizes.
  std::vector<std::string> formatted_axes_names;
  formatted_axes_names.reserve(axes_names_.size());
  for (int64_t i = 0; i < axes_names_.size(); ++i) {
    formatted_axes_names.push_back(
        absl::StrCat(axes_names_[i], "=", device_assignment_.dim(i)));
  }

  // Add the device assignment if it is not an iota case.
  std::optional<IotaTileAssignment> iota = device_assignment_.iota();
  std::string device_assignment_str = "";
  bool simple_iota = iota.has_value() && iota->reshape_dims().size() == 1;
  if (!simple_iota && device_assignment_.num_elements() != 0) {
    device_assignment_str =
        absl::StrCat(", device_ids=(", device_assignment_.ArrayToString(), ")");
  }
  absl::StrAppend(&mesh_str, "[", absl::StrJoin(formatted_axes_names, ","), "]",
                  device_assignment_str);
  return mesh_str;
}

MeshProto Mesh::ToProto() const {
  MeshProto proto;

  if (num_axes() == 0) {
    if (device_assignment_.num_elements() == 0) {
      return MeshProto();
    }
    // Maximal mesh
    // TODO(b/454008727): Validate device_ids_size is 1.
    proto.add_device_ids(*device_assignment_.array().begin());
    return proto;
  }

  std::vector<MeshProto::MeshAxis> axes;
  axes.reserve(num_axes());

  for (auto [name, size] :
       llvm::zip_equal(axes_names_, device_assignment_.dimensions())) {
    MeshProto::MeshAxis axis;
    axis.set_name(name);
    axis.set_size(size);
    axes.push_back(std::move(axis));
  }
  proto.mutable_axes()->Assign(axes.begin(), axes.end());

  std::optional<IotaTileAssignment> iota = device_assignment_.iota();
  // Only add device ids for non-iota cases.
  if (!(iota.has_value() && iota->reshape_dims().size() == 1)) {
    proto.mutable_device_ids()->Assign(device_assignment_.array().begin(),
                                       device_assignment_.array().end());
  }
  return proto;
}

Mesh Mesh::FromProto(const MeshProto& proto) {
  if (proto.axes_size() == 0) {
    if (proto.device_ids_size() == 0) {
      return Mesh();
    }
    // Maximal mesh
    CHECK_EQ(proto.device_ids_size(), 1)
        << "Maximal mesh must have exactly 1 device id.";
    return Mesh(proto.device_ids(0));
  }

  std::vector<int64_t> mesh_axis_sizes;
  std::vector<absl::string_view> mesh_axis_names;
  mesh_axis_sizes.reserve(proto.axes_size());
  mesh_axis_names.reserve(proto.axes_size());
  for (const auto& axis : proto.axes()) {
    CHECK_GT(axis.size(), 0) << "Mesh axis size must be positive.";
    mesh_axis_sizes.push_back(axis.size());
    mesh_axis_names.push_back(axis.name());
  }
  absl::Span<const absl::string_view> mesh_axis_names_span =
      absl::MakeSpan(mesh_axis_names);

  // If device ids are not specified, create a mesh with iota tiling.
  if (proto.device_ids_size() == 0) {
    TileAssignment device_assignment =
        TileAssignment(IotaTileAssignment::Create(mesh_axis_sizes));
    return Mesh(device_assignment, mesh_axis_names_span);
  }
  // Otherwise, create a mesh with the specific device id ordering.
  std::vector<int64_t> device_ids(proto.device_ids().begin(),
                                  proto.device_ids().end());
  Array<int64_t> device_ids_array(mesh_axis_sizes);
  CHECK_EQ(device_ids.size(), device_ids_array.num_elements())
      << "Number of device ids must match the product of mesh axis sizes.";
  absl::c_copy(device_ids, device_ids_array.begin());

  TileAssignment tile_assignment =
      TileAssignment(std::make_shared<Array<int64_t>>(device_ids_array));
  return Mesh(tile_assignment, mesh_axis_names_span);
}

bool Mesh::ContainsAllMeshAxesInOrder(absl::Span<const AxisRef> axes) const {
  if (num_axes() != axes.size()) {
    return false;
  }
  for (int i = 0; i < axes.size(); ++i) {
    if (axes[i].sub_axis_info().has_value() || axes[i].mesh_axis_index() != i) {
      return false;
    }
  }
  return true;
}

std::string AxisRef::ToString(const Mesh* mesh) const {
  // TODO(b/474013054): Remove these checks if they have significant overhead.
  CHECK_GE(mesh_axis_index_, 0);
  if (mesh) {
    CHECK_LT(mesh_axis_index_, mesh->num_axes());
  }
  std::string axis_str = mesh ? mesh->axis_names()[mesh_axis_index_]
                              : std::to_string(mesh_axis_index_);
  if (sub_axis_info_.has_value()) {
    absl::StrAppend(&axis_str, ":(", sub_axis_info_->pre_size, ")",
                    sub_axis_info_->size);
  }
  return axis_str;
}

AxisRefProto AxisRef::ToProto() const {
  AxisRefProto proto;
  proto.set_mesh_axis_index(mesh_axis_index_);
  if (sub_axis_info_.has_value()) {
    proto.mutable_sub_axis_info()->set_pre_size(sub_axis_info_->pre_size);
    proto.mutable_sub_axis_info()->set_size(sub_axis_info_->size);
  }
  return proto;
}

AxisRef AxisRef::FromProto(const AxisRefProto& proto) {
  if (proto.has_sub_axis_info()) {
    return AxisRef(proto.mesh_axis_index(),
                   SubAxis{proto.sub_axis_info().pre_size(),
                           proto.sub_axis_info().size()});
  }
  return AxisRef(proto.mesh_axis_index());
}

AxisRef::AxisRef(int64_t mesh_axis_index) : mesh_axis_index_(mesh_axis_index) {}

AxisRef::AxisRef(int64_t mesh_axis_index, SubAxis sub_axis_info)
    : mesh_axis_index_(mesh_axis_index), sub_axis_info_(sub_axis_info) {
  CHECK_GT(sub_axis_info_->pre_size, 0) << "sub-axis pre-size must be >= 1";
  CHECK_GT(sub_axis_info_->size, 1) << "sub-axis size must be > 1";
}

bool AxisRef::CanCoexistWithoutOverlap(const AxisRef& other) const {
  // Check if the axes are on different mesh dimensions. If so, they can always
  // coexist and never overlap.
  if (mesh_axis_index() != other.mesh_axis_index()) {
    return true;
  }

  // If one AxisRef is a full axis it will always overlap the other axis on the
  // same dimension.
  if (!sub_axis_info_.has_value() || !other.sub_axis_info_.has_value()) {
    return false;
  }

  const SubAxis& this_sub_axis = sub_axis_info_.value();
  const SubAxis& other_sub_axis = other.sub_axis_info_.value();

  int64_t this_pre_size = this_sub_axis.pre_size;
  int64_t other_pre_size = other_sub_axis.pre_size;
  int64_t this_next_pre_size = this_sub_axis.next_pre_size();
  int64_t other_next_pre_size = other_sub_axis.next_pre_size();

  // Check for overlapping sub-axes
  bool overlaps = (this_next_pre_size > other_pre_size) &&
                  (other_next_pre_size > this_pre_size);
  if (overlaps) {
    return false;
  }
  // Assert that sub-axes can coexist.
  auto [min_pre_size, max_pre_size] =
      std::minmax(this_pre_size, other_pre_size);
  auto [min_next_pre_size, max_next_pre_size] =
      std::minmax(this_next_pre_size, other_next_pre_size);

  // Sub-axes don't overlap, check if the gap is valid.
  return max_pre_size % min_next_pre_size == 0;
}

bool AxisRef::CanMerge(const AxisRef& other) const {
  if (mesh_axis_index_ != other.mesh_axis_index()) {
    return false;
  }
  if (!sub_axis_info_.has_value() || !other.sub_axis_info_.has_value()) {
    return false;
  }
  return sub_axis_info_->next_pre_size() == other.sub_axis_info_->pre_size;
}

bool AxisRef::Merge(const AxisRef& other, const Mesh& mesh) {
  if (!CanMerge(other)) {
    return false;
  }

  sub_axis_info_->size *= other.sub_axis_info_->size;
  if (sub_axis_info_->size == mesh.axis_size(mesh_axis_index_)) {
    assert(sub_axis_info_->pre_size == 1);
    sub_axis_info_ = std::nullopt;
  }
  return true;
}

absl::Status AxisRef::Validate(const Mesh& mesh) const {
  if (mesh_axis_index_ >= mesh.num_axes()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Axis index must be less than number of axes. Axis index: ",
        mesh_axis_index_, ", Number of axes: ", mesh.axis_names().size()));
  }
  if (!sub_axis_info_.has_value()) {
    return absl::OkStatus();
  }

  int64_t axis_size = mesh.axis_size(mesh_axis_index_);
  if (axis_size % sub_axis_info_->next_pre_size() != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Sub-axis next_pre_size must divide the full axis size. Next "
        "pre-size: ",
        sub_axis_info_->next_pre_size(), ", Axis size: ", axis_size));
  }
  if (sub_axis_info_->size >= axis_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Sub-axis size must be strictly less than the full axis size. Sub-axis "
        "size: ",
        sub_axis_info_->size, ", Axis size: ", axis_size));
  }
  return absl::OkStatus();
}

int64_t AxisRef::size(const Mesh& mesh) const {
  if (sub_axis_info_.has_value()) {
    return sub_axis_info_->size;
  }

  return mesh.axis_size(mesh_axis_index_);
}

std::ostream& operator<<(std::ostream& out, const Mesh& mesh) {
  return out << mesh.ToString();
}

std::ostream& operator<<(std::ostream& out, const AxisRef& axis) {
  return out << axis.ToString();
}

bool AxesCanCoexistWithoutOverlap(absl::Span<const AxisRef> axes) {
  if (axes.size() < 2) {
    return true;
  }
  for (auto it1 = axes.begin(); it1 != std::prev(axes.end()); ++it1) {
    for (auto it2 = std::next(it1); it2 != axes.end(); ++it2) {
      if (!it1->CanCoexistWithoutOverlap(*it2)) {
        return false;
      }
    }
  }
  return true;
}

absl::Status ValidateSpanOfAxes(absl::Span<const AxisRef> axes,
                                const Mesh& mesh,
                                bool allow_mergeable_neighbors) {
  if (axes.empty()) {
    return absl::OkStatus();
  }
  for (const AxisRef& axis : axes) {
    TF_RETURN_IF_ERROR(axis.Validate(mesh));
  }
  if (!AxesCanCoexistWithoutOverlap(axes)) {
    return absl::InvalidArgumentError("Axes cannot coexist or axes overlap.");
  }
  if (allow_mergeable_neighbors) {
    return absl::OkStatus();
  }
  for (auto it = axes.begin(); it != std::prev(axes.end()); ++it) {
    if (it->CanMerge(*std::next(it))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Adjacent axes in dimension sharding can be merged: ",
          it->ToString(&mesh), ", ", std::next(it)->ToString(&mesh)));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
