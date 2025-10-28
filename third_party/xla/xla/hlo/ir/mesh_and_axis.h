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

#ifndef XLA_HLO_IR_MESH_AND_AXIS_H_
#define XLA_HLO_IR_MESH_AND_AXIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {

// C++ representation for corresponding `OpSharding::Mesh` proto so same
// documentation applies, except device assignment is represented in the array
// format instead of list of device ids to align with various array specific
// queries. Note that `TileAssignment` is used instead of `xla::Array` for
// optimized array representation in iota based cases which is the most common
// case.
//
// Example: device_assignment {{3, 0, 2}, {1, 4, 5}} with axes names
// {"data", "model"} represents a 2 * 3 mesh of 6 devices, with "data" axis of
// size 2 and "model" axis of size 3.
class Mesh {
 public:
  explicit Mesh(TileAssignment device_assignment,
                absl::Span<const std::string> axes_names)
      : device_assignment_(std::move(device_assignment)),
        axes_names_(axes_names.begin(), axes_names.end()) {
    CHECK_EQ(device_assignment_.dimensions().size(), axes_names_.size())
        << "Number of axes names must match number of dimensions in the "
           "device assignment.";
  }

  bool operator==(const Mesh& other) const {
    return device_assignment_ == other.device_assignment_ &&
           axes_names_ == other.axes_names_;
  }

  bool operator!=(const Mesh& other) const { return !(*this == other); }

  MeshProto ToProto() const {
    MeshProto proto;
    std::vector<MeshProto::MeshAxis> axes;
    axes.reserve(axes_names_.size());

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

  static Mesh FromProto(const MeshProto& proto) {
    // TODO(b/454008727): Add validators for Mesh and AxisRef FromProto methods.
    std::vector<int64_t> mesh_axis_sizes;
    std::vector<std::string> mesh_axis_names;
    mesh_axis_sizes.reserve(proto.axes_size());
    mesh_axis_names.reserve(proto.axes_size());
    for (const auto& axis : proto.axes()) {
      mesh_axis_sizes.push_back(axis.size());
      mesh_axis_names.push_back(axis.name());
    }

    // If device ids are not specified, create a mesh with iota tiling.
    if (proto.device_ids_size() == 0) {
      TileAssignment device_assignment =
          TileAssignment(IotaTileAssignment::Create(mesh_axis_sizes));
      return Mesh(device_assignment, mesh_axis_names);
    }
    // Otherwise, create a mesh with the specific device id ordering.
    std::vector<int64_t> device_ids(proto.device_ids().begin(),
                                    proto.device_ids().end());
    Array<int64_t> device_ids_array(mesh_axis_sizes);
    absl::c_copy(device_ids, device_ids_array.begin());

    TileAssignment tile_assignment =
        TileAssignment(std::make_shared<Array<int64_t>>(device_ids_array));
    return Mesh(tile_assignment, absl::MakeSpan(mesh_axis_names));
  }

  TileAssignment device_assignment() const { return device_assignment_; }

 private:
  // Dimensions of the `device_assignment_` array correspond to the axes of the
  // mesh.
  TileAssignment device_assignment_;
  // Axes names correspond to names of axes represented by dimensions of
  // `device_assignment_`. Size of `axes_names_` should be equal to the number
  // of dimensions in the device_assignment_.
  std::vector<std::string> axes_names_;
};

// C++ representation for corresponding `OpSharding::AxisRef`proto so same
// documentation applies.
class AxisRef {
 private:
  struct SubAxis {
    int64_t pre_size;
    int64_t size;
  };

  // Index corresponding to axis in the mesh. It should be a valid index into
  // `mesh.axes_names_`.
  int64_t mesh_axis_index_;
  std::optional<SubAxis> sub_axis_info_;

 public:
  explicit AxisRef(int64_t mesh_axis_index)
      : mesh_axis_index_(mesh_axis_index) {}

  explicit AxisRef(int64_t mesh_axis_index, SubAxis sub_axis_info)
      : mesh_axis_index_(mesh_axis_index), sub_axis_info_(sub_axis_info) {}

  explicit AxisRef(int64_t mesh_axis_index, int64_t sub_axis_pre_size,
                   int64_t sub_axis_size)
      : mesh_axis_index_(mesh_axis_index),
        sub_axis_info_({sub_axis_pre_size, sub_axis_size}) {}

  bool operator==(const xla::AxisRef& other) const {
    if (mesh_axis_index_ != other.mesh_axis_index_) {
      return false;
    }
    if (sub_axis_info_.has_value() != other.sub_axis_info_.has_value()) {
      return false;
    }
    if (sub_axis_info_.has_value()) {
      return sub_axis_info_->pre_size == other.sub_axis_info_->pre_size &&
             sub_axis_info_->size == other.sub_axis_info_->size;
    }
    return true;
  }

  bool operator!=(const xla::AxisRef& other) const { return !(*this == other); }

  AxisRefProto ToProto() const {
    AxisRefProto proto;
    proto.set_mesh_axis_index(mesh_axis_index_);
    if (sub_axis_info_.has_value()) {
      proto.mutable_sub_axis_info()->set_pre_size(sub_axis_info_->pre_size);
      proto.mutable_sub_axis_info()->set_size(sub_axis_info_->size);
    }
    return proto;
  }

  static AxisRef FromProto(const AxisRefProto& proto) {
    AxisRef axis_ref(proto.mesh_axis_index());
    if (proto.has_sub_axis_info()) {
      axis_ref.sub_axis_info_ = {proto.sub_axis_info().pre_size(),
                                 proto.sub_axis_info().size()};
    }
    return axis_ref;
  }

  int64_t mesh_axis_index() const { return mesh_axis_index_; }
  std::optional<SubAxis> sub_axis_info() const { return sub_axis_info_; }
};

}  // namespace xla

#endif  // XLA_HLO_IR_MESH_AND_AXIS_H_
