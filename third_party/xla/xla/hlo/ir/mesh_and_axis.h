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

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {

class AxisRef;

// C++ representation for corresponding `OpSharding::Mesh` proto so same
// documentation applies, except device assignment is represented in the array
// format instead of list of device ids to align with various array specific
// queries. Note that `TileAssignment` is used instead of `xla::Array` for
// optimized array representation in iota based cases which is the most common
// case.
//
// Example: device_assignment {{3, 0, 2}, {1, 4, 5}} with axes names {"data",
// "model"} represents the mesh ["data"=2, "model"=3].
class Mesh {
 public:
  // Constructs an iota device assignment mesh with given axes sizes and names.
  //
  // Example: axes_sizes {2, 3} and axes_names {"data", "model"} represent the
  // mesh ["data"=2, "model"=3] with iota device list. We use `TileAssignment`
  // optimized for iota based cases which will not store the entire array.
  explicit Mesh(absl::Span<const int64_t> axes_sizes,
                absl::Span<const absl::string_view> axes_names)
      : Mesh(TileAssignment(axes_sizes), axes_names) {}

  // Constructs a mesh with given device assignment and axes names. This ctor
  // should **ONLY** be used for non-iota based device assignments.
  explicit Mesh(Array<int64_t> device_assignment,
                absl::Span<const absl::string_view> axes_names)
      : Mesh(TileAssignment(std::make_shared<Array<int64_t>>(
                 std::move(device_assignment))),
             axes_names) {}

  explicit Mesh(TileAssignment device_assignment,
                absl::Span<const absl::string_view> axes_names);

  bool operator==(const Mesh& other) const {
    return device_assignment_ == other.device_assignment_ &&
           axes_names_ == other.axes_names_;
  }

  bool operator!=(const Mesh& other) const { return !(*this == other); }

  std::string ToString() const {
    std::string mesh_str = "@mesh";
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
    if (!(iota.has_value() && iota->reshape_dims().size() == 1)) {
      device_assignment_str =
          absl::StrCat("(", device_assignment_.ArrayToString(), ")");
    }
    absl::StrAppend(&mesh_str, "<", absl::StrJoin(formatted_axes_names, ","),
                    ">", device_assignment_str);
    return mesh_str;
  }

  bool DeviceAssignmentEquals(const Mesh& other) const {
    return device_assignment_ == other.device_assignment_;
  }

  MeshProto ToProto() const;

  static Mesh FromProto(const MeshProto& proto);

  TileAssignment device_assignment() const { return device_assignment_; }
  std::vector<std::string> axis_names() const { return axes_names_; }
  absl::Span<const int64_t> axis_sizes() const {
    return device_assignment_.dimensions();
  }
  int64_t axis_size(int64_t axis_index) const {
    return device_assignment_.dim(axis_index);
  }

 private:
  absl::Status ValidateMesh();
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
    int64_t next_pre_size() const { return pre_size * size; }
  };

  // Index corresponding to axis in the mesh. It should be a valid index into
  // `mesh.axes_names_`.
  int64_t mesh_axis_index_;
  std::optional<SubAxis> sub_axis_info_;

 public:
  explicit AxisRef(int64_t mesh_axis_index);

  explicit AxisRef(int64_t mesh_axis_index, SubAxis sub_axis_info);

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

  std::string ToString(const Mesh& mesh) const {
    CHECK_GE(mesh_axis_index_, 0);
    CHECK_LT(mesh_axis_index_, mesh.axis_names().size());
    std::string axis_str = mesh.axis_names()[mesh_axis_index()];
    if (sub_axis_info_.has_value()) {
      absl::StrAppend(&axis_str, ":(", sub_axis_info_->pre_size, ")",
                      sub_axis_info_->size);
    }
    return axis_str;
  }

  AxisRefProto ToProto() const;

  static AxisRef FromProto(const AxisRefProto& proto);

  bool CanCoexist(const AxisRef& other) const;
  bool Overlaps(const AxisRef& other) const;
  bool CanCoexistWithoutOverlap(const AxisRef& other) const;

  // Validates that the given mesh is compatible for this axis ref.
  absl::Status Validate(const Mesh& mesh) const;
  int64_t mesh_axis_index() const { return mesh_axis_index_; }
  std::optional<SubAxis> sub_axis_info() const { return sub_axis_info_; }

 private:
  absl::Status ValidateAxisRef();
};

bool AxesCanCoexistWithoutOverlap(absl::Span<const AxisRef> axes);

// The span of axes is valid if (1) all axes are valid for the given mesh, and
// (2) the axes can coexist without overlap.
absl::Status ValidateSpanOfAxes(absl::Span<const AxisRef> axes,
                                const Mesh& mesh);

}  // namespace xla

#endif  // XLA_HLO_IR_MESH_AND_AXIS_H_
