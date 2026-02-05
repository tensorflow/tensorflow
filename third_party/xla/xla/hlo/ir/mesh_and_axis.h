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
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {

class AxisRef;

// C++ representation for corresponding OpSharding::Mesh proto so same
// documentation applies, except device assignment is represented in the array
// format instead of list of device ids to align with various array specific
// queries. `TileAssignment` is used instead of `xla::Array` for optimized array
// representation in the most common iota-based cases.
//
// - device_assignment_.dimensions() represents the axis sizes.
// - device_assignment_.array() represents the list of device IDs.
//
// For maximal mesh, axes_names is empty and device_assignment_ contains the
// single device id.
//
// Example: device_assignment {{3, 0, 2}, {1, 4, 5}} with axes names
// {"data", "model"} represents the mesh ["data"=2, "model"=3].
class Mesh {
 public:
  // Empty mesh
  explicit Mesh() = default;

  // Maximal Mesh
  explicit Mesh(int64_t device_id) : device_assignment_(device_id) {}

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

  // Returns whether this mesh is a maximal-sharding mesh.
  //
  // A maximal-sharding mesh contains an empty axis list and a single device id.
  bool IsMaximal() const {
    return axes_names_.empty() && device_assignment_.num_elements() == 1;
  }

  bool operator==(const Mesh& other) const {
    return device_assignment_ == other.device_assignment_ &&
           axes_names_ == other.axes_names_;
  }

  bool operator!=(const Mesh& other) const { return !(*this == other); }

  bool DeviceAssignmentEquals(const Mesh& other) const {
    return device_assignment_ == other.device_assignment_;
  }

  std::string ToString() const;

  MeshProto ToProto() const;

  static Mesh FromProto(const MeshProto& proto);

  const TileAssignment& device_assignment() const { return device_assignment_; }
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

    template <typename H>
    friend H AbslHashValue(H h, const SubAxis& s) {
      return H::combine(std::move(h), s.pre_size, s.size);
    }
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

  template <typename H>
  friend H AbslHashValue(H h, const AxisRef& a) {
    return H::combine(std::move(h), a.mesh_axis_index_, a.sub_axis_info_);
  }

  std::string ToString(const Mesh* mesh = nullptr) const;

  AxisRefProto ToProto() const;

  static AxisRef FromProto(const AxisRefProto& proto);

  bool CanCoexistWithoutOverlap(const AxisRef& other) const;

  // Returns true if this AxisRef can be merged with the `other`, i.e., they are
  // consecutive sub-axes of same full axis and this sub-axis is major to other.
  bool CanMerge(const AxisRef& other) const;

  // Returns true if this AxisRef is merged with the `other` and this AxisRef
  // is updated, otherwise returns false.
  bool Merge(const AxisRef& other, const Mesh& mesh);

  // Validates that the given mesh is compatible for this axis ref.
  absl::Status Validate(const Mesh& mesh) const;

  int64_t mesh_axis_index() const { return mesh_axis_index_; }
  std::optional<SubAxis> sub_axis_info() const { return sub_axis_info_; }

  int64_t size(const Mesh& mesh) const;

 private:
  absl::Status ValidateAxisRef();
};

std::ostream& operator<<(std::ostream& out, const Mesh& mesh);

std::ostream& operator<<(std::ostream& out, const AxisRef& axis);

bool AxesCanCoexistWithoutOverlap(absl::Span<const AxisRef> axes);

// The span of axes is valid if (1) all axes are valid for the given mesh, and
// (2) the axes can coexist without overlap.
absl::Status ValidateSpanOfAxes(absl::Span<const AxisRef> axes,
                                const Mesh& mesh);

}  // namespace xla

#endif  // XLA_HLO_IR_MESH_AND_AXIS_H_
