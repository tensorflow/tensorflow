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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
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

  MeshProto ToProto() const;

  static Mesh FromProto(const MeshProto& proto);

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

  AxisRefProto ToProto() const;

  static AxisRef FromProto(const AxisRefProto& proto);

  int64_t mesh_axis_index() const { return mesh_axis_index_; }
  std::optional<SubAxis> sub_axis_info() const { return sub_axis_info_; }
};

}  // namespace xla

#endif  // XLA_HLO_IR_MESH_AND_AXIS_H_
