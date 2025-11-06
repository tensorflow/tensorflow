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
#include <vector>

#include "xla/hlo/ir/tile_assignment.h"

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
};

}  // namespace xla

#endif  // XLA_HLO_IR_MESH_AND_AXIS_H_
