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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {

MeshProto Mesh::ToProto() const {
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

Mesh Mesh::FromProto(const MeshProto& proto) {
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
  AxisRef axis_ref(proto.mesh_axis_index());
  if (proto.has_sub_axis_info()) {
    axis_ref.sub_axis_info_ = {proto.sub_axis_info().pre_size(),
                               proto.sub_axis_info().size()};
  }
  return axis_ref;
}

}  // namespace xla
