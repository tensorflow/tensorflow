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

#ifndef XLA_HLO_IR_NAMED_SHARDING_H_
#define XLA_HLO_IR_NAMED_SHARDING_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {

// C++ representation for corresponding `OpSharding::NamedSharding` proto so
// same documentation applies.
class NamedSharding {
 public:
  class DimensionSharding {
   public:
    bool operator==(const DimensionSharding& other) const {
      return axes_ == other.axes_ && is_closed_ == other.is_closed_;
    }

    // TODO: Confirm default value for is_closed = true ? to match JAX behavior
    explicit DimensionSharding() : is_closed_(true) {};

    explicit DimensionSharding(std::vector<AxisRef> axes, bool is_closed)
        : axes_(std::move(axes)), is_closed_(is_closed) {}

    absl::Span<const AxisRef> axes() const { return axes_; }

   private:
    std::vector<AxisRef> axes_;
    bool is_closed_;
  };

  // Shardings using mesh with similar device assignment should compare equal
  bool operator==(const NamedSharding& other) const {
    return mesh_.DeviceAssignmentEquals(other.mesh_) &&
           dim_shardings_ == other.dim_shardings_ &&
           replicated_axes_ == other.replicated_axes_ &&
           unreduced_axes_ == other.unreduced_axes_;
  }

  bool operator!=(const NamedSharding& other) const {
    return !(*this == other);
  }

  // TODO(b/456212087): Add validation checks
  explicit NamedSharding(Mesh mesh,
                         absl::Span<const DimensionSharding> dim_shardings = {},
                         absl::Span<const AxisRef> replicated_axes = {},
                         absl::Span<const AxisRef> unreduced_axes = {},
                         absl::Span<const OpMetadata> metadata = {})
      : mesh_(std::move(mesh)),
        dim_shardings_(dim_shardings.begin(), dim_shardings.end()),
        replicated_axes_(replicated_axes.begin(), replicated_axes.end()),
        unreduced_axes_(unreduced_axes.begin(), unreduced_axes.end()),
        metadata_(metadata.begin(), metadata.end()) {
    // TODO: We can do a optimization here if all dim_shardings are empty we can
    // just not have all empty vectors or should we do it at user side itself ?
  }

  const Mesh& mesh() const { return mesh_; }
  absl::Span<const DimensionSharding> dim_shardings() const {
    return dim_shardings_;
  }
  absl::Span<const AxisRef> replicated_axes() const { return replicated_axes_; }
  absl::Span<const AxisRef> unreduced_axes() const { return unreduced_axes_; }
  absl::Span<const OpMetadata> metadata() const { return metadata_; }

  // Construct sharding with given mesh and dim_shardings, replicated_axes,
  // unreduced_axes referring to axis names in the mesh.
  explicit NamedSharding(
      Mesh mesh, absl::Span<const std::vector<std::string>> dim_shardings,
      absl::Span<const std::string> replicated_axes = {},
      absl::Span<const std::string> unreduced_axes = {},
      absl::Span<const OpMetadata> metadata = {})
      : mesh_(std::move(mesh)), metadata_(metadata.begin(), metadata.end()) {
    std::map<std::string, int64_t> mesh_axis_to_index;
    for (int64_t i = 0; i < mesh_.axis_names().size(); ++i) {
      mesh_axis_to_index[mesh_.axis_names()[i]] = i;
    }

    dim_shardings_.reserve(dim_shardings.size());
    for (const auto& axes_for_dim : dim_shardings) {
      std::vector<AxisRef> axis_refs;
      axis_refs.reserve(axes_for_dim.size());
      for (const std::string& axis_name : axes_for_dim) {
        CHECK(mesh_axis_to_index.contains(axis_name))
            << "Axis " << axis_name << " not found in mesh "
            << mesh_.ToString();
        axis_refs.push_back(AxisRef(mesh_axis_to_index[axis_name]));
      }
      dim_shardings_.push_back(
          DimensionSharding(std::move(axis_refs), /*is_closed=*/true));
    }

    replicated_axes_.reserve(replicated_axes.size());
    for (const std::string& axis_name : replicated_axes) {
      CHECK(mesh_axis_to_index.contains(axis_name))
          << "Axis " << axis_name << " not found in mesh " << mesh_.ToString();
      replicated_axes_.push_back(AxisRef(mesh_axis_to_index[axis_name]));
    }

    unreduced_axes_.reserve(unreduced_axes.size());
    for (const std::string& axis_name : unreduced_axes) {
      CHECK(mesh_axis_to_index.contains(axis_name))
          << "Axis " << axis_name << " not found in mesh " << mesh_.ToString();
      unreduced_axes_.push_back(AxisRef(mesh_axis_to_index[axis_name]));
    }
  }

 private:
  friend class HloSharding;

  // Creates a sharding with empty mesh and no sharding axes depicting it is
  // replicated across all devices.
  static NamedSharding Replicate(absl::Span<const OpMetadata> metadata = {}) {
    return NamedSharding(
        /*mesh=*/Mesh(),
        /*dim_shardings=*/absl::Span<const DimensionSharding>{},
        /*replicated_axes=*/{},
        /*unreduced_axes=*/{}, metadata);
  }

  static NamedSharding MaximalSharding(
      int64_t device_id, absl::Span<const OpMetadata> metadata = {}) {
    return NamedSharding(
        Mesh(device_id),
        /*dim_shardings=*/absl::Span<const DimensionSharding>{},
        /*replicated_axes=*/{},
        /*unreduced_axes=*/{}, metadata);
  }

  bool IsReplicated() const {
    return !IsMaximal() &&
           absl::c_all_of(dim_shardings_, [](const DimensionSharding& s) {
             return s.axes().empty();
           });
  }

  bool IsMaximal() const { return mesh_.IsMaximal(); }

  // Returns true if the tile size is the same as the input size.
  //
  // This checks for both replicated and maximal sharding, as in both cases tile
  // size is same as input size.
  bool IsTileMaximal() const { return IsReplicated() || IsMaximal(); }

  const TileAssignment& device_assignment() const {
    return mesh_.device_assignment();
  }

  Mesh mesh_;
  std::vector<DimensionSharding> dim_shardings_;
  std::vector<AxisRef> replicated_axes_;
  std::vector<AxisRef> unreduced_axes_;
  std::vector<OpMetadata> metadata_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_NAMED_SHARDING_H_
