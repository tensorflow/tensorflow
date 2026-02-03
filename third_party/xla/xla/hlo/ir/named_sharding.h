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
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
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

    bool operator!=(const DimensionSharding& other) const {
      return !(*this == other);
    }

    std::string ToString(const Mesh* mesh = nullptr) const;

    NamedShardingProto::DimensionSharding ToProto() const;
    static DimensionSharding FromProto(
        const NamedShardingProto::DimensionSharding& proto);

    // Note that by default we assume closed sharding.
    explicit DimensionSharding() : is_closed_(true) {};

    explicit DimensionSharding(absl::Span<const AxisRef> axes, bool is_closed)
        : axes_(axes.begin(), axes.end()), is_closed_(is_closed) {}

    absl::Span<const AxisRef> axes() const { return axes_; }
    bool is_closed() const { return is_closed_; }

    int64_t getShardedSize(const Mesh& mesh) const;

    // Appends `other` to this dimension sharding. This function assumes that
    // both the dimension shardings correspond to the same mesh represented by
    // `mesh` argument.
    void Append(const DimensionSharding& other, const Mesh& mesh);

    // Slice axes of size `slice_size` from this dimension sharding and update
    // this dimension sharding with remaining axes.
    //
    // Axes can only be sliced from major to minor.
    // For example, given an input {a, b, c}, we can slice it as
    // 1. {a} + {b, c}
    // 2. {a, b} + {c}
    // or other slices with sub-axis, we cannot slice it to {a, c} + {b}.
    std::optional<DimensionSharding> Slice(const Mesh& mesh,
                                           int64_t slice_size);

    // Returns true if this dimension sharding is a prefix of `other`.
    //
    // This means that the sequence of axes in this sharding matches the
    // beginning of the sequence of axes in `other` sharding.
    bool IsPrefixOf(const DimensionSharding& other, const Mesh& mesh,
                    const Mesh& other_mesh) const;

   private:
    std::vector<AxisRef> axes_;
    bool is_closed_;
  };

  // Shardings using mesh with similar device assignment should compare equal
  bool operator==(const NamedSharding& other) const {
    return mesh_.DeviceAssignmentEquals(other.mesh_) &&
           dim_shardings_ == other.dim_shardings_ &&
           replicated_axes_ == other.replicated_axes_ &&
           unreduced_axes_ == other.unreduced_axes_ &&
           manual_axes_ == other.manual_axes_;
  }

  bool operator!=(const NamedSharding& other) const {
    return !(*this == other);
  }

  std::string ToString(bool include_metadata = false) const;

  NamedShardingProto ToProto() const;
  static NamedSharding FromProto(const NamedShardingProto& proto);

  explicit NamedSharding(Mesh mesh,
                         absl::Span<const DimensionSharding> dim_shardings = {},
                         absl::Span<const AxisRef> replicated_axes = {},
                         absl::Span<const AxisRef> unreduced_axes = {},
                         absl::Span<const AxisRef> manual_axes = {},
                         absl::Span<const OpMetadata> metadata = {});

  const Mesh& mesh() const { return mesh_; }
  absl::Span<const DimensionSharding> dim_shardings() const {
    return dim_shardings_;
  }
  const DimensionSharding& dim_sharding(int64_t dim) const {
    return dim_shardings_[dim];
  }
  absl::Span<const AxisRef> replicated_axes() const { return replicated_axes_; }
  absl::Span<const AxisRef> unreduced_axes() const { return unreduced_axes_; }
  absl::Span<const AxisRef> manual_axes() const { return manual_axes_; }
  absl::Span<const OpMetadata> metadata() const { return metadata_; }

  // Returns number of dimensions.
  int64_t num_dimensions() const { return dim_shardings_.size(); }

  // Returns size of the given dimension.
  int64_t dimension(int64_t dim) const {
    return dim_shardings_[dim].getShardedSize(mesh_);
  }

  // Returns all sharding dimensions.
  absl::Span<const int64_t> dimensions() const { return sharded_sizes_; }

  // Returns the total number of devices used by sharding.
  int64_t num_devices() const {
    return mesh_.device_assignment().num_elements();
  }

  bool IsReplicated() const {
    return !IsMaximal() && AllDimShardingsEmpty(dim_shardings_) &&
           unreduced_axes_.empty() && manual_axes_.empty();
  }

  bool IsMaximal() const { return mesh_.IsMaximal(); }

  bool IsManual() const {
    return !IsMaximal() && AllDimShardingsEmpty(dim_shardings_) &&
           replicated_axes_.empty() && unreduced_axes_.empty() &&
           mesh_.ContainsAllMeshAxesInOrder(manual_axes_);
  }

  bool IsUnreduced() const {
    return !IsMaximal() && AllDimShardingsEmpty(dim_shardings_) &&
           replicated_axes_.empty() && manual_axes_.empty() &&
           mesh_.ContainsAllMeshAxesInOrder(unreduced_axes_);
  }

  // Returns true if the tile size is the same as the input size.
  //
  // This checks for both replicated and maximal sharding, as in both cases tile
  // size is same as input size.
  bool IsTileMaximal() const { return IsReplicated() || IsMaximal(); }

  // Creates a sharding with empty mesh and no sharding axes depicting it is
  // replicated across all devices.
  static NamedSharding Replicate(absl::Span<const OpMetadata> metadata = {}) {
    return NamedSharding(/*mesh=*/Mesh(), /*dim_shardings=*/{},
                         /*replicated_axes=*/{},
                         /*unreduced_axes=*/{},
                         /*manual_axes=*/{}, metadata);
  }

  static NamedSharding MaximalSharding(
      int64_t device_id, absl::Span<const OpMetadata> metadata = {}) {
    return NamedSharding(Mesh(device_id), /*dim_shardings=*/{},
                         /*replicated_axes=*/{},
                         /*unreduced_axes=*/{},
                         /*manual_axes=*/{}, metadata);
  }

 private:
  friend class HloSharding;

  void InitShardedSizes() {
    sharded_sizes_.reserve(dim_shardings_.size());
    for (const DimensionSharding& dim_sharding : dim_shardings_) {
      sharded_sizes_.push_back(dim_sharding.getShardedSize(mesh_));
    }
  }

  bool AllDimShardingsEmpty(
      absl::Span<const DimensionSharding> dim_shardings) const {
    return absl::c_all_of(dim_shardings, [](const DimensionSharding& s) {
      return s.axes().empty();
    });
  }

  std::vector<DimensionSharding> CanonicalizedDimShardings(
      absl::Span<const DimensionSharding> dim_shardings) const {
    if (AllDimShardingsEmpty(dim_shardings)) {
      return {};
    }
    return std::vector<DimensionSharding>(dim_shardings.begin(),
                                          dim_shardings.end());
  }

  const TileAssignment& device_assignment() const {
    return mesh_.device_assignment();
  }

  Mesh mesh_;
  std::vector<DimensionSharding> dim_shardings_;
  std::vector<AxisRef> replicated_axes_;
  std::vector<AxisRef> unreduced_axes_;
  std::vector<AxisRef> manual_axes_;
  std::vector<OpMetadata> metadata_;

  // Stores sharded sizes for each dimension. Required to maintain backward
  // compatibility with existing `HloSharding::dimensions()` implementation
  // returning a span.
  // Once we make API change for `HloSharding::dimensions()` to return a vector,
  // we can remove this field.
  std::vector<int64_t> sharded_sizes_;
};

std::ostream& operator<<(std::ostream& out,
                         const NamedSharding::DimensionSharding& sharding);

std::ostream& operator<<(std::ostream& out, const NamedSharding& sharding);

// Verifies that the `NamedSharding` is valid.
// Checks:
// - All axis indices are within mesh bounds.
// - All sub-axes are valid (pre-size * size divides the full axis size).
// - For a single vector of axes, mergeable neighbors is not allowed.
// - For the concat(all axes), we check (1) no overlap, and (2) all axes can
//   co-exist.
// - Replicated axes and unreduced axes are sorted by mesh axis index and
//   sub-axis pre-size.
absl::Status VerifyNamedSharding(const NamedSharding& named_sharding);

// Contains test only helper functions.
namespace test_utils {
// Construct sharding with given mesh. `dim_shardings`, `replicated_axes`,
// `unreduced_axes`, and `manual_axes` refer to axis names in the mesh.
NamedSharding FromAxisNames(
    Mesh mesh, absl::Span<const std::vector<std::string>> dim_shardings,
    absl::Span<const std::string> replicated_axes = {},
    absl::Span<const std::string> unreduced_axes = {},
    absl::Span<const std::string> manual_axes = {},
    absl::Span<const OpMetadata> metadata = {});
}  // namespace test_utils

}  // namespace xla

#endif  // XLA_HLO_IR_NAMED_SHARDING_H_
