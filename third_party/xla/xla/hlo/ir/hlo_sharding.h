/* Copyright 2017 The OpenXLA Authors.

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

// HLO shardings describe how an HLO instruction is split across multiple
// computations.

#ifndef XLA_HLO_IR_HLO_SHARDING_H_
#define XLA_HLO_IR_HLO_SHARDING_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"  // IWYU pragma: export
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// HLO shardings describe how an HLO instruction is split across multiple
// computations.
class HloSharding {
 public:
  // Creates a trivial sharding that replicates a maximal tile across all
  // devices.
  static HloSharding Replicate(absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(/*manual=*/false, /*replicated=*/true, /*unknown=*/false,
                       metadata);
  }

  // Creates a sharding that represents the op is manually partitioned.
  static HloSharding Manual(absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(/*manual=*/true, /*replicated=*/false, /*unknown=*/false,
                       metadata);
  }

  // Creates a sharding that represents the op has a placeholder sharding.
  static HloSharding Unknown(absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(/*manual=*/false, /*replicated=*/false, /*unknown=*/true,
                       metadata);
  }

  // Creates a sharding that emulates device placement; a tile shape equal to
  // the input shape (one tile) assigned to a single device.
  static HloSharding AssignDevice(int64_t device_id,
                                  absl::Span<const OpMetadata> metadata = {});

  // Creates a new sharding which splits a shape into tiles amongst the devices
  // specified by `tile_assignment`.
  static HloSharding Tile(TileAssignment tile_assignment,
                          absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(tile_assignment, /*replicate_on_last_tile_dim=*/false,
                       metadata);
  }
  static HloSharding Tile(Array<int64_t> tile_assignment,
                          absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(TileAssignment(std::make_shared<const Array<int64_t>>(
                           std::move(tile_assignment))),
                       /*replicate_on_last_tile_dim=*/false, metadata);
  }
  // Similar to `Tile` but use IotaTileAssignment format.
  static HloSharding IotaTile(absl::Span<const int64_t> tile_assignment_dims,
                              absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(TileAssignment(tile_assignment_dims),
                       /*replicate_on_last_tile_dim=*/false, metadata);
  }
  static HloSharding IotaTile(absl::Span<const int64_t> tile_assignment_dims,
                              absl::Span<const int64_t> reshape_dims,
                              absl::Span<const int> transpose_perm,
                              absl::Span<const OpMetadata> metadata = {}) {
    return HloSharding(
        TileAssignment(tile_assignment_dims, reshape_dims, transpose_perm),
        /*replicate_on_last_tile_dim=*/false, metadata);
  }

  // Creates a partially replicated tiled sharding with device-level tile
  // assignment, where the last dimension is the additional replication
  // dimension. Replication group members will be sorted.
  static HloSharding PartialTile(
      const TileAssignment& tile_assignment_last_dim_replicate,
      absl::Span<const OpMetadata> metadata = {});
  static HloSharding PartialTile(
      Array<int64_t> tile_assignment_last_dim_replicate,
      absl::Span<const OpMetadata> metadata = {}) {
    return PartialTile(TileAssignment(std::make_shared<const Array<int64_t>>(
                           std::move(tile_assignment_last_dim_replicate))),
                       metadata);
  }

  // Creates a subgroup sharding with device-level tile assignment, the
  // sharding type of each subgroup is defined by subgroup_types. When creating
  // the HloSharding, subgroup dims of the same type will be merged.
  static HloSharding Subgroup(const TileAssignment& tile_assignment,
                              absl::Span<const OpSharding::Type> subgroup_types,
                              absl::Span<const OpMetadata> metadata = {});
  static HloSharding Subgroup(Array<int64_t> tile_assignment,
                              absl::Span<const OpSharding::Type> subgroup_types,
                              absl::Span<const OpMetadata> metadata = {}) {
    return Subgroup(
        TileAssignment(std::make_shared<const Array<int64_t>>(tile_assignment)),
        subgroup_types, metadata);
  }

  // Creates a new sharding which splits a one-dimensional input shape into
  // `num_tiles` tiles.
  static HloSharding Tile1D(const Shape& input_shape, int64_t num_tiles,
                            absl::Span<const OpMetadata> metadata = {});

  // Creates a new sharding for a tuple type. The given ShapeTree must have
  // elements for every leaf shape contained in the tuple.
  static HloSharding Tuple(const ShapeTree<HloSharding>& sub_shardings);

  // Creates a new sharding for a tuple type. The number of elements in
  // shardings must match the number of leaf nodes in tuple_shape. For
  // empty tuples, the shardings array must have one element.
  static HloSharding Tuple(const Shape& tuple_shape,
                           absl::Span<const HloSharding> shardings);

  // Creates a new sharding for a flat tuple type.
  static HloSharding FlatTuple(std::vector<HloSharding> sub_shardings) {
    return HloSharding(std::move(sub_shardings));
  }

  // Creates a new sharding for a tuple type, with a single input sharding
  // repeated on each leaf.
  static HloSharding SingleTuple(const Shape& tuple_shape,
                                 const HloSharding& sharding);

  // If shape is an array, returns sharding, otherwise returns the tuple shaped
  // sharding with all the leaf nodes having the same input sharding.
  static HloSharding Single(const Shape& shape, const HloSharding& sharding);

  // Create a new sharding from a protobuf OpSharding.
  static absl::StatusOr<HloSharding> FromProto(const OpSharding& proto);

  // Checks whether device is a reserved device number. A reserved device number
  // has usually a special meaning, with dedicated handling logic.
  static bool IsReservedDevice(int64_t device) { return device < 0; }

  OpSharding ToProto() const;

  // Prints the string representation of this sharding.Note that this string
  // canonically has outer curly braces, e.g. "{replicated}".
  void Print(Printer* printer, bool include_metadata = false) const;

  // Returns the content printed by Print as a string.
  std::string ToString(bool include_metadata = false) const;

  // Validate that this sharding can be applied to a tensor with shape `shape`.
  absl::Status Validate(const Shape& shape,
                        std::optional<int64_t> num_devices = {}) const;

  // Returns true if the sharding has tuple type.
  bool IsTuple() const { return tuple_; }

  // Returns true if the sharding is trivial: replicate on all devices.
  bool IsReplicated() const {
    if (!IsTuple()) {
      return replicated_;
    }
    return absl::c_all_of(
        tuple_elements_, [](const HloSharding& s) { return s.IsReplicated(); });
  }
  bool IsReplicatedLeaf() const {
    DCHECK(!IsTuple());
    return replicated_;
  }

  // Returns true if the tile size is the same as the input size.
  bool IsTileMaximal() const {
    if (!IsTuple()) {
      return maximal_;
    }
    return absl::c_all_of(tuple_elements_, [](const HloSharding& s) {
      return s.IsTileMaximal();
    });
  }
  bool IsTileMaximalLeaf() const {
    DCHECK(!IsTuple());
    return maximal_;
  }

  // Returns whether the sharding represents manual partitioning.
  bool IsManual() const {
    if (!IsTuple()) {
      return manual_;
    }
    return absl::c_all_of(tuple_elements_,
                          [](const HloSharding& s) { return s.IsManual(); });
  }
  bool IsManualLeaf() const {
    DCHECK(!IsTuple());
    return manual_;
  }

  // Returns whether the sharding represents a placeholder sharding.
  bool IsUnknown() const {
    if (!IsTuple()) {
      return unknown_;
    }
    return absl::c_all_of(tuple_elements_,
                          [](const HloSharding& s) { return s.IsUnknown(); });
  }
  bool IsUnknownLeaf() const {
    DCHECK(!IsTuple());
    return unknown_;
  }

  bool IsShardGroup() const {
    if (!IsTuple()) {
      return shard_group_.shard_group_id != -1 &&
             (shard_group_.shard_like || shard_group_.shard_as);
    }
    return !tuple_elements_.empty() &&
           absl::c_all_of(tuple_elements_, [](const HloSharding& s) {
             return s.IsShardGroup();
           });
  }

  bool IsShardAs() const {
    if (!IsTuple()) {
      return shard_group_.shard_group_id != -1 && shard_group_.shard_as;
    }
    return !tuple_elements_.empty() &&
           absl::c_all_of(tuple_elements_,
                          [](const HloSharding& s) { return s.IsShardAs(); });
  }

  bool IsShardLike() const {
    if (!IsTuple()) {
      return shard_group_.shard_group_id != -1 && shard_group_.shard_like;
    }
    return !tuple_elements_.empty() &&
           absl::c_all_of(tuple_elements_,
                          [](const HloSharding& s) { return s.IsShardLike(); });
  }

  // Returns whether the sharding represents manual subgroup sharding.
  bool IsManualSubgroup() const {
    if (!IsTuple()) {
      return absl::c_linear_search(subgroup_types_, OpSharding::MANUAL);
    }
    return absl::c_all_of(tuple_elements_, [](const HloSharding& s) {
      return s.IsManualSubgroup();
    });
  }

  // Returns weather the sharding represents a tiled sharding where the mapping
  // between devices and tiles is represented through 'tile_assignment()'.
  bool IsTiled() const {
    return !IsTileMaximal() && !IsManual() && !IsUnknown();
  }
  bool IsTiledLeaf() const {
    return !IsTileMaximalLeaf() && !IsManualLeaf() && !IsUnknownLeaf();
  }

  // Returns if the sharding has partial replication and partial sharding. If
  // true, data is sharded according to other dimensions of tile_assignment(),
  // but replicated across devices along the last dimension.
  bool ReplicateOnLastTileDim() const { return replicate_on_last_tile_dim_; }

  // Returns whether there is any partial replication. This can be using
  // ReplicateOnLastTileDim or subgroups with REPLICATED.
  bool HasPartialReplication() const {
    return replicate_on_last_tile_dim_ ||
           absl::c_linear_search(subgroup_types_, OpSharding::REPLICATED);
  }

  // Returns true if the sharding defines an operation on the given device.
  bool UsesDevice(int64_t device) const;

  // Retrieves a histogram of the devices used by the sharding. The returned
  // map has the device number as key, and the occurrence count as value.
  // If a sharding does not have a device, it will not be included in the
  // histogram. The count argument, if not nullptr, will receive the total
  // number of elements this sharding is made of (one for array, N leaves for
  // tuples).
  std::map<int64_t, int64_t> UsedDevices(int64_t* count) const;

  // Returns the tile that should be executed on the given device.
  // REQUIRES: !IsTuple()
  std::vector<int64_t> TileIndexForDevice(int64_t device) const;

  // Returns the device that should execute the given tile.
  // It is an error to call this if is_replicated() is true.
  // When ReplicateOnLastTileDim() == true, if index.size() == data rank, it
  // returns the first device in that replicated subgroup; otherwise,
  // index.size() should be the same as tile_assignment()'s rank and specifies
  // the member of the replication subgroup.
  // REQUIRES: !IsTuple()
  int64_t DeviceForTileIndex(absl::Span<const int64_t> index) const;

  // Given a device ID, returns the offset within the specified shape of the
  // tile that should be executed on the given core. This returns the lower
  // extent of the tile in the input space.
  // REQUIRES: !IsTuple()
  std::vector<int64_t> TileOffsetForDevice(const Shape& shape,
                                           int64_t device) const;

  // Given a device ID, returns the limit within the specified shape of the
  // tile that should be executed on the given core. This returns the upper
  // extent of the tile in the input space.
  // REQUIRES: !IsTuple()
  std::vector<int64_t> TileLimitForDevice(const Shape& shape,
                                          int64_t device) const;

  // Returns the single device this op operates on. If the sharding does not
  // span a single device, the return value will be empty.
  // In order for a sharding to span a single device, every leaf sharding must
  // be maximal and not replicated, and the used device must match.
  std::optional<int64_t> UniqueDevice() const;

  // Retrieves the unique device or fails with a CHECK.
  int64_t GetUniqueDevice() const;

  // Returns true if this op only uses a single device.
  bool HasUniqueDevice() const { return UniqueDevice().has_value(); }

  // Returns the ShapeTree containing the shardings for each element of this
  // tuple, if IsTuple, or a ShapeTree with a single element containing this
  // sharding. Only the leaf elements are populated. This creates a new
  // ShapeTree object so is not cheap.
  absl::StatusOr<ShapeTree<HloSharding>> AsShapeTree(const Shape& shape) const;
  ShapeTree<HloSharding> GetAsShapeTree(const Shape& shape) const {
    return AsShapeTree(shape).value();
  }

  // Retrieves the sub sharding at a given index, out of a tuple sharding.
  // REQUIRES: IsTuple()
  HloSharding GetSubSharding(const Shape& shape, const ShapeIndex& index) const;

  // If the current sharding is a tuple sharding, return itself as result.
  // Otherwise returns a tuple sharding for the input shape, with all the leaves
  // having this object sharding.
  absl::StatusOr<HloSharding> GetTupleSharding(const Shape& shape) const;

  // If the shape is tuple and the current sharding is not a tuple, attempt to
  // construct a sharding that is compatible with the shape by replicating the
  // current sharding across all tuple elements. Note that the returned
  // sharding is not guaranteed to be compatible with the input shape.
  HloSharding NormalizeTupleSharding(const Shape& shape) const;

  // Extracts the sharding that is common within the current sharding.
  // If the current sharding is not a tuple sharding, the current sharding will
  // be returned. If it is a tuple, and all the tuple elements are common, the
  // common element will be returned. Otherwise the optional will contain no
  // value.
  std::optional<HloSharding> ExtractSingleSharding() const;

  // Returns a copy of the sharding with no metadata. If sharding is of tuple
  // type, sub shardings will have no metadata.
  HloSharding WithoutMetadata() const;

  // Returns a copy of the sharding with specified metadata. If metadata is
  // already present, that metadata will not be replaced unless `overwrite` is
  // set to true. If sharding is of tuple type, sub shardings metadata will be
  // assigned instead.
  HloSharding WithMetadata(absl::Span<const OpMetadata> metadata,
                           bool overwrite) const;

  bool operator==(const HloSharding& other) const {
    return replicated_ == other.replicated_ && maximal_ == other.maximal_ &&
           manual_ == other.manual_ && unknown_ == other.unknown_ &&
           tile_assignment_ == other.tile_assignment_ &&
           tuple_elements_ == other.tuple_elements_ &&
           replicate_on_last_tile_dim_ == other.replicate_on_last_tile_dim_ &&
           subgroup_types_ == other.subgroup_types_ &&
           shard_group_ == other.shard_group_;
  }
  bool operator!=(const HloSharding& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const HloSharding& sharding) {
    if (sharding.tuple_) {
      return H::combine(std::move(h), sharding.tuple_elements_);
    }
    return H::combine(std::move(h), sharding.replicated_, sharding.manual_,
                      sharding.unknown_, sharding.tile_assignment_.array(),
                      sharding.replicate_on_last_tile_dim_,
                      sharding.shard_group_.ToString());
  }

  // Gets the tile assignment tensor.
  // REQUIRES: !IsReplicated() && !IsTuple()
  const TileAssignment& tile_assignment() const { return tile_assignment_; }

  // Gets the subgroup types array.
  // REQUIRES: !IsTuple()
  const std::vector<OpSharding::Type>& subgroup_types() const {
    return subgroup_types_;
  }

  // Returns the flattened list of all the leaf shardings in a tuple shape, by
  // pre-order walk (ShapeTree iterator order).
  // REQUIRES: IsTuple().
  std::vector<HloSharding>& tuple_elements() { return tuple_elements_; }
  const std::vector<HloSharding>& tuple_elements() const {
    return tuple_elements_;
  }

  // Gets the tile shape.
  // REQUIRES: !IsTuple()
  Shape TileShape(const Shape& shape) const;

  // Gets the tile shape on the device.
  // REQUIRES: !IsTuple()
  Shape TileShape(const Shape& shape, int64_t device) const;

  // Gets the total number of tiles including subgroups and partial replication.
  int64_t TotalNumTiles() const;
  // Gets the number of tiles. If it has partial replication, this will not
  // equal the device count.
  int64_t NumTiles() const;
  int64_t NumTilesLeaf() const;
  // Like NumTiles() but considers only some specific dimensions passed as
  // argument
  int64_t NumTiles(absl::Span<const int64_t> dims) const;

  // Gets metadata from sharding.
  std::vector<OpMetadata>& metadata() { return metadata_; }
  const std::vector<OpMetadata>& metadata() const { return metadata_; }

  // Returns the replication subgroup dim, or -1 if it doesn't exist.
  int64_t SubgroupReplicationDim() const {
    auto it = absl::c_find(subgroup_types_, OpSharding::REPLICATED);
    if (it != subgroup_types_.end()) {
      return (it - subgroup_types_.begin()) + TiledDataRank();
    }
    if (replicate_on_last_tile_dim_) {
      return tile_assignment_.num_dimensions() - 1;
    }
    return -1;
  }

  // Returns the manual subgroup dim, or -1 if it doesn't exist.
  int64_t SubgroupManualDim() const {
    auto it = absl::c_find(subgroup_types_, OpSharding::MANUAL);
    if (it != subgroup_types_.end()) {
      return (it - subgroup_types_.begin()) + TiledDataRank();
    }
    return -1;
  }

  // Returns the data rank for tiled sharding. It doesn't include subgroup dims.
  int64_t TiledDataRank() const {
    CHECK(IsTiled());
    int64_t rank = tile_assignment_.num_dimensions();
    if (ReplicateOnLastTileDim()) {
      rank--;
    }
    rank -= subgroup_types_.size();
    return rank;
  }
  int64_t TiledDataRankLeaf() const {
    DCHECK(!IsTuple());
    CHECK(IsTiledLeaf());
    int64_t rank = tile_assignment_.num_dimensions();
    if (ReplicateOnLastTileDim()) {
      rank--;
    }
    rank -= subgroup_types_.size();
    return rank;
  }

  // Returns the number of tuple_elements_ entries to fit the shape.
  static int64_t RequiredLeaves(const Shape& shape);

  struct ShardGroup {
    ShardGroup(int64_t shard_group_id, bool shard_as, bool shard_like)
        : shard_group_id(shard_group_id),
          shard_as(shard_as),
          shard_like(shard_like) {}

    bool operator==(const ShardGroup& rhs) const {
      return shard_group_id == rhs.shard_group_id && shard_as == rhs.shard_as &&
             shard_like == rhs.shard_like;
    }

    std::string ToString() const {
      std::ostringstream result;
      if (shard_as) {
        result << "shard_as " << shard_group_id;
      } else if (shard_like) {
        result << "shard_like " << shard_group_id;
      }
      return result.str();
    }

    int64_t shard_group_id = 0;
    bool shard_as;
    bool shard_like;
  };
  static ShardGroup NotShardGroup() {
    return ShardGroup(
        /*shard_group_id=*/-1,
        /*shard_as=*/false,
        /*shard_like=*/false);
  }

  static ShardGroup ShardAs(int64_t shard_group_id) {
    return ShardGroup(shard_group_id,
                      /*shard_as=*/true,
                      /*shard_like=*/false);
  }

  static ShardGroup ShardLike(int64_t shard_group_id) {
    return ShardGroup(shard_group_id,
                      /*shard_as=*/false,
                      /*shard_like=*/true);
  }

  HloSharding& SetShardGroup(const ShardGroup& shard_group) {
    shard_group_ = shard_group;
    return *this;
  }

  HloSharding& SetShardGroupFromProto(const OpSharding& proto) {
    ShardGroup shard_group = NotShardGroup();
    if (proto.is_shard_group()) {
      if (proto.shard_group_type() == OpSharding::AS) {
        shard_group = ShardAs(proto.shard_group_id());
      } else {
        shard_group = ShardLike(proto.shard_group_id());
      }
    }
    SetShardGroup(shard_group);
    return *this;
  }

  HloSharding& ClearShardGroup() {
    shard_group_ = NotShardGroup();
    return *this;
  }

  const ShardGroup& GetShardGroup() const { return shard_group_; }

 private:
  explicit HloSharding(bool manual, bool replicated, bool unknown,
                       absl::Span<const OpMetadata> metadata)
      : metadata_(metadata.begin(), metadata.end()),
        replicated_(replicated),
        maximal_(replicated),
        tuple_(false),
        manual_(manual),
        unknown_(unknown),
        replicate_on_last_tile_dim_(false) {}
  // device_id values:
  // -2: magic number to mean unassigned device, used by spatial partitioning
  // -1: the id of the host
  //  0 or positive: the id of a device
  // NOTE(dimvar): -1 is needed for outside compilation. It can be removed once
  // we have fully switched to the side-effect tokens.
  explicit HloSharding(int64_t device_id, absl::Span<const OpMetadata> metadata)
      : tile_assignment_(device_id),
        metadata_(metadata.begin(), metadata.end()),
        replicated_(false),
        maximal_(true),
        tuple_(false),
        manual_(false),
        unknown_(false),
        replicate_on_last_tile_dim_(false) {}
  explicit HloSharding(TileAssignment tile_assignment,
                       bool replicate_on_last_tile_dim,
                       absl::Span<const OpMetadata> metadata = {})
      : tile_assignment_(std::move(tile_assignment)),
        metadata_(metadata.begin(), metadata.end()),
        replicated_(false),
        maximal_(false),
        tuple_(false),
        manual_(false),
        unknown_(false),
        replicate_on_last_tile_dim_(replicate_on_last_tile_dim) {}
  explicit HloSharding(TileAssignment tile_assignment,
                       absl::Span<const OpSharding::Type> subgroup_types,
                       absl::Span<const OpMetadata> metadata = {})
      : tile_assignment_(std::move(tile_assignment)),
        metadata_(metadata.begin(), metadata.end()),
        subgroup_types_(subgroup_types.begin(), subgroup_types.end()),
        replicated_(false),
        maximal_(false),
        tuple_(false),
        manual_(false),
        unknown_(false),
        replicate_on_last_tile_dim_(false) {}
  explicit HloSharding(std::vector<HloSharding> tuple_shardings)
      : tuple_elements_(std::move(tuple_shardings)),
        replicated_(false),
        maximal_(false),
        tuple_(true),
        manual_(false),
        unknown_(false),
        replicate_on_last_tile_dim_(false) {}

  // Test-only constructor for sharding format code coverage. Copies the
  // original sharding with provided tile assignment.
  explicit HloSharding(const HloSharding& other, TileAssignment tile_assignment)
      : tile_assignment_(std::move(tile_assignment)),
        tuple_elements_(other.tuple_elements_),
        metadata_(other.metadata_),
        subgroup_types_(other.subgroup_types_),
        replicated_(other.replicated_),
        maximal_(other.maximal_),
        tuple_(other.tuple_),
        manual_(other.manual_),
        unknown_(other.unknown_),
        replicate_on_last_tile_dim_(other.replicate_on_last_tile_dim_) {
    CHECK(tile_assignment_ == other.tile_assignment_)
        << tile_assignment_.ToString() << " v.s. "
        << other.tile_assignment_.ToString();
  }
  friend class HloShardingTestHelper;

  // Checks that the number of elements in tuple_elements_ is consistent with
  // the tuple shape passes as argument.
  absl::Status CheckLeafCount(const Shape& shape) const;

  // Internal helper to validate a tuple sharding.
  absl::Status ValidateTuple(const Shape& shape,
                             std::optional<int64_t> num_devices) const;

  // Internal helper to validate a non-tuple (leaf) sharding.
  absl::Status ValidateNonTuple(const Shape& shape,
                                std::optional<int64_t> num_devices) const;

  // This field is only used if replicated_ is false. If maximal_ is true, then
  // the field contains a rank 1 array with a single element, which is the
  // device the HLO is assigned to. If maximal_ is false, the field contains an
  // array with the same rank as the corresponding HLO. The dimension sizes of
  // the array describe the number of ways the HLO is partitioned along each
  // dimension. The values of the array specify which device each tile of
  // the HLO is assigned to. The index of each value determines which tile it
  // takes.
  // For example, {{{2, 3}}, {{5, 7}}} (whose ToString representation is
  // "{devices=[2,1,2]2,3,5,7}"), means that dimension 1 is split two way and
  // dimension 3 is split 2 way. Core 5, whose index is [2,1,1] will take the
  // tile that contains the 2nd half of dimension 1 and the 1st half of
  // dimension 3.
  TileAssignment tile_assignment_;
  // Only non-empty when tuple_ is true. If a tuple is empty then one entry is
  // present for the root. This is a flattened list of all the leaf shardings in
  // a tuple shape, by pre-order walk (ShapeTree iterator order).
  std::vector<HloSharding> tuple_elements_;
  // This field is used to track the source of this sharding, usually derived
  // from instructions. Multiple metadata may be populated if sharding is
  // combined with other shardings. Metadata are to not be populated when
  // tuple_ == true and instead metadata should be set on individual tuple
  // elements.
  std::vector<OpMetadata> metadata_;
  // This field is used to represented the sharding type of each subgroup.
  // For example, sharding={devices=[2,2,2,2]0,1,2,...,15 last_tile_dims={
  // replicate, manual, unreduced}} means that each of the last 3 dimensions
  // in [2,2,2,2] represents a subgrouping in replicate, manual.
  // When creating HloSharding, subgroup dims of the same type will be merged,
  // so that there is at most one dim with a given type.
  std::vector<OpSharding::Type> subgroup_types_;
  bool replicated_ : 1;  // When non-tuple, true if the sharding is trivial.
  bool maximal_ : 1;     // When non-tuple, true if the tile size is the same as
                         // the input size.
  bool tuple_ : 1;       // True if this is a tuple.
  bool manual_ : 1;   // When non-tuple, true if the sharding represents manual
                      // partitioning.
  bool unknown_ : 1;  // When non-tuple, true if the sharding represents a
                      // placeholder sharding.
  // This flag is to support partial replication and partial sharding. If it is
  // true, tile_assignment_ will have an extra dimension in addition to the data
  // shape rank, and the added last dimension represents the subgroups of
  // replications, i.e., elements in slice [..., :] will be replicated.
  bool replicate_on_last_tile_dim_ : 1;
  // This field is used to store the shard group information. Instructions
  // within the same shard group(i.e. under the same shard_group_id) will be
  // sharded alike or exactly the same as each other.
  ShardGroup shard_group_ = NotShardGroup();
};

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding);

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_SHARDING_H_
