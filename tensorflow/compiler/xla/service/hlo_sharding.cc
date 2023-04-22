/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_sharding.h"

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/compiler/xla/service/hlo_op_metadata.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

using absl::StrCat;
using absl::StrJoin;

HloSharding HloSharding::AssignDevice(int64 device_id,
                                      absl::Span<const OpMetadata> metadata) {
  return HloSharding(device_id, metadata);
}

HloSharding HloSharding::Tile1D(const Shape& input_shape, int64 num_tiles,
                                absl::Span<const OpMetadata> metadata) {
  CHECK_EQ(1, input_shape.rank());
  CHECK_GT(num_tiles, 1);
  std::vector<int64> dimensions(1, num_tiles);
  Array<int64> assignment(dimensions);
  std::iota(assignment.begin(), assignment.end(), 0);
  return HloSharding(assignment, /*replicate_on_last_tile_dim=*/false,
                     metadata);
}

HloSharding HloSharding::PartialTile(
    const Array<int64>& group_tile_assignment,
    absl::Span<const absl::Span<const int64>> replication_groups,
    absl::Span<const OpMetadata> metadata) {
  CHECK_EQ(group_tile_assignment.num_elements(), replication_groups.size());
  if (replication_groups.size() == 1) {
    return Replicate(metadata);
  }
  auto new_tile_dims = group_tile_assignment.dimensions();
  new_tile_dims.push_back(replication_groups[0].size());
  auto new_tile_assignment = Array<int64>(new_tile_dims);
  new_tile_assignment.Each([&](absl::Span<const int64> indices, int64* device) {
    std::vector<int64> group_index(indices.begin(), indices.end());
    group_index.pop_back();
    int64 group = group_tile_assignment(group_index);
    *device = replication_groups[group][indices.back()];
  });
  return PartialTile(new_tile_assignment, metadata);
}

HloSharding HloSharding::PartialTile(
    const Array<int64>& tile_assignment_last_dim_replicate,
    absl::Span<const OpMetadata> metadata) {
  if (tile_assignment_last_dim_replicate.num_dimensions() == 1 ||
      tile_assignment_last_dim_replicate.dimensions().back() ==
          tile_assignment_last_dim_replicate.num_elements()) {
    return Replicate(metadata);
  }
  if (tile_assignment_last_dim_replicate.dimensions().back() == 1) {
    auto new_tile_dims = tile_assignment_last_dim_replicate.dimensions();
    new_tile_dims.pop_back();
    auto fully_tiled = tile_assignment_last_dim_replicate;
    fully_tiled.Reshape(new_tile_dims);
    return HloSharding(fully_tiled, /*replicate_on_last_tile_dim=*/false,
                       metadata);
  }
  std::vector<std::set<int64>> sorted_groups(
      tile_assignment_last_dim_replicate.num_elements() /
      tile_assignment_last_dim_replicate.dimensions().back());
  auto get_group_id = [&](absl::Span<const int64> indices) {
    int64 group_id = 0;
    for (int64 i = 0; i < indices.size() - 1; ++i) {
      group_id *= tile_assignment_last_dim_replicate.dim(i);
      group_id += indices[i];
    }
    return group_id;
  };
  tile_assignment_last_dim_replicate.Each(
      [&](absl::Span<const int64> indices, const int64 device) {
        sorted_groups[get_group_id(indices)].insert(device);
      });
  Array<int64> sorted_tile(tile_assignment_last_dim_replicate.dimensions());
  sorted_tile.Each([&](absl::Span<const int64> indices, int64* device) {
    const int64 group_id = get_group_id(indices);
    auto begin = sorted_groups[group_id].begin();
    *device = *begin;
    sorted_groups[group_id].erase(begin);
  });
  return HloSharding(sorted_tile, /*replicate_on_last_tile_dim=*/true,
                     metadata);
}

HloSharding HloSharding::Tuple(const ShapeTree<HloSharding>& sub_shardings) {
  std::vector<HloSharding> flattened_list;
  flattened_list.reserve(sub_shardings.leaf_count());
  for (const auto& index_to_sharding : sub_shardings.leaves()) {
    flattened_list.push_back(index_to_sharding.second);
  }
  if (flattened_list.empty()) {
    // Empty tuple sharding ends up having no leaves, but we want to allow
    // empty tuple HLO instruction results to have sharding, so we fetch the
    // root ({}) sharding value from the ShapeTree.
    // A ShapeTree created with ShapeTree<HloSharding>(shape, init) will have
    // init as value at its root.
    flattened_list.push_back(sub_shardings.element(ShapeIndex({})));
  }
  return HloSharding(flattened_list);
}

HloSharding HloSharding::Tuple(const Shape& tuple_shape,
                               absl::Span<const HloSharding> shardings) {
  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  for (auto& sharding : shardings) {
    CHECK(!sharding.IsTuple()) << sharding.ToString();
  }
  std::vector<HloSharding> flattened_list(shardings.begin(), shardings.end());
  CHECK_EQ(flattened_list.size(), RequiredLeaves(tuple_shape))
      << "Flat list has " << flattened_list.size() << ", required "
      << RequiredLeaves(tuple_shape);
  return HloSharding(flattened_list);
}

HloSharding HloSharding::SingleTuple(const Shape& tuple_shape,
                                     const HloSharding& sharding) {
  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  CHECK(!sharding.IsTuple()) << sharding.ToString();
  int64 leaf_count = RequiredLeaves(tuple_shape);
  std::vector<HloSharding> flattened_list;
  flattened_list.resize(leaf_count, sharding);
  return HloSharding(flattened_list);
}

HloSharding HloSharding::Single(const Shape& shape,
                                const HloSharding& sharding) {
  return shape.IsTuple() ? SingleTuple(shape, sharding) : sharding;
}

string HloSharding::ToString(bool include_metadata) const {
  if (IsTuple()) {
    CHECK(metadata_.empty());
    std::string result = "{";
    for (int i = 0; i < tuple_elements_.size(); ++i) {
      const HloSharding& element = tuple_elements_[i];
      if (i != 0) {
        absl::StrAppend(&result, ", ");
        if (i % 5 == 0) {
          absl::StrAppend(&result, "/*index=", i, "*/");
        }
      }
      absl::StrAppend(&result, element.ToString(include_metadata));
    }
    absl::StrAppend(&result, "}");
    return result;
  }

  std::string metadata;
  if (include_metadata) {
    if (metadata_.size() == 1) {
      metadata =
          StrCat(" metadata={", OpMetadataToString(metadata_.front()), "}");
    } else if (metadata_.size() > 1) {
      std::vector<std::string> metadata_strings;
      metadata_strings.reserve(metadata_.size());
      for (const auto& single_metadata : metadata_) {
        metadata_strings.push_back(
            StrCat("{", OpMetadataToString(single_metadata), "}"));
      }
      metadata = StrCat(" metadata={", StrJoin(metadata_strings, ", "), "}");
    }
  }

  if (replicated_) {
    return StrCat("{replicated", metadata, "}");
  }

  if (manual_) {
    return StrCat("{manual", metadata, "}");
  }
  if (maximal_) {
    return StrCat("{maximal device=",
                  static_cast<int64>(*tile_assignment_.begin()), metadata, "}");
  }
  return StrCat("{devices=[", StrJoin(tile_assignment_.dimensions(), ","), "]",
                StrJoin(tile_assignment_, ","),
                replicate_on_last_tile_dim_ ? " last_tile_dim_replicate" : "",
                metadata, "}");
}

bool HloSharding::UsesDevice(int64 device) const {
  if (IsTuple()) {
    return absl::c_any_of(tuple_elements_, [&](const HloSharding& s) {
      return s.UsesDevice(device);
    });
  }
  const auto& devices = tile_assignment_;
  return replicated_ || manual_ || absl::c_linear_search(devices, device);
}

std::map<int64, int64> HloSharding::UsedDevices(int64* count) const {
  int64 element_count = 1;
  std::map<int64, int64> device_map;
  if (IsTuple()) {
    for (auto& tuple_element_sharding : tuple_elements()) {
      auto unique_device = tuple_element_sharding.UniqueDevice();
      if (unique_device) {
        device_map[*unique_device] += 1;
      }
    }
    element_count = tuple_elements().size();
  } else {
    auto unique_device = UniqueDevice();
    if (unique_device) {
      device_map[*unique_device] += 1;
    }
  }
  if (count != nullptr) {
    *count = element_count;
  }
  return device_map;
}

std::vector<int64> HloSharding::TileIndexForDevice(int64 device) const {
  CHECK(!maximal_);
  CHECK(!manual_);
  CHECK(!IsTuple());
  std::vector<int64> ret_index;
  tile_assignment_.Each([&](absl::Span<const int64> index, int64 d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  if (replicate_on_last_tile_dim_) {
    ret_index.pop_back();
  }
  return ret_index;
}

int64 HloSharding::DeviceForTileIndex(absl::Span<const int64> index) const {
  CHECK(!replicated_);
  CHECK(!manual_);
  CHECK(!IsTuple());
  if (maximal_) {
    return *tile_assignment_.begin();
  }
  if (replicate_on_last_tile_dim_ &&
      index.size() < tile_assignment().num_dimensions()) {
    std::vector<int64> first_replicated_index(index.begin(), index.end());
    first_replicated_index.push_back(0);
    return tile_assignment_(first_replicated_index);
  }
  return tile_assignment_(index);
}

std::vector<int64> HloSharding::TileOffsetForDevice(const Shape& shape,
                                                    int64 device) const {
  CHECK(!IsTuple());
  CHECK(!manual_);

  if (maximal_) {
    return std::vector<int64>(shape.dimensions_size(), 0);
  }
  if (replicate_on_last_tile_dim_) {
    CHECK_EQ(shape.dimensions_size(), tile_assignment_.num_dimensions() - 1);
  } else {
    CHECK_EQ(shape.dimensions_size(), tile_assignment_.num_dimensions());
  }
  std::vector<int64> index = TileIndexForDevice(device);
  for (int64 i = 0; i < index.size(); ++i) {
    const int64 shape_dim = shape.dimensions(i);
    index[i] = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
  }
  return index;
}

std::vector<int64> HloSharding::TileLimitForDevice(const Shape& shape,
                                                   int64 device) const {
  CHECK(!IsTuple());
  CHECK(!manual_);

  if (maximal_) {
    return std::vector<int64>(shape.dimensions().begin(),
                              shape.dimensions().end());
  }

  CHECK_EQ(shape.dimensions_size() + (ReplicateOnLastTileDim() ? 1 : 0),
           tile_assignment_.num_dimensions());
  std::vector<int64> index = TileIndexForDevice(device);
  for (int64 i = 0; i < index.size(); ++i) {
    const int64 shape_dim = shape.dimensions(i);
    index[i] = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
  }
  return index;
}

int64 HloSharding::RequiredLeaves(const Shape& shape) {
  // Empty tuples (with arbitrary nesting) have no leaf nodes as far as
  // ShapeUtil and ShapeTree are concerned, but they do have a single
  // tuple_elements_ entry since we want to allow empty tuple results to
  // have sharding.
  const int64 leaf_count = ShapeUtil::GetLeafCount(shape);
  return (leaf_count == 0) ? 1 : leaf_count;
}

Status HloSharding::CheckLeafCount(const Shape& shape) const {
  int64 shape_leaves = RequiredLeaves(shape);
  TF_RET_CHECK(shape_leaves == tuple_elements_.size())
      << "Shape " << ShapeUtil::HumanString(shape) << " has " << shape_leaves
      << " leaf nodes while this sharding has " << tuple_elements_.size();
  return Status::OK();
}

StatusOr<ShapeTree<HloSharding>> HloSharding::AsShapeTree(
    const Shape& shape) const {
  if (IsTuple()) {
    ShapeTree<HloSharding> result(shape, HloSharding::Replicate());
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    auto it = tuple_elements_.begin();
    for (auto& index_to_sharding : result.leaves()) {
      index_to_sharding.second = *it++;
    }
    if (ShapeUtil::IsEmptyTuple(shape)) {
      // Empty tuples have no leaves, but we want to assign them a sharding
      // anyway, so we use the root element sharding.
      *result.mutable_element(ShapeIndex({})) = *it;
    }
    return std::move(result);
  } else {
    return ShapeTree<HloSharding>(shape, *this);
  }
}

StatusOr<HloSharding> HloSharding::GetTupleSharding(const Shape& shape) const {
  if (IsTuple()) {
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    return *this;
  }
  return Tuple(ShapeTree<HloSharding>(shape, *this));
}

absl::optional<int64> HloSharding::UniqueDevice() const {
  if (IsTuple()) {
    if (tuple_elements_.empty()) {
      return absl::nullopt;
    }
    absl::optional<int64> unique_device;
    for (auto& tuple_sharding : tuple_elements_) {
      auto device = tuple_sharding.UniqueDevice();
      if (!device || (unique_device && *device != *unique_device)) {
        return absl::nullopt;
      }
      unique_device = device;
    }
    return unique_device;
  }
  if (!replicated_ && maximal_) {
    return static_cast<int64>(*tile_assignment_.begin());
  }
  return absl::nullopt;
}

int64 HloSharding::GetUniqueDevice() const {
  auto device = UniqueDevice();
  CHECK(device) << "Sharding does not have a unique device: " << *this;
  return *device;
}

Status HloSharding::ValidateTuple(const Shape& shape, int64 num_devices) const {
  if (!shape.IsTuple()) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Sharding is tuple-shaped but validation shape is not."));
  }
  TF_RETURN_IF_ERROR(CheckLeafCount(shape));

  // Now we've validated the number of tuple elements, it's safe to request a
  // shape tree.
  ShapeTree<HloSharding> shape_tree = GetAsShapeTree(shape);
  for (const auto& index_to_sharding : shape_tree.leaves()) {
    Status status = index_to_sharding.second.ValidateNonTuple(
        ShapeUtil::GetSubshape(shape, index_to_sharding.first), num_devices);
    if (!status.ok()) {
      tensorflow::errors::AppendToMessage(
          &status, StrCat("Note: While validating sharding tuple element ",
                          index_to_sharding.first.ToString(), " which is ",
                          index_to_sharding.second.ToString()));
      return status;
    }
  }
  return Status::OK();
}

Status HloSharding::Validate(const Shape& shape, int64 num_devices) const {
  if (shape.IsToken()) {
    return Status::OK();
  }
  Status status = IsTuple() ? ValidateTuple(shape, num_devices)
                            : ValidateNonTuple(shape, num_devices);
  if (!status.ok()) {
    tensorflow::errors::AppendToMessage(
        &status, StrCat("Note: While validating sharding ", ToString(),
                        " against shape ", ShapeUtil::HumanString(shape)));
  }
  return status;
}

Status HloSharding::ValidateNonTuple(const Shape& shape,
                                     int64 num_devices) const {
  if (shape.IsTuple()) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Validation shape is a tuple but sharding is not."));
  }
  if (replicated_) {
    return Status::OK();
  }

  // All tile assignments must be less than the number of available cores and
  // unique.
  Status status = Status::OK();
  absl::flat_hash_set<int64> seen_cores;
  tile_assignment_.Each([&](absl::Span<const int64> indices, int32 core) {
    // Don't overwrite a bad status, so we report the first error.
    if (status.ok()) {
      if (core >= num_devices) {
        status = tensorflow::errors::InvalidArgument(
            StrCat("core ", core, " > ", num_devices, " in tile assignment"));
      } else if (seen_cores.contains(core)) {
        status = tensorflow::errors::InvalidArgument(
            StrCat("core ", core, " is not unique in tile assignment"));
      }
      seen_cores.insert(core);
    }
  });
  if (!status.ok()) {
    return status;
  }

  if (IsTileMaximal() || IsManual()) {
    return Status::OK();
  }

  // The tile assignment tensor must have the same rank as the input, or input
  // rank + 1 for replicate_on_last_tile_dim_.
  if (shape.rank() + (replicate_on_last_tile_dim_ ? 1 : 0) !=
      tile_assignment_.num_dimensions()) {
    return tensorflow::errors::InvalidArgument(
        "Number of tile assignment dimensions is different to the input rank. "
        "sharding=",
        ToString(), ", input_shape=", ShapeUtil::HumanString(shape));
  }

  // The correct constructor has to be used to create tile maximal shardings.
  if (tile_assignment_.num_elements() == 1) {
    return tensorflow::errors::InvalidArgument(
        "Tile assignment only contains a single device. If a replicated "
        "sharding was intended, use HloSharding::Replicated(). If a device "
        "placement was intended, use HloSharding::AssignDevice()");
  }
  return Status::OK();
}

/*static*/ StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
  std::vector<OpMetadata> metadata(proto.metadata().begin(),
                                   proto.metadata().end());
  if (proto.type() == OpSharding::TUPLE) {
    TF_RET_CHECK(metadata.empty())
        << "Tuple sharding is expected to have no metadata.";
    std::vector<HloSharding> tuple_shardings;
    tuple_shardings.reserve(proto.tuple_shardings().size());
    for (const OpSharding& tuple_sharding_proto : proto.tuple_shardings()) {
      TF_ASSIGN_OR_RETURN(HloSharding sharding,
                          HloSharding::FromProto(tuple_sharding_proto));
      tuple_shardings.push_back(sharding);
    }
    return HloSharding(tuple_shardings);
  } else if (proto.type() == OpSharding::REPLICATED) {
    return Replicate(metadata);
  } else if (proto.type() == OpSharding::MANUAL) {
    return Manual(metadata);
  } else if (proto.tile_assignment_devices().size() == 1) {
    return HloSharding(proto.tile_assignment_devices(0), metadata);
  }

  TF_RET_CHECK(proto.type() != OpSharding::MAXIMAL)
      << "Maximal sharding is expected to have single device assignment, but "
      << proto.tile_assignment_devices().size() << " has provided.";

  TF_RET_CHECK(proto.tile_assignment_devices().size() > 1);
  TF_RET_CHECK(!proto.tile_assignment_dimensions().empty());

  // RE: the product of tile assignment tensor dimensions must be
  // equal to tile_assignment_devices.size().
  int64 product_of_dimensions = 1;
  for (auto dimension : proto.tile_assignment_dimensions()) {
    TF_RET_CHECK(dimension > 0);
    product_of_dimensions =
        MultiplyWithoutOverflow(product_of_dimensions, dimension);
    TF_RET_CHECK(product_of_dimensions > 0);
  }
  TF_RET_CHECK(product_of_dimensions == proto.tile_assignment_devices().size());

  // Some versions of gcc cannot infer the TileAssignment constructor from a
  // braced initializer-list, so create one manually.
  std::vector<int64> devices(proto.tile_assignment_devices().begin(),
                             proto.tile_assignment_devices().end());
  Array<int64> tile_assignment(
      std::vector<int64>(proto.tile_assignment_dimensions().begin(),
                         proto.tile_assignment_dimensions().end()));
  std::copy(proto.tile_assignment_devices().begin(),
            proto.tile_assignment_devices().end(), tile_assignment.begin());
  return proto.replicate_on_last_tile_dim()
             ? PartialTile(tile_assignment, metadata)
             : HloSharding(tile_assignment,
                           /*replicate_on_last_tile_dim=*/false, metadata);
}

OpSharding HloSharding::ToProto() const {
  OpSharding result;

  if (IsTuple()) {
    CHECK(metadata_.empty());
    for (const HloSharding& element : tuple_elements_) {
      *result.add_tuple_shardings() = element.ToProto();
    }
    result.set_type(OpSharding::TUPLE);
    return result;
  }

  result.mutable_metadata()->Reserve(metadata_.size());
  for (const auto& metadata : metadata_) {
    *result.add_metadata() = metadata;
  }

  for (int64 dim : tile_assignment_.dimensions()) {
    result.add_tile_assignment_dimensions(dim);
  }
  for (auto device : tile_assignment_) {
    result.add_tile_assignment_devices(device);
  }
  if (IsReplicated()) {
    result.set_type(OpSharding::REPLICATED);
    result.clear_tile_assignment_dimensions();
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::MAXIMAL);
  } else if (IsManual()) {
    result.set_type(OpSharding::MANUAL);
    result.clear_tile_assignment_dimensions();
  } else {
    result.set_type(OpSharding::OTHER);
    result.set_replicate_on_last_tile_dim(ReplicateOnLastTileDim());
  }
  return result;
}

Shape HloSharding::TileShape(const Shape& shape) const {
  if (IsTileMaximal() || IsManual()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64 i = 0; i < shape.dimensions_size(); ++i) {
    result_shape.set_dimensions(
        i, CeilOfRatio<int64>(shape.dimensions(i), tile_assignment_.dim(i)));
  }
  return result_shape;
}

Shape HloSharding::TileShape(const Shape& shape, int64 device) const {
  if (IsTileMaximal() || IsManual()) {
    return shape;
  }

  std::vector<int64> index = TileIndexForDevice(device);
  Shape result_shape = shape;
  for (int64 i = 0; i < index.size(); ++i) {
    const int64 shape_dim = shape.dimensions(i);
    int64 offset = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
    int64 limit = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
    result_shape.set_dimensions(i, limit - offset);
  }
  return result_shape;
}

int64 HloSharding::NumTiles() const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  if (ReplicateOnLastTileDim()) {
    return tile_assignment().num_elements() /
           tile_assignment().dimensions().back();
  }
  return tile_assignment().num_elements();
}

int64 HloSharding::NumTiles(absl::Span<const int64> dims) const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!ReplicateOnLastTileDim() ||
        !absl::c_linear_search(dims, tile_assignment().num_dimensions() - 1));
  int64 num_tiles = 1;
  for (auto d : dims) {
    CHECK(d < tile_assignment().num_dimensions());
    num_tiles *= tile_assignment().dim(d);
  }
  return num_tiles;
}

HloSharding HloSharding::GetSubSharding(const Shape& shape,
                                        const ShapeIndex& index) const {
  CHECK(IsTuple());
  int64 sharding_index = 0;
  const Shape* sub_shape = &shape;
  for (int64 idx : index) {
    for (int64 i = 0; i < idx; ++i) {
      sharding_index +=
          ShapeUtil::GetLeafCount(ShapeUtil::GetSubshape(*sub_shape, {i}));
    }
    sub_shape = &ShapeUtil::GetSubshape(*sub_shape, {idx});
  }
  if (sub_shape->IsTuple()) {
    auto begin_it = tuple_elements_.begin() + sharding_index;
    std::vector<HloSharding> sub_shardings(
        begin_it, begin_it + ShapeUtil::GetLeafCount(*sub_shape));
    return HloSharding::Tuple(*sub_shape, sub_shardings);
  } else {
    return tuple_elements_[sharding_index];
  }
}

absl::optional<HloSharding> HloSharding::ExtractSingleSharding() const {
  if (!IsTuple()) {
    return *this;
  }
  if (tuple_elements_.empty()) {
    return absl::nullopt;
  }
  for (int64 i = 1; i < tuple_elements_.size(); ++i) {
    if (tuple_elements_[0] != tuple_elements_[i]) {
      return absl::nullopt;
    }
  }
  return tuple_elements_.front();
}

HloSharding HloSharding::WithMetadata(absl::Span<const OpMetadata> metadata,
                                      bool overwrite) const {
  auto assign_metadata = [&](HloSharding& sharding) {
    if (sharding.metadata_.empty() || overwrite) {
      sharding.metadata_.assign(metadata.begin(), metadata.end());
    }
  };

  HloSharding sharding = *this;
  if (sharding.IsTuple()) {
    for (HloSharding& sub_sharding : sharding.tuple_elements()) {
      assign_metadata(sub_sharding);
    }
  } else {
    assign_metadata(sharding);
  }
  return sharding;
}

HloSharding HloSharding::WithoutMetadata() const {
  HloSharding sharding = *this;
  sharding.metadata_.clear();
  for (HloSharding& sub_sharding : sharding.tuple_elements()) {
    sub_sharding.metadata_.clear();
  }
  return sharding;
}

size_t HloSharding::Hash() const {
  if (tuple_) {
    size_t h = 0;
    for (const auto& element : tuple_elements_) {
      h = tensorflow::Hash64Combine(h, element.Hash());
    }
    return h;
  }
  if (replicated_) {
    return 0;
  }
  if (manual_) {
    return 1;
  }
  size_t h = 0;
  for (uint32 v : tile_assignment_) {
    h = tensorflow::Hash64Combine(h, std::hash<uint32>{}(v));
  }
  if (replicate_on_last_tile_dim_) {
    h = tensorflow::Hash64Combine(h, std::hash<uint32>{}(1));
  }
  return h;
}

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding) {
  out << sharding.ToString();
  return out;
}

}  // namespace xla
