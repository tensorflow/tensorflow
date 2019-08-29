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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

using absl::StrCat;
using absl::StrJoin;

HloSharding HloSharding::AssignDevice(int64 device_id) {
  return HloSharding(device_id);
}

HloSharding HloSharding::Tile1D(const Shape& input_shape, int64 num_tiles) {
  CHECK_EQ(1, input_shape.rank());
  CHECK_GT(num_tiles, 1);
  std::vector<int64> dimensions(1, num_tiles);
  Array<int64> assignment(dimensions);
  std::iota(assignment.begin(), assignment.end(), 0);
  return HloSharding(assignment);
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

string HloSharding::ToString() const {
  if (IsTuple()) {
    std::vector<string> parts;
    parts.reserve(tuple_elements_.size());
    for (const HloSharding& element : tuple_elements_) {
      parts.push_back(element.ToString());
    }
    return StrCat("{", absl::StrJoin(parts, ", "), "}");
  }

  if (replicated_) {
    return "{replicated}";
  }
  if (maximal_) {
    return StrCat(
        "{maximal device=", static_cast<int64>(*tile_assignment_.begin()), "}");
  }
  return StrCat("{devices=[", StrJoin(tile_assignment_.dimensions(), ","), "]",
                StrJoin(tile_assignment_, ","), "}");
}

bool HloSharding::UsesDevice(int64 device) const {
  if (IsTuple()) {
    return absl::c_any_of(tuple_elements_, [&](const HloSharding& s) {
      return s.UsesDevice(device);
    });
  }
  const auto& devices = tile_assignment_;
  return replicated_ || absl::c_linear_search(devices, device);
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
  CHECK(!IsTuple());
  std::vector<int64> ret_index;
  tile_assignment_.Each([&](absl::Span<const int64> index, int64 d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  return ret_index;
}

int64 HloSharding::DeviceForTileIndex(absl::Span<const int64> index) const {
  CHECK(!replicated_);
  CHECK(!IsTuple());
  if (maximal_) {
    return *tile_assignment_.begin();
  }
  return tile_assignment_(index);
}

std::vector<int64> HloSharding::TileOffsetForDevice(const Shape& shape,
                                                    int64 device) const {
  CHECK(!IsTuple());

  if (maximal_) {
    return std::vector<int64>(shape.dimensions_size(), 0);
  }

  CHECK_EQ(shape.dimensions_size(), tile_assignment_.num_dimensions());
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

  if (maximal_) {
    return std::vector<int64>(shape.dimensions().begin(),
                              shape.dimensions().end());
  }

  CHECK_EQ(shape.dimensions_size(), tile_assignment_.num_dimensions());
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
  // Empty tuples have no leaf nodes as far as ShapeUtil and ShapeTree are
  // concerned, but they do have a single tuple_elements_ entry since we want
  // to allow empty tuple results to have sharding.
  return ShapeUtil::IsEmptyTuple(shape) ? 1 : ShapeUtil::GetLeafCount(shape);
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
  tile_assignment_.Each(
      [&](absl::Span<const int64> indices, int32 core) {
        // Don't overwrite a bad status, so we report the first error.
        if (status.ok()) {
          if (core >= num_devices) {
            status = tensorflow::errors::InvalidArgument(StrCat(
                "core ", core, " > ", num_devices, " in tile assignment"));
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

  if (IsTileMaximal()) {
    return Status::OK();
  }

  // The tile assignment tensor must have the same rank as the input.
  if (shape.rank() != tile_assignment_.num_dimensions()) {
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
  if (proto.type() == OpSharding::TUPLE) {
    std::vector<HloSharding> tuple_shardings;
    tuple_shardings.reserve(proto.tuple_shardings().size());
    for (const OpSharding& tuple_sharding_proto : proto.tuple_shardings()) {
      TF_ASSIGN_OR_RETURN(HloSharding sharding,
                          HloSharding::FromProto(tuple_sharding_proto));
      tuple_shardings.push_back(sharding);
    }
    return HloSharding(tuple_shardings);
  } else if (proto.type() == OpSharding::REPLICATED) {
    return Replicate();
  } else if (proto.tile_assignment_devices().size() == 1) {
    return HloSharding(proto.tile_assignment_devices(0));
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
  return HloSharding(tile_assignment);
}

OpSharding HloSharding::ToProto() const {
  OpSharding result;

  if (IsTuple()) {
    for (const HloSharding& element : tuple_elements_) {
      *result.add_tuple_shardings() = element.ToProto();
    }
    result.set_type(OpSharding::TUPLE);
    return result;
  }

  for (int64 dim : tile_assignment_.dimensions()) {
    result.add_tile_assignment_dimensions(dim);
  }
  for (auto device : tile_assignment_) {
    result.add_tile_assignment_devices(device);
  }
  if (IsReplicated()) {
    result.set_type(OpSharding::REPLICATED);
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::MAXIMAL);
  } else {
    result.set_type(OpSharding::OTHER);
  }
  return result;
}

Shape HloSharding::TileShape(const Shape& shape) const {
  if (IsTileMaximal()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64 i = 0; i < shape.dimensions_size(); ++i) {
    result_shape.set_dimensions(
        i, CeilOfRatio<int64>(shape.dimensions(i), tile_assignment_.dim(i)));
  }
  return result_shape;
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
  size_t h = 0;
  for (uint32 v : tile_assignment_) {
    h = tensorflow::Hash64Combine(h, std::hash<uint32>{}(v));
  }
  return h;
}

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding) {
  out << sharding.ToString();
  return out;
}

}  // namespace xla
