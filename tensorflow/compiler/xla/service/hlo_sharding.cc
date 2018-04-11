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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {

using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrCat;

HloSharding HloSharding::AssignDevice(int64 device_id) {
  return HloSharding(device_id);
}

HloSharding HloSharding::Tile1D(const Shape& input_shape, int64 num_tiles) {
  CHECK_EQ(1, ShapeUtil::Rank(input_shape));
  CHECK_GT(num_tiles, 1);
  std::vector<int64> dimensions(1, num_tiles);
  Shape tile_shape = input_shape;
  auto& tile_dimension = (*tile_shape.mutable_dimensions())[0];
  tile_dimension = CeilOfRatio(static_cast<int64>(tile_dimension), num_tiles);
  Array<int64> assignment(dimensions);
  std::iota(assignment.begin(), assignment.end(), 0);
  return HloSharding(tile_shape, assignment);
}

string HloSharding::ToString() const {
  if (IsTuple()) {
    std::vector<string> parts;
    parts.reserve(tuple_elements_.size());
    for (const HloSharding& element : tuple_elements_) {
      parts.push_back(element.ToString());
    }
    return StrCat("{", tensorflow::str_util::Join(parts, ", "), "}");
  }

  string result = StrCat("{", (replicated_ ? " replicated" : ""),
                         (maximal_ ? " maximal" : ""));

  if (replicated_) {
    return "{replicated}";
  } else if (maximal_) {
    return StrCat(
        "{maximal device=", static_cast<int64>(*tile_assignment_.begin()), "}");
  } else {
    return StrCat("{", ShapeUtil::HumanString(tile_shape_), " ", "devices=[",
                  Join(tile_assignment_.dimensions(), ","), "]",
                  Join(tile_assignment_, ","), "}");
  }
}

bool HloSharding::UsesDevice(int64 device) const {
  if (IsTuple()) {
    return std::any_of(
        tuple_elements_.begin(), tuple_elements_.end(),
        [&](const HloSharding& s) { return s.UsesDevice(device); });
  }
  const auto& devices = tile_assignment_;
  return replicated_ ||
         std::find(devices.begin(), devices.end(), device) != devices.end();
}

std::vector<int64> HloSharding::TileIndexForDevice(int64 device) const {
  CHECK(!ShapeUtil::IsTuple(tile_shape_));
  CHECK(!maximal_);
  CHECK(!IsTuple());
  std::vector<int64> ret_index;
  tile_assignment_.Each([&](tensorflow::gtl::ArraySlice<int64> index, int64 d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  return ret_index;
}

int64 HloSharding::DeviceForTileIndex(
    tensorflow::gtl::ArraySlice<int64> index) const {
  CHECK(!replicated_);
  CHECK(!IsTuple());
  if (maximal_) {
    return *tile_assignment_.begin();
  }
  CHECK_EQ(ShapeUtil::Rank(tile_shape_), tile_assignment_.dimensions().size());
  return tile_assignment_(index);
}

std::vector<int64> HloSharding::TileOffsetForDevice(int64 device) const {
  CHECK(!IsTuple());

  std::vector<int64> index = TileIndexForDevice(device);
  if (maximal_) {
    // Index will always be all zeroes if we're maximal, and tile_shape_ is not
    // valid.
    return index;
  }
  for (int64 i = 0; i < index.size(); ++i) {
    index[i] *= tile_shape_.dimensions(i);
  }
  return index;
}

std::vector<int64> HloSharding::TileLimitForDevice(int64 device) const {
  CHECK(!IsTuple());
  CHECK(!maximal_);  // Maximal shardings do not have a valid tile shape.

  std::vector<int64> index = TileIndexForDevice(device);
  for (int64 i = 0; i < index.size(); ++i) {
    index[i] = (index[i] + 1) * tile_shape_.dimensions(i);
  }
  return index;
}

StatusOr<int64> HloSharding::UniqueDevice() const {
  if (IsTuple()) {
    if (tuple_elements_.empty()) {
      return tensorflow::errors::InvalidArgument(
          "UniqueDevice() called on empty tuple");
    }
    std::vector<StatusOr<int64>> results;
    std::transform(tuple_elements_.begin(), tuple_elements_.end(),
                   std::back_inserter(results),
                   [](const HloSharding& s) { return s.UniqueDevice(); });
    if (std::all_of(results.begin(), results.end(),
                    [&](const StatusOr<int64>& s) {
                      return s.ok() && results[0].ok() &&
                             s.ValueOrDie() == results[0].ValueOrDie();
                    })) {
      return results[0];
    } else {
      return tensorflow::errors::InvalidArgument(
          "Tuple did not contain a unique device");
    }
  }
  if (!replicated_ && maximal_ && !IsTuple()) {
    return static_cast<int64>(*tile_assignment_.begin());
  }
  return tensorflow::errors::InvalidArgument(
      "UniqueDevice() called on sharding that executes on multiple devices");
}

bool HloSharding::HasUniqueDevice() const {
  if (IsTuple()) {
    return UniqueDevice().status().ok();
  } else {
    return !IsReplicated() && IsTileMaximal();
  }
}

Status HloSharding::ValidateTuple(const Shape& shape, int64 num_devices) const {
  if (!ShapeUtil::IsTuple(shape)) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Sharding is tuple-shaped but validation shape is not."));
  }
  // The easiest way to get the number of elements in a nested tuple is just to
  // create a shape tree. We could call GetAsShapeTree, but that will try and
  // apply our tuple_shardings_ to the shape tree, and that might cause a crash
  // at this point as we haven't validated them.
  ShapeTree<bool> bool_shape_tree(shape, false);
  int64 num_leaves =
      std::distance(bool_shape_tree.leaf_begin(), bool_shape_tree.leaf_end());
  if (num_leaves != tuple_elements_.size()) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Validation tuple shape has ", num_leaves,
               " leaf elements, but this sharding contains ",
               tuple_elements_.size(), " elements."));
  }

  // Now we've validated the number of tuple elements, it's safe to request a
  // shape tree.
  ShapeTree<HloSharding> shape_tree = GetAsShapeTree(shape);
  for (const auto& index_to_sharding : shape_tree.leaves()) {
    if (index_to_sharding.first.empty()) {
      // An empty tuple has a ShapeTree with a single leaf at the empty index.
      continue;
    }
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
  if (ShapeUtil::IsTuple(shape)) {
    return tensorflow::errors::InvalidArgument(
        StrCat("Validation shape is a tuple but sharding is not."));
  }
  if (replicated_) {
    return Status::OK();
  }

  // All tile assignments must be less than the number of available cores and
  // unique.
  Status status = Status::OK();
  std::set<int64> seen_cores;
  tile_assignment_.Each(
      [&](tensorflow::gtl::ArraySlice<int64> indices, int32 core) {
        // Don't overwrite a bad status, so we report the first error.
        if (status.ok()) {
          if (core >= num_devices) {
            status = tensorflow::errors::InvalidArgument(StrCat(
                "core ", core, " > ", num_devices, " in tile assignment"));
          } else if (seen_cores.count(core) != 0) {
            status = tensorflow::errors::InvalidArgument(
                StrCat("core ", core, " is not unique in tile assignment"));
          }
        }
        seen_cores.insert(core);
      });
  if (!status.ok()) {
    return status;
  }

  if (IsTileMaximal()) {
    return Status::OK();
  }

  // The tile rank must be the same as the input rank.
  if (ShapeUtil::Rank(shape) != ShapeUtil::Rank(tile_shape_)) {
    return tensorflow::errors::InvalidArgument(
        "Tile rank is different to the input rank. sharding=", ToString(),
        ", input_shape=", ShapeUtil::HumanString(shape));
  }

  // The tile shape must not be the same as the input shape without maximal_
  // also set. If this is the case, we're not actually sharded and the correct
  // constructor should have been used.
  if (ShapeUtil::Equal(shape, tile_shape_)) {
    return tensorflow::errors::InvalidArgument(
        "Tile shape is the same as the input shape. If a replicated sharding "
        "was intended, use HloSharding::Replicated(). If a device placement "
        "was intended, use HloSharding::AssignDevice()");
  }

  // The tile shape must not be greater than the input shape in any dimension.
  for (int64 i = 0, e = ShapeUtil::Rank(shape); i != e; ++i) {
    auto tile_dim = tile_shape_.dimensions(i);
    auto shape_dim = shape.dimensions(i);
    if (tile_dim > shape_dim) {
      return tensorflow::errors::InvalidArgument(
          StrCat("Tile is larger than input shape (dimension ", i, ", ",
                 tile_dim, " > ", shape_dim));
    }
  }

  // The tile assignment tensor must be exactly dimensioned to ceil(shape[dim]
  // tile[dim]) for every dimension contained within tile.
  for (int64 i = 0, e = tile_assignment_.dimensions().size(); i != e; ++i) {
    int64 expected_dim =
        CeilOfRatio(shape.dimensions(i), tile_shape_.dimensions(i));
    if (tile_assignment_.dimensions()[i] != expected_dim) {
      return tensorflow::errors::InvalidArgument(
          StrCat("Tile assignment tensor has incorrect shape. Dimension ", i,
                 " expected ", expected_dim, " but got ",
                 tile_assignment_.dimensions()[i]));
    }
  }

  return Status::OK();
}

/*static*/ StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
  if (proto.type() == OpSharding::Type::OpSharding_Type_TUPLE) {
    std::vector<HloSharding> tuple_shardings;
    tuple_shardings.reserve(proto.tuple_shardings().size());
    for (const OpSharding& tuple_sharding_proto : proto.tuple_shardings()) {
      TF_ASSIGN_OR_RETURN(HloSharding sharding,
                          HloSharding::FromProto(tuple_sharding_proto));
      tuple_shardings.push_back(sharding);
    }
    return HloSharding(tuple_shardings);
  } else if (proto.type() == OpSharding::Type::OpSharding_Type_REPLICATED) {
    return Replicate();
  } else if (proto.type() == OpSharding::Type::OpSharding_Type_MAXIMAL ||
             proto.tile_assignment_devices().size() == 1) {
    return HloSharding(proto.tile_assignment_devices(0));
  }
  // Some versions of gcc cannot infer the TileAssignment constructor from a
  // braced initializer-list, so create one manually.
  std::vector<int64> devices(proto.tile_assignment_devices().begin(),
                             proto.tile_assignment_devices().end());
  Array<int64> tile_assignment(
      std::vector<int64>(proto.tile_assignment_dimensions().begin(),
                         proto.tile_assignment_dimensions().end()));
  std::copy(proto.tile_assignment_devices().begin(),
            proto.tile_assignment_devices().end(), tile_assignment.begin());
  return HloSharding(proto.tile_shape(), tile_assignment);
}

OpSharding HloSharding::ToProto() const {
  OpSharding result;

  if (IsTuple()) {
    for (const HloSharding& element : tuple_elements_) {
      *result.add_tuple_shardings() = element.ToProto();
    }
    result.set_type(OpSharding::Type::OpSharding_Type_TUPLE);
    return result;
  }

  *result.mutable_tile_shape() = tile_shape_;
  for (int64 dim : tile_assignment_.dimensions()) {
    result.add_tile_assignment_dimensions(dim);
  }
  for (auto device : tile_assignment_) {
    result.add_tile_assignment_devices(device);
  }
  if (IsReplicated()) {
    result.set_type(OpSharding::Type::OpSharding_Type_REPLICATED);
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::Type::OpSharding_Type_MAXIMAL);
  } else {
    result.set_type(OpSharding::Type::OpSharding_Type_OTHER);
  }
  return result;
}

HloSharding HloSharding::TransformShardedTileShape(
    const Shape& new_shape,
    const std::function<int64(int64, int64)>& transform) const {
  CHECK(!IsTuple());
  if (IsTileMaximal()) {
    return *this;
  }
  CHECK_EQ(ShapeUtil::Rank(new_shape), ShapeUtil::Rank(tile_shape()));
  Shape new_tile_shape;
  new_tile_shape.set_element_type(tile_shape().element_type());
  for (int64 i = 0; i < ShapeUtil::Rank(new_shape); ++i) {
    int64 dim;
    if (tile_assignment().dim(i) == 1) {
      dim = new_shape.dimensions(i);
    } else if (transform) {
      dim = transform(i, tile_shape().dimensions(i));
    } else {
      dim = tile_shape().dimensions(i);
    }
    new_tile_shape.add_dimensions(dim);
  }
  TF_CHECK_OK(
      LayoutUtil::CopyLayoutBetweenShapes(tile_shape_, &new_tile_shape));
  return HloSharding::Tile(new_tile_shape, tile_assignment());
}

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding) {
  out << sharding.ToString();
  return out;
}

}  // namespace xla
