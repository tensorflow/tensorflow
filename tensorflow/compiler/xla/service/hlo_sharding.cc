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

namespace xla {

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
  string result = StrCat("{", (replicated_ ? " replicated" : ""),
                         (maximal_ ? " maximal" : ""));

  if (replicated_) {
    return "{replicated}";
  } else if (maximal_) {
    return StrCat(
        "{maximal device=", static_cast<int64>(*tile_assignment_.begin()), "}");
  } else {
    return StrCat("{", ShapeUtil::HumanString(tile_shape_), " ",
                  "devices=", VectorString(tile_assignment_), "}");
  }
}

bool HloSharding::UsesDevice(int64 device) const {
  const auto& devices = tile_assignment_;
  return replicated_ ||
         std::find(devices.begin(), devices.end(), device) != devices.end();
}

std::vector<int64> HloSharding::TileIndexForDevice(int64 device) const {
  CHECK(!ShapeUtil::IsTuple(tile_shape_));
  CHECK(!maximal_);
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
  if (maximal_) {
    return *tile_assignment_.begin();
  }
  CHECK_EQ(ShapeUtil::Rank(tile_shape_), tile_assignment_.dimensions().size());
  return tile_assignment_(index);
}

std::vector<int64> HloSharding::TileOffsetForDevice(int64 device) const {
  CHECK(!ShapeUtil::IsTuple(tile_shape_));

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
  CHECK(!ShapeUtil::IsTuple(tile_shape_));
  CHECK(!maximal_);  // Maximal shardings do not have a valid tile shape.

  std::vector<int64> index = TileIndexForDevice(device);
  for (int64 i = 0; i < index.size(); ++i) {
    index[i] = (index[i] + 1) * tile_shape_.dimensions(i);
  }
  return index;
}

StatusOr<int64> HloSharding::UniqueDevice() const {
  if (!replicated_ && maximal_) {
    return static_cast<int64>(*tile_assignment_.begin());
  }
  return tensorflow::errors::InvalidArgument(
      "UniqueDevice() called on sharding that executes on multiple devices");
}

Status HloSharding::Validate(const Shape& shape, int64 num_devices) const {
  if (replicated_) {
    return Status::OK();
  }

  // All tile assignments must be less than the number of available cores and
  // unique.
  Status status = Status::OK();
  std::set<int64> seen_cores;
  tile_assignment_.Each(
      [&](tensorflow::gtl::ArraySlice<int64> indices, uint32 core) {
        // Don't overwrite a bad status, so we report the first error.
        if (status.ok()) {
          if (core >= num_devices) {
            status =
                tensorflow::errors::InvalidArgument(tensorflow::strings::StrCat(
                    "core ", core, " > ", num_devices, " in tile assignment"));
          } else if (seen_cores.count(core) != 0) {
            status =
                tensorflow::errors::InvalidArgument(tensorflow::strings::StrCat(
                    "core ", core, " is not unique in tile assignment"));
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
        "Tile rank is different to the input rank");
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
      return tensorflow::errors::InvalidArgument(tensorflow::strings::StrCat(
          "Tile is larger than input shape (dimension ", i, ", ", tile_dim,
          " > ", shape_dim));
    }
  }

  // The tile assignment tensor must be exactly dimensioned to ceil(shape[dim]
  // tile[dim]) for every dimension contained within tile.
  for (int64 i = 0, e = tile_assignment_.dimensions().size(); i != e; ++i) {
    int64 expected_dim =
        CeilOfRatio(shape.dimensions(i), tile_shape_.dimensions(i));
    if (tile_assignment_.dimensions()[i] != expected_dim) {
      return tensorflow::errors::InvalidArgument(tensorflow::strings::StrCat(
          "Tile assignment tensor has incorrect shape. Dimension ", i,
          " expected ", expected_dim, " but got ",
          tile_assignment_.dimensions()[i]));
    }
  }

  return Status::OK();
}

/*static*/ StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
  if (proto.type() == OpSharding::Type::OpSharding_Type_REPLICATED) {
    return Replicate();
  } else if (proto.type() == OpSharding::Type::OpSharding_Type_MAXIMAL) {
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

}  // namespace xla
