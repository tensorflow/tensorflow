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

#include "xla/hlo/ir/hlo_sharding.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/overflow_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using absl::StrCat;

// Helper to group minor dimensions totaling a given group size while preserving
// V2 format. Returns true if such grouping is successful, otherwise returns
// false and will need to fallback to V1 sharding.
bool GroupMinorIotaDimsSorted(absl::Span<const int64_t> dims,
                              absl::Span<const int> perm, int64_t group_size,
                              absl::InlinedVector<int64_t, 6>& new_dims,
                              absl::InlinedVector<int, 6>& new_perm) {
  DCHECK_GT(group_size, 1);
  int grouped_dims = 0;
  std::optional<std::pair<int, int64_t>> split_dim_and_size;
  for (int i = perm.size() - 1; i >= 0; --i) {
    const int dim = perm[i];
    const int64_t dim_size = dims[dim];
    if (dim_size <= group_size) {
      if (group_size % dim_size != 0) {
        return false;
      }
      group_size /= dim_size;
      ++grouped_dims;
    } else {
      if (dim_size % group_size != 0) {
        return false;
      }
      split_dim_and_size.emplace(dim, dim_size / group_size);
      ++grouped_dims;
      group_size = 1;
      break;
    }
  }
  if (!split_dim_and_size) {
    new_dims.assign(dims.begin(), dims.end());
    new_perm.assign(perm.begin(), perm.end());
    std::stable_sort(new_perm.end() - grouped_dims, new_perm.end());
    return true;
  }
  new_dims.resize(dims.size() + 1);
  new_perm.resize(perm.size() + 1);
  const int split_i = split_dim_and_size->first;
  for (int i = 0; i < split_i; ++i) {
    new_dims[i] = dims[i];
  }
  new_dims[split_i] = split_dim_and_size->second;
  new_dims[split_i + 1] = dims[split_i] / split_dim_and_size->second;
  for (int i = split_i + 2; i < new_perm.size(); ++i) {
    new_dims[i] = dims[i - 1];
  }
  int perm_split = 0;
  for (int i = 0; i < perm.size(); ++i) {
    const int perm_dim = perm[i];
    new_perm[i] = perm_dim <= split_i ? perm_dim : (perm_dim + 1);
    if (perm_dim == split_i) {
      perm_split = i;
      break;
    }
  }
  new_perm[perm_split + 1] = new_perm[perm_split] + 1;
  for (int i = perm_split + 2; i < new_perm.size(); ++i) {
    const int perm_dim = perm[i - 1];
    new_perm[i] = perm_dim <= split_i ? perm_dim : (perm_dim + 1);
  }
  std::stable_sort(new_perm.end() - grouped_dims, new_perm.end());
  return true;
}

}  // namespace

HloSharding HloSharding::AssignDevice(int64_t device_id,
                                      absl::Span<const OpMetadata> metadata) {
  return HloSharding(device_id, metadata);
}

HloSharding HloSharding::Tile1D(const Shape& input_shape, int64_t num_tiles,
                                absl::Span<const OpMetadata> metadata) {
  CHECK_EQ(1, input_shape.dimensions().size());
  CHECK_GT(num_tiles, 1);
  absl::Span<const int64_t> dimensions(&num_tiles, 1);
  return HloSharding(TileAssignment(dimensions, dimensions, {0}),
                     /*replicate_on_last_tile_dim=*/false, metadata);
}

HloSharding HloSharding::PartialTile(
    const TileAssignment& tile_assignment_last_dim_replicate,
    absl::Span<const OpMetadata> metadata) {
  const size_t num_elements = tile_assignment_last_dim_replicate.num_elements();
  if (tile_assignment_last_dim_replicate.num_dimensions() == 1 ||
      tile_assignment_last_dim_replicate.dimensions().back() == num_elements) {
    return Replicate(metadata);
  }
  if (tile_assignment_last_dim_replicate.dimensions().back() == 1) {
    auto new_tile_dims = tile_assignment_last_dim_replicate.dimensions();
    new_tile_dims.remove_suffix(1);
    return HloSharding(
        tile_assignment_last_dim_replicate.Reshape(new_tile_dims),
        /*replicate_on_last_tile_dim=*/false, metadata);
  }
  const int64_t group_size =
      tile_assignment_last_dim_replicate.dimensions().back();
  if (tile_assignment_last_dim_replicate.iota_) {
    // Iota tile assignments are always sorted in the minor dimension.
    // Additionally if the minor most dimension is the combination multiple
    // dimensions in the transposed iota, these dimensions can be folded into
    // one.
    auto& iota = tile_assignment_last_dim_replicate.iota_.value();
    if (iota.reshape_dims()[iota.transpose_perm().back()] == group_size) {
      return HloSharding(tile_assignment_last_dim_replicate,
                         /*replicate_on_last_tile_dim=*/true, metadata);
    }
    absl::InlinedVector<int64_t, 6> new_reshape_dims;
    absl::InlinedVector<int, 6> new_transpose_perm;
    if (GroupMinorIotaDimsSorted(iota.reshape_dims(), iota.transpose_perm(),
                                 group_size, new_reshape_dims,
                                 new_transpose_perm)) {
      return HloSharding(
          TileAssignment(iota.dims(), new_reshape_dims, new_transpose_perm),
          /*replicate_on_last_tile_dim=*/true, metadata);
    }
  }

  std::shared_ptr<Array<int64_t>> sorted_tile =
      tile_assignment_last_dim_replicate.shared_array_clone();
  int64_t* sorted_tile_data = sorted_tile->data();
  int64_t* sorted_tile_data_end = sorted_tile_data + num_elements;
  while (sorted_tile_data < sorted_tile_data_end) {
    std::sort(sorted_tile_data, sorted_tile_data + group_size);
    sorted_tile_data += group_size;
  }
  DCHECK_EQ(sorted_tile_data, sorted_tile_data_end);

  return HloSharding(TileAssignment(std::move(sorted_tile)),
                     /*replicate_on_last_tile_dim=*/true, metadata);
}

HloSharding HloSharding::Subgroup(
    const TileAssignment& tile_assignment,
    absl::Span<const OpSharding::Type> subgroup_types,
    absl::Span<const OpMetadata> metadata) {
  if (subgroup_types.empty()) {
    return HloSharding(tile_assignment,
                       /*replicate_on_last_tile_dim=*/false, metadata);
  }
  // If there is only one type of subgrouping and there is no tiling on data
  // dimensions, it can be canonicalized to a simple manual/replicated sharding.
  if (absl::c_all_of(
          subgroup_types,
          [&](const OpSharding::Type t) { return t == subgroup_types[0]; }) &&
      Product(tile_assignment.dimensions().subspan(
          0, tile_assignment.num_dimensions() - subgroup_types.size())) == 1) {
    if (subgroup_types[0] == OpSharding::MANUAL) {
      return Manual(metadata);
    }
    if (subgroup_types[0] == OpSharding::REPLICATED) {
      return Replicate(metadata);
    }
  }
  // Normalize the subgroups to simplify two cases:
  //   - Remove trivial dims of size 1.
  //   - Merge dims of the same type.
  //   - Sort types.
  int64_t data_dims = tile_assignment.num_dimensions() - subgroup_types.size();
  absl::InlinedVector<int, 6> perm(data_dims);
  absl::c_iota(perm, 0);
  static_assert(sizeof(std::vector<int>) >=
                sizeof(absl::InlinedVector<int, 2>));
  std::array<absl::InlinedVector<int, 2>, OpSharding::Type_ARRAYSIZE>
      type_to_dims;
  int subgroup_count = 0;
  bool needs_merging = false;
  absl::InlinedVector<int, 4> removed_dims;
  for (int i = 0; i < subgroup_types.size(); ++i) {
    if (tile_assignment.dim(i + data_dims) == 1) {
      removed_dims.push_back(i + data_dims);
      needs_merging = true;
      continue;
    }
    auto& dims = type_to_dims[subgroup_types[i]];
    if (!dims.empty()) {
      needs_merging = true;
    } else {
      ++subgroup_count;
    }
    needs_merging |= !dims.empty();
    dims.push_back(i + data_dims);
  }
  needs_merging |= subgroup_count > 1;
  // Make sure the replicate dims are at the end so that we can leverage
  // PartialTile() to sort the elements.
  auto create_sharding = [](const TileAssignment& tiles,
                            absl::Span<const OpSharding::Type> types,
                            absl::Span<const OpMetadata> metadata) {
    if (types.size() == 1 && types.back() == OpSharding::REPLICATED) {
      // Normalize to partial tile.
      return PartialTile(tiles, metadata);
    }
    if (types.size() == 1 && types.back() == OpSharding::MANUAL &&
        tiles.num_elements() == tiles.dimensions().back()) {
      // Normalize to manual.
      return Manual(metadata);
    }
    if (!types.empty() && types.back() == OpSharding::REPLICATED) {
      // If the last type is REPLICATED, we first create a partially replicated
      // sharding without other subgroups so that the elements are sorted. Then
      // we fix the subgroup types.
      HloSharding sharding = PartialTile(tiles, metadata);
      sharding.replicate_on_last_tile_dim_ = false;
      for (const OpSharding::Type type : types) {
        sharding.subgroup_types_.push_back(type);
      }
      return sharding;
    }
    return HloSharding(tiles, types, metadata);
  };
  if (needs_merging) {
    auto data_tile_shape = tile_assignment.dimensions().subspan(0, data_dims);
    absl::InlinedVector<int64_t, 6> merged_shape(data_tile_shape.begin(),
                                                 data_tile_shape.end());
    absl::InlinedVector<int64_t, 6> transposed_shape = merged_shape;
    std::vector<OpSharding::Type> merged_types;
    static constexpr std::array<OpSharding::Type, OpSharding::Type_ARRAYSIZE>
        kOrderedTypes = {OpSharding::MAXIMAL,    OpSharding::TUPLE,
                         OpSharding::OTHER,      OpSharding::MANUAL,
                         OpSharding::REPLICATED, OpSharding::UNKNOWN};
    static_assert(kOrderedTypes[0] == 1 && kOrderedTypes[1] == 2 &&
                  kOrderedTypes[2] == 3 && kOrderedTypes[3] == 4 &&
                  kOrderedTypes[4] == 0 && kOrderedTypes[5] == 5);
    for (OpSharding::Type type : kOrderedTypes) {
      auto& dims = type_to_dims[type];
      if (dims.empty()) continue;
      int64_t dim_size = 1;
      for (int64_t dim : dims) {
        perm.push_back(dim);
        dim_size *= tile_assignment.dim(dim);
        transposed_shape.push_back(tile_assignment.dim(dim));
      }
      merged_shape.push_back(dim_size);
      merged_types.push_back(type);
    }
    TileAssignment new_tile_assignment = [&] {
      if (tile_assignment.iota_) {
        absl::c_copy(removed_dims, std::back_inserter(perm));
        auto transposed_iota = tile_assignment.iota_->Transpose(perm);
        if (transposed_iota) {
          return TileAssignment(merged_shape, transposed_iota->reshape_dims(),
                                transposed_iota->transpose_perm());
        }
      }
      auto new_tiles = std::make_shared<Array<int64_t>>(transposed_shape);
      new_tiles->Each([&](absl::Span<const int64_t> indices, int64_t* value) {
        std::vector<int64_t> src_indices(tile_assignment.num_dimensions(), 0);
        for (int64_t i = 0; i < indices.size(); ++i) {
          src_indices[perm[i]] = indices[i];
        }
        *value = tile_assignment(src_indices);
      });
      new_tiles->Reshape(merged_shape);
      return TileAssignment(std::move(new_tiles));
    }();

    return create_sharding(new_tile_assignment, merged_types, metadata);
  }
  return create_sharding(tile_assignment, subgroup_types, metadata);
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
    CHECK(!sharding.IsTuple())
        << sharding.ToString()
        << ", tuple shape = " << ShapeUtil::HumanString(tuple_shape);
  }
  std::vector<HloSharding> flattened_list(shardings.begin(), shardings.end());
  if (!flattened_list.empty()) {
    CHECK_EQ(flattened_list.size(), RequiredLeaves(tuple_shape))
        << "Flat list has " << flattened_list.size() << ", required "
        << RequiredLeaves(tuple_shape);
  }
  return HloSharding(std::move(flattened_list));
}

HloSharding HloSharding::SingleTuple(const Shape& tuple_shape,
                                     const HloSharding& sharding) {
  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  CHECK(!sharding.IsTuple()) << sharding.ToString();
  int64_t leaf_count = RequiredLeaves(tuple_shape);
  std::vector<HloSharding> flattened_list;
  flattened_list.resize(leaf_count, sharding);
  return HloSharding(std::move(flattened_list));
}

HloSharding HloSharding::Single(const Shape& shape,
                                const HloSharding& sharding) {
  return shape.IsTuple() ? SingleTuple(shape, sharding) : sharding;
}

void HloSharding::Print(Printer* printer, bool include_metadata) const {
  if (IsTuple()) {
    CHECK(metadata_.empty());
    if (ABSL_PREDICT_FALSE(tuple_elements_.empty())) {
      printer->Append("{}");
      return;
    }
    printer->Append("{");
    tuple_elements_[0].Print(printer, include_metadata);
    for (int i = 1; i < tuple_elements_.size(); ++i) {
      if (i % 5 == 0) {
        AppendCat(printer, ", /*index=", i, "*/");
      } else {
        printer->Append(", ");
      }
      tuple_elements_[i].Print(printer, include_metadata);
    }
    printer->Append("}");
    return;
  }

  auto print_metadata = [&] {
    if (include_metadata && !metadata_.empty()) {
      printer->Append(" metadata={");
      if (metadata_.size() == 1) {
        printer->Append(OpMetadataToString(metadata_.front()));
      } else {
        AppendJoin(printer, metadata_, ", ",
                   [](Printer* printer, auto& metadata) {
                     AppendCat(printer, "{", OpMetadataToString(metadata), "}");
                   });
      }
      printer->Append("}");
    }
  };
  auto print_shard_group = [&] {
    auto shard_group_str = shard_group_.ToString();
    if (!shard_group_str.empty()) {
      printer->Append(" " + shard_group_str);
    }
  };

  if (replicated_) {
    printer->Append("{replicated");
    print_shard_group();
    print_metadata();
    printer->Append("}");
    return;
  }

  if (manual_) {
    printer->Append("{manual");
    print_shard_group();
    print_metadata();
    printer->Append("}");
    return;
  }

  if (unknown_) {
    printer->Append("{unknown");
    print_shard_group();
    print_metadata();
    printer->Append("}");
    return;
  }

  if (maximal_) {
    AppendCat(printer, "{maximal device=",
              static_cast<int64_t>(*tile_assignment_.array().begin()));
    print_shard_group();
    print_metadata();
    printer->Append("}");
    return;
  }

  auto print_last_tile_dims = [&] {
    if (!subgroup_types_.empty()) {
      auto op_sharding_type_to_string = [](OpSharding::Type type) {
        switch (type) {
          case OpSharding::MANUAL:
            return "manual";
          case OpSharding::MAXIMAL:
            return "maximul";
          case OpSharding::REPLICATED:
            return "replicated";
          default:
            return "error_type.";
        }
      };
      printer->Append(" last_tile_dims={");
      AppendJoin(printer, subgroup_types_, ", ",
                 [&](Printer* printer, OpSharding::Type sharding_type) {
                   printer->Append(op_sharding_type_to_string(sharding_type));
                 });
      printer->Append("}");
    }
  };

  printer->Append("{");
  tile_assignment_.Print(printer);
  if (replicate_on_last_tile_dim_) {
    printer->Append(" last_tile_dim_replicate");
  }
  print_last_tile_dims();
  print_shard_group();
  print_metadata();
  printer->Append("}");
}

std::string HloSharding::ToString(bool include_metadata) const {
  StringPrinter printer;
  Print(&printer, include_metadata);
  return std::move(printer).ToString();
}

bool HloSharding::UsesDevice(int64_t device) const {
  if (IsTuple()) {
    return absl::c_any_of(tuple_elements_, [&](const HloSharding& s) {
      return s.UsesDevice(device);
    });
  }
  return replicated_ || manual_ || tile_assignment_.UsesDevice(device);
}

std::map<int64_t, int64_t> HloSharding::UsedDevices(int64_t* count) const {
  int64_t element_count = 1;
  std::map<int64_t, int64_t> device_map;
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

std::vector<int64_t> HloSharding::TileIndexForDevice(int64_t device) const {
  CHECK(!maximal_);
  CHECK(!IsManual());
  CHECK(!IsUnknown());
  CHECK(!IsTuple());
  std::vector<int64_t> ret_index;
  tile_assignment_.Each([&](absl::Span<const int64_t> index, int64_t d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  ret_index.resize(TiledDataRank());
  return ret_index;
}

int64_t HloSharding::DeviceForTileIndex(absl::Span<const int64_t> index) const {
  CHECK(!replicated_);
  CHECK(!IsManual());
  CHECK(!IsUnknown());
  CHECK(!IsTuple());
  if (maximal_) {
    return *tile_assignment_.array().begin();
  }
  if (index.size() == TiledDataRank() &&
      index.size() < tile_assignment_.num_dimensions()) {
    std::vector<int64_t> first_subgroup_index(index.begin(), index.end());
    for (int64_t i = 0; i < tile_assignment_.num_dimensions() - index.size();
         ++i) {
      first_subgroup_index.push_back(0);
    }
    return tile_assignment_(first_subgroup_index);
  }
  return tile_assignment_(index);
}

std::vector<int64_t> HloSharding::TileOffsetForDevice(const Shape& shape,
                                                      int64_t device) const {
  CHECK(!IsTuple());
  CHECK(!IsManual());
  CHECK(!IsUnknown());

  if (maximal_) {
    return std::vector<int64_t>(shape.dimensions().size(), 0);
  }
  CHECK_EQ(shape.dimensions().size(), TiledDataRank());
  std::vector<int64_t> index = TileIndexForDevice(device);
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    index[i] = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
  }
  return index;
}

std::vector<int64_t> HloSharding::TileLimitForDevice(const Shape& shape,
                                                     int64_t device) const {
  CHECK(!IsTuple());
  CHECK(!IsManual());
  CHECK(!IsUnknown());

  if (maximal_) {
    return std::vector<int64_t>(shape.dimensions().begin(),
                                shape.dimensions().end());
  }

  CHECK_EQ(shape.dimensions().size(), TiledDataRank());
  std::vector<int64_t> index = TileIndexForDevice(device);
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    index[i] = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
  }
  return index;
}

int64_t HloSharding::RequiredLeaves(const Shape& shape) {
  // Empty tuples (with arbitrary nesting) have no leaf nodes as far as
  // ShapeUtil and ShapeTree are concerned, but they do have a single
  // tuple_elements_ entry since we want to allow empty tuple results to
  // have sharding.
  const int64_t leaf_count = ShapeUtil::GetLeafCount(shape);
  return (leaf_count == 0) ? 1 : leaf_count;
}

absl::Status HloSharding::CheckLeafCount(const Shape& shape) const {
  int64_t leaf_count = ShapeUtil::GetLeafCount(shape);
  if (leaf_count == 0 && tuple_elements_.size() == 1) {
    // Allow (but don't require) empty tuples to have a single sharding
    return absl::OkStatus();
  }
  TF_RET_CHECK(leaf_count == tuple_elements_.size())
      << "Shape " << ShapeUtil::HumanString(shape) << " has " << leaf_count
      << " leaf nodes while this sharding has " << tuple_elements_.size();
  return absl::OkStatus();
}

absl::StatusOr<ShapeTree<HloSharding>> HloSharding::AsShapeTree(
    const Shape& shape) const {
  if (IsTuple()) {
    ShapeTree<HloSharding> result(shape, HloSharding::Replicate());
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    auto it = tuple_elements_.begin();
    for (auto& index_to_sharding : result.leaves()) {
      index_to_sharding.second = *it++;
    }
    return result;
  } else {
    return ShapeTree<HloSharding>(shape, *this);
  }
}

absl::StatusOr<HloSharding> HloSharding::GetTupleSharding(
    const Shape& shape) const {
  if (IsTuple()) {
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    return *this;
  }
  return SingleTuple(shape, *this);
}

HloSharding HloSharding::NormalizeTupleSharding(const Shape& shape) const {
  if (shape.IsTuple() && !IsTuple()) {
    return HloSharding::SingleTuple(shape, *this);
  }
  return *this;
}

std::optional<int64_t> HloSharding::UniqueDevice() const {
  if (IsTuple()) {
    if (tuple_elements_.empty()) {
      return std::nullopt;
    }
    std::optional<int64_t> unique_device;
    for (auto& tuple_sharding : tuple_elements_) {
      auto device = tuple_sharding.UniqueDevice();
      if (!device || (unique_device && *device != *unique_device)) {
        return std::nullopt;
      }
      unique_device = device;
    }
    return unique_device;
  }
  if (!replicated_ && maximal_) {
    return static_cast<int64_t>(*tile_assignment_.array().begin());
  }
  return std::nullopt;
}

int64_t HloSharding::GetUniqueDevice() const {
  auto device = UniqueDevice();
  CHECK(device) << "Sharding does not have a unique device: " << *this;
  return *device;
}

absl::Status HloSharding::ValidateTuple(
    const Shape& shape, std::optional<int64_t> num_devices) const {
  if (!shape.IsTuple()) {
    return tsl::errors::InvalidArgument(
        "Sharding is tuple-shaped but validation shape is not.");
  }
  TF_RETURN_IF_ERROR(CheckLeafCount(shape));
  if (ShapeUtil::GetLeafCount(shape) == 0 && tuple_elements_.empty()) {
    // Empty tuples are allowed to not have sharding
    return absl::OkStatus();
  }

  // Now we've validated the number of tuple elements, it's safe to request a
  // shape tree.
  ShapeTree<HloSharding> shape_tree = GetAsShapeTree(shape);
  for (const auto& index_to_sharding : shape_tree.leaves()) {
    absl::Status status = index_to_sharding.second.ValidateNonTuple(
        ShapeUtil::GetSubshape(shape, index_to_sharding.first), num_devices);
    if (!status.ok()) {
      tsl::errors::AppendToMessage(
          &status, StrCat("Note: While validating sharding tuple element ",
                          index_to_sharding.first.ToString(), " which is ",
                          index_to_sharding.second.ToString()));
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status HloSharding::Validate(const Shape& shape,
                                   std::optional<int64_t> num_devices) const {
  if (shape.IsToken()) {
    return absl::OkStatus();
  }
  absl::Status status = IsTuple() ? ValidateTuple(shape, num_devices)
                                  : ValidateNonTuple(shape, num_devices);
  if (!status.ok()) {
    tsl::errors::AppendToMessage(
        &status, StrCat("Note: While validating sharding ", ToString(),
                        " against shape ", ShapeUtil::HumanString(shape)));
  }
  return status;
}

absl::Status HloSharding::ValidateNonTuple(
    const Shape& shape, std::optional<int64_t> num_devices) const {
  if (shape.IsTuple()) {
    return absl::InvalidArgumentError(
        "Validation shape is a tuple but sharding is not.");
  }
  if (replicated_) {
    return absl::OkStatus();
  }

  // All tile assignments must be less than the number of available devices and
  // unique.
  bool all_devices_seen;
  if (!tile_assignment_.iota_) {
    absl::flat_hash_set<int64_t> seen_devices;
    absl::Status status = tile_assignment_.array().EachStatus(
        [&num_devices, &seen_devices](absl::Span<const int64_t> indices,
                                      int32_t device) {
          if (num_devices.has_value() && device >= *num_devices) {
            return absl::InvalidArgumentError(
                absl::StrCat("device ", device, " > num_devices (",
                             *num_devices, ") in tile assignment"));
          } else if (seen_devices.contains(device)) {
            return absl::InvalidArgumentError(absl::StrCat(
                "device ", device, " is not unique in tile assignment"));
          }
          seen_devices.insert(device);
          return absl::OkStatus();
        });
    TF_RETURN_IF_ERROR(status);
    all_devices_seen =
        !num_devices.has_value() || seen_devices.size() == *num_devices;
  } else {
    all_devices_seen = !num_devices.has_value() ||
                       tile_assignment_.iota_->num_elements() == *num_devices;
  }

  if (IsTileMaximal() || IsManual() || IsUnknown()) {
    return absl::OkStatus();
  }

  // The tile assignment tensor must have the same rank as the tiled data rank.
  if (shape.dimensions().size() != TiledDataRank()) {
    return tsl::errors::InvalidArgument(
        "Number of tile assignment dimensions (excluding subgroups) is "
        "different than the input rank. "
        "sharding=",
        ToString(), ", input_shape=", ShapeUtil::HumanString(shape));
  }

  // All devices should be seen in the tile assignment.
  if (!all_devices_seen) {
    return tsl::errors::InvalidArgument("tile_assignment should have ",
                                        *num_devices, " devices");
  }

  // The correct constructor has to be used to create tile maximal shardings.
  if (tile_assignment_.num_elements() == 1) {
    return tsl::errors::InvalidArgument(
        "Tile assignment only contains a single device. If a replicated "
        "sharding was intended, use HloSharding::Replicated(). If a device "
        "placement was intended, use HloSharding::AssignDevice()");
  }
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
  std::vector<OpMetadata> metadata(proto.metadata().begin(),
                                   proto.metadata().end());
  std::vector<int> subgroup_types_int(proto.last_tile_dims().begin(),
                                      proto.last_tile_dims().end());
  std::vector<OpSharding::Type> subgroup_types;
  absl::c_transform(
      subgroup_types_int, std::back_inserter(subgroup_types),
      [](const int type) { return static_cast<OpSharding::Type>(type); });
  if (proto.type() == OpSharding::TUPLE) {
    TF_RET_CHECK(metadata.empty())
        << "Tuple sharding is expected to have no metadata.";
    std::vector<HloSharding> tuple_shardings;
    tuple_shardings.reserve(proto.tuple_shardings().size());
    for (const OpSharding& tuple_sharding_proto : proto.tuple_shardings()) {
      TF_ASSIGN_OR_RETURN(HloSharding sharding,
                          HloSharding::FromProto(tuple_sharding_proto));
      tuple_shardings.push_back(std::move(sharding));
    }
    return std::move(
        HloSharding(std::move(tuple_shardings)).SetShardGroupFromProto(proto));
  } else if (proto.type() == OpSharding::REPLICATED) {
    return std::move(Replicate(metadata).SetShardGroupFromProto(proto));
  } else if (proto.type() == OpSharding::MANUAL) {
    return std::move(Manual(metadata).SetShardGroupFromProto(proto));
  } else if (proto.type() == OpSharding::UNKNOWN) {
    return std::move(Unknown(metadata).SetShardGroupFromProto(proto));
  } else if (proto.tile_assignment_devices().size() == 1) {
    return std::move(HloSharding(proto.tile_assignment_devices(0), metadata)
                         .SetShardGroupFromProto(proto));
  } else if (!proto.iota_reshape_dims().empty() &&
             absl::c_all_of(proto.iota_reshape_dims(),
                            [](int64_t d) { return d == 1; })) {
    return std::move(HloSharding(0, metadata).SetShardGroupFromProto(proto));
  }

  TF_RET_CHECK(proto.type() != OpSharding::MAXIMAL)
      << "Maximal sharding is expected to have single device assignment, but "
      << proto.tile_assignment_devices().size() << " has provided.";

  const bool use_iota_tile_assignments = !proto.iota_reshape_dims().empty();
  if (use_iota_tile_assignments) {
    TF_RET_CHECK(proto.tile_assignment_devices().empty());
    TF_RET_CHECK(proto.iota_reshape_dims().size() ==
                 proto.iota_transpose_perm().size());
  } else {
    TF_RET_CHECK(proto.tile_assignment_devices().size() > 1)
        << proto.ShortDebugString();
  }

  TF_RET_CHECK(!proto.tile_assignment_dimensions().empty());

  auto product_no_overflow =
      [](absl::Span<const int64_t> dims) -> absl::StatusOr<int64_t> {
    int64_t product_of_dimensions = 1;
    bool any_overflow = false;
    for (auto dimension : dims) {
      bool overflow = false;
      std::tie(product_of_dimensions, overflow) =
          OverflowSafeMultiply(product_of_dimensions, dimension);
    }
    TF_RET_CHECK(!any_overflow);
    return product_of_dimensions;
  };

  // RE: the product of tile assignment tensor dimensions must be
  // equal to tile_assignment_devices.size() or the product of iota_dimensions.
  TF_ASSIGN_OR_RETURN(int64_t product_of_dimensions,
                      product_no_overflow(proto.tile_assignment_dimensions()));
  if (use_iota_tile_assignments) {
    TF_ASSIGN_OR_RETURN(int64_t product_of_iota_dimensions,
                        product_no_overflow(proto.iota_reshape_dims()));
    TF_RET_CHECK(product_of_dimensions == product_of_iota_dimensions);
  } else {
    TF_RET_CHECK(product_of_dimensions ==
                 proto.tile_assignment_devices().size());
  }

  auto create_tile_assignment = [&] {
    if (use_iota_tile_assignments) {
      return TileAssignment(proto.tile_assignment_dimensions(),
                            proto.iota_reshape_dims(),
                            proto.iota_transpose_perm());
    }
    auto tiles =
        std::make_shared<Array<int64_t>>(proto.tile_assignment_dimensions());
    absl::c_copy(proto.tile_assignment_devices(), tiles->begin());
    return TileAssignment(std::move(tiles));
  };
  if (!subgroup_types.empty()) {
    TF_RET_CHECK(!proto.replicate_on_last_tile_dim());
    return std::move(
        Subgroup(create_tile_assignment(), subgroup_types, metadata)
            .SetShardGroupFromProto(proto));
  }
  if (proto.replicate_on_last_tile_dim()) {
    return std::move(PartialTile(create_tile_assignment(), metadata)
                         .SetShardGroupFromProto(proto));
  }
  return std::move(HloSharding(create_tile_assignment(),
                               /*replicate_on_last_tile_dim=*/false, metadata)
                       .SetShardGroupFromProto(proto));
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

  result.mutable_tile_assignment_dimensions()->Reserve(
      tile_assignment_.num_dimensions());
  absl::c_copy(tile_assignment_.dimensions(),
               tsl::protobuf::RepeatedFieldBackInserter(
                   result.mutable_tile_assignment_dimensions()));

  if (tile_assignment_.iota_) {
    result.mutable_iota_reshape_dims()->Reserve(
        tile_assignment_.iota_->reshape_dims().size());
    absl::c_copy(tile_assignment_.iota_->reshape_dims(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_iota_reshape_dims()));
    result.mutable_iota_transpose_perm()->Reserve(
        tile_assignment_.iota_->transpose_perm().size());
    absl::c_copy(tile_assignment_.iota_->transpose_perm(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_iota_transpose_perm()));
  } else {
    result.mutable_tile_assignment_devices()->Reserve(
        tile_assignment_.num_elements());
    absl::c_copy(tile_assignment_.array(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_tile_assignment_devices()));
  }
  if (IsReplicated()) {
    result.set_type(OpSharding::REPLICATED);
    result.clear_tile_assignment_dimensions();
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::MAXIMAL);
  } else if (IsManual()) {
    result.set_type(OpSharding::MANUAL);
    result.clear_tile_assignment_dimensions();
  } else if (IsUnknown()) {
    result.set_type(OpSharding::UNKNOWN);
    result.clear_tile_assignment_dimensions();
  } else {
    result.set_type(OpSharding::OTHER);
    result.set_replicate_on_last_tile_dim(ReplicateOnLastTileDim());
    for (auto type : subgroup_types_) {
      result.add_last_tile_dims(type);
    }
  }

  if (IsShardGroup()) {
    result.set_is_shard_group(true);
    result.set_shard_group_id(shard_group_.shard_group_id);
    if (shard_group_.shard_as) {
      result.set_shard_group_type(OpSharding::AS);
    } else {
      result.set_shard_group_type(OpSharding::LIKE);
    }
  }
  return result;
}

Shape HloSharding::TileShape(const Shape& shape) const {
  if (IsTileMaximal() || IsManual() || IsUnknown()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64_t i = 0; i < TiledDataRank(); ++i) {
    result_shape.set_dimensions(
        i, CeilOfRatio<int64_t>(shape.dimensions(i), tile_assignment_.dim(i)));
  }
  return result_shape;
}

Shape HloSharding::TileShape(const Shape& shape, int64_t device) const {
  if (IsTileMaximal() || IsManual() || IsUnknown()) {
    return shape;
  }

  std::vector<int64_t> index = TileIndexForDevice(device);
  Shape result_shape = shape;
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    int64_t offset = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
    int64_t limit = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
    result_shape.set_dimensions(i, limit - offset);
  }
  return result_shape;
}

int64_t HloSharding::TotalNumTiles() const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!IsUnknown());
  return Product(absl::Span<const int64_t>(tile_assignment_.dimensions()));
}

int64_t HloSharding::NumTiles() const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!IsUnknown());
  return Product(absl::Span<const int64_t>(tile_assignment_.dimensions())
                     .subspan(0, TiledDataRank()));
}

int64_t HloSharding::NumTilesLeaf() const {
  DCHECK(!IsTuple());
  if (IsTileMaximalLeaf()) {
    return 1;
  }
  CHECK(!IsManualLeaf() && !IsUnknownLeaf());
  return Product(absl::Span<const int64_t>(tile_assignment_.dimensions())
                     .subspan(0, TiledDataRankLeaf()));
}

int64_t HloSharding::NumTiles(absl::Span<const int64_t> dims) const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!ReplicateOnLastTileDim() ||
        !absl::c_linear_search(dims, tile_assignment().num_dimensions() - 1));
  int64_t num_tiles = 1;
  for (auto d : dims) {
    CHECK(d < tile_assignment().num_dimensions());
    num_tiles *= tile_assignment().dim(d);
  }
  return num_tiles;
}

HloSharding HloSharding::GetSubSharding(const Shape& shape,
                                        const ShapeIndex& index) const {
  CHECK(IsTuple());
  int64_t sharding_index = 0;
  const Shape* sub_shape = &shape;
  for (int64_t idx : index) {
    for (int64_t i = 0; i < idx; ++i) {
      sharding_index += ShapeUtil::GetLeafCount(sub_shape->tuple_shapes(i));
    }
    sub_shape = &sub_shape->tuple_shapes(idx);
  }
  if (sub_shape->IsTuple()) {
    auto begin_it = tuple_elements_.begin() + sharding_index;
    return HloSharding::Tuple(
        *sub_shape,
        absl::MakeConstSpan(
            &*begin_it,
            &*(begin_it + ShapeUtil::GetLeafCountTuple(*sub_shape))));
  } else {
    return tuple_elements_[sharding_index];
  }
}

std::optional<HloSharding> HloSharding::ExtractSingleSharding() const {
  if (!IsTuple()) {
    return *this;
  }
  if (tuple_elements_.empty()) {
    return std::nullopt;
  }
  for (int64_t i = 1; i < tuple_elements_.size(); ++i) {
    if (tuple_elements_[0] != tuple_elements_[i]) {
      return std::nullopt;
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

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding) {
  out << sharding.ToString();
  return out;
}

}  // namespace xla
