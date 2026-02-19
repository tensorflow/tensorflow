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
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
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

// Advances the specified set of indexes and returns true if we haven't
// wrapped around (i.e. result isn't {0, 0, ...}).
bool NextIndex(absl::InlinedVector<int64_t, 6>* index,
               absl::Span<const int64_t> limit) {
  DCHECK_LE(index->size(), limit.size());
  for (int64_t i = index->size() - 1; i >= 0; --i) {
    ++(*index)[i];
    if ((*index)[i] < limit[i]) {
      return true;
    }
    (*index)[i] = 0;
  }
  return false;
}

std::vector<AxisRef> GetOrderedAxisRefs(const NamedSharding& sharding) {
  absl::flat_hash_map<int64_t, std::vector<int64_t>> axis_index_to_pre_sizes;
  const Mesh& mesh = sharding.mesh();
  for (int64_t i = 0; i < mesh.axis_sizes().size(); ++i) {
    axis_index_to_pre_sizes[i].push_back(1);
    axis_index_to_pre_sizes[i].push_back(mesh.axis_sizes()[i]);
  }

  auto collect_axis_ref = [&](const AxisRef& axis_ref) {
    if (axis_ref.sub_axis_info()) {
      axis_index_to_pre_sizes[axis_ref.mesh_axis_index()].push_back(
          axis_ref.sub_axis_info()->pre_size);
      axis_index_to_pre_sizes[axis_ref.mesh_axis_index()].push_back(
          axis_ref.sub_axis_info()->next_pre_size());
    }
  };

  for (const NamedSharding::DimensionSharding& dim_sharding :
       sharding.dim_shardings()) {
    for (const AxisRef& axis_ref : dim_sharding.axes()) {
      collect_axis_ref(axis_ref);
    }
  }
  for (const AxisRef& axis_ref : sharding.replicated_axes()) {
    collect_axis_ref(axis_ref);
  }
  for (const AxisRef& axis_ref : sharding.unreduced_axes()) {
    collect_axis_ref(axis_ref);
  }
  for (const AxisRef& axis_ref : sharding.manual_axes()) {
    collect_axis_ref(axis_ref);
  }

  std::vector<AxisRef> axis_refs;
  for (int64_t i = 0; i < mesh.axis_sizes().size(); ++i) {
    std::vector<int64_t>& pre_sizes = axis_index_to_pre_sizes[i];
    absl::c_sort(pre_sizes);
    pre_sizes.erase(std::unique(pre_sizes.begin(), pre_sizes.end()),
                    pre_sizes.end());
    if (pre_sizes.size() == 2) {
      axis_refs.push_back(AxisRef(i));
      continue;
    }
    for (int64_t j = 0; j < pre_sizes.size() - 1; ++j) {
      int64_t pre_size = pre_sizes[j];
      int64_t size = pre_sizes[j + 1] / pre_size;
      axis_refs.push_back(AxisRef(i, {pre_size, size}));
    }
  }
  return axis_refs;
}

}  // namespace

HloSharding HloSharding::AssignDevice(int64_t device_id,
                                      absl::Span<const OpMetadata> metadata,
                                      bool use_named_sharding) {
  if (use_named_sharding) {
    return HloSharding(NamedSharding::MaximalSharding(device_id, metadata));
  }
  return HloSharding(device_id, metadata);
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
  // dimensions, it can be canonicalized to a simple manual/replicated/unreduced
  // sharding.
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
    if (subgroup_types[0] == OpSharding::UNREDUCED) {
      return Unreduced(metadata);
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
        kOrderedTypes = {OpSharding::MAXIMAL,   OpSharding::TUPLE,
                         OpSharding::OTHER,     OpSharding::MANUAL,
                         OpSharding::UNREDUCED, OpSharding::REPLICATED,
                         OpSharding::UNKNOWN};
    static_assert(kOrderedTypes[0] == 1 && kOrderedTypes[1] == 2 &&
                  kOrderedTypes[2] == 3 && kOrderedTypes[3] == 4 &&
                  kOrderedTypes[4] == 6 && kOrderedTypes[5] == 0 &&
                  kOrderedTypes[6] == 5);
    for (OpSharding::Type type : kOrderedTypes) {
      auto& dims = type_to_dims[type];
      if (dims.empty()) {
        continue;
      }
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

  if (UseNamedShardingLeaf()) {
    printer->Append(named_sharding_->ToString(include_metadata));
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

  if (unreduced_) {
    printer->Append("{unreduced");
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
            return "maximal";
          case OpSharding::REPLICATED:
            return "replicated";
          case OpSharding::UNREDUCED:
            return "unreduced";
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

  return IsReplicatedLeaf() || IsManualLeaf() ||
         TileAgnosticDeviceAssignment().UsesDevice(device);
}

std::vector<int64_t> HloSharding::TileIndexForDevice(int64_t device) const {
  CHECK(!maximal_);
  CHECK(!IsManual());
  CHECK(!IsUnknown());
  CHECK(!IsTuple());
  std::vector<int64_t> ret_index;
  EachTile([&](absl::Span<const int64_t> index, int64_t d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  ret_index.resize(TiledDataRank());
  return ret_index;
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
    index[i] =
        std::min(index[i] * CeilOfRatio(shape_dim, dimension(i)), shape_dim);
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
    index[i] = std::min((index[i] + 1) * CeilOfRatio(shape_dim, dimension(i)),
                        shape_dim);
  }
  return index;
}

void HloSharding::EachTile(
    absl::FunctionRef<void(absl::Span<const int64_t>, int64_t)> f) const {
  if (UseNamedShardingLeaf()) {
    V3ToV2Sharding(*named_sharding_).EachTile(f);
    return;
  }
  return tile_assignment_.Each(f);
}

absl::Status HloSharding::EachTile(
    absl::Span<const int64_t> dims,
    absl::FunctionRef<void(int64_t, absl::Span<const int64_t>,
                           absl::Span<const int64_t>)>
        f) const {
  CHECK(!IsTuple());
  CHECK(!IsManual());
  CHECK(!IsUnknown());
  CHECK(!maximal_);

  // At the high-level, sharding_dims[i] describes the number of ways the shape
  // is partitioned along i-th dimension. Note that sharding_dims[i] with i >=
  // dims.size() encodes other information such as subgroups to express partial
  // replication/sharding and other semantics.  They do not participate in
  // determining the tile origin and shape.
  const absl::Span<const int64_t> sharding_dims = dimensions();

  if (dims.size() != TiledDataRank()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Shape rank is not same as tile rank: %d vs %d",
                        dims.size(), TiledDataRank()));
  }

  if (UseNamedShardingLeaf()) {
    return V3ToV2Sharding(*named_sharding_).EachTile(dims, f);
  }

  absl::InlinedVector<int64_t, 6> tile_dims;
  tile_dims.reserve(dims.size());
  for (int64_t i = 0; i < dims.size(); ++i) {
    tile_dims.push_back(CeilOfRatio(dims[i], sharding_dims[i]));
  }

  const int64_t replication_dim = SubgroupReplicationDim();
  int64_t num_replicas;
  if (replication_dim == -1) {
    num_replicas = 1;
  } else {
    num_replicas = sharding_dims[replication_dim];
  }

  // Enumerate over all indices of tiles. For instance, if sharding_dims is [3,
  // 2], iterate over [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]. If
  // sharding_dims includes replication, we only enumerate over the sharding
  // portion, and copy the same indices multiple times.
  absl::InlinedVector<int64_t, 6> unique_tile_index(dims.size());
  absl::InlinedVector<int64_t, 6> tile_offset(dims.size());
  absl::InlinedVector<int64_t, 6> tile_limit(dims.size());
  int64_t flat_tile_index = 0;
  const int64_t* flat_tile_assignment = tile_assignment().array().data();
  do {
    for (int64_t i = 0; i < dims.size(); ++i) {
      tile_offset[i] = std::min(tile_dims[i] * unique_tile_index[i], dims[i]);
      tile_limit[i] =
          std::min(tile_dims[i] * (unique_tile_index[i] + 1), dims[i]);
    }
    for (int64_t i = 0; i < num_replicas; ++i) {
      CHECK_LT(flat_tile_index, num_devices());
      const int64_t device_id = flat_tile_assignment[flat_tile_index];
      if (device_id < 0 || device_id >= num_devices()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Out of range device id in device_assignment: %d; "
                            "valid range: [0, %d)",
                            device_id, num_devices()));
      }
      f(device_id, tile_offset, tile_limit);
      ++flat_tile_index;
    }
  } while (NextIndex(&unique_tile_index, sharding_dims));
  return absl::OkStatus();
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
  }
  return ShapeTree<HloSharding>(shape, *this);
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

  if (!IsReplicatedLeaf() && IsTileMaximalLeaf()) {
    return static_cast<int64_t>(
        *TileAgnosticDeviceAssignment().array().begin());
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
    return absl::InvalidArgumentError(
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

namespace {
absl::Status DeviceInRange(int64_t device, std::optional<int64_t> num_devices) {
  if (device < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("device %d is negative in tile assignment", device));
  }
  if (num_devices.has_value() && device >= *num_devices) {
    return absl::InvalidArgumentError(
        absl::StrFormat("device %d >= num_devices (%d) in tile assignment",
                        device, *num_devices));
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status HloSharding::ValidateNonTuple(
    const Shape& shape, std::optional<int64_t> num_devices) const {
  if (shape.IsTuple()) {
    return absl::InvalidArgumentError(
        "Validation shape is a tuple but sharding is not.");
  }
  if (IsReplicatedLeaf() || IsManualLeaf() || IsUnreducedLeaf() ||
      IsUnknownLeaf()) {
    return absl::OkStatus();
  }

  if (IsTileMaximalLeaf()) {
    CHECK(!TileAgnosticDeviceAssignment().iota_);
    if (TileAgnosticDeviceAssignment().array().num_elements() != 1) {
      return absl::InvalidArgumentError(
          "Tile maximal sharding must have a single device assignment.");
    }
    return DeviceInRange(TileAgnosticDeviceAssignment().first(), num_devices);
  }

  // The correct constructor has to be used to create tile maximal shardings.
  if (TileAgnosticDeviceAssignment().num_elements() == 1) {
    return absl::InvalidArgumentError(
        "Tile assignment only contains a single device. If a replicated "
        "sharding was intended, use HloSharding::Replicated(). If a device "
        "placement was intended, use HloSharding::AssignDevice()");
  }

  // The tile assignment tensor must have the same rank as the tiled data rank.
  if (shape.dimensions().size() != TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of tile assignment dimensions (excluding subgroups) is "
        "different than the input rank. sharding=",
        ToString(), ", input_shape=", ShapeUtil::HumanString(shape)));
  }

  if (UseNamedShardingLeaf()) {
    if (num_devices.has_value() && this->num_devices() != *num_devices) {
      return absl::InvalidArgumentError(
          absl::StrFormat("sharding should have %d devices but has %d",
                          *num_devices, this->num_devices()));
    }
    return absl::OkStatus();
  }

  if (tile_assignment_.iota_) {
    if (num_devices.has_value() &&
        tile_assignment_.iota_->num_elements() != *num_devices) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "tile_assignment should have %d devices but has %d", *num_devices,
          tile_assignment_.iota_->num_elements()));
    }
    return absl::OkStatus();
  }

  absl::flat_hash_set<int64_t> seen_devices;
  absl::Status status = tile_assignment_.array().EachStatus(
      [&num_devices, &seen_devices](absl::Span<const int64_t> indices,
                                    int64_t device) {
        TF_RETURN_IF_ERROR(DeviceInRange(device, num_devices));
        if (!seen_devices.insert(device).second) {
          return absl::InvalidArgumentError(absl::StrCat(
              "device ", device, " is not unique in tile assignment"));
        }
        return absl::OkStatus();
      });
  TF_RETURN_IF_ERROR(status);
  if (num_devices.has_value() && seen_devices.size() != *num_devices) {
    return absl::InvalidArgumentError(
        absl::StrFormat("tile_assignment should have %d devices but has %d",
                        *num_devices, seen_devices.size()));
  }

  return absl::OkStatus();
}

const TileAssignment& HloSharding::TileAgnosticDeviceAssignment() const {
  // Returns device assignment regardless of sharding tiling.
  //  - named_sharding_->device_assignment() only contains the information of
  //    the mesh without information of how axes are used.
  //  - tile_assignment_ keeps the information of mesh and how axes are used.
  //
  // For example, a NamedSharding [mesh= ['a'=2, 'b'=2] {'a'}, {}] and
  // HloSharding [2,1,2]<=4 last_tile_dim_replicate would have the same
  // underlying device order as: {{0, 1}, {2, 3}}.
  if (UseNamedShardingLeaf()) {
    return named_sharding_->device_assignment();
  }
  return tile_assignment_;
}

/*static*/ absl::StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
  if (proto.has_named_sharding()) {
    return HloSharding(NamedSharding::FromProto(proto.named_sharding()));
  }

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
  }
  if (proto.type() == OpSharding::REPLICATED) {
    return std::move(Replicate(metadata).SetShardGroupFromProto(proto));
  }
  if (proto.type() == OpSharding::MANUAL) {
    return std::move(Manual(metadata).SetShardGroupFromProto(proto));
  }
  if (proto.type() == OpSharding::UNREDUCED) {
    return std::move(Unreduced(metadata).SetShardGroupFromProto(proto));
  }
  if (proto.type() == OpSharding::UNKNOWN) {
    return std::move(Unknown(metadata).SetShardGroupFromProto(proto));
  }
  if (proto.type() == OpSharding::MAXIMAL) {
    TF_RET_CHECK(proto.tile_assignment_devices().size() == 1)
        << "Maximal sharding is expected to have single device assignment, but "
        << proto.tile_assignment_devices().size() << " has provided.";
    return std::move(HloSharding(proto.tile_assignment_devices(0), metadata)
                         .SetShardGroupFromProto(proto));
  }

  TF_RET_CHECK(proto.type() == OpSharding::OTHER);
  const bool use_iota_tile_assignments = !proto.iota_reshape_dims().empty();
  if (use_iota_tile_assignments) {
    TF_RET_CHECK(proto.tile_assignment_devices().empty());
    TF_RET_CHECK(proto.iota_reshape_dims().size() ==
                 proto.iota_transpose_perm().size());
  } else {
    TF_RET_CHECK(!proto.tile_assignment_devices().empty());
  }

  // If there is only one device, returns replicated sharding.
  if (use_iota_tile_assignments &&
      absl::c_all_of(proto.iota_reshape_dims(),
                     [](int64_t d) { return d == 1; })) {
    return std::move(Replicate(metadata).SetShardGroupFromProto(proto));
  }
  if (!use_iota_tile_assignments &&
      proto.tile_assignment_devices().size() == 1) {
    return std::move(Replicate(metadata).SetShardGroupFromProto(proto));
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

  if (UseNamedShardingLeaf()) {
    *result.mutable_named_sharding() = named_sharding_->ToProto();
    return result;
  }

  result.mutable_metadata()->Reserve(metadata_.size());
  for (const auto& metadata : metadata_) {
    *result.add_metadata() = metadata;
  }

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
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::MAXIMAL);
  } else if (IsManual()) {
    result.set_type(OpSharding::MANUAL);
  } else if (IsUnreduced()) {
    result.set_type(OpSharding::UNREDUCED);
  } else if (IsUnknown()) {
    result.set_type(OpSharding::UNKNOWN);
  } else {
    result.set_type(OpSharding::OTHER);
    result.set_replicate_on_last_tile_dim(ReplicateOnLastTileDim());
    for (auto type : subgroup_types_) {
      result.add_last_tile_dims(type);
    }
    result.mutable_tile_assignment_dimensions()->Reserve(num_dimensions());
    absl::c_copy(dimensions(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_tile_assignment_dimensions()));
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

/*static*/ HloSharding HloSharding::V3ToV2Sharding(
    const NamedSharding& sharding) {
  // TODO(b/477900810): Remove sharding conversions.
  LOG(WARNING) << "V3ToV2Sharding method involves sharding conversions for "
                  "HloShardingV3, its use cases should be avoided.";
  const Mesh& mesh = sharding.mesh();
  absl::Span<const OpMetadata> metadata = sharding.metadata();
  if (sharding.IsReplicated()) {
    return HloSharding::Replicate(metadata);
  }
  if (sharding.IsMaximal()) {
    return HloSharding::AssignDevice(mesh.device_assignment()(0), metadata);
  }

  std::vector<int64_t> tile_assignment_dims;
  tile_assignment_dims.reserve(sharding.dim_shardings().size());
  absl::flat_hash_map<AxisRef, int64_t> axis_ref_to_sharded_pos;
  int64_t sharded_pos = 0;
  for (const NamedSharding::DimensionSharding& dim_sharding :
       sharding.dim_shardings()) {
    tile_assignment_dims.push_back(dim_sharding.getShardedSize(mesh));
    for (const AxisRef& axis_ref : dim_sharding.axes()) {
      axis_ref_to_sharded_pos[axis_ref] = sharded_pos++;
    }
  }

  std::vector<OpSharding::Type> types;
  auto add_subgroup_axes = [&](absl::Span<const AxisRef> axes,
                               OpSharding::Type type) {
    if (axes.empty()) {
      return;
    }
    types.push_back(type);
    int64_t& dim = tile_assignment_dims.emplace_back(1);
    for (const AxisRef& axis_ref : axes) {
      dim *= axis_ref.size(mesh);
      axis_ref_to_sharded_pos[axis_ref] = sharded_pos++;
    }
  };
  add_subgroup_axes(sharding.manual_axes(), OpSharding::MANUAL);
  add_subgroup_axes(sharding.unreduced_axes(), OpSharding::UNREDUCED);

  std::vector<AxisRef> mesh_axis_refs = GetOrderedAxisRefs(sharding);
  std::vector<int64_t> reshape_dims;
  reshape_dims.reserve(mesh_axis_refs.size());
  std::vector<int> transpose_perm(mesh_axis_refs.size());

  int64_t total_replicated_size = 1;
  int64_t replicated_pos = sharded_pos;
  for (int64_t i = 0; i < mesh_axis_refs.size(); ++i) {
    const AxisRef& axis_ref = mesh_axis_refs[i];
    reshape_dims.push_back(axis_ref.size(mesh));

    auto sharded_pos_it = axis_ref_to_sharded_pos.find(axis_ref);
    if (sharded_pos_it == axis_ref_to_sharded_pos.end()) {
      transpose_perm[replicated_pos++] = i;
      total_replicated_size *= axis_ref.size(mesh);
    } else {
      transpose_perm[sharded_pos_it->second] = i;
    }
  }

  if (total_replicated_size > 1) {
    tile_assignment_dims.push_back(total_replicated_size);
    types.push_back(OpSharding::REPLICATED);
  }

  if (mesh.device_assignment().iota().has_value() &&
      mesh.device_assignment().iota()->reshape_dims().size() == 1) {
    // Simple iota case
    return HloSharding::Subgroup(
        TileAssignment(tile_assignment_dims, reshape_dims, transpose_perm),
        types, metadata);
  }

  TileAssignment tile_assignment = mesh.device_assignment()
                                       .Reshape(reshape_dims)
                                       .Transpose(transpose_perm)
                                       .Reshape(tile_assignment_dims);
  return HloSharding::Subgroup(tile_assignment, types, metadata);
}

/*static*/ HloSharding HloSharding::ToNamedSharding(
    const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    std::vector<HloSharding> v3_elements;
    v3_elements.reserve(sharding.tuple_elements().size());
    for (const HloSharding& element : sharding.tuple_elements()) {
      v3_elements.push_back(ToNamedSharding(element));
    }
    return HloSharding::FlatTuple(std::move(v3_elements));
  }
  if (sharding.UseNamedShardingLeaf()) {
    return sharding;
  }
  if (sharding.IsReplicated()) {
    return HloSharding(NamedSharding::Replicate(sharding.metadata()));
  }
  if (sharding.IsTileMaximal()) {
    return HloSharding(NamedSharding::MaximalSharding(
        sharding.tile_assignment().first(), sharding.metadata()));
  }

  // Tiled sharding.
  const TileAssignment& tile_assignment = sharding.tile_assignment();
  std::vector<int64_t> mesh_dims;
  std::vector<int64_t> perm;

  // Check if the device assignment is equivalent to a simple iota (0, 1, 2,
  // ...). If so, we can treat the mesh as having the same dimensions as the
  // tile assignment and use an identity permutation.
  bool effective_iota = true;
  int64_t expected_device = 0;
  tile_assignment.array().Each([&](absl::Span<const int64_t>, int64_t device) {
    if (device != expected_device++) {
      effective_iota = false;
    }
  });

  // Determine the dimensions of the underlying mesh and the permutation from
  // mesh axes to tile dimensions.
  if (effective_iota) {
    mesh_dims.assign(tile_assignment.dimensions().begin(),
                     tile_assignment.dimensions().end());
    perm.resize(mesh_dims.size());
    absl::c_iota(perm, 0);
  } else if (tile_assignment.iota()) {
    mesh_dims.assign(tile_assignment.iota()->reshape_dims().begin(),
                     tile_assignment.iota()->reshape_dims().end());
    perm.assign(tile_assignment.iota()->transpose_perm().begin(),
                tile_assignment.iota()->transpose_perm().end());
  } else {
    mesh_dims.assign(tile_assignment.dimensions().begin(),
                     tile_assignment.dimensions().end());
    perm.resize(mesh_dims.size());
    absl::c_iota(perm, 0);
  }

  // When converting back to NamedSharding (v3), we construct a new mesh where
  // the axis names are simply their indices ("axis_0", "axis_1", ...).
  std::vector<std::string> axes_names_storage;
  axes_names_storage.reserve(mesh_dims.size());
  for (int64_t i = 0; i < mesh_dims.size(); ++i) {
    axes_names_storage.push_back(absl::StrCat("axis_", i));
  }
  // Note that we use `tile_assignment.dimensions()` which are the dimensions
  // after any reshape/transpose operations defined in the V2 sharding.
  // Consequently, any transpose in V2 is 'baked' into the resulting Mesh, and
  // the dimension mappings in V3 are always identity (Tensor Dim i -> Mesh Axis
  // i) for the tiled dimensions.
  std::vector<absl::string_view> axes_names_views(axes_names_storage.begin(),
                                                  axes_names_storage.end());

  Mesh mesh = (effective_iota || tile_assignment.iota())
                  ? Mesh(mesh_dims, axes_names_views)
                  : Mesh(tile_assignment.array(), axes_names_views);

  // Map the data dimensions to the corresponding mesh axes using the calculated
  // permutation.
  std::vector<NamedSharding::DimensionSharding> dim_shardings;
  int64_t tiled_data_rank = sharding.TiledDataRank();
  dim_shardings.reserve(tiled_data_rank);

  int64_t perm_idx = 0;
  auto consume_axes = [&](int64_t target_size) {
    std::vector<AxisRef> axes;
    int64_t accumulated_size = 1;
    while (accumulated_size < target_size && perm_idx < perm.size()) {
      int64_t axis_idx = perm[perm_idx];
      accumulated_size *= mesh_dims[axis_idx];
      axes.push_back(AxisRef(axis_idx));
      perm_idx++;
    }
    CHECK_EQ(accumulated_size, target_size)
        << "Unable to map tile dimension size " << target_size
        << " to mesh axes. Accumulated size: " << accumulated_size;
    return axes;
  };

  for (int64_t i = 0; i < tiled_data_rank; ++i) {
    dim_shardings.push_back(NamedSharding::DimensionSharding(
        consume_axes(tile_assignment.dimensions()[i]), /*is_closed=*/true));
  }

  std::vector<AxisRef> replicated_axes;
  std::vector<AxisRef> unreduced_axes;
  std::vector<AxisRef> manual_axes;

  // Handle subgroup types which correspond to additional dimensions in the tile
  // assignment (beyond TiledDataRank). These are mapped to specific axis types
  // (manual, unreduced, replicated) in V3.
  int64_t dim_idx = tiled_data_rank;
  for (OpSharding::Type type : sharding.subgroup_types()) {
    std::vector<AxisRef> axes =
        consume_axes(tile_assignment.dimensions()[dim_idx++]);
    CHECK(type == OpSharding::REPLICATED || type == OpSharding::MANUAL ||
          type == OpSharding::UNREDUCED)
        << "Unsupported sharding type: " << OpSharding::Type_Name(type);
    if (type == OpSharding::MANUAL) {
      manual_axes.insert(manual_axes.end(), axes.begin(), axes.end());
    } else if (type == OpSharding::UNREDUCED) {
      unreduced_axes.insert(unreduced_axes.end(), axes.begin(), axes.end());
    } else if (type == OpSharding::REPLICATED) {
      replicated_axes.insert(replicated_axes.end(), axes.begin(), axes.end());
    }
  }

  return HloSharding(NamedSharding(mesh, dim_shardings, replicated_axes,
                                   unreduced_axes, manual_axes,
                                   sharding.metadata()));
}

Shape HloSharding::TileShape(const Shape& shape) const {
  if (IsTileMaximal() || IsManual() || IsUnreduced() || IsUnknown()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64_t i = 0; i < TiledDataRank(); ++i) {
    result_shape.set_dimensions(
        i, CeilOfRatio<int64_t>(shape.dimensions(i), dimension(i)));
  }
  return result_shape;
}

Shape HloSharding::TileShape(const Shape& shape, int64_t device) const {
  if (IsTileMaximal() || IsManual() || IsUnreduced() || IsUnknown()) {
    return shape;
  }

  std::vector<int64_t> index = TileIndexForDevice(device);
  Shape result_shape = shape;
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    int64_t offset =
        std::min(index[i] * CeilOfRatio(shape_dim, dimension(i)), shape_dim);
    int64_t limit = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, dimension(i)), shape_dim);
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
  return Product(dimensions());
}

int64_t HloSharding::NumTiles() const {
  if (IsTileMaximalLeaf()) {
    return 1;
  }
  CHECK(!IsManualLeaf() && !IsUnknownLeaf());
  return Product(dimensions().subspan(0, TiledDataRank()));
}

int64_t HloSharding::NumTiles(absl::Span<const int64_t> dims) const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!ReplicateOnLastTileDim() ||
        !absl::c_linear_search(dims, num_dimensions() - 1));
  int64_t num_tiles = 1;
  for (auto d : dims) {
    CHECK(d < num_dimensions());
    num_tiles *= dimension(d);
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
  }
  return tuple_elements_[sharding_index];
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
