/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/spmd/spmd_partitioner_util.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

namespace {
using hlo_sharding_util::DeviceGroupTileAssignment;
using hlo_sharding_util::GroupedSharding;
}  // namespace

Window GenNewWindow(const HloInstruction* original_dot,
                    const HloInstruction* dot_lhs,
                    const HloInstruction* dot_rhs, int64_t lhs_concat_dim,
                    int64_t rhs_concat_dim, bool windowed_at_contracting_dims,
                    bool windowed_at_batch_dims) {
  auto new_window = original_dot->window();
  const ConvolutionDimensionNumbers& conv_dnums =
      original_dot->convolution_dimension_numbers();
  if (lhs_concat_dim != -1) {
    for (int64_t i = 0; i < conv_dnums.input_spatial_dimensions_size(); ++i) {
      if (conv_dnums.input_spatial_dimensions(i) == lhs_concat_dim) {
        auto wd = new_window.mutable_dimensions(i);
        auto lhs_size = dot_lhs->shape().dimensions(lhs_concat_dim + 1);
        if (windowed_at_contracting_dims) {
          wd->set_size(lhs_size);
        }
        if (windowed_at_batch_dims) {
          wd->set_size(lhs_size);
          wd->set_padding_low(0);
          wd->set_padding_high(0);
          wd->set_stride(std::max<int64_t>(1, lhs_size - 1));
          wd->set_window_dilation(1);
          wd->set_base_dilation(lhs_size);
          wd->set_window_reversal(false);
        }
      }
    }
  }
  if (rhs_concat_dim != -1) {
    for (int64_t i = 0; i < conv_dnums.kernel_spatial_dimensions_size(); ++i) {
      if (conv_dnums.kernel_spatial_dimensions(i) == rhs_concat_dim &&
          !windowed_at_contracting_dims && !windowed_at_batch_dims &&
          lhs_concat_dim == -1) {
        auto wd = new_window.mutable_dimensions(i);
        auto rhs_size = dot_rhs->shape().dimensions(rhs_concat_dim + 1);
        wd->set_size(rhs_size);
        wd->set_padding_low(rhs_size - 1);
        wd->set_padding_high(rhs_size - 1);
      }
    }
  }
  // Add the extra dimension to window.
  WindowDimension* new_dim = new_window.add_dimensions();
  if (windowed_at_contracting_dims) {
    new_dim->set_size(2);
    new_dim->set_padding_low(0);
    new_dim->set_padding_high(0);
    new_dim->set_stride(1);
    new_dim->set_window_dilation(1);
    new_dim->set_base_dilation(1);
    new_dim->set_window_reversal(false);
  } else if (windowed_at_batch_dims) {
    new_dim->set_size(2);
    new_dim->set_padding_low(0);
    new_dim->set_padding_high(0);
    new_dim->set_stride(1);  // std::max<int64_t>(1, 2 - 1)
    new_dim->set_window_dilation(1);
    new_dim->set_base_dilation(2);
    new_dim->set_window_reversal(false);
  } else {
    if (lhs_concat_dim != -1) {
      new_dim->set_size(1);
      new_dim->set_padding_low(0);
      new_dim->set_padding_high(0);
      new_dim->set_stride(1);
      new_dim->set_window_dilation(1);
      new_dim->set_base_dilation(1);
      new_dim->set_window_reversal(false);
    }
    if (rhs_concat_dim != -1) {
      new_dim->set_size(2);          // rhs_size
      new_dim->set_padding_low(1);   // rhs_size - 1
      new_dim->set_padding_high(1);  // rhs_size - 1
      new_dim->set_stride(1);
      new_dim->set_window_dilation(1);
      new_dim->set_base_dilation(1);
      new_dim->set_window_reversal(true);
    }
  }

  VLOG(2) << "new_window: " << new_window.ShortDebugString();
  return new_window;
}

ConvolutionDimensionNumbers GenNewConvDNums(
    const HloInstruction* original_dot, const HloInstruction* dot_lhs,
    const HloInstruction* dot_rhs, int64_t lhs_concat_dim,
    int64_t rhs_concat_dim, bool windowed_at_contracting_dims,
    bool windowed_at_batch_dims,
    const std::vector<int64_t>& lhs_to_output_indices,
    const std::vector<int64_t>& rhs_to_output_indices,
    const Shape& new_dot_shape) {
  // Generate the new conv dimension numbers.
  const ConvolutionDimensionNumbers& dnums =
      original_dot->convolution_dimension_numbers();
  // Handle the LHS dimension numbers.
  int64_t input_batch_dimension = dnums.input_batch_dimension();
  int64_t input_feature_dimension = dnums.input_feature_dimension();
  std::vector<int64_t> input_spatial_dimensions(
      dnums.input_spatial_dimensions().begin(),
      dnums.input_spatial_dimensions().end());
  if (lhs_concat_dim != -1) {
    if (lhs_concat_dim <= input_batch_dimension) {
      input_batch_dimension++;
    }
    if (lhs_concat_dim <= input_feature_dimension) {
      input_feature_dimension++;
    }
    for (int64_t i = 0; i < input_spatial_dimensions.size(); ++i) {
      if (lhs_concat_dim <= input_spatial_dimensions[i]) {
        input_spatial_dimensions[i]++;
      }
    }
    input_spatial_dimensions.push_back(lhs_concat_dim);
  }
  if (rhs_concat_dim != -1 && !windowed_at_contracting_dims &&
      !windowed_at_batch_dims) {
    input_spatial_dimensions.push_back(dot_lhs->shape().dimensions().size() -
                                       1);
  }
  // Handle the RHS dimension numbers.
  int64_t kernel_input_feature_dimension =
      dnums.kernel_input_feature_dimension();
  int64_t kernel_output_feature_dimension =
      dnums.kernel_output_feature_dimension();
  std::vector<int64_t> kernel_spatial_dimensions(
      dnums.kernel_spatial_dimensions().begin(),
      dnums.kernel_spatial_dimensions().end());
  if (rhs_concat_dim != -1) {
    if (rhs_concat_dim <= kernel_input_feature_dimension) {
      kernel_input_feature_dimension++;
    }
    if (rhs_concat_dim <= kernel_output_feature_dimension) {
      kernel_output_feature_dimension++;
    }
    for (int64_t i = 0; i < kernel_spatial_dimensions.size(); ++i) {
      if (rhs_concat_dim <= kernel_spatial_dimensions[i]) {
        kernel_spatial_dimensions[i]++;
      }
    }
    kernel_spatial_dimensions.push_back(rhs_concat_dim);
  }
  if (lhs_concat_dim != -1 && !windowed_at_contracting_dims &&
      !windowed_at_batch_dims) {
    kernel_spatial_dimensions.push_back(dot_rhs->shape().dimensions().size() -
                                        1);
  }
  // Handle the Output dimension numbers.
  int64_t output_batch_dimension = dnums.output_batch_dimension();
  int64_t output_feature_dimension = dnums.output_feature_dimension();
  std::vector<int64_t> output_spatial_dimensions(
      dnums.output_spatial_dimensions().begin(),
      dnums.output_spatial_dimensions().end());
  if (!windowed_at_contracting_dims) {
    auto output_slice_dim = lhs_concat_dim != -1
                                ? lhs_to_output_indices[lhs_concat_dim]
                                : rhs_to_output_indices[rhs_concat_dim];
    if (output_slice_dim <= output_batch_dimension) {
      output_batch_dimension++;
    }
    if (output_slice_dim <= output_feature_dimension) {
      output_feature_dimension++;
    }
    for (int64_t i = 0; i < output_spatial_dimensions.size(); ++i) {
      if (output_slice_dim <= output_spatial_dimensions[i]) {
        output_spatial_dimensions[i]++;
      }
    }
    output_spatial_dimensions.push_back(output_slice_dim);
  } else {
    output_spatial_dimensions.push_back(new_dot_shape.dimensions().size() - 1);
  }
  // Construct the new dot dimension numbers.
  ConvolutionDimensionNumbers new_dnums;
  new_dnums.set_input_batch_dimension(input_batch_dimension);
  new_dnums.set_input_feature_dimension(input_feature_dimension);
  for (auto dim : input_spatial_dimensions) {
    new_dnums.add_input_spatial_dimensions(dim);
  }
  new_dnums.set_kernel_input_feature_dimension(kernel_input_feature_dimension);
  new_dnums.set_kernel_output_feature_dimension(
      kernel_output_feature_dimension);
  for (auto dim : kernel_spatial_dimensions) {
    new_dnums.add_kernel_spatial_dimensions(dim);
  }
  new_dnums.set_output_batch_dimension(output_batch_dimension);
  new_dnums.set_output_feature_dimension(output_feature_dimension);
  for (auto dim : output_spatial_dimensions) {
    new_dnums.add_output_spatial_dimensions(dim);
  }

  return new_dnums;
}

bool HasReplicatedSharding(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return absl::c_any_of(sharding.tuple_elements(), HasReplicatedSharding);
  }
  return sharding.IsReplicated();
}

HloComputation* MakeBinaryAdd(PrimitiveType type, HloModule* module) {
  HloComputation::Builder sum_b("add");
  auto x = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(type, {}), "x"));
  auto y = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(type, {}), "y"));
  if (type == PRED) {
    sum_b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), HloOpcode::kOr, x, y));
  } else {
    sum_b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), HloOpcode::kAdd, x, y));
  }
  HloComputation* reduction = module->AddEmbeddedComputation(sum_b.Build());
  return reduction;
}

bool EvenlyPartitions(const Shape& shape, const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      if (!EvenlyPartitions(ShapeUtil::GetTupleElementShape(shape, i),
                            sharding.GetSubSharding(shape, {i}))) {
        return false;
      }
    }
  }
  if (sharding.IsManual()) {
    return true;
  }
  if (sharding.IsTileMaximal()) {
    return sharding.IsReplicated();
  }
  if (shape.IsArray()) {
    for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
      if (shape.dimensions(i) % sharding.tile_assignment().dim(i) != 0) {
        return false;
      }
    }
  }
  return true;
}

Shape MakePartitionedShape(const Shape& shape, const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    std::vector<Shape> subshapes;
    const int64_t shape_n = ShapeUtil::TupleElementCount(shape);
    subshapes.reserve(shape_n);
    for (int64_t i = 0; i < shape_n; ++i) {
      subshapes.push_back(
          MakePartitionedShape(ShapeUtil::GetTupleElementShape(shape, i),
                               sharding.GetSubSharding(shape, {i})));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  }
  return sharding.TileShape(shape);
}

int64_t ShapeSizeInBytes(const Shape& shape) {
  if (shape.IsTuple()) {
    int64_t total_size = 0;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      total_size += ShapeSizeInBytes(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return total_size;
  }
  return ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) *
         ShapeUtil::ElementsIn(shape);
}

Shape MakeNonPaddedShapeForGivenPartition(const Shape& shape,
                                          const HloSharding& sharding,
                                          int64_t partition_id) {
  if (sharding.IsTuple()) {
    std::vector<Shape> subshapes;
    const int64_t shape_n = ShapeUtil::TupleElementCount(shape);
    subshapes.reserve(shape_n);
    for (int64_t i = 0; i < shape_n; ++i) {
      subshapes.push_back(MakeNonPaddedShapeForGivenPartition(
          ShapeUtil::GetTupleElementShape(shape, i),
          sharding.GetSubSharding(shape, {i}), partition_id));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  }

  if (sharding.IsReplicated()) {
    return shape;
  }
  if (sharding.IsTileMaximal()) {
    if (partition_id == *sharding.UniqueDevice()) {
      return shape;
    }
    return ShapeUtil::MakeTupleShape({});
  }

  auto partition_shape = shape;
  std::vector<int64_t> tile_offset =
      sharding.TileOffsetForDevice(shape, partition_id);
  std::vector<int64_t> tile_limit =
      sharding.TileLimitForDevice(shape, partition_id);
  for (int64_t i = 0; i < tile_offset.size(); ++i) {
    if (sharding.UsesDevice(partition_id)) {
      partition_shape.set_dimensions(i, tile_limit[i] - tile_offset[i]);
    } else {
      partition_shape.set_dimensions(i, 0);
    }
  }
  return partition_shape;
}

std::vector<HloInstruction*> MakePartitionOffsets(
    const Shape& shape, const HloSharding& sharding,
    HloInstruction* partition_id, SpmdBuilder* b,
    absl::Span<const int64_t> dims) {
  CHECK(!shape.IsTuple());

  auto shard_shape = MakePartitionedShape(shape, sharding);
  std::vector<HloInstruction*> offsets;

  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    if (sharding.tile_assignment().dim(i) == 1 ||
        (!dims.empty() && !absl::c_linear_search(dims, i))) {
      offsets.push_back(b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32))));
    } else {
      std::vector<int32_t> offset_array(
          sharding.tile_assignment().num_elements());
      sharding.tile_assignment().Each(
          [&](absl::Span<const int64_t> indices, int64_t device) {
            offset_array[device] = indices[i] * shard_shape.dimensions(i);
          });
      offsets.push_back(
          TableLookup<int32_t>(offset_array, S32, partition_id, b));
    }
  }

  return offsets;
}

std::vector<HloInstruction*> MakeTiledPartitionOrdinals(
    const HloSharding& sharding, HloInstruction* partition_id, SpmdBuilder* b) {
  CHECK(!sharding.IsTileMaximal());
  auto dimensions = sharding.tile_assignment().dimensions();
  if (sharding.ReplicateOnLastTileDim()) {
    dimensions.remove_suffix(1);
  }
  auto table_shape = ShapeUtil::MakeShape(S32, dimensions);
  return MakePartitionOffsets(table_shape, sharding, partition_id, b);
}

Shape GetPaddedShapeForUnevenPartitioning(const Shape& base_shape,
                                          const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    return base_shape;
  }
  if (EvenlyPartitions(base_shape, sharding)) {
    return base_shape;
  }
  auto shard_shape = MakePartitionedShape(base_shape, sharding);
  Shape padded_base_shape = base_shape;
  for (int64_t i = 0; i < padded_base_shape.dimensions().size(); ++i) {
    padded_base_shape.set_dimensions(
        i, shard_shape.dimensions(i) * sharding.tile_assignment().dim(i));
  }
  return padded_base_shape;
}

HloInstruction* GetInGroupPartitionId(
    HloInstruction* partition_id,
    const DeviceGroupTileAssignment& device_groups, SpmdBuilder* b) {
  std::vector<uint32_t> in_group_ids(device_groups.num_total_devices());
  for (uint32_t i = 0; i < device_groups.num_groups(); ++i) {
    for (uint32_t j = 0; j < device_groups.num_devices_per_group(); ++j) {
      in_group_ids[device_groups(i, j)] = j;
    }
  }
  return TableLookup<uint32_t>(in_group_ids, U32, partition_id, b);
}

namespace {

bool IsIota(const Array<int64_t>& x) {
  for (int64_t i = 0; i < x.num_elements(); ++i) {
    if (x(0, i) != i) {
      return false;
    }
  }
  return true;
}

// Expand the device groups, making each device group follow the format of the
// partition group.
std::vector<std::vector<int64_t>> ExpandDeviceGroups(
    const DeviceGroupTileAssignment& device_groups,
    const std::vector<std::vector<int64_t>>& partition_subgroups) {
  // Example: Given device groups of {{0,1,2,3},{4,5,6,7}} and partition
  // subgroups of {{0,2}, {1,3}} returns device groups of {{0,2}, {1,3}, {4,6},
  // {5,7}}
  if (partition_subgroups.empty()) {
    return device_groups.flattened_device_groups();
  }
  std::vector<std::vector<int64_t>> result(partition_subgroups.size() *
                                           device_groups.num_groups());
  for (int64_t g = 0; g < device_groups.num_groups(); ++g) {
    for (int64_t i = 0; i < partition_subgroups.size(); ++i) {
      result[g * partition_subgroups.size() + i].resize(
          partition_subgroups[i].size());
      for (int64_t j = 0; j < partition_subgroups[i].size(); ++j) {
        result[g * partition_subgroups.size() + i][j] =
            device_groups(g, partition_subgroups[i][j]);
      }
    }
  }
  return result;
}

// Expand the device groups, making each device group follow the format of the
// iota partition group list.
std::optional<IotaReplicaGroupList> ExpandDeviceGroupsWithIota(
    const DeviceGroupTileAssignment& device_groups,
    const IotaReplicaGroupList& partition_group_list) {
  // Example: Given device groups of devices=[2,4]<=[8] and partition
  // group list of devices=[2,2]<=[2,2]T(1,0) returns device groups of
  // devices=[4,2]<=[2,2,2]T(0,2,1).

  // If device groups are not in iota format, we cannot expand the device
  // groups into an iota format.
  if (!device_groups.has_iota()) {
    return std::nullopt;
  }
  CHECK(partition_group_list.num_replica_groups() *
            partition_group_list.num_devices_per_group() ==
        device_groups.num_devices_per_group());
  int64_t final_num_replica_groups =
      partition_group_list.num_replica_groups() * device_groups.num_groups();
  int64_t final_num_devices_per_group =
      partition_group_list.num_devices_per_group();
  // 1. Split the 2nd dimension of device groups to match dimensions of
  // partition group list.
  // 2. Apply transpose to the split dimensions matching the partition group
  // list transpose perm.
  std::vector<int64_t> reshape_dims =
      std::vector<int64_t>(partition_group_list.reshape_dims().begin(),
                           partition_group_list.reshape_dims().end());
  reshape_dims.insert(reshape_dims.begin(), device_groups.num_groups());
  std::vector<int> transpose_perm =
      std::vector<int>(partition_group_list.transpose_perm().begin(),
                       partition_group_list.transpose_perm().end());
  for (int64_t i = 0; i < transpose_perm.size(); ++i) {
    transpose_perm[i] += 1;
  }
  transpose_perm.insert(transpose_perm.begin(), 0);
  TileAssignment processed_device_groups =
      device_groups.Reshape(reshape_dims).Transpose(transpose_perm);
  // If after transpose we don't have an iota, then we can't expand the device
  // groups with iota.
  if (!processed_device_groups.iota().has_value()) {
    return std::nullopt;
  }

  if (processed_device_groups.iota().has_value()) {
    return IotaReplicaGroupList(
        final_num_replica_groups, final_num_devices_per_group,
        processed_device_groups.iota()->reshape_dims(),
        processed_device_groups.iota()->transpose_perm());
  }
  return std::nullopt;
}

SPMDCollectiveOpsCreator GetPerGroupCollectiveOpsCreator(
    const SPMDCollectiveOpsCreator& creator,
    const DeviceGroupTileAssignment& device_groups) {
  if (device_groups.num_groups() == 1) {
    // If the device_groups has an IotaTileAssignment and has a naive reshape
    // dim of size 1, it is an iota device group.
    if (device_groups.has_iota() &&
        device_groups.iota()->reshape_dims().size() == 1) {
      return creator;
    }
    if (IsIota(device_groups.array())) {
      return creator;
    }
  }

  SPMDCollectiveOpsCreator result;
  auto device_groups_ptr =
      std::make_shared<const DeviceGroupTileAssignment>(device_groups);
  result.create_partition_id = [creator, device_groups_ptr](SpmdBuilder* b) {
    return GetInGroupPartitionId(creator.create_partition_id(b),
                                 *device_groups_ptr, b);
  };
  result.create_cross_partition_all_reduce =
      [creator, device_groups_ptr](
          SpmdBuilder* b, HloInstruction* operand, HloComputation* reduction,
          const std::vector<std::vector<int64_t>>& partition_subgroups,
          int64_t channel_id) {
        return creator.create_cross_partition_all_reduce(
            b, operand, reduction,
            ExpandDeviceGroups(*device_groups_ptr, partition_subgroups),
            channel_id);
      };
  result.create_cross_partition_all_reduce_with_iota_device_list =
      [creator, device_groups_ptr](
          SpmdBuilder* b, HloInstruction* operand, HloComputation* reduction,
          const IotaReplicaGroupList& partition_group_list,
          int64_t channel_id) {
        // Try to expand the device group list, but if this fails fallback
        // to creating collective with list of list of integers representation.
        std::optional<IotaReplicaGroupList> expanded_iota_partition_group_list =
            ExpandDeviceGroupsWithIota(*device_groups_ptr,
                                       partition_group_list);
        if (!expanded_iota_partition_group_list.has_value()) {
          return creator.create_cross_partition_all_reduce(
              b, operand, reduction,
              ExpandDeviceGroups(
                  *device_groups_ptr,
                  partition_group_list.flattened_replica_groups()),
              channel_id);
        }
        return creator.create_cross_partition_all_reduce_with_iota_device_list(
            b, operand, reduction, *expanded_iota_partition_group_list,
            channel_id);
      };
  result.create_cross_partition_collective_permute =
      [creator, device_groups_ptr](
          SpmdBuilder* b, HloInstruction* operand,
          std::vector<std::pair<int64_t, int64_t>>& src_dst_pairs,
          int64_t next_channel_id) {
        std::vector<std::pair<int64_t, int64_t>> expanded_pairs(
            src_dst_pairs.size() * device_groups_ptr->num_groups());
        for (int64_t g = 0; g < device_groups_ptr->num_groups(); ++g) {
          for (int64_t i = 0; i < src_dst_pairs.size(); ++i) {
            expanded_pairs[g * src_dst_pairs.size() + i] =
                std::pair<int64_t, int64_t>{
                    device_groups_ptr->array()(g, src_dst_pairs[i].first),
                    device_groups_ptr->array()(g, src_dst_pairs[i].second)};
          }
        }
        return creator.create_cross_partition_collective_permute(
            b, operand, expanded_pairs, next_channel_id);
      };
  result.create_cross_partition_all_to_all =
      [creator, device_groups_ptr](
          SpmdBuilder* b, absl::Span<HloInstruction* const> operands,
          const std::vector<std::vector<int64_t>>& partition_subgroups,
          int64_t channel_id, std::optional<int64_t> split_dimension) {
        return creator.create_cross_partition_all_to_all(
            b, operands,
            ExpandDeviceGroups(*device_groups_ptr, partition_subgroups),
            channel_id, split_dimension);
      };
  result.create_cross_partition_all_to_all_with_iota_device_list =
      [creator, device_groups_ptr](
          SpmdBuilder* b, absl::Span<HloInstruction* const> operands,
          const IotaReplicaGroupList& partition_group_list, int64_t channel_id,
          std::optional<int64_t> split_dimension) {
        // Try to expand the partition group list, but if this fails fallback
        // to creating collective with list of list of integers representation.
        std::optional<IotaReplicaGroupList> expanded_iota_partition_group_list =
            ExpandDeviceGroupsWithIota(*device_groups_ptr,
                                       partition_group_list);
        if (!expanded_iota_partition_group_list.has_value()) {
          return creator.create_cross_partition_all_to_all(
              b, operands,
              ExpandDeviceGroups(
                  *device_groups_ptr,
                  partition_group_list.flattened_replica_groups()),
              channel_id, split_dimension);
        }
        return creator.create_cross_partition_all_to_all_with_iota_device_list(
            b, operands, *expanded_iota_partition_group_list, channel_id,
            split_dimension);
      };
  if (creator.create_cross_partition_all_gather) {
    result.create_cross_partition_all_gather =
        [creator, device_groups_ptr](
            SpmdBuilder* b, HloInstruction* operand, const Shape& ag_shape,
            const std::vector<std::vector<int64_t>>& partition_subgroups,
            int64_t channel_id, int64_t all_gather_dimension) {
          return creator.create_cross_partition_all_gather(
              b, operand, ag_shape,
              ExpandDeviceGroups(*device_groups_ptr, partition_subgroups),
              channel_id, all_gather_dimension);
        };
  }
  if (creator.create_cross_partition_all_gather_with_iota_device_list) {
    result.create_cross_partition_all_gather_with_iota_device_list =
        [creator, device_groups_ptr](
            SpmdBuilder* b, HloInstruction* operand, const Shape& ag_shape,
            const IotaReplicaGroupList& partition_group_list,
            int64_t channel_id, int64_t all_gather_dimension) {
          // Try to expand the device group list, but if this fails fallback
          // to creating collective with list of list of integers
          // representation.
          std::optional<IotaReplicaGroupList>
              expanded_iota_partition_group_list = ExpandDeviceGroupsWithIota(
                  *device_groups_ptr, partition_group_list);
          if (!expanded_iota_partition_group_list.has_value()) {
            return creator.create_cross_partition_all_gather(
                b, operand, ag_shape,
                ExpandDeviceGroups(
                    *device_groups_ptr,
                    partition_group_list.flattened_replica_groups()),
                channel_id, all_gather_dimension);
          }
          return creator
              .create_cross_partition_all_gather_with_iota_device_list(
                  b, operand, ag_shape, *expanded_iota_partition_group_list,
                  channel_id, all_gather_dimension);
        };
  }
  return result;
}

}  // namespace

std::optional<HloSharding> PartialReplicateReshardCompatibleSharding(
    const HloSharding& partial_sharding, const HloSharding& target_sharding) {
  if (!partial_sharding.ReplicateOnLastTileDim()) {
    return std::nullopt;
  }
  if (partial_sharding.tile_assignment().num_elements() !=
      target_sharding.tile_assignment().num_elements()) {
    return std::nullopt;
  }
  const int64_t rank = partial_sharding.TiledDataRank();
  if (rank != target_sharding.TiledDataRank()) {
    return std::nullopt;
  }

  // A dimension is expanded when target_tile_size > partial_tile_size and
  // target_tile_size % partial_tile_size == 0.
  // expand_tile_dims_positions is the index of the expand_dim.
  std::vector<int64_t> expand_tile_dims_indices(rank, -1);
  // expand_tile_size = target_tile_size / partial_tile_size.
  std::vector<int64_t> expand_tile_sizes;
  int64_t num_expand_dims = 0;
  for (int64_t dim = 0; dim < rank; dim++) {
    int64_t partial_tile_size = partial_sharding.tile_assignment().dim(dim);
    int64_t target_tile_size = target_sharding.tile_assignment().dim(dim);
    if (target_tile_size % partial_tile_size != 0) {
      return std::nullopt;
    }

    if (target_tile_size > partial_tile_size) {
      expand_tile_dims_indices[dim] = num_expand_dims++;
      expand_tile_sizes.emplace_back(target_tile_size / partial_tile_size);
    }
  }

  const std::vector<int64_t> shape_dims(
      target_sharding.tile_assignment().dimensions().begin(),
      target_sharding.tile_assignment().dimensions().begin() + rank);
  if (hlo_sharding_util::IsSubTilingOrEqualSharding(
          ShapeUtil::MakeShape(F32, shape_dims), target_sharding,
          partial_sharding)) {
    return target_sharding;
  }

  // Now that target_sharding is not a subtiling of partial_sharding, we
  // decompose partial_sharding on the last tile dimension (replicated one) and
  // move the decomposed tile dimensions to the expanded tile dimensions.
  std::vector<int64_t> reshape_dimensions(
      partial_sharding.tile_assignment().dimensions().begin(),
      partial_sharding.tile_assignment().dimensions().begin() + rank);
  reshape_dimensions.insert(reshape_dimensions.end(), expand_tile_sizes.begin(),
                            expand_tile_sizes.end());

  std::vector<int> perm;
  perm.reserve(rank + expand_tile_sizes.size());
  for (int64_t dim = 0; dim < rank; dim++) {
    perm.push_back(dim);
    if (expand_tile_dims_indices[dim] > -1) {
      perm.emplace_back(expand_tile_dims_indices[dim] + rank);
    }
  }

  if (target_sharding.ReplicateOnLastTileDim()) {
    reshape_dimensions.push_back(
        target_sharding.tile_assignment().dimensions().back());
    perm.push_back(reshape_dimensions.size() - 1);
  }

  auto transpose_tile_assignment =
      partial_sharding.tile_assignment()
          .Reshape(reshape_dimensions)
          .Transpose(perm)
          .Reshape(target_sharding.tile_assignment().dimensions());

  return target_sharding.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(transpose_tile_assignment)
             : HloSharding::Tile(transpose_tile_assignment);
}

std::optional<HloInstruction*> TileToPartialReplicateHaloExchange(
    HloInstruction* hlo, const Shape& base_shape,
    const HloSharding& src_sharding, const HloSharding& dst_sharding,
    const std::vector<int64_t>& replicate_dims,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b) {
  // Source is tile sharding.
  auto padded_src_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, src_sharding);
  // Target is partial replicate.
  auto padded_dst_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, dst_sharding);
  if (ShapeUtil::Compatible(padded_dst_shape, hlo->shape())) {
    return hlo;
  }

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(src_sharding, partition_id, b);

  auto result = hlo;
  auto hlo_shape = hlo->shape();
  for (auto dim : replicate_dims) {
    int64_t src_shard_count = src_sharding.tile_assignment().dim(dim);
    int64_t dst_shard_count = dst_sharding.tile_assignment().dim(dim);
    int64_t src_per_dst_shard_size =
        padded_src_shape.dimensions(dim) / dst_shard_count;
    // Calculate per shard size using the sharding to compare if dst_sharding
    // needs more padding at the end.
    int64_t dst_per_shard_size =
        padded_dst_shape.dimensions(dim) / dst_shard_count;

    // If src per shard doesn't have redundant data.
    if (src_per_dst_shard_size <= dst_per_shard_size || dst_shard_count == 1) {
      continue;
    }

    // If src_per_shard * replicate_factor > dst_per_shard , need to
    // re-distribute the data between each shard using collective permute. For
    // example, if dimension size is 6 and shard 4 ways in the src but needs to
    // shard 2 ways in the dst. 4 way sharding has 2 element in each shard,
    // while 2 way sharding has 3 elements, the last element in the first shard
    // will be sliced out. re-distribution is needed.
    int64_t replicate_factor = src_shard_count / dst_shard_count;

    // 1. Calculate left_halo size.
    // left-halo size is
    //   (src_per_dst_shard_size - dst_per_shard_size) *
    //   floor(i / replicate_factor)
    OffsetCalculation left_halo_size_function = OffsetCalculation(
        HloOpcode::kMultiply,
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            0, src_per_dst_shard_size - dst_per_shard_size, 1)),
        OffsetCalculation(
            MultiplyAddDivideOffsetCalculation(1, 0, replicate_factor)));

    // 2. Calculate right_halo size.
    // right-halo size is -left_halo_size_function;
    OffsetCalculation right_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(0, 0, 1)) -
        left_halo_size_function;

    // 3. Halo exchange.
    result = ExchangeHaloCompact(result, base_shape, left_halo_size_function,
                                 right_halo_size_function, nullptr, dim,
                                 src_sharding, partition_ordinals[dim],
                                 collective_ops_creator, next_channel_id, b);
  }
  return result;
}

std::optional<HloInstruction*> PadFromPartialReplicateShape(
    HloInstruction* hlo, const Shape& base_shape,
    const HloSharding& src_sharding, const HloSharding& dst_sharding,
    const std::vector<int64_t>& expand_tile_dims,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, HloInstruction* partition_id, SpmdBuilder* b) {
  auto padded_src_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, src_sharding);
  auto padded_dst_shape =
      GetPaddedShapeForUnevenPartitioning(base_shape, dst_sharding);
  if (ShapeUtil::Compatible(padded_dst_shape, hlo->shape())) {
    return hlo;
  }

  auto partition_ordinals =
      MakeTiledPartitionOrdinals(src_sharding, partition_id, b);

  HloInstruction* result = hlo;
  auto zero = b->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(hlo->shape().element_type())));
  std::vector<int64_t> expand_dims_without_halo_exchange;
  // Pad the dimensions needs halo exchange and record the padded dims that
  // won't need halo exchange.
  for (auto dim : expand_tile_dims) {
    int64_t src_shard_count = src_sharding.tile_assignment().dim(dim);
    int64_t src_per_shard_size =
        padded_src_shape.dimensions(dim) / src_shard_count;
    // Calculate per shard size using the sharding to compare if dst_sharding
    // needs more padding at the end.
    int64_t dst_per_shard_size =
        padded_dst_shape.dimensions(dim) / src_shard_count;

    // If dst_sharding doesn't need more padding at the end.
    if (src_per_shard_size >= dst_per_shard_size) {
      continue;
    }
    // If src sharding at this dimension is not partitioned, simply pad to
    // the desired shape.
    if (src_shard_count == 1) {
      expand_dims_without_halo_exchange.push_back(dim);
      continue;
    }

    // If dst_padding needs more padding at the end, need to re-distribute the
    // data between each shard using collective permute.
    // For example, if dimension size is 6 and shard 2 ways in the src but
    // needs to shard 4 ways in the dst. 4 ways needs padding 2 0s at the end
    // and has 2 elements at each shard, while 2 way sharding has 3 elements
    // in each shard, re-distribution is needed.
    //
    // 1. Calculate left_halo size.
    // left-halo size is -(D * i - S * i) = (S - D) * i
    OffsetCalculation left_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            src_per_shard_size - dst_per_shard_size, 0, 1));
    // 2. Calculate right_halo size.
    // right-halo size is D * (i + 1) - S * (i + 1) = (D - S) * i + (D - S)
    OffsetCalculation right_halo_size_function =
        OffsetCalculation(MultiplyAddDivideOffsetCalculation(
            dst_per_shard_size - src_per_shard_size,
            dst_per_shard_size - src_per_shard_size, 1));
    // 3. Halo exchange.
    result = ExchangeHaloCompact(result, base_shape, left_halo_size_function,
                                 right_halo_size_function, nullptr, dim,
                                 src_sharding, partition_ordinals[dim],
                                 collective_ops_creator, next_channel_id, b);
  }

  // Pad other dimensions that won't need halo exchange with a single pad.
  if (!expand_dims_without_halo_exchange.empty()) {
    std::vector<int64_t> zero_padding(result->shape().dimensions().size());
    PaddingConfig pad_config = window_util::MakeSymmetricPadding(zero_padding);

    auto padded_shape = result->shape();
    for (auto dim : expand_dims_without_halo_exchange) {
      pad_config.mutable_dimensions(dim)->set_edge_padding_low(0);
      pad_config.mutable_dimensions(dim)->set_edge_padding_high(
          padded_dst_shape.dimensions(dim) - padded_src_shape.dimensions(dim));
      padded_shape.set_dimensions(dim, result->shape().dimensions(dim) +
                                           padded_dst_shape.dimensions(dim) -
                                           padded_src_shape.dimensions(dim));
    }
    result = b->AddInstruction(
        HloInstruction::CreatePad(padded_shape, result, zero, pad_config));
  }

  return result;
}

std::optional<int64_t> UniqueTiledDim(const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    return std::nullopt;
  }
  int64_t dim = -1;
  int64_t rank = sharding.ReplicateOnLastTileDim()
                     ? sharding.tile_assignment().num_dimensions() - 1
                     : sharding.tile_assignment().num_dimensions();
  for (int64_t i = 0; i < rank; ++i) {
    if (sharding.tile_assignment().dim(i) > 1) {
      if (dim != -1) {
        return std::nullopt;
      }
      dim = i;
    }
  }
  CHECK_NE(dim, -1);
  return dim;
}

MultiplyAddDivideOffsetCalculation::MultiplyAddDivideOffsetCalculation(
    int64_t multiplier, int64_t offset, int64_t divisor)
    : multiplier_(multiplier), offset_(offset), divisor_(divisor) {
  CHECK_GT(divisor_, 0);
  Simplify();
}

OffsetCalculation MultiplyAddDivideOffsetCalculation::operator-(
    const MultiplyAddDivideOffsetCalculation& other) const {
  if (divisor_ == 1 && other.divisor_ == 1) {
    return OffsetCalculation(MultiplyAddDivideOffsetCalculation(
        multiplier_ - other.multiplier_, offset_ - other.offset_, 1));
  }
  return OffsetCalculation(HloOpcode::kSubtract, *this, other);
}

OffsetCalculation MultiplyAddDivideOffsetCalculation::operator+(
    const MultiplyAddDivideOffsetCalculation& other) const {
  if (divisor_ == 1 && other.divisor_ == 1) {
    return OffsetCalculation(MultiplyAddDivideOffsetCalculation(
        multiplier_ + other.multiplier_, offset_ + other.offset_, 1));
  }
  return OffsetCalculation(HloOpcode::kAdd, *this, other);
}

void MultiplyAddDivideOffsetCalculation::Simplify() {
  // We could simplify the calculation when multiplier is a multiple of
  // divisor_. However, when offset_ is not a multiple of divisor_, we must
  // make sure that offset_ and multiplier_ are both non-negative or both
  // non-positive. E.g., (3 * i  - 1) / 3 is not equivalent to i or i - 1.
  if (divisor_ != 1 && multiplier_ % divisor_ == 0 &&
      (offset_ % divisor_ == 0 || offset_ * multiplier_ > 0)) {
    multiplier_ /= divisor_;
    offset_ /= divisor_;
    divisor_ = 1;
  }
}

int64_t MultiplyAddDivideOffsetCalculation::Calculate(
    int64_t shard_ordinal) const {
  return (shard_ordinal * multiplier_ + offset_) / divisor_;
}

HloInstruction* MultiplyAddDivideOffsetCalculation::Calculate(
    HloInstruction* shard_ordinal, SpmdBuilder* b) const {
  auto scalar_shape = ShapeUtil::MakeShape(S32, {});
  if (multiplier_ == 0) {
    return b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(offset_ / divisor_)));
  }
  HloInstruction* result = shard_ordinal;
  if (multiplier_ != 1) {
    result = b->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMultiply, shard_ordinal,
        b->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(multiplier_)))));
  }
  if (offset_ != 0) {
    auto offset = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(offset_)));
    result = b->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, result, offset));
  }
  if (divisor_ != 1) {
    auto divisor = b->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(divisor_)));
    result = b->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kDivide, result, divisor));
  }
  return result;
}

int64_t MultiplyAddDivideOffsetCalculation::MaxInRange(
    int64_t start_ordinal, int64_t limit_ordinal) const {
  int64_t max = Calculate(start_ordinal);
  for (int64_t i = start_ordinal + 1; i < limit_ordinal; ++i) {
    max = std::max(max, Calculate(i));
  }
  return max;
}

OffsetCalculation& OffsetCalculation::operator=(
    const OffsetCalculation& other) {
  opcode_ = other.opcode_;
  copy_from_ = other.copy_from_;
  if (opcode_ != HloOpcode::kCopy) {
    lhs_ = std::make_unique<OffsetCalculation>(*other.lhs_);
    rhs_ = std::make_unique<OffsetCalculation>(*other.rhs_);
  }
  return *this;
}

bool OffsetCalculation::IsConstant() const {
  if (opcode_ == HloOpcode::kCopy) {
    return copy_from_.IsConstant();
  }
  if (opcode_ == HloOpcode::kSubtract && *lhs_ == *rhs_) {
    return true;
  }
  return lhs_->IsConstant() && rhs_->IsConstant();
}

OffsetCalculation OffsetCalculation::operator-(
    const OffsetCalculation& other) const {
  if (opcode_ == HloOpcode::kCopy && other.opcode_ == HloOpcode::kCopy) {
    return copy_from_ - other.copy_from_;
  }
  return OffsetCalculation(HloOpcode::kSubtract, *this, other);
}

OffsetCalculation OffsetCalculation::operator+(
    const OffsetCalculation& other) const {
  if (opcode_ == HloOpcode::kCopy && other.opcode_ == HloOpcode::kCopy) {
    return copy_from_ + other.copy_from_;
  }
  return OffsetCalculation(HloOpcode::kAdd, *this, other);
}

bool OffsetCalculation::operator==(const OffsetCalculation& other) const {
  if (opcode_ != other.opcode_) {
    return false;
  }
  if (opcode_ == HloOpcode::kCopy) {
    return copy_from_ == other.copy_from_;
  }
  return *lhs_ == *other.lhs_ && *rhs_ == *other.rhs_;
}

int64_t OffsetCalculation::Calculate(int64_t shard_ordinal) const {
  switch (opcode_) {
    case HloOpcode::kAdd:
      return lhs_->Calculate(shard_ordinal) + rhs_->Calculate(shard_ordinal);
    case HloOpcode::kCopy:
      return copy_from_.Calculate(shard_ordinal);
    case HloOpcode::kSubtract:
      return lhs_->Calculate(shard_ordinal) - rhs_->Calculate(shard_ordinal);
    case HloOpcode::kMultiply:
      return lhs_->Calculate(shard_ordinal) * rhs_->Calculate(shard_ordinal);
    default:
      LOG(FATAL) << "Should not happen";
  }
}

HloInstruction* OffsetCalculation::Calculate(HloInstruction* shard_ordinal,
                                             SpmdBuilder* b) const {
  if (opcode_ == HloOpcode::kCopy) {
    return copy_from_.Calculate(shard_ordinal, b);
  }
  auto lhs = lhs_->Calculate(shard_ordinal, b);
  auto rhs = rhs_->Calculate(shard_ordinal, b);
  return b->AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), opcode_, lhs, rhs));
}

int64_t OffsetCalculation::MaxInRange(int64_t start_ordinal,
                                      int64_t limit_ordinal) const {
  if (IsConstant()) {
    return Calculate(start_ordinal);
  }
  if (opcode_ == HloOpcode::kCopy) {
    return std::max(Calculate(start_ordinal), Calculate(limit_ordinal - 1));
  }
  int64_t max = Calculate(start_ordinal);
  for (int64_t i = start_ordinal + 1; i < limit_ordinal; ++i) {
    max = std::max(max, Calculate(i));
  }
  return max;
}

std::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo, const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function, int64_t dim,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b) {
  int64_t input_shard_size = hlo->shape().dimensions(dim);
  int64_t shard_count = target.tile_assignment().dim(dim);

  std::vector<HloInstruction*> concat_pieces;

  int64_t max_left_halo_size =
      left_halo_size_function.MaxInRange(1, shard_count);
  int64_t max_right_halo_size =
      right_halo_size_function.MaxInRange(0, shard_count - 1);
  if (max_left_halo_size + max_right_halo_size + input_shard_size >=
          input_shard_size * shard_count &&
      (max_left_halo_size > input_shard_size ||
       max_right_halo_size > input_shard_size)) {
    return std::nullopt;
  }
  // Since max halo sizes could be negative, we only need to include data within
  // certain bounds. Useful region is [left_bound, right_bound).
  const int64_t left_bound =
      -left_halo_size_function.MaxInRange(0, shard_count);
  const int64_t right_bound =
      input_shard_size + right_halo_size_function.MaxInRange(0, shard_count);
  if (left_bound >= right_bound) {
    return std::nullopt;
  }
  // Left halo.
  for (int64_t i = CeilOfRatio(max_left_halo_size, input_shard_size) - 1;
       i >= 0 && (-i - 1) * input_shard_size < right_bound; --i) {
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
    target.tile_assignment().Each(
        [&](absl::Span<const int64_t> indices, int64_t device) {
          if (indices[dim] > i) {
            std::vector<int64_t> source_indices(indices.begin(), indices.end());
            source_indices[dim] -= i + 1;
            source_target_pairs.emplace_back(
                target.tile_assignment()(source_indices), device);
          }
        });
    int64_t halo_size_including_skips =
        std::min(max_left_halo_size - input_shard_size * i, input_shard_size);
    int64_t halo_right_skips =
        std::max<int64_t>(-i * input_shard_size - right_bound, 0);
    int64_t halo_size = halo_size_including_skips - halo_right_skips;
    auto halo_shape = hlo->shape();
    auto source_halo_slice = hlo;
    if (halo_size != hlo->shape().dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64_t> halo_start_indices(halo_shape.dimensions().size(),
                                              0);
      halo_start_indices[dim] =
          hlo->shape().dimensions(dim) - halo_size_including_skips;
      std::vector<int64_t> halo_limit_indices(hlo->shape().dimensions().begin(),
                                              hlo->shape().dimensions().end());
      halo_limit_indices[dim] -= halo_right_skips;
      std::vector<int64_t> halo_slice_strides(halo_shape.dimensions().size(),
                                              1);
      source_halo_slice = b->AddInstruction(
          HloInstruction::CreateSlice(halo_shape, hlo, halo_start_indices,
                                      halo_limit_indices, halo_slice_strides));
    }
    auto left_halo =
        collective_ops_creator.create_cross_partition_collective_permute(
            b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    concat_pieces.push_back(left_halo);
  }

  if (left_bound < input_shard_size && right_bound > 0) {
    int64_t self_start = std::max<int64_t>(0, left_bound);
    int64_t self_limit = std::min<int64_t>(input_shard_size, right_bound);
    if (self_start == 0 && self_limit == input_shard_size) {
      concat_pieces.push_back(hlo);
    } else {
      auto self_shape = hlo->shape();
      self_shape.set_dimensions(dim, self_limit - self_start);
      std::vector<int64_t> start_indices(self_shape.dimensions().size(), 0);
      start_indices[dim] = self_start;
      std::vector<int64_t> limit_indices(hlo->shape().dimensions().begin(),
                                         hlo->shape().dimensions().end());
      limit_indices[dim] = self_limit;
      std::vector<int64_t> slice_strides(self_shape.dimensions().size(), 1);
      concat_pieces.push_back(b->AddInstruction(HloInstruction::CreateSlice(
          self_shape, hlo, start_indices, limit_indices, slice_strides)));
    }
  }

  int64_t skipped_right_halos =
      std::min<int64_t>(std::max<int64_t>(left_bound - input_shard_size, 0),
                        std::max<int64_t>(max_right_halo_size, 0)) /
      input_shard_size;
  // Right halo.
  for (int64_t i = skipped_right_halos;
       i < CeilOfRatio(max_right_halo_size, input_shard_size); ++i) {
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
    target.tile_assignment().Each(
        [&](absl::Span<const int64_t> indices, int64_t device) {
          if (indices[dim] > i) {
            std::vector<int64_t> target_indices(indices.begin(), indices.end());
            target_indices[dim] -= i + 1;
            source_target_pairs.emplace_back(
                device, target.tile_assignment()(target_indices));
          }
        });
    int64_t halo_size_including_skips =
        std::min(max_right_halo_size - input_shard_size * i, input_shard_size);
    int64_t halo_left_skips =
        std::max<int64_t>(left_bound - (i + 1) * input_shard_size, 0);
    int64_t halo_size = halo_size_including_skips - halo_left_skips;
    auto halo_shape = hlo->shape();
    HloInstruction* source_halo_slice = hlo;
    if (halo_size != halo_shape.dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64_t> halo_start_indices(halo_shape.dimensions().size(),
                                              0);
      halo_start_indices[dim] = halo_left_skips;
      std::vector<int64_t> halo_limit_indices(halo_shape.dimensions().begin(),
                                              halo_shape.dimensions().end());
      halo_limit_indices[dim] += halo_left_skips;
      std::vector<int64_t> halo_slice_strides(halo_shape.dimensions().size(),
                                              1);
      source_halo_slice = b->AddInstruction(
          HloInstruction::CreateSlice(halo_shape, hlo, halo_start_indices,
                                      halo_limit_indices, halo_slice_strides));
    }
    auto right_halo =
        collective_ops_creator.create_cross_partition_collective_permute(
            b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    concat_pieces.push_back(right_halo);
  }

  auto concat = concat_pieces[0];
  // Concat with halos/padding.
  if (concat_pieces.size() > 1) {
    auto concat_shape = hlo->shape();
    int64_t concat_dim_size = 0;
    for (auto piece : concat_pieces) {
      concat_dim_size += piece->shape().dimensions(dim);
    }
    concat_shape.set_dimensions(dim, concat_dim_size);
    concat = b->AddInstruction(
        HloInstruction::CreateConcatenate(concat_shape, concat_pieces, dim));
  }

  return concat;
}

HloInstruction* ExchangeHaloCompact(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    HloInstruction* pad_value, int64_t dim, const HloSharding& sharding,
    HloInstruction* shard_ordinal,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b) {
  int64_t input_shard_size = hlo->shape().dimensions(dim);
  int64_t shard_count = sharding.tile_assignment().dim(dim);
  auto grouped =
      hlo_sharding_util::GroupShardingOnAllDimsExcept(sharding, {dim});
  auto g_creator = GetPerGroupCollectiveOpsCreator(collective_ops_creator,
                                                   grouped.device_groups);
  const bool ignore_pad_vale = pad_value == nullptr;
  if (ignore_pad_vale) {
    pad_value = CreateR0WithType(hlo->shape().element_type(), 0, b);
  }
  struct Halo {
    int64_t my_index;
    int64_t start;
    int64_t limit;
    int64_t cp_idx;
    int64_t halo_offset;
    int64_t halo_at_shard;
  };

  // Find a list of halos for each shard. Each halo can be a real collective-
  // permute, a slice of the self tensor, or all padding.
  std::vector<std::vector<Halo>> halos(shard_count);
  constexpr int64_t kPaddingShard = -2;
  constexpr int64_t kSelfShard = -1;
  int64_t max_window_size = 0;
  for (int64_t i = 0; i < shard_count; ++i) {
    const int64_t start =
        i * input_shard_size - left_halo_size_function.Calculate(i);
    int64_t next_start = start;
    const int64_t limit =
        (i + 1) * input_shard_size + right_halo_size_function.Calculate(i);
    max_window_size = std::max(max_window_size, limit - start);
    while (next_start < limit) {
      Halo& halo = halos[i].emplace_back();
      halo.my_index = i;
      halo.halo_offset = next_start - start;
      halo.start = next_start % input_shard_size;
      if (halo.start < 0) {
        halo.start += input_shard_size;
      }
      int64_t size = limit - next_start;
      if (next_start < 0 || next_start >= base_shape.dimensions(dim)) {
        if (next_start < 0) {
          // Left padding bounded by offset zero.
          size = std::min(size, 0 - next_start);
        }
        VLOG(3) << "Halo for shard i " << i << ": pad, size " << size;
        halo.limit = halo.start + size;
        halo.cp_idx = kPaddingShard;
        next_start += size;
        continue;
      }
      size = std::min(input_shard_size - halo.start, size);
      halo.limit = halo.start + size;
      int64_t shard = next_start / input_shard_size;
      halo.halo_at_shard = shard;
      // To be assigned.
      halo.cp_idx = kSelfShard;
      next_start += size;
      VLOG(3) << "Halo for shard i " << i << ": shard " << shard << ", size "
              << size << ", start " << halo.start;
    }
  }
  // Element at index i: dst halos for src core i, where each halo is
  // represented as a pair (shard_ordinal, offset in halos[shard_ordinal]).
  std::vector<std::vector<std::pair<int64_t, int64_t>>> src_to_dst(shard_count);
  {
    // At each offset, unless all shards have padding data, we limit the size of
    // the paddings to input_shard_size so that they don't force to pad the
    // non-padding buffers too much.
    std::vector<std::vector<Halo>> halos2(shard_count);
    std::vector<int64_t> next_halo_idx(halos2.size(), 0);
    while (true) {
      bool all_padding = true;
      bool empty = true;
      for (int64_t i = 0; i < halos.size(); ++i) {
        if (next_halo_idx[i] >= halos[i].size()) {
          continue;
        }
        if (halos[i][next_halo_idx[i]].cp_idx != kPaddingShard) {
          all_padding = false;
        }
        empty = false;
      }
      if (empty) {
        break;
      }
      for (int64_t i = 0; i < halos.size(); ++i) {
        if (next_halo_idx[i] >= halos[i].size()) {
          continue;
        }
        Halo& h = halos[i][next_halo_idx[i]];
        halos2[i].push_back(h);
        Halo& new_h = halos2[i].back();
        if (!all_padding && h.cp_idx == kPaddingShard &&
            h.limit > input_shard_size) {
          new_h.limit = input_shard_size;
          h.start = 0;
          h.limit -= input_shard_size;
          VLOG(3) << "Split padding halo for shard i " << i << ": size "
                  << new_h.limit - new_h.start;
        } else {
          next_halo_idx[i] += 1;
        }
        if (h.cp_idx != kPaddingShard && h.halo_at_shard != i) {
          src_to_dst[h.halo_at_shard].emplace_back(i, halos2[i].size() - 1);
        }
      }
    }
    halos = std::move(halos2);
  }
  // Sort halos that are from the same src according to halo_offset, so that
  // they are more likely to have similar characteristics.
  for (int64_t i = 0; i < src_to_dst.size(); ++i) {
    absl::c_stable_sort(src_to_dst[i],
                        [&](const std::pair<int64_t, int64_t>& a,
                            const std::pair<int64_t, int64_t>& b) {
                          return halos[a.first][a.second].halo_offset <
                                 halos[b.first][b.second].halo_offset;
                        });
  }

  // Build collective permutes with distinct src/dst values.
  // A list of [<cp, offset_in_shard>].
  std::vector<std::pair<HloInstruction*, int64_t>> cps;
  std::vector<int64_t> next_dst_idx(src_to_dst.size(), 0);
  while (true) {
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
    std::vector<bool> dst_seen(shard_count, false);
    int64_t start = input_shard_size;
    int64_t limit = 0;
    for (int64_t i = 0; i < src_to_dst.size(); ++i) {
      if (src_to_dst[i].size() <= next_dst_idx[i]) {
        continue;
      }
      const auto& halo_idx = src_to_dst[i][next_dst_idx[i]];
      Halo& halo = halos[halo_idx.first][halo_idx.second];
      // Delay this pair if there is dst collision, or it increases the range by
      // too much (we just test if we are combining small left halos with small
      // right halos).
      if (!source_target_pairs.empty() &&
          (dst_seen[halo.my_index] ||
           (start > halo.limit && limit == input_shard_size &&
            halo.start == 0) ||
           (limit < halo.start && start == 0 &&
            halo.limit == input_shard_size))) {
        continue;
      }
      halo.cp_idx = cps.size();
      dst_seen[halo.my_index] = true;
      source_target_pairs.emplace_back(i, halo.my_index);
      start = std::min(start, halo.start);
      limit = std::max(limit, halo.limit);
      next_dst_idx[i] += 1;
    }
    if (source_target_pairs.empty()) {
      break;
    }
    CHECK_LT(start, limit);
    const int64_t halo_size = limit - start;
    Shape halo_shape = hlo->shape();
    HloInstruction* source_halo_slice = hlo;
    if (halo_size != hlo->shape().dimensions(dim)) {
      halo_shape.set_dimensions(dim, halo_size);
      std::vector<int64_t> halo_start_indices(halo_shape.dimensions().size(),
                                              0);
      halo_start_indices[dim] = start;
      std::vector<int64_t> halo_limit_indices(hlo->shape().dimensions().begin(),
                                              hlo->shape().dimensions().end());
      halo_limit_indices[dim] = limit;
      std::vector<int64_t> halo_slice_strides(halo_shape.dimensions().size(),
                                              1);
      source_halo_slice = b->AddInstruction(
          HloInstruction::CreateSlice(halo_shape, hlo, halo_start_indices,
                                      halo_limit_indices, halo_slice_strides));
    }
    HloInstruction* cp = g_creator.create_cross_partition_collective_permute(
        b, source_halo_slice, source_target_pairs, (*next_channel_id)++);
    VLOG(3) << "Halo collective-permute created: " << cp->ToString();
    cps.emplace_back(cp, start);
  }

  // Build pieces to concat in order. Each piece may be a select from different
  // collectiver permutes/self/padding, because different shards may not share
  // the same collective permute at each offset.
  std::vector<HloInstruction*> concat_pieces;
  Shape concat_shape = hlo->shape();
  concat_shape.set_dimensions(dim, 0);
  int64_t self_piece_start = input_shard_size;
  bool all_padding = true;
  for (int64_t current_halo_idx = 0; true; ++current_halo_idx) {
    int64_t max_size = 0;
    constexpr int64_t kUnseen = -5;
    std::vector<int64_t> cp_index(halos.size(), kUnseen);
    int64_t min_self_start = input_shard_size;
    int64_t max_self_limit = 0;
    for (int64_t i = 0; i < halos.size(); ++i) {
      if (current_halo_idx >= halos[i].size()) {
        continue;
      }
      const Halo& halo = halos[i][current_halo_idx];
      cp_index[i] = halo.cp_idx;
      if (halo.cp_idx >= 0) {
        max_size =
            std::max(max_size, cps[cp_index[i]].first->shape().dimensions(dim));
      } else if (halo.cp_idx == kSelfShard) {
        // Use the full input to allow all possible slicing offsets.
        min_self_start = std::min(min_self_start, halo.start);
        max_self_limit = std::max(max_self_limit, halo.limit);
        max_size = std::max(max_size, max_self_limit - min_self_start);
      } else {
        max_size = std::max(max_size, halo.limit - halo.start);
      }
    }
    if (absl::c_all_of(cp_index, [&](int64_t idx) { return idx == kUnseen; })) {
      break;
    }
    // Adjust self start for max_size.
    min_self_start -= max_size - (max_self_limit - min_self_start);
    min_self_start = std::max<int64_t>(min_self_start, 0);
    if (current_halo_idx == 0) {
      self_piece_start = min_self_start;
    }
    concat_shape.set_dimensions(dim, max_size + concat_shape.dimensions(dim));
    Shape piece_shape = hlo->shape();
    piece_shape.set_dimensions(dim, max_size);
    // Create the concat piece, which can be a select over different buffers.
    // For each unique buffer we cache its value, then we use a table lookup to
    // find which buffer each device uses.
    HloInstruction* padding = b->AddInstruction(
        HloInstruction::CreateBroadcast(piece_shape, pad_value, {}));
    std::vector<HloInstruction*> unique_pieces;
    std::vector<int64_t> slices_cache(cps.size() + 2, kUnseen);
    std::vector<int32_t> piece_index(halos.size());
    for (int64_t i = 0; i < halos.size(); ++i) {
      HloInstruction* piece;
      int64_t cache_idx = cp_index[i];
      if (cp_index[i] >= 0) {
        all_padding = false;
        piece = cps[cp_index[i]].first;
      } else if (cp_index[i] == kSelfShard) {
        if (hlo->shape().dimensions(dim) == max_size) {
          piece = hlo;
        } else {
          std::vector<int64_t> starts(piece_shape.dimensions().size(), 0);
          starts[dim] = min_self_start;
          std::vector<int64_t> limits(piece_shape.dimensions().begin(),
                                      piece_shape.dimensions().end());
          std::vector<int64_t> strides(piece_shape.dimensions().size(), 1);
          limits[dim] += min_self_start;
          piece = b->AddInstruction(HloInstruction::CreateSlice(
              piece_shape, hlo, starts, limits, strides));
        }
        cache_idx = cps.size();
        all_padding = false;
      } else {
        piece = padding;
        cache_idx = cps.size() + 1;
      }
      if (slices_cache[cache_idx] != kUnseen) {
        piece_index[i] = slices_cache[cache_idx];
        continue;
      }
      if (piece->shape().dimensions(dim) != max_size) {
        PaddingConfig pc;
        for (int64_t k = 0; k < piece_shape.dimensions().size(); ++k) {
          auto pc_dim = pc.add_dimensions();
          pc_dim->set_interior_padding(0);
          pc_dim->set_edge_padding_low(0);
          pc_dim->set_edge_padding_high(0);
          if (k != dim) {
            continue;
          }
          int64_t padding_size = max_size - piece->shape().dimensions(dim);
          if (concat_pieces.empty()) {
            pc_dim->set_edge_padding_low(padding_size);
          } else {
            pc_dim->set_edge_padding_high(padding_size);
          }
        }
        piece = b->AddInstruction(
            HloInstruction::CreatePad(piece_shape, piece, pad_value, pc));
      }
      piece_index[i] = unique_pieces.size();
      unique_pieces.push_back(piece);
      slices_cache[cache_idx] = piece_index[i];
    }
    // Select a buffer based on the shard ordinal.
    HloInstruction* selector =
        TableLookup<int32_t>(piece_index, S32, shard_ordinal, b);
    int64_t init_piece = 0;
    if (unique_pieces.size() > 1 && unique_pieces[init_piece] == padding) {
      // Init with a non-padding piece.
      init_piece = 1;
    }
    HloInstruction* selected = unique_pieces[init_piece];
    for (int64_t i = init_piece + 1; i < unique_pieces.size(); ++i) {
      if (unique_pieces[i] == padding) {
        // Padding masking will be applied later.
        continue;
      }
      HloInstruction* pred = b->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeScalarShape(PRED), selector,
          CreateR0WithType(S32, i, b), ComparisonDirection::kEq));
      pred = b->AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(PRED, selected->shape().dimensions()), pred,
          {}));
      selected = b->AddInstruction(
          HloInstruction::CreateTernary(selected->shape(), HloOpcode::kSelect,
                                        pred, unique_pieces[i], selected));
    }
    concat_pieces.push_back(selected);
  }
  if (all_padding) {
    concat_shape.set_dimensions(dim, max_window_size);
    return b->AddInstruction(
        HloInstruction::CreateBroadcast(concat_shape, pad_value, {}));
  }
  CHECK_GE(concat_shape.dimensions(dim), max_window_size);
  HloInstruction* concat;
  if (concat_pieces.size() == 1) {
    concat = concat_pieces[0];
  } else {
    concat = b->AddInstruction(
        HloInstruction::CreateConcatenate(concat_shape, concat_pieces, dim));
  }

  // Find extra paddings on the left to slice out.
  std::vector<int32_t> slice_offset(halos.size(), 0);
  std::vector<int32_t> non_padding_starts(halos.size(), 0);
  std::vector<int32_t> non_padding_limits(halos.size(), 0);
  const int64_t first_piece_size = concat_pieces[0]->shape().dimensions(dim);
  int64_t padded_concat_size = concat_shape.dimensions(dim);
  for (int64_t i = 0; i < halos.size(); ++i) {
    if (halos[i].empty()) {
      // All padding.
      continue;
    }
    const Halo& halo = halos[i][0];
    for (int64_t j = 0; j < halos[i].size(); ++j) {
      if (halos[i][j].cp_idx != kPaddingShard) {
        break;
      }
      non_padding_starts[i] += halos[i][j].limit - halos[i][j].start;
    }
    non_padding_limits[i] = left_halo_size_function.Calculate(i) +
                            right_halo_size_function.Calculate(i) +
                            input_shard_size;
    int64_t high_padding = right_halo_size_function.Calculate(i) +
                           input_shard_size * (i + 1) -
                           base_shape.dimensions(dim);
    if (high_padding > 0) {
      non_padding_limits[i] -= high_padding;
    }
    if (halo.cp_idx >= 0) {
      slice_offset[i] = halo.start - cps[halo.cp_idx].second +
                        first_piece_size -
                        cps[halo.cp_idx].first->shape().dimensions(dim);
    } else if (halo.cp_idx == kSelfShard) {
      slice_offset[i] = halo.start - self_piece_start;
    } else {
      slice_offset[i] = first_piece_size - (halo.limit - halo.start);
    }
    padded_concat_size =
        std::max(padded_concat_size, slice_offset[i] + max_window_size);
  }
  if (padded_concat_size > concat_shape.dimensions(dim)) {
    // Need increase the shape size before slicing.
    PaddingConfig pc;
    for (int64_t k = 0; k < concat_shape.dimensions().size(); ++k) {
      auto pc_dim = pc.add_dimensions();
      pc_dim->set_interior_padding(0);
      pc_dim->set_edge_padding_low(0);
      pc_dim->set_edge_padding_high(0);
      if (k != dim) {
        continue;
      }
      pc_dim->set_edge_padding_high(padded_concat_size -
                                    concat_shape.dimensions(dim));
    }
    concat_shape.set_dimensions(dim, padded_concat_size);
    concat = b->AddInstruction(
        HloInstruction::CreatePad(concat_shape, concat, pad_value, pc));
  }
  if (concat_shape.dimensions(dim) > max_window_size) {
    Shape result_shape = concat_shape;
    result_shape.set_dimensions(dim, max_window_size);
    std::vector<HloInstruction*> offsets(result_shape.dimensions().size(),
                                         CreateR0WithType(S32, 0, b));
    offsets[dim] = TableLookup<int32_t>(slice_offset, S32, shard_ordinal, b);
    concat = b->AddInstruction(HloInstruction::CreateDynamicSlice(
        result_shape, concat, offsets, result_shape.dimensions()));
  }
  if (ignore_pad_vale) {
    return concat;
  }
  // Compute the range of non-padding data, then mask the padding with the given
  // pad_value.
  HloInstruction* iota = b->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::ChangeElementType(concat->shape(), S32), dim));
  HloInstruction* valid_limit =
      b->AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::ChangeElementType(concat->shape(), S32),
          TableLookup<int32_t>(non_padding_limits, S32, shard_ordinal, b), {}));
  HloInstruction* mask = b->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::ChangeElementType(concat->shape(), PRED), iota, valid_limit,
      ComparisonDirection::kLt));
  if (absl::c_any_of(non_padding_starts,
                     [](const int32_t s) { return s > 0; })) {
    HloInstruction* valid_start =
        b->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::ChangeElementType(concat->shape(), S32),
            TableLookup<int32_t>(non_padding_starts, S32, shard_ordinal, b),
            {}));
    mask = b->AddInstruction(HloInstruction::CreateBinary(
        mask->shape(), HloOpcode::kAnd, mask,
        b->AddInstruction(HloInstruction::CreateCompare(
            mask->shape(), iota, valid_start, ComparisonDirection::kGe))));
  }
  HloInstruction* padding = b->AddInstruction(
      HloInstruction::CreateBroadcast(concat->shape(), pad_value, {}));
  return b->AddInstruction(HloInstruction::CreateTernary(
      concat->shape(), HloOpcode::kSelect, mask, concat, padding));
}

std::optional<HloInstruction*> ExchangeHalo(
    HloInstruction* hlo,
    std::vector<OffsetCalculation> left_halo_size_functions,
    std::vector<OffsetCalculation> right_halo_size_functions,
    const HloSharding& target,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b) {
  CHECK(left_halo_size_functions.size() == hlo->shape().dimensions().size());
  CHECK(right_halo_size_functions.size() == hlo->shape().dimensions().size());

  HloInstruction* visiting_hlo = hlo;
  for (int dim = 0; dim < hlo->shape().dimensions().size(); ++dim) {
    auto concat = ExchangeHalo(visiting_hlo, left_halo_size_functions[dim],
                               right_halo_size_functions[dim], dim, target,
                               collective_ops_creator, next_channel_id, b);
    if (!concat) {
      return std::nullopt;
    }
    visiting_hlo = *concat;
  }
  return visiting_hlo;
}

std::optional<HloInstruction*> ExchangeHaloAndGetValidData(
    HloInstruction* hlo, const Shape& base_shape,
    const OffsetCalculation& left_halo_size_function,
    const OffsetCalculation& right_halo_size_function,
    int64_t explicit_left_padding_on_full_shape, int64_t padded_full_shape_size,
    int64_t shard_size_with_halo, int64_t dim, const HloSharding& target,
    HloInstruction* offset_on_padded_shape, HloInstruction* pad_value,
    HloInstruction* partition_ordinal,
    const SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, SpmdBuilder* b, bool mask_invalid_region,
    bool force_mask_in_compact) {
  int64_t shard_count = target.tile_assignment().dim(dim);
  if (explicit_left_padding_on_full_shape ==
      left_halo_size_function.Calculate(0)) {
    int64_t max_halo =
        std::max(left_halo_size_function.MaxInRange(0, shard_count),
                 right_halo_size_function.MaxInRange(0, shard_count));
    int64_t max_shard_size =
        hlo->shape().dimensions(dim) +
        (left_halo_size_function + right_halo_size_function)
            .MaxInRange(0, shard_count);
    // TODO(xla): Check if it's safe ot use it in other cases: e.g.
    // explicit_left_padding_on_full_shape != left_halo(0), or
    // max_shard_size != shard_size_with_halo.
    if (max_shard_size == shard_size_with_halo &&
        max_halo > 2 * shard_size_with_halo) {
      if (max_shard_size * 2 >= shard_count * hlo->shape().dimensions(dim)) {
        // Easier to fallback to replication.
        return std::nullopt;
      }
      return ExchangeHaloCompact(
          hlo, base_shape, left_halo_size_function, right_halo_size_function,
          mask_invalid_region || force_mask_in_compact ? pad_value : nullptr,
          dim, target, partition_ordinal, collective_ops_creator,
          next_channel_id, b);
    }
  }
  auto halo_exchange_result =
      ExchangeHalo(hlo, left_halo_size_function, right_halo_size_function, dim,
                   target, collective_ops_creator, next_channel_id, b);
  if (!halo_exchange_result) {
    return std::nullopt;
  }
  auto concat = *halo_exchange_result;
  int64_t max_left_halo_size =
      left_halo_size_function.MaxInRange(1, shard_count);

  // Now we determine if we need extra padding after the concat.
  //
  // The max of halo size or the first shard's explicit left padding.
  int64_t max_left_halo_or_padding_size =
      std::max(max_left_halo_size, explicit_left_padding_on_full_shape);
  // The calculation that returns the dynamic slice index for a shard on the
  // padded concat, which is the difference between
  // max_left_halo_or_padding_size and its left halo size.
  auto start_offset_on_padded_concat_calculation =
      OffsetCalculation(MultiplyAddDivideOffsetCalculation(
          0, max_left_halo_or_padding_size, 1)) -
      left_halo_size_function;

  // See if we need to pad the concat before dynamic slice.
  int64_t extra_left_padding =
      std::max(int64_t{0}, max_left_halo_or_padding_size -
                               std::max(int64_t{0}, max_left_halo_size));
  int64_t extra_right_padding =
      start_offset_on_padded_concat_calculation.MaxInRange(0, shard_count) +
      shard_size_with_halo - concat->shape().dimensions(dim) -
      extra_left_padding;
  extra_right_padding = std::max(int64_t{0}, extra_right_padding);
  if (extra_left_padding > 0 || extra_right_padding > 0) {
    PaddingConfig padding_config;
    auto padded_concat_shape = concat->shape();
    for (int64_t i = 0; i < base_shape.dimensions().size(); ++i) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_interior_padding(0);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_edge_padding_high(0);
      if (i != dim) {
        continue;
      }
      padding_config_dim->set_edge_padding_low(extra_left_padding);
      padding_config_dim->set_edge_padding_high(extra_right_padding);
      padded_concat_shape.set_dimensions(dim, concat->shape().dimensions(dim) +
                                                  extra_left_padding +
                                                  extra_right_padding);
    }
    concat = b->AddInstruction(HloInstruction::CreatePad(
        padded_concat_shape, concat, pad_value, padding_config));
  }

  auto valid_slice = concat;
  if (shard_size_with_halo != concat->shape().dimensions(dim)) {
    // Concat is bigger than the shard shape, so we need a dynamic slice.
    CHECK_LT(shard_size_with_halo, concat->shape().dimensions(dim));
    auto slice_shape = concat->shape();
    slice_shape.set_dimensions(dim, shard_size_with_halo);

    if (left_halo_size_function.IsConstant() &&
        left_halo_size_function.Calculate(0) ==
            explicit_left_padding_on_full_shape) {
      std::vector<int64_t> start_indices(slice_shape.dimensions().size(), 0);
      std::vector<int64_t> strides(slice_shape.dimensions().size(), 1);
      valid_slice = b->AddInstruction(
          HloInstruction::CreateSlice(slice_shape, concat, start_indices,
                                      slice_shape.dimensions(), strides));
    } else {
      auto zero = b->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
      std::vector<HloInstruction*> slice_offsets(base_shape.dimensions().size(),
                                                 zero);
      slice_offsets[dim] = start_offset_on_padded_concat_calculation.Calculate(
          partition_ordinal, b);
      valid_slice = b->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape, concat, slice_offsets, slice_shape.dimensions()));
    }
  }

  if (!mask_invalid_region) {
    return valid_slice;
  }

  int64_t total_right_padding = padded_full_shape_size -
                                base_shape.dimensions(dim) -
                                explicit_left_padding_on_full_shape;
  // Mask off garbage data due to uneven partition or low/high padding.
  if (explicit_left_padding_on_full_shape > 0 || total_right_padding > 0) {
    auto index_shape = ShapeUtil::ChangeElementType(valid_slice->shape(), S32);
    auto iota = b->AddInstruction(HloInstruction::CreateIota(index_shape, dim));
    auto broadcast_start_index_in_padded_shape =
        b->AddInstruction(HloInstruction::CreateBroadcast(
            index_shape, offset_on_padded_shape, {}));
    auto index_in_padded_shape = b->AddInstruction(
        HloInstruction::CreateBinary(index_shape, HloOpcode::kAdd, iota,
                                     broadcast_start_index_in_padded_shape));
    auto mask_shape = ShapeUtil::ChangeElementType(index_shape, PRED);
    std::vector<HloInstruction*> predicates;
    if (explicit_left_padding_on_full_shape > 0) {
      auto valid_index_start =
          b->AddInstruction(HloInstruction::CreateBroadcast(
              index_shape,
              b->AddInstruction(
                  HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                      explicit_left_padding_on_full_shape))),
              {}));
      predicates.push_back(b->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, index_in_padded_shape, valid_index_start,
          ComparisonDirection::kGe)));
    }
    if (total_right_padding > 0) {
      auto valid_index_limit =
          b->AddInstruction(HloInstruction::CreateBroadcast(
              index_shape,
              b->AddInstruction(
                  HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                      base_shape.dimensions(dim) +
                      explicit_left_padding_on_full_shape))),
              {}));
      predicates.push_back(b->AddInstruction(HloInstruction::CreateCompare(
          mask_shape, index_in_padded_shape, valid_index_limit,
          ComparisonDirection::kLt)));
    }
    CHECK(!predicates.empty());
    auto is_valid =
        predicates.size() == 2
            ? b->AddInstruction(HloInstruction::CreateBinary(
                  mask_shape, HloOpcode::kAnd, predicates[0], predicates[1]))
            : predicates[0];
    if (pad_value->shape().element_type() !=
        valid_slice->shape().element_type()) {
      pad_value = b->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeShape(valid_slice->shape().element_type(),
                               pad_value->shape().dimensions()),
          pad_value));
    }
    auto masking_value = b->AddInstruction(
        HloInstruction::CreateBroadcast(valid_slice->shape(), pad_value, {}));
    valid_slice = b->AddInstruction(
        HloInstruction::CreateTernary(valid_slice->shape(), HloOpcode::kSelect,
                                      is_valid, valid_slice, masking_value));
  }
  return valid_slice;
}

HloInstruction* HaloExchangeToPadOnLeft(PartitionedHlo& original,
                                        absl::Span<const int64_t> dims) {
  if (original.sharding().IsTileMaximal()) {
    return original.hlo();
  }
  // Create a window config to halo exchange for unevenly partitioned reverse
  // dimensions.
  Window window;
  for (int64_t i = 0; i < original.base_shape().dimensions().size(); ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_window_reversal(false);
    int64_t low_padding = 0;
    if (absl::c_linear_search(dims, i)) {
      low_padding = RoundUpTo(original.base_shape().dimensions(i),
                              original.sharding().tile_assignment().dim(i)) -
                    original.base_shape().dimensions(i);
    }
    dim->set_padding_low(low_padding);
    dim->set_padding_high(0);
    dim->set_base_dilation(1);
  }

  auto reshard_window = original.ReshardAsWindowedInput(
      window, original.sharding(),
      CreateZero(ShapeUtil::MakeShape(original.base_shape().element_type(), {}),
                 original.state().b),
      /*mask_invalid_region=*/false);
  if (!reshard_window.has_value()) {
    return nullptr;
  }
  CHECK(!reshard_window->dynamic_slice_index_on_output.has_value());
  return reshard_window->sharded_input;
}

bool IsNanSafeGt(HloComputation* comp) {
  namespace m = match;
  auto match_bitcast_f32 = [](int64_t parameter_number) {
    auto param = m::Parameter(parameter_number)
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32_t>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };
  auto match_bitcast_bf16 = [](int64_t parameter_number) {
    auto param = m::Convert(m::Parameter(parameter_number)
                                .WithShape(m::Shape().WithElementType(BF16)))
                     .WithShape(m::Shape().WithElementType(F32));
    auto param_s32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(S32));
    auto param_u32 =
        m::BitcastConvert(param).WithShape(m::Shape().WithElementType(U32));
    return m::Select(
        m::Lt(param_s32, m::ConstantScalar(0)),
        m::BitcastConvert(
            m::Subtract(m::ConstantScalar(std::numeric_limits<int32_t>::max()),
                        param_u32))
            .WithShape(m::Shape().WithElementType(S32)),
        param_s32);
  };
  // If root instruction is kSelect and compares indices if values are equal.
  if (comp->root_instruction()->opcode() == HloOpcode::kSelect) {
    return Match(comp->root_instruction()->operand(2),
                 m::Gt(match_bitcast_f32(0), match_bitcast_f32(1))) ||
           Match(comp->root_instruction()->operand(2),
                 m::Gt(match_bitcast_bf16(0), match_bitcast_bf16(1)));
  }
  return Match(comp->root_instruction(),
               m::Gt(match_bitcast_f32(0), match_bitcast_f32(1))) ||
         Match(comp->root_instruction(),
               m::Gt(match_bitcast_bf16(0), match_bitcast_bf16(1)));
}

std::optional<int64_t> GetKValueInTopKWhenPartitionSortDim(
    HloInstruction* hlo) {
  HloSortInstruction* sort = DynCast<HloSortInstruction>(hlo);
  if (sort == nullptr || sort->operand_count() != 2) {
    return std::nullopt;
  }
  if (!IsNanSafeGt(sort->to_apply())) {
    return std::nullopt;
  }
  HloInstruction* data = sort->mutable_operand(0);
  HloIotaInstruction* iota =
      DynCast<HloIotaInstruction>(sort->mutable_operand(1));
  const PrimitiveType element_type = data->shape().element_type();
  if (iota == nullptr || iota->shape().element_type() != S32 ||
      iota->opcode() != HloOpcode::kIota ||
      iota->iota_dimension() != sort->sort_dimension()) {
    return std::nullopt;
  }

  const int64_t sort_dim = sort->sort_dimension();

  if (element_type != F32 && element_type != BF16 && element_type != S32 &&
      element_type != U32) {
    return std::nullopt;
  }

  bool supported = true;
  std::optional<int64_t> k;
  for (HloInstruction* gte : sort->users()) {
    if (gte->opcode() != HloOpcode::kGetTupleElement) {
      supported = false;
      break;
    }

    const HloInstruction* slice = gte->users()[0];
    if (slice->opcode() != HloOpcode::kSlice) {
      // Non-slice user means we are not doing a TopK
      supported = false;
      break;
    }
    if (absl::c_any_of(slice->slice_starts(), [](int x) { return x != 0; }) ||
        absl::c_any_of(slice->slice_strides(), [](int x) { return x != 1; })) {
      // Strided slice or slicing at the beginning isn't supported.
      supported = false;
      break;
    }
    for (int64_t dim = 0; dim < data->shape().dimensions().size(); dim++) {
      if (dim == sort_dim) {
        continue;
      }
      if (slice->slice_limits(dim) !=
          slice->operand(0)->shape().dimensions(dim)) {
        // Slicing along the other dimension isn't supported.
        supported = false;
        break;
      }
    }
    if (!k.has_value()) {
      k = slice->slice_limits(sort_dim);
    } else if (k != slice->slice_limits(sort_dim)) {
      // Different k for the different operands isn't supported.
      supported = false;
      break;
    }
  }
  if (k == std::nullopt || !supported) {
    return std::nullopt;
  }

  // Only support when sort dim is sharded.
  if (!data->has_sharding()) {
    return std::nullopt;
  }
  const HloSharding& sharding = sort->operand(0)->sharding();

  if (sharding.IsTileMaximal()) {
    return std::nullopt;
  }

  // Check if partitioned at sort dimension.
  for (int64_t dim = 0; dim < sort->shape().tuple_shapes(0).dimensions().size();
       ++dim) {
    if (sharding.tile_assignment().dim(dim) > 1) {
      if (dim != sort_dim) {
        return std::nullopt;
      }
    }
  }

  // Checks if partition size is smaller than k.
  const int64_t shard_count = sharding.tile_assignment().dim(sort_dim);

  if (shard_count <= 1) {
    return std::nullopt;
  }

  const int64_t input_size = hlo->operand(0)->shape().dimensions(sort_dim);
  const int64_t per_partition_size = CeilOfRatio(input_size, shard_count);

  if (k.value() >= per_partition_size) {
    return std::nullopt;
  }

  return k;
}

// Slice first k elements from sort_dim.
HloInstruction* SliceFirstK(HloInstruction* hlo, SpmdBuilder* builder,
                            int64_t slice_dim, int64_t k) {
  const Shape& hlo_shape = hlo->shape();
  auto hlo_dims = hlo_shape.dimensions();
  std::vector<int64_t> start_indices(hlo_shape.dimensions().size(), 0);
  std::vector<int64_t> limit_indices(hlo_dims.begin(), hlo_dims.end());
  std::vector<int64_t> strides(hlo_shape.dimensions().size(), 1);
  limit_indices[slice_dim] = k;
  auto output_shape = hlo_shape;
  output_shape.set_dimensions(slice_dim, k);
  return builder->AddInstruction(HloInstruction::CreateSlice(
      output_shape, hlo, start_indices, limit_indices, strides));
}

// Check if a dimension is sharded.
int64_t ShardCountAtDim(const HloSharding& sharding, int64_t dim) {
  if (sharding.IsTileMaximal()) {
    return 1;
  }
  if (dim == -1) {
    // -1 is used as a placeholder in non-existing dims like
    // DotConvolutionDimsInfo.
    return 1;
  }
  return sharding.tile_assignment().dim(dim);
}

std::optional<std::vector<std::pair<int64_t, int64_t>>>
GetReshardAllToAllSourceTargetDims(const HloSharding& source,
                                   const HloSharding& target) {
  if (source.IsTileMaximal() || target.IsTileMaximal() ||
      source.tile_assignment().num_dimensions() !=
          target.tile_assignment().num_dimensions() ||
      source.NumTiles() != target.NumTiles()) {
    return std::nullopt;
  }
  // Record partition count to index for indices that have different partition
  // counts on source and target.
  std::map<int64_t, std::vector<int64_t>> source_size_to_dim;
  std::map<int64_t, std::vector<int64_t>> target_size_to_dim;
  for (int64_t i = 0; i < source.tile_assignment().num_dimensions(); ++i) {
    if (source.tile_assignment().dim(i) == target.tile_assignment().dim(i)) {
      continue;
    }
    source_size_to_dim[source.tile_assignment().dim(i)].push_back(i);
    target_size_to_dim[target.tile_assignment().dim(i)].push_back(i);
  }
  // In order to shard via AllToAll, source_size_to_dim and target_size_to_dim
  // must have the same distribution.
  if (source_size_to_dim.empty() ||
      source_size_to_dim.size() != target_size_to_dim.size()) {
    return std::nullopt;
  }
  for (const auto& entry : source_size_to_dim) {
    auto target_it = target_size_to_dim.find(entry.first);
    if (target_it == target_size_to_dim.end() ||
        target_it->second.size() != entry.second.size()) {
      return std::nullopt;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> result;
  auto remove_entry = [](int64_t size, int64_t dim,
                         std::map<int64_t, std::vector<int64_t>>& size_to_dim) {
    size_to_dim[size].erase(
        std::remove_if(size_to_dim[size].begin(), size_to_dim[size].end(),
                       [dim](int64_t a) { return a == dim; }),
        size_to_dim[size].end());
    if (size_to_dim[size].empty()) {
      size_to_dim.erase(size);
    }
  };
  // Find one pair of dimensions to swap at a time.
  while (!source_size_to_dim.empty()) {
    int64_t source_size = source_size_to_dim.begin()->first;
    int64_t i = source_size_to_dim.begin()->second.back();
    int64_t target_i_size = target.tile_assignment().dim(i);
    if (target_i_size == source_size) {
      remove_entry(source_size, i, source_size_to_dim);
      remove_entry(source_size, i, target_size_to_dim);
      continue;
    }
    auto j_it = source_size_to_dim[target_i_size].begin();
    int64_t j = *j_it;
    if (source_size == 1) {
      // If possible, find a j where the target partition count is not one, so
      // that when we swap, the resulting size-1 dimension will still be useful
      // to other dimensions.
      while (target.tile_assignment().dim(j) == 1) {
        if (++j_it == source_size_to_dim[target_i_size].end()) {
          break;
        }
        j = *j_it;
      }
    } else if (target_i_size % source_size == 0) {
      // If possible, find a j where the target partition count is source_size,
      // so that we can do a single swap.
      while (target.tile_assignment().dim(j) != source_size) {
        if (++j_it == source_size_to_dim[target_i_size].end()) {
          break;
        }
        j = *j_it;
      }
    } else {
      return std::nullopt;
    }
    result.emplace_back(j, i);
    remove_entry(target_i_size, i, target_size_to_dim);
    source_size_to_dim.begin()->second.back() = j;
    remove_entry(target_i_size, j, source_size_to_dim);
  }
  return result;
}

bool CanReshardWithCollectivePermute(const HloSharding& source,
                                     const HloSharding& target) {
  return !source.IsTileMaximal() && !target.IsTileMaximal() &&
         source.tile_assignment().dimensions() ==
             target.tile_assignment().dimensions() &&
         source.ReplicateOnLastTileDim() == target.ReplicateOnLastTileDim() &&
         source.tile_assignment() != target.tile_assignment();
}

std::optional<GroupedSharding> AlignGroupsWithInternal(
    GroupedSharding grouped_sharding, const GroupedSharding& reference,
    bool requires_compatibility, bool ignore_group_order) {
  CHECK_EQ(grouped_sharding.device_groups.num_groups(),
           reference.device_groups.num_groups());
  CHECK_EQ(grouped_sharding.device_groups.num_devices_per_group(),
           reference.device_groups.num_devices_per_group());
  // If the device groups are both in iota, and the iota is the same, return the
  // grouped sharding.
  if (grouped_sharding.device_groups.has_iota() &&
      reference.device_groups.has_iota() &&
      grouped_sharding.device_groups.iota() == reference.device_groups.iota()) {
    return grouped_sharding;
  }
  // Returns src -> dst index mapping.
  auto get_permutation = [&](int64_t base_group_idx,
                             int64_t reference_group_idx) {
    int64_t num_devices_per_group =
        grouped_sharding.device_groups.num_devices_per_group();
    absl::flat_hash_map<int64_t, int64_t> dst_reverse_map(
        num_devices_per_group);

    for (int64_t i = 0; i < num_devices_per_group; ++i) {
      dst_reverse_map[reference.device_groups(reference_group_idx, i)] = i;
    }
    std::vector<int64_t> permutation(num_devices_per_group);
    for (int64_t i = 0; i < num_devices_per_group; ++i) {
      auto it = dst_reverse_map.find(
          grouped_sharding.device_groups(base_group_idx, i));
      CHECK(it != dst_reverse_map.end());
      permutation[i] = it->second;
    }
    return permutation;
  };
  std::vector<int64_t> device_to_ref_group(
      reference.device_groups.num_total_devices());
  for (int64_t g = 0; g < reference.device_groups.num_groups(); ++g) {
    for (int64_t d = 0; d < reference.device_groups.num_devices_per_group();
         ++d) {
      device_to_ref_group[reference.device_groups(g, d)] = g;
    }
  }
  auto unique_ref_dev_group = [&](int64_t group) -> int64_t {
    int64_t ref_g = -1;
    for (int64_t device_idx = 0;
         device_idx < grouped_sharding.device_groups.num_devices_per_group();
         ++device_idx) {
      int64_t device = grouped_sharding.device_groups(group, device_idx);
      if (ref_g == -1) {
        ref_g = device_to_ref_group[device];
      } else if (ref_g != device_to_ref_group[device]) {
        return -1;
      }
    }
    return ref_g;
  };
  bool matching_groups = true;
  std::vector<int64_t> original_src_to_ref_permutation;
  for (int64_t g = 0; g < grouped_sharding.device_groups.num_groups(); ++g) {
    int64_t ref_g = unique_ref_dev_group(g);
    if (ref_g < 0 || (!ignore_group_order && g != ref_g)) {
      if (requires_compatibility) {
        return std::nullopt;
      }
      matching_groups = false;
      break;
    }
    if (g == 0) {
      original_src_to_ref_permutation = get_permutation(g, ref_g);
    } else if (requires_compatibility) {
      if (original_src_to_ref_permutation != get_permutation(g, ref_g)) {
        return std::nullopt;
      }
    }
  }
  if (matching_groups && !grouped_sharding.sharding.IsTileMaximal()) {
    auto tiles = [&] {
      auto array =
          grouped_sharding.sharding.tile_assignment().shared_array_clone();
      array->Each([&](absl::Span<const int64_t> indices, int64_t* device) {
        *device = original_src_to_ref_permutation[*device];
      });
      return TileAssignment(std::move(array));
    }();
    grouped_sharding.sharding =
        grouped_sharding.sharding.ReplicateOnLastTileDim()
            ? HloSharding::PartialTile(tiles)
            : HloSharding::Tile(tiles);
  }
  grouped_sharding.device_groups = reference.device_groups;
  return grouped_sharding;
}

GroupedSharding AlignGroupsWith(GroupedSharding grouped_sharding,
                                const GroupedSharding& reference,
                                bool ignore_group_order) {
  return *AlignGroupsWithInternal(std::move(grouped_sharding), reference,
                                  /*requires_compatibility=*/false,
                                  ignore_group_order);
}

std::optional<GroupedSharding> AlignGroupsWithIfCompatible(
    GroupedSharding grouped_sharding, const GroupedSharding& reference) {
  return AlignGroupsWithInternal(std::move(grouped_sharding), reference,
                                 /*requires_compatibility=*/true,
                                 /*ignore_group_order=*/false);
}

HloSharding AlignShardingOnDims(const HloSharding& sharding,
                                absl::Span<const int64_t> sharding_dims,
                                const HloSharding& reference,
                                absl::Span<const int64_t> reference_dims) {
  auto sharding_grouped =
      hlo_sharding_util::GroupShardingOnDims(sharding, sharding_dims);
  auto reference_grouped =
      hlo_sharding_util::GroupShardingOnDims(reference, reference_dims);
  return hlo_sharding_util::UngroupSharding(
      AlignGroupsWith(sharding_grouped, reference_grouped));
}

Shape GetPerGroupBaseShape(const GroupedSharding& grouped_sharding,
                           const Shape& original_base_shape) {
  auto result = original_base_shape;
  for (int64_t i = 0; i < grouped_sharding.group_dims.size(); ++i) {
    int64_t dim = grouped_sharding.group_dims[i];
    if (dim >= original_base_shape.dimensions().size()) {
      continue;
    }
    int64_t groups = grouped_sharding.group_dim_sizes[i];
    result.set_dimensions(dim, CeilOfRatio(result.dimensions(dim), groups));
  }
  return result;
}

PartitionedHlo::PartitioningState CreatePerGroupPartitioningState(
    const PartitionedHlo::PartitioningState& state,
    const DeviceGroupTileAssignment& device_groups, SpmdBuilder* b) {
  auto result = state;
  result.collective_ops_creator = GetPerGroupCollectiveOpsCreator(
      state.collective_ops_creator, device_groups);
  result.partition_id =
      GetInGroupPartitionId(state.partition_id, device_groups, b);

  std::string key;
  // If the device groups are an iota, use the iota string as the key.
  if (device_groups.has_iota()) {
    key = device_groups.iota()->ToString();
  } else {
    // Create a string key for the groups.
    std::vector<std::string> per_group_strings(device_groups.num_groups());
    for (int64_t i = 0; i < per_group_strings.size(); ++i) {
      for (int64_t j = 0; j < device_groups.num_devices_per_group(); ++j) {
        absl::StrAppend(&per_group_strings[i], device_groups(i, j), ",");
      }
    }
    key = absl::StrJoin(per_group_strings, ";");
  }
  auto& grouped_cache = state.reshard_cache->groupd_caches[key];
  if (!grouped_cache) {
    grouped_cache = std::make_unique<PartitionedHlo::ReshardCache>();
  }
  result.reshard_cache = grouped_cache.get();
  return result;
}

HloInstruction* PerGroupSliceFromReplicated(
    HloInstruction* replicated, HloInstruction* partition_id,
    const DeviceGroupTileAssignment& device_groups,
    absl::Span<const int64_t> group_dims,
    absl::Span<const int64_t> group_dim_sizes, SpmdBuilder* b) {
  std::vector<uint32_t> group_ids(device_groups.num_total_devices());
  for (int64_t g = 0; g < device_groups.num_groups(); ++g) {
    for (int64_t d = 0; d < device_groups.num_devices_per_group(); ++d) {
      group_ids[device_groups(g, d)] = g;
    }
  }
  auto group_id = TableLookup<uint32_t>(group_ids, U32, partition_id, b);
  std::vector<int64_t> group_level_tile_dims(
      replicated->shape().dimensions().size(), 1);
  for (int64_t i = 0; i < group_dims.size(); ++i) {
    group_level_tile_dims[group_dims[i]] = group_dim_sizes[i];
  }
  auto group_level_tile = [&] {
    absl::InlinedVector<int, 6> perm_dims(group_dims.begin(), group_dims.end());
    absl::c_sort(perm_dims);
    absl::InlinedVector<int, 6> perm_dim_map(group_level_tile_dims.size(), -1);
    for (int i = 0; i < perm_dims.size(); ++i) {
      perm_dim_map[perm_dims[i]] = i;
    }
    absl::InlinedVector<int, 6> transpose_perm(group_dims.size());
    for (int i = 0; i < group_dims.size(); ++i) {
      transpose_perm[i] = perm_dim_map[group_dims[i]];
      CHECK_NE(transpose_perm[i], -1);
    }
    return TileAssignment(group_level_tile_dims, group_dim_sizes,
                          transpose_perm);
  }();
  auto group_level_sharding = HloSharding::Tile(std::move(group_level_tile));
  auto padded_hlo = PadBaseShapeBeforeUnevenTiledSharding(
      replicated, group_level_sharding, b);
  auto shard_shape =
      MakePartitionedShape(replicated->shape(), group_level_sharding);
  return b->AddInstruction(HloInstruction::CreateDynamicSlice(
      shard_shape, padded_hlo,
      MakePartitionOffsets(replicated->shape(), group_level_sharding, group_id,
                           b),
      shard_shape.dimensions()));
}

std::optional<std::vector<int64_t>> FindMatchingPartitionedDimsForGrouping(
    const HloSharding& sharding,
    const DeviceGroupTileAssignment& device_groups) {
  if (sharding.IsTileMaximal() || device_groups.num_groups() < 2) {
    return std::nullopt;
  }
  const int64_t num_devices = sharding.tile_assignment().num_elements();
  if (num_devices != device_groups.num_elements()) {
    return std::nullopt;
  }
  std::vector<int64_t> dims;
  if (device_groups.num_devices_per_group() < 2) {
    // Trivial case: single member groups
    for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
      if (sharding.tile_assignment().dim(i) > 1) {
        dims.push_back(i);
      }
    }
    return dims;
  }

  std::vector<std::vector<int64_t>> device_to_index(
      num_devices,
      std::vector<int64_t>(sharding.tile_assignment().num_dimensions()));
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> index, int64_t device) {
        device_to_index[device].assign(index.begin(), index.end());
      });
  int64_t group_count = 1;
  for (int64_t i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (device_to_index[device_groups(0, 0)][i] ==
        device_to_index[device_groups(0, 1)][i]) {
      dims.push_back(i);
      group_count *= sharding.tile_assignment().dim(i);
    }
  }
  if (group_count != device_groups.num_groups()) {
    return std::nullopt;
  }
  for (int64_t g = 0; g < device_groups.num_groups(); ++g) {
    for (int64_t device = 1; device < device_groups.num_devices_per_group();
         ++device) {
      if (absl::c_any_of(dims, [&](const int64_t dim) {
            return device_to_index[device_groups(g, device)][dim] !=
                   device_to_index[device_groups(g, 0)][dim];
          })) {
        return std::nullopt;
      }
    }
  }
  return dims;
}

HloSharding CreateMatchingShardingOnDims(
    const Shape& target_shape, const HloSharding& source_sharding,
    absl::Span<const int64_t> target_dims,
    absl::Span<const int64_t> source_dims) {
  CHECK(target_dims.size() == source_dims.size())
      << "Expected 1:1 match between parallel dimensions";
  if (source_sharding.IsReplicated()) {
    return HloSharding::Replicate();
  }
  absl::InlinedVector<int64_t, 4> tile_dims(target_shape.dimensions().size(),
                                            1);
  int num_tiles = 1;
  for (int i = 0, end = target_dims.size(); i < end; ++i) {
    num_tiles *= source_sharding.tile_assignment().dim(source_dims[i]);
    tile_dims[target_dims[i]] =
        source_sharding.tile_assignment().dim(source_dims[i]);
  }
  // If there is some partition across non-parallel dimensions in the
  // other operand then partially replicate for the new
  bool to_be_partially_replicated = false;
  if (num_tiles != source_sharding.tile_assignment().num_elements()) {
    CHECK_EQ(source_sharding.tile_assignment().num_elements() % num_tiles, 0);
    to_be_partially_replicated = true;
    tile_dims.push_back(source_sharding.tile_assignment().num_elements() /
                        num_tiles);
  }
  auto tgt_tile_assignment =
      source_sharding.tile_assignment().Reshape(tile_dims);
  if (to_be_partially_replicated) {
    return AlignShardingOnDims(HloSharding::PartialTile(tgt_tile_assignment),
                               target_dims, source_sharding, source_dims);
  } else {
    return AlignShardingOnDims(HloSharding::Tile(tgt_tile_assignment),
                               target_dims, source_sharding, source_dims);
  }
}

std::optional<GatherScatterParallelDimSharding>
GatherScatterOperandsShardedAcrossParallelDims(
    const HloInstruction& operand, const HloInstruction& indices,
    const hlo_sharding_util::GatherScatterDims& parallel_dims) {
  const auto& indices_parallel_dims = parallel_dims.indices_dims;
  const auto& operand_parallel_dims = parallel_dims.operand_dims;
  if (indices_parallel_dims.size() != operand_parallel_dims.size()) {
    return std::nullopt;
  }
  auto new_index_shard = indices.sharding();
  auto new_operand_shard = operand.sharding();
  int idx_parallel_tiles_num = new_index_shard.NumTiles(indices_parallel_dims);
  int op_parallel_tiles_num = new_operand_shard.NumTiles(operand_parallel_dims);
  if (idx_parallel_tiles_num == 1 && op_parallel_tiles_num == 1) {
    return std::nullopt;
  }

  if (new_index_shard.IsReplicated()) {
    return GatherScatterParallelDimSharding{
        CreateMatchingShardingOnDims(indices.shape(), new_operand_shard,
                                     indices_parallel_dims,
                                     operand_parallel_dims),
        new_operand_shard};
  }
  if (new_operand_shard.IsReplicated()) {
    return GatherScatterParallelDimSharding{
        new_index_shard, CreateMatchingShardingOnDims(
                             operand.shape(), new_index_shard,
                             operand_parallel_dims, indices_parallel_dims)};
  }

  // Parallel dimension distribution needs to be the same, so try to steal
  // sharding from partial replication to compensate.
  if (idx_parallel_tiles_num != op_parallel_tiles_num) {
    auto to_adjust_dims = operand_parallel_dims;
    auto target_dims = indices_parallel_dims;
    HloSharding* target = &new_index_shard;
    HloSharding* to_adjust = &new_operand_shard;
    if (idx_parallel_tiles_num < op_parallel_tiles_num) {
      std::swap(to_adjust_dims, target_dims);
      std::swap(to_adjust, target);
    }
    if (!to_adjust->ReplicateOnLastTileDim()) {
      return std::nullopt;
    }
    std::vector<int64_t> new_tile_assignment_dims(
        to_adjust->tile_assignment().dimensions().begin(),
        to_adjust->tile_assignment().dimensions().end());
    for (int i = 0; i < to_adjust_dims.size(); ++i) {
      int64_t target_dim = target->tile_assignment().dim(target_dims[i]);
      int64_t to_adjust_dim =
          to_adjust->tile_assignment().dim(to_adjust_dims[i]);
      if (target_dim < to_adjust_dim) {
        return std::nullopt;
      }
      if (target_dim == to_adjust_dim) {
        continue;
      }
      int64_t ratio = target_dim / to_adjust_dim;
      if (target_dim % to_adjust_dim != 0 ||
          new_tile_assignment_dims.back() % ratio != 0) {
        return std::nullopt;
      }
      new_tile_assignment_dims[to_adjust_dims[i]] *= ratio;
      new_tile_assignment_dims.back() /= ratio;
    }
    CHECK_GE(new_tile_assignment_dims.back(), 1);
    bool to_partially_replicate = true;
    if (new_tile_assignment_dims.back() == 1) {
      new_tile_assignment_dims.pop_back();
      to_partially_replicate = false;
    }
    auto new_tile_assignment =
        to_adjust->tile_assignment().Reshape(new_tile_assignment_dims);
    if (to_partially_replicate) {
      *to_adjust =
          AlignShardingOnDims(HloSharding::PartialTile(new_tile_assignment),
                              to_adjust_dims, *target, target_dims);
    } else {
      *to_adjust = AlignShardingOnDims(HloSharding::Tile(new_tile_assignment),
                                       to_adjust_dims, *target, target_dims);
    }
  }
  // Make sure that the parallel dimensions are aligned.
  std::vector<int64_t> operand_shard_tile_dims(
      new_operand_shard.tile_assignment().dimensions().begin(),
      new_operand_shard.tile_assignment().dimensions().end());
  for (int i = 0; i < indices_parallel_dims.size(); ++i) {
    operand_shard_tile_dims[operand_parallel_dims[i]] =
        new_index_shard.tile_assignment().dim(indices_parallel_dims[i]);
  }
  auto operand_shard_tiles =
      new_operand_shard.tile_assignment().Reshape(operand_shard_tile_dims);
  new_operand_shard = AlignShardingOnDims(
      new_operand_shard.ReplicateOnLastTileDim()
          ? HloSharding::PartialTile(operand_shard_tiles)
          : HloSharding::Tile(operand_shard_tiles),
      operand_parallel_dims, new_index_shard, indices_parallel_dims);
  return GatherScatterParallelDimSharding{new_index_shard, new_operand_shard};
}

namespace {

const HloInstruction* SkipCopyOperands(const HloInstruction* operand,
                                       bool check_single_use = true) {
  while (operand->user_count() == 1 && operand->opcode() == HloOpcode::kCopy) {
    operand = operand->operand(0);
  }
  if (check_single_use && operand->user_count() != 1) {
    return nullptr;
  }
  return operand;
}

}  // namespace

std::optional<int64_t> FindRotateRightPattern(const HloInstruction* concat) {
  if (concat->operand_count() != 2) {
    return std::nullopt;
  }
  const HloInstruction* lhs = SkipCopyOperands(concat->operand(0));
  const HloInstruction* rhs = SkipCopyOperands(concat->operand(1));
  if (!lhs || !rhs) {
    return std::nullopt;
  }

  if (lhs->opcode() != HloOpcode::kSlice ||
      rhs->opcode() != HloOpcode::kSlice ||
      lhs->operand(0) != rhs->operand(0)) {
    return std::nullopt;
  }
  const HloInstruction* to_rotate = lhs->operand(0);
  if (!ShapeUtil::Compatible(to_rotate->shape(), concat->shape()) ||
      concat->sharding() != to_rotate->sharding()) {
    return std::nullopt;
  }
  const int64_t dim = concat->concatenate_dimension();
  if (lhs->slice_strides(dim) != 1 || rhs->slice_strides(dim) != 1 ||
      lhs->slice_starts(dim) != rhs->slice_limits(dim)) {
    return std::nullopt;
  }
  return lhs->shape().dimensions(dim);
}

std::optional<PadWithWrapPattern> FindPadWithWrapPattern(
    const HloInstruction* concat) {
  if (concat->operand_count() != 3) {
    return std::nullopt;
  }
  const HloInstruction* lhs = SkipCopyOperands(concat->operand(0));
  const HloInstruction* mid = SkipCopyOperands(concat->operand(1),
                                               /*check_single_use=*/false);
  const HloInstruction* rhs = SkipCopyOperands(concat->operand(2));
  if (!lhs || !mid || !rhs) {
    return std::nullopt;
  }

  // Skip elementwise unary operations applied to inst, returning
  // a list of applied operations that were skipped.
  auto skip_elementwise_ops = [&](const HloInstruction* inst) {
    std::vector<const HloInstruction*> modifiers;
    while (inst->IsElementwise() && inst->operand_count() == 1 &&
           inst->user_count() == 1) {
      if (inst->opcode() != HloOpcode::kCopy) {
        modifiers.push_back(inst);
      }
      inst = inst->operand(0);
    }
    return std::make_pair(modifiers, inst);
  };

  PadWithWrapPattern pad_pattern;
  auto skip_result = skip_elementwise_ops(lhs);
  pad_pattern.lhs_modifiers = std::move(skip_result.first);
  lhs = skip_result.second;

  skip_result = skip_elementwise_ops(rhs);
  pad_pattern.rhs_modifiers = std::move(skip_result.first);
  rhs = skip_result.second;

  const int64_t dim = concat->concatenate_dimension();
  if (lhs->opcode() != HloOpcode::kSlice ||
      rhs->opcode() != HloOpcode::kSlice || lhs->operand(0) != mid ||
      rhs->operand(0) != mid || lhs->slice_strides(dim) != 1 ||
      rhs->slice_strides(dim) != 1) {
    return std::nullopt;
  }
  pad_pattern.lhs_slice_start = lhs->slice_starts(dim);
  pad_pattern.rhs_slice_start = rhs->slice_starts(dim);
  return pad_pattern;
}

std::optional<PartitionedHlo::WindowedInputShardReturnValue>
ReshardDataForSlicing(absl::Span<const int64_t> strides,
                      absl::Span<const int64_t> starts,
                      absl::Span<const int64_t> limits,
                      PartitionedHlo to_reshard,
                      const HloSharding& target_sharding, SpmdBuilder* b) {
  // Create a window config to represent the slice.
  Window window;
  for (int64_t i = 0; i < starts.size(); ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_stride(strides[i]);
    dim->set_window_dilation(1);
    dim->set_window_reversal(false);
    dim->set_padding_low(-starts[i]);
    dim->set_padding_high(limits[i] - to_reshard.base_shape().dimensions(i));
    dim->set_base_dilation(1);
  }

  return to_reshard.ReshardAsWindowedInput(
      window, target_sharding,
      CreateZero(
          ShapeUtil::MakeShape(to_reshard.hlo()->shape().element_type(), {}),
          b),
      /*mask_invalid_region=*/false);
}

HloInstruction* SliceDataFromWindowReshard(
    const PartitionedHlo::WindowedInputShardReturnValue& reshard_operand,
    absl::Span<const int64_t> strides, const Shape& base_shape,
    const HloSharding& target_sharding, SpmdBuilder* b) {
  std::vector<int64_t> start_indices(strides.size());
  std::vector<int64_t> limit_indices(strides.size());
  bool need_slice = false;
  for (int64_t i = 0; i < strides.size(); ++i) {
    auto dim = reshard_operand.shard_window.dimensions(i);
    start_indices[i] = -dim.padding_low();
    limit_indices[i] = reshard_operand.sharded_input->shape().dimensions(i) +
                       dim.padding_high();
    if (start_indices[i] != 0 || strides[i] != 1 ||
        limit_indices[i] !=
            reshard_operand.sharded_input->shape().dimensions(i)) {
      need_slice = true;
    }
  }
  if (need_slice) {
    auto shard_shape = MakePartitionedShape(base_shape, target_sharding);
    return b->AddInstruction(
        HloInstruction::CreateSlice(shard_shape, reshard_operand.sharded_input,
                                    start_indices, limit_indices, strides));
  }
  return reshard_operand.sharded_input;
}

std::optional<PartitionedHlo::WindowedInputShardReturnValue> ReshardDataForPad(
    HloInstruction* pad_value, PaddingConfig pc, PartitionedHlo to_reshard,
    const HloSharding& target_sharding, SpmdBuilder* b) {
  // Create a window config to represent the pad.
  Window window;
  bool needs_masking = false;
  const bool pad_value_is_zero =
      pad_value->IsConstant() && pad_value->literal().IsZero({});
  for (int64_t i = 0; i < to_reshard.hlo()->shape().dimensions().size(); ++i) {
    WindowDimension* dim = window.add_dimensions();
    auto pd = pc.dimensions(i);
    dim->set_size(1);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_window_reversal(false);
    dim->set_padding_low(pd.edge_padding_low());
    dim->set_padding_high(pd.edge_padding_high());
    dim->set_base_dilation(pd.interior_padding() + 1);
    const int64_t shard_count = target_sharding.tile_assignment().dim(i);
    // Need masking only if there is non-zero padding value or the operand is
    // unevenly partitioned. Halo exchange fills 0 in collective permute result
    // for non-destination cores.
    needs_masking |= shard_count > 1 &&
                     (pd.edge_padding_low() > 0 || pd.edge_padding_high() > 0 ||
                      pd.interior_padding() > 0) &&
                     (!pad_value_is_zero ||
                      to_reshard.base_shape().dimensions(i) % shard_count != 0);
  }
  // In compact halo exchange, we can't skip masking.
  return to_reshard.ReshardAsWindowedInput(
      window, target_sharding, pad_value,
      /*mask_invalid_region=*/needs_masking, /*force_mask_in_compact=*/true);
}

HloInstruction* PadDataFromWindowReshard(
    const PartitionedHlo::WindowedInputShardReturnValue& reshard_operand,
    HloInstruction* pad_value, SpmdBuilder* b) {
  PaddingConfig sharded_padding_config;
  bool need_pad = false;
  for (int64_t i = 0;
       i < reshard_operand.sharded_input->shape().dimensions().size(); ++i) {
    auto dim = sharded_padding_config.add_dimensions();
    const auto& wd = reshard_operand.shard_window.dimensions(i);
    dim->set_edge_padding_low(wd.padding_low());
    dim->set_edge_padding_high(wd.padding_high());
    dim->set_interior_padding(wd.base_dilation() - 1);
    if (wd.padding_low() != 0 || wd.padding_high() != 0 ||
        wd.base_dilation() != 1) {
      need_pad = true;
    }
  }
  auto sharded_data = reshard_operand.sharded_input;
  if (need_pad) {
    auto sharded_data_shape =
        ShapeInference::InferPadShape(sharded_data->shape(), pad_value->shape(),
                                      sharded_padding_config)
            .value();
    return b->AddInstruction(HloInstruction::CreatePad(
        sharded_data_shape, sharded_data, pad_value, sharded_padding_config));
  }
  return sharded_data;
}

std::vector<std::vector<int64_t>> GetPartitionGroupsForReplication(
    const HloSharding& sharding, absl::Span<const int64_t> replication_dims) {
  absl::Span<const int64_t> tile_assignment_dims =
      sharding.tile_assignment().dimensions();
  DCHECK_GE(tile_assignment_dims.size(), replication_dims.size());
  int64_t group_size = 1;
  for (int64_t i : replication_dims) {
    DCHECK_LT(i, tile_assignment_dims.size());
    group_size *= tile_assignment_dims[i];
  }
  std::vector<int64_t> non_replication_indices;
  non_replication_indices.reserve(tile_assignment_dims.size() -
                                  replication_dims.size());
  for (int64_t i = 0; i < tile_assignment_dims.size(); ++i) {
    if (!absl::c_linear_search(replication_dims, i)) {
      non_replication_indices.push_back(i);
    }
  }
  DCHECK_EQ(replication_dims.size() + non_replication_indices.size(),
            tile_assignment_dims.size());
  std::vector<int64_t> non_replication_strides(non_replication_indices.size());
  if (!non_replication_strides.empty()) {
    non_replication_strides.back() = 1;
    for (int64_t i = non_replication_indices.size() - 1; i > 0; --i) {
      non_replication_strides[i - 1] =
          non_replication_strides[i] *
          tile_assignment_dims[non_replication_indices[i]];
    }
  }

  std::vector<std::vector<int64_t>> partition_groups(
      sharding.tile_assignment().num_elements() / group_size);
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t partition) {
        int64_t group_id = 0;
        auto non_replication_strides_it = non_replication_strides.begin();
        for (int64_t non_replication_index : non_replication_indices) {
          group_id +=
              indices[non_replication_index] * (*non_replication_strides_it);
          ++non_replication_strides_it;
        }
        DCHECK(non_replication_strides_it == non_replication_strides.end());
        DCHECK_LT(group_id, partition_groups.size());
        partition_groups[group_id].push_back(partition);
      });
  return partition_groups;
}

std::vector<std::vector<int64_t>> GetPartitionGroupsAcrossTargetDims(
    const HloSharding& sharding, std::vector<int64_t> target_dims,
    std::vector<int64_t> group_sizes) {
  CHECK(target_dims.size() == group_sizes.size());
  int64_t total_group_size = std::accumulate(
      group_sizes.begin(), group_sizes.end(), 1, std::multiplies<int64_t>());
  std::vector<std::vector<int64_t>> groups(
      sharding.tile_assignment().num_elements() / total_group_size);
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64_t> indices, int64_t device) {
        int64_t group_id = 0;
        for (int64_t dim = 0; dim < indices.size(); ++dim) {
          auto it = absl::c_find(target_dims, dim);
          if (it != target_dims.end()) {
            int64_t group_size =
                group_sizes[std::distance(target_dims.begin(), it)];
            group_id *= sharding.tile_assignment().dim(dim) / group_size;
            group_id += indices[dim] / group_size;
          } else {
            group_id *= sharding.tile_assignment().dim(dim);
            group_id += indices[dim];
          }
        }
        groups[group_id].push_back(device);
      });
  return groups;
}

std::optional<IotaReplicaGroupList> GetIotaPartitionGroupsAcrossTargetDims(
    const HloSharding& sharding, std::vector<int64_t> target_dims,
    std::vector<int64_t> group_sizes, int64_t num_partitions) {
  CHECK(target_dims.size() == group_sizes.size());
  // If provided sharding is not HloShardingV2, we cannot generate partition
  // groups in an iota format.
  if (!sharding.tile_assignment().iota().has_value()) {
    return std::nullopt;
  }

  // The goal of this function is to generate partition groups which span across
  // target dims without using explicit indexing and instead using transposes
  // and reshapes of the tile assignment. We do this by reshaping the tile
  // assignment by expanding each target dim to [target_dim/group_size,
  // group_size]. We then transpose the tile assignment, making the newly
  // created target dims the most minor dims, preserving the order of the target
  // dims. Consider the following example:
  // Tile assignment: [8,8,16]<=[1024]
  // Target dims: [0,1]
  // Group sizes: [4,4]
  //
  // In this case we would generate 64 replica groups of size 16.
  // These replica groups would span the target dims 0 and 1.
  // We perform the following steps on the original tile assignment:
  // 1. Expand target dims: [8,8,16]->[2,4,2,4,16]
  // 2. Transpose to make target dims minor: [2,4,2,4,16]->[2,2,16,4,4] with
  // (0,1,2,3,4) -> (0,2,4,1,3)
  // 3. Reshape to get groups of size 16: [2,4,16,2,4]->[2,2,16,16]
  int64_t total_group_size = std::accumulate(
      group_sizes.begin(), group_sizes.end(), 1, std::multiplies<int64_t>());
  int64_t num_replica_groups =
      sharding.tile_assignment().num_elements() / total_group_size;

  std::vector<int64_t> reshape_dimensions;
  reshape_dimensions.reserve(sharding.tile_assignment().num_dimensions());
  std::vector<int64_t> target_dim_locations;
  for (int64_t dim = 0; dim < sharding.tile_assignment().num_dimensions();
       ++dim) {
    auto it = std::find(target_dims.begin(), target_dims.end(), dim);
    if (it != target_dims.end()) {
      int64_t current_val = sharding.tile_assignment().dim(dim);
      int64_t group_size = group_sizes[std::distance(target_dims.begin(), it)];
      reshape_dimensions.push_back(current_val / group_size);
      reshape_dimensions.push_back(group_size);
      target_dim_locations.push_back(reshape_dimensions.size() - 1);
    } else {
      reshape_dimensions.push_back(sharding.tile_assignment().dim(dim));
    }
  }

  std::vector<int> transpose_dims(reshape_dimensions.size());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  for (int64_t loc : target_dim_locations) {
    auto it = std::find(transpose_dims.begin(), transpose_dims.end(), loc);
    if (it != transpose_dims.end()) {
      transpose_dims.erase(it);
      transpose_dims.push_back(loc);
    }
  }

  // Step 1: Expand target dims.
  auto reshaped_tile_assignment =
      sharding.tile_assignment().Reshape(reshape_dimensions);

  // If after the reshape we do not have an iota tile assignment
  // (HloShardingV2), we cannot generate a compressed format.
  if (!reshaped_tile_assignment.iota().has_value()) {
    return std::nullopt;
  }

  // Step 2: Transpose the tile assignment to make the target dims minor.
  auto tranposed_tile_assignment =
      reshaped_tile_assignment.iota()->Transpose(transpose_dims);
  // If after the transpose we do not have an iota tile assignment
  // (HloShardingV2), we cannot generate a compressed format.
  if (!tranposed_tile_assignment.has_value()) {
    return std::nullopt;
  }

  // Step 3: Final reshape to get groups of size total_group_size. This is done
  // implicitly by creating an IotaReplicaGroupList with num_replica_groups,
  // total_group_size.
  IotaReplicaGroupList groups(
      num_replica_groups, total_group_size,
      tranposed_tile_assignment.value().reshape_dims(),
      tranposed_tile_assignment.value().transpose_perm());
  return groups;
}

// Returns partition groups in an iota format.
std::optional<IotaReplicaGroupList> GetIotaPartitionGroupsForReplication(
    const HloSharding& sharding, absl::Span<const int64_t> replication_dims,
    int64_t num_partitions) {
  // If provided sharding is not HloShardingV2, we cannot generate partition
  // groups in an iota format.
  if (!sharding.tile_assignment().iota().has_value()) {
    return std::nullopt;
  }

  int64_t group_size = 1;
  for (int64_t i : replication_dims) {
    group_size *= sharding.tile_assignment().dim(i);
  }

  int64_t num_replica_groups =
      sharding.tile_assignment().num_elements() / group_size;

  // The compressed replica group list involves transposing and reshaping the
  // initial tile assignment. We transpose the original tile assignment so that
  // the dims we want to replicate across are at the end, essentially making
  // these dims the most minor dims. We then reshape the initial tile assignment
  // into a tile assignment with dims [M, N], where M is the number of replica
  // groups and N is the size of each replica group.
  std::vector<int> transpose_dims(sharding.tile_assignment().num_dimensions());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);

  // Sorting is not necessary but is done to match the non-optimized equivalent
  // function.
  std::vector<int> replication_dims_sorted(replication_dims.begin(),
                                           replication_dims.end());
  std::sort(replication_dims_sorted.begin(), replication_dims_sorted.end());
  for (int64_t i : replication_dims_sorted) {
    auto it = std::find(transpose_dims.begin(), transpose_dims.end(), i);
    if (it != transpose_dims.end()) {
      transpose_dims.erase(it);
      transpose_dims.push_back(i);
    }
  }

  // If after the transpose, we do not have an iota tile assignment
  // (HloShardingV2), we cannot generate a compressed format. Currently sharding
  // V2 only supports some types of transposes.
  auto transpose_iota_tile_assignment =
      sharding.tile_assignment().iota()->Transpose(transpose_dims);
  if (!transpose_iota_tile_assignment.has_value()) {
    return std::nullopt;
  }

  // We generate a compressed device list with this information.
  return IotaReplicaGroupList(num_replica_groups, group_size,
                              transpose_iota_tile_assignment->reshape_dims(),
                              transpose_iota_tile_assignment->transpose_perm());
}

// Expands partition group list across all replicas. Expects that provided
// partition group list utilizes all the partitions.
CollectiveDeviceList ExpandPartitionGroupListAcrossReplicas(
    IotaReplicaGroupList partition_group_list, int num_replicas,
    int num_partitions) {
  int partition_group_count = partition_group_list.num_replica_groups();
  int partition_group_size = partition_group_list.num_devices_per_group();
  // Verify that partition group list utilizes all partitions.
  CHECK_EQ((partition_group_count * partition_group_size), num_partitions);

  // Total number of replica groups is number of partitions groups * num of
  // replicas.
  int replica_group_count = partition_group_count * num_replicas;

  // Newly generated replica group list expands the pattern within one replica
  // across all replicas. For example, if we want to expand a partition group
  // list with 10 groups of size 5 with reshape dims [2,5,5] and
  // transpose dims of [0,2,1] to 2 replicas, we would generate a replica group
  // list with 20 groups of size 5 with reshape dims [2,2,5,5] and
  // transpose dims [0,1,3,2].
  std::vector<int64_t> new_reshape_dims(
      partition_group_list.reshape_dims().begin(),
      partition_group_list.reshape_dims().end());
  new_reshape_dims.insert(new_reshape_dims.begin(), num_replicas);
  std::vector<int> new_transpose_dims;
  new_transpose_dims.push_back(0);
  for (int64_t dim : partition_group_list.transpose_perm()) {
    new_transpose_dims.push_back(dim + 1);
  }

  return CollectiveDeviceList(
      IotaReplicaGroupList(replica_group_count, partition_group_size,
                           new_reshape_dims, new_transpose_dims));
}

PartitionedHlo MakeACopyAndReturnItsPartitionedHlo(const PartitionedHlo& phlo,
                                                   SpmdBuilder* b) {
  HloInstruction* copy_hlo = b->AddInstruction(HloInstruction::CreateUnary(
      phlo.hlo()->shape(), HloOpcode::kCopy, phlo.hlo()));
  copy_hlo->copy_sharding(phlo.hlo());
  return PartitionedHlo(copy_hlo, phlo.base_shape(), phlo.state());
}

DynamicUpdateSliceAnalysis AnalyzeDynamicUpdateSlice(
    const HloInstruction* hlo) {
  CHECK(!hlo->sharding().IsTileMaximal());

  DynamicUpdateSliceAnalysis analysis;

  bool update_on_a_single_partition = true;
  bool has_partitioned_slice_dim_with_dynamic_index = false;
  for (int64_t i = 0; i < hlo->shape().dimensions().size(); ++i) {
    if (hlo->operand(1)->shape().dimensions(i) == hlo->shape().dimensions(i)) {
      continue;
    }
    analysis.slice_dims.push_back(i);

    if (hlo->sharding().tile_assignment().dim(i) == 1) {
      continue;
    }
    analysis.partitioned_slice_dims.push_back(i);

    int64_t slice_size = hlo->operand(1)->shape().dimensions(i);
    if (slice_size == 1) {
      continue;
    }

    if (hlo->operand(i + 2)->IsConstant()) {
      const PrimitiveType elemType =
          hlo->operand(i + 2)->shape().element_type();
      int64_t start_index =
          elemType == S64 ? hlo->operand(i + 2)->literal().Get<int64_t>({})
                          : hlo->operand(i + 2)->literal().Get<int>({});
      int64_t end_index = start_index + slice_size - 1;

      int64_t per_partition_size = CeilOfRatio(
          hlo->shape().dimensions(i), hlo->sharding().tile_assignment().dim(i));
      if (start_index / per_partition_size != end_index / per_partition_size) {
        update_on_a_single_partition = false;
      }
    } else {
      update_on_a_single_partition = false;
      has_partitioned_slice_dim_with_dynamic_index = true;
    }
  }

  if (analysis.partitioned_slice_dims.empty()) {
    analysis.method = DynamicUpdateSliceMethod::kDefault;
  } else if (update_on_a_single_partition) {
    analysis.method = DynamicUpdateSliceMethod::kUpdateOnASinglePartition;
  } else if (has_partitioned_slice_dim_with_dynamic_index) {
    analysis.method = DynamicUpdateSliceMethod::kDefault;
  } else {
    analysis.method =
        DynamicUpdateSliceMethod::kAllPartitionedSliceDimsHaveConstantIndices;
  }

  return analysis;
}

}  // namespace spmd
}  // namespace xla
