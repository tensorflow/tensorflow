/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"

#include <map>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace hlo_sharding_util {

absl::optional<int64> SelectDominantDevice(
    const std::map<int64, int64>& device_map, int64* top_count) {
  int64 device = 0;
  int64 count = 0;
  for (auto& it : device_map) {
    if (it.second > count) {
      count = it.second;
      device = it.first;
    }
  }
  if (top_count != nullptr) {
    *top_count = count;
  }
  return count > 0 ? absl::optional<int64>(device) : absl::optional<int64>();
}

Status AssignComputationDevice(HloComputation* computation, int64 device) {
  VLOG(4) << "Assigning device " << device << " to " << computation->name()
          << " computation";
  for (HloInstruction* instruction : computation->instructions()) {
    if (!instruction->has_sharding()) {
      VLOG(4) << "Assigning device " << device << " to " << instruction->name();
      instruction->set_device_sharding(device);
    }
  }
  return Status::OK();
}

absl::optional<int64> GetMostOccurringDevice(
    absl::Span<HloInstruction* const> instructions) {
  std::map<int64, int64> device_map;
  for (HloInstruction* instruction : instructions) {
    if (instruction->has_sharding()) {
      for (auto& it : instruction->sharding().UsedDevices(nullptr)) {
        // The UsedDevices() API returns a map<device, occurrence_count>.
        device_map[it.first] += it.second;
      }
    }
  }
  return SelectDominantDevice(device_map, nullptr);
}

StatusOr<absl::optional<int64>> GetDominantDevice(
    absl::Span<HloComputation* const> computations, double dominant_factor) {
  int64 instruction_count = 0;
  std::map<int64, int64> device_map;
  for (HloComputation* computation : computations) {
    for (HloInstruction* instruction : computation->instructions()) {
      int64 count = 1;
      if (instruction->has_sharding()) {
        for (auto& it : instruction->sharding().UsedDevices(&count)) {
          // The UsedDevices() API returns a map<device, occurrence_count>.
          device_map[it.first] += it.second;
        }
      }
      instruction_count += count;
    }
  }
  int64 count;
  absl::optional<int64> device = SelectDominantDevice(device_map, &count);
  absl::optional<int64> dominant_device;
  if (device) {
    double factor =
        static_cast<double>(count) / static_cast<double>(instruction_count);
    if (factor >= dominant_factor) {
      dominant_device = device;
    }
  }
  return dominant_device;
}

HloSharding TransposeSharding(const HloSharding& sharding,
                              const std::vector<int64>& dimensions) {
  if (sharding.IsTileMaximal()) {
    return sharding;
  }
  const int64 rank = dimensions.size();
  std::vector<int64> tile_assignment_dim(rank);
  for (int64 i = 0; i < rank; ++i) {
    tile_assignment_dim[i] = sharding.tile_assignment().dim(dimensions[i]);
  }
  Array<int64> tile_assignment = sharding.tile_assignment();
  tile_assignment.Reshape(tile_assignment_dim);
  tile_assignment.Each([&](absl::Span<const int64> indices, int64* value) {
    std::vector<int64> src_indices(indices.size(), -1);
    for (int64 i = 0; i < indices.size(); ++i) {
      src_indices[dimensions[i]] = indices[i];
    }
    *value = sharding.tile_assignment()(src_indices);
  });
  return HloSharding::Tile(tile_assignment);
}

absl::optional<HloSharding> ReshapeSharding(const Shape& source_shape,
                                            const Shape& target_shape,
                                            const HloSharding& sharding) {
  if (sharding.IsTileMaximal()) {
    return sharding;
  }

  // In case of a tiled sharding the reshaped sharding will be a valid if the
  // reshape is composed from the following operations:
  // * Adding or removing dimensions with size 1.
  // * Merging consecutive dimensions where only the most major is sharded.
  // * Splitting a dimension to consecutive dimensions.
  // * Any reshaping of unsharded dimensions.
  // Note that merge and split can happen consecutively on the same dimension,
  // e.g., f32[1024,256,1024] to f32[128,2048,1024] can be considered that 1024
  // gets split into 128 and 8, but 8 then gets merged with 256. We use stacks
  // to make supporting such cases easy.
  const Shape tile_shape = sharding.TileShape(source_shape);
  std::vector<int64> target_tile_assignment_dimensions;
  std::vector<int64> source_dims_stack(source_shape.rank());
  std::vector<int64> target_dims_stack(target_shape.rank());
  std::vector<int64> sharding_tile_dims_stack(source_shape.rank());
  for (int64 i = 0; i < source_shape.rank(); ++i) {
    source_dims_stack[i] = source_shape.dimensions(source_shape.rank() - 1 - i);
    sharding_tile_dims_stack[i] =
        sharding.tile_assignment().dim(source_shape.rank() - 1 - i);
  }
  for (int64 i = 0; i < target_shape.rank(); ++i) {
    target_dims_stack[i] = target_shape.dimensions(target_shape.rank() - 1 - i);
  }
  while (!source_dims_stack.empty() || !target_dims_stack.empty()) {
    if (target_dims_stack.empty()) {
      if (Product(sharding_tile_dims_stack) != 1) {
        return absl::nullopt;
      }
      break;
    }
    int64 s_size = 1;
    int64 t_size = 1;
    int64 s_partitions = 1;
    if (!source_dims_stack.empty()) {
      s_size = source_dims_stack.back();
      source_dims_stack.pop_back();
      s_partitions = sharding_tile_dims_stack.back();
      sharding_tile_dims_stack.pop_back();
    }
    t_size = target_dims_stack.back();
    target_dims_stack.pop_back();
    if (s_partitions * Product(sharding_tile_dims_stack) == 1) {
      // No more partitions left.
      target_tile_assignment_dimensions.push_back(1);
      continue;
    }
    if (s_size == t_size) {
      // Same dimension.
      target_tile_assignment_dimensions.push_back(s_partitions);
    } else if (t_size == 1) {
      // Trivial dimension added.
      target_tile_assignment_dimensions.push_back(1);
      source_dims_stack.push_back(s_size);
      sharding_tile_dims_stack.push_back(s_partitions);
    } else if (s_size == 1) {
      // Trivial dimension removed.
      if (s_partitions != 1) {
        return absl::nullopt;
      }
      target_dims_stack.push_back(t_size);
    } else if (s_size > t_size) {
      // Dimension split.
      if (s_size % t_size != 0 || t_size % s_partitions != 0) {
        return absl::nullopt;
      }
      target_tile_assignment_dimensions.push_back(s_partitions);
      // We have part of the s_size unprocessed, so put it back to stack.
      source_dims_stack.push_back(s_size / t_size);
      sharding_tile_dims_stack.push_back(1);
    } else {
      // Dimension merge. Also merge the source dimension with the next, and
      // process it next time.
      if (s_size % s_partitions != 0) {
        return absl::nullopt;
      }
      CHECK(!source_dims_stack.empty());
      if (sharding_tile_dims_stack.back() != 1 && s_size != s_partitions) {
        // If the next dimension to combine is sharded, we require that the
        // current dimension's shard size to be 1. Otherwise, the new shard
        // would be non-contiguous.
        return absl::nullopt;
      }
      source_dims_stack.back() *= s_size;
      sharding_tile_dims_stack.back() *= s_partitions;
      target_dims_stack.push_back(t_size);
    }
  }
  Array<int64> new_tile_assignment = sharding.tile_assignment();
  new_tile_assignment.Reshape(target_tile_assignment_dimensions);
  return HloSharding::Tile(new_tile_assignment);
}

HloSharding ReshapeToTileDimension(const HloSharding& sharding, int64 dim,
                                   absl::Span<const int64> dims) {
  CHECK(!sharding.IsTuple() && !sharding.IsTileMaximal());
  CHECK_NE(absl::c_find(dims, dim), dims.end()) << "dim is not in dims";
  // We optimize the tile assignment on the single dimension dim in a way to
  // minimize communication among devices caused by the reshard:
  // +---+---+               +---+---+              +-+-+-+-+
  // |   |   |               |   0   |              | | | | |
  // | 0 | 1 |               +-------+              | | | | |
  // |   |   |  reshape on   |   1   |  reshape on  | | | | |
  // +---+---+   dim 0  =>   +-------+   dim 1  =>  |0|2|1|3|
  // |   |   |               |   2   |              | | | | |
  // | 2 | 3 |               +-------+              | | | | |
  // |   |   |               |   3   |              | | | | |
  // +---+---+               +---+---+              +-+-+-+-+

  std::vector<int64> tile_dims(sharding.tile_assignment().num_dimensions(), 1);
  // Handle ignore dimensions.
  std::vector<int64> ignore_sizes;
  int64 ignore_size = 1;
  for (int64 i = 0; i < sharding.tile_assignment().num_dimensions(); ++i) {
    if (absl::c_find(dims, i) == dims.end()) {
      int64 size = sharding.tile_assignment().dim(i);
      ignore_sizes.push_back(size);
      tile_dims[i] = size;
      ignore_size *= size;
    }
  }

  using Buckets = std::vector<std::vector<int64>>;
  Array<Buckets> buckets(ignore_sizes,
                         Buckets(sharding.tile_assignment().dim(dim)));
  sharding.tile_assignment().Each(
      [&](absl::Span<const int64> index, int64 device) {
        std::vector<int64> ignore_index;
        for (int64 i = 0; i < index.size(); ++i) {
          if (absl::c_find(dims, i) == dims.end()) {
            ignore_index.push_back(index[i]);
          }
        }
        buckets(ignore_index)[index[dim]].push_back(device);
      });
  std::vector<int64> devices;
  buckets.Each([&](absl::Span<const int64> index, const Buckets& buckets) {
    for (auto& bucket : buckets) {
      devices.insert(devices.end(), bucket.begin(), bucket.end());
    }
  });
  tile_dims[dim] = devices.size() / ignore_size;
  Array<int64> tile_assignment(tile_dims);
  tile_assignment.SetValues(devices);
  return HloSharding::Tile(tile_assignment);
}

bool ContainsTileSharding(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->has_sharding() &&
          !instruction->sharding().IsTileMaximal()) {
        return true;
      }
    }
  }
  return false;
}

HloSharding GatherOutputSharding(const HloSharding& index_sharding,
                                 const HloInstruction* hlo) {
  if (index_sharding.IsTileMaximal()) {
    return index_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  std::vector<int64> output_tile_assignment_dims;
  for (int64 i = 0, index_dim = 0; i < hlo->shape().rank(); ++i) {
    if (absl::c_binary_search(dnums.offset_dims(), i)) {
      output_tile_assignment_dims.push_back(1);
    } else {
      output_tile_assignment_dims.push_back(
          index_sharding.tile_assignment().dim(index_dim));
      index_dim++;
    }
  }
  Array<int64> new_tile_assignment = index_sharding.tile_assignment();
  new_tile_assignment.Reshape(output_tile_assignment_dims);
  return HloSharding::Tile(new_tile_assignment);
}

HloSharding GatherIndexSharding(const HloSharding& output_sharding,
                                const HloInstruction* hlo) {
  if (output_sharding.IsTileMaximal()) {
    return output_sharding;
  }

  const GatherDimensionNumbers& dnums = hlo->gather_dimension_numbers();
  std::vector<int64> index_tile_assignment_dims;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      index_tile_assignment_dims.push_back(
          output_sharding.tile_assignment().dim(i));
    }
  }
  Array<int64> new_tile_assignment = output_sharding.tile_assignment();
  new_tile_assignment.Reshape(index_tile_assignment_dims);
  return HloSharding::Tile(new_tile_assignment);
}

HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo) {
  if (hlo.sharding().IsTileMaximal()) {
    return hlo.sharding();
  }

  const GatherDimensionNumbers& dnums = hlo.gather_dimension_numbers();
  std::vector<int64> tile_assignment_dims(hlo.shape().rank());
  int64 num_elements = 1;
  for (int64 i = 0; i < hlo.shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      tile_assignment_dims[i] = hlo.sharding().tile_assignment().dim(i);
      num_elements *= hlo.sharding().tile_assignment().dim(i);
    } else {
      tile_assignment_dims[i] = 1;
    }
  }
  if (num_elements == hlo.sharding().tile_assignment().num_elements()) {
    // Output sharding is only on non offset dimensions. We use output sharding
    // to shard this gather op directly.
    return hlo.sharding();
  }

  if (num_elements == 1) {
    // Output sharding is only on offset dimensions. We do not shard this gather
    // op. Return a tile maximal sharding with the first device in output
    // sharding tile assignment.
    return HloSharding::AssignDevice(*hlo.sharding().tile_assignment().begin());
  }

  // Output sharding is on both offset and non offset dimensions. We shard the
  // gather op only on non offset dimensions.
  // For example:
  // - the gather op has sharding [2,2]{0,1,2,3},
  // - first dimension is non offset dimension,
  // - second dimension is offset dimension,
  // Then the result sharding will be [2,1]{0,2}.
  std::vector<int64> slice_starts(hlo.shape().rank(), 0LL),
      slice_limits(hlo.shape().rank());
  for (int64 i = 0; i < hlo.shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.offset_dims(), i)) {
      slice_limits[i] = hlo.sharding().tile_assignment().dim(i);
    } else {
      slice_limits[i] = 1;
    }
  }
  Array<int64> tile_assignment =
      hlo.sharding().tile_assignment().Slice(slice_starts, slice_limits);
  return HloSharding::Tile(tile_assignment);
}

HloSharding ScatterIndexSharding(const HloSharding& data_sharding,
                                 const HloInstruction* hlo) {
  if (data_sharding.IsTileMaximal()) {
    return data_sharding;
  }

  const ScatterDimensionNumbers& dnums = hlo->scatter_dimension_numbers();
  std::vector<int64> index_tile_assignment_dims;
  for (int64 i = 0; i < hlo->shape().rank(); ++i) {
    if (!absl::c_binary_search(dnums.update_window_dims(), i)) {
      index_tile_assignment_dims.push_back(
          data_sharding.tile_assignment().dim(i));
    }
  }
  if (index_tile_assignment_dims.size() < hlo->operand(1)->shape().rank()) {
    index_tile_assignment_dims.push_back(1);
  }
  Array<int64> new_tile_assignment = data_sharding.tile_assignment();
  new_tile_assignment.Reshape(index_tile_assignment_dims);
  return HloSharding::Tile(new_tile_assignment);
}

HloSharding ScatterDataSharding(const HloSharding& index_sharding,
                                const HloInstruction* hlo) {
  if (index_sharding.IsTileMaximal()) {
    return index_sharding;
  }

  const ScatterDimensionNumbers& dnums = hlo->scatter_dimension_numbers();
  std::vector<int64> data_tile_assignment_dims;
  for (int64 i = 0, index_dim = 0; i < hlo->shape().rank(); ++i) {
    if (absl::c_binary_search(dnums.update_window_dims(), i)) {
      data_tile_assignment_dims.push_back(1);
    } else {
      data_tile_assignment_dims.push_back(
          index_sharding.tile_assignment().dim(index_dim));
      index_dim++;
    }
  }
  Array<int64> new_tile_assignment = index_sharding.tile_assignment();
  new_tile_assignment.Reshape(data_tile_assignment_dims);
  return HloSharding::Tile(new_tile_assignment);
}

HloSharding ScatterEffectiveIndexSharding(const HloSharding& index_sharding,
                                          const HloInstruction& hlo) {
  if (index_sharding.IsTileMaximal()) {
    return index_sharding;
  }

  // Only shard on first "number of scatter_window_dims" dimensions.
  const ScatterDimensionNumbers& dnums = hlo.scatter_dimension_numbers();
  int64 num_elements = 1;
  int64 index_dim = 0;
  for (int64 i = 0; i < hlo.shape().rank(); ++i) {
    if (absl::c_binary_search(dnums.inserted_window_dims(), i)) {
      num_elements *= index_sharding.tile_assignment().dim(index_dim);
      index_dim++;
    }
  }
  if (num_elements == index_sharding.tile_assignment().num_elements()) {
    // Index sharding is only on scatter_window_dims. We use this index sharding
    // directly.
    return index_sharding;
  }

  // Index sharding is only on update_window_dims. We do not shard this scatter
  // op. Return a tile maximal sharding with the first device in index sharding
  // tile assignment.
  if (num_elements == 1) {
    return HloSharding::AssignDevice(*index_sharding.tile_assignment().begin());
  }

  const int64 index_rank = hlo.operand(1)->shape().rank();
  std::vector<int64> slice_starts(index_rank, 0LL), slice_limits(index_rank);
  for (int64 i = 0; i < index_rank; ++i) {
    if (i < index_dim) {
      slice_limits[i] = index_sharding.tile_assignment().dim(i);
    } else {
      slice_limits[i] = 1;
    }
  }
  Array<int64> tile_assignment =
      index_sharding.tile_assignment().Slice(slice_starts, slice_limits);
  return HloSharding::Tile(tile_assignment);
}

HloSharding ScatterEffectiveDataSharding(const HloSharding& data_sharding,
                                         const HloInstruction& hlo) {
  if (data_sharding.IsTileMaximal()) {
    return data_sharding;
  }

  const ScatterDimensionNumbers& dnums = hlo.scatter_dimension_numbers();
  const int64 data_rank = hlo.operand(2)->shape().rank();
  std::vector<int64> tile_assignment_dims(data_rank, 1LL);
  int64 num_elements = 1;
  for (int64 i = 0; i < hlo.shape().rank(); ++i) {
    if (absl::c_binary_search(dnums.inserted_window_dims(), i)) {
      CHECK_LT(i, data_rank);
      tile_assignment_dims[i] = data_sharding.tile_assignment().dim(i);
      num_elements *= data_sharding.tile_assignment().dim(i);
    }
  }
  if (num_elements == data_sharding.tile_assignment().num_elements()) {
    // Data sharding is only on scatter_window_dims. We use this data sharding
    // directly.
    return data_sharding;
  }

  if (num_elements == 1) {
    // Data sharding is only on update_window_dims. We do not shard this
    // scatter op. Return a tile maximal sharding with the first device in
    // data sharding tile assignment.
    return HloSharding::AssignDevice(*data_sharding.tile_assignment().begin());
  }

  // Data sharding is on both update_window_dims and scatter_window_dims. We
  // shard the scatter op only on scatter_window_dims. For example:
  // - the scatter data has sharding [2,2]{0,1,2,3},
  // - first dimension is scatter_window_dims,
  // - second dimension is update_window_dims,
  // Then the result sharding will be [2,1]{0,2}.
  std::vector<int64> slice_starts(data_rank, 0LL);
  Array<int64> tile_assignment =
      data_sharding.tile_assignment().Slice(slice_starts, tile_assignment_dims);
  return HloSharding::Tile(tile_assignment);
}

StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter) {
  auto computation = scatter.to_apply();
  // We only handle computations with 2 parameters and only 1 calculation.
  if (computation->instruction_count() != 3) {
    return Status(
        tensorflow::error::Code::INVALID_ARGUMENT,
        "Expected scatter reduce computation with 2 parameters and only 1 "
        "calculation");
  }

  auto root_instruction = computation->root_instruction();
  if (root_instruction->opcode() == HloOpcode::kAdd ||
      root_instruction->opcode() == HloOpcode::kOr) {
    return std::make_pair(HloInstruction::CreateConstant(LiteralUtil::Zero(
                              scatter.shape().element_type())),
                          root_instruction->opcode());
  } else if (root_instruction->opcode() == HloOpcode::kMultiply ||
             root_instruction->opcode() == HloOpcode::kAnd) {
    return std::make_pair(HloInstruction::CreateConstant(
                              LiteralUtil::One(scatter.shape().element_type())),
                          root_instruction->opcode());
  } else if (root_instruction->opcode() == HloOpcode::kMaximum) {
    return std::make_pair(HloInstruction::CreateConstant(LiteralUtil::MinValue(
                              scatter.shape().element_type())),
                          root_instruction->opcode());
  } else if (root_instruction->opcode() == HloOpcode::kMinimum) {
    return std::make_pair(HloInstruction::CreateConstant(LiteralUtil::MaxValue(
                              scatter.shape().element_type())),
                          root_instruction->opcode());
  }

  return Status(tensorflow::error::Code::INVALID_ARGUMENT,
                "Expected scatter reduce computation which is "
                "add/or/multiply/add/min/max");
}

std::vector<int64> DevicesForSharding(
    const HloSharding& sharding, const std::vector<int64>& available_devices) {
  std::vector<int64> devices;
  if (sharding.IsReplicated()) {
    for (int64 d : available_devices) {
      if (!HloSharding::IsReservedDevice(d)) {
        devices.push_back(d);
      }
    }
    return devices;
  }

  for (int64 i : available_devices) {
    if (sharding.UsesDevice(i)) {
      devices.push_back(i);
    }
  }
  DCHECK(std::all_of(sharding.tile_assignment().begin(),
                     sharding.tile_assignment().end(), [&](int64 device) {
                       return std::find(available_devices.begin(),
                                        available_devices.end(),
                                        device) != available_devices.end();
                     }));
  return devices;
}

}  // namespace hlo_sharding_util
}  // namespace xla
