/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/support/sharding_conversions.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace support {

StatusOr<OpSharding> ToOpSharding(const ShardingParam& sharding_param,
                                  absl::Span<const int> device_mapping) {
  OpSharding op_sharding;
  op_sharding.set_type(OpSharding::OTHER);

  // Populate tile_assignment_dimensions.
  auto* tile_assignment_dims = op_sharding.mutable_tile_assignment_dimensions();
  int64_t cum_size = 1;
  tile_assignment_dims->Reserve(sharding_param.dim_shards().size() + 1);
  for (const int64_t dim_shard : sharding_param.dim_shards()) {
    cum_size *= dim_shard;
    tile_assignment_dims->Add(dim_shard);
  }
  int device_count = 1;
  for (const int axis_size : sharding_param.minor_to_major().axis_sizes) {
    device_count *= axis_size;
  }
  if (device_count != cum_size) {
    op_sharding.set_replicate_on_last_tile_dim(true);
    tile_assignment_dims->Add(device_count / cum_size);
  }

  // Populate tile_assignment_devices.
  llvm::SmallVector<int, 4> devices;
  sharding_param.minor_to_major().ToDeviceList(devices);
  auto* tile_assignment_devices = op_sharding.mutable_tile_assignment_devices();
  tile_assignment_devices->Reserve(devices.size());
  for (const int device : devices) {
    if (device < 0 || device >= device_mapping.size()) {
      return absl::OutOfRangeError(absl::StrCat("Can't map device ", device));
    }
    tile_assignment_devices->Add(device_mapping[device]);
  }

  return op_sharding;
}

StatusOr<ShardingParam> ToShardingParam(const HloSharding& hlo_sharding,
                                        absl::Span<const int64_t> shape,
                                        absl::Span<const int> axis_sizes) {
  // Dim shards matches the rank of the tensor, with each entry representing
  // the number of shards for the corresponding dimension.
  llvm::SmallVector<int64_t> dim_shards;
  dim_shards.reserve(shape.size());
  // `axis_sizes` with the sizes of the mesh dimensions
  // `permutation` of the same length as `axis_sizes` telling how the shards
  // are mapped over the axis in `minor_to_major` order.
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.axis_sizes.reserve(axis_sizes.size());
  minor_to_major.permutation.reserve(axis_sizes.size());
  for (auto axis_size : axis_sizes) {
    minor_to_major.axis_sizes.push_back(axis_size);
  }
  if (hlo_sharding.IsReplicated()) {
    for (int i = 0; i < shape.size(); ++i) {
      dim_shards.push_back(1);
    }
    for (int axis_idx = 0; axis_idx < axis_sizes.size(); ++axis_idx) {
      minor_to_major.permutation.push_back(axis_idx);
    }
    return ShardingParam(dim_shards, std::move(minor_to_major));
  }
  return absl::UnimplementedError(
      absl::StrCat("Only converting from replicated HloSharding is supported.",
                   hlo_sharding.ToString()));
}

}  // namespace support
}  // namespace ifrt
}  // namespace xla
