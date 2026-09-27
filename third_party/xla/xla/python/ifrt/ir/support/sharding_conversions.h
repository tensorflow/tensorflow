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

#ifndef XLA_PYTHON_IFRT_IR_SUPPORT_SHARDING_CONVERSIONS_H_
#define XLA_PYTHON_IFRT_IR_SUPPORT_SHARDING_CONVERSIONS_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace support {

// Converts ShardingParam to OpSharding.
//
// The function assumes that `sharding_param` is valid.
absl::StatusOr<xla::OpSharding> ToOpSharding(
    const ShardingParam& sharding_param);

// Converts ShardingParam to HloSharding.
//
// If `logical_device_ids` is provided, then a HloShardingV1 is returned.
// Otherwise, IOTA is assumed and a HloShardingV2 is returned.
//
// The function assumes that `sharding_param` is valid.
absl::StatusOr<xla::HloSharding> ToHloSharding(
    const ShardingParam& sharding_param,
    std::optional<llvm::ArrayRef<int>> logical_device_ids = std::nullopt);

// Converts HloSharding to ShardingParam.
//
// It assumes that `hlo_sharding` is a valid HloShardingV2.
//
// Returns error when `hlo_sharding` cannot be converted to sharding param.
// Only a subset of HloShardings are supported: REPLICATED (including MAXIMAL
// on single-device), partially replicated, fully partitioned shardings.
// (Non-fully-replicated) MAXIMAL and MANUAL shardings are not supported.
absl::StatusOr<ShardingParam> ToShardingParam(
    const xla::HloSharding& hlo_sharding, int rank, int num_devices);

struct ShardingParamWithDeviceIds {
  ShardingParam sharding_param;
  // If `logical_device_ids` is nullopt, then the logical device ids are assumed
  // to be IOTA.
  std::optional<std::vector<int>> logical_device_ids;
};

// Converts HloSharding to ShardingParam and device ids.
//
// It assumes that `hlo_sharding` is valid. Has the same limitations as
// `ToShardingParam`, but also supports converting HloShardingV1 (i.e.,
// with non-iota tile assignments).
absl::StatusOr<ShardingParamWithDeviceIds> ToShardingParamAndDevices(
    const xla::HloSharding& hlo_sharding, int rank, int num_devices);

}  // namespace support
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_SUPPORT_SHARDING_CONVERSIONS_H_
