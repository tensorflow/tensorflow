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

#ifndef XLA_PYTHON_IFRT_SUPPORT_SHARDING_CONVERSIONS_H_
#define XLA_PYTHON_IFRT_SUPPORT_SHARDING_CONVERSIONS_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace support {

// Converts ShardingParam and a device_mapping to OpSharding.
//
// The function assumes that `sharding_param` is valid. The logical device
// ids from `sharding_param` are used as indices into the device_mapping to
// obtain the device ids to create the OpSharding.
//
// Returns error when `device_mapping` can't map the logical devices in
// `sharding_param`.
absl::StatusOr<OpSharding> ToOpSharding(const ShardingParam& sharding_param,
                                        absl::Span<const int> device_mapping);

// Converts ShardingParam to HloSharding.
//
// This assumes that `sharding_param` is valid.
// The returned HloSharding uses the same logical device ids as the
// given ShardingParam.
absl::StatusOr<HloSharding> ToHloSharding(const ShardingParam& sharding_param);

// Converts HloSharding to ShardingParam.
//
// It assumes that `hlo_sharding` is valid.
//
// Returns error when `hlo_sharding` cannot be converted to sharding param.
// Only a subset of HloShardings are supported: REPLICATED (including MAXIMAL
// on single-device), partially replicated, fully partitioned shardings.
// (Non-fully-replicated) MAXIMAL and MANUAL shardings are not supported.
absl::StatusOr<ShardingParam> ToShardingParam(const HloSharding& hlo_sharding,
                                              int rank, int num_devices);

}  // namespace support
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SUPPORT_SHARDING_CONVERSIONS_H_
