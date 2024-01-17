/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace support {

// Converts ShardingParam to OpSharding.
//
// This assumes that `sharding_param` is valid.
//
// Returns error when `device_mapping` can't map the logical devices in
// `sharding_param`.
StatusOr<OpSharding> ToOpSharding(const ShardingParam& sharding_param,
                                  absl::Span<const int> device_mapping);

// Converts HloSharding to OpSharding.
//
// It assumes that `hlo_sharding` is valid.
//
// Returns error when `hlo_sharding` cannot be converted to sharding param.
StatusOr<ShardingParam> ToShardingParam(const HloSharding& hlo_sharding,
                                        absl::Span<const int64_t> shape,
                                        absl::Span<const int> axis_sizes);

}  // namespace support
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SUPPORT_SHARDING_CONVERSIONS_H_
