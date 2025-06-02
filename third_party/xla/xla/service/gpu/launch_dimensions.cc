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

#include "xla/service/gpu/launch_dimensions.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

WorkDimensions LaunchDimensions::AsWorkDimensions() const {
  return WorkDimensions{
      NumWorkClusters{},
      NumWorkGroups{block_counts_.x, block_counts_.y, block_counts_.z},
      NumWorkItems{thread_counts_per_block_.x, thread_counts_per_block_.y,
                   thread_counts_per_block_.z}};
}

LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config) {
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  if (num_elements <= 1) {
    return LaunchDimensions();
  }
  num_elements = CeilOfRatio(num_elements, int64_t{dim_config.unroll_factor});
  const int kWarpSchedulers = 4;

  if (xla::PlatformUtil::CanonicalPlatformName("gpu").value() == "rocm") {
    int64_t threads_per_block_x = std::min<int64_t>(
        gpu_device_info.threads_per_warp() * kWarpSchedulers, num_elements);

    int64_t num_blocks = CeilOfRatio(num_elements, threads_per_block_x);
    CHECK(num_blocks < gpu_device_info.block_dim_limit().x);

    int threads_per_block_y = 1;
    while ((num_blocks * threads_per_block_x) >
           std::numeric_limits<uint32_t>::max()) {
      threads_per_block_x /= 2;
      threads_per_block_y *= 2;
    }

    return LaunchDimensions(
        se::BlockDim(num_blocks, 1, 1),
        se::ThreadDim(threads_per_block_x, threads_per_block_y, 1));

  } else {
    int64_t threads_per_block = std::min<int64_t>(
        gpu_device_info.threads_per_warp() * kWarpSchedulers, num_elements);

    int64_t num_blocks_total = CeilOfRatio(num_elements, threads_per_block);
    int64_t num_blocks_y = CeilOfRatio<uint64_t>(
        num_blocks_total, gpu_device_info.block_dim_limit().x);
    int64_t num_blocks_x = CeilOfRatio(num_blocks_total, num_blocks_y);

    return LaunchDimensions(se::BlockDim(num_blocks_x, num_blocks_y, 1),
                            se::ThreadDim(threads_per_block, 1, 1));
  }
}

LaunchDimensionsProto LaunchDimensions::ToProto() const {
  LaunchDimensionsProto proto;
  *proto.mutable_block_counts() = block_counts_.ToProto();
  *proto.mutable_thread_counts_per_block() = thread_counts_per_block_.ToProto();
  return proto;
}

absl::StatusOr<LaunchDimensions> LaunchDimensions::FromProto(
    const LaunchDimensionsProto& proto) {
  TF_ASSIGN_OR_RETURN(
      stream_executor::BlockDim block_counts,
      stream_executor::BlockDim::FromProto(proto.block_counts()));
  TF_ASSIGN_OR_RETURN(
      stream_executor::ThreadDim thread_counts_per_block,
      stream_executor::ThreadDim::FromProto(proto.thread_counts_per_block()));
  return LaunchDimensions{block_counts, thread_counts_per_block};
}
}  // namespace gpu
}  // namespace xla
