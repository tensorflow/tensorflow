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
#include <atomic>
#include <cstdint>
#include <ostream>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims) {
  se::BlockDim block_counts = launch_dims.block_counts();
  se::ThreadDim thread_counts = launch_dims.thread_counts_per_block();
  out << absl::StrFormat("[block: {%d, %d, %d}, thread: {%d, %d, %d}]",
                         block_counts.x, block_counts.y, block_counts.z,
                         thread_counts.x, thread_counts.y, thread_counts.z);
  return out;
}

static int64_t ThreadsPerBlockLimit(
    const se::DeviceDescription& gpu_device_info) {
  int64_t threads_per_block = gpu_device_info.threads_per_block_limit();
  if (threads_per_block <= 0) {
    static std::atomic<int64_t> log_count{0};
    if (log_count.fetch_add(1) < 8) {
      LOG(WARNING) << "Attempting to calculate launch dimensions for GPU "
                      "without full information about its capabilities.  "
                      "StreamExecutor's PopulateDeviceDescription should be "
                      "updated for this device.";
    }
    threads_per_block = gpu_device_info.threads_per_warp();
    if (threads_per_block == 0) {
      // Fall back to *something* if we can't even get num threads per warp.
      threads_per_block = 32;
    }
  }
  return threads_per_block;
}

int64_t ThreadsPerBlockRowVectorized(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config) {
  if (shape.dimensions().empty()) {
    return -1;
  }
  int64_t threads_per_block_row_vectorized =
      shape.dimensions().back() / dim_config.unroll_factor;
  if (dim_config.row_vectorized &&
      shape.dimensions().back() % dim_config.unroll_factor == 0 &&
      // If the row size is a multiple of 256, then use the old code
      // path that use a block size of 256. This give small speed up on V100.
      // Vectorization of the row load was already happening.
      (shape.dimensions().back() % 256) != 0 &&
      // We do not support row that do not fit in one block.
      threads_per_block_row_vectorized <=
          gpu_device_info.threads_per_block_limit()) {
    return threads_per_block_row_vectorized;
  }
  return -1;
}

namespace {

struct BlockSizes {
  int64_t threads_per_block_x;
  int64_t threads_per_block_y;
  int64_t block_count;
};

BlockSizes GetBlockSizes(LaunchDimensionsConfig dim_config,
                         const se::DeviceDescription& gpu_device_info,
                         const Shape& shape, int64_t num_elements) {
  if (!dim_config.row_vectorized && !dim_config.few_waves) {
    BlockSizes result;
    const int kWarpSchedulers = 4;
    result.threads_per_block_x = std::min<int64_t>(
        gpu_device_info.threads_per_warp() * kWarpSchedulers, num_elements);
    result.threads_per_block_y = 1;
    result.block_count = CeilOfRatio(
        num_elements, result.threads_per_block_x * result.threads_per_block_y);
    return result;
  }

  int64_t threads_per_block_row_vectorized =
      ThreadsPerBlockRowVectorized(shape, gpu_device_info, dim_config);
  // If row vectorized, threads_per_block_x is the vectorized size.
  // Otherwise, we unroll kernels to make use of vectorized
  // loads/stores. This means we need more registers to hold
  // intermediate values. Reduce the number of threads per block to
  // increase the number of registers available to ptxas.  Make sure
  // we still have a multiple of 32.
  BlockSizes result;
  int64_t max_threads_per_block_x =
      threads_per_block_row_vectorized > 0
          ? threads_per_block_row_vectorized
          : RoundUpTo(ThreadsPerBlockLimit(gpu_device_info) /
                          dim_config.unroll_factor,
                      int64_t{32});
  result.threads_per_block_x = std::min(num_elements, max_threads_per_block_x);
  // threads_per_block_y > 1 when we row vectorize and have small row size.
  result.threads_per_block_y =
      threads_per_block_row_vectorized > 0 &&
              threads_per_block_row_vectorized < 128 && num_elements > 128
          ? CeilOfRatio(static_cast<int64_t>(128),
                        threads_per_block_row_vectorized)
          : 1;
  VLOG(2) << "Set # of threads per block to (.x=" << result.threads_per_block_x
          << ", .y=" << result.threads_per_block_y << ")";

  result.block_count = CeilOfRatio(
      num_elements, result.threads_per_block_x * result.threads_per_block_y);
  if (dim_config.few_waves) {
    if (dim_config.row_vectorized) {
      // This multiple of 32 was tuned to not cause regression on multiple
      // benchmarks. It isn't a value that is optimal for all kernels. Maybe
      // looking at the arithmetic intensity of the kernels can specialize the
      // multiple per kernel.
      int64_t max_block_count =
          32 * gpu_device_info.core_count() *
          (gpu_device_info.threads_per_core_limit() /
           (result.threads_per_block_x * result.threads_per_block_y));
      int64_t capped_block_count = result.block_count;
      while (capped_block_count > max_block_count) {
        capped_block_count /= 2;
      }
      if (capped_block_count < result.block_count) {
        result.block_count = capped_block_count;
        VLOG(2) << "Update # of blocks to " << result.block_count
                << " as few_waves is enabled.";
      }
    } else {
      int64_t capped_threads_per_block_x =
          std::min<int64_t>(result.threads_per_block_x, 128);
      int64_t capped_block_count =
          gpu_device_info.core_count() *
          (gpu_device_info.threads_per_core_limit() /
           (capped_threads_per_block_x * result.threads_per_block_y));
      if (capped_block_count < result.block_count) {
        result.threads_per_block_x = capped_threads_per_block_x;
        result.block_count = capped_block_count;
        VLOG(2) << "Update the # of blocks to " << result.block_count
                << " and the # of threads per blocks to "
                << result.threads_per_block_x
                << " as the few_waves mode is enabled.";
      }
    }
  }
  return result;
}

}  // namespace

LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config) {
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  if (num_elements <= 1) {
    return LaunchDimensions();
  }
  num_elements = CeilOfRatio(num_elements, int64_t{dim_config.unroll_factor});
  BlockSizes sizes =
      GetBlockSizes(dim_config, gpu_device_info, shape, num_elements);

  return LaunchDimensions(
      se::BlockDim(sizes.block_count, 1, 1),
      se::ThreadDim(sizes.threads_per_block_x, sizes.threads_per_block_y, 1));
}

}  // namespace gpu
}  // namespace xla
