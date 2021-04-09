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

#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"

#include <ostream>
#include <string>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims) {
  LaunchDimensions::Dim3D block_counts = launch_dims.block_counts();
  LaunchDimensions::Dim3D thread_counts = launch_dims.thread_counts_per_block();
  out << absl::StrFormat("[block: {%d, %d, %d}, thread: {%d, %d, %d}]",
                         block_counts.x, block_counts.y, block_counts.z,
                         thread_counts.x, thread_counts.y, thread_counts.z);
  return out;
}

static int64 ThreadsPerBlockLimit(GpuDeviceInfo gpu_device_info) {
  int64 threads_per_block = gpu_device_info.threads_per_block_limit;
  if (threads_per_block <= 0) {
    static std::atomic<int64> log_count{0};
    if (log_count.fetch_add(1) < 8) {
      LOG(WARNING) << "Attempting to calculate launch dimensions for GPU "
                      "without full information about its capabilities.  "
                      "StreamExecutor's PopulateDeviceDescription should be "
                      "updated for this device.";
    }
    threads_per_block = gpu_device_info.threads_per_warp;
    if (threads_per_block == 0) {
      // Fall back to *something* if we can't even get num threads per warp.
      threads_per_block = 32;
    }
  }
  return threads_per_block;
}

// Calculates the launch dimensions used to invoke `hlo`.
StatusOr<LaunchDimensions> CalculateLaunchDimensions(
    const Shape& shape, GpuDeviceInfo gpu_device_info, int unroll_factor,
    bool few_waves) {
  int64 num_elements = ShapeUtil::ElementsIn(shape);
  if (num_elements <= 1) {
    return LaunchDimensions();
  }

  CHECK_EQ(num_elements % unroll_factor, 0);
  num_elements = num_elements / unroll_factor;

  // Since we don't do any inter-warp communication, we're free to choose any
  // block size we want, subject to hardware constraints.  We choose the largest
  // block size allowed, as empirically, this is a performance win on almost
  // (but not all) benchmarks.
  //
  // My guess is that using a larger block size encourages ptxas to decrease
  // per-thread register usage, thus allowing for higher occupancy, but I
  // haven't verified this.
  //
  // TODO(jlebar): Investigate this further, and tune this heuristic so we can
  // run faster on the few benchmarks where smaller block size helps.
  int64 threads_per_block = ThreadsPerBlockLimit(gpu_device_info);
  // We unroll kernels to make use of vectorized loads/stores. This means we
  // need more registers to hold intermediate values. Reduce the number of
  // blocks per thread to increase the number of registers available to ptxas.
  // Make sure we still have a multiple of 32.
  threads_per_block =
      RoundUpToNearest(threads_per_block / unroll_factor, int64{32});
  if (num_elements < threads_per_block) {
    threads_per_block = num_elements;
    VLOG(2) << "Update # of threads per block to the element count ("
            << threads_per_block << ") because the latter is smaller.";
  }

  int64 block_count = CeilOfRatio(num_elements, threads_per_block);
  if (few_waves) {
    int64 capped_threads_per_block = std::min<int64>(threads_per_block, 128);
    int64 capped_block_count =
        gpu_device_info.core_count *
        (gpu_device_info.threads_per_core_limit / capped_threads_per_block);
    // Do not increase the number of blocks. This can happens for
    // small num_elements.
    if (capped_block_count < block_count) {
      threads_per_block = capped_threads_per_block;
      block_count = capped_block_count;
    }
  }

  if (gpu_device_info.block_dim_limit_x > 0 &&
      block_count >= gpu_device_info.block_dim_limit_x) {
    return tensorflow::errors::Unimplemented(
        "Kernel launch needs more blocks (", block_count,
        ") than allowed by hardware (", gpu_device_info.block_dim_limit_x,
        ").");
  }

  VLOG(2) << absl::StrFormat(
      "Initialized the block count to ceil(# of elements / threads per "
      "block) = ceil(%d/%d) = %d",
      num_elements, threads_per_block, block_count);

  return LaunchDimensions(block_count, threads_per_block);
}

}  // namespace gpu
}  // namespace xla
