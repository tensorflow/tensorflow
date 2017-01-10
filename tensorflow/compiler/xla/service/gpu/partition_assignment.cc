/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"

#include <ostream>
#include <string>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims) {
  out << tensorflow::strings::Printf("[block: %lld, thread: %lld]",
                                     launch_dims.block_count(),
                                     launch_dims.threads_per_block());
  return out;
}

// Calculates the launch dimensions used to invoke `hlo`.
LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& device_desc,
    PartitionStrategy partition_strategy) {
  int64 warp_size = device_desc.threads_per_warp();

  int64 num_elements = ShapeUtil::ElementsIn(shape);
  if (num_elements <= 1) {
    return LaunchDimensions();
  }

  // Calculate the number of threads per block.
  // Initialize threads_per_block as the threads-per-block limit.
  int64 threads_per_block = device_desc.threads_per_block_limit();
  VLOG(2) << "Initial # of threads per block = " << threads_per_block;

  if (partition_strategy == PartitionStrategy::kLatency) {
    // Limit the thread count to allow maximum number of registers per thread.
    // TODO(b/28560520): We don't have to assume the emitted kernel will use up
    // all the registers. We could use ptxas to examine the actual number of
    // register used, and set the thread count accordingly.
    int64 threads_per_block_limit_due_to_registers =
        device_desc.registers_per_core_limit() /
        device_desc.registers_per_thread_limit();
    CHECK_NE(0, threads_per_block_limit_due_to_registers);
    if (threads_per_block_limit_due_to_registers < threads_per_block) {
      threads_per_block =
          // Make `threads_per_block` a multiple of warp size to use GPU
          // efficiently.
          warp_size *
          std::max(1LL, threads_per_block_limit_due_to_registers / warp_size);
      VLOG(2) << "Update # of threads per block due to register pressure = "
              << threads_per_block;
    }
  }

  if (num_elements < threads_per_block) {
    threads_per_block = num_elements;
    VLOG(2) << "Update # of threads per block to the element count ("
            << threads_per_block << ") because the latter is smaller.";
  }

  // Calculate the block count. We copy the strategy used by Eigen:
  // eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorExecutor.h
  int64 block_count = CeilOfRatio(num_elements, threads_per_block);
  VLOG(2) << tensorflow::strings::Printf(
      "Initialized the block count to ceil(# of elements / threads per "
      "block) = ceil(%lld/%lld) = %lld",
      num_elements, threads_per_block, block_count);

  return LaunchDimensions(block_count, threads_per_block);
}

}  // namespace gpu
}  // namespace xla
