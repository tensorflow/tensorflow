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
    const Shape& shape, const se::DeviceDescription& device_desc) {
  int64 num_elements = ShapeUtil::ElementsIn(shape);
  if (num_elements <= 1) {
    return LaunchDimensions();
  }

  // Since we don't do any inter-warp communication, we're free to choose any
  // block size we want, subject to hardware constraints.  We choose the
  // smallest block size that allows the GPU to reach full occupancy (assuming
  // the kernel uses sufficiently few registers).  This gives us max performance
  // when the kernel uses few registers, and lets us scale down gracefully as
  // the kernel uses more registers.
  //
  // Specifically, we choose the number of threads per block such that
  //
  //   <num threads per block> * <max blocks per core> = <max threads per core>

  auto threads_per_core = device_desc.threads_per_core_limit();
  auto blocks_per_core = device_desc.blocks_per_core_limit();
  int64 threads_per_block;
  if (threads_per_core != 0 && blocks_per_core != 0) {
    threads_per_block = device_desc.threads_per_core_limit() /
                        device_desc.blocks_per_core_limit();
  } else {
    static std::atomic<int64> log_count{0};
    if (log_count.fetch_add(1) < 8) {
      LOG(WARNING) << "Attempting to calculate launch dimensions for GPU "
                      "without full information about its capabilities.  "
                      "StreamExecutor's PopulateDeviceDescription should be "
                      "updated for this device.";
    }
    threads_per_block = device_desc.threads_per_warp();
    if (threads_per_block == 0) {
      // Fall back to *something* if we can't even get num threads per warp.
      threads_per_block = 32;
    }
  }

  if (num_elements < threads_per_block) {
    threads_per_block = num_elements;
    VLOG(2) << "Update # of threads per block to the element count ("
            << threads_per_block << ") because the latter is smaller.";
  }

  int64 block_count = CeilOfRatio(num_elements, threads_per_block);
  VLOG(2) << tensorflow::strings::Printf(
      "Initialized the block count to ceil(# of elements / threads per "
      "block) = ceil(%lld/%lld) = %lld",
      num_elements, threads_per_block, block_count);

  return LaunchDimensions(block_count, threads_per_block);
}

}  // namespace gpu
}  // namespace xla
