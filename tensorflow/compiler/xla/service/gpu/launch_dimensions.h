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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_

#include <map>
#include <memory>

#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {
namespace gpu {

// Encapsulates the launch dimensions of a kernel, e.g., the block count and the
// number of threads per block.
class LaunchDimensions {
 public:
  // The default constructor creates a launch dimension that indicate
  // single-threaded execution.
  LaunchDimensions() : block_count_(1), threads_per_block_(1) {}

  LaunchDimensions(int64 block_count, int64 threads_per_block)
      : block_count_(block_count), threads_per_block_(threads_per_block) {}

  bool IsSinglethreaded() const {
    return block_count_ == 1 && threads_per_block_ == 1;
  }

  int64 block_count() const { return block_count_; }
  int64 threads_per_block() const { return threads_per_block_; }
  int64 launch_bound() const { return block_count() * threads_per_block(); }

 private:
  int64 block_count_;
  int64 threads_per_block_;
};

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims);

LaunchDimensions CalculateLaunchDimensions(const Shape& shape,
                                           GpuDeviceInfo gpu_device_info,
                                           int unroll_factor = 1);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
