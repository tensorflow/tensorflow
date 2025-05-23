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

#ifndef XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
#define XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {
namespace gpu {

// Encapsulates the launch dimensions of a kernel, e.g., the block count and the
// number of threads per block.
class LaunchDimensions {
 public:
  // The default constructor creates a launch dimension that indicate
  // single-threaded execution.
  constexpr LaunchDimensions()
      : block_counts_(se::BlockDim()),
        thread_counts_per_block_(se::ThreadDim()) {}

  constexpr LaunchDimensions(uint64_t block_x_count,
                             uint64_t thread_x_count_per_block)
      : block_counts_(block_x_count, 1, 1),
        thread_counts_per_block_(thread_x_count_per_block, 1, 1) {}

  constexpr LaunchDimensions(const se::BlockDim& block_counts,
                             const se::ThreadDim& thread_counts_per_block)
      : block_counts_(block_counts),
        thread_counts_per_block_(thread_counts_per_block) {}

  se::BlockDim block_counts() const { return block_counts_; }

  se::ThreadDim thread_counts_per_block() const {
    return thread_counts_per_block_;
  }

  // Returns the total number of blocks.
  uint64_t num_blocks() const {
    return block_counts_.x * block_counts_.y * block_counts_.z;
  }

  // Returns the total number of threads in a block.
  uint64_t num_threads_per_block() const {
    return thread_counts_per_block_.x * thread_counts_per_block_.y *
           thread_counts_per_block_.z;
  }

  uint64_t launch_bound() const {
    return num_blocks() * num_threads_per_block();
  }

  std::string ToString() const {
    return absl::StrCat("blocks: {", block_counts_.x, ", ", block_counts_.y,
                        ", ", block_counts_.z, "}, threads/block: {",
                        thread_counts_per_block_.x, ", ",
                        thread_counts_per_block_.y, ", ",
                        thread_counts_per_block_.z, "}");
  }

 private:
  se::BlockDim block_counts_;
  se::ThreadDim thread_counts_per_block_;
};

struct LaunchDimensionsConfig {
  // The kernel implementation will be unrolled if `unroll_factor` is
  // greater than one.
  int unroll_factor = 1;
};

// Returns -1 if the shape doesn't allow the row vectorization code path.
// If supported, return the number of threads to use in that case.
int64_t ThreadsPerBlockRowVectorized(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config);

// Calculates the launch dimensions used to invoke `hlo`.
LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config = {});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
