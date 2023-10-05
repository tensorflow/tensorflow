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

#ifndef XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
#define XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_

#include <ostream>
#include <string>

#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Encapsulates the launch dimensions of a kernel, e.g., the block count and the
// number of threads per block.
class LaunchDimensions {
 public:
  struct Dim3D {
    int64_t x, y, z;

    bool operator==(const Dim3D& other) const {
      return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Dim3D& other) const { return !(*this == other); }
  };

  // The default constructor creates a launch dimension that indicate
  // single-threaded execution.
  LaunchDimensions()
      : block_counts_({1, 1, 1}), thread_counts_per_block_({1, 1, 1}) {}

  LaunchDimensions(int64_t block_x_count, int64_t thread_x_count_per_block)
      : block_counts_({block_x_count, 1, 1}),
        thread_counts_per_block_({thread_x_count_per_block, 1, 1}) {}

  LaunchDimensions(const Dim3D& block_counts,
                   const Dim3D& thread_counts_per_block)
      : block_counts_(block_counts),
        thread_counts_per_block_(thread_counts_per_block) {}

  Dim3D block_counts() const { return block_counts_; }

  Dim3D thread_counts_per_block() const { return thread_counts_per_block_; }

  // Returns the total number of threads in a block.
  int64_t total_nb_threads() const {
    return thread_counts_per_block_.x * thread_counts_per_block_.y *
           thread_counts_per_block_.z;
  }

  int64_t launch_bound() const {
    return block_counts_.x * thread_counts_per_block_.x * block_counts_.y *
           thread_counts_per_block_.y * block_counts_.z *
           thread_counts_per_block_.z;
  }

  std::string ToString() const {
    return absl::StrCat("blocks: {", block_counts_.x, ", ", block_counts_.y,
                        ", ", block_counts_.z, "}, threads/block: {",
                        thread_counts_per_block_.x, ", ",
                        thread_counts_per_block_.y, ", ",
                        thread_counts_per_block_.z, "}");
  }

  bool operator==(const LaunchDimensions& other) const {
    return block_counts_ == other.block_counts_ &&
           thread_counts_per_block_ == other.thread_counts_per_block_;
  }

  bool operator!=(const LaunchDimensions& other) const {
    return !(*this == other);
  }

 private:
  Dim3D block_counts_;
  Dim3D thread_counts_per_block_;
};

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims);

struct LaunchDimensionsConfig {
  // The kernel implementation will be unrolled if `unroll_factor` is
  // greater than one.
  int unroll_factor = 1;
  // A wave is a group of blocks that execute at the same time on the
  // GPU. If there are more blocks then the number that can run
  // concurrently, there are multiple waves of blocks running
  // sequentially.  If `few_waves` is true, each thread will loop over
  // a block of unroll_factor elements. Otherwise each thread will
  // handle only unroll_factor.
  bool few_waves = false;
  // If `row_optimized` is true, then the block size will equal to
  // `hlo.shape().dimensions().back()/unroll_factor`.
  // Currently few_waves and row_vectorized do not work together.
  bool row_vectorized = false;

  std::string ToString() {
    return absl::StrCat("unroll_factor=", unroll_factor,
                        ", few_waves=", few_waves,
                        ", row_vectorized=", row_vectorized);
  }
};

// Returns -1 if the shape doesn't allow the row vectorization code path.
// If supported, return the number of threads to use in that case.
int64_t ThreadsPerBlockRowVectorized(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config);

// Calculates the launch dimensions used to invoke `hlo`.
StatusOr<LaunchDimensions> CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    LaunchDimensionsConfig dim_config = {});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
