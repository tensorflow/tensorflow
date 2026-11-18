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
#include <optional>
#include <string>
#include <tuple>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/gpu/launch_dimensions.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {
namespace gpu {

// Encapsulates the launch dimensions of a kernel, e.g., the block count, the
// number of threads per block, and optional cluster dimensions.
class LaunchDimensions {
 public:
  // The default constructor creates a launch dimension that indicate
  // single-threaded execution.
  constexpr LaunchDimensions() = default;

  constexpr LaunchDimensions(uint64_t block_x_count,
                             uint64_t thread_x_count_per_block)
      : block_counts_(block_x_count, 1, 1),
        thread_counts_per_block_(thread_x_count_per_block, 1, 1) {}

  constexpr LaunchDimensions(
      const se::BlockDim& block_counts,
      const se::ThreadDim& thread_counts_per_block,
      std::optional<se::ClusterDim> cluster_dims = std::nullopt)
      : block_counts_(block_counts),
        thread_counts_per_block_(thread_counts_per_block),
        cluster_dims_(cluster_dims) {}

  constexpr const se::BlockDim& block_counts() const { return block_counts_; }

  constexpr const se::ThreadDim& thread_counts_per_block() const {
    return thread_counts_per_block_;
  }

  constexpr const std::optional<se::ClusterDim>& cluster_dims() const {
    return cluster_dims_;
  }

  // Returns the total number of blocks.
  constexpr uint64_t num_blocks() const {
    return block_counts_.x * block_counts_.y * block_counts_.z;
  }

  // Returns the total number of threads in a block.
  constexpr uint64_t num_threads_per_block() const {
    return thread_counts_per_block_.x * thread_counts_per_block_.y *
           thread_counts_per_block_.z;
  }

  constexpr uint64_t launch_bound() const {
    return num_blocks() * num_threads_per_block();
  }

  std::string ToString() const {
    std::string str = absl::StrCat(
        "blocks: {", block_counts_.x, ", ", block_counts_.y, ", ",
        block_counts_.z, "}, threads/block: {", thread_counts_per_block_.x,
        ", ", thread_counts_per_block_.y, ", ", thread_counts_per_block_.z,
        "}");
    if (cluster_dims_.has_value()) {
      absl::StrAppend(&str, ", cluster: {", cluster_dims_->x, ", ",
                      cluster_dims_->y, ", ", cluster_dims_->z, "}");
    }
    return str;
  }

  static absl::StatusOr<LaunchDimensions> FromWorkDimensions(
      const WorkDimensions& work_dimensions);
  WorkDimensions AsWorkDimensions() const;

  LaunchDimensionsProto ToProto() const;
  static absl::StatusOr<LaunchDimensions> FromProto(
      const LaunchDimensionsProto& proto);

  friend constexpr bool operator==(const LaunchDimensions& lhs,
                                   const LaunchDimensions& rhs) {
    return std::tie(lhs.block_counts_, lhs.thread_counts_per_block_,
                    lhs.cluster_dims_) == std::tie(rhs.block_counts_,
                                                   rhs.thread_counts_per_block_,
                                                   rhs.cluster_dims_);
  }

  friend constexpr bool operator!=(const LaunchDimensions& lhs,
                                   const LaunchDimensions& rhs) {
    return !(lhs == rhs);
  }

 private:
  se::BlockDim block_counts_;
  se::ThreadDim thread_counts_per_block_;
  std::optional<se::ClusterDim> cluster_dims_;
};

// Returns -1 if the shape doesn't allow the row vectorization code path.
// If supported, return the number of threads to use in that case.
int64_t ThreadsPerBlockRowVectorized(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    int unroll_factor);

// Calculates the launch dimensions used to invoke `hlo`.
LaunchDimensions CalculateLaunchDimensions(
    const Shape& shape, const se::DeviceDescription& gpu_device_info,
    int unroll_factor = 1);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_LAUNCH_DIMENSIONS_H_
