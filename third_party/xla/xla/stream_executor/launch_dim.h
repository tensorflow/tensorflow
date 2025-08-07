/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_LAUNCH_DIM_H_
#define XLA_STREAM_EXECUTOR_LAUNCH_DIM_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/launch_dim.pb.h"

namespace stream_executor {

struct Dim3D {
  uint64_t x, y, z;

  bool operator==(const Dim3D& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator!=(const Dim3D& other) const { return !(*this == other); }

  Dim3DProto ToProto() const;
  static absl::StatusOr<Dim3D> FromProto(const Dim3DProto& proto);
};

// Types to express dimensionality of a kernel launch. Blocks, threads and
// clusters are (up to) 3-dimensional.
//
// See NVIDIA documentation for a thread hierarchy:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy

// Thread dimensionality for use in a kernel launch.
// details.
struct ThreadDim : Dim3D {
  explicit constexpr ThreadDim(uint64_t x_arg = 1, uint64_t y_arg = 1,
                               uint64_t z_arg = 1)
      : Dim3D({x_arg, y_arg, z_arg}) {}

  explicit ThreadDim(const Dim3D& other) : Dim3D(other) {}

  std::string ToString() const {
    return absl::StrCat("ThreadDim{", x, ", ", y, ", ", z, "}");
  }

  ThreadDimProto ToProto() const;
  static absl::StatusOr<ThreadDim> FromProto(const ThreadDimProto& proto);
};

// Block dimensionality for use in a kernel launch.
// details.
struct BlockDim : Dim3D {
  explicit constexpr BlockDim(uint64_t x_arg = 1, uint64_t y_arg = 1,
                              uint64_t z_arg = 1)
      : Dim3D({x_arg, y_arg, z_arg}) {}

  explicit BlockDim(const Dim3D& other) : Dim3D(other) {}

  std::string ToString() const {
    return absl::StrCat("BlockDim{", x, ", ", y, ", ", z, "}");
  }

  BlockDimProto ToProto() const;
  static absl::StatusOr<BlockDim> FromProto(const BlockDimProto& proto);
};

// Cluster dimensionality for use in a kernel launch.
struct ClusterDim : Dim3D {
  explicit ClusterDim(uint64_t x_arg = 1, uint64_t y_arg = 1,
                      uint64_t z_arg = 1)
      : Dim3D({x_arg, y_arg, z_arg}) {}

  explicit ClusterDim(const Dim3D& other) : Dim3D(other) {}

  std::string ToString() const {
    return absl::StrCat("ClusterDim{", x, ", ", y, ", ", z, "}");
  }

  ClusterDimProto ToProto() const;
  static absl::StatusOr<ClusterDim> FromProto(const ClusterDimProto& proto);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_LAUNCH_DIM_H_
