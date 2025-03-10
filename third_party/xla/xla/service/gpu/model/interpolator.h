/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_INTERPOLATOR_H_
#define XLA_SERVICE_GPU_MODEL_INTERPOLATOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"

namespace xla::gpu {

template <typename R, size_t N>
class InterpolatorBase {
 public:
  InterpolatorBase() = default;

  virtual ~InterpolatorBase() = default;

  // Adds point to the interpolation space.
  void Add(std::array<int64_t, N>& point, R val) {
    plane_.emplace_back(point, val);
  };

  // Returns interpolated value.
  virtual R Eval(std::array<int64_t, N>& point) = 0;

 protected:
  std::vector<std::pair<std::array<int64_t, N>, R>> plane_;
};

// `Interpolates` any point in euclidean space by just returning the nearest
// neighbour, measured by euclidean distance.
// It's not very efficient as it runs the linear algorithm. A possible extension
// is to make it aware of the n-dimensional grid properties (like a constant
// distance per dimension between neighbouring points) which in turn can make
// shave off a bunch of time complexity.
// TODO: Speed up NN retrieval if it happens to be a compilation bottleneck (by
// rounding, k-d trees etc).
template <typename R, size_t N>
class EuclideanNNInterpolator : public InterpolatorBase<R, N> {
 public:
  R Eval(std::array<int64_t, N>& point) override {
    CHECK_GT(this->plane_.size(), 0);

    R result;
    uint64_t min_dist = std::numeric_limits<uint64_t>::max();

    for (const auto& [plane_point, val] : this->plane_) {
      int64_t dist = Norm2(plane_point, point);
      if (dist < min_dist) {
        result = val;
        min_dist = dist;
      }
    }
    return result;
  }

 private:
  int64_t Norm2(const std::array<int64_t, N>& lhs,
                const std::array<int64_t, N>& rhs) {
    int64_t dist = 0;
    for (int i = 0; i < lhs.size(); ++i) {
      int coord = lhs[i];
      int64_t abs_dist = coord - rhs[i];
      dist += abs_dist * abs_dist;
    }
    return dist;
  }
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_INTERPOLATOR_H_
