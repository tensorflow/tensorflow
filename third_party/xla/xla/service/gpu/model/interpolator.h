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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_join.h"

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
  virtual R Eval(std::array<int64_t, N>& point) const = 0;

 protected:
  std::vector<std::pair<std::array<int64_t, N>, R>> plane_;
};

// `Interpolates` any point in euclidean space by just returning the nearest
// neighbour, measured by euclidean distance.
// It's not very efficient as it runs the linear algorithm. A possible extension
// is to make it aware of the n-dimensional grid properties (like a constant
// distance per dimension between neighbouring points) which in turn can make
// shave off a bunch of time complexity.
template <typename R, size_t N>
class EuclideanNNInterpolator : public InterpolatorBase<R, N> {
 public:
  R Eval(std::array<int64_t, N>& point) const override {
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
                const std::array<int64_t, N>& rhs) const {
    int64_t dist = 0;
    for (int i = 0; i < lhs.size(); ++i) {
      int coord = lhs[i];
      int64_t abs_dist = coord - rhs[i];
      dist += abs_dist * abs_dist;
    }
    return dist;
  }
};

template <typename R, size_t N>
class EuclideanComplementInterpolator : public EuclideanNNInterpolator<R, N> {
 public:
  explicit EuclideanComplementInterpolator(
      std::array<int64_t, N> next_context,
      std::array<int64_t, N> next_power_context,
      std::array<int64_t, N> max_context, std::array<int64_t, N> min_context)
      : retrieval_ctx_(next_context),
        retrieval_pow_ctx_(next_power_context),
        max_ctx_(max_context),
        min_ctx_(min_context) {}

  void Add(std::array<int64_t, N>& point, R val) {
    EuclideanNNInterpolator<R, N>::Add(point, val);
    retrieval_[point] = val;
  }

  R Eval(std::array<int64_t, N>& point) const override {
    CHECK_GT(this->plane_.size(), 0);
    std::array<int64_t, N> interpolation_point;
    for (int i = 0; i < point.size(); ++i) {
      std::optional<int64_t> next_potential_dim;
      if (retrieval_ctx_[i] != -1) {
        int64_t next = retrieval_ctx_[i];
        next_potential_dim = Closest(point[i], PrevComplement(point[i], next),
                                     NextComplement(point[i], next));
      }
      if (retrieval_pow_ctx_[i] != -1) {
        next_potential_dim = Closest(point[i], PrevPowerOfTwo(point[i]),
                                     NextPowerOfTwo(point[i]));
      }
      CHECK(next_potential_dim.has_value());
      interpolation_point[i] =
          std::max(std::min(*next_potential_dim, max_ctx_[i]), min_ctx_[i]);
    }
    return retrieval_.at(interpolation_point);
  }

 private:
  int64_t Closest(int64_t n, int64_t prev, int64_t next) const {
    if (n - prev < next - n) {
      return prev;
    }
    return next;
  }

  int64_t NextComplement(int64_t n, int64_t complement) const {
    return (n + complement) & ~(complement - 1);
  }

  int64_t PrevComplement(int64_t n, int64_t complement) const {
    return n & ~(complement - 1);
  }

  bool IsPowerOfTwo(int n) {
    if (n <= 0) {
      return false;
    }
    return (n & (n - 1)) == 0;
  }

  int64_t PrevPowerOfTwo(int64_t n) const { return NextPowerOfTwo(n << 1); }

  int64_t NextPowerOfTwo(int64_t n) const {
    if (n == 0) {
      return 1;
    }
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
  }

  std::string PointStr(std::array<int64_t, N> point) const {
    return absl::StrJoin(point, ", ");
  }

  std::array<int64_t, N> retrieval_ctx_;
  std::array<int64_t, N> retrieval_pow_ctx_;
  std::array<int64_t, N> max_ctx_;
  std::array<int64_t, N> min_ctx_;

  absl::flat_hash_map<std::array<int64_t, N>, R> retrieval_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_INTERPOLATOR_H_
