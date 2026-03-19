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
#include <functional>
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
  virtual void Add(std::array<int64_t, N>& point, R val) = 0;

  // Returns interpolated value.
  virtual R Eval(std::array<int64_t, N>& point) const = 0;

  static int64_t Norm2(const std::array<int64_t, N>& lhs,
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

// `Interpolates` any point in euclidean space by just returning the nearest
// neighbour, measured by euclidean distance.
// It's not very efficient as it runs the linear algorithm. A possible extension
// is to make it aware of the n-dimensional grid properties (like a constant
// distance per dimension between neighbouring points) which in turn can make
// shave off a bunch of time complexity.
template <typename R, size_t N>
class EuclideanNNInterpolator : public InterpolatorBase<R, N> {
 public:
  void Add(std::array<int64_t, N>& point, R val) override {
    plane_.emplace_back(point, val);
  };

  R Eval(std::array<int64_t, N>& point) const override {
    CHECK_GT(plane_.size(), 0);

    R result;
    uint64_t min_dist = std::numeric_limits<uint64_t>::max();

    for (const auto& [plane_point, val] : plane_) {
      int64_t dist = InterpolatorBase<R, N>::Norm2(plane_point, point);
      if (dist < min_dist) {
        result = val;
        min_dist = dist;
      }
    }
    return result;
  }

 private:
  std::vector<std::pair<std::array<int64_t, N>, R>> plane_;
};

// `EuclideanComplementInterpolator` takes `next_context`, `next_power_context`,
// `max_context` and `min_context` and then lookups out the closes neighbour in
// N-dimensional euclidean space. The additional input context is necessary for
// fast interpolation.
//
// The constructor API is as follows: `next_context` just specifies the next
// potential dimension for each dimension. `next_power_context` specifies the
// next power of two dimension for each dimension. `max_context` and
// `min_context` specify the maximum and minimum value for each dimension.
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

  void Add(std::array<int64_t, N>& point, R val) override {
    retrieval_[point] = val;
  }

  R Eval(std::array<int64_t, N>& point) const override {
    CHECK_GT(retrieval_.size(), 0);
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

 protected:
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

  int64_t PrevPowerOfTwo(int64_t n) const {
    return NextPowerOfTwo((n >> 1) + 1);
  }

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

template <size_t N>
struct Neighbour {
  std::array<int64_t, N> point;
  double weight;
};

// `EuclideanWeightedAverageInterpolator` takes `next_context`,
// `next_power_context`, `max_context` and `min_context`, then lookups convex
// hull of the point of interest and returns the weighted average of the values
// at the convex hull. The additional input context is necessary for fast
// nearest neighbour search.
//
// The constructor API is as follows: `next_context` just specifies the next
// potential dimension for each dimension. `next_power_context` specifies the
// next power of two dimension for each dimension. `max_context` and
// `min_context` specify the maximum and minimum value for each dimension.
template <size_t N>
class EuclideanWeightedAverageInterpolator
    : public EuclideanComplementInterpolator<double, N> {
 public:
  explicit EuclideanWeightedAverageInterpolator(
      std::array<int64_t, N> next_context,
      std::array<int64_t, N> next_power_context,
      std::array<int64_t, N> max_context, std::array<int64_t, N> min_context)
      : EuclideanComplementInterpolator<double, N>(
            next_context, next_power_context, max_context, min_context) {}

  double Eval(std::array<int64_t, N>& point) const override {
    CHECK_GT(this->retrieval_.size(), 0) << "Retrieval map is empty.";
    double result = 0;
    double total_weight = 0.0f;

    for (const Neighbour<N>& neighbour : GetNeighbours(point)) {
      result += this->retrieval_.at(neighbour.point) * neighbour.weight;
      total_weight += neighbour.weight;
    }
    return result / total_weight;
  }

 private:
  int64_t ClampDim(int64_t val, int64_t dim) const {
    return std::min(std::max(this->min_ctx_[dim], val), this->max_ctx_[dim]);
  }

  int64_t SmallerNeighbour(std::array<int64_t, N> point, int64_t dim) const {
    int64_t neighbour_dim = -1;
    if (this->retrieval_ctx_[dim] != -1) {
      int64_t next = this->retrieval_ctx_[dim];
      neighbour_dim = ClampDim(this->PrevComplement(point[dim], next), dim);
    }
    if (this->retrieval_pow_ctx_[dim] != -1) {
      neighbour_dim = ClampDim(this->PrevPowerOfTwo(point[dim]), dim);
    }
    return neighbour_dim;
  }

  int64_t LargerNeighbour(std::array<int64_t, N> point, int64_t dim) const {
    int64_t neighbour_dim = -1;
    if (this->retrieval_ctx_[dim] != -1) {
      int64_t next = this->retrieval_ctx_[dim];
      neighbour_dim = ClampDim(this->NextComplement(point[dim], next), dim);
    }
    if (this->retrieval_pow_ctx_[dim] != -1) {
      neighbour_dim = ClampDim(this->NextPowerOfTwo(point[dim]), dim);
    }
    return neighbour_dim;
  }

  std::vector<Neighbour<N>> GetNeighbours(std::array<int64_t, N>& point) const {
    static constexpr float kEpsilon = 1.0;

    std::function<std::vector<std::array<int64_t, N>>(int)> convex_hull =
        [&](int dim) -> std::vector<std::array<int64_t, N>> {
      std::vector<std::array<int64_t, N>> result;
      if (dim == point.size() - 1) {
        std::array<int64_t, N> min, max;
        min[dim] = SmallerNeighbour(point, dim);
        max[dim] = LargerNeighbour(point, dim);
        return {min, max};
      }

      std::vector<std::array<int64_t, N>> intermediete_results =
          convex_hull(dim + 1);

      for (const std::array<int64_t, N>& pt : intermediete_results) {
        std::array<int64_t, N> min_point = pt, max_point = pt;
        min_point[dim] = SmallerNeighbour(point, dim);
        max_point[dim] = LargerNeighbour(point, dim);
        result.push_back(min_point);
        result.push_back(max_point);
      }
      return result;
    };
    std::vector<Neighbour<N>> neighbours;
    for (const std::array<int64_t, N> neighbour : convex_hull(/*dim=*/0)) {
      float weight =
          1.0f /
          (InterpolatorBase<double, N>::Norm2(neighbour, point) + kEpsilon);
      Neighbour<N> n;
      n.point = neighbour;
      n.weight = weight;
      neighbours.push_back(n);
    }
    return neighbours;
  }
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_INTERPOLATOR_H_
