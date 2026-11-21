/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_ISOTONIC_REGRESSION_OP_H_
#define TENSORFLOW_CORE_KERNELS_ISOTONIC_REGRESSION_OP_H_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tensorflow {
namespace isotonic_regression {

// In separable chain-constrained problems, i.e., those of the form
//
//  min_{y_1 >= y_2 >= ... >= y_n} \sum_{i=1}^n h_i(y_i)
//
// for any set of convex functions h_i, contiguous segments of coordinates are
// represented by this class. The interval is half-closed [col_start, col_limit).
class Segment {
 public:
  explicit Segment(int col_index)
      : col_start_(col_index), col_limit_(col_index + 1) {}

  int num_points() const { return col_limit_ - col_start_; }

  void merge_with(const Segment& other) {
    col_start_ = std::min(col_start_, other.col_start());
    col_limit_ = std::max(col_limit_, other.col_limit());
  }

  int col_start() const { return col_start_; }
  int col_limit() const { return col_limit_; }

 private:
  int col_start_;
  int col_limit_;
};

// Implements the oracle for squared (L2) loss: h_i(y) = (y - x_i)^2.
template <typename T>
class L2PavaSegment : public Segment {
 public:
  L2PavaSegment(T y, int col_index)
      : Segment(col_index), y_sum_(y), minimum_(y) {}

  void merge_with(const L2PavaSegment& other) {
    Segment::merge_with(other);
    y_sum_ += other.y_sum_;
    minimum_ = y_sum_ / static_cast<T>(num_points());
  }

  T minimum() const { return minimum_; }

 private:
  T y_sum_;
  T minimum_;
};

// Solves a single 1D isotonic regression problem of length `n` using PAVA,
// reusing the provided `pools` vector to avoid heap allocations.
template <typename Tin, typename Tout>
void SolvePava1D(const Tin* input, Tout* solution, int32_t* segments,
                 int64_t n, std::vector<L2PavaSegment<Tout>>* pools) {
  if (n <= 0) return;
  pools->clear();
  pools->reserve(n);

  for (int64_t col_index = 0; col_index < n; ++col_index) {
    pools->push_back(
        L2PavaSegment<Tout>(static_cast<Tout>(input[col_index]), col_index));

    // While the last two pools violate the monotonic constraint, merge them.
    while (pools->size() > 1 &&
           pools->rbegin()->minimum() > (pools->rbegin() + 1)->minimum()) {
      (pools->rbegin() + 1)->merge_with(*pools->rbegin());
      pools->pop_back();
    }
  }

  int32_t segment_id = 0;
  for (const auto& pool : *pools) {
    const auto pool_minimum = pool.minimum();
    Tout* solution_ptr = solution + pool.col_start();
    int32_t* segments_ptr = segments + pool.col_start();
    for (int i = pool.col_start(); i < pool.col_limit(); ++i) {
      *solution_ptr++ = pool_minimum;
      *segments_ptr++ = segment_id;
    }
    ++segment_id;
  }
}

// Overload for single-call usage that manages its own `pools` vector.
template <typename Tin, typename Tout>
void SolvePava1D(const Tin* input, Tout* solution, int32_t* segments,
                 int64_t n) {
  std::vector<L2PavaSegment<Tout>> pools;
  SolvePava1D<Tin, Tout>(input, solution, segments, n, &pools);
}

// Solves a batch of 1D isotonic regression problems, each of length `n`.
// Reuses a single `pools` vector across the entire batch to eliminate
// per-element heap allocations.
template <typename Tin, typename Tout>
void SolvePavaBatch(const Tin* input, Tout* solution, int32_t* segments,
                    int64_t batch_size, int64_t n) {
  std::vector<L2PavaSegment<Tout>> pools;
  pools.reserve(n);
  for (int64_t b = 0; b < batch_size; ++b) {
    SolvePava1D<Tin, Tout>(input + b * n, solution + b * n,
                           segments + b * n, n, &pools);
  }
}

}  // namespace isotonic_regression
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ISOTONIC_REGRESSION_OP_H_

