// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_STATS_GRADIENT_STATS_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_STATS_GRADIENT_STATS_H_

#include <math.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

const double kEps = 1e-6;

// A data structure for accumulating a Tensor value.
struct TensorStat {
  TensorStat() {}

  explicit TensorStat(const float v) : t(DT_FLOAT, TensorShape({1})) {
    t.flat<float>()(0) = v;
  }

  explicit TensorStat(const Tensor& rt) : t(tensor::DeepCopy(rt)) {}

  TensorStat(const TensorStat& ts) : t(tensor::DeepCopy(ts.t)) {}

  TensorStat& operator+=(const TensorStat& other) {
    if (t.NumElements() == 0) {
      t = tensor::DeepCopy(other.t);
      return (*this);
    }
    CHECK(t.shape() == other.t.shape())
        << "My shape = " << t.shape().DebugString()
        << " Other shape = " << other.t.shape().DebugString();
    auto me_flat = t.unaligned_flat<float>();
    auto other_flat = other.t.unaligned_flat<float>();
    for (int i = 0; i < me_flat.size(); i++) {
      me_flat(i) += other_flat(i);
    }
    return (*this);
  }

  TensorStat& operator-=(const TensorStat& other) {
    if (other.t.NumElements() == 0) {
      return (*this);
    }
    CHECK(t.shape() == other.t.shape())
        << "My shape = " << t.shape().DebugString()
        << " Other shape = " << other.t.shape().DebugString();
    auto me_flat = t.unaligned_flat<float>();
    auto other_flat = other.t.unaligned_flat<float>();
    for (int i = 0; i < me_flat.size(); i++) {
      me_flat(i) -= other_flat(i);
    }
    return (*this);
  }

  TensorStat& operator*=(float value) {
    auto me_flat = t.unaligned_flat<float>();
    for (size_t i = 0; i < me_flat.size(); i++) {
      me_flat(i) *= value;
    }
    return (*this);
  }

  bool IsZero() const {
    auto me_flat = t.unaligned_flat<float>();
    for (int i = 0; i < me_flat.size(); i++) {
      if (me_flat(i) != 0.0f) {
        return false;
      }
    }
    return true;
  }

  // Checks if the L^2 magnitude of the tensor is less than eps.
  bool IsAlmostZero(const float eps = kEps) const {
    auto me_flat = t.unaligned_flat<float>();
    double s = 0.0;
    for (int i = 0; i < me_flat.size(); i++) {
      s += me_flat(i) * me_flat(i);
      if (s > eps * eps) {
        return false;
      }
    }
    return true;
  }

  float Magnitude() const {
    auto me_flat = t.unaligned_flat<float>();
    double s = 0.0;
    for (int i = 0; i < me_flat.size(); i++) {
      s += me_flat(i) * me_flat(i);
    }
    return sqrt(s);
  }

  string DebugString() const { return t.DebugString(); }

  Tensor t;
};

// GradientStats holds first and second order gradient stats.
struct GradientStats {
  GradientStats() {}

  // Legacy constructor for tests
  GradientStats(float g, float h) : first(g), second(h) {}

  GradientStats(const Tensor& g, const Tensor& h) : first(g), second(h) {}

  GradientStats(const Tensor& g, const Tensor& h, int64 example_index)
      : first(g.Slice(example_index, example_index + 1)),
        second(h.Slice(example_index, example_index + 1)) {}

  GradientStats& operator+=(const GradientStats& other) {
    first += other.first;
    second += other.second;
    return (*this);
  }

  GradientStats& operator*=(float value) {
    first *= value;
    second *= value;
    return (*this);
  }

  GradientStats& operator-=(const GradientStats& other) {
    first -= other.first;
    second -= other.second;
    return (*this);
  }

  bool IsZero() const { return first.IsZero() && second.IsZero(); }

  bool IsAlmostZero(const float eps = kEps) const {
    return first.IsAlmostZero(eps) && second.IsAlmostZero(eps);
  }

  float Magnitude() const { return second.Magnitude(); }

  string DebugString() const {
    return "First = " + first.DebugString() +
           " Second = " + second.DebugString();
  }

  TensorStat first;
  TensorStat second;
};

struct GradientStatsAccumulator {
  void operator()(const GradientStats& from, GradientStats* to) const {
    (*to) += from;
  }
};

inline GradientStats operator+(const GradientStats& a, const GradientStats& b) {
  GradientStats ret(a);
  ret += b;
  return ret;
}

inline GradientStats operator-(const GradientStats& a, const GradientStats& b) {
  GradientStats ret(a);
  ret -= b;
  return ret;
}

// Helper macro to check gradient stats approximate equality.
#define EXPECT_GRADIENT_STATS_EQ(val1, val2) \
  EXPECT_TRUE((val1 - val2).IsAlmostZero());

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_STATS_GRADIENT_STATS_H_
