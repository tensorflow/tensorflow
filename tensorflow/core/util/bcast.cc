/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/bcast.h"

#include "tensorflow/core/platform/logging.h"
namespace tensorflow {

/* static */
void BCast::Reverse(Vec* shape) { std::reverse(shape->begin(), shape->end()); }

BCast::BCast(const Vec& sx, const Vec& sy, const bool fewer_dims_optimization) {
  if (sx == sy && TF_PREDICT_TRUE(fewer_dims_optimization)) {
    // Fast path for common case of identical shapes for sx and sy
    int64 elements = 1;
    const int n = sx.size();
    output_.resize(n);
    for (int i = 0; i < n; i++) {
      const int64 dim = sx[i];
      elements *= dim;
      output_[i] = dim;
    }
    result_.push_back(elements);
    x_reshape_.push_back(elements);
    y_reshape_.push_back(elements);
    x_bcast_.push_back(1);
    y_bcast_.push_back(1);
    // grad_x_reduce_ and grad_y_reduce_ are left as empty
  } else {
    // Reverse the shape of x and y for convenience.
    // After the reverse, 0-th is the inner-most dimension.
    Vec x = sx;
    Vec y = sy;
    Reverse(&x);
    Reverse(&y);

    // 1-extend and align x and y so that they are the same size.
    if (x.size() > y.size()) {
      y.resize(x.size(), 1);
    } else {
      x.resize(y.size(), 1);
    }

    // Going through each dimension starting from the inner-most
    // dimension, compares dimension of x and y. They are compatible if
    // they are equal or either is 1.
    enum State {
      UNKNOWN,
      SAME,
      X_ONE,
      Y_ONE,
    };
    State prev = UNKNOWN;
    const int64 n = x.size();
    for (int i = 0; i < n; ++i) {
      // Output shape.
      State curr = UNKNOWN;
      const int64 x_i = x[i];  // i-th dimension of x.
      const int64 y_i = y[i];  // i-th dimension of y.
      int64 o_i;               // i-th dimension of the output.
      int64 bx_i;              // i-th broadcast for x.
      int64 by_i;              // i-th broadcast for y.
      // Invariant:
      //   o_i = x_i * bx_i = y_i * by_i
      if (x_i == y_i) {
        // No broadcast.
        o_i = x_i;
        bx_i = 1;
        by_i = 1;
        curr = SAME;
      } else if (x_i == 1) {
        // x broadcast to y on this dimension.
        o_i = y_i;
        bx_i = y_i;
        by_i = 1;
        grad_x_reduce_idx_.push_back(n - 1 - i);
        curr = X_ONE;
      } else if (y_i == 1) {
        // y broadcast to x on this dimension.
        o_i = x_i;
        bx_i = 1;
        by_i = x_i;
        grad_y_reduce_idx_.push_back(n - 1 - i);
        curr = Y_ONE;
      } else {
        valid_ = false;
        return;
      }
      output_.push_back(o_i);
      // Reshape/broadcast.
      // Invariant:
      //  result[i] == x_reshape[i] * x_bcast[i] == y_reshape_[i] * y_bcast_[i]
      if (curr == SAME && x_i == 1) {
        // Both side are 1s.
        grad_x_reduce_idx_.push_back(n - 1 - i);
        grad_y_reduce_idx_.push_back(n - 1 - i);
        if (!TF_PREDICT_TRUE(fewer_dims_optimization)) {
          result_.push_back(o_i);
          x_reshape_.push_back(x_i);
          x_bcast_.push_back(bx_i);
          y_reshape_.push_back(y_i);
          y_bcast_.push_back(by_i);
        }
        continue;
      } else if (TF_PREDICT_TRUE(fewer_dims_optimization) && prev == curr) {
        // It is a run of the same cases(no broadcast, x broadcast to y, y
        // broadcast to x). We can reshape the input so that fewer dimensions
        // are involved in the intermediate computation.
        result_.back() *= o_i;
        x_reshape_.back() *= x_i;
        x_bcast_.back() *= bx_i;
        y_reshape_.back() *= y_i;
        y_bcast_.back() *= by_i;
      } else {
        result_.push_back(o_i);
        x_reshape_.push_back(x_i);
        x_bcast_.push_back(bx_i);
        y_reshape_.push_back(y_i);
        y_bcast_.push_back(by_i);
      }
      prev = curr;
    }

    if (result_.empty()) {
      // Can happen when both x and y are effectively scalar.
      result_.push_back(1);
      x_reshape_.push_back(1);
      x_bcast_.push_back(1);
      y_reshape_.push_back(1);
      y_bcast_.push_back(1);
    }

    // Reverse all vectors since x and y were reversed at very
    // beginning.
    Reverse(&x_reshape_);
    Reverse(&x_bcast_);
    Reverse(&y_reshape_);
    Reverse(&y_bcast_);
    Reverse(&result_);
    Reverse(&output_);
    Reverse(&grad_x_reduce_idx_);
    Reverse(&grad_y_reduce_idx_);
  }
}

BCast::Vec BCast::FromShape(const TensorShape& shape) {
  const int N = shape.dims();
  BCast::Vec ret(N);
  for (int i = 0; i < N; ++i) {
    ret[i] = shape.dim_size(i);
  }
  return ret;
}

TensorShape BCast::ToShape(const BCast::Vec& vec) {
  TensorShape shape(vec);
  return shape;
}

}  // end namespace tensorflow
