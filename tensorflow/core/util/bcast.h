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

#ifndef TENSORFLOW_CORE_UTIL_BCAST_H_
#define TENSORFLOW_CORE_UTIL_BCAST_H_

#include <algorithm>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// BCast is a helper for broadcasting binary tensor operation.
// TensorFlow's broadcasting rule follows that of numpy (See
// http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
//
// The rule has the following properties:
//
//   1. suffix matching: the rule starts with the right-most
//      dimension, and works towards the left-most dimension. Since
//      TensorFlow is row-major, the right-most dimension (the last
//      element in the shape of a tensor) is the inner-most, a.k.a.
//      the fastest changing, dimension.
//
//   2. Two dimensions are compatible for broadcasting if both are the
//      same or either is 1.
//
// BCast takes the shape of two tensors and computes a few vectors of
// int32 that are useful for the caller to reshape the tensors, apply
// the right broadcasts to them, compute the broadcasted operation,
// and possibly the gradients. In a nutshell, the caller is expected
// to compute the broadcasted operation as following:
//
//   BCast b(x.shape(), y.shape());
//   output = x.reshape(b.x_reshape()).broadcast(b.x_bcast())
//            _op_
//            y.reshape(b.y_reshape()).broadcast(b.y_bcast())
//
// For the gradient computation,
//   grad_x = sum(grad * backprop_x(x, y), grad_x_reduce_idx)
//            .reshape(x.shape())
//   grad_y = sum(grad * backprop_y(x, y), grad_y_reduce_idx)
//            .reshape(y.shape())
// backprop_x and backprop_y are functionals of the binary function "op",
// e.g.,
//   for +, backprop_x(x, y) = backprop_y(x, y) = 1;
//   for *, backprop_x(x, y) =  y, backprop_y(x, y) = x;
//   for /, backprop_x(x, y) = 1/y, backprop_y(x, y) = -x/y^2;
//
// The multiplication in the grad * backprop_x itself is also
// broadcasting following the same rule.
//
// TODO(zhifengc): Adds support for n-ary (n >= 2).
class BCast {
 public:
  // A vector of int64 representing the shape of tensor. The 0-th
  // element is the outer-most dimension and the last element is the
  // inner-most dimension. Note that we do not use TensorShape since
  // it's more convenient to manipulate Vec directly for this module.
  typedef gtl::InlinedVector<int64, 4> Vec;

  // Constructs all helper shapes, following the aforementioned rules.
  //
  // If "fewer_dims_optimization" is set to true (the default), the
  // implementation tries to reduce intermediate dimensions needed to be more
  // efficient.  This is transparent to the caller.
  //
  // If false, all intermediate shapes (except for grad_{x,y}_reduce_idx()) have
  // the same number of dimensions as the larger of the two inputs.
  BCast(const Vec& x, const Vec& y, const bool fewer_dims_optimization = true);
  ~BCast() {}

  // Returns true iff two operands are compatible according to the
  // broadcasting rule.
  bool IsValid() const { return valid_; }

  // If and only if IsValid(), the following fields can be used in
  // implementing a broadcasted binary tensor operation according to
  // the broadcasting rule.
  const Vec& x_reshape() const { return x_reshape_; }
  const Vec& x_bcast() const { return x_bcast_; }
  const Vec& y_reshape() const { return y_reshape_; }
  const Vec& y_bcast() const { return y_bcast_; }
  const Vec& result_shape() const { return result_; }
  const Vec& output_shape() const { return output_; }
  const Vec& grad_x_reduce_idx() const { return grad_x_reduce_idx_; }
  const Vec& grad_y_reduce_idx() const { return grad_y_reduce_idx_; }

  // Static helpers.
  static Vec FromShape(const TensorShape& shape);
  static TensorShape ToShape(const BCast::Vec& vec);

  template <int NDIMS>
  static Eigen::array<Eigen::DenseIndex, NDIMS> ToIndexArray(
      const BCast::Vec& vec) {
    CHECK_EQ(vec.size(), NDIMS);
    Eigen::array<Eigen::DenseIndex, NDIMS> ret;
    for (int i = 0; i < NDIMS; ++i) ret[i] = vec[i];
    return ret;
  }

 private:
  bool valid_ = true;
  Vec x_reshape_;
  Vec x_bcast_;
  Vec y_reshape_;
  Vec y_bcast_;
  Vec result_;
  Vec output_;
  Vec grad_x_reduce_idx_;
  Vec grad_y_reduce_idx_;

  static void Reverse(Vec* shape);

  TF_DISALLOW_COPY_AND_ASSIGN(BCast);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_BCAST_H_
