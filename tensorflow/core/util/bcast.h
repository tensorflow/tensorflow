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

// Returns the mapping from the output batch indices to the corresponding
// input's batch indices, given the input's "reshape" and "bcast" shapes as
// returned by the BCastList helper class. The i'th element denotes the
// (flattened) batch index of the input that must be used to compute the i'th
// batch output.
//
inline void ComputeBatchIndices(const int64_t output_batch_size,
                                const gtl::InlinedVector<int64_t, 4>& reshape,
                                const gtl::InlinedVector<int64_t, 4>& bcast,
                                std::vector<int64_t>* out_indices) {
  // Populates the mapping in out_indices. This algorithm is identical to
  // the following steps:
  //  - Reshape {0, 1, ..., input_batch_size - 1} to the input shape.
  //  - Broadcast to the output shape.
  //  - Reshape back to a flat 1D vector.
  out_indices->resize(output_batch_size);
  int64_t num_output_elements = 1;
  int64_t num_input_elements = 1;
  for (int64_t i = reshape.size() - 1; i >= 0; --i) {
    // Replicate the already populated mapping an additional (dim - 1) times.
    // If we are broadcasting, just copy the existing mapping.
    // Otherwise, add another dimension from the input shape.
    const int64_t dim = std::max(reshape[i], bcast[i]);
    const int64_t incr = bcast[i] > 1 ? 0 : num_input_elements;
    for (int64_t k = 0; k < (dim - 1) * num_output_elements; ++k) {
      (*out_indices)[num_output_elements + k] = (*out_indices)[k] + incr;
    }
    num_output_elements *= dim;
    num_input_elements *= reshape[i];
  }
}

template <int N>
class BCastList {
 public:
  // A vector of int64 representing the shape of tensor. The 0-th
  // element is the outer-most dimension and the last element is the
  // inner-most dimension. Note that we do not use TensorShape since
  // it's more convenient to manipulate Vec directly for this module.
  typedef gtl::InlinedVector<int64_t, 4> Vec;

  // Constructs all helper shapes, following the aforementioned rules.
  //
  // If "fewer_dims_optimization" is set to true (the default), the
  // implementation tries to reduce intermediate dimensions needed to be more
  // efficient.  This is transparent to the caller.
  //
  // If false, all intermediate shapes (except for grad_{x,y}_reduce_idx()) have
  // the same number of dimensions as the larger of the two inputs.
  //
  // If return_flattened_batch_indices is true, the implementation will compute
  // for each output member of the flattened output, which batch indices of
  // each input correspond to it. This is disabled by default.
  explicit BCastList(const Vec (&x)[N],
                     const bool fewer_dims_optimization = true,
                     const bool return_flattened_batch_indices = false);
  ~BCastList() {}

  // Returns true iff two operands are compatible according to the
  // broadcasting rule.
  bool IsValid() const { return valid_; }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  // If and only if IsValid(), the following fields can be used in
  // implementing a broadcasted binary tensor operation according to
  // the broadcasting rule.
  const Vec& reshape(int i) const { return reshape_[i]; }
  const Vec& bcast(int i) const { return bcast_[i]; }
  const Vec& result_shape() const { return result_; }
  const Vec& output_shape() const { return output_; }
  const Vec& grad_reduce_idx(int i) const { return grad_reduce_idx_[i]; }
  const int64_t output_batch_size() const { return output_batch_size_; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& batch_indices(int i) const {
    return batch_indices_[i];
  }

 protected:
  bool valid_ = true;
  bool broadcasting_required_ = true;
  Vec reshape_[N];
  Vec bcast_[N];
  Vec result_;
  Vec output_;
  Vec grad_reduce_idx_[N];

  int64_t output_batch_size_;
  std::vector<int64_t> batch_indices_[N];

  static void Reverse(Vec* shape) {
    std::reverse(shape->begin(), shape->end());
  }

  TF_DISALLOW_COPY_AND_ASSIGN(BCastList);
};

template <int N>
BCastList<N>::BCastList(const BCastList::Vec (&x)[N],
                        const bool fewer_dims_optimization,
                        const bool return_flattened_batch_indices) {
  typedef BCastList::Vec Vec;

  // Safely multiplies dimensions taking into account symbolic shapes.
  auto mul_dims = [](int64_t dim1, int64_t dim2) -> int64 {
    return dim1 != 0 && dim2 != 0 && (dim1 < 0 || dim2 < 0) ? -1 : dim1 * dim2;
  };

  bool all_equal = true;
  size_t largest_rank = 0;
  output_batch_size_ = 1;
  for (int i = 0; i < N; ++i) {
    if (x[i] != x[0]) {
      all_equal = false;
    }
    if (x[i].size() > largest_rank) {
      largest_rank = x[i].size();
    }
  }
  if (all_equal) {
    broadcasting_required_ = false;
  }
  if (all_equal && TF_PREDICT_TRUE(fewer_dims_optimization)) {
    // Fast path for common case of identical shapes.
    int64_t elements = 1;
    const int rank = x[0].size();
    output_.resize(rank);
    for (int i = 0; i < rank; i++) {
      const int64_t dim = x[0][i];
      elements = mul_dims(elements, dim);
      output_[i] = dim;
    }
    result_.push_back(elements);
    output_batch_size_ = elements;
    for (int i = 0; i < N; ++i) {
      reshape_[i].push_back(elements);
      bcast_[i].push_back(1);
    }
    // grad_reduce_ is left as empty
    return;
  }

  // Reverse all the shapes for convenience
  // After the reverse, 0-th is the inner-most dimension.
  Vec copy[N];
  for (int i = 0; i < N; ++i) {
    copy[i] = x[i];
    Reverse(&copy[i]);
  }

  // 1-extend and align all vectors.
  for (int i = 0; i < N; ++i) {
    if (copy[i].size() < largest_rank) {
      copy[i].resize(largest_rank, 1);
    }
  }
  // Going through each dimension starting from the inner-most
  // dimension, compares dimension of x and y. They are compatible if
  // they are equal or either is 1.

  // indices of j-th component of each input.
  bool prev_is_one[N];
  bool current_is_one[N];
  for (int i = 0; i < N; ++i) {
    prev_is_one[i] = false;
    current_is_one[i] = false;
  }
  Vec output;
  bool output_dim_set = false;
  int output_dim = -1;
  bool none_is_one = true;
  bool set_one = false;
  for (int j = 0; j < largest_rank; ++j) {
    output_dim = -1;
    output_dim_set = false;
    none_is_one = true;
    // Find which indices are 1.
    for (int i = 0; i < N; ++i) {
      // Keep track of which indices are 1.
      if (copy[i][j] == 1) {
        current_is_one[i] = true;
        none_is_one = false;
      } else {
        current_is_one[i] = false;
        if (!output_dim_set || copy[i][j] == output_dim) {
          output_dim = copy[i][j];
          output_dim_set = true;
        } else {
          valid_ = false;
          return;
        }
      }
    }
    output_.push_back(output_dim_set ? output_dim : 1);
    output_batch_size_ = mul_dims(output_batch_size_, output_.back());
    // All dimensions are 1.
    if (!output_dim_set) {
      if (!TF_PREDICT_TRUE(fewer_dims_optimization)) {
        for (int i = 0; i < N; ++i) {
          bcast_[i].push_back(1);
          reshape_[i].push_back(1);
        }
        result_.push_back(1);
      }
      for (int i = 0; i < N; ++i) {
        grad_reduce_idx_[i].push_back(largest_rank - 1 - j);
      }
      // This will skip updating the previous state to the current one. We'll
      // explain why this is safe below.
      // Consider the previous state P, current state C and the next state N.
      // In the case where N also is all ones (N == C), we'll do the same
      // optimization here (push back one dimensions if we need to), which is
      // safe and is expected.
      //
      // When N != C, we'll continue as usual. However, we might trigger the
      // next block if N == P (because we didn't update the previous state).
      // We trigger the next block if `fewer_dims_optimization` is true.
      // This means that we did not modify and broadcast / reshapes in this
      // block (we skipped updating, since the one dimensions can be ignored).
      // In essence, we only need to check whether the previous non-one state is
      // equal to the current non-one state.

      continue;
    } else if (TF_PREDICT_TRUE(fewer_dims_optimization) &&
               std::equal(current_is_one, current_is_one + N, prev_is_one) &&
               set_one) {
      // It is a run of the same broadcasting case as last time.
      // We can reshape the input so that fewer dimensions
      // are involved in the intermediate computation.
      result_.back() = mul_dims(result_.back(), output_dim);
      for (int i = 0; i < N; ++i) {
        reshape_[i].back() = mul_dims(reshape_[i].back(), copy[i][j]);
        bcast_[i].back() =
            mul_dims(bcast_[i].back(), current_is_one[i] ? output_dim : 1);
        if (current_is_one[i] && !none_is_one) {
          grad_reduce_idx_[i].push_back(largest_rank - 1 - j);
        }
      }
    } else {
      result_.push_back(output_dim);
      for (int i = 0; i < N; ++i) {
        reshape_[i].push_back(copy[i][j]);
        bcast_[i].push_back(current_is_one[i] ? output_dim : 1);
        if (current_is_one[i] && !none_is_one) {
          grad_reduce_idx_[i].push_back(largest_rank - 1 - j);
        }
      }
    }
    set_one = true;
    for (int i = 0; i < N; ++i) {
      prev_is_one[i] = current_is_one[i];
    }
  }
  if (result_.empty()) {
    result_.push_back(1);
    for (int i = 0; i < N; ++i) {
      reshape_[i].push_back(1);
      bcast_[i].push_back(1);
    }
  }
  // Do something about batches.
  for (int i = 0; i < N; ++i) {
    Reverse(&reshape_[i]);
    Reverse(&bcast_[i]);
    Reverse(&grad_reduce_idx_[i]);
  }
  Reverse(&result_);
  Reverse(&output_);
  // Only compute batch indices when we need broadcasting, and we aren't doing
  // needless work (when the output size is 0 or the
  // return_flattened_batch_indices isn't enabled).
  if (return_flattened_batch_indices && broadcasting_required_ &&
      output_batch_size_ > 0) {
    for (int i = 0; i < N; ++i) {
      ComputeBatchIndices(output_batch_size_, reshape_[i], bcast_[i],
                          &batch_indices_[i]);
    }
  }
}

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
class BCast : public BCastList<2> {
 public:
  // Constructs all helper shapes, following the aforementioned rules.
  //
  // If "fewer_dims_optimization" is set to true (the default), the
  // implementation tries to reduce intermediate dimensions needed to be more
  // efficient.  This is transparent to the caller.
  //
  // If false, all intermediate shapes (except for grad_{x,y}_reduce_idx()) have
  // the same number of dimensions as the larger of the two inputs.
  typedef gtl::InlinedVector<int64_t, 4> Vec;

  BCast(const Vec& x, const Vec& y, const bool fewer_dims_optimization = true,
        const bool return_flattened_batch_indices = false)
      : BCastList<2>({x, y}, fewer_dims_optimization,
                     return_flattened_batch_indices) {}

  ~BCast() {}

  // If and only if IsValid(), the following fields can be used in
  // implementing a broadcasted binary tensor operation according to
  // the broadcasting rule.
  const Vec& x_reshape() const { return reshape_[0]; }
  const Vec& x_bcast() const { return bcast_[0]; }
  const Vec& y_reshape() const { return reshape_[1]; }
  const Vec& y_bcast() const { return bcast_[1]; }
  const Vec& result_shape() const { return result_; }
  const Vec& output_shape() const { return output_; }
  const Vec& grad_x_reduce_idx() const { return grad_reduce_idx_[0]; }
  const Vec& grad_y_reduce_idx() const { return grad_reduce_idx_[1]; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& x_batch_indices() const {
    return batch_indices_[0];
  }
  // Returns the mapping from the flattened output batch indices to y's
  // flattened batch indices. Similar to x_batch_indices().
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64_t>& y_batch_indices() const {
    return batch_indices_[1];
  }

  template <typename IndexType, int NDIMS>
  static Eigen::array<IndexType, NDIMS> ToIndexArrayType(
      const BCast::Vec& vec) {
    CHECK_EQ(vec.size(), NDIMS);
    Eigen::array<IndexType, NDIMS> ret;
    for (int i = 0; i < NDIMS; ++i) ret[i] = vec[i];
    return ret;
  }

  template <int NDIMS>
  static Eigen::array<Eigen::DenseIndex, NDIMS> ToIndexArray(
      const BCast::Vec& vec) {
    return ToIndexArrayType<Eigen::DenseIndex, NDIMS>(vec);
  }

  // Static helpers.
  static Vec FromShape(const TensorShape& shape);
  static TensorShape ToShape(const Vec& vec);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BCast);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_BCAST_H_
