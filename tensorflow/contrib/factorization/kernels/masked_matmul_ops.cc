// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
// =============================================================================

// TensorFlow kernels and Ops for computing a masked matrix product.

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_BOOL;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT64;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::errors::InvalidArgument;

namespace tensorflow {

typedef Eigen::Map<
    Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    EigenMatInt64Map;
typedef Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    EigenMatFloatMap;
typedef Eigen::Map<
    const Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    ConstEigenMatInt64Map;
typedef Eigen::Map<
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    ConstEigenMatFloatMap;

class MaskedMatmulOp : public OpKernel {
 public:
  explicit MaskedMatmulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->MatchSignature(
        {DT_FLOAT, DT_FLOAT, DT_INT64, DT_BOOL, DT_BOOL},
        {DT_FLOAT}));
  }

  void Compute(OpKernelContext* context) override {
    // Computes the product a * b, but only for indices (i, j) in mask_indices.
    // The result is stored in prod_values, a 1-tensor, such that for all i,
    // prod_values[i] = (a * b)[mask_indices[i, 0], mask_indices[i, 1]].
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& mask_indices = context->input(2);
    const Tensor& transpose_a = context->input(3);
    const Tensor& transpose_b = context->input(4);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a.shape()),
                InvalidArgument("Input a should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(b.shape()),
                InvalidArgument("Input b should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(mask_indices.shape()),
                InvalidArgument("Input mask_indices should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(transpose_a.shape()),
                InvalidArgument("Input transpose_a should be a scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(transpose_b.shape()),
                InvalidArgument("Input transpose_b should be a scalar."));

    const bool adj_a = transpose_a.scalar<bool>()();
    const bool adj_b = transpose_b.scalar<bool>()();
    const int64 a_dim_0 = a.dim_size(adj_a ? 1 : 0);
    const int64 a_dim_1 = a.dim_size(adj_a ? 0 : 1);
    const int64 b_dim_0 = b.dim_size(adj_b ? 1 : 0);
    const int64 b_dim_1 = b.dim_size(adj_b ? 0 : 1);
    const int64 num_nonzero_elements = mask_indices.dim_size(0);

    OP_REQUIRES(context, a_dim_1 == b_dim_0,
                InvalidArgument("Matrix shapes are incompatible: a has shape ",
                                a.shape().DebugString(), ", while b has shape ",
                                b.shape().DebugString(), "."));
    OP_REQUIRES(context, mask_indices.dim_size(1) == 2,
                InvalidArgument("mask_indices should be a matrix of shape ",
                                "[nnz 2], where nnz is the number of non-zero ",
                                "elements."));

    ConstEigenMatFloatMap a_mat(a.matrix<float>().data(), a.dim_size(0),
                                a.dim_size(1));
    ConstEigenMatFloatMap b_mat(b.matrix<float>().data(), b.dim_size(0),
                                b.dim_size(1));
    ConstEigenMatInt64Map indices_mat(mask_indices.matrix<int64>().data(),
                                      num_nonzero_elements, 2);

    Tensor* prod_values_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({num_nonzero_elements}),
                       &prod_values_tensor));
    EigenMatFloatMap prod_values(prod_values_tensor->vec<float>().data(),
                                 1, num_nonzero_elements);

    auto get_a_index = [&indices_mat, &a_dim_0](int64 i) {
      int64 a_index = internal::SubtleMustCopy(indices_mat(i, 0));
      CHECK(FastBoundsCheck(a_index, a_dim_0))
          << "In mask_indices[" << i << ", :], the row index " << a_index
          << " is out of bounds [0, " << a_dim_0 << ").";
      return a_index;
    };
    auto get_b_index = [&indices_mat, &b_dim_1](int64 i) {
      int64 b_index = internal::SubtleMustCopy(indices_mat(i, 1));
      CHECK(FastBoundsCheck(b_index, b_dim_1))
          << "In mask_indices[" << i << ", :], the column index " << b_index
          << " is out of bounds [0, " << b_dim_1 << ").";
      return b_index;
    };
    auto get_dot_product = [&adj_a, &adj_b, &a_mat, &b_mat](int64 i, int64 j) {
      if (adj_a) {
        if (adj_b) {
          return a_mat.col(i).dot(b_mat.row(j));
        } else {
          return a_mat.col(i).dot(b_mat.col(j));
        }
      } else {
        if (adj_b) {
          return a_mat.row(i).dot(b_mat.row(j));
        } else {
          return a_mat.row(i).dot(b_mat.col(j));
        }
      }
    };

    std::vector<int64> perm(num_nonzero_elements);
    std::iota(perm.begin(), perm.end(), 0);
    // TODO(walidk): improve performance in the case adj_a and not adj_b
    // TODO(walidk): benchmark smaller inputs, and potentially skip the sort
    // when the input fits in L3 cache.
    // Compute a permutation to sort either the a or b matrix, to take advantage
    // of CPU caching. Since row access is efficient (given the RowMajor
    // ordering), we prefer to
    //   sort according to a when a is transposed,
    //   sort according to b when b is not transpose.
    auto compare_a_index = [&get_a_index](int64 i, int64 j) {
      return get_a_index(i) < get_a_index(j);
    };
    auto compare_b_index = [&get_b_index](int64 i, int64 j) {
      return get_b_index(i) < get_b_index(j);
    };
    if (adj_a) {
      std::stable_sort(perm.begin(), perm.end(), compare_a_index);
    } else if (!adj_b) {
      std::stable_sort(perm.begin(), perm.end(), compare_b_index);
    }

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // Based on benchmarks, the cost is on the order of 20 cycles per dimension
    const int64 cost_per_unit = 20 * a_dim_1;
    // Lambda encapsulating the per-shard computation.
    auto work = [&](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        const int64 p = perm[i];
        const int64 a_index = get_a_index(p);
        const int64 b_index = get_b_index(p);
        prod_values(p) = get_dot_product(a_index, b_index);
      }
    };
    // Shard the work.
    worker_threads.workers->ParallelFor(
        num_nonzero_elements, cost_per_unit, work);
  }
};
REGISTER_KERNEL_BUILDER(Name("MaskedMatmul").Device(DEVICE_CPU),
                        MaskedMatmulOp);

}  // namespace tensorflow
