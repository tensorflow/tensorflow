// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
// ==============================================================================

// TensorFlow kernels and Ops for constructing WALS normal equations.
// TODO(agarwal,rmlarsen): Add security checks to the code.

#include <algorithm>
#include <numeric>
#include <vector>

// This is only used for std::this_thread::get_id()
#include <thread>  // NOLINT

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"

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

// TODO(ataei): Consider using RowMajor maps.
typedef Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
    EigenMatrixFloatMap;
typedef Eigen::Map<
    const Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
    ConstEigenMatrixInt64Map;
typedef Eigen::Map<
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
    ConstEigenMatrixFloatMap;

class WALSComputePartialLhsAndRhsOp : public OpKernel {
 public:
  explicit WALSComputePartialLhsAndRhsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->MatchSignature(
                                {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                 DT_INT64, DT_FLOAT, DT_INT64, DT_BOOL},
                                {DT_FLOAT, DT_FLOAT}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& factors = context->input(0);
    const Tensor& factor_weights = context->input(1);
    const Tensor& unobserved_weights = context->input(2);
    const Tensor& input_weights = context->input(3);
    const Tensor& input_indices = context->input(4);
    const Tensor& input_values = context->input(5);
    const Tensor& input_block_size = context->input(6);
    const Tensor& input_is_transpose = context->input(7);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(factors.shape()),
                InvalidArgument("Input factors should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(factor_weights.shape()),
                InvalidArgument("Input factor_weights should be a vector."));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(unobserved_weights.shape()),
        InvalidArgument("Input unobserved_weights should be a scalar."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_weights.shape()),
                InvalidArgument("Input input_weights should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                InvalidArgument("Input input_indices should be a matrix."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values.shape()),
                InvalidArgument("Input input_values should be a vector"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_block_size.shape()),
                InvalidArgument("Input input_block_size should be a scalar."));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(input_is_transpose.shape()),
        InvalidArgument("Input input_is_transpose should be a scalar."));

    const int64 factor_dim = factors.dim_size(1);
    const int64 factors_size = factors.dim_size(0);
    const int64 num_nonzero_elements = input_indices.dim_size(0);
    const int64 block_size = input_block_size.scalar<int64>()();
    const auto& factor_weights_vec = factor_weights.vec<float>();
    const auto& input_weights_vec = input_weights.vec<float>();
    const float w_0 = unobserved_weights.scalar<float>()();
    const auto& input_values_vec = input_values.vec<float>();

    ConstEigenMatrixFloatMap factors_mat(factors.matrix<float>().data(),
                                         factor_dim, factors_size);
    ConstEigenMatrixInt64Map indices_mat(input_indices.matrix<int64>().data(),
                                         2, num_nonzero_elements);

    Tensor* output_lhs_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({block_size, factor_dim, factor_dim}),
                       &output_lhs_tensor));
    auto output_lhs_t = output_lhs_tensor->tensor<float, 3>();
    output_lhs_t.setZero();
    Tensor* output_rhs_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({block_size, factor_dim}),
                                &output_rhs_tensor));
    EigenMatrixFloatMap rhs_mat(output_rhs_tensor->matrix<float>().data(),
                                factor_dim, block_size);
    rhs_mat.setZero();
    const bool is_transpose = input_is_transpose.scalar<bool>()();

    auto get_input_index = [is_transpose, &indices_mat](int64 i) {
      return is_transpose ? indices_mat(1, i) : indices_mat(0, i);
    };
    auto get_factor_index = [is_transpose, &indices_mat](int64 i) {
      return is_transpose ? indices_mat(0, i) : indices_mat(1, i);
    };

    // TODO(rmlarsen): In principle, we should be using the SparseTensor class
    // and machinery for iterating over groups, but the fact that class
    // SparseTensor makes a complete copy of the matrix makes me reluctant to
    // use it.
    std::vector<int64> perm(num_nonzero_elements);
    std::iota(perm.begin(), perm.end(), 0);

    typedef std::pair<int64, int64> Shard;
    std::vector<Shard> shards;
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const int num_threads = worker_threads.num_threads;
    int64 shard_total = 0;
    if (num_threads == 1) {
      shards.emplace_back(0, num_nonzero_elements);
      shard_total += num_nonzero_elements;
    } else {
      // Compute a permutation such that get_input_index(perm[i]) is sorted, use
      // stable_sort to preserve spatial locality.
      std::stable_sort(perm.begin(), perm.end(),
                       [&get_input_index](int64 i, int64 j) {
                         return get_input_index(i) < get_input_index(j);
                       });

      // Compute the start and end of runs with identical input_index.
      // These are the shards of work that can be processed in parallel
      // without locking.
      int64 start = 0;
      int64 end = 0;
      while (end < num_nonzero_elements) {
        start = end;
        while (end < num_nonzero_elements &&
               get_input_index(perm[start]) == get_input_index(perm[end])) {
          ++end;
        }
        shards.emplace_back(start, end);
        shard_total += end - start;
      }
    }
    CHECK_EQ(shard_total, num_nonzero_elements);
    CHECK_LE(shards.size(), num_nonzero_elements);
    CHECK_GT(shards.size(), 0);

    // Batch the rank-one updates into a rank-k update to lower memory traffic
    const int kMaxBatchSize = 128;

    // Since we do not have an easy way of generating thread id's within the
    // range [0,num_threads), we can instead call out to an std::unordered_map
    // of matrices and initialize the matrix on the first call.
    // However, this might have a performance penalty, as memory allocation can
    // cause the OS kernel to enter a critical section and temporarily disable
    // parallelism, and the unordered_map must be protected with a read/write
    // mutex.
    //
    // TODO(jpoulson): Simplify after the thread rank can be queried
    std::unordered_map<size_t, Eigen::MatrixXf> factor_batch_map;
    mutex map_mutex;

    BlockingCounter counter(shards.size());
    // Lambda encapsulating the per-shard computation.
    auto work = [&](const Shard& shard) {
      const std::thread::id thread_id = std::this_thread::get_id();
      const size_t id_hash = std::hash<std::thread::id>()(thread_id);
      // If this thread's unique factors_mat.rows() x kMaxBatchSize
      // batching matrix has not yet been created, then emplace it into the
      // map using the hash of the thread id as the key.
      //
      // TODO(jpoulson): Switch to try_emplace once C++17 is supported
      map_mutex.lock();
      const auto key_count = factor_batch_map.count(id_hash);
      map_mutex.unlock();
      if (!key_count) {
        map_mutex.lock();
        factor_batch_map.emplace(
            std::piecewise_construct, std::forward_as_tuple(id_hash),
            std::forward_as_tuple(factors_mat.rows(), kMaxBatchSize));
        map_mutex.unlock();
      }
      map_mutex.lock();
      auto& factor_batch = factor_batch_map[id_hash];
      map_mutex.unlock();

      CHECK_GE(shard.first, 0);
      CHECK_LE(shard.second, perm.size());
      CHECK_LE(shard.first, shard.second);
      const int64 input_index = get_input_index(perm[shard.first]);
      // Acccumulate the rhs and lhs terms in the normal equations
      // for the non-zero elements in the row or column of the sparse matrix
      // corresponding to input_index.
      int num_batched = 0;
      EigenMatrixFloatMap lhs_mat(output_lhs_tensor->flat<float>().data() +
                                      input_index * factor_dim * factor_dim,
                                  factor_dim, factor_dim);
      auto lhs_symm = lhs_mat.selfadjointView<Eigen::Lower>();
      for (int64 p = shard.first; p < shard.second; ++p) {
        const int64 i = perm[p];
        // Check that all entries in the shard have the same input index.
        CHECK_EQ(input_index, get_input_index(i));
        const int64 factor_index = get_factor_index(i);
        const float input_value = input_values_vec(i);
        const float weight =
            input_weights_vec(input_index) * factor_weights_vec(factor_index);
        CHECK_GE(weight, 0);
        factor_batch.col(num_batched) =
            factors_mat.col(factor_index) * std::sqrt(weight);
        ++num_batched;
        if (num_batched == kMaxBatchSize) {
          lhs_symm.rankUpdate(factor_batch);
          num_batched = 0;
        }

        rhs_mat.col(input_index) +=
            input_value * (w_0 + weight) * factors_mat.col(factor_index);
      }
      if (num_batched != 0) {
        auto factor_block =
            factor_batch.block(0, 0, factors_mat.rows(), num_batched);
        lhs_symm.rankUpdate(factor_block);
      }
      // Copy lower triangular to upper triangular part of normal equation
      // matrix.
      lhs_mat = lhs_symm;
      counter.DecrementCount();
    };
    for (int i = 1; i < shards.size(); ++i) {
      worker_threads.workers->Schedule(std::bind(work, shards[i]));
    }
    // Inline execute the 1st shard.
    work(shards[0]);
    counter.Wait();
  }
};

REGISTER_KERNEL_BUILDER(Name("WALSComputePartialLhsAndRhs").Device(DEVICE_CPU),
                        WALSComputePartialLhsAndRhsOp);

}  // namespace tensorflow
