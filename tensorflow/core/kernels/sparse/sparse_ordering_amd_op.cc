/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>

#include "tensorflow/core/framework/types.pb.h"

#define EIGEN_USE_THREADS

#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/SparseCholesky"  // from @eigen_archive
#include "Eigen/SparseCore"  // from @eigen_archive
#include "Eigen/OrderingMethods"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

// Op to compute the Approximate Minimum Degree (AMD) ordering for a sparse
// matrix.
//
// Accepts a CSRSparseMatrix which may represent a single sparse matrix (rank 2)
// or a batch of sparse matrices (rank 3). Each component must be a square
// matrix. The input is assumed to be symmetric; only the lower triangular part
// of each component matrix is read. The numeric values of the sparse matrix
// does not affect the returned AMD ordering; only the sparsity pattern does.
//
// For each component sparse matrix A, the corresponding output Tensor
// represents the AMD ordering of A's rows and columns. The ordering is returned
// as a 1D Tensor (per batch) containing the list of indices, i.e. it contains
// each of the integers {0, .. N-1} exactly once; where N is the number of rows
// of the sparse matrix. The ith element represents the index of the row that
// the ith row should map to.

// If P represents the permutation matrix corresponding to the indices, then the
// matrix:
//   P^{-1} * A * P
// would have a sparse Cholesky decomposition with fewer structural non-zero
// elements than the sparse Cholesky decomposition of A itself.
class CSROrderingAMDCPUOp : public OpKernel {
  using SparseMatrix = Eigen::SparseMatrix<int, Eigen::RowMajor>;
  using Indices =
      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IndicesMap = Eigen::Map<Indices>;
  using ConstIndicesMap = Eigen::Map<const Indices>;

 public:
  explicit CSROrderingAMDCPUOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) final {
    // Extract the input CSRSparseMatrix.
    const CSRSparseMatrix* input_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix));

    const Tensor& dense_shape = input_matrix->dense_shape();
    const int rank = dense_shape.dim_size(0);
    OP_REQUIRES(ctx, rank == 2 || rank == 3,
                errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
                                        "but dense_shape has size ", rank));

    auto dense_shape_vec = dense_shape.vec<int64_t>();
    const int64_t num_rows = dense_shape_vec((rank == 2) ? 0 : 1);
    const int64_t num_cols = dense_shape_vec((rank == 2) ? 1 : 2);

    OP_REQUIRES(ctx, num_rows == num_cols,
                errors::InvalidArgument("sparse matrix must be square; got: ",
                                        num_rows, " != ", num_cols));

    // Allocate the output permutation indices.
    const int batch_size = input_matrix->batch_size();
    TensorShape permutation_indices_shape =
        (rank == 2) ? TensorShape{num_rows} : TensorShape{batch_size, num_rows};
    Tensor permutation_indices(cpu_allocator(), DT_INT32,
                               permutation_indices_shape);
    ctx->set_output(0, permutation_indices);

    // Parallelize AMD computation across batches using a threadpool.
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64_t amd_cost_per_batch =
        10 * num_rows * (input_matrix->total_nnz() / batch_size);
    Shard(
        worker_threads.num_threads, worker_threads.workers, batch_size,
        amd_cost_per_batch, [&](int64_t batch_begin, int64_t batch_end) {
          for (int64_t batch_index = batch_begin; batch_index < batch_end;
               ++batch_index) {
            // Define an Eigen SparseMatrix Map to operate on the
            // CSRSparseMatrix component without copying the data.
            // The values doesn't matter for computing the ordering, hence we
            // reuse the column pointers as dummy values.
            Eigen::Map<const SparseMatrix> sparse_matrix(
                num_rows, num_rows, input_matrix->nnz(batch_index),
                input_matrix->row_pointers_vec(batch_index).data(),
                input_matrix->col_indices_vec(batch_index).data(),
                input_matrix->col_indices_vec(batch_index).data());
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>
                permutation_matrix;
            // Compute the AMD ordering.
            Eigen::AMDOrdering<int> amd_ordering;
            amd_ordering(sparse_matrix.template selfadjointView<Eigen::Lower>(),
                         permutation_matrix);
            // Define an Eigen Map over the allocated output Tensor so that it
            // can be mutated in place.
            IndicesMap permutation_map(
                permutation_indices.flat<int>().data() + batch_index * num_rows,
                num_rows, 1);
            permutation_map = permutation_matrix.indices();
          }
        });
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseMatrixOrderingAMD").Device(DEVICE_CPU),
                        CSROrderingAMDCPUOp);

}  // namespace tensorflow
