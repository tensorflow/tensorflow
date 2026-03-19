/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/mkl_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

static Graph *SparseMatrixMatmulGenerate(int nnz, int m, int k, int n,
                                         Tensor **csr_matrix_t,
                                         Tensor **dense_matrix_t) {
  Graph *g = new Graph(OpRegistry::Global());
  CSRSparseMatrix csr_matrix;

  // Generate the random COO matrix.
  Tensor a_values_t(DT_FLOAT, TensorShape({nnz}));
  Tensor a_indices_t(DT_INT64, TensorShape({nnz, 2}));
  Tensor a_shape_t(DT_INT64, TensorShape({2}));
  auto a_shape_vec = a_shape_t.vec<int64_t>();
  a_shape_vec(0) = m;
  a_shape_vec(1) = k;
  a_values_t.flat<float>().setRandom();
  auto a_indices_mat = a_indices_t.matrix<int64_t>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> a_lhs_dist(0, a_shape_vec(0) - 1);
  std::uniform_int_distribution<> a_rhs_dist(0, a_shape_vec(1) - 1);
  for (int32_t i = 0; i < nnz; ++i) {
    a_indices_mat(i, 0) = (const int64_t)a_lhs_dist(gen);
    a_indices_mat(i, 1) = (const int64_t)a_rhs_dist(gen);
  }

  // Calculate some constants for the conversion.
  const int64_t batch_size = 1;
  const int num_rows = a_shape_vec(0);
  const int num_cols = a_shape_vec(1);

  // Allocate memory for the output CSR.
  Tensor csr_batch_pointers(DT_INT32, TensorShape({batch_size + 1}));
  Tensor csr_column_indices(DT_INT32, TensorShape({nnz}));
  Tensor csr_row_pointers(DT_INT32, TensorShape({(num_rows + 1) * batch_size}));

  // Cast the indices matrix to const.
  auto a_indices_mat_const = std::as_const(a_indices_t).matrix<int64_t>();

  // Zero out the row pointers.
  memset(csr_row_pointers.flat<int32>().data(), 0,
         (num_rows + 1) * batch_size * sizeof(int32));

  // Convert from COO to CSR.
  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  TF_CHECK_OK(coo_to_csr(batch_size, num_rows, num_cols, a_indices_mat_const,
                         csr_batch_pointers.vec<int32>(),
                         csr_row_pointers.vec<int32>(),
                         csr_column_indices.vec<int32>()));

  // Construct a CSRSparseMatrix.
  TF_CHECK_OK(CSRSparseMatrix::CreateCSRSparseMatrix(
      DT_FLOAT, a_shape_t, csr_batch_pointers, csr_row_pointers,
      csr_column_indices, a_values_t, &csr_matrix));
  *csr_matrix_t = new Tensor(cpu_allocator(), DT_VARIANT, TensorShape({}));
  (*csr_matrix_t)->scalar<Variant>()() = std::move(csr_matrix);

  // Generate the dense tensor to multiply against.
  *dense_matrix_t = new Tensor(DT_FLOAT, TensorShape({k, n}));
  (*dense_matrix_t)->flat<float>().setRandom();

  return g;
}

static Graph *SparseMatrixMatmul(const string &kind, Graph *g,
                                 Tensor *csr_matrix_t, Tensor *dense_matrix_t) {
  const bool isDefault = (kind == "Default");
  Node *ret = nullptr;

  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(g->NewName("n1"), "SparseMatrixMatMul")
                    .Input(test::graph::Constant(g, *csr_matrix_t))
                    .Input(test::graph::Constant(g, *dense_matrix_t))
                    .Attr("T", DT_FLOAT)
                    .Finalize(g, &ret));
  } else {
    test::graph::oneDNNSparseCSRMatmul(
        g, test::graph::Constant(g, *csr_matrix_t),
        test::graph::Constant(g, *dense_matrix_t));
  }
  return g;
}

// NOLINTBEGIN
#define BM_SparseMatrixMatmulDev(kind, NNZ, M, K, N, DEVICE)                  \
  static void BM_SparseMatrixMatmul_##kind##NNZ##_##M##_##K##_##N##_##DEVICE( \
      ::testing::benchmark::State &state) {                                   \
    Tensor *csr_matrix_t, *dense_matrix_t;                                    \
    Graph *g;                                                                 \
    int64_t items_per_iter = (static_cast<int64_t>(NNZ) * N);                 \
    g = SparseMatrixMatmulGenerate(NNZ, M, K, N, &csr_matrix_t,               \
                                   &dense_matrix_t);                          \
    test::Benchmark(                                                          \
        #DEVICE, SparseMatrixMatmul(#kind, g, csr_matrix_t, dense_matrix_t),  \
        /*old_benchmark_api*/ false)                                          \
        .Run(state);                                                          \
    state.SetItemsProcessed(state.iterations() * items_per_iter);             \
    state.SetBytesProcessed(state.iterations() * items_per_iter *             \
                            sizeof(float));                                   \
  }                                                                           \
  BENCHMARK(BM_SparseMatrixMatmul_##kind##NNZ##_##M##_##K##_##N##_##DEVICE)   \
      ->Arg(/* unused arg */ 1);
// NOLINTEND

#define BM_SparseMatrixMatmul(NNZ, M, K, N)             \
  BM_SparseMatrixMatmulDev(Default, NNZ, M, K, N, cpu); \
  BM_SparseMatrixMatmulDev(Mkl, NNZ, M, K, N, cpu);

BM_SparseMatrixMatmul(128, 8, 512, 1);
BM_SparseMatrixMatmul(128, 16, 512, 1);
BM_SparseMatrixMatmul(128, 128, 512, 1);

BM_SparseMatrixMatmul(128, 4096, 4096, 1);
BM_SparseMatrixMatmul(1024, 4096, 4096, 1);
BM_SparseMatrixMatmul(16384, 4096, 4096, 1);

BM_SparseMatrixMatmul(128, 8, 1024, 16);
BM_SparseMatrixMatmul(128, 16, 1024, 16);
BM_SparseMatrixMatmul(128, 128, 1024, 16);
BM_SparseMatrixMatmul(128, 4096, 4096, 128);
BM_SparseMatrixMatmul(128, 4096, 4096, 1024);

BM_SparseMatrixMatmul(1024, 8, 1024, 16);
BM_SparseMatrixMatmul(1024, 16, 1024, 16);
BM_SparseMatrixMatmul(1024, 128, 1024, 16);
BM_SparseMatrixMatmul(1024, 4096, 4096, 128);
BM_SparseMatrixMatmul(1024, 4096, 4096, 1024);

BM_SparseMatrixMatmul(16384, 8, 1024, 16);
BM_SparseMatrixMatmul(16384, 16, 1024, 16);
BM_SparseMatrixMatmul(16384, 128, 1024, 16);
BM_SparseMatrixMatmul(16384, 4096, 4096, 128);
BM_SparseMatrixMatmul(16384, 4096, 4096, 1024);

BM_SparseMatrixMatmul(16384, 4096, 4096, 4096);

// The big ones.
BM_SparseMatrixMatmul(100, 1, 1000000, 100);
BM_SparseMatrixMatmul(200, 1, 2000000, 100);
BM_SparseMatrixMatmul(400, 1, 4000000, 100);

BM_SparseMatrixMatmul(400, 4, 1000000, 100);
BM_SparseMatrixMatmul(800, 4, 2000000, 100);
BM_SparseMatrixMatmul(1600, 4, 4000000, 100);

BM_SparseMatrixMatmul(800, 8, 1000000, 100);
BM_SparseMatrixMatmul(1600, 8, 2000000, 100);
BM_SparseMatrixMatmul(3200, 8, 4000000, 100);

// The bigger ones.
// BM_SparseMatrixMatmul(100, 1, 1000000, 1000);
// BM_SparseMatrixMatmul(200, 1, 2000000, 1000);
// BM_SparseMatrixMatmul(400, 1, 4000000, 1000);

// BM_SparseMatrixMatmul(400, 4, 1000000, 1000);
// BM_SparseMatrixMatmul(800, 4, 2000000, 1000);
// BM_SparseMatrixMatmul(1600, 4, 4000000, 1000);

// BM_SparseMatrixMatmul(800, 8, 1000000, 1000);
// BM_SparseMatrixMatmul(1600, 8, 2000000, 1000);
// BM_SparseMatrixMatmul(3200, 8, 4000000, 1000);

}  // namespace
}  // end namespace tensorflow

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
