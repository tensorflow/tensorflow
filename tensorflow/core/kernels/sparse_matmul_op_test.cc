#include "tensorflow/core/framework/types.pb.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
random::PhiloxRandom philox(1, 1);
random::SimplePhilox rnd(&philox);

void Sparsify(Tensor* t, float sparsity) {
  const int64 N = t->NumElements();
  CHECK_LE(sparsity, 1);
  if (sparsity <= 0) return;
  auto flat = t->flat<float>();
  static const uint32 K = 10000;
  for (int64 i = 0; i < N; ++i) {
    if (rnd.Uniform(K) < sparsity * K) {
      flat(i) = 0;
    }
  }
}

Node* SparseMatMulNode(Graph* g, Node* in0, Node* in1, bool transpose_a,
                       bool transpose_b, bool a_sparse, bool b_sparse) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseMatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Attr("a_is_sparse", a_sparse)
                  .Attr("b_is_sparse", b_sparse)
                  .Finalize(g, &ret));
  return ret;
}

static Graph* SparseMatMulHelper(Graph* g, int m, int n, int d, float sparsity,
                                 bool transpose_a, bool transpose_b,
                                 bool a_sparse, bool b_sparse) {
  a_sparse = a_sparse && (sparsity > 0);
  b_sparse = b_sparse && (sparsity > 0);

  auto left_shape = transpose_a ? TensorShape({d, m}) : TensorShape({m, d});
  Tensor left(DataTypeToEnum<float>::value, left_shape);
  left.flat<float>().setRandom();
  if (a_sparse) {
    Sparsify(&left, sparsity);
  }

  auto right_shape = transpose_b ? TensorShape({n, d}) : TensorShape({d, n});
  Tensor right(DataTypeToEnum<float>::value, right_shape);
  right.flat<float>().setRandom();
  if (b_sparse) {
    Sparsify(&right, sparsity);
  }

  SparseMatMulNode(g, test::graph::Constant(g, left),
                   test::graph::Constant(g, right), transpose_a, transpose_b,
                   a_sparse, b_sparse);
  return g;
}

static Graph* SparseMatMul(int m, int n, int d, float sparsity,
                           bool transpose_a, bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  return SparseMatMulHelper(g, m, n, d, sparsity, transpose_a, transpose_b,
                            true, false);
}

static Graph* MultiSparseMatMul(int m, int n, int d, float sparsity_a,
                                float sparsity_b) {
  Graph* g = new Graph(OpRegistry::Global());
  if (sparsity_a == 0 && sparsity_b > 0) {
    SparseMatMulHelper(g, m, n, d, sparsity_a, false, false, false, false);
    SparseMatMulHelper(g, n, d, m, sparsity_b, true, true, true, false);
    SparseMatMulHelper(g, m, d, n, sparsity_b, false, false, true, false);
  } else {
    SparseMatMulHelper(g, m, n, d, sparsity_a, false, true, true, false);
    SparseMatMulHelper(g, d, n, m, sparsity_a, true, false, true, true);
    SparseMatMulHelper(g, m, d, n, sparsity_b, false, false, true, false);
  }
  return g;
}

#define BM_SPARSE(M, K, N, S)                                                  \
  static void BM_Sparse##_##M##_##K##_##N##_##S(int iters) {                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    std::string label = strings::Printf("%d_%d_%d_%0.2f", M, K, N, S / 100.0); \
    testing::SetLabel(label);                                                  \
    test::Benchmark("cpu", SparseMatMul(M, N, K, S / 100.0, false, false))     \
        .Run(iters);                                                           \
  }                                                                            \
  BENCHMARK(BM_Sparse##_##M##_##K##_##N##_##S);

BM_SPARSE(2048, 2048, 2048, 0);
BM_SPARSE(2048, 2048, 2048, 1);
BM_SPARSE(2048, 2048, 2048, 85);

BM_SPARSE(1024, 1024, 1024, 0);
BM_SPARSE(1024, 1024, 1024, 1);
BM_SPARSE(1024, 1024, 1024, 85);

BM_SPARSE(256, 256, 256, 1);
BM_SPARSE(512, 512, 512, 1);

#define BM_SPARSE_MULTI(M, K, N, S1, S2)                                       \
  static void BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2(int iters) {     \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2 * 3);    \
    std::string label = strings::Printf("%d_%d_%d_%0.2f_%0.2f", M, K, N,       \
                                        S1 / 100.0, S2 / 100.0);               \
    testing::SetLabel(label);                                                  \
    test::Benchmark("cpu", MultiSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0)) \
        .Run(iters);                                                           \
  }                                                                            \
  BENCHMARK(BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2);

BM_SPARSE_MULTI(512, 2140, 4096, 0, 82);
BM_SPARSE_MULTI(512, 4096, 2048, 83, 83);

#define BM_SPARSE_TR(M, K, N, S, TA, TB)                                     \
  static void BM_Sparse##_##M##_##K##_##N##_##S##_##TA##_##TB(int iters) {   \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);      \
    std::string label =                                                      \
        strings::Printf("%d_%d_%d_%d_%d_%0.2f", M, K, N, TA, TB, S / 100.0); \
    testing::SetLabel(label);                                                \
    test::Benchmark("cpu", SparseMatMul(M, N, K, S / 100.0, TA, TB))         \
        .Run(iters);                                                         \
  }                                                                          \
  BENCHMARK(BM_Sparse##_##M##_##K##_##N##_##S##_##TA##_##TB);

BM_SPARSE_TR(2048, 2048, 2048, 1, true, false);
BM_SPARSE_TR(2048, 2048, 2048, 1, false, true);
BM_SPARSE_TR(2048, 2048, 2048, 1, true, true);

}  // end namespace tensorflow
