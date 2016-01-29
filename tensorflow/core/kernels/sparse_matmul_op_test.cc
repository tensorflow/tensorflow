/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
random::PhiloxRandom philox(1, 1);
random::SimplePhilox rnd(&philox);

void Sparsify(Tensor* t, float sparsity) {
  const int64 N = t->NumElements();
  CHECK_LE(sparsity, 1);
  auto flat = t->flat<float>();
  if (sparsity == 1) {
    flat.setZero();
    return;
  }
  static const uint32 K = 10000;
  for (int64 i = 0; i < N; ++i) {
    if (rnd.Uniform(K) < sparsity * K) {
      flat(i) = 0;
    } else if (flat(i) == 0) {
      flat(i) = 0.1;
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

static Graph* SparseMatMulHelper(Graph* g, int m, int n, int d,
                                 float sparsity_a, float sparsity_b,
                                 bool transpose_a, bool transpose_b) {
  bool a_sparse = (sparsity_a > 0);
  bool b_sparse = (sparsity_b > 0);

  auto left_shape = transpose_a ? TensorShape({d, m}) : TensorShape({m, d});
  Tensor left(DataTypeToEnum<float>::value, left_shape);
  left.flat<float>().setRandom();
  Sparsify(&left, sparsity_a);

  auto right_shape = transpose_b ? TensorShape({n, d}) : TensorShape({d, n});
  Tensor right(DataTypeToEnum<float>::value, right_shape);
  right.flat<float>().setRandom();
  Sparsify(&right, sparsity_b);

  SparseMatMulNode(g, test::graph::Constant(g, left),
                   test::graph::Constant(g, right), transpose_a, transpose_b,
                   a_sparse, b_sparse);
  return g;
}

static Graph* SparseMatMul(int m, int n, int d, float sparsity_a,
                           float sparsity_b, bool transpose_a,
                           bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  return SparseMatMulHelper(g, m, n, d, sparsity_a, sparsity_b, transpose_a,
                            transpose_b);
}

#define BM_SPARSE(M, K, N, S1, S2, TA, TB)                                   \
  static void BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TA##_##TB(       \
      int iters) {                                                           \
    testing::StopTiming();                                                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);      \
    std::string label =                                                      \
        strings::Printf("tr_a: %d tr_b: %d sp_a: %0.2f sp_b: %0.2f", TA, TB, \
                        S1 / 100.0, S2 / 100.0);                             \
    testing::SetLabel(label);                                                \
    testing::UseRealTime();                                                  \
    auto g = SparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, TA, TB);          \
    testing::StartTiming();                                                  \
    test::Benchmark("cpu", g).Run(iters);                                    \
  }                                                                          \
  BENCHMARK(BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TA##_##TB);

BM_SPARSE(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE(2048, 2048, 2048, 50, 0, false, false);
BM_SPARSE(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE(2048, 2048, 2048, 99, 0, false, false);

BM_SPARSE(2048, 2048, 2048, 0, 50, false, false);
BM_SPARSE(2048, 2048, 2048, 0, 85, false, false);

BM_SPARSE(2048, 2048, 2048, 85, 0, true, false);
BM_SPARSE(2048, 2048, 2048, 85, 0, false, true);
BM_SPARSE(2048, 2048, 2048, 85, 0, true, true);

BM_SPARSE(2048, 2048, 2048, 0, 85, true, false);
BM_SPARSE(2048, 2048, 2048, 0, 85, false, true);
BM_SPARSE(2048, 2048, 2048, 0, 85, true, true);

BM_SPARSE(1024, 1024, 1024, 0, 0, false, false);
BM_SPARSE(1024, 1024, 1024, 1, 0, false, false);
BM_SPARSE(1024, 1024, 1024, 85, 0, false, false);

BM_SPARSE(256, 256, 256, 1, 0, false, false);
BM_SPARSE(512, 512, 512, 1, 0, false, false);

static Graph* MultiSparseMatMul(int m, int n, int d, float sparsity_1,
                                float sparsity_2) {
  Graph* g = new Graph(OpRegistry::Global());
  SparseMatMulHelper(g, d, n, m, sparsity_1, sparsity_2, true, false);
  SparseMatMulHelper(g, m, d, n, sparsity_2, 0, false, true);
  return g;
}

#define BM_SPARSE_MULTI(M, K, N, S1, S2)                                    \
  static void BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2(int iters) {  \
    testing::StopTiming();                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2 * 3); \
    std::string label = strings::Printf("%d_%d_%d_%0.2f_%0.2f", M, K, N,    \
                                        S1 / 100.0, S2 / 100.0);            \
    testing::SetLabel(label);                                               \
    testing::UseRealTime();                                                 \
    auto g = MultiSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0);            \
    testing::StartTiming();                                                 \
    test::Benchmark("cpu", g).Run(iters);                                   \
  }                                                                         \
  BENCHMARK(BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2);

BM_SPARSE_MULTI(1024, 2140, 4096, 0, 82);
BM_SPARSE_MULTI(1024, 4096, 2048, 83, 83);

}  // end namespace tensorflow
