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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

Node* SparseTensorDenseMatMulNode(Graph* g, Node* a_indices, Node* a_values,
                                  Node* a_shape, Node* b, bool adjoint_a,
                                  bool adjoint_b) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseTensorDenseMatMul")
                  .Input(a_indices)
                  .Input(a_values)
                  .Input(a_shape)
                  .Input(b)
                  .Attr("T", DT_FLOAT)
                  .Attr("adjoint_a", adjoint_a)
                  .Attr("adjoint_b", adjoint_b)
                  .Finalize(g, &ret));
  return ret;
}

static Graph* SparseTensorDenseMatmul(int nnz, int m, int k, int n,
                                      bool adjoint_a, bool adjoint_b) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor a_values(DT_FLOAT, TensorShape({nnz}));
  Tensor a_indices(DT_INT64, TensorShape({nnz, 2}));
  Tensor a_shape(DT_INT64, TensorShape({2}));
  auto a_shape_t = a_shape.vec<int64>();
  a_shape_t(0) = adjoint_a ? k : m;
  a_shape_t(1) = adjoint_a ? m : k;
  a_values.flat<float>().setRandom();
  auto a_indices_t = a_indices.matrix<int64>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> a_lhs_dist(0, a_shape_t(0) - 1);
  std::uniform_int_distribution<> a_rhs_dist(0, a_shape_t(1) - 1);
  for (int32 i = 0; i < nnz; ++i) {
    a_indices_t(i, 0) = a_lhs_dist(gen);
    a_indices_t(i, 1) = a_rhs_dist(gen);
  }
  Tensor b(DT_FLOAT, adjoint_b ? TensorShape({n, k}) : TensorShape({k, n}));
  b.flat<float>().setRandom();

  SparseTensorDenseMatMulNode(
      g, test::graph::Constant(g, a_indices),
      test::graph::Constant(g, a_values), test::graph::HostConstant(g, a_shape),
      test::graph::Constant(g, b), adjoint_a, adjoint_b);
  return g;
}

// NOLINTBEGIN
#define BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, DEVICE)                  \
  static void                                                                        \
      BM_SparseTensorDenseMatmul##_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE( \
          ::testing::benchmark::State& state) {                                      \
    int64 items_per_iter = (static_cast<int64>(NNZ) * (TB ? K : N));                 \
    test::Benchmark(#DEVICE, SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB),          \
                    /*old_benchmark_api*/ false)                                     \
        .Run(state);                                                                 \
    state.SetItemsProcessed(state.iterations() * items_per_iter);                    \
    state.SetBytesProcessed(state.iterations() * items_per_iter *                    \
                            sizeof(float));                                          \
  }                                                                                  \
  BENCHMARK(                                                                         \
      BM_SparseTensorDenseMatmul##_##NNZ##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE);
// NOLINTEND

#define BM_SparseTensorDenseMatmul(NNZ, M, K, N, TA, TB)    \
  BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, cpu); \
  BM_SparseTensorDenseMatmulDev(NNZ, M, K, N, TA, TB, gpu);

BM_SparseTensorDenseMatmul(128, 8, 512, 1, false, false);
BM_SparseTensorDenseMatmul(128, 16, 512, 1, false, false);
BM_SparseTensorDenseMatmul(128, 128, 512, 1, false, false);

BM_SparseTensorDenseMatmul(128, 4096, 4096, 1, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 1, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 1, false, false);

BM_SparseTensorDenseMatmul(128, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(128, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(128, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(1024, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(1024, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(16384, 8, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 16, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 128, 1024, 16, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 128, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 1024, false, false);

BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, false, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, false, true);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, true, false);
BM_SparseTensorDenseMatmul(16384, 4096, 4096, 4096, true, true);

}  // end namespace tensorflow
