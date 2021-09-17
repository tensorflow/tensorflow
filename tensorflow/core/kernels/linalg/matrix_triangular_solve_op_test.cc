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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/broadcast_to_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

Node* BroadcastTo(Graph* g, Node* input, Node* shape) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BroadcastTo")
                  .Input(input)
                  .Input(shape)
                  .Attr("Tidx", DT_INT64)
                  .Finalize(g, &ret));
  return ret;
}

Node* MatrixTriangularSolve(Graph* g, Node* in0, Node* in1, bool adjoint) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MatrixTriangularSolve")
                  .Input(in0)
                  .Input(in1)
                  .Attr("lower", true)
                  .Attr("adjoint", adjoint)
                  .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* MatrixTriangularSolveWithBroadcast(int64_t b0, int64_t b1,
                                                 int64_t m, int64_t n,
                                                 bool manual_broadcast,
                                                 DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, TensorShape({b0, m, m}));
  // Set diagonal to non-zero to guarantee invertibility.
  in0.flat<T>().setRandom();
  auto matrix = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      in0.flat<T>().data(), in0.dim_size(1), in0.dim_size(2));

  matrix.diagonal() =
      (matrix.diagonal().cwiseAbs().array() + static_cast<T>(0.5));
  Tensor in1(type, TensorShape({b1, m, n}));
  in1.flat<T>().setRandom();

  Tensor broadcasted_in0_shape(DT_INT64, TensorShape({3}));
  Tensor broadcasted_in1_shape(DT_INT64, TensorShape({3}));

  Node* in0_node = nullptr;
  Node* in1_node = nullptr;
  if (manual_broadcast) {
    auto vec0 = broadcasted_in0_shape.vec<int64_t>();
    auto vec1 = broadcasted_in1_shape.vec<int64_t>();
    for (int i = 0; i < 3; ++i) {
      vec0(i) = (i == 0 ? std::max(b0, b1) : in0.shape().dim_size(i));
      vec1(i) = (i == 0 ? std::max(b0, b1) : in1.shape().dim_size(i));
    }
    in0_node = BroadcastTo(g, test::graph::Constant(g, in0),
                           test::graph::Constant(g, broadcasted_in0_shape));
    in1_node = BroadcastTo(g, test::graph::Constant(g, in1),
                           test::graph::Constant(g, broadcasted_in1_shape));
  } else {
    in0_node = test::graph::Constant(g, in0);
    in1_node = test::graph::Constant(g, in1);
  }

  MatrixTriangularSolve(g, in0_node, in1_node, false);
  return g;
}

// Macro arguments names: --------------------------------------------------- //
//   B1: batch size of LHS
//   B2: batch size of RHS
//    M: inner dimensions of LHS and RHS, outer dimension of LHS
//    N: outer dimension of RHS
//   MB: boolean indicating whether to use manual broadcasting
//    T: C++ type of scalars (e.g. float, std::complex)
//   TT: TensorFlow type of scalars (e.g. DT_FLOAT, DT_COMPLEX128
//    D: Device (e.g. cpu, gpu)
#define BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, T, TT, D)               \
  static void                                                                 \
      BM_MatrixTriangularSolve##_##B1##_##B2##_##M##_##N##_##MB##_##TT##_##D( \
          ::testing::benchmark::State& state) {                               \
    state.SetItemsProcessed(state.iterations() * std::max(B1, B2) * M * M *   \
                            N * 2);                                           \
    test::Benchmark(                                                          \
        #D, MatrixTriangularSolveWithBroadcast<T>(B1, B2, M, N, MB, TT),      \
        /*old_benchmark_api*/ false)                                          \
        .Run(state);                                                          \
  }                                                                           \
  BENCHMARK(                                                                  \
      BM_MatrixTriangularSolve##_##B1##_##B2##_##M##_##N##_##MB##_##TT##_##D);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define BM_MatrixTriangularSolve(B1, B2, M, N, MB)                       \
  BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, float, DT_FLOAT, cpu);   \
  BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, double, DT_DOUBLE, cpu); \
  BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, float, DT_FLOAT, gpu);   \
  BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, double, DT_DOUBLE, gpu);

#else

#define BM_MatrixTriangularSolve(B1, B2, M, N, MB)                     \
  BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, float, DT_FLOAT, cpu); \
  BM_MatrixTriangularSolveDev(B1, B2, M, N, MB, double, DT_DOUBLE, cpu);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Square matrix triangular solve.
BM_MatrixTriangularSolve(32, 32, 512, 512, true);
BM_MatrixTriangularSolve(32, 32, 512, 512, false);
BM_MatrixTriangularSolve(1, 32, 512, 512, true);
BM_MatrixTriangularSolve(1, 32, 512, 512, false);
BM_MatrixTriangularSolve(32, 1, 512, 512, true);
BM_MatrixTriangularSolve(32, 1, 512, 512, false);
BM_MatrixTriangularSolve(128, 128, 512, 512, true);
BM_MatrixTriangularSolve(128, 128, 512, 512, false);
BM_MatrixTriangularSolve(1, 128, 512, 512, true);
BM_MatrixTriangularSolve(1, 128, 512, 512, false);
BM_MatrixTriangularSolve(128, 1, 512, 512, true);
BM_MatrixTriangularSolve(128, 1, 512, 512, false);
BM_MatrixTriangularSolve(1, 128, 1024, 1024, true);
BM_MatrixTriangularSolve(1, 128, 1024, 1024, false);
BM_MatrixTriangularSolve(128, 1, 1024, 1024, true);
BM_MatrixTriangularSolve(128, 1, 1024, 1024, false);

// Matrix-vector triangular solve.
BM_MatrixTriangularSolve(1, 128, 200, 1, true);
BM_MatrixTriangularSolve(1, 128, 200, 1, false);
BM_MatrixTriangularSolve(128, 1, 200, 1, true);
BM_MatrixTriangularSolve(128, 1, 200, 1, false);

// Matrix-vector triangular solve, large dimension.
BM_MatrixTriangularSolve(1, 128, 200, 10000, true);
BM_MatrixTriangularSolve(1, 128, 200, 10000, false);
BM_MatrixTriangularSolve(128, 1, 200, 10000, true);
BM_MatrixTriangularSolve(128, 1, 200, 10000, false);

}  // namespace
}  // namespace tensorflow
