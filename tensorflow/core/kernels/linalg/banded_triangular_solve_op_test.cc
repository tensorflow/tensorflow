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
#include "tensorflow/core/kernels/linalg/matrix_set_diag_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

Node* SetDiag(int num_bands, Graph* g, Node* bands, Node* triangular) {
  Node* ret;
  Tensor bandwidth(DT_INT32, TensorShape({2}));
  bandwidth.flat<int32>()(0) = -(num_bands - 1);
  bandwidth.flat<int32>()(1) = 0;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MatrixSetDiagV3")
                  .Input(triangular)
                  .Input(bands)
                  .Input(test::graph::Constant(g, bandwidth))
                  .Attr("align", "RIGHT_LEFT")
                  .Finalize(g, &ret));
  return ret;
}

Node* BandedTriangularSolve(Graph* g, Node* in0, Node* in1) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BandedTriangularSolve")
                  .Input(in0)
                  .Input(in1)
                  .Attr("lower", true)
                  .Attr("adjoint", false)
                  .Finalize(g, &ret));
  return ret;
}

Node* MatrixTriangularSolve(Graph* g, Node* in0, Node* in1) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MatrixTriangularSolve")
                  .Input(in0)
                  .Input(in1)
                  .Attr("lower", true)
                  .Attr("adjoint", false)
                  .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* BandedTriangularSolve(int64 num_bands, int64 n, int64 m,
                                    bool use_banded_solver, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, TensorShape({num_bands, n}));
  // Set diagonal to nonzero to guarantee invertibility.
  in0.flat<T>().setRandom();
  in0.flat<T>() =
      in0.flat<T>().abs() + in0.flat<T>().constant(static_cast<T>(0.5));
  Tensor in1(type, TensorShape({n, m}));
  in1.flat<T>().setRandom();
  if (use_banded_solver) {
    BandedTriangularSolve(g, test::graph::Constant(g, in0),
                          test::graph::Constant(g, in1));
  } else {
    // Create a zero tensor.
    Tensor in2(type, TensorShape({n, n}));
    in2.flat<T>().setZero();
    Node* triangular_matrix =
        SetDiag(num_bands, g, test::graph::Constant(g, in0),
                test::graph::Constant(g, in2));
    MatrixTriangularSolve(g, triangular_matrix, test::graph::Constant(g, in1));
  }
  return g;
}

// Macro arguments names: --------------------------------------------------- //
//   K: Number of bands
//   N: Inner dimension of LHS, Inner dimension of RHS.
//   M: Outer dimensions of RHS
//   BS: boolean indicating whether to use the banded solver
//    T: C++ type of scalars (e.g. float, std::complex)
//   TT: TensorFlow type of scalars (e.g. DT_FLOAT, DT_COMPLEX128
#define BM_BandedTriangularSolveDev(K, N, M, BS, T, TT, D)              \
  static void BM_BandedTriangularSolve##_##K##_##N##_##M##_##BS##_##TT( \
      ::testing::benchmark::State& state) {                             \
    test::Benchmark(#D, BandedTriangularSolve<T>(K, N, M, BS, TT),      \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * K * N + N * M);        \
  }                                                                     \
  BENCHMARK(BM_BandedTriangularSolve##_##K##_##N##_##M##_##BS##_##TT)   \
      ->UseRealTime();

#define BM_BandedTriangularSolve(K, N, M, BS, D)                \
  BM_BandedTriangularSolveDev(K, N, M, BS, float, DT_FLOAT, D); \
  BM_BandedTriangularSolveDev(K, N, M, BS, double, DT_DOUBLE, D);

// Small number of bands, few rhs
BM_BandedTriangularSolve(2, 32, 1, true, cpu);
BM_BandedTriangularSolve(2, 32, 1, false, cpu);
BM_BandedTriangularSolve(4, 32, 1, true, cpu);
BM_BandedTriangularSolve(4, 32, 1, false, cpu);
BM_BandedTriangularSolve(8, 32, 1, true, cpu);
BM_BandedTriangularSolve(8, 32, 1, false, cpu);
BM_BandedTriangularSolve(16, 32, 1, true, cpu);
BM_BandedTriangularSolve(16, 32, 1, false, cpu);
BM_BandedTriangularSolve(2, 128, 1, true, cpu);
BM_BandedTriangularSolve(2, 128, 1, false, cpu);
BM_BandedTriangularSolve(4, 128, 1, true, cpu);
BM_BandedTriangularSolve(4, 128, 1, false, cpu);
BM_BandedTriangularSolve(8, 128, 1, true, cpu);
BM_BandedTriangularSolve(8, 128, 1, false, cpu);
BM_BandedTriangularSolve(16, 128, 1, true, cpu);
BM_BandedTriangularSolve(16, 128, 1, false, cpu);
BM_BandedTriangularSolve(2, 512, 1, true, cpu);
BM_BandedTriangularSolve(2, 512, 1, false, cpu);
BM_BandedTriangularSolve(4, 512, 1, true, cpu);
BM_BandedTriangularSolve(4, 512, 1, false, cpu);
BM_BandedTriangularSolve(8, 512, 1, true, cpu);
BM_BandedTriangularSolve(8, 512, 1, false, cpu);
BM_BandedTriangularSolve(16, 512, 1, true, cpu);
BM_BandedTriangularSolve(16, 512, 1, false, cpu);

// Larger # rhs
BM_BandedTriangularSolve(2, 32, 32, true, cpu);
BM_BandedTriangularSolve(2, 32, 32, false, cpu);
BM_BandedTriangularSolve(4, 32, 32, true, cpu);
BM_BandedTriangularSolve(4, 32, 32, false, cpu);
BM_BandedTriangularSolve(8, 32, 32, true, cpu);
BM_BandedTriangularSolve(8, 32, 32, false, cpu);
BM_BandedTriangularSolve(16, 32, 32, true, cpu);
BM_BandedTriangularSolve(16, 32, 32, false, cpu);
BM_BandedTriangularSolve(2, 128, 128, true, cpu);
BM_BandedTriangularSolve(2, 128, 128, false, cpu);
BM_BandedTriangularSolve(4, 128, 128, true, cpu);
BM_BandedTriangularSolve(4, 128, 128, false, cpu);
BM_BandedTriangularSolve(8, 128, 128, true, cpu);
BM_BandedTriangularSolve(8, 128, 128, false, cpu);
BM_BandedTriangularSolve(16, 128, 128, true, cpu);
BM_BandedTriangularSolve(16, 128, 128, false, cpu);
BM_BandedTriangularSolve(2, 512, 512, true, cpu);
BM_BandedTriangularSolve(2, 512, 512, false, cpu);
BM_BandedTriangularSolve(4, 512, 512, true, cpu);
BM_BandedTriangularSolve(4, 512, 512, false, cpu);
BM_BandedTriangularSolve(8, 512, 512, true, cpu);
BM_BandedTriangularSolve(8, 512, 512, false, cpu);
BM_BandedTriangularSolve(16, 512, 512, true, cpu);
BM_BandedTriangularSolve(16, 512, 512, false, cpu);

BM_BandedTriangularSolve(2, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(2, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(4, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(4, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(8, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(8, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(16, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(16, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(32, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(32, 2048, 2048, false, cpu);
BM_BandedTriangularSolve(64, 2048, 2048, true, cpu);
BM_BandedTriangularSolve(64, 2048, 2048, false, cpu);

}  // namespace
}  // namespace tensorflow
