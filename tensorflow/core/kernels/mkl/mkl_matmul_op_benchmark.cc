/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/mkl_layout_pass.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {
namespace {

template <typename T>
static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();

  Node* src0 = test::graph::Constant(g, in0);
  Node* src1 = test::graph::Constant(g, in1);
  g->AddEdge(g->source_node(), 0, src0, 0);
  g->AddEdge(g->source_node(), 1, src1, 0);
  // Add shape sizes
  AttrValue attr_input_shape;
  TensorShapeProto* proto = attr_input_shape.mutable_list()->add_shape();
  proto->add_dim()->set_size(m);
  proto->add_dim()->set_size(k);
  proto = attr_input_shape.mutable_list()->add_shape();
  proto->add_dim()->set_size(k);
  proto->add_dim()->set_size(n);

  Node* ret = nullptr;
  TF_CHECK_OK(NodeBuilder(g->NewName("matmul"), "MatMul")
                  .Input(src0)
                  .Input(src1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Attr("_input_shapes", attr_input_shape)
                  .Finalize(g, &ret));
#ifdef INTEL_MKL
  if (IsMKLEnabled()) {
    std::unique_ptr<Graph>* ug = new std::unique_ptr<Graph>(g);
    RunMklLayoutRewritePass(ug);
  }
#endif  // INTEL_MKL
  return g;
}

#define BM_oneDNNMatmulDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)               \
  static void                                                                \
      BM_oneDNNMatmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
          ::testing::benchmark::State& state) {                              \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(state); \
    state.SetItemsProcessed(state.iterations() * M * K * N * 2);             \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_oneDNNMatmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE) \
      ->MeasureProcessCPUTime();

#define BM_Matmul(M, K, N, TA, TB)                           \
  BM_oneDNNMatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu); \
  BM_oneDNNMatmulDev(M, K, N, TA, TB, bfloat16, DT_BFLOAT16, cpu);

// Benchmarking the same matmul shapes as `matmul_op_test.cc`
// LINT.IfChange

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(1, 1024, 1024, false, false);
BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(1, 1024, 1024, false, true);
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(1, 200, 10000, false, false);
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(1, 10000, 200, false, true);
BM_Matmul(1, 10000, 200, false, false);
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_Matmul(50, 50, 1, false, false);
BM_Matmul(50, 50, 1, true, false);
BM_Matmul(50, 50, 1, false, true);
BM_Matmul(50, 50, 1, true, true);
BM_Matmul(500, 500, 1, false, false);
BM_Matmul(500, 500, 1, true, false);
BM_Matmul(500, 500, 1, false, true);
BM_Matmul(500, 500, 1, true, true);
BM_Matmul(2000, 2000, 1, false, false);
BM_Matmul(2000, 2000, 1, true, false);
BM_Matmul(2000, 2000, 1, false, true);
BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 50, 50, true, false);
BM_Matmul(1, 50, 50, false, true);
BM_Matmul(1, 50, 50, true, true);
BM_Matmul(1, 500, 500, false, false);
BM_Matmul(1, 500, 500, true, false);
BM_Matmul(1, 500, 500, false, true);
BM_Matmul(1, 500, 500, true, true);
BM_Matmul(1, 2000, 2000, false, false);
BM_Matmul(1, 2000, 2000, true, false);
BM_Matmul(1, 2000, 2000, false, true);
BM_Matmul(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_Matmul(50, 1, 50, false, false);
BM_Matmul(50, 1, 50, true, false);
BM_Matmul(50, 1, 50, false, true);
BM_Matmul(50, 1, 50, true, true);
BM_Matmul(500, 1, 500, false, false);
BM_Matmul(500, 1, 500, true, false);
BM_Matmul(500, 1, 500, false, true);
BM_Matmul(500, 1, 500, true, true);
BM_Matmul(2000, 1, 2000, false, false);
BM_Matmul(2000, 1, 2000, true, false);
BM_Matmul(2000, 1, 2000, false, true);
BM_Matmul(2000, 1, 2000, true, true);

// LINT.ThenChange(//tensorflow/core/kernels/matmul_op_test.cc)

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
