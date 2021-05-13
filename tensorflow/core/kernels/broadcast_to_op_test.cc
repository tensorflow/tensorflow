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
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <typename InputShape>
static Graph* BroadcastTo(int dim0, int dim1, InputShape input_shape) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_FLOAT, input_shape(dim0, dim1));
  input.flat<float>() = input.flat<float>().setRandom();

  Tensor shape(DT_INT32, TensorShape({2}));
  shape.flat<int32>()(0) = dim0;
  shape.flat<int32>()(1) = dim1;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BroadcastTo")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, shape))
                  .Attr("T", DT_FLOAT)
                  .Attr("Tidx", DT_INT32)
                  .Finalize(g, &node));
  return g;
}

#define BM_BroadcastTo_InnerDim(DIM0, DIM1, type)                           \
  static void BM_BroadcastTo_Inner##_##type##_##DIM0##_##DIM1(              \
      ::testing::benchmark::State& state) {                                 \
    test::Benchmark(#type,                                                  \
                    BroadcastTo(DIM0, DIM1,                                 \
                                [](int dim0, int dim1) {                    \
                                  return TensorShape({dim0, 1});            \
                                }),                                         \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    state.SetItemsProcessed(static_cast<int64>(state.iterations()) * DIM0 * \
                            DIM1);                                          \
  }                                                                         \
  BENCHMARK(BM_BroadcastTo_Inner##_##type##_##DIM0##_##DIM1)->UseRealTime();

#define BM_BroadcastTo_OuterDim(DIM0, DIM1, type)                           \
  static void BM_BroadcastTo_Outer##_##type##_##DIM0##_##DIM1(              \
      ::testing::benchmark::State& state) {                                 \
    test::Benchmark(#type,                                                  \
                    BroadcastTo(DIM0, DIM1,                                 \
                                [](int dim0, int dim1) {                    \
                                  return TensorShape({1, dim1});            \
                                }),                                         \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    state.SetItemsProcessed(static_cast<int64>(state.iterations()) * DIM0 * \
                            DIM1);                                          \
  }                                                                         \
  BENCHMARK(BM_BroadcastTo_Outer##_##type##_##DIM0##_##DIM1)->UseRealTime();

BM_BroadcastTo_InnerDim(64, 64, cpu);
BM_BroadcastTo_InnerDim(128, 128, cpu);
BM_BroadcastTo_InnerDim(256, 256, cpu);
BM_BroadcastTo_InnerDim(512, 512, cpu);
BM_BroadcastTo_InnerDim(1024, 1024, cpu);
BM_BroadcastTo_InnerDim(500, 20000, cpu);

BM_BroadcastTo_OuterDim(64, 64, cpu);
BM_BroadcastTo_OuterDim(128, 128, cpu);
BM_BroadcastTo_OuterDim(256, 256, cpu);
BM_BroadcastTo_OuterDim(512, 512, cpu);
BM_BroadcastTo_OuterDim(1024, 1024, cpu);
BM_BroadcastTo_OuterDim(500, 20000, cpu);

}  // end namespace tensorflow
