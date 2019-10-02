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
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <typename InputShape>
static Graph* BroadcastTo(int size, InputShape input_shape) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_FLOAT, input_shape(size));
  input.flat<float>() = input.flat<float>().setRandom();

  Tensor shape(DT_INT32, TensorShape({2}));
  shape.flat<int32>()(0) = size;
  shape.flat<int32>()(1) = size;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BroadcastTo")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, shape))
                  .Attr("T", DT_FLOAT)
                  .Attr("Tidx", DT_INT32)
                  .Finalize(g, &node));
  return g;
}

#define BM_BroadcastTo_InnerDim(SIZE, type)                             \
  static void BM_BroadcastTo_Inner##_##type##_##SIZE(int iters) {       \
    testing::UseRealTime();                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * SIZE * SIZE);   \
    test::Benchmark(#type, BroadcastTo(SIZE,                            \
                                       [](int size) {                   \
                                         return TensorShape({size, 1}); \
                                       }))                              \
        .Run(iters);                                                    \
  }                                                                     \
  BENCHMARK(BM_BroadcastTo_Inner##_##type##_##SIZE);

#define BM_BroadcastTo_OuterDim(SIZE, type)                             \
  static void BM_BroadcastTo_Outer##_##type##_##SIZE(int iters) {       \
    testing::UseRealTime();                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * SIZE * SIZE);   \
    test::Benchmark(#type, BroadcastTo(SIZE,                            \
                                       [](int size) {                   \
                                         return TensorShape({1, size}); \
                                       }))                              \
        .Run(iters);                                                    \
  }                                                                     \
  BENCHMARK(BM_BroadcastTo_Outer##_##type##_##SIZE);

BM_BroadcastTo_InnerDim(64, cpu);
BM_BroadcastTo_InnerDim(128, cpu);
BM_BroadcastTo_InnerDim(256, cpu);
BM_BroadcastTo_InnerDim(512, cpu);
BM_BroadcastTo_InnerDim(1024, cpu);

BM_BroadcastTo_OuterDim(64, cpu);
BM_BroadcastTo_OuterDim(128, cpu);
BM_BroadcastTo_OuterDim(256, cpu);
BM_BroadcastTo_OuterDim(512, cpu);
BM_BroadcastTo_OuterDim(1024, cpu);

}  // end namespace tensorflow
