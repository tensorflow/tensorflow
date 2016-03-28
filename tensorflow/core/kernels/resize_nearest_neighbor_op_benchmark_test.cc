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
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* BM_ResizeNearestNeighbor(int batches, int width, int height) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(DT_FLOAT, TensorShape({batches, width, height, 3}));
  in.flat<float>().setRandom();

  Tensor out_size(DT_INT32, TensorShape({2}));
  auto out_size_flat = out_size.flat<int32>();
  out_size_flat(0) = width * 2;
  out_size_flat(1) = height * 2;

  Node* ret;
  NodeBuilder(g->NewName("n"), "ResizeNearestNeighbor")
      .Input(test::graph::Constant(g, in))
      .Input(test::graph::Constant(g, out_size))
      .Finalize(g, &ret);
  return g;
}

#define BM_ResizeNearestNeighborDev(DEVICE, B, W, H)                           \
  static void BM_ResizeNearestNeighbor_##DEVICE##_##B##_##W##_##H(int iters) { \
    testing::ItemsProcessed(iters* B* W* H * 3);                               \
    test::Benchmark(#DEVICE, BM_ResizeNearestNeighbor(B, W, H)).Run(iters);    \
  }                                                                            \
  BENCHMARK(BM_ResizeNearestNeighbor_##DEVICE##_##B##_##W##_##H)

BM_ResizeNearestNeighborDev(cpu, 1, 499, 499);
BM_ResizeNearestNeighborDev(gpu, 1, 499, 499);

}  // namespace tensorflow
