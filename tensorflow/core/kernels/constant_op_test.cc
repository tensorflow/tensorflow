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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Returns graph containing "num" const nodes.  If 'sequential' is
// true, make sure all constants are executed sequentially in the
// graph by adding control dependencies.
static Graph* ManyConsts(int num, bool sequential) {
  Graph* g = new Graph(OpRegistry::Global());
  Node* prev = nullptr;
  for (int i = 0; i < num; ++i) {
    Tensor c(DT_FLOAT, TensorShape({}));
    c.scalar<float>()() = i;
    Node* curr = test::graph::Constant(g, c);
    if (sequential && prev != nullptr) {
      g->AddControlEdge(prev, curr);
    }
    prev = curr;
  }
  return g;
}

static void BM_ManyConsts_Parallel(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  test::Benchmark("cpu", ManyConsts(num, false /* !sequential */)).Run(iters);
}
BENCHMARK(BM_ManyConsts_Parallel)->Range(1, 1 << 10);

static void BM_ManyConsts_Sequential(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  test::Benchmark("cpu", ManyConsts(num, true /* sequential */)).Run(iters);
}
BENCHMARK(BM_ManyConsts_Sequential)->Range(1, 1 << 10);

}  // end namespace tensorflow
