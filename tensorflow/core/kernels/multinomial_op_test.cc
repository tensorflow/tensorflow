/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Node* Multinomial(Graph* g, Node* logits, Node* num_samples,
                         DataType data_type) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("multinomial"), "Multinomial")
                  .Input(logits)
                  .Input(num_samples)
                  .Attr("T", data_type)
                  .Finalize(g, &ret));
  return ret;
}

static Graph* Multinomial(int batch_size, int num_classes, int num_samples) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor logits_t(DT_FLOAT, TensorShape({batch_size, num_classes}));
  Tensor num_samples_t(DT_INT32, TensorShape());

  logits_t.flat<float>().setRandom();
  num_samples_t.scalar<int32>().setConstant(num_samples);

  Multinomial(g, test::graph::Constant(g, logits_t),
              test::graph::Constant(g, num_samples_t), DT_FLOAT);

  return g;
}

static void BM_Multinomial(int iters, int batch_size, int num_classes,
                           int num_samples) {
  test::Benchmark("cpu", Multinomial(batch_size, num_classes, num_samples))
      .Run(iters);
}

#define BM_MultinomialBCS(B, C, S)                        \
  static void BM_Multinomial_##B##_##C##_##S(int iters) { \
    BM_Multinomial(iters, B, C, S);                       \
  }                                                       \
  BENCHMARK(BM_Multinomial_##B##_##C##_##S);

BM_MultinomialBCS(1, 10000, 4);
BM_MultinomialBCS(1, 10000, 128);
BM_MultinomialBCS(1, 100000, 4);
BM_MultinomialBCS(1, 100000, 128);

BM_MultinomialBCS(32, 10000, 4);
BM_MultinomialBCS(32, 10000, 128);
BM_MultinomialBCS(32, 100000, 4);
BM_MultinomialBCS(32, 100000, 128);

BM_MultinomialBCS(128, 100000, 1);
BM_MultinomialBCS(128, 100000, 128);

}  // namespace tensorflow
