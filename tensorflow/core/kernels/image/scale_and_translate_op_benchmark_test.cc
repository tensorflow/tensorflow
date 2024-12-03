/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/platform/test_benchmark.h"

namespace tensorflow {
namespace {

void BM_ScaleAndTranslateOp(benchmark::State& state) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor in(DT_FLOAT, TensorShape({1, 768, 768, 3}));
  in.flat<float>().setRandom();
  Tensor size(DT_INT32, TensorShape({2}));
  size.flat<int32>()(0) = 772;
  size.flat<int32>()(1) = 772;
  Tensor scale(DT_FLOAT, TensorShape({2}));
  scale.flat<float>()(0) = 1.0052;
  scale.flat<float>()(1) = 1.0052;
  Tensor translate(DT_FLOAT, TensorShape({2}));
  translate.flat<float>()(0) = 0.0;
  translate.flat<float>()(1) = 0.0;
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "ScaleAndTranslate")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, size))
                  .Input(test::graph::Constant(g, scale))
                  .Input(test::graph::Constant(g, translate))
                  .Attr("antialias", true)
                  .Finalize(g, &ret));
  test::Benchmark("cpu", g).Run(state);
}

BENCHMARK(BM_ScaleAndTranslateOp)->UseRealTime()->MeasureProcessCPUTime();

}  // namespace
}  // namespace tensorflow
