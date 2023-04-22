/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

Graph* BM_ScalarSummaryOp(TensorShape shape, std::string tag, float value) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor tags(DT_STRING, shape);
  Tensor values(DT_FLOAT, shape);
  for (int i = 0; i < tags.NumElements(); ++i) {
    tags.flat<tstring>()(i) = tag;
    values.flat<float>()(i) = value;
  }
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("dummy"), "ScalarSummary")
                  .Input(test::graph::Constant(g, tags))
                  .Input(test::graph::Constant(g, values))
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &ret));
  return g;
}

// Macro used to parse initializer list for tensorshape
#define DIMARGS(...) \
  { __VA_ARGS__ }
// // Random parameters for testing
constexpr char longTagParam[] = "LONGTAG____________________________";
constexpr float largeValueParam = 2352352.2623433;

#define BM_ScalarSummaryDev(device, dims, name, tag, value)                 \
  void BM_ScalarSummary##name##device(::testing::benchmark::State& state) { \
    TensorShape tensorshape(DIMARGS dims);                                  \
    auto g = BM_ScalarSummaryOp(tensorshape, #tag, value);                  \
    test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);      \
  }                                                                         \
  BENCHMARK(BM_ScalarSummary##name##device);

BM_ScalarSummaryDev(Cpu, (5, 10, 100), Base, Tag, 5.2);
// Benchmark for large shapes
BM_ScalarSummaryDev(Cpu, (500, 100, 100), LargeShape, Tag, 5.2);
// Benchmark for large tag tstring
BM_ScalarSummaryDev(Cpu, (5, 10, 100), LongTag, longTagParam, 5.2);
// Benchmark for large values
BM_ScalarSummaryDev(Cpu, (500, 100, 100), LargeValue, Tag, largeValueParam);

}  // namespace
}  // namespace tensorflow
