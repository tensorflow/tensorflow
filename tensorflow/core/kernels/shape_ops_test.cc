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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

static void BM_ExpandDims(int iters) {
  testing::StopTiming();
  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({1, 1, 1, 1}));
  input.flat<int32>()(0) = 10;

  Tensor axis(DT_INT32, TensorShape({}));
  axis.flat<int32>()(0) = 2;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "ExpandDims")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, axis))
                  .Attr("T", DT_INT32)
                  .Attr("Tdim", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  testing::StartTiming();
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR")
      .Run(iters);

  testing::UseRealTime();
}

BENCHMARK(BM_ExpandDims);

}  // namespace
}  // namespace tensorflow
