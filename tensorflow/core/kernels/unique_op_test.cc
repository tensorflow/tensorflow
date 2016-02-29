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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

static void BM_Unique(int iters, int dim) {
  testing::StopTiming();
  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim}));
  input.flat<int32>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));

  testing::BytesProcessed(static_cast<int64>(iters) * dim * sizeof(int32));
  testing::UseRealTime();
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
}

BENCHMARK(BM_Unique)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4 * 1024)
    ->Arg(16 * 1024)
    ->Arg(64 * 1024)
    ->Arg(256 * 1024);

}  // namespace
}  // namespace tensorflow
