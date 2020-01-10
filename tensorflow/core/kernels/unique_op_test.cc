#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

namespace {

static void BM_Unique(int iters, int dim) {
  testing::StopTiming();
  RequireDefaultOps();
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
