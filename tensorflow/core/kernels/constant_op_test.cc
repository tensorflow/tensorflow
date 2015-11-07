#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

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
