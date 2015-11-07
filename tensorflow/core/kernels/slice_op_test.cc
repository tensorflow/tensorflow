#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
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

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension"
template <typename T>
static void SliceHelper(int iters, int size) {
  testing::StopTiming();
  RequireDefaultOps();
  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  int kDim = 100;
  int kMaxSize = 15000;
  CHECK_LT(size, kMaxSize);

  Tensor begin(DT_INT32, TensorShape({2}));
  begin.flat<int32>()(0) = 10;
  begin.flat<int32>()(1) = 10;

  Tensor sizes(DT_INT32, TensorShape({2}));
  sizes.flat<int32>()(0) = kDim;
  sizes.flat<int32>()(1) = size;

  Tensor input(dt, TensorShape({2 * kDim, kMaxSize}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Slice")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, begin))
                  .Input(test::graph::Constant(g, sizes))
                  .Attr("T", dt)
                  .Finalize(g, &node));

  testing::BytesProcessed(static_cast<int64>(iters) * kDim * size * sizeof(T));
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
  testing::UseRealTime();
}

static void BM_SliceFloat(int iters, int dim2) {
  SliceHelper<float>(iters, dim2);
}

BENCHMARK(BM_SliceFloat)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SliceBFloat16(int iters, int dim2) {
  SliceHelper<bfloat16>(iters, dim2);
}

BENCHMARK(BM_SliceBFloat16)->Arg(100)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow
