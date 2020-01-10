#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Creates a Graph which "reduce"s a 3D float tensor of "num" elements
// into a scalar.
static Graph* ToScalar(const string& reduce, int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({3}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 1;
  axes.flat<int32>()(2) = 2;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

// Creates a bench which reduces a 3D tensor with total "num" floats
// into a scalar on a "device". Runs the bench for "iters" times.
static void ReduceToScalar(int iters, const string& device,
                           const string& reduce, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num * sizeof(float));
  test::Benchmark(device, ToScalar(reduce, num)).Run(iters);
}

static void BM_Sum3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Sum", num);
}
BENCHMARK(BM_Sum3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Max3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Max", num);
}
BENCHMARK(BM_Max3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Prod3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Prod", num);
}
BENCHMARK(BM_Prod3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Mean3DToScalarCPU(int iters, int num) {
  ReduceToScalar(iters, "cpu", "Mean", num);
}
BENCHMARK(BM_Mean3DToScalarCPU)->Range(1 << 13, 1 << 20);

static void BM_Sum3DToScalarGPU(int iters, int num) {
  ReduceToScalar(iters, "gpu", "Sum", num);
}
BENCHMARK(BM_Sum3DToScalarGPU)->Range(1 << 13, 1 << 20);

static void BM_Max3DToScalarGPU(int iters, int num) {
  ReduceToScalar(iters, "gpu", "Max", num);
}
BENCHMARK(BM_Max3DToScalarGPU)->Range(1 << 13, 1 << 20);

static void BM_Prod3DToScalarGPU(int iters, int num) {
  ReduceToScalar(iters, "gpu", "Prod", num);
}
BENCHMARK(BM_Prod3DToScalarGPU)->Range(1 << 13, 1 << 20);

// Once Mean is available on GPU, enable this.
// static void BM_Mean3DToScalarGPU(int iters, int num) {
//   ReduceToScalar(iters, "gpu", "Mean", num);
// }
// BENCHMARK(BM_Mean3DToScalarGPU)->Range(1 << 13, 1 << 20);

}  // end namespace tensorflow
