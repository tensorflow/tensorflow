#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Creates a Graph which applies a unary "func" on a 3D float tensor
// of "num" elements.
static Graph* Unary(const string& func, int num) {
  RequireDefaultOps();
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  CHECK_GT(data.NumElements(), 0);
  data.flat<float>().setRandom();
  test::graph::Unary(g, func, test::graph::Constant(g, data), 0);
  return g;
}

static int kRows = 100000;

static int RowsAndColsArg(int r, int c) { return r * kRows + c; }
static int RowsFromArg(int arg) { return (arg / kRows); }
static int ColsFromArg(int arg) { return (arg % kRows); }

#define BM_UNARY(DEVICE, FUNC)                              \
  static void BM_##DEVICE##_##FUNC(int iters, int num) {    \
    const int64 tot = static_cast<int64>(iters) * num;      \
    testing::ItemsProcessed(tot);                           \
    testing::BytesProcessed(tot * sizeof(float));           \
    test::Benchmark(#DEVICE, Unary(#FUNC, num)).Run(iters); \
  }                                                         \
  BENCHMARK(BM_##DEVICE##_##FUNC)->Range(4 << 10, 1 << 20);

BM_UNARY(cpu, Floor);
BM_UNARY(gpu, Floor);

// data func scalar.
static Graph* BinaryScalar(int num, const string& func) {
  RequireDefaultOps();
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  lhs.flat<float>().setRandom();
  Tensor rhs(DT_FLOAT, TensorShape({}));
  rhs.flat<float>().setRandom();
  test::graph::Binary(g, func, test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BINARY_SCALAR(DEVICE, FUNC)                             \
  static void BM_##DEVICE##_##FUNC##_scalar(int iters, int num) {  \
    const int64 tot = static_cast<int64>(iters) * num;             \
    testing::ItemsProcessed(tot);                                  \
    testing::BytesProcessed(tot * sizeof(float));                  \
    test::Benchmark(#DEVICE, BinaryScalar(num, #FUNC)).Run(iters); \
  }                                                                \
  BENCHMARK(BM_##DEVICE##_##FUNC##_scalar)                         \
      ->Arg(4096) /* must >= 4096 */                               \
      ->Arg(32768)                                                 \
      ->Arg(131072)                                                \
      ->Arg(1048576);

BM_BINARY_SCALAR(cpu, Less);
BM_BINARY_SCALAR(gpu, Less);
BM_BINARY_SCALAR(cpu, Add);
BM_BINARY_SCALAR(gpu, Add);
#undef BM_BINARY_SCALAR

static Graph* BiasAdd(int rows, int cols) {
  RequireDefaultOps();
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({rows, cols}));
  lhs.flat<float>().setRandom();
  TensorShape rhs_shape;
  rhs_shape = TensorShape({cols});
  Tensor rhs(DT_FLOAT, rhs_shape);
  rhs.flat<float>().setRandom();
  test::graph::Binary(g, "BiasAdd", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BIAS_ADD(DEVICE, R, C)                                     \
  static void BM_##DEVICE##_BiasAdd_R##R##_C##C(int iters, int arg) { \
    const int rows = RowsFromArg(arg);                                \
    const int cols = ColsFromArg(arg);                                \
    const int64 tot = static_cast<int64>(iters) * rows * cols;        \
    testing::ItemsProcessed(tot);                                     \
    testing::BytesProcessed(tot * sizeof(float));                     \
    test::Benchmark(#DEVICE, BiasAdd(rows, cols)).Run(iters);         \
  }                                                                   \
  BENCHMARK(BM_##DEVICE##_BiasAdd_R##R##_C##C)->Arg(RowsAndColsArg(R, C));

#define BM_BIAS_ADD_ALL(DEVICE)   \
  BM_BIAS_ADD(DEVICE, 512, 2048); \
  BM_BIAS_ADD(DEVICE, 512, 4096); \
  BM_BIAS_ADD(DEVICE, 2048, 512); \
  BM_BIAS_ADD(DEVICE, 4096, 512);

BM_BIAS_ADD_ALL(cpu);
BM_BIAS_ADD_ALL(gpu);
#undef BM_BIAS_ADD_ALL
#undef BM_BIAS_ADD

static Graph* BcastAdd(int rows, int cols, int dim) {
  RequireDefaultOps();
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({rows, cols}));
  lhs.flat<float>().setRandom();
  TensorShape rhs_shape;
  if (dim == 0) {
    rhs_shape = TensorShape({rows, 1});
  } else {
    rhs_shape = TensorShape({cols});
  }
  Tensor rhs(DT_FLOAT, rhs_shape);
  rhs.flat<float>().setRandom();
  test::graph::Binary(g, "Add", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BCAST_ADD_ROW(DEVICE, R, C)                                    \
  static void BM_##DEVICE##_BcastAddRow_R##R##_C##C(int iters, int arg) { \
    const int rows = RowsFromArg(arg);                                    \
    const int cols = ColsFromArg(arg);                                    \
    const int64 tot = static_cast<int64>(iters) * rows * cols;            \
    testing::ItemsProcessed(tot);                                         \
    testing::BytesProcessed(tot * sizeof(float));                         \
    test::Benchmark(#DEVICE, BcastAdd(rows, cols, 0)).Run(iters);         \
  }                                                                       \
  BENCHMARK(BM_##DEVICE##_BcastAddRow_R##R##_C##C)->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_ROW_ALL(DEVICE)   \
  BM_BCAST_ADD_ROW(DEVICE, 512, 2048); \
  BM_BCAST_ADD_ROW(DEVICE, 512, 4096); \
  BM_BCAST_ADD_ROW(DEVICE, 2048, 512); \
  BM_BCAST_ADD_ROW(DEVICE, 4096, 512);
BM_BCAST_ADD_ROW_ALL(cpu);
BM_BCAST_ADD_ROW_ALL(gpu);
#undef BM_BCAST_ADD_ROW_ALL
#undef BM_BCAST_ADD_ROW

#define BM_BCAST_ADD_COL(DEVICE, R, C)                                    \
  static void BM_##DEVICE##_BcastAddCol_R##R##_C##C(int iters, int arg) { \
    const int rows = RowsFromArg(arg);                                    \
    const int cols = ColsFromArg(arg);                                    \
    const int64 tot = static_cast<int64>(iters) * rows * cols;            \
    testing::ItemsProcessed(tot);                                         \
    testing::BytesProcessed(tot * sizeof(float));                         \
    test::Benchmark(#DEVICE, BcastAdd(rows, cols, 1)).Run(iters);         \
  }                                                                       \
  BENCHMARK(BM_##DEVICE##_BcastAddCol_R##R##_C##C)->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_COL_ALL(DEVICE)   \
  BM_BCAST_ADD_COL(DEVICE, 512, 2048); \
  BM_BCAST_ADD_COL(DEVICE, 512, 4096); \
  BM_BCAST_ADD_COL(DEVICE, 2048, 512); \
  BM_BCAST_ADD_COL(DEVICE, 4096, 512);
BM_BCAST_ADD_COL_ALL(cpu);
BM_BCAST_ADD_COL_ALL(gpu);
#undef BM_BCAST_ADD_COL_ALL
#undef BM_BCAST_ADD_COL

}  // end namespace tensorflow
