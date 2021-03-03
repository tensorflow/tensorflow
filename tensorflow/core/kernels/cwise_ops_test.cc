/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

// Creates a Graph which applies a unary "func" on a 3D tensor of
// type T with "num" elements.
template <typename T>
static Graph* Unary(const string& func, int num, DataType dtype) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(dtype, TensorShape({64, 64, num / (64 * 64)}));
  CHECK_GT(data.NumElements(), 0);
  data.flat<T>().setRandom();
  test::graph::Unary(g, func, test::graph::Constant(g, data), 0);
  return g;
}

const int kRows = 100000;

int RowsAndColsArg(int r, int c) { return r * kRows + c; }
int RowsFromArg(int arg) { return (arg / kRows); }
int ColsFromArg(int arg) { return (arg % kRows); }

#define BM_UNARY(DEVICE, FUNC, T, TYPE)                                    \
  void BM_##DEVICE##_##FUNC##_##TYPE(::testing::benchmark::State& state) { \
    const int num = state.range(0);                                        \
    test::Benchmark(#DEVICE, Unary<T>(#FUNC, num, TYPE),                   \
                    /*old_benchmark_api*/ false)                           \
        .Run(state);                                                       \
    const int64 tot = static_cast<int64>(state.iterations()) * num;        \
    state.SetItemsProcessed(tot);                                          \
    state.SetBytesProcessed(tot * sizeof(T));                              \
  }                                                                        \
  BENCHMARK(BM_##DEVICE##_##FUNC##_##TYPE)                                 \
      ->UseRealTime()                                                      \
      ->Range(4 << 10, 1 << 20);

BM_UNARY(cpu, Floor, float, DT_FLOAT);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Floor, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

BM_UNARY(cpu, Floor, double, DT_DOUBLE);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Floor, double, DT_DOUBLE);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

BM_UNARY(cpu, Conj, std::complex<float>, DT_COMPLEX64);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Conj, std::complex<float>, DT_COMPLEX64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(cpu, Conj, std::complex<double>, DT_COMPLEX128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Conj, std::complex<double>, DT_COMPLEX128);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

BM_UNARY(cpu, Rint, double, DT_DOUBLE);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Rint, double, DT_DOUBLE);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(cpu, Rint, float, DT_FLOAT);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Rint, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

BM_UNARY(cpu, Round, double, DT_DOUBLE);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Round, double, DT_DOUBLE);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(cpu, Round, float, DT_FLOAT);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_UNARY(gpu, Round, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// data func scalar.
Graph* BinaryScalar(int num, const string& func) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  lhs.flat<float>().setRandom();
  Tensor rhs(DT_FLOAT, TensorShape({}));
  rhs.flat<float>().setRandom();
  test::graph::Binary(g, func, test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BINARY_SCALAR(DEVICE, FUNC)                                     \
  void BM_##DEVICE##_##FUNC##_scalar(::testing::benchmark::State& state) { \
    const int num = state.range(0);                                        \
                                                                           \
    test::Benchmark(#DEVICE, BinaryScalar(num, #FUNC),                     \
                    /*old_benchmark_api=*/false)                           \
        .Run(state);                                                       \
    const int64 tot = static_cast<int64>(state.iterations()) * num;        \
    state.SetItemsProcessed(tot);                                          \
    state.SetBytesProcessed(tot * sizeof(float));                          \
  }                                                                        \
  BENCHMARK(BM_##DEVICE##_##FUNC##_scalar)                                 \
      ->Arg(1 << 12) /* must >= 4096 */                                    \
      ->Arg(1 << 13)                                                       \
      ->Arg(1 << 14)                                                       \
      ->Arg((1 << 15) - (1 << 13))                                         \
      ->Arg(1 << 15)                                                       \
      ->Arg((1 << 15) + (1 << 14))                                         \
      ->Arg(1 << 16)                                                       \
      ->Arg((1 << 17) - (1 << 15))                                         \
      ->Arg(1 << 17)                                                       \
      ->Arg((1 << 17) + (1 << 16))                                         \
      ->Arg(1 << 18)                                                       \
      ->Arg(1 << 19)                                                       \
      ->Arg(1 << 20);

BM_BINARY_SCALAR(cpu, Less);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BINARY_SCALAR(gpu, Less);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

BM_BINARY_SCALAR(cpu, Add);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BINARY_SCALAR(gpu, Add);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

BM_BINARY_SCALAR(cpu, DivNoNan);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BINARY_SCALAR(gpu, DivNoNan);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef BM_BINARY_SCALAR

// Three implementations of x^3.
Graph* CubeWithPow3(int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  lhs.flat<float>().setRandom();
  Tensor rhs(DT_FLOAT, TensorShape({}));
  rhs.flat<float>().setConstant(3);
  test::graph::Binary(g, "Pow", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

Graph* CubeWithTwoMuls(int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  lhs.flat<float>().setRandom();
  auto* x = test::graph::Constant(g, lhs);
  auto* inner = test::graph::Binary(g, "Mul", x, x);
  test::graph::Binary(g, "Mul", x, inner);
  return g;
}

Graph* CubeWithMulSquare(int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(DT_FLOAT, TensorShape({64, 64, num / (64 * 64)}));
  lhs.flat<float>().setRandom();
  auto* x = test::graph::Constant(g, lhs);
  auto* inner = test::graph::Unary(g, "Square", x);
  test::graph::Binary(g, "Mul", test::graph::Constant(g, lhs), inner);
  return g;
}

#define BM_CUBE(DEVICE, Impl)                                          \
  void BM_##DEVICE##_Cube_##Impl(::testing::benchmark::State& state) { \
    const int num = state.range(0);                                    \
                                                                       \
    test::Benchmark(#DEVICE, Impl(num), /*old_benchmark_api*/ false)   \
        .Run(state);                                                   \
    const int64 tot = static_cast<int64>(state.iterations()) * num;    \
    state.SetItemsProcessed(tot);                                      \
    state.SetBytesProcessed(tot * sizeof(float));                      \
  }                                                                    \
  BENCHMARK(BM_##DEVICE##_Cube_##Impl)                                 \
      ->UseRealTime()                                                  \
      ->Arg(1 << 12) /* must >= 4096 */                                \
      ->Arg(1 << 16)                                                   \
      ->Arg(1 << 20);

BM_CUBE(cpu, CubeWithPow3);
BM_CUBE(cpu, CubeWithTwoMuls);
BM_CUBE(cpu, CubeWithMulSquare);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_CUBE(gpu, CubeWithPow3);
BM_CUBE(gpu, CubeWithTwoMuls);
BM_CUBE(gpu, CubeWithMulSquare);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef BM_CUBE

template <class T>
Graph* BiasAdd(int rows, int cols, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor lhs(type, TensorShape({rows, cols}));
  lhs.template flat<T>().setRandom();
  TensorShape rhs_shape;
  rhs_shape = TensorShape({cols});
  Tensor rhs(type, rhs_shape);
  rhs.template flat<T>().setRandom();
  test::graph::Binary(g, "BiasAdd", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, R, C)                          \
  void BM_##DEVICE##_##C_TYPE##_BiasAdd_R##R##_C##C(                        \
      ::testing::benchmark::State& state) {                                 \
    const int arg = state.range(0);                                         \
    const int rows = RowsFromArg(arg);                                      \
    const int cols = ColsFromArg(arg);                                      \
    const int64 tot = static_cast<int64>(state.iterations()) * rows * cols; \
    test::Benchmark(#DEVICE, BiasAdd<C_TYPE>(rows, cols, TF_TYPE),          \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    state.SetItemsProcessed(tot);                                           \
    state.SetBytesProcessed(tot * sizeof(C_TYPE));                          \
  }                                                                         \
  BENCHMARK(BM_##DEVICE##_##C_TYPE##_BiasAdd_R##R##_C##C)                   \
      ->UseRealTime()                                                       \
      ->Arg(RowsAndColsArg(R, C));

#define BM_BIAS_ADD_ALL(DEVICE, C_TYPE, TF_TYPE)   \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 512, 2048); \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 512, 4096); \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 2048, 512); \
  BM_BIAS_ADD(DEVICE, C_TYPE, TF_TYPE, 4096, 512);

using Eigen::half;
BM_BIAS_ADD_ALL(cpu, float, DT_FLOAT);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_ALL(gpu, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_ALL(cpu, half, DT_HALF);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_ALL(gpu, half, DT_HALF);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef BM_BIAS_ADD_ALL
#undef BM_BIAS_ADD

template <class T>
Graph* BiasAddGrad(int rows, int cols, int channels, DataType type,
                   TensorFormat format) {
  Graph* g = new Graph(OpRegistry::Global());
  TensorShape lhs_shape;
  if (format == FORMAT_NCHW) {
    lhs_shape = TensorShape({channels, rows, cols});
  } else {
    lhs_shape = TensorShape({rows, cols, channels});
  }
  Tensor lhs(type, lhs_shape);
  lhs.template flat<T>().setRandom();
  Node* n;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BiasAddGrad")
                  .Attr("data_format", ToString(format))
                  .Input(test::graph::Constant(g, lhs), /*src_index=*/0)
                  .Finalize(g, &n));
  return g;
}

#define BM_BIAS_ADD_GRAD(DEVICE, FMT, C_TYPE, TF_TYPE, R, C, CH)               \
  void BM_##DEVICE##_##FMT##_##C_TYPE##_BiasAddGrad_R##R##_C##C##_CH##CH(      \
      ::testing::benchmark::State& state) {                                    \
    const int arg = state.range(0);                                            \
    const int channels = state.range(1);                                       \
                                                                               \
    const int rows = RowsFromArg(arg);                                         \
    const int cols = ColsFromArg(arg);                                         \
    test::Benchmark(                                                           \
        #DEVICE,                                                               \
        BiasAddGrad<C_TYPE>(rows, cols, channels, TF_TYPE, FORMAT_##FMT),      \
        /*old_benchmark_api=*/false)                                           \
        .Run(state);                                                           \
    const int64 tot =                                                          \
        static_cast<int64>(state.iterations()) * rows * cols * channels;       \
    state.SetItemsProcessed(tot);                                              \
    state.SetBytesProcessed(tot * sizeof(C_TYPE));                             \
  }                                                                            \
  BENCHMARK(BM_##DEVICE##_##FMT##_##C_TYPE##_BiasAddGrad_R##R##_C##C##_CH##CH) \
      ->ArgPair(RowsAndColsArg(R, C), CH);

#define BM_BIAS_ADD_GRAD_ALL(DEVICE, FORMAT, C_TYPE, TF_TYPE)       \
  BM_BIAS_ADD_GRAD(DEVICE, FORMAT, C_TYPE, TF_TYPE, 64, 64, 64);    \
  BM_BIAS_ADD_GRAD(DEVICE, FORMAT, C_TYPE, TF_TYPE, 512, 512, 4);   \
  BM_BIAS_ADD_GRAD(DEVICE, FORMAT, C_TYPE, TF_TYPE, 512, 512, 1);   \
  BM_BIAS_ADD_GRAD(DEVICE, FORMAT, C_TYPE, TF_TYPE, 4096, 4096, 4); \
  BM_BIAS_ADD_GRAD(DEVICE, FORMAT, C_TYPE, TF_TYPE, 4096, 4096, 1);

using Eigen::half;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_GRAD_ALL(gpu, NCHW, float, DT_FLOAT);
BM_BIAS_ADD_GRAD_ALL(gpu, NCHW, half, DT_HALF);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_GRAD_ALL(cpu, NHWC, float, DT_FLOAT);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_GRAD_ALL(gpu, NHWC, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_GRAD_ALL(cpu, NHWC, half, DT_HALF);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BIAS_ADD_GRAD_ALL(gpu, NHWC, half, DT_HALF);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef BM_BIAS_ADD_GRAD_ALL
#undef BM_BIAS_ADD_GRAD

Graph* BcastAdd(int rows, int cols, int dim) {
  Graph* g = new Graph(OpRegistry::Global());
  TensorShape lhs_shape, rhs_shape;
  if (dim == 0) {  // row
    lhs_shape = TensorShape({rows, cols});
    rhs_shape = TensorShape({rows, 1});
  } else if (dim == 1) {  // col
    lhs_shape = TensorShape({rows, cols});
    rhs_shape = TensorShape({cols});
  } else if (dim == 2) {  // cross_rc
    lhs_shape = TensorShape({rows, 1});
    rhs_shape = TensorShape({1, cols});
  } else {  // cross_cr
    lhs_shape = TensorShape({1, cols});
    rhs_shape = TensorShape({rows, 1});
  }
  Tensor lhs(DT_FLOAT, lhs_shape);
  lhs.flat<float>().setRandom();
  Tensor rhs(DT_FLOAT, rhs_shape);
  rhs.flat<float>().setRandom();
  test::graph::Binary(g, "Add", test::graph::Constant(g, lhs),
                      test::graph::Constant(g, rhs));
  return g;
}

#define BM_BCAST_ADD_ROW(DEVICE, R, C)                                      \
  void BM_##DEVICE##_BcastAddRow_R##R##_C##C(                               \
      ::testing::benchmark::State& state) {                                 \
    const int arg = state.range(0);                                         \
                                                                            \
    const int rows = RowsFromArg(arg);                                      \
    const int cols = ColsFromArg(arg);                                      \
    test::Benchmark(#DEVICE, BcastAdd(rows, cols, 0),                       \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    const int64 tot = static_cast<int64>(state.iterations()) * rows * cols; \
    state.SetItemsProcessed(tot);                                           \
    state.SetBytesProcessed(tot * sizeof(float));                           \
  }                                                                         \
  BENCHMARK(BM_##DEVICE##_BcastAddRow_R##R##_C##C)->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_ROW_ALL(DEVICE)   \
  BM_BCAST_ADD_ROW(DEVICE, 512, 2048); \
  BM_BCAST_ADD_ROW(DEVICE, 512, 4096); \
  BM_BCAST_ADD_ROW(DEVICE, 2048, 512); \
  BM_BCAST_ADD_ROW(DEVICE, 4096, 512);
BM_BCAST_ADD_ROW_ALL(cpu);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BCAST_ADD_ROW_ALL(gpu);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef BM_BCAST_ADD_ROW_ALL
#undef BM_BCAST_ADD_ROW

#define BM_BCAST_ADD_COL(DEVICE, R, C)                                      \
  void BM_##DEVICE##_BcastAddCol_R##R##_C##C(                               \
      ::testing::benchmark::State& state) {                                 \
    const int arg = state.range(0);                                         \
                                                                            \
    const int rows = RowsFromArg(arg);                                      \
    const int cols = ColsFromArg(arg);                                      \
    test::Benchmark(#DEVICE, BcastAdd(rows, cols, 1),                       \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    const int64 tot = static_cast<int64>(state.iterations()) * rows * cols; \
                                                                            \
    state.SetItemsProcessed(tot);                                           \
    state.SetBytesProcessed(tot * sizeof(float));                           \
  }                                                                         \
  BENCHMARK(BM_##DEVICE##_BcastAddCol_R##R##_C##C)                          \
      ->UseRealTime()                                                       \
      ->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_COL_ALL(DEVICE)   \
  BM_BCAST_ADD_COL(DEVICE, 512, 2048); \
  BM_BCAST_ADD_COL(DEVICE, 512, 4096); \
  BM_BCAST_ADD_COL(DEVICE, 2048, 512); \
  BM_BCAST_ADD_COL(DEVICE, 4096, 512);
BM_BCAST_ADD_COL_ALL(cpu);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BCAST_ADD_COL_ALL(gpu);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef BM_BCAST_ADD_COL_ALL
#undef BM_BCAST_ADD_COL

#define BM_BCAST_ADD_CROSS_RC(DEVICE, R, C)                                 \
  void BM_##DEVICE##_BcastAddCrossRC_R##R##_C##C(                           \
      ::testing::benchmark::State& state) {                                 \
    const int arg = state.range(0);                                         \
                                                                            \
    const int rows = RowsFromArg(arg);                                      \
    const int cols = ColsFromArg(arg);                                      \
    test::Benchmark(#DEVICE, BcastAdd(rows, cols, 2),                       \
                    /*old_benchmark_api=*/false)                            \
        .Run(state);                                                        \
    const int64 tot = static_cast<int64>(state.iterations()) * rows * cols; \
                                                                            \
    state.SetItemsProcessed(tot);                                           \
    state.SetBytesProcessed(tot * sizeof(float));                           \
  }                                                                         \
  BENCHMARK(BM_##DEVICE##_BcastAddCrossRC_R##R##_C##C)                      \
      ->UseRealTime()                                                       \
      ->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_CROSS_RC_ALL(DEVICE)   \
  BM_BCAST_ADD_CROSS_RC(DEVICE, 512, 2048); \
  BM_BCAST_ADD_CROSS_RC(DEVICE, 512, 4096); \
  BM_BCAST_ADD_CROSS_RC(DEVICE, 2048, 512); \
  BM_BCAST_ADD_CROSS_RC(DEVICE, 4096, 512);
BM_BCAST_ADD_CROSS_RC_ALL(cpu);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BCAST_ADD_CROSS_RC_ALL(gpu);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef BM_BCAST_ADD_CROSS_RC_ALL
#undef BM_BCAST_ADD_CROSS_RC

#define BM_BCAST_ADD_CROSS_CR(DEVICE, R, C)                                 \
  void BM_##DEVICE##_BcastAddCrossCR_R##R##_C##C(                           \
      ::testing::benchmark::State& state) {                                 \
    const int arg = state.range(0);                                         \
                                                                            \
    const int rows = RowsFromArg(arg);                                      \
    const int cols = ColsFromArg(arg);                                      \
    test::Benchmark(#DEVICE, BcastAdd(rows, cols, 3),                       \
                    /*old_benchmark_api*/ false)                            \
        .Run(state);                                                        \
    const int64 tot = static_cast<int64>(state.iterations()) * rows * cols; \
    state.SetItemsProcessed(tot);                                           \
    state.SetBytesProcessed(tot * sizeof(float));                           \
  }                                                                         \
  BENCHMARK(BM_##DEVICE##_BcastAddCrossCR_R##R##_C##C)                      \
      ->UseRealTime()                                                       \
      ->Arg(RowsAndColsArg(R, C));

#define BM_BCAST_ADD_CROSS_CR_ALL(DEVICE)   \
  BM_BCAST_ADD_CROSS_CR(DEVICE, 512, 2048); \
  BM_BCAST_ADD_CROSS_CR(DEVICE, 512, 4096); \
  BM_BCAST_ADD_CROSS_CR(DEVICE, 2048, 512); \
  BM_BCAST_ADD_CROSS_CR(DEVICE, 4096, 512);
BM_BCAST_ADD_CROSS_CR_ALL(cpu);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_BCAST_ADD_CROSS_CR_ALL(gpu);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef BM_BCAST_ADD_CROSS_CR_ALL
#undef BM_BCAST_ADD_CROSS_CR

}  // namespace
}  // namespace tensorflow
