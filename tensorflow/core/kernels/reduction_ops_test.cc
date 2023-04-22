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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Creates a Graph which "reduce"s a 3D float tensor of "num" elements
// into a scalar.
template <typename T>
static Graph* ToScalar(const string& reduce, int num_x, int num_y) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<T>::value, TensorShape({num_x, num_y}));
  data.flat<T>().setRandom();
  Tensor axes(DT_INT32, TensorShape({2}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ColReduce(const string& reduce, int num_x, int num_y) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 0;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* RowReduce(const string& reduce, int num_x, int num_y) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDYReduce(const string& reduce, int num_y, int num_z) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({4, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = 1;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDXZReduce(const string& reduce, int num_y, int num_z) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({4, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({2}));
  axes.flat<int32>()(0) = 0;
  axes.flat<int32>()(1) = 2;
  test::graph::Reduce(g, reduce, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

// Creates a bench which reduces a 3D tensor with total "num" floats
// into a scalar on a "device". Runs the bench for "iters" times.
template <typename T>
static void ReduceToScalar(::testing::benchmark::State& state,
                           const string& device, const string& reduce,
                           int num_x, int num_y) {
  test::Benchmark(device, ToScalar<T>(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y * sizeof(T));
}

static void DoRowReduce(::testing::benchmark::State& state,
                        const string& device, const string& reduce, int num_x,
                        int num_y) {
  test::Benchmark(device, RowReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void DoColReduce(::testing::benchmark::State& state,
                        const string& device, const string& reduce, int num_x,
                        int num_y) {
  test::Benchmark(device, ColReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void Do3DYReduce(::testing::benchmark::State& state,
                        const string& device, const string& reduce, int num_x,
                        int num_y) {
  test::Benchmark(device, ThreeDYReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void Do3DXZReduce(::testing::benchmark::State& state,
                         const string& device, const string& reduce, int num_x,
                         int num_y) {
  test::Benchmark(device, ThreeDXZReduce(reduce, num_x, num_y),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * num_x *
                          num_y * sizeof(float));
}

static void BM_Sum2DToScalarGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DToScalarGPUComplex(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<std::complex<float>>(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPUComplex)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DToScalarGPUHalf(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<Eigen::half>(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DToScalarGPUHalf)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DRowReduceGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  DoRowReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DRowReduceGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DColumnReduceGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  DoColReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum2DColumnReduceGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum3DYReduceGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  Do3DYReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum3DYReduceGPU)->RangePair(64, 4096, 64, 4096);

static void BM_Sum3DXZReduceGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  Do3DXZReduce(state, "gpu", "Sum", num_x, num_y);
}
BENCHMARK(BM_Sum3DXZReduceGPU)->RangePair(64, 4096, 64, 4096);

static void BM_Mean2DToScalarGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Mean", num_x, num_y);
}
BENCHMARK(BM_Mean2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_EuclideanNorm2DToScalarGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "EuclideanNorm", num_x, num_y);
}
BENCHMARK(BM_EuclideanNorm2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Max2DToScalarGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Max", num_x, num_y);
}
BENCHMARK(BM_Max2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Min2DToScalarGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<float>(state, "gpu", "Min", num_x, num_y);
}
BENCHMARK(BM_Min2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

static void BM_Min2DToScalarGPUHalf(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<Eigen::half>(state, "gpu", "Min", num_x, num_y);
}
BENCHMARK(BM_Min2DToScalarGPUHalf)->RangePair(2048, 8192, 2048, 8192);

static void BM_Bool2DToScalarGPU(::testing::benchmark::State& state) {
  const int num_x = state.range(0);
  const int num_y = state.range(1);

  ReduceToScalar<bool>(state, "gpu", "All", num_x, num_y);
}
BENCHMARK(BM_Bool2DToScalarGPU)->RangePair(2048, 8192, 2048, 8192);

}  // end namespace tensorflow
