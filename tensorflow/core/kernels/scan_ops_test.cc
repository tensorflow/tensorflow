/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

template <typename T>
static Graph* LargeOneDCumsum(int num_x, bool reverse = false) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<T>::value, TensorShape({num_x}));
  data.flat<T>().setRandom();
  Tensor axes(DT_INT32, TensorShape({}));
  axes.flat<int32>()(0) = 0;
  test::graph::Cumsum(g, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ColCumsum(int num_x, int num_y, bool reverse = false) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({}));
  axes.flat<int32>()(0) = 0;
  test::graph::Cumsum(g, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* RowCumsum(int num_x, int num_y, bool reverse = false) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({num_x, num_y}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({}));
  axes.flat<int32>()(0) = 1;
  test::graph::Cumsum(g, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

static Graph* ThreeDYCumsum(int num_y, int num_z, bool reverse = false) {
  auto* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({32, num_y, num_z}));
  data.flat<float>().setRandom();
  Tensor axes(DT_INT32, TensorShape({}));
  axes.flat<int32>()(0) = 1;
  test::graph::Cumsum(g, test::graph::Constant(g, data),
                      test::graph::Constant(g, axes));
  return g;
}

template <typename T>
static void LargeOneDimensional(int iters, const string& device, int num_x,
                                bool reverse = false) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * sizeof(T));
  test::Benchmark(device, LargeOneDCumsum<T>(num_x, reverse)).Run(iters);
}

static void DoRowCumsum(int iters, const string& device, int num_x, int num_y,
                        bool reverse = false) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, RowCumsum(num_x, num_y, reverse)).Run(iters);
}

static void DoColCumsum(int iters, const string& device, int num_x, int num_y,
                        bool reverse = false) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, ColCumsum(num_x, num_y, reverse)).Run(iters);
}

static void Do3DYCumsum(int iters, const string& device, int num_x, int num_y,
                        bool reverse = false) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num_x * num_y);
  testing::BytesProcessed(static_cast<int64>(iters) * num_x * num_y *
                          sizeof(float));
  test::Benchmark(device, ThreeDYCumsum(num_x, num_y, reverse)).Run(iters);
}

static void BM_OneDCumsumGPU(int iters, int num_x) {
  LargeOneDimensional<float>(iters, "gpu", num_x);
}
BENCHMARK(BM_OneDCumsumGPU)->Range(1, 1 << 21);

static void BM_OneDCumsumGPUHalf(int iters, int num_x) {
  LargeOneDimensional<Eigen::half>(iters, "gpu", num_x);
}
BENCHMARK(BM_OneDCumsumGPUHalf)->Range(1, 1 << 21);

static void BM_Sum2DRowCumsumGPU(int iters, int num_x, int num_y) {
  DoRowCumsum(iters, "gpu", num_x, num_y);
}
BENCHMARK(BM_Sum2DRowCumsumGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DColumnCumsumGPU(int iters, int num_x, int num_y) {
  DoColCumsum(iters, "gpu", num_x, num_y);
}
BENCHMARK(BM_Sum2DColumnCumsumGPU)->RangePair(1, 8192, 1, 8192);

static void BM_Sum3DYCumsumGPU(int iters, int num_x, int num_y) {
  Do3DYCumsum(iters, "gpu", num_x, num_y);
}
BENCHMARK(BM_Sum3DYCumsumGPU)->RangePair(64, 4096, 64, 4096);

static void BM_OneDCumsumGPU_reverse(int iters, int num_x) {
  LargeOneDimensional<float>(iters, "gpu", num_x, true);
}
BENCHMARK(BM_OneDCumsumGPU_reverse)->Range(1, 1 << 21);

static void BM_Sum2DRowCumsumGPU_reverse(int iters, int num_x, int num_y) {
  DoRowCumsum(iters, "gpu", num_x, num_y, true);
}
BENCHMARK(BM_Sum2DRowCumsumGPU_reverse)->RangePair(1, 8192, 1, 8192);

static void BM_Sum2DColumnCumsumGPU_reverse(int iters, int num_x, int num_y) {
  DoColCumsum(iters, "gpu", num_x, num_y, true);
}
BENCHMARK(BM_Sum2DColumnCumsumGPU_reverse)->RangePair(1, 8192, 1, 8192);

static void BM_Sum3DYCumsumGPU_reverse(int iters, int num_x, int num_y) {
  Do3DYCumsum(iters, "gpu", num_x, num_y, true);
}
BENCHMARK(BM_Sum3DYCumsumGPU_reverse)->RangePair(32, 2048, 32, 2048);

}  // end namespace tensorflow
