/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stdlib.h>
#include <initializer_list>
#include <iterator>
#include <vector>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Generate "count" random positive integers (not including zero) with sum
// "sum". Technique based on one from https://math.stackexchange.com/a/1276225
// but simplified (especially for zero-based indexing).
static std::vector<int64> GenerateRandomIntsWithSum(int64 sum, int count) {
  CHECK_GE(count, 1);
  CHECK_GE(sum, count);
  std::vector<int64> temp(count);
  for (int i = 0; i + 1 < count; ++i) {
    temp[i] = lrand48() % (sum - count);
  }
  temp[count - 1] = sum - count;
  std::sort(temp.begin(), std::prev(temp.end()));
  std::vector<int64> result(count);
  std::adjacent_difference(temp.begin(), temp.end(), result.begin());
  for (int i = 0; i < count; ++i) {
    ++result[i];
  }
  CHECK(std::all_of(result.begin(), result.end(),
                    [sum](int64 x) { return x >= 1 && x <= sum; }));
  CHECK_EQ(std::accumulate(result.begin(), result.end(), static_cast<int64>(0)),
           sum);
  CHECK_EQ(result.size(), count);
  return result;
}

static Graph* MakeGraph(int split_dim, const std::vector<int64>& size_splits,
                        std::initializer_list<int64> total_size) {
  Graph* g = new Graph(OpRegistry::Global());
  TensorShape in_shape(total_size);
  Tensor in(DataTypeToEnum<float>::value, in_shape);
  in.flat<float>().setRandom();
  Tensor split_dim_tensor = test::AsScalar<int32>(split_dim);
  Tensor size_splits_tensor = test::AsTensor<int64>(size_splits);
  Node* splitv;
  TF_CHECK_OK(NodeBuilder(g->NewName("splitv"), "SplitV")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, size_splits_tensor))
                  .Input(test::graph::Constant(g, split_dim_tensor))
                  .Attr("num_split", static_cast<int64>(size_splits.size()))
                  .Finalize(g, &splitv));
  return g;
}

#define BM_SPLITV_1D(num_split, total_size)                                  \
  static void BM_SplitV_1d_##num_split##_##total_size(int iters) {           \
    testing::StopTiming();                                                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * total_size);         \
    auto label =                                                             \
        strings::Printf("1-D %d chunks totaling %d", num_split, total_size); \
    testing::SetLabel(label);                                                \
    testing::UseRealTime();                                                  \
    auto g = MakeGraph(/* split_dim = */ 0,                                  \
                       GenerateRandomIntsWithSum(total_size, num_split),     \
                       {total_size});                                        \
    testing::StartTiming();                                                  \
    test::Benchmark("cpu", g).Run(iters);                                    \
  }                                                                          \
  BENCHMARK(BM_SplitV_1d_##num_split##_##total_size);

#define BM_SPLITV_2D(split_dim, num_split, total_size0, total_size1)          \
  static void                                                                 \
      BM_SplitV_2d_##split_dim##_##num_split##_##total_size0##_##total_size1( \
          int iters) {                                                        \
    testing::StopTiming();                                                    \
    std::vector<int64> total_size_vec{total_size0, total_size1};              \
    testing::ItemsProcessed(static_cast<int64>(iters) * total_size0 *         \
                            total_size1);                                     \
    auto label =                                                              \
        strings::Printf("2-D %d chunks in dim %d totaling (%d * %d)",         \
                        num_split, split_dim, total_size0, total_size1);      \
    testing::SetLabel(label);                                                 \
    testing::UseRealTime();                                                   \
    auto g = MakeGraph(                                                       \
        split_dim,                                                            \
        GenerateRandomIntsWithSum(total_size_vec[split_dim], num_split),      \
        {total_size0, total_size1});                                          \
    testing::StartTiming();                                                   \
    test::Benchmark("cpu", g).Run(iters);                                     \
  }                                                                           \
  BENCHMARK(                                                                  \
      BM_SplitV_2d_##split_dim##_##num_split##_##total_size0##_##total_size1);

BM_SPLITV_1D(5, 20);
BM_SPLITV_1D(262144, 1000000);
BM_SPLITV_1D(1, 100000);
BM_SPLITV_1D(5, 100000);
BM_SPLITV_1D(5, 250000);
BM_SPLITV_1D(5, 500000);
BM_SPLITV_1D(5, 1000000);
BM_SPLITV_1D(10, 4194304);
BM_SPLITV_1D(2, 4194304);
BM_SPLITV_1D(100, 10240);
BM_SPLITV_1D(32768, 1048576);

BM_SPLITV_2D(0, 1024, 10247, 10);
BM_SPLITV_2D(0, 1024, 100000, 10);
BM_SPLITV_2D(0, 512, 1024, 256);
BM_SPLITV_2D(0, 20, 100000, 5);
BM_SPLITV_2D(0, 2, 7, 524288);
BM_SPLITV_2D(0, 100, 4096, 512);

BM_SPLITV_2D(1, 1024, 15, 10240);
BM_SPLITV_2D(1, 1024, 10, 100000);
BM_SPLITV_2D(1, 512, 1024, 2563);
BM_SPLITV_2D(1, 20, 100000, 52);
BM_SPLITV_2D(1, 2, 3, 524288);
BM_SPLITV_2D(1, 100, 4096, 512);

}  // namespace tensorflow
