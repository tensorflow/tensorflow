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

#include <initializer_list>
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

static Graph* MakeGraph(int split_dim, int num_split,
                        std::initializer_list<int64_t> chunk_size) {
  Graph* g = new Graph(OpRegistry::Global());
  TensorShape in_shape(chunk_size);
  in_shape.set_dim(split_dim, in_shape.dim_size(split_dim) * num_split);
  Tensor in(DataTypeToEnum<float>::value, in_shape);
  in.flat<float>().setRandom();
  Tensor split_dim_tensor = test::AsScalar<int32>(split_dim);
  Node* split;
  TF_CHECK_OK(NodeBuilder(g->NewName("split"), "Split")
                  .Input(test::graph::Constant(g, split_dim_tensor))
                  .Input(test::graph::Constant(g, in))
                  .Attr("num_split", num_split)
                  .Finalize(g, &split));
  return g;
}

#define BM_SPLIT_1D(num_split, chunk_size)                                  \
  static void BM_Split_1d_##num_split##_##chunk_size(                       \
      ::testing::benchmark::State& state) {                                 \
    auto label =                                                            \
        strings::Printf("1-D %d chunks of %d each", num_split, chunk_size); \
    state.SetLabel(label);                                                  \
    auto g = MakeGraph(/* split_dim = */ 0, num_split, {chunk_size});       \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);      \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *      \
                            num_split * chunk_size);                        \
  }                                                                         \
  BENCHMARK(BM_Split_1d_##num_split##_##chunk_size)->UseRealTime();

#define BM_SPLIT_2D(split_dim, num_split, chunk_size0, chunk_size1)          \
  static void                                                                \
      BM_Split_2d_##split_dim##_##num_split##_##chunk_size0##_##chunk_size1( \
          ::testing::benchmark::State& state) {                              \
    auto label =                                                             \
        strings::Printf("2-D %d chunks in dim %d of (%d * %d) each",         \
                        num_split, split_dim, chunk_size0, chunk_size1);     \
    state.SetLabel(label);                                                   \
    auto g = MakeGraph(split_dim, num_split, {chunk_size0, chunk_size1});    \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);       \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *       \
                            num_split * chunk_size0 * chunk_size1);          \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_Split_2d_##split_dim##_##num_split##_##chunk_size0##_##chunk_size1) \
      ->UseRealTime();

BM_SPLIT_1D(5, 1);
BM_SPLIT_1D(262144, 1);
BM_SPLIT_1D(1, 100000);
BM_SPLIT_1D(5, 100000);
BM_SPLIT_1D(10, 4194304);
BM_SPLIT_1D(2, 4194304);
BM_SPLIT_1D(100, 1024);
BM_SPLIT_1D(32768, 1024);

BM_SPLIT_2D(0, 1024, 1, 10);
BM_SPLIT_2D(0, 1024, 10, 10);
BM_SPLIT_2D(0, 512, 1024, 256);
BM_SPLIT_2D(0, 20, 100000, 5);
BM_SPLIT_2D(0, 2, 3, 524288);
BM_SPLIT_2D(0, 100, 4096, 512);

BM_SPLIT_2D(1, 1024, 1, 10);
BM_SPLIT_2D(1, 1024, 10, 10);
BM_SPLIT_2D(1, 512, 1024, 256);
BM_SPLIT_2D(1, 20, 100000, 5);
BM_SPLIT_2D(1, 2, 3, 524288);
BM_SPLIT_2D(1, 100, 4096, 512);

}  // namespace tensorflow
