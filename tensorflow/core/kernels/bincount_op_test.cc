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
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* Bincount(int arr_size, int nbins) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor arr(DT_INT32, TensorShape({arr_size}));
  arr.flat<int32>() = arr.flat<int32>().setRandom().abs();

  Tensor size(DT_INT32, TensorShape({static_cast<int32>(1)}));
  size.flat<int32>()(0) = static_cast<int32>(nbins);

  Tensor weights(DT_INT32, TensorShape({0}));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Bincount")
                  .Input(test::graph::Constant(g, arr))
                  .Input(test::graph::Constant(g, size))
                  .Input(test::graph::Constant(g, weights))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  return g;
}

#define BM_BincountDev(K, NBINS, type)                             \
  static void BM_Bincount##_##type##_##K##_##NBINS(int iters) {    \
    testing::ItemsProcessed(static_cast<int64>(iters) * K * 1024); \
    test::Benchmark(#type, Bincount(K * 1024, NBINS)).Run(iters);  \
  }                                                                \
  BENCHMARK(BM_Bincount##_##type##_##K##_##NBINS);

BM_BincountDev(32, 1000, cpu);
BM_BincountDev(32, 2000, cpu);
BM_BincountDev(32, 5000, cpu);
BM_BincountDev(64, 1000, cpu);
BM_BincountDev(64, 2000, cpu);
BM_BincountDev(64, 5000, cpu);
BM_BincountDev(128, 1000, cpu);
BM_BincountDev(128, 2000, cpu);
BM_BincountDev(128, 5000, cpu);

BM_BincountDev(32, 1000, gpu);
BM_BincountDev(32, 2000, gpu);
BM_BincountDev(32, 5000, gpu);
BM_BincountDev(64, 1000, gpu);
BM_BincountDev(64, 2000, gpu);
BM_BincountDev(64, 5000, gpu);
BM_BincountDev(128, 1000, gpu);
BM_BincountDev(128, 2000, gpu);
BM_BincountDev(128, 5000, gpu);

}  // end namespace tensorflow
