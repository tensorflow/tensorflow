/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <typename T>
static Graph* InTopK(int num_targets, int num_classes, T top_k) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;

  Tensor predictions_t(DT_FLOAT, TensorShape({num_targets, num_classes}));
  predictions_t.flat<float>().setRandom();

  Tensor targets_t(dtype, TensorShape({num_targets}));
  targets_t.flat<T>().setRandom();

  Tensor k_t(dtype, TensorShape({}));
  k_t.scalar<T>() = k_t.scalar<T>().constant(top_k);

  Node* predictions = test::graph::Constant(g, predictions_t, "predictions");
  Node* targets = test::graph::Constant(g, targets_t, "targets");
  Node* k = test::graph::Constant(g, k_t, "k");

  Node* in_topk;
  TF_CHECK_OK(NodeBuilder(g->NewName("in_topk"), "InTopKV2")
                  .Input(predictions)
                  .Input(targets)
                  .Input(k)
                  .Attr("T", dtype)
                  .Finalize(g, &in_topk));

  return g;
}

#define BM_NAME(T, TARGETS, CLASSES, K, DEVICE) \
  BM_InTopK##_##T##_##TARGETS##_##CLASSES##_##K##_##DEVICE

#define BM_InTopK(T, TARGETS, CLASSES, K, DEVICE)                              \
  static void BM_NAME(T, TARGETS, CLASSES, K,                                  \
                      DEVICE)(::testing::benchmark::State & state) {           \
    test::Benchmark(#DEVICE, InTopK<T>(TARGETS, CLASSES, K),                   \
                    /*old_benchmark_api=*/false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64>(state.iterations()) * TARGETS * \
                            CLASSES);                                          \
  }                                                                            \
  BENCHMARK(BM_NAME(T, TARGETS, CLASSES, K, DEVICE))->UseRealTime();

BM_InTopK(int64, 64, 1000, 10, cpu);
BM_InTopK(int64, 64, 10000, 10, cpu);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
BM_InTopK(int64, 64, 1000, 10, gpu);
BM_InTopK(int64, 64, 10000, 10, gpu);
#endif  // defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)

}  // namespace tensorflow
