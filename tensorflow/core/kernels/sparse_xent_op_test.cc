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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/xent_op.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <class T>
static Graph* SparseXent(int batch_size, int num_classes, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor logits(type, TensorShape({batch_size, num_classes}));
  logits.flat<T>().setRandom();
  Tensor labels(DT_INT64, TensorShape({batch_size}));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, num_classes - 1);
  auto labels_t = labels.flat<int64_t>();
  for (int i = 0; i < batch_size; ++i) {
    labels_t(i) = dist(gen);
  }
  test::graph::Binary(g, "SparseSoftmaxCrossEntropyWithLogits",
                      test::graph::Constant(g, logits),
                      test::graph::Constant(g, labels));
  return g;
}

#define BM_SparseXentDev(BATCH, CLASS, DEVICE, C_TYPE, TF_TYPE)         \
  static void BM_SparseXent##_##BATCH##_##CLASS##_##DEVICE##_##C_TYPE(  \
      ::testing::benchmark::State& state) {                             \
    test::Benchmark(#DEVICE, SparseXent<C_TYPE>(BATCH, CLASS, TF_TYPE), \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    const int64_t tot =                                                 \
        static_cast<int64_t>(state.iterations()) * BATCH * CLASS;       \
    state.SetItemsProcessed(tot);                                       \
    state.SetBytesProcessed(tot * sizeof(C_TYPE));                      \
  }                                                                     \
  BENCHMARK(BM_SparseXent##_##BATCH##_##CLASS##_##DEVICE##_##C_TYPE);

/// The representative tests for ptb_word on GPU
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
BM_SparseXentDev(8, 1000000, gpu, float, DT_FLOAT);

BM_SparseXentDev(16, 10000, gpu, float, DT_FLOAT);
BM_SparseXentDev(16, 30000, gpu, float, DT_FLOAT);
BM_SparseXentDev(16, 100000, gpu, float, DT_FLOAT);

BM_SparseXentDev(32, 10000, gpu, float, DT_FLOAT);
BM_SparseXentDev(32, 30000, gpu, float, DT_FLOAT);
BM_SparseXentDev(32, 100000, gpu, float, DT_FLOAT);

BM_SparseXentDev(64, 10000, gpu, float, DT_FLOAT);
BM_SparseXentDev(64, 30000, gpu, float, DT_FLOAT);
BM_SparseXentDev(64, 100000, gpu, float, DT_FLOAT);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// CPU
#define BM_SparseXentDev_CPU(C_TYPE, TF_TYPE)         \
  BM_SparseXentDev(8, 1000000, cpu, C_TYPE, TF_TYPE); \
  BM_SparseXentDev(16, 10000, cpu, C_TYPE, TF_TYPE);  \
  BM_SparseXentDev(16, 100000, cpu, C_TYPE, TF_TYPE); \
  BM_SparseXentDev(32, 10000, cpu, C_TYPE, TF_TYPE);  \
  BM_SparseXentDev(32, 100000, cpu, C_TYPE, TF_TYPE); \
  BM_SparseXentDev(64, 10000, cpu, C_TYPE, TF_TYPE);  \
  BM_SparseXentDev(64, 100000, cpu, C_TYPE, TF_TYPE);

BM_SparseXentDev_CPU(float, DT_FLOAT);
BM_SparseXentDev_CPU(bfloat16, DT_BFLOAT16);

}  // end namespace tensorflow
