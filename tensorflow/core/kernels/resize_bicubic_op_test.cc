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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* ResizeBicubic(int batch_size, int size, int channels) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor input(DT_FLOAT, TensorShape({batch_size, size, size, channels}));
  input.flat<float>().setRandom();
  Tensor shape(DT_INT32, TensorShape({2}));
  auto shape_t = shape.flat<int32>();
  shape_t(0) = 0.3 * size;
  shape_t(1) = 0.7 * size;
  test::graph::Binary(g, "ResizeBicubic", test::graph::Constant(g, input),
                      test::graph::Constant(g, shape));
  return g;
}

#define BM_ResizeBicubicDev(BATCH, SIZE, CHANNELS)                            \
  static void BM_ResizeBicubic##_##BATCH##_##SIZE##_##CHANNELS(int iters) {   \
    testing::ItemsProcessed(static_cast<int64>(iters) * BATCH * SIZE * SIZE * \
                            CHANNELS);                                        \
    test::Benchmark("cpu", ResizeBicubic(BATCH, SIZE, CHANNELS)).Run(iters);  \
  }                                                                           \
  BENCHMARK(BM_ResizeBicubic##_##BATCH##_##SIZE##_##CHANNELS);

BM_ResizeBicubicDev(8, 32, 3);
BM_ResizeBicubicDev(8, 128, 3);
BM_ResizeBicubicDev(8, 512, 3);
BM_ResizeBicubicDev(8, 1024, 3);
BM_ResizeBicubicDev(16, 32, 3);
BM_ResizeBicubicDev(16, 128, 3);
BM_ResizeBicubicDev(16, 512, 3);
BM_ResizeBicubicDev(16, 1024, 3);
BM_ResizeBicubicDev(32, 32, 3);
BM_ResizeBicubicDev(32, 128, 3);
BM_ResizeBicubicDev(32, 512, 3);
BM_ResizeBicubicDev(32, 1024, 3);

}  // end namespace tensorflow
