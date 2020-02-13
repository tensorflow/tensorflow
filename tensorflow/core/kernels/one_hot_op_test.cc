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

#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* OneHot(int batch_size, int num_classes, int axis) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor indices(DT_INT32, TensorShape({batch_size}));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, num_classes - 1);

  auto indices_t = indices.flat<int32>();
  for (int i = 0; i < batch_size; ++i) {
    indices_t(i) = dist(gen);
  }

  Tensor depth(DT_INT32, TensorShape({}));
  depth.scalar<int32>()() = num_classes;

  Tensor on_value(DT_FLOAT, TensorShape({}));
  on_value.scalar<float>()() = 1.0f;

  Tensor off_value(DT_FLOAT, TensorShape({}));
  off_value.scalar<float>()() = 0.0f;

  test::graph::Multi(g, "OneHot",
                     {
                         test::graph::Constant(g, indices),
                         test::graph::Constant(g, depth),
                         test::graph::Constant(g, on_value),
                         test::graph::Constant(g, off_value),
                     })
      ->AddAttr("axis", axis);
  return g;
}

#define BM_OneHot(BATCH, CLASS, AXIS, DEVICE)                                \
  static void BM_OneHot##_##BATCH##_##CLASS##_##AXIS##_##DEVICE(int iters) { \
    testing::ItemsProcessed(static_cast<int64>(iters) * BATCH * CLASS);      \
    test::Benchmark(#DEVICE, OneHot(BATCH, CLASS, AXIS)).Run(iters);         \
  }                                                                          \
  BENCHMARK(BM_OneHot##_##BATCH##_##CLASS##_##AXIS##_##DEVICE);

// CPU
BM_OneHot(32, 512, 1, cpu);
BM_OneHot(64, 512, 1, cpu);
BM_OneHot(128, 512, 1, cpu);

BM_OneHot(32, 1024, 1, cpu);
BM_OneHot(64, 1024, 1, cpu);
BM_OneHot(128, 1024, 1, cpu);

BM_OneHot(32, 10000, 1, cpu);
BM_OneHot(64, 10000, 1, cpu);
BM_OneHot(128, 10000, 1, cpu);

BM_OneHot(32, 512, 0, cpu);
BM_OneHot(64, 512, 0, cpu);
BM_OneHot(128, 512, 0, cpu);

BM_OneHot(32, 1024, 0, cpu);
BM_OneHot(64, 1024, 0, cpu);
BM_OneHot(128, 1024, 0, cpu);

BM_OneHot(32, 10000, 0, cpu);
BM_OneHot(64, 10000, 0, cpu);
BM_OneHot(128, 10000, 0, cpu);

}  // end namespace tensorflow
