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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* BM_AdjustContrast(int batches, int width, int height) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(DT_FLOAT, TensorShape({batches, width, height, 3}));
  in.flat<float>().setRandom();
  Tensor factor(DT_FLOAT, TensorShape({}));
  factor.flat<float>().setConstant(1.2);

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "AdjustContrastv2")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, factor))
                  .Finalize(g, &ret));
  return g;
}

#define BM_AdjustContrastDev(DEVICE, B, W, H)                           \
  static void BM_AdjustContrast_##DEVICE##_##B##_##W##_##H(int iters) { \
    testing::ItemsProcessed(iters* B* W* H * 3);                        \
    test::Benchmark(#DEVICE, BM_AdjustContrast(B, W, H)).Run(iters);    \
  }                                                                     \
  BENCHMARK(BM_AdjustContrast_##DEVICE##_##B##_##W##_##H)

// Benchmark results as of cl/106323955
// BM_AdjustContrast_cpu_1_299_299  3416770  22008951  100  11.6M items/s
// BM_AdjustContrast_gpu_32_299_299  37117844  45512374  100  179.8M items/s
// Benchmark results as of cl/109478777
// (note that the definition has changed to perform no min/max or clamping,
// so a comparison to cl/106323955 is inherently unfair)
// The GPU test ran with -c opt --config=cuda --copt=-mavx, CPU ran without
// --config=cuda because for some reason that killed throughput measurement.
// CPU: Intel Haswell with HyperThreading (6 cores) dL1:32KB dL2:256KB dL3:15MB
// GPU: Tesla K40m
// BM_AdjustContrast_cpu_1_299_299     179084     340186  2181  751.9M items/s
// BM_AdjustContrast_gpu_32_299_299     85276     123665  4189  2.9G items/s
BM_AdjustContrastDev(cpu, 1, 299, 299);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
BM_AdjustContrastDev(gpu, 32, 299, 299);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#ifdef TENSORFLOW_USE_SYCL
BM_AdjustContrastDev(sycl, 32, 299, 299);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
