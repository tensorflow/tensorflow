#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* BM_AdjustContrast(int batches, int width, int height) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(DT_UINT8, TensorShape({batches, width, height, 3}));
  in.flat<uint8>().setRandom();
  Tensor factor(DT_FLOAT, TensorShape({}));
  factor.flat<float>().setConstant(1.2);
  Tensor min_value(DT_FLOAT, TensorShape({}));
  min_value.flat<float>().setConstant(7.);
  Tensor max_value(DT_FLOAT, TensorShape({}));
  max_value.flat<float>().setConstant(250.);

  Node* ret;
  NodeBuilder(g->NewName("n"), "AdjustContrast")
      .Input(test::graph::Constant(g, in))
      .Input(test::graph::Constant(g, factor))
      .Input(test::graph::Constant(g, min_value))
      .Input(test::graph::Constant(g, max_value))
      .Finalize(g, &ret);
  return g;
}

#define BM_AdjustContrastDev(DEVICE, B, W, H)                           \
  static void BM_AdjustContrast_##DEVICE##_##B##_##W##_##H(int iters) { \
    testing::ItemsProcessed(iters* B* W* H * 3);                        \
    test::Benchmark(#DEVICE, BM_AdjustContrast(B, W, H)).Run(iters);    \
  }                                                                     \
  BENCHMARK(BM_AdjustContrast_##DEVICE##_##B##_##W##_##H);

// Benchmark results as of cl/106323955
// BM_AdjustContrast_cpu_1_299_299  3416770  22008951  100  11.6M items/s

// BM_AdjustContrast_gpu_32_299_299  37117844  45512374  100  179.8M items/s
BM_AdjustContrastDev(cpu, 1, 299, 299) BM_AdjustContrastDev(gpu, 32, 299, 299)

}  // namespace tensorflow
