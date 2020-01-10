#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include <gtest/gtest.h>
#include "tensorflow/core/kernels/xent_op.h"

namespace tensorflow {

static Graph* Xent(int batch_size, int num_classes) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor logits(DT_FLOAT, TensorShape({batch_size, num_classes}));
  logits.flat<float>().setRandom();
  Tensor labels(DT_FLOAT, TensorShape({batch_size, num_classes}));
  labels.flat<float>().setRandom();
  test::graph::Binary(g, "SoftmaxCrossEntropyWithLogits",
                      test::graph::Constant(g, logits),
                      test::graph::Constant(g, labels));
  return g;
}

#define BM_XentDev(BATCH, CLASS, DEVICE)                                \
  static void BM_Xent##_##BATCH##_##CLASS##_##DEVICE(int iters) {       \
    testing::ItemsProcessed(static_cast<int64>(iters) * BATCH * CLASS); \
    test::Benchmark(#DEVICE, Xent(BATCH, CLASS)).Run(iters);            \
  }                                                                     \
  BENCHMARK(BM_Xent##_##BATCH##_##CLASS##_##DEVICE);

/// The representative tests for ptb_word on GPU
BM_XentDev(16, 10000, gpu);
BM_XentDev(16, 30000, gpu);
BM_XentDev(16, 100000, gpu);

BM_XentDev(32, 10000, gpu);
BM_XentDev(32, 30000, gpu);
BM_XentDev(32, 100000, gpu);

BM_XentDev(64, 10000, gpu);
BM_XentDev(64, 30000, gpu);
BM_XentDev(64, 100000, gpu);

/// Only the smaller tests for CPU. Otherwise, it's too slow
BM_XentDev(16, 10000, cpu);
BM_XentDev(32, 10000, cpu);
BM_XentDev(64, 10000, cpu);

}  // end namespace tensorflow
