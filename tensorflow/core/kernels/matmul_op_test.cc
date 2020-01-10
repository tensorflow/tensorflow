#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(DT_FLOAT, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<float>().setRandom();
  Tensor in1(DT_FLOAT, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<float>().setRandom();
  test::graph::Matmul(g, test::graph::Constant(g, in0),
                      test::graph::Constant(g, in1), transpose_a, transpose_b);
  return g;
}

#define BM_MatmulDev(M, K, N, TA, TB, DEVICE)                           \
  static void BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE(     \
      int iters) {                                                      \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2); \
    test::Benchmark(#DEVICE, Matmul(M, K, N, TA, TB)).Run(iters);       \
  }                                                                     \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##DEVICE);

#define BM_Matmul(M, K, N, TA, TB)    \
  BM_MatmulDev(M, K, N, TA, TB, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, gpu);

// Typical fully connected layers
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

}  // end namespace tensorflow
