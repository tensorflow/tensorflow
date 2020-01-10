#include "tensorflow/core/framework/bfloat16.h"

#include "tensorflow/core/platform/test_benchmark.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace {

TEST(Bfloat16Test, Simple) {
  bfloat16 a(12);
  EXPECT_EQ(12, a.value);
}

TEST(Bfloat16Test, Conversion) {
  float a[100];
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100];
  float c[100];
  FloatToBFloat16(a, b, 100);
  BFloat16ToFloat(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

static void BM_FloatToBFloat16(int iters) {
  testing::StopTiming();
  static const int N = 32 << 20;
  const int64 tot = static_cast<int64>(iters) * N;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  float* inp = new float[N];
  bfloat16* out = new bfloat16[N];

  testing::StartTiming();
  while (iters--) {
    FloatToBFloat16(inp, out, N);
  }
  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_FloatToBFloat16);

static void BM_BFloat16ToFloat(int iters) {
  testing::StopTiming();
  static const int N = 32 << 20;
  const int64 tot = static_cast<int64>(iters) * N;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  bfloat16* inp = new bfloat16[N];
  float* out = new float[N];

  testing::StartTiming();
  while (iters--) {
    BFloat16ToFloat(inp, out, N);
  }
  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_BFloat16ToFloat);

}  // namespace
}  // namespace tensorflow
