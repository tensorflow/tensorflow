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

#include "tensorflow/core/framework/bfloat16.h"

#include "absl/base/casts.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

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

void BM_FloatToBFloat16(::testing::benchmark::State& state) {
  static const int N = 32 << 20;

  float* inp = new float[N];
  bfloat16* out = new bfloat16[N];

  for (auto s : state) {
    FloatToBFloat16(inp, out, N);
  }

  const int64 tot = static_cast<int64>(state.iterations()) * N;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_FloatToBFloat16);

void BM_RoundFloatToBFloat16(::testing::benchmark::State& state) {
  static const int N = 32 << 20;

  float* inp = new float[N];
  bfloat16* out = new bfloat16[N];

  for (auto s : state) {
    RoundFloatToBFloat16(inp, out, N);
    tensorflow::testing::DoNotOptimize(inp);
    tensorflow::testing::DoNotOptimize(out);
  }

  const int64 tot = static_cast<int64>(state.iterations()) * N;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_RoundFloatToBFloat16);

void BM_BFloat16ToFloat(::testing::benchmark::State& state) {
  static const int N = 32 << 20;

  bfloat16* inp = new bfloat16[N];
  float* out = new float[N];

  for (auto s : state) {
    BFloat16ToFloat(inp, out, N);
  }

  const int64 tot = static_cast<int64>(state.iterations()) * N;
  state.SetItemsProcessed(tot);
  state.SetBytesProcessed(tot * (sizeof(float) + sizeof(bfloat16)));

  delete[] inp;
  delete[] out;
}
BENCHMARK(BM_BFloat16ToFloat);

}  // namespace
}  // namespace tensorflow
