/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/attention_model.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"

namespace tflite {
namespace ynnpack {
namespace {

void BenchAttention(benchmark::State& state, int b, int query_len = 0,
                    bool transpose_io = false) {
  const int s = state.range(0);
  const int t = query_len == 0 ? s : query_len;
  const int h = state.range(1);
  const int n = state.range(2);
  const int num_threads = state.range(3);
  const int s_active = std::min<int>(s, state.range(4));
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = num_threads;
  options.static_shape = true;

  // Initialize input data
  std::vector<float> q_data(b * n * t * h);
  std::vector<float> k_data(b * n * s * h);
  std::vector<float> v_data(b * n * h * s);
  std::vector<float> mask_data(b * 1 * t * s);

  for (int i = 0; i < b * t; ++i) {
    for (int j = 0; j < s; ++j) {
      mask_data[i * s + j] = (j < s_active) ? 0.0f : -1e9f;
    }
  }

  // Run delegate model (YNNPACK)
  AttentionModel model(b, t, s, h, n, scale, transpose_io,
                       /*use_delegate=*/true, options);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data);
  model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  model.PopulateTensor(model.mask(), mask_data);
  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("Failed to invoke delegate interpreter (warmup)");
    return;
  }

  // Run benchmark on delegate model
  for (auto _ : state) {
    if (model.Invoke() != kTfLiteOk) {
      state.SkipWithError("Failed to invoke interpreter");
      return;
    }
  }

  const size_t flops = 2ull * b * n * (query_len == 0 ? s_active : t) *
                       s_active * h * 2;  // QK^T and P@V
  state.counters["FLOP"] =
      benchmark::Counter(static_cast<double>(state.iterations() * flops),
                         benchmark::Counter::kIsRate);
}

void Attention(benchmark::State& state) { BenchAttention(state, /*b=*/1); }

void AttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0,
                 /*transpose_io=*/true);
}

void AttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1,
                 /*transpose_io=*/true);
}

void AttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1);
}

void AttentionArguments(benchmark::Benchmark* b) {
  b->ArgNames({"seq", "head", "heads", "threads", "seq_active"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  std::vector<std::vector<int>> shapes = {{256, 64, 8},
                                          {512, 64, 8},
                                          {1024, 64, 8},
                                          {1024, 64, 32},
                                          {4096, 64, 32}};
  for (const auto& shape : shapes) {
    for (int threads : {1, 2, 4}) {
      for (float s_fraction : {0.01f, 0.5f, 0.99f, 1.0f}) {
        const int s_active = std::ceil(s_fraction * shape[0]);
        b->Args({shape[0], shape[1], shape[2], threads, s_active});
      }
    }
  }
}

BENCHMARK(Attention)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(AttentionDecodeTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(AttentionDecode)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
