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
#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/attention_model.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"

namespace tflite {
namespace ynnpack {
namespace {

void BenchAttention(benchmark::State& state, int b, int query_len = 0,
                    bool transpose_io = false,
                    AttentionImpl impl = AttentionImpl::kOdmlRuntimeBmm,
                    bool quantize_kv = false, bool int4_kv = false,
                    TfLiteType sdpa_dot_type = kTfLiteNoType) {
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
  options.sdpa_dot_type = sdpa_dot_type;

  // Initialize input data
  std::vector<float> q_data(b * n * t * h);
  std::vector<float> mask_data(b * 1 * t * s);

  for (int i = 0; i < b * t; ++i) {
    for (int j = 0; j < s; ++j) {
      mask_data[i * s + j] = (j < s_active) ? 0.0f : -1e9f;
    }
  }

  // Run delegate model (YNNPACK)
  AttentionModel model(b, t, s, h, n, scale, transpose_io,
                       /*use_delegate=*/true, options, impl, /*n_kv=*/-1,
                       quantize_kv, /*bool_mask=*/false, /*kv_zero_point=*/0,
                       /*quantize_q=*/false, /*q_zero_point=*/0,
                       /*quantize_output=*/false, /*output_zero_point=*/0,
                       /*quantize_mask=*/false, /*mask_zero_point=*/0, int4_kv);
  model.PopulateTensor(model.query(), q_data);
  if (int4_kv) {
    std::vector<int8_t> k_qdata(b * n * s * h);
    std::vector<int8_t> v_qdata(b * n * h * s);
    model.PopulateInt4Tensor(model.key(), k_qdata);
    model.PopulateInt4Tensor(model.value(), v_qdata);
  } else if (quantize_kv) {
    std::vector<int8_t> k_qdata(b * n * s * h);
    std::vector<int8_t> v_qdata(b * n * h * s);
    model.PopulateTensor(model.key(), k_qdata);
    model.PopulateTensor(model.value(), v_qdata);
  } else {
    std::vector<float> k_data(b * n * s * h);
    std::vector<float> v_data(b * n * h * s);
    model.PopulateTensor(model.key(), k_data);
    model.PopulateTensor(model.value(), v_data);
  }
  if (model.runtime_bmm_params() != -1) {
    model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  }
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

// BMM-based benchmarks (with dynamic slicing)
void Attention(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 AttentionImpl::kOdmlRuntimeBmm);
}

void AttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/true,
                 AttentionImpl::kOdmlRuntimeBmm);
}

void AttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/true,
                 AttentionImpl::kOdmlRuntimeBmm);
}

void AttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 AttentionImpl::kOdmlRuntimeBmm);
}

// SDPA-based benchmarks (with dynamic slicing)
void SdpaAttention(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 AttentionImpl::kOdmlSdpa);
}

void SdpaAttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/true,
                 AttentionImpl::kOdmlSdpa);
}

void SdpaAttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/true,
                 AttentionImpl::kOdmlSdpa);
}

void SdpaAttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 AttentionImpl::kOdmlSdpa);
}

void SdpaAttentionInt8Kv(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 AttentionImpl::kOdmlSdpa, /*quantize_kv=*/true,
                 /*int4_kv=*/false, /*sdpa_dot_type=*/kTfLiteInt8);
}

void SdpaAttentionDecodeInt8Kv(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 AttentionImpl::kOdmlSdpa, /*quantize_kv=*/true,
                 /*int4_kv=*/false, /*sdpa_dot_type=*/kTfLiteInt8);
}

void SdpaAttentionInt4Kv(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 AttentionImpl::kOdmlSdpa, /*quantize_kv=*/false,
                 /*int4_kv=*/true);
}

void SdpaAttentionDecodeInt4Kv(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 AttentionImpl::kOdmlSdpa, /*quantize_kv=*/false,
                 /*int4_kv=*/true);
}

// Full sequence benchmarks (standard ops, no dynamic slicing)
void FullSequenceAttention(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/false,
                 AttentionImpl::kFullSequence);
}

void FullSequenceAttentionTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/0, /*transpose_io=*/true,
                 AttentionImpl::kFullSequence);
}

void FullSequenceAttentionDecodeTransposed(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/true,
                 AttentionImpl::kFullSequence);
}

void FullSequenceAttentionDecode(benchmark::State& state) {
  BenchAttention(state, /*b=*/1, /*query_len=*/1, /*transpose_io=*/false,
                 AttentionImpl::kFullSequence);
}

void AttentionArguments(benchmark::internal::Benchmark* b) {
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

BENCHMARK(SdpaAttention)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionDecodeTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionDecode)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionInt8Kv)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionDecodeInt8Kv)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionInt4Kv)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(SdpaAttentionDecodeInt4Kv)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(FullSequenceAttention)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FullSequenceAttentionTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FullSequenceAttentionDecodeTransposed)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);
BENCHMARK(FullSequenceAttentionDecode)
    ->Apply(AttentionArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
