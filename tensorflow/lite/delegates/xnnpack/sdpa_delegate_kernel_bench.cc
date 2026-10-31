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

// Benchmarks odml.sdpa_transposed and odml.fused_sdpa_cache_update on the
// XNNPACK delegate, comparing the fused SdpaDelegateKernel against the
// decomposed TFLite + XNNPACK graph.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {
namespace {

class FusedSdpaBenchModel : public SingleOpModel {
 public:
  FusedSdpaBenchModel(int h, int g, int t, int d, int w, int num_threads) {
    q_ = AddInput({TensorType_FLOAT32, {1, h, g * t, d}});
    kc_ = AddInput({TensorType_FLOAT32, {1, h, w, d}});
    vc_ = AddInput({TensorType_FLOAT32, {1, h, d, w}});
    kn_ = AddInput({TensorType_FLOAT32, {1, h, t, d}});
    vn_ = AddInput({TensorType_FLOAT32, {1, h, d, t}});
    mask_ = AddInput({TensorType_BOOL, {1, 1, t, w + t}});
    param_ = AddInput({TensorType_INT32, {1, 1, 1, 7}});
    AddOutput({TensorType_FLOAT32, {1, h, g * t, d}});
    AddOutput({TensorType_FLOAT32, {1, h, w, d}});
    AddOutput({TensorType_FLOAT32, {1, h, d, w}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("cache_size", w);
      fbb.Bool("update_cache", true);
    });
    fbb.Finish();
    std::vector<uint8_t> attrs = fbb.GetBuffer();
    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions,
                 CreateStableHLOCompositeOptionsDirect(
                     builder_, "odml.fused_sdpa_cache_update",
                     /*decomposition_subgraph_index=*/0, &attrs)
                     .Union());
    BuildInterpreter({GetShape(q_), GetShape(kc_), GetShape(vc_), GetShape(kn_),
                      GetShape(vn_), GetShape(mask_), GetShape(param_)},
                     -1, false, false, /*allocate_and_delegate=*/false);
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete));
    ApplyDelegate();
    interpreter_->AllocateTensors();
  }

  bool Delegated() const {
    return interpreter_->execution_plan().size() == 1 &&
           interpreter_
                   ->node_and_registration(interpreter_->execution_plan()[0])
                   ->first.delegate != nullptr;
  }

  void Fill(int t, int w, int start) {
    for (int id : {q_, kc_, vc_, kn_, vn_}) {
      float* p = interpreter_->typed_tensor<float>(id);
      const size_t n = GetTensorSize(id);
      for (size_t i = 0; i < n; ++i) p[i] = 0.001f * static_cast<float>(i % 97);
    }
    bool* m = interpreter_->typed_tensor<bool>(mask_);
    const int valid = std::max(1, t - 1);
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < w + t; ++j) {
        bool keep = false;
        if (i < valid) {
          if (j < w) {
            int p = start - 1 - ((start - 1 - j) % w + w) % w;
            keep = start > 0 && p >= 0 && p > start + i - w;
          } else {
            keep = (j - w) <= i;
          }
        }
        m[i * (w + t) + j] = keep;
      }
    }
    int32_t* pr = interpreter_->typed_tensor<int32_t>(param_);
    pr[0] = start;
    pr[1] = start + valid;
    pr[2] = start + valid;
    for (int i = 3; i < 7; ++i) pr[i] = 0;
  }

 private:
  int q_, kc_, vc_, kn_, vn_, mask_, param_;
};

// Decomposed equivalent of odml.fused_sdpa_cache_update (CONCATENATION +
// BATCH_MATMUL + TILE + SELECT_V2 + SOFTMAX + BATCH_MATMUL + per-token
// DYNAMIC_UPDATE_SLICE cache update) delegated to XNNPACK.
class DecomposedFusedSdpaBenchModel : public MultiOpModel {
 public:
  DecomposedFusedSdpaBenchModel(int h, int g, int t, int d, int w,
                                int num_threads)
      : h_(h), w_(w), t_(t), d_(d) {
    const std::vector<int> q_shape = {1, h, g * t, d};
    const std::vector<int> kc_shape = {1, h, w, d};
    const std::vector<int> vc_shape = {1, h, d, w};
    const std::vector<int> kn_shape = {1, h, t, d};
    const std::vector<int> vn_shape = {1, h, d, t};
    const std::vector<int> mask_shape = {1, 1, t, w + t};

    q_ = AddInput({TensorType_FLOAT32, q_shape});
    kc_ = AddInput({TensorType_FLOAT32, kc_shape});
    vc_ = AddInput({TensorType_FLOAT32, vc_shape});
    kn_ = AddInput({TensorType_FLOAT32, kn_shape});
    vn_ = AddInput({TensorType_FLOAT32, vn_shape});
    mask_ = AddInput({TensorType_BOOL, mask_shape});
    out_ = AddOutput({TensorType_FLOAT32, q_shape});

    const int k_all = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_CONCATENATION,
                 BuiltinOptions_ConcatenationOptions,
                 CreateConcatenationOptions(builder_, /*axis=*/2).Union(),
                 {kc_, kn_}, {k_all});

    const int v_all = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_CONCATENATION,
                 BuiltinOptions_ConcatenationOptions,
                 CreateConcatenationOptions(builder_, /*axis=*/3).Union(),
                 {vc_, vn_}, {v_all});

    const int scores = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, /*adj_x=*/false, /*adj_y=*/true)
            .Union(),
        {q_, k_all}, {scores});

    int mask_for_select = mask_;
    if (g > 1) {
      const int multiples =
          AddConstInput<int32_t>(TensorType_INT32, {1, 1, g, 1}, {4});
      const int tiled_mask = AddIntermediate(TensorType_BOOL, {}, {});
      AddBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions,
                   CreateTileOptions(builder_).Union(), {mask_, multiples},
                   {tiled_mask});
      mask_for_select = tiled_mask;
    }

    const int fill_const =
        AddConstInput<float>(TensorType_FLOAT32, {-10000.0f}, {1, 1, 1, 1});
    const int masked_scores = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_SELECT_V2, BuiltinOptions_SelectV2Options,
                 CreateSelectV2Options(builder_).Union(),
                 {mask_for_select, scores, fill_const}, {masked_scores});

    const int probs = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, 1.0f).Union(), {masked_scores},
                 {probs});

    AddBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, /*adj_x=*/false, /*adj_y=*/true)
            .Union(),
        {probs, v_all}, {out_});

    BuildInterpreter(
        {q_shape, kc_shape, vc_shape, kn_shape, vn_shape, mask_shape},
        num_threads, false, false, /*allocate_and_delegate=*/false);
    interpreter_->AllocateTensors();
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete));
    ApplyDelegate();
    interpreter_->AllocateTensors();
    kc_out_buf_.resize(h * w * d);
    vc_out_buf_.resize(h * d * w);
  }

  void Fill(int t, int w, int start) {
    start_ = start;
    valid_ = std::max(1, t - 1);
    for (int id : {q_, kc_, vc_, kn_, vn_}) {
      float* p = interpreter_->typed_tensor<float>(id);
      const size_t n = GetTensorSize(id);
      for (size_t i = 0; i < n; ++i) p[i] = 0.001f * static_cast<float>(i % 97);
    }
    bool* m = interpreter_->typed_tensor<bool>(mask_);
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < w + t; ++j) {
        bool keep = false;
        if (i < valid_) {
          if (j < w) {
            int p = start - 1 - ((start - 1 - j) % w + w) % w;
            keep = start > 0 && p >= 0 && p > start + i - w;
          } else {
            keep = (j - w) <= i;
          }
        }
        m[i * (w + t) + j] = keep;
      }
    }
  }

  TfLiteStatus InvokeWithDecomposedCacheLoop() {
    TfLiteStatus st = Invoke();
    if (st != kTfLiteOk) return st;
    // Simulate the T-iteration decomposed DYNAMIC_UPDATE_SLICE loop over [0, T)
    const float* kc = interpreter_->typed_tensor<float>(kc_);
    const float* vc = interpreter_->typed_tensor<float>(vc_);
    const float* kn = interpreter_->typed_tensor<float>(kn_);
    const float* vn = interpreter_->typed_tensor<float>(vn_);
    std::memcpy(kc_out_buf_.data(), kc, h_ * w_ * d_ * sizeof(float));
    std::memcpy(vc_out_buf_.data(), vc, h_ * d_ * w_ * sizeof(float));
    for (int i = 0; i < t_; ++i) {
      const int pos = start_ + i;
      const int slot = (pos < start_ + valid_) ? (pos % w_) : -1;
      if (slot >= 0) {
        for (int hi = 0; hi < h_; ++hi) {
          std::memcpy(&kc_out_buf_[(hi * w_ + slot) * d_],
                      &kn[(hi * t_ + i) * d_], d_ * sizeof(float));
          for (int di = 0; di < d_; ++di) {
            vc_out_buf_[(hi * d_ + di) * w_ + slot] =
                vn[(hi * d_ + di) * t_ + i];
          }
        }
      }
    }
    return kTfLiteOk;
  }

 private:
  int h_, w_, t_, d_;
  int start_ = 0;
  int valid_ = 1;
  int q_, kc_, vc_, kn_, vn_, mask_, out_;
  std::vector<float> kc_out_buf_;
  std::vector<float> vc_out_buf_;
};

class SdpaTransposedBenchModel : public SingleOpModel {
 public:
  SdpaTransposedBenchModel(int h, int g, int t, int d, int cache,
                           int num_threads) {
    q_ = AddInput({TensorType_FLOAT32, {1, h * g, t, d}});
    k_ = AddInput({TensorType_FLOAT32, {1, h, cache, d}});
    v_ = AddInput({TensorType_FLOAT32, {1, h, d, cache}});
    mask_ = AddInput({TensorType_BOOL, {1, 1, t, cache}});
    param_ = AddInput({TensorType_INT32, {1, 1, 1, 7}});
    AddOutput({TensorType_FLOAT32, {1, h * g, t, d}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Float("scale", 1.0f); });
    fbb.Finish();
    std::vector<uint8_t> attrs = fbb.GetBuffer();
    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions,
                 CreateStableHLOCompositeOptionsDirect(
                     builder_, "odml.sdpa_transposed",
                     /*decomposition_subgraph_index=*/0, &attrs)
                     .Union());
    BuildInterpreter({GetShape(q_), GetShape(k_), GetShape(v_), GetShape(mask_),
                      GetShape(param_)},
                     -1, false, false, /*allocate_and_delegate=*/false);
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete));
    ApplyDelegate();
    interpreter_->AllocateTensors();
  }

  bool Delegated() const {
    return interpreter_->execution_plan().size() == 1 &&
           interpreter_
                   ->node_and_registration(interpreter_->execution_plan()[0])
                   ->first.delegate != nullptr;
  }

  void Fill(int t, int cache, int kv) {
    for (int id : {q_, k_, v_}) {
      float* p = interpreter_->typed_tensor<float>(id);
      const size_t n = GetTensorSize(id);
      for (size_t i = 0; i < n; ++i) p[i] = 0.001f * static_cast<float>(i % 97);
    }
    bool* m = interpreter_->typed_tensor<bool>(mask_);
    const int start = kv - t;
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < cache; ++j) m[i * cache + j] = j <= start + i;
    }
    int32_t* pr = interpreter_->typed_tensor<int32_t>(param_);
    pr[0] = start;
    pr[1] = kv;
    pr[2] = kv;
    for (int i = 3; i < 7; ++i) pr[i] = 0;
  }

 private:
  int q_, k_, v_, mask_, param_;
};

// Decomposed equivalent of odml.sdpa_transposed (RESHAPE + BATCH_MATMUL +
// TILE + SELECT_V2 + SOFTMAX + BATCH_MATMUL + RESHAPE) over the full static
// KV cache size (since XNNPACK does not support dynamic runtime_bmm slicing).
class DecomposedSdpaTransposedBenchModel : public MultiOpModel {
 public:
  DecomposedSdpaTransposedBenchModel(int h, int g, int t, int d, int cache,
                                     int num_threads) {
    q_ = AddInput({TensorType_FLOAT32, {1, h * g, t, d}});
    k_ = AddInput({TensorType_FLOAT32, {1, h, cache, d}});
    v_ = AddInput({TensorType_FLOAT32, {1, h, d, cache}});
    mask_ = AddInput({TensorType_BOOL, {1, 1, t, cache}});
    out_ = AddOutput({TensorType_FLOAT32, {1, h * g, t, d}});

    const int q_shape_const =
        AddConstInput<int32_t>(TensorType_INT32, {1, h, g * t, d}, {4});
    const int q_packed = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
                 CreateReshapeOptions(
                     builder_, builder_.CreateVector<int>({1, h, g * t, d}))
                     .Union(),
                 {q_, q_shape_const}, {q_packed});

    const int scores = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, /*adj_x=*/false, /*adj_y=*/true)
            .Union(),
        {q_packed, k_}, {scores});

    int mask_for_select = mask_;
    if (h > 1 || g > 1) {
      const int multiples =
          AddConstInput<int32_t>(TensorType_INT32, {1, h, g, 1}, {4});
      const int tiled_mask = AddIntermediate(TensorType_BOOL, {}, {});
      AddBuiltinOp(BuiltinOperator_TILE, BuiltinOptions_TileOptions,
                   CreateTileOptions(builder_).Union(), {mask_, multiples},
                   {tiled_mask});
      mask_for_select = tiled_mask;
    }

    const int fill_const =
        AddConstInput<float>(TensorType_FLOAT32, {-10000.0f}, {1, 1, 1, 1});
    const int masked_scores = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_SELECT_V2, BuiltinOptions_SelectV2Options,
                 CreateSelectV2Options(builder_).Union(),
                 {mask_for_select, scores, fill_const}, {masked_scores});

    const int probs = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, 1.0f).Union(), {masked_scores},
                 {probs});

    const int attn_packed = AddIntermediate(TensorType_FLOAT32, {}, {});
    AddBuiltinOp(
        BuiltinOperator_BATCH_MATMUL, BuiltinOptions_BatchMatMulOptions,
        CreateBatchMatMulOptions(builder_, /*adj_x=*/false, /*adj_y=*/true)
            .Union(),
        {probs, v_}, {attn_packed});

    const int out_shape_const =
        AddConstInput<int32_t>(TensorType_INT32, {1, h * g, t, d}, {4});
    AddBuiltinOp(BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
                 CreateReshapeOptions(
                     builder_, builder_.CreateVector<int>({1, h * g, t, d}))
                     .Union(),
                 {attn_packed, out_shape_const}, {out_});

    BuildInterpreter(
        {GetShape(q_), GetShape(k_), GetShape(v_), GetShape(mask_)},
        num_threads, false, false, /*allocate_and_delegate=*/false);
    interpreter_->AllocateTensors();
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete));
    ApplyDelegate();
    interpreter_->AllocateTensors();
  }

  void Fill(int t, int cache, int kv) {
    for (int id : {q_, k_, v_}) {
      float* p = interpreter_->typed_tensor<float>(id);
      const size_t n = GetTensorSize(id);
      for (size_t i = 0; i < n; ++i) p[i] = 0.001f * static_cast<float>(i % 97);
    }
    bool* m = interpreter_->typed_tensor<bool>(mask_);
    const int start = kv - t;
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < cache; ++j) m[i * cache + j] = j <= start + i;
    }
  }

 private:
  int q_, k_, v_, mask_, out_;
};

void BM_FusedSdpa_XnnKernel(benchmark::State& state) {
  const int h = state.range(0), g = state.range(1), t = state.range(2),
            d = state.range(3), w = state.range(4), start = state.range(5),
            threads = state.range(6);
  FusedSdpaBenchModel model(h, g, t, d, w, threads);
  if (!model.Delegated()) {
    state.SkipWithError("not delegated");
    return;
  }
  model.Fill(t, w, start);
  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("invoke failed");
    return;
  }
  for (auto _ : state) {
    if (model.Invoke() != kTfLiteOk) {
      state.SkipWithError("invoke failed");
      return;
    }
  }
  const double flops = 2.0 * 2.0 * h * g * t * (w + t) * d;
  state.counters["GFLOPS"] = benchmark::Counter(
      flops * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}

void BM_FusedSdpa_DecomposedXnn(benchmark::State& state) {
  const int h = state.range(0), g = state.range(1), t = state.range(2),
            d = state.range(3), w = state.range(4), start = state.range(5),
            threads = state.range(6);
  DecomposedFusedSdpaBenchModel model(h, g, t, d, w, threads);
  model.Fill(t, w, start);
  if (model.InvokeWithDecomposedCacheLoop() != kTfLiteOk) {
    state.SkipWithError("invoke failed");
    return;
  }
  for (auto _ : state) {
    if (model.InvokeWithDecomposedCacheLoop() != kTfLiteOk) {
      state.SkipWithError("invoke failed");
      return;
    }
  }
  const double flops = 2.0 * 2.0 * h * g * t * (w + t) * d;
  state.counters["GFLOPS"] = benchmark::Counter(
      flops * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}

void BM_SdpaTransposed_XnnKernel(benchmark::State& state) {
  const int h = state.range(0), g = state.range(1), t = state.range(2),
            d = state.range(3), cache = state.range(4), kv = state.range(5),
            threads = state.range(6);
  SdpaTransposedBenchModel model(h, g, t, d, cache, threads);
  if (!model.Delegated()) {
    state.SkipWithError("not delegated");
    return;
  }
  model.Fill(t, cache, kv);
  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("invoke failed");
    return;
  }
  for (auto _ : state) {
    if (model.Invoke() != kTfLiteOk) {
      state.SkipWithError("invoke failed");
      return;
    }
  }
  const double flops = 2.0 * 2.0 * h * g * t * static_cast<double>(kv) * d;
  state.counters["GFLOPS"] = benchmark::Counter(
      flops * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}

void BM_SdpaTransposed_DecomposedXnn(benchmark::State& state) {
  const int h = state.range(0), g = state.range(1), t = state.range(2),
            d = state.range(3), cache = state.range(4), kv = state.range(5),
            threads = state.range(6);
  DecomposedSdpaTransposedBenchModel model(h, g, t, d, cache, threads);
  model.Fill(t, cache, kv);
  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("invoke failed");
    return;
  }
  for (auto _ : state) {
    if (model.Invoke() != kTfLiteOk) {
      state.SkipWithError("invoke failed");
      return;
    }
  }
  const double flops = 2.0 * 2.0 * h * g * t * static_cast<double>(kv) * d;
  state.counters["GFLOPS"] = benchmark::Counter(
      flops * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}

BENCHMARK(BM_FusedSdpa_XnnKernel)
    ->ArgNames({"h", "g", "t", "d", "w", "start", "thr"})
    ->Args({1, 4, 128, 256, 512, 0, 4})    // gemma3-1b prefill_128 chunk 0
    ->Args({1, 4, 128, 256, 512, 127, 4})  // gemma3-1b prefill_128 chunk 1
    ->Args({1, 4, 128, 256, 512, 600, 4})  // gemma3-1b prefill_128 wrapped
    ->Args({1, 4, 1024, 256, 512, 0, 4})   // gemma3-1b prefill_1024 chunk 0
    ->Args({1, 8, 128, 256, 512, 0, 4})    // gemma4-e2b prefill_128 chunk 0
    ->Args({1, 8, 128, 256, 512, 600, 4})  // gemma4-e2b prefill_128 wrapped
    ->Args({1, 8, 1024, 256, 512, 0, 4})   // gemma4-e2b prefill_1024 chunk 0
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_FusedSdpa_DecomposedXnn)
    ->ArgNames({"h", "g", "t", "d", "w", "start", "thr"})
    ->Args({1, 4, 128, 256, 512, 0, 4})
    ->Args({1, 4, 128, 256, 512, 127, 4})
    ->Args({1, 4, 128, 256, 512, 600, 4})
    ->Args({1, 4, 1024, 256, 512, 0, 4})
    ->Args({1, 8, 128, 256, 512, 0, 4})
    ->Args({1, 8, 128, 256, 512, 600, 4})
    ->Args({1, 8, 1024, 256, 512, 0, 4})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SdpaTransposed_XnnKernel)
    ->ArgNames({"h", "g", "t", "d", "cache", "kv", "thr"})
    ->Args({1, 4, 1, 256, 4096, 128, 4})     // gemma3-1b decode at kv=128
    ->Args({1, 4, 1, 256, 4096, 1024, 4})    // gemma3-1b decode at kv=1024
    ->Args({1, 4, 128, 256, 4096, 128, 4})   // gemma3-1b prefill_128 chunk 1
    ->Args({1, 4, 128, 256, 4096, 1024, 4})  // gemma3-1b prefill_128 chunk 8
    ->Args({1, 8, 1, 512, 10240, 1024, 4})   // gemma4-e2b decode at kv=1024
    ->Args({1, 8, 1, 512, 10240, 9216, 4})   // gemma4-e2b decode at kv=9216
    ->Args({1, 8, 1024, 512, 10240, 1024,
            4})  // gemma4-e2b prefill_1024 chunk 1
    ->Args({1, 8, 1024, 512, 10240, 5120,
            4})                               // gemma4-e2b prefill_1024 chunk 5
    ->Args({8, 2, 1, 128, 4096, 1024, 4})     // qwen3-0.6b decode at kv=1024
    ->Args({8, 2, 1, 128, 10240, 9216, 4})    // qwen3-0.6b decode at kv=9216
    ->Args({8, 2, 1024, 128, 4096, 1024, 4})  // qwen3-0.6b prefill_1024 chunk 1
    ->Args({8, 2, 1024, 128, 10240, 5120,
            4})  // qwen3-0.6b prefill_1024 chunk 5
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SdpaTransposed_DecomposedXnn)
    ->ArgNames({"h", "g", "t", "d", "cache", "kv", "thr"})
    ->Args({1, 4, 1, 256, 4096, 128, 4})
    ->Args({1, 4, 1, 256, 4096, 1024, 4})
    ->Args({1, 4, 128, 256, 4096, 128, 4})
    ->Args({1, 4, 128, 256, 4096, 1024, 4})
    ->Args({1, 8, 1, 512, 10240, 1024, 4})
    ->Args({1, 8, 1, 512, 10240, 9216, 4})
    ->Args({1, 8, 1024, 512, 10240, 1024, 4})
    ->Args({1, 8, 1024, 512, 10240, 5120, 4})
    ->Args({8, 2, 1, 128, 4096, 1024, 4})
    ->Args({8, 2, 1, 128, 10240, 9216, 4})
    ->Args({8, 2, 1024, 128, 4096, 1024, 4})
    ->Args({8, 2, 1024, 128, 10240, 5120, 4})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
