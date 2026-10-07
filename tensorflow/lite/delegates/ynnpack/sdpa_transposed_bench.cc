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

// Benchmarks odml.sdpa_transposed (full/global attention over a KV cache that
// is sliced to the active length by the param tensor) on the YNNPACK delegate.
// Args: {g, t, d, cache, kv, num_threads}; one KV head, g query heads.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ynnpack {
namespace {

class SdpaModel : public SingleOpModel {
 public:
  SdpaModel(int g, int t, int d, int cache, int num_threads) {
    q_ = AddInput({TensorType_FLOAT32, {1, g, t, d}});
    k_ = AddInput({TensorType_FLOAT32, {1, 1, cache, d}});
    v_ = AddInput({TensorType_FLOAT32, {1, 1, d, cache}});
    mask_ = AddInput({TensorType_BOOL, {1, 1, t, cache}});
    param_ = AddInput({TensorType_INT32, {1, 1, 1, 7}});
    AddOutput({TensorType_FLOAT32, {1, g, t, d}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Float("scale", 1.0f); });
    fbb.Finish();
    std::vector<uint8_t> attrs = fbb.GetBuffer();
    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions,
                 CreateStableHLOCompositeOptionsDirect(
                     builder_, "odml.sdpa_transposed",
                     /*decomposition_subgraph_index=*/1, &attrs)
                     .Union());
    BuildInterpreter({GetShape(q_), GetShape(k_), GetShape(v_), GetShape(mask_),
                      GetShape(param_)},
                     -1, false, false, /*allocate_and_delegate=*/false);
    TfLiteYNNPackDelegateOptions options =
        TfLiteYNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete));
    ApplyDelegate();
    interpreter_->AllocateTensors();
  }
  bool Delegated() const {
    return interpreter_->execution_plan().size() == 1 &&
           interpreter_
                   ->node_and_registration(interpreter_->execution_plan()[0])
                   ->first.delegate != nullptr;
  }
  // Chunk of t queries ending at position kv - 1 (causal).
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

void BM_SdpaTransposed(benchmark::State& state) {
  const int g = state.range(0), t = state.range(1), d = state.range(2),
            cache = state.range(3), kv = state.range(4),
            threads = state.range(5);
  SdpaModel model(g, t, d, cache, threads);
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
  const double flops = 2.0 * 2.0 * g * t * static_cast<double>(kv) * d;
  const double kv_bytes = 2.0 * kv * d * sizeof(float);
  state.counters["GFLOPS"] = benchmark::Counter(
      flops * state.iterations() / 1e9, benchmark::Counter::kIsRate);
  state.counters["KV_GBps"] = benchmark::Counter(
      kv_bytes * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}

// {g, t, d, cache, kv, threads}: gemma4-e2b global layers (8 q heads, 1 kv
// head, head_dim 512), cache 10240.
BENCHMARK(BM_SdpaTransposed)
    ->ArgNames({"g", "t", "d", "cache", "kv", "thr"})
    ->Args({8, 1, 512, 10240, 1024, 4})     // decode after p=1024
    ->Args({8, 1, 512, 10240, 9216, 4})     // decode after p=9216
    ->Args({8, 1024, 512, 10240, 1024, 4})  // prefill chunk 1
    ->Args({8, 1024, 512, 10240, 5120, 4})  // prefill chunk 5
    ->Args({8, 1024, 512, 10240, 9216, 4})  // prefill chunk 9
    ->Args({4, 1, 256, 10240, 9216, 4})     // gemma3-1b decode after p=9216
    ->Args({4, 1024, 256, 10240, 9216, 4})  // gemma3-1b prefill chunk 9
    // Thread scaling of the heaviest prefill chunk.
    ->Args({8, 1024, 512, 10240, 9216, 1})
    ->Args({8, 1024, 512, 10240, 9216, 2})
    ->Args({8, 1024, 512, 10240, 9216, 6})
    ->Args({8, 1024, 512, 10240, 9216, 8})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
