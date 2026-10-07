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

// Benchmarks odml.fused_sdpa_cache_update on the YNNPACK delegate.
// Args: {h, g, t, d, w, start, num_threads}.

#include <cstddef>
#include <cstdint>
#include <memory>
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

class FusedModel : public SingleOpModel {
 public:
  FusedModel(int h, int g, int t, int d, int w, int num_threads) {
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
                     /*decomposition_subgraph_index=*/1, &attrs)
                     .Union());
    BuildInterpreter({GetShape(q_), GetShape(kc_), GetShape(vc_), GetShape(kn_),
                      GetShape(vn_), GetShape(mask_), GetShape(param_)},
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
  void Fill(int t, int w, int start) {
    for (int id : {q_, kc_, vc_, kn_, vn_}) {
      float* p = interpreter_->typed_tensor<float>(id);
      const size_t n = GetTensorSize(id);
      for (size_t i = 0; i < n; ++i) p[i] = 0.001f * static_cast<float>(i % 97);
    }
    // Causal window mask for a chunk at `start` with t - 1 valid tokens (the
    // last row is LiteRT-LM's pending-token padding row).
    bool* m = interpreter_->typed_tensor<bool>(mask_);
    const int valid = t - 1;
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < w + t; ++j) {
        bool keep = false;
        if (i < valid) {
          if (j < w) {
            // Slot j holds position p with p % w == j and p < start.
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

void BM_Fused(benchmark::State& state) {
  const int h = state.range(0), g = state.range(1), t = state.range(2),
            d = state.range(3), w = state.range(4), start = state.range(5),
            threads = state.range(6);
  FusedModel model(h, g, t, d, w, threads);
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

// {h, g, t, d, w, start, threads}
BENCHMARK(BM_Fused)
    ->ArgNames({"h", "g", "t", "d", "w", "start", "thr"})
    ->Args({1, 8, 1024, 256, 512, 0, 4})   // gemma4-e2b prefill_1024
    ->Args({1, 4, 1024, 256, 512, 0, 4})   // gemma3-1b prefill_1024
    ->Args({1, 8, 128, 256, 512, 600, 4})  // gemma4 prefill_128, wrapped
    ->Args({1, 8, 1024, 256, 512, 0, 1})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
