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

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"

namespace tflite {
namespace ynnpack {
namespace {

class SdpaBenchModel : public SingleOpModel {
 public:
  SdpaBenchModel(int s_q, int s_kv, int heads, int head_dim) {
    q_id_ = AddInput({TensorType_FLOAT32, {1, s_q, heads, head_dim}});
    k_id_ = AddInput({TensorType_FLOAT32, {1, s_kv, heads, head_dim}});
    v_id_ = AddInput({TensorType_FLOAT32, {1, s_kv, heads, head_dim}});
    out_id_ = AddOutput({TensorType_FLOAT32, {1, s_q, heads, head_dim}});

    SetCustomOp("odml.scaled_dot_product_attention", {},
                /*register_fn=*/[]() -> TfLiteRegistration* {
                  static TfLiteRegistration reg = {
                      /*init=*/nullptr,
                      /*free=*/nullptr,
                      /*prepare=*/nullptr,
                      /*invoke=*/nullptr,
                      /*profiling_string=*/nullptr,
                      /*builtin_code=*/0,
                      "odml.scaled_dot_product_attention",
                      /*version=*/1};
                  return &reg;
                });

    BuildInterpreter({GetShape(q_id_), GetShape(k_id_), GetShape(v_id_)});
  }

  void SetRandomData() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto fill = [&](int id) {
      auto shape = GetTensorShape(id);
      int size = 1;
      for (int i = 0; i < shape.size(); ++i) size *= shape[i];
      std::vector<float> data(size);
      for (float& x : data) x = dist(rng);
      PopulateTensor<float>(id, data);
    };

    fill(q_id_);
    fill(k_id_);
    fill(v_id_);
  }

  TfLiteStatus ApplyCustomDelegate(TfLiteDelegate* delegate) {
    return interpreter_->ModifyGraphWithDelegate(delegate);
  }

  tflite::Interpreter* get_interpreter() { return interpreter_.get(); }

 private:
  int q_id_;
  int k_id_;
  int v_id_;
  int out_id_;
};

void BM_YnnpackSdpaPrefill(benchmark::State& state) {
  const int seq = state.range(0);
  const int head_dim = state.range(1);
  const int heads = state.range(2);
  const int threads = state.range(3);

  SdpaBenchModel model(/*s_q=*/seq, /*s_kv=*/seq, heads, head_dim);
  model.SetRandomData();

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = threads;
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete);

  if (model.ApplyCustomDelegate(delegate.get()) != kTfLiteOk) {
    state.SkipWithError("Failed to apply YNNPACK delegate");
    return;
  }

  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("Failed to warm-up invocation");
    return;
  }

  for (auto _ : state) {
    model.Invoke();
  }

  const double flops = 4.0 * seq * seq * heads * head_dim;
  state.counters["FLOPS"] = benchmark::Counter(flops * state.iterations(),
                                               benchmark::Counter::kIsRate);
}

void BM_YnnpackSdpaDecode(benchmark::State& state) {
  const int seq = state.range(0);
  const int head_dim = state.range(1);
  const int heads = state.range(2);
  const int threads = state.range(3);

  SdpaBenchModel model(/*s_q=*/1, /*s_kv=*/seq, heads, head_dim);
  model.SetRandomData();

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = threads;
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete);

  if (model.ApplyCustomDelegate(delegate.get()) != kTfLiteOk) {
    state.SkipWithError("Failed to apply YNNPACK delegate");
    return;
  }

  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("Failed to warm-up invocation");
    return;
  }

  for (auto _ : state) {
    model.Invoke();
  }

  const double flops = 4.0 * 1 * seq * heads * head_dim;
  state.counters["FLOPS"] = benchmark::Counter(flops * state.iterations(),
                                               benchmark::Counter::kIsRate);
}

BENCHMARK(BM_YnnpackSdpaPrefill)
    ->Args({64, 64, 32, 1})
    ->Args({128, 64, 32, 1})
    ->Args({256, 64, 32, 1})
    ->Args({512, 64, 32, 1})
    ->Args({1024, 64, 32, 1})
    ->Args({2048, 64, 32, 1})
    ->Args({4096, 64, 32, 1})
    ->Args({8192, 64, 32, 1})
    ->Args({1024, 64, 32, 2})
    ->Args({1024, 64, 32, 4})
    ->Args({4096, 64, 32, 4});

BENCHMARK(BM_YnnpackSdpaDecode)
    ->Args({64, 64, 32, 1})
    ->Args({128, 64, 32, 1})
    ->Args({256, 64, 32, 1})
    ->Args({512, 64, 32, 1})
    ->Args({1024, 64, 32, 1})
    ->Args({2048, 64, 32, 1})
    ->Args({4096, 64, 32, 1})
    ->Args({8192, 64, 32, 1})
    ->Args({1024, 64, 32, 2})
    ->Args({1024, 64, 32, 4})
    ->Args({4096, 64, 32, 4});

}  // namespace
}  // namespace ynnpack
}  // namespace tflite

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  std::cout << "\n================ PER-OP PROFILING SUMMARY ================\n";
  tflite::ynnpack::SdpaBenchModel model(/*s_q=*/1024, /*s_kv=*/1024,
                                        /*heads=*/32, /*head_dim=*/64);
  model.SetRandomData();

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  auto delegate = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(&options), TfLiteYNNPackDelegateDelete);

  if (model.ApplyCustomDelegate(delegate.get()) == kTfLiteOk) {
    tflite::profiling::BufferedProfiler profiler(1024);
    model.get_interpreter()->SetProfiler(&profiler);
    profiler.StartProfiling();
    model.Invoke();
    profiler.StopProfiling();

    tflite::profiling::ProfileSummarizer summarizer;
    summarizer.ProcessProfiles(profiler.GetProfileEvents(),
                               *model.get_interpreter());
    std::cout << summarizer.GetOutputString();
  }
  std::cout << "=========================================================\n\n";

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
