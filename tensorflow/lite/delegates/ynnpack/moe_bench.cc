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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ynnpack {
namespace {

// Parameters from Gemma4 26B-A4B MoE model:
// embed_dim = 2816 (D_in / D_out), expert_dim = 704 (D_mid),
// num_experts = 128 (E), top_k = 8 (K).
constexpr int kGemma4Din = 2816;
constexpr int kGemma4Dmid = 704;
constexpr int kGemma4NumExperts = 128;
constexpr int kGemma4TopK = 8;

class MoeModel : public SingleOpModel {
 public:
  MoeModel(int B, int N, int D_in, int D_mid, int E, int K, int num_threads,
           bool use_delegate)
      : w_gate_data_(D_mid * E * D_in),
        w_up_data_(D_mid * E * D_in),
        w_down_data_(D_in * E * D_mid),
        scale_data_(E) {
    std::vector<int> tokens_shape = {B, N, D_in};
    std::vector<int> rw_shape = {B, N, K};
    std::vector<int> ei_shape = {B, N, K};
    std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int> scale_shape = {1, 1, 1, E};
    std::vector<int> out_shape = {B, N, D_in};

    tokens_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    rw_id_ = AddInput({TensorType_FLOAT32, rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_shape});
    w_gate_id_ = AddInput({TensorType_FLOAT32, w_gate_shape});
    w_up_id_ = AddInput({TensorType_FLOAT32, w_up_shape});
    w_down_id_ = AddInput({TensorType_FLOAT32, w_down_shape});
    scale_id_ = AddInput({TensorType_FLOAT32, scale_shape});

    out_id_ = AddOutput({TensorType_FLOAT32, out_shape});

    std::vector<uint8_t> empty_attrs;
    flatbuffers::Offset<StableHLOCompositeOptions> options =
        CreateStableHLOCompositeOptionsDirect(
            builder_, "odml.moe_experts",
            /*decomposition_subgraph_index=*/1, &empty_attrs);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions, options.Union());

    BuildInterpreter({tokens_shape, rw_shape, ei_shape, w_gate_shape,
                      w_up_shape, w_down_shape, scale_shape},
                     -1, false, false,
                     /*allocate_and_delegate=*/false);

    std::fill(w_gate_data_.begin(), w_gate_data_.end(), 0.02f);
    std::fill(w_up_data_.begin(), w_up_data_.end(), 0.02f);
    std::fill(w_down_data_.begin(), w_down_data_.end(), 0.02f);
    std::fill(scale_data_.begin(), scale_data_.end(), 1.0f);

    interpreter_->SetTensorParametersReadOnly(
        w_gate_id_, kTfLiteFloat32, "gate_weights", w_gate_shape,
        TfLiteQuantization(),
        reinterpret_cast<const char*>(w_gate_data_.data()),
        w_gate_data_.size() * sizeof(float));
    interpreter_->SetTensorParametersReadOnly(
        w_up_id_, kTfLiteFloat32, "up_weights", w_up_shape,
        TfLiteQuantization(), reinterpret_cast<const char*>(w_up_data_.data()),
        w_up_data_.size() * sizeof(float));
    interpreter_->SetTensorParametersReadOnly(
        w_down_id_, kTfLiteFloat32, "down_weights", w_down_shape,
        TfLiteQuantization(),
        reinterpret_cast<const char*>(w_down_data_.data()),
        w_down_data_.size() * sizeof(float));
    interpreter_->SetTensorParametersReadOnly(
        scale_id_, kTfLiteFloat32, "scale", scale_shape, TfLiteQuantization(),
        reinterpret_cast<const char*>(scale_data_.data()),
        scale_data_.size() * sizeof(float));

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = num_threads;
      delegate_options.static_shape = true;
      SetDelegate(Interpreter::TfLiteDelegatePtr(
          TfLiteYNNPackDelegateCreate(&delegate_options),
          TfLiteYNNPackDelegateDelete));
      ApplyDelegate();
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors\n");
    }
  }

  int tokens() const { return tokens_id_; }
  int rw() const { return rw_id_; }
  int ei() const { return ei_id_; }
  int out() const { return out_id_; }

 private:
  int tokens_id_;
  int rw_id_;
  int ei_id_;
  int w_gate_id_;
  int w_up_id_;
  int w_down_id_;
  int scale_id_;
  int out_id_;

  std::vector<float> w_gate_data_;
  std::vector<float> w_up_data_;
  std::vector<float> w_down_data_;
  std::vector<float> scale_data_;
};

void BenchMoe(benchmark::State& state, int B, int N, int D_in, int D_mid, int E,
              int K, int num_threads) {
  MoeModel model(B, N, D_in, D_mid, E, K, num_threads, /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in, 0.5f);
  std::vector<float> rw(B * N * K, 1.0f / static_cast<float>(K));
  std::vector<int32_t> ei(B * N * K);
  for (int n = 0; n < B * N; ++n) {
    for (int k = 0; k < K; ++k) {
      ei[n * K + k] = (n * K + k) % E;
    }
  }

  model.PopulateTensor(model.tokens(), tokens);
  model.PopulateTensor(model.rw(), rw);
  model.PopulateTensor(model.ei(), ei);

  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("Failed to invoke delegate interpreter (warmup)");
    return;
  }

  for (auto _ : state) {
    if (model.Invoke() != kTfLiteOk) {
      state.SkipWithError("Failed to invoke interpreter");
      return;
    }
  }

  // 2 matmuls of [D_in, D_mid] (gate and up) + 1 matmul of [D_mid, D_in] (down)
  // per expert choice.
  const size_t flops = 6ull * static_cast<size_t>(B) * static_cast<size_t>(N) *
                       static_cast<size_t>(K) * static_cast<size_t>(D_in) *
                       static_cast<size_t>(D_mid);
  state.counters["FLOP"] =
      benchmark::Counter(static_cast<double>(state.iterations() * flops),
                         benchmark::Counter::kIsRate);
}

void Gemma4MoEDecode(benchmark::State& state) {
  const int num_threads = state.range(0);
  BenchMoe(state, /*B=*/1, /*N=*/1, kGemma4Din, kGemma4Dmid, kGemma4NumExperts,
           kGemma4TopK, num_threads);
}

void Gemma4MoEPrefill(benchmark::State& state) {
  const int seq_len = state.range(0);
  const int num_threads = state.range(1);
  BenchMoe(state, /*B=*/1, seq_len, kGemma4Din, kGemma4Dmid, kGemma4NumExperts,
           kGemma4TopK, num_threads);
}

void DecodeArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"threads"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int threads : {1, 4}) {
    b->Args({threads});
  }
}

void PrefillArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"seq_len", "threads"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int seq_len : {64, 256, 1024}) {
    for (int threads : {1, 4}) {
      b->Args({seq_len, threads});
    }
  }
}

BENCHMARK(Gemma4MoEDecode)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma4MoEPrefill)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
