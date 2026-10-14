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
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <random>
#include <streambuf>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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

// Shapes of one MoE block in Qwen1.5-MoE-A2.7B, from the `qwen2_moe` config in
// //third_party/py/transformers (whose defaults are that model): hidden_size =
// 2048, moe_intermediate_size = 1408, num_experts = 60, num_experts_per_tok =
// 4. This is the "large expert" regime: one expert's weights do not fit in L2.
//
// The model also has a shared expert (shared_expert_intermediate_size = 5632)
// that every token passes through. That is a plain dense FFN, not part of the
// routed MoE op measured here, so it is not modelled.
constexpr int kQwenDin = 2048;
constexpr int kQwenDmid = 1408;
constexpr int kQwenNumExperts = 60;
constexpr int kQwenTopK = 4;

// Shapes of one MoE block in Gemma 4 700M (the "tiny MoE" / E2B model) as used
// for the end-to-end LiteRT LM measurements. These were read back from the
// YNNPACK subgraph dump of the real model, not from a config file. This is the
// "small expert" regime: one expert's weights fit comfortably in L2.
constexpr int kGemma700MDin = 512;
constexpr int kGemma700MDmid = 448;
constexpr int kGemma700MNumExperts = 32;
constexpr int kGemma700MTopK = 4;

// Shapes of one routed MoE block in Gemma 4 26B-A4B, from the
// `gemma4_26b_a4b` config in //third_party/py/tunix/models/gemma4/config.py
// (cross-checked against `ConfigGemma4_26B_MoE` in
// //third_party/gemma_cpp/gemma/configs.cc and the HF `moe_intermediate_size`
// used in //third_party/py/qwix/google/offline_quantization_example/gemma4.py):
// embed_dim = 2816, expert_dim = 704, num_experts = 128,
// num_experts_per_tok = 8.
//
// Every layer also runs a dense shared MLP (hidden_dim = 2112) in parallel
// with the routed experts. That is a plain FFN, not part of the routed MoE op
// measured here, so it is not modelled.
//
// This is the "many small experts" regime: one expert is ~5.9 MB at int8,
// a third smaller than a Qwen expert, but 8 of the 128 are active per token,
// so a decode step streams ~48 MB of int8 weights against Qwen's ~35 MB and
// 700M's ~2.8 MB, out of a ~760 MB per-block working set.
//
// These arms are expensive: a 26B prefill iteration at seq_len 512 takes
// ~1.4 s at int8, and the FP32 arms allocate ~3 GB of weights, so pass
// --benchmark_filter rather than running the whole file.
constexpr int kGemma26BDin = 2816;
constexpr int kGemma26BDmid = 704;
constexpr int kGemma26BNumExperts = 128;
constexpr int kGemma26BTopK = 8;

// Number of distinct routing buffers the timed loop cycles through. With a
// single fixed buffer the same few experts stay resident in cache across all
// iterations, which is not what a real decode loop does: there, consecutive
// tokens pick different experts and the weights are streamed from memory.
constexpr int kRoutingRingSize = 64;

// Generates expert assignments that look like the output of a trained MoE
// router rather than a round-robin counter.
//
// Experts are given Zipf-distributed popularities w[e] ~ 1 / rank[e]^skew,
// where `rank` is a random permutation of the expert ids so that popular
// experts are not adjacent in memory. Each token then draws K *distinct*
// experts with probability proportional to w, using the Gumbel top-k trick
// (equivalent to Plackett-Luce sampling without replacement).
//
// The `skew` knob is not a load imbalance in itself; the resulting max/mean
// expert load also depends on E and K. Measured over 512 tokens:
//
//   skew   E=32,K=4   E=60,K=4
//   0.0      1.27        1.40     (floor: pure sampling noise)
//   0.2      1.58        1.88
//   0.3      1.89        2.44
//   0.7      3.91        6.05
//   1.2      6.63       11.78
//
// A model trained with a load-balancing auxiliary loss usually lands near
// 1.2-2.0, i.e. skew in the 0.2-0.3 range. Rather than trusting the knob,
// read the `load_imbalance` counter the benchmark emits.
class ExpertRouter {
 public:
  ExpertRouter(int num_experts, int top_k, double skew, uint64_t seed)
      : num_experts_(num_experts), top_k_(top_k), rng_(seed) {
    std::vector<int> rank(num_experts);
    std::iota(rank.begin(), rank.end(), 0);
    std::shuffle(rank.begin(), rank.end(), rng_);
    log_weight_.resize(num_experts);
    for (int e = 0; e < num_experts; ++e) {
      log_weight_[e] = -skew * std::log(static_cast<double>(rank[e]) + 1.0);
    }
  }

  // Fills `out` with `num_tokens * top_k` expert ids.
  void Route(int num_tokens, int32_t* out) {
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 1.0);
    std::vector<std::pair<double, int>> keys(num_experts_);
    for (int t = 0; t < num_tokens; ++t) {
      for (int e = 0; e < num_experts_; ++e) {
        const double gumbel = -std::log(-std::log(uniform(rng_)));
        keys[e] = {log_weight_[e] + gumbel, e};
      }
      absl::c_partial_sort(
          keys, keys.begin() + top_k_,
          [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
          });
      for (int k = 0; k < top_k_; ++k) {
        out[t * top_k_ + k] = keys[k].second;
      }
    }
  }

 private:
  int num_experts_;
  int top_k_;
  std::mt19937_64 rng_;
  std::vector<double> log_weight_;
};

struct RoutingStats {
  // Mean over invocations of (max expert load / mean expert load) within one
  // invocation. 1.0 is perfectly balanced.
  double load_imbalance = 0.0;
  // Mean over invocations of the number of distinct experts one invocation
  // touches.
  double experts_per_invocation = 0.0;
  // Number of distinct experts touched across the whole routing ring. Times
  // the per-expert weight footprint, this is the working set the timed loop
  // actually streams.
  int experts_in_ring = 0;
};

RoutingStats ComputeRoutingStats(
    const std::vector<std::vector<int32_t>>& routing, int num_experts) {
  RoutingStats stats;
  std::vector<int> ring_seen(num_experts, 0);
  for (const std::vector<int32_t>& buffer : routing) {
    std::vector<int> count(num_experts, 0);
    for (int32_t e : buffer) {
      ++count[e];
      ring_seen[e] = 1;
    }
    int max_count = 0;
    int distinct = 0;
    for (int c : count) {
      max_count = std::max(max_count, c);
      distinct += c > 0 ? 1 : 0;
    }
    const double mean =
        static_cast<double>(buffer.size()) / static_cast<double>(num_experts);
    stats.load_imbalance += static_cast<double>(max_count) / mean;
    stats.experts_per_invocation += distinct;
  }
  const double n = static_cast<double>(routing.size());
  stats.load_imbalance /= n;
  stats.experts_per_invocation /= n;
  stats.experts_in_ring =
      std::accumulate(ring_seen.begin(), ring_seen.end(), 0);
  return stats;
}

class MoeModel : public SingleOpModel {
 public:
  MoeModel(int B, int N, int D_in, int D_mid, int E, int K, int num_threads,
           bool use_delegate)
      : w_gate_data_(static_cast<size_t>(D_mid) * E * D_in),
        w_up_data_(static_cast<size_t>(D_mid) * E * D_in),
        w_down_data_(static_cast<size_t>(D_in) * E * D_mid),
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

    // Note: there is no undelegated arm here. The model contains a single
    // stablehlo.composite op whose decomposition subgraph is not materialized
    // by SingleOpModel, so it can only run through the YNNPACK delegate. Use
    // the end-to-end LiteRT LM benchmark to get a delegated/undelegated ratio.
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

  // Size of one stored weight element, for the working-set counter.
  static constexpr double kWeightBytes = sizeof(float);

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

// Storage precision of the expert weights in `QuantizedMoeModel`. int4 and
// int2 weights are stored packed, 2 respectively 4 values per byte.
enum class WeightPrecision { kInt8, kInt4, kInt2 };

template <WeightPrecision P>
struct WeightTraits;

template <>
struct WeightTraits<WeightPrecision::kInt8> {
  static constexpr TensorType kTensorType = TensorType_INT8;
  static constexpr TfLiteType kTfLiteType = kTfLiteInt8;
  static constexpr int kValuesPerByte = 1;
  // Weights cycle through [-kValuePeriod / 2, kValuePeriod / 2].
  static constexpr int kValuePeriod = 31;
};

template <>
struct WeightTraits<WeightPrecision::kInt4> {
  static constexpr TensorType kTensorType = TensorType_INT4;
  static constexpr TfLiteType kTfLiteType = kTfLiteInt4;
  static constexpr int kValuesPerByte = 2;
  static constexpr int kValuePeriod = 15;
};

template <>
struct WeightTraits<WeightPrecision::kInt2> {
  static constexpr TensorType kTensorType = TensorType_INT2;
  static constexpr TfLiteType kTfLiteType = kTfLiteInt2;
  static constexpr int kValuesPerByte = 4;
  static constexpr int kValuePeriod = 3;
};

// The dynamically quantized counterpart of `MoeModel`, with the weights stored
// at the precision `P` (W8A8, W4A8 or W2A8).
//
// This is the 10-input form of the op: each weight tensor is followed by its
// own FP32 per-output-channel scale tensor. Activations are quantized by the
// delegate at run time, so the inputs and the output stay FP32 and the
// benchmark loop is identical to the float one.
//
// Like `MoeModel`, the weights are attached with
// `SetTensorParametersReadOnly` rather than `AddConstInput`. At Qwen shapes
// the three weight tensors are ~600 MB and embedding them in the flatbuffer
// would be slow and would risk the 2 GB flatbuffer limit. It also matters
// for what this benchmark measures: read-only tensors are what the delegate
// treats as constant, which is the precondition for the weight-gather
// fusions to fire.
template <WeightPrecision P>
class QuantizedMoeModel : public SingleOpModel {
 public:
  QuantizedMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                    int num_threads, bool use_delegate)
      : w_gate_data_(static_cast<size_t>(D_mid) * E * D_in),
        w_up_data_(static_cast<size_t>(D_mid) * E * D_in),
        w_down_data_(static_cast<size_t>(D_in) * E * D_mid),
        gate_scale_data_(static_cast<size_t>(D_mid) * E),
        up_scale_data_(static_cast<size_t>(D_mid) * E),
        down_scale_data_(static_cast<size_t>(D_in) * E),
        scale_data_(E) {
    std::vector<int> tokens_shape = {B, N, D_in};
    std::vector<int> rw_shape = {B, N, K};
    std::vector<int> ei_shape = {B, N, K};
    std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int> gate_scale_shape = {D_mid, E, 1, 1};
    std::vector<int> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int> up_scale_shape = {D_mid, E, 1, 1};
    std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int> down_scale_shape = {D_in, E, 1, 1};
    std::vector<int> scale_shape = {1, 1, 1, E};
    std::vector<int> out_shape = {B, N, D_in};

    tokens_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    rw_id_ = AddInput({TensorType_FLOAT32, rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_shape});
    w_gate_id_ = AddInput({WeightTraits<P>::kTensorType, w_gate_shape});
    gate_scale_id_ = AddInput({TensorType_FLOAT32, gate_scale_shape});
    w_up_id_ = AddInput({WeightTraits<P>::kTensorType, w_up_shape});
    up_scale_id_ = AddInput({TensorType_FLOAT32, up_scale_shape});
    w_down_id_ = AddInput({WeightTraits<P>::kTensorType, w_down_shape});
    down_scale_id_ = AddInput({TensorType_FLOAT32, down_scale_shape});
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
                      gate_scale_shape, w_up_shape, up_scale_shape,
                      w_down_shape, down_scale_shape, scale_shape},
                     -1, false, false,
                     /*allocate_and_delegate=*/false);

    // Spread the weights over the representable range rather than filling with
    // a constant, so that neither the quantized dot nor the zero-point
    // correction can be short-circuited by a degenerate value.
    constexpr int period = WeightTraits<P>::kValuePeriod;
    for (size_t i = 0; i < w_gate_data_.size(); ++i) {
      w_gate_data_[i] = static_cast<int8_t>((i % period) - period / 2);
    }
    for (size_t i = 0; i < w_up_data_.size(); ++i) {
      w_up_data_[i] = static_cast<int8_t>(((i + 7) % period) - period / 2);
    }
    for (size_t i = 0; i < w_down_data_.size(); ++i) {
      w_down_data_[i] = static_cast<int8_t>(((i + 13) % period) - period / 2);
    }
    std::fill(gate_scale_data_.begin(), gate_scale_data_.end(), 0.005f);
    std::fill(up_scale_data_.begin(), up_scale_data_.end(), 0.005f);
    std::fill(down_scale_data_.begin(), down_scale_data_.end(), 0.005f);
    std::fill(scale_data_.begin(), scale_data_.end(), 1.0f);

    SetReadOnlyWeights(w_gate_id_, "gate_weights", w_gate_shape, w_gate_data_);
    SetReadOnlyWeights(w_up_id_, "up_weights", w_up_shape, w_up_data_);
    SetReadOnlyWeights(w_down_id_, "down_weights", w_down_shape, w_down_data_);
    SetReadOnlyFloat(gate_scale_id_, "gate_scale", gate_scale_shape,
                     gate_scale_data_);
    SetReadOnlyFloat(up_scale_id_, "up_scale", up_scale_shape, up_scale_data_);
    SetReadOnlyFloat(down_scale_id_, "down_scale", down_scale_shape,
                     down_scale_data_);
    SetReadOnlyFloat(scale_id_, "scale", scale_shape, scale_data_);

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

  static constexpr double kWeightBytes = 1.0 / WeightTraits<P>::kValuesPerByte;

 private:
  // Packs `data` the way LiteRT stores sub-byte tensors (lowest-index value in
  // the least significant bits) and hands the result to the interpreter as a
  // read-only tensor. `packed_weights_` keeps the storage alive: the
  // interpreter does not copy it.
  void SetReadOnlyWeights(int id, const char* name,
                          const std::vector<int>& shape,
                          const std::vector<int8_t>& data) {
    constexpr int kValuesPerByte = WeightTraits<P>::kValuesPerByte;
    const char* bytes = reinterpret_cast<const char*>(data.data());
    size_t num_bytes = data.size();
    if (kValuesPerByte > 1) {
      constexpr int kBits = 8 / kValuesPerByte;
      constexpr int kMask = (1 << kBits) - 1;
      std::vector<uint8_t> packed(
          (data.size() + kValuesPerByte - 1) / kValuesPerByte, 0);
      for (size_t i = 0; i < data.size(); ++i) {
        packed[i / kValuesPerByte] |= static_cast<uint8_t>(
            (data[i] & kMask) << (kBits * (i % kValuesPerByte)));
      }
      num_bytes = packed.size();
      packed_weights_.push_back(std::move(packed));
      bytes = reinterpret_cast<const char*>(packed_weights_.back().data());
    }
    interpreter_->SetTensorParametersReadOnly(id, WeightTraits<P>::kTfLiteType,
                                              name, shape, TfLiteQuantization(),
                                              bytes, num_bytes);
  }

  void SetReadOnlyFloat(int id, const char* name, const std::vector<int>& shape,
                        const std::vector<float>& data) {
    interpreter_->SetTensorParametersReadOnly(
        id, kTfLiteFloat32, name, shape, TfLiteQuantization(),
        reinterpret_cast<const char*>(data.data()),
        data.size() * sizeof(float));
  }

  int tokens_id_;
  int rw_id_;
  int ei_id_;
  int w_gate_id_;
  int gate_scale_id_;
  int w_up_id_;
  int up_scale_id_;
  int w_down_id_;
  int down_scale_id_;
  int scale_id_;
  int out_id_;

  std::vector<int8_t> w_gate_data_;
  std::vector<int8_t> w_up_data_;
  std::vector<int8_t> w_down_data_;
  // Packed copies of the weights above, for the sub-byte precisions.
  std::vector<std::vector<uint8_t>> packed_weights_;
  std::vector<float> gate_scale_data_;
  std::vector<float> up_scale_data_;
  std::vector<float> down_scale_data_;
  std::vector<float> scale_data_;
};

// `SingleOpModel::ApplyDelegate` logs a warning every time a delegate is set
// manually, and `TFLITE_LOG` writes to `std::cout` unconditionally: it has no
// severity filter, and despite its comment it does not use stderr. Google
// Benchmark also writes its report to `std::cout`, so with
// `--benchmark_format=json` those lines land in the middle of the JSON
// document and make it unparseable. benchy then fails the whole run with
// "invalid character 'W' looking for beginning of value".
//
// Point `std::cout` at `std::cerr` for the duration of the noisy window so
// the diagnostics are still visible, just not on the machine-readable
// stream.
class ScopedCoutToCerr {
 public:
  ScopedCoutToCerr() : saved_(std::cout.rdbuf(std::cerr.rdbuf())) {}
  ~ScopedCoutToCerr() { std::cout.rdbuf(saved_); }

  ScopedCoutToCerr(const ScopedCoutToCerr&) = delete;
  ScopedCoutToCerr& operator=(const ScopedCoutToCerr&) = delete;

 private:
  std::streambuf* saved_;
};

template <typename ModelT>
void BenchMoe(benchmark::State& state, int B, int N, int D_in, int D_mid, int E,
              int K, int num_threads, double skew, bool vary_routing) {
  // Held for the whole body: the benchmark framework emits its report after
  // this function returns, so nothing of ours is diverted.
  ScopedCoutToCerr redirect_tflite_logs;

  ModelT model(B, N, D_in, D_mid, E, K, num_threads, /*use_delegate=*/true);

  const int num_tokens = B * N;
  std::vector<float> tokens(static_cast<size_t>(num_tokens) * D_in, 0.5f);
  std::vector<float> rw(static_cast<size_t>(num_tokens) * K,
                        1.0f / static_cast<float>(K));

  const int ring_size = vary_routing ? kRoutingRingSize : 1;
  ExpertRouter router(E, K, skew, /*seed=*/0x5eed1234u);
  std::vector<std::vector<int32_t>> routing(ring_size);
  for (std::vector<int32_t>& buffer : routing) {
    buffer.resize(static_cast<size_t>(num_tokens) * K);
    router.Route(num_tokens, buffer.data());
  }

  model.PopulateTensor(model.tokens(), tokens);
  model.PopulateTensor(model.rw(), rw);
  model.PopulateTensor(model.ei(), routing[0]);

  if (model.Invoke() != kTfLiteOk) {
    state.SkipWithError("Failed to invoke delegate interpreter (warmup)");
    return;
  }

  // The routing tensor is repopulated on every iteration in both arms so that
  // the fixed and varying arms differ only in the *content* of the routing,
  // never in the work done outside the delegate.
  int index = 0;
  for (auto _ : state) {
    model.PopulateTensor(model.ei(), routing[index]);
    index = index + 1 == ring_size ? 0 : index + 1;
    if (model.Invoke() != kTfLiteOk) {
      state.SkipWithError("Failed to invoke interpreter");
      return;
    }
  }

  // 2 matmuls of [D_in, D_mid] (gate and up) + 1 matmul of [D_mid, D_in]
  // (down) per (token, expert) pair.
  const size_t flops = 6ull * static_cast<size_t>(num_tokens) *
                       static_cast<size_t>(K) * static_cast<size_t>(D_in) *
                       static_cast<size_t>(D_mid);
  state.counters["FLOP"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * static_cast<double>(flops),
      benchmark::Counter::kIsRate);

  const RoutingStats stats = ComputeRoutingStats(routing, E);
  state.counters["load_imbalance"] = stats.load_imbalance;
  state.counters["experts_per_invoke"] = stats.experts_per_invocation;
  // Weight bytes touched over one full cycle of the routing ring.
  const double expert_bytes =
      3.0 * static_cast<double>(D_in) * D_mid * ModelT::kWeightBytes;
  state.counters["working_set_MB"] =
      stats.experts_in_ring * expert_bytes / (1024.0 * 1024.0);
}

// The argument unpacking is shared so that the FP32 and INT8 arms of a given
// shape regime are guaranteed to see identical sequence lengths, thread
// counts, routing skew and routing ring.
template <typename ModelT>
void DecodeBench(benchmark::State& state, int D_in, int D_mid, int E, int K) {
  BenchMoe<ModelT>(state, /*B=*/1, /*N=*/1, D_in, D_mid, E, K,
                   /*num_threads=*/state.range(0),
                   /*skew=*/state.range(1) / 10.0,
                   /*vary_routing=*/state.range(2) != 0);
}

template <typename ModelT>
void PrefillBench(benchmark::State& state, int D_in, int D_mid, int E, int K) {
  BenchMoe<ModelT>(state, /*B=*/1, /*N=*/state.range(0), D_in, D_mid, E, K,
                   /*num_threads=*/state.range(1),
                   /*skew=*/state.range(2) / 10.0, /*vary_routing=*/true);
}

void QwenMoeDecodeFloat32(benchmark::State& state) {
  DecodeBench<MoeModel>(state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void QwenMoeDecodeInt8(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt8>>(
      state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void QwenMoeDecodeInt4(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt4>>(
      state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void QwenMoeDecodeInt2(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt2>>(
      state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void QwenMoePrefillFloat32(benchmark::State& state) {
  PrefillBench<MoeModel>(state, kQwenDin, kQwenDmid, kQwenNumExperts,
                         kQwenTopK);
}

void QwenMoePrefillInt8(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt8>>(
      state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void QwenMoePrefillInt4(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt4>>(
      state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void QwenMoePrefillInt2(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt2>>(
      state, kQwenDin, kQwenDmid, kQwenNumExperts, kQwenTopK);
}

void Gemma700MMoeDecodeFloat32(benchmark::State& state) {
  DecodeBench<MoeModel>(state, kGemma700MDin, kGemma700MDmid,
                        kGemma700MNumExperts, kGemma700MTopK);
}

void Gemma700MMoeDecodeInt8(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt8>>(
      state, kGemma700MDin, kGemma700MDmid, kGemma700MNumExperts,
      kGemma700MTopK);
}

void Gemma700MMoeDecodeInt4(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt4>>(
      state, kGemma700MDin, kGemma700MDmid, kGemma700MNumExperts,
      kGemma700MTopK);
}

void Gemma700MMoeDecodeInt2(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt2>>(
      state, kGemma700MDin, kGemma700MDmid, kGemma700MNumExperts,
      kGemma700MTopK);
}

void Gemma700MMoePrefillFloat32(benchmark::State& state) {
  PrefillBench<MoeModel>(state, kGemma700MDin, kGemma700MDmid,
                         kGemma700MNumExperts, kGemma700MTopK);
}

void Gemma700MMoePrefillInt8(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt8>>(
      state, kGemma700MDin, kGemma700MDmid, kGemma700MNumExperts,
      kGemma700MTopK);
}

void Gemma700MMoePrefillInt4(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt4>>(
      state, kGemma700MDin, kGemma700MDmid, kGemma700MNumExperts,
      kGemma700MTopK);
}

void Gemma700MMoePrefillInt2(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt2>>(
      state, kGemma700MDin, kGemma700MDmid, kGemma700MNumExperts,
      kGemma700MTopK);
}

// 26B-A4B holds 3 * 2816 * 704 * 128 = 761M weight values per MoE block, so
// the three weight tensors are ~3.0 GB in FP32, ~760 MB at int8, ~380 MB at
// int4 and ~190 MB at int2. The FP32 arm is a reference point, not a
// deployable configuration.
void Gemma26BMoeDecodeFloat32(benchmark::State& state) {
  DecodeBench<MoeModel>(state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts,
                        kGemma26BTopK);
}

void Gemma26BMoeDecodeInt8(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt8>>(
      state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts, kGemma26BTopK);
}

void Gemma26BMoeDecodeInt4(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt4>>(
      state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts, kGemma26BTopK);
}

void Gemma26BMoeDecodeInt2(benchmark::State& state) {
  DecodeBench<QuantizedMoeModel<WeightPrecision::kInt2>>(
      state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts, kGemma26BTopK);
}

void Gemma26BMoePrefillFloat32(benchmark::State& state) {
  PrefillBench<MoeModel>(state, kGemma26BDin, kGemma26BDmid,
                         kGemma26BNumExperts, kGemma26BTopK);
}

void Gemma26BMoePrefillInt8(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt8>>(
      state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts, kGemma26BTopK);
}

void Gemma26BMoePrefillInt4(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt4>>(
      state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts, kGemma26BTopK);
}

void Gemma26BMoePrefillInt2(benchmark::State& state) {
  PrefillBench<QuantizedMoeModel<WeightPrecision::kInt2>>(
      state, kGemma26BDin, kGemma26BDmid, kGemma26BNumExperts, kGemma26BTopK);
}

void DecodeArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"threads", "skew_x10", "vary_routing"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int threads : {1, 4}) {
    // vary_routing = 0 keeps the same experts hot in cache across iterations
    // and is only kept so the cache effect can be quantified; 1 is the
    // realistic arm.
    for (int vary : {0, 1}) {
      b->Args({threads, 3, vary});
    }
  }
}

void PrefillArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"seq_len", "threads", "skew_x10"});
  b->UseRealTime();
  b->MeasureProcessCPUTime();
  for (int seq_len : {128, 256, 512}) {
    for (int threads : {1, 4}) {
      // Uniform / realistic / pathological expert load.
      for (int skew : {0, 3, 12}) {
        b->Args({seq_len, threads, skew});
      }
    }
  }
}

BENCHMARK(QwenMoeDecodeFloat32)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoeDecodeInt8)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoeDecodeInt4)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoeDecodeInt2)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoePrefillFloat32)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoePrefillInt8)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoePrefillInt4)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(QwenMoePrefillInt2)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoeDecodeFloat32)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoeDecodeInt8)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoeDecodeInt4)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoeDecodeInt2)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoePrefillFloat32)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoePrefillInt8)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoePrefillInt4)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma700MMoePrefillInt2)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoeDecodeFloat32)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoeDecodeInt8)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoeDecodeInt4)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoeDecodeInt2)
    ->Apply(DecodeArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoePrefillFloat32)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoePrefillInt8)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoePrefillInt4)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

BENCHMARK(Gemma26BMoePrefillInt2)
    ->Apply(PrefillArguments)
    ->Unit(benchmark::TimeUnit::kMillisecond);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
