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

// Tests for the QSA StableHLO composite kernels of the XNNPACK delegate
// (`odml.sparse_sdpa_transposed` and `odml.qsa_indexer_update`). Each test
// builds a single-composite interpreter, applies the XNNPACK delegate and
// compares the result with a brute-force transcription of the composites'
// PyTorch decompositions.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace xnnpack {
namespace {

constexpr float kMaskFill = -10000.0f;

// ---------------------------------------------------------------------------
// fp16 helpers.
// ---------------------------------------------------------------------------

uint16_t ToHalf(float f) {
  // Round-to-nearest-even conversion, adequate for finite test values.
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t half_mant = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half_mant & 1))) ++half_mant;
    return static_cast<uint16_t>(sign | half_mant);
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
  return static_cast<uint16_t>(h);
}

float FromHalf(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1F;
  const uint32_t mant = h & 0x3FFu;
  float f;
  if (exp == 0) {
    f = std::ldexp(static_cast<float>(mant), -24);
  } else if (exp == 31) {
    f = mant ? NAN : INFINITY;
  } else {
    f = std::ldexp(static_cast<float>(mant | 0x400u),
                   static_cast<int>(exp) - 25);
  }
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  bits |= sign;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

float Round(TfLiteType type, float v) {
  return type == kTfLiteFloat16 ? FromHalf(ToHalf(v)) : v;
}

std::vector<float> RandomValues(std::mt19937& rng, size_t n, TfLiteType type,
                                float lo = -1.0f, float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = Round(type, dist(rng));
  return v;
}

size_t Product(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         [](size_t a, int b) { return a * b; });
}

// ---------------------------------------------------------------------------
// Single-composite interpreter.
// ---------------------------------------------------------------------------

TfLiteStatus UnexpectedInvoke(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context, "QSA composite was not delegated");
  return kTfLiteError;
}

class CompositeModel {
 public:
  CompositeModel() : interpreter_(std::make_unique<Interpreter>()) {
    registration_ = {};
    registration_.builtin_code = kTfLiteBuiltinStablehloComposite;
    registration_.invoke = UnexpectedInvoke;
    registration_.version = 1;
  }

  ~CompositeModel() { interpreter_.reset(); }

  int AddInput(TfLiteType type, const std::vector<int>& shape) {
    const int id = AddTensor(type, shape);
    inputs_.push_back(id);
    return id;
  }

  int AddOutput(TfLiteType type, const std::vector<int>& shape) {
    const int id = AddTensor(type, shape);
    outputs_.push_back(id);
    return id;
  }

  void Build(const char* name, const std::vector<uint8_t>& attributes) {
    attributes_ = attributes;
    auto* params = static_cast<TfLiteStablehloCompositeParams*>(
        std::malloc(sizeof(TfLiteStablehloCompositeParams)));
    params->name = name;
    params->subgraph_index = 0;
    params->version = 1;
    params->attributes = attributes_.data();
    params->attributes_size = attributes_.size();
    ASSERT_EQ(interpreter_->SetInputs(inputs_), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs(outputs_), kTfLiteOk);
    ASSERT_EQ(interpreter_->AddNodeWithParameters(inputs_, outputs_, nullptr, 0,
                                                  params, &registration_),
              kTfLiteOk);
  }

  // Applies the XNNPACK delegate and checks that the composite was claimed.
  void Delegate(int num_threads) {
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete);
    ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(delegate_.get()),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->execution_plan().size(), 1);
    const auto* node_and_reg =
        interpreter_->node_and_registration(interpreter_->execution_plan()[0]);
    ASSERT_NE(node_and_reg->second.custom_name, nullptr);
    ASSERT_STREQ(node_and_reg->second.custom_name, "TfLiteXNNPackDelegate");
  }

  // Applies the XNNPACK delegate and returns whether the composite was
  // claimed by it.
  bool TryDelegate(int num_threads) {
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads;
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        TfLiteXNNPackDelegateCreate(&options), TfLiteXNNPackDelegateDelete);
    if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
      return false;
    }
    const auto* node_and_reg =
        interpreter_->node_and_registration(interpreter_->execution_plan()[0]);
    return node_and_reg->second.custom_name != nullptr &&
           std::strcmp(node_and_reg->second.custom_name,
                       "TfLiteXNNPackDelegate") == 0;
  }

  // Makes `a` and `b` share one buffer (in-place state update).
  void Alias(int a, int b) {
    const size_t bytes = interpreter_->tensor(a)->bytes;
    ASSERT_EQ(bytes, interpreter_->tensor(b)->bytes);
    alias_buffers_.emplace_back(bytes + kDefaultTensorAlignment);
    auto& buffer = alias_buffers_.back();
    const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.data());
    void* aligned = reinterpret_cast<void*>(
        (base + kDefaultTensorAlignment - 1) & ~(kDefaultTensorAlignment - 1));
    const TfLiteCustomAllocation alloc{aligned, bytes};
    ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(a, alloc), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(b, alloc), kTfLiteOk);
  }

  void Allocate() { ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk); }

  void Invoke() { ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk); }

  void SetFloat(int id, const std::vector<float>& values) {
    TfLiteTensor* t = interpreter_->tensor(id);
    ASSERT_EQ(values.size(), Product(Shape(id)));
    if (t->type == kTfLiteFloat16) {
      uint16_t* dst = reinterpret_cast<uint16_t*>(t->data.raw);
      for (size_t i = 0; i < values.size(); ++i) dst[i] = ToHalf(values[i]);
    } else {
      ASSERT_EQ(t->type, kTfLiteFloat32);
      std::memcpy(t->data.raw, values.data(), values.size() * sizeof(float));
    }
  }

  void SetBool(int id, const std::vector<bool>& values) {
    TfLiteTensor* t = interpreter_->tensor(id);
    ASSERT_EQ(values.size(), Product(Shape(id)));
    for (size_t i = 0; i < values.size(); ++i) t->data.b[i] = values[i];
  }

  void SetInt32(int id, const std::vector<int32_t>& values) {
    TfLiteTensor* t = interpreter_->tensor(id);
    ASSERT_EQ(values.size(), Product(Shape(id)));
    std::memcpy(t->data.raw, values.data(), values.size() * sizeof(int32_t));
  }

  std::vector<float> GetFloat(int id) {
    const TfLiteTensor* t = interpreter_->tensor(id);
    const size_t n = Product(Shape(id));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
      out[i] = t->type == kTfLiteFloat16
                   ? FromHalf(reinterpret_cast<const uint16_t*>(t->data.raw)[i])
                   : t->data.f[i];
    }
    return out;
  }

  std::vector<int32_t> GetInt32(int id) {
    const TfLiteTensor* t = interpreter_->tensor(id);
    return std::vector<int32_t>(t->data.i32, t->data.i32 + Product(Shape(id)));
  }

  std::vector<int> Shape(int id) {
    const TfLiteTensor* t = interpreter_->tensor(id);
    return std::vector<int>(t->dims->data, t->dims->data + t->dims->size);
  }

 private:
  int AddTensor(TfLiteType type, const std::vector<int>& shape) {
    int id = -1;
    EXPECT_EQ(interpreter_->AddTensors(1, &id), kTfLiteOk);
    EXPECT_EQ(interpreter_->SetTensorParametersReadWrite(id, type, "", shape,
                                                         TfLiteQuantization{}),
              kTfLiteOk);
    return id;
  }

  std::unique_ptr<Interpreter> interpreter_;
  TfLiteRegistration registration_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<uint8_t> attributes_;
  std::vector<std::vector<char>> alias_buffers_;
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_{
      nullptr, TfLiteXNNPackDelegateDelete};
};

// Builds a causal [B=1, 1, S, T] mask: row s sees tokens [0, pos0 + s], except
// rows >= valid_rows, which see only [0, pos0 + valid_rows) (right padding).
std::vector<bool> CausalMask(int seq_len, int kv_len, int pos0,
                             int valid_rows) {
  std::vector<bool> mask(static_cast<size_t>(seq_len) * kv_len, false);
  for (int s = 0; s < seq_len; ++s) {
    const int last = s < valid_rows ? pos0 + s : pos0 + valid_rows - 1;
    for (int t = 0; t <= last && t < kv_len; ++t) {
      mask[static_cast<size_t>(s) * kv_len + t] = true;
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// odml.sparse_sdpa_transposed
// ---------------------------------------------------------------------------

struct SdpaConfig {
  int batch = 1;
  TfLiteType type = kTfLiteFloat16;
  TfLiteType mask_type = kTfLiteBool;
  int num_q_heads = 8;
  int num_kv_heads = 2;
  int seq_len = 1;
  int head_dim = 32;
  int kv_len = 130;
  int ratio = 4;
  int block_topk = 8;
  int k_ts_idx = 2;
  int v_ts_idx = 3;
  int pos0 = 97;
  int num_threads = 4;
};

std::vector<uint8_t> SdpaAttributes(const SdpaConfig& c, float scale) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("compress_ratio", c.ratio);
    fbb.Int("block_topk", c.block_topk);
    fbb.Float("scale", scale);
    fbb.Int("k_ts_idx", c.k_ts_idx);
    fbb.Int("v_ts_idx", c.v_ts_idx);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// Brute-force transcription of `apply_sparse_sdpa_transposed`.
std::vector<float> ReferenceSparseSdpa(const SdpaConfig& c, float scale,
                                       const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       const std::vector<int32_t>& topk,
                                       const std::vector<bool>& vis) {
  const int H = c.num_q_heads, S = c.seq_len, D = c.head_dim, T = c.kv_len;
  const int G = c.num_q_heads / c.num_kv_heads;
  const int max_blocks = T / c.ratio;
  const int block_tokens = max_blocks * c.ratio;
  auto key = [&](int h, int t, int d) {
    return c.k_ts_idx == 2 ? k[(static_cast<size_t>(h) * T + t) * D + d]
                           : k[(static_cast<size_t>(h) * D + d) * T + t];
  };
  auto value = [&](int h, int t, int d) {
    return c.v_ts_idx == 3 ? v[(static_cast<size_t>(h) * D + d) * T + t]
                           : v[(static_cast<size_t>(h) * T + t) * D + d];
  };
  std::vector<float> out(static_cast<size_t>(S) * H * D, 0.0f);
  for (int s = 0; s < S; ++s) {
    std::vector<bool> keep(max_blocks, false);
    for (int i = 0; i < c.block_topk; ++i) {
      const int idx = topk[static_cast<size_t>(s) * c.block_topk + i];
      if (idx >= 0 && idx < max_blocks) keep[idx] = true;
    }
    for (int j = 0; j < max_blocks; ++j) {
      bool complete = true;
      for (int r = 0; r < c.ratio; ++r) {
        complete =
            complete && vis[static_cast<size_t>(s) * T + j * c.ratio + r];
      }
      keep[j] = keep[j] || !complete;
    }
    std::vector<bool> eff(T);
    bool any = false;
    for (int t = 0; t < T; ++t) {
      eff[t] = vis[static_cast<size_t>(s) * T + t] &&
               (t >= block_tokens || keep[t / c.ratio]);
      any = any || eff[t];
    }
    for (int h = 0; h < H; ++h) {
      const int kvh = h / G;
      std::vector<double> logits(T);
      double mx = -1e30;
      for (int t = 0; t < T; ++t) {
        double dot = 0.0;
        for (int d = 0; d < D; ++d) {
          dot +=
              static_cast<double>(q[(static_cast<size_t>(h) * S + s) * D + d]) *
              key(kvh, t, d);
        }
        logits[t] = eff[t] ? dot * scale : kMaskFill;
        mx = std::max(mx, logits[t]);
      }
      double sum = 0.0;
      for (int t = 0; t < T; ++t) sum += std::exp(logits[t] - mx);
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        if (any) {
          for (int t = 0; t < T; ++t) {
            acc += std::exp(logits[t] - mx) / sum * value(kvh, t, d);
          }
        }
        out[(static_cast<size_t>(s) * H + h) * D + d] =
            Round(c.type, static_cast<float>(acc));
      }
    }
  }
  return out;
}

void RunSparseSdpaTest(const SdpaConfig& c, int valid_rows = -1) {
  if (valid_rows < 0) valid_rows = c.seq_len;
  std::mt19937 rng(1234 + c.seq_len * 7 + c.kv_len);
  const int T = c.kv_len, D = c.head_dim, S = c.seq_len;
  const int max_blocks = T / c.ratio;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int B = c.batch;
  const std::vector<int> k_shape =
      c.k_ts_idx == 2 ? std::vector<int>{B, c.num_kv_heads, T, D}
                      : std::vector<int>{B, c.num_kv_heads, D, T};
  const std::vector<int> v_shape =
      c.v_ts_idx == 3 ? std::vector<int>{B, c.num_kv_heads, D, T}
                      : std::vector<int>{B, c.num_kv_heads, T, D};

  CompositeModel model;
  const int q_id = model.AddInput(c.type, {B, c.num_q_heads, S, D});
  const int k_id = model.AddInput(c.type, k_shape);
  const int v_id = model.AddInput(c.type, v_shape);
  const int topk_id = model.AddInput(kTfLiteInt32, {B, S, c.block_topk});
  // The mask has batch 1 and is broadcast over the batch.
  const int mask_id = model.AddInput(c.mask_type, {1, 1, S, T});
  const int out_id = model.AddOutput(c.type, {B, S, c.num_q_heads, D});
  model.Build("odml.sparse_sdpa_transposed", SdpaAttributes(c, scale));
  model.Delegate(c.num_threads);
  model.Allocate();

  const std::vector<float> q =
      RandomValues(rng, Product({B, c.num_q_heads, S, D}), c.type, -2.0f, 2.0f);
  const std::vector<float> k = RandomValues(rng, Product(k_shape), c.type);
  const std::vector<float> v = RandomValues(rng, Product(v_shape), c.type);
  const std::vector<bool> vis = CausalMask(S, T, c.pos0, valid_rows);
  // Select a random subset of the complete blocks per row, padded with -1.
  // Some rows also select an incomplete block, which must be a no-op.
  std::vector<int32_t> topk(static_cast<size_t>(B) * S * c.block_topk, -1);
  for (int row = 0; row < B * S; ++row) {
    const int s = row % S;
    std::vector<int> complete;
    for (int j = 0; j < max_blocks; ++j) {
      bool all = true;
      for (int r = 0; r < c.ratio; ++r) {
        all = all && vis[static_cast<size_t>(s) * T + j * c.ratio + r];
      }
      if (all) complete.push_back(j);
    }
    std::shuffle(complete.begin(), complete.end(), rng);
    const int n = std::min<int>(complete.size(), c.block_topk - 2);
    for (int i = 0; i < n; ++i) {
      topk[static_cast<size_t>(row) * c.block_topk + i] = complete[i];
    }
    if (s % 2 == 1) {
      topk[static_cast<size_t>(row) * c.block_topk + n] = max_blocks - 1;
    }
  }
  std::vector<float> mask_f(vis.size());
  for (size_t i = 0; i < vis.size(); ++i) mask_f[i] = vis[i] ? 0.0f : kMaskFill;

  model.SetFloat(q_id, q);
  model.SetFloat(k_id, k);
  model.SetFloat(v_id, v);
  model.SetInt32(topk_id, topk);
  if (c.mask_type == kTfLiteBool) {
    model.SetBool(mask_id, vis);
  } else {
    model.SetFloat(mask_id, mask_f);
  }
  model.Invoke();

  // Batch entries are independent; run the reference on each slice.
  auto slice = [](const auto& v, int b, int num_batches) {
    const size_t n = v.size() / num_batches;
    return std::vector<typename std::decay_t<decltype(v)>::value_type>(
        v.begin() + b * n, v.begin() + (b + 1) * n);
  };
  std::vector<float> expected;
  for (int b = 0; b < B; ++b) {
    const std::vector<float> out_b =
        ReferenceSparseSdpa(c, scale, slice(q, b, B), slice(k, b, B),
                            slice(v, b, B), slice(topk, b, B), vis);
    expected.insert(expected.end(), out_b.begin(), out_b.end());
  }
  const std::vector<float> actual = model.GetFloat(out_id);
  ASSERT_EQ(actual.size(), expected.size());
  const float tol = c.type == kTfLiteFloat16 ? 2e-3f : 1e-5f;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i],
                tol * std::max(1.0f, std::fabs(expected[i])))
        << "at " << i;
  }
}

TEST(QsaSparseSdpaTest, DecodeFp16) {
  SdpaConfig c;
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, DecodeBatched) {
  SdpaConfig c;
  c.batch = 3;
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, PrefillBatched) {
  SdpaConfig c;
  c.batch = 2;
  c.seq_len = 5;
  c.pos0 = 70;
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, DecodeFp16SingleThread) {
  SdpaConfig c;
  c.num_threads = 1;
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, DecodeLongContextSplitsRows) {
  SdpaConfig c;
  c.kv_len = 4096 + 2;
  c.head_dim = 64;
  c.block_topk = 128;
  c.pos0 = 3000;
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, DecodeTailTokensVisible) {
  SdpaConfig c;
  c.pos0 = 129;  // Last two tokens are beyond the last complete block.
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, PrefillFp16WithPadding) {
  SdpaConfig c;
  c.seq_len = 9;
  c.pos0 = 80;
  RunSparseSdpaTest(c, /*valid_rows=*/6);
}

TEST(QsaSparseSdpaTest, PrefillFp32FloatMaskSeqMajorLayouts) {
  SdpaConfig c;
  c.type = kTfLiteFloat32;
  c.mask_type = kTfLiteFloat32;
  c.seq_len = 5;
  c.pos0 = 60;
  c.k_ts_idx = 3;
  c.v_ts_idx = 2;
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, EmptyRowProducesZeros) {
  SdpaConfig c;
  c.seq_len = 2;
  c.pos0 = -1;  // Row 0 sees nothing.
  RunSparseSdpaTest(c);
}

TEST(QsaSparseSdpaTest, Fp16MaskAndMaxBlocksSmallerThanTopk) {
  SdpaConfig c;
  c.mask_type = kTfLiteFloat16;
  c.kv_len = 16;
  c.block_topk = 8;
  c.pos0 = 13;
  RunSparseSdpaTest(c);
}

// ---------------------------------------------------------------------------
// odml.qsa_indexer_update
// ---------------------------------------------------------------------------

struct IndexerConfig {
  int batch = 1;
  TfLiteType type = kTfLiteFloat16;
  TfLiteType mask_type = kTfLiteBool;
  int seq_len = 1;
  int num_heads = 4;
  int dim = 32;
  int ratio = 4;
  int max_blocks = 16;
  int kv_len = 64;
  int rotary_dim = 16;
  int block_topk = 4;
  int pos0 = 23;
  int valid_rows = -1;
  float eps = 1e-6f;
  bool alias_state = false;
  int num_threads = 4;
};

std::vector<uint8_t> IndexerAttributes(const IndexerConfig& c, float scale) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("compress_ratio", c.ratio);
    fbb.Int("block_topk", c.block_topk);
    fbb.Float("rms_norm_eps", c.eps);
    fbb.Float("scale", scale);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// `_rms_norm` followed by `_apply_rope_single` on one pooled block.
std::vector<float> ReferencePool(const IndexerConfig& c,
                                 const std::vector<float>& pooled_f32,
                                 const std::vector<float>& weight,
                                 const float* cos, const float* sin) {
  const TfLiteType t = c.type;
  const int D = c.dim;
  std::vector<float> x(D);
  float ms = 0.0f;
  for (int d = 0; d < D; ++d) {
    x[d] = Round(t, pooled_f32[d]);
    ms += x[d] * x[d];
  }
  const float inv = 1.0f / std::sqrt(ms / D + c.eps);
  for (int d = 0; d < D; ++d) {
    x[d] = Round(t, Round(t, x[d] * inv) * Round(t, weight[d]));
  }
  std::vector<float> out = x;
  const int half = c.rotary_dim / 2;
  for (int i = 0; i < c.rotary_dim; ++i) {
    const float rot = i < half ? -x[i + half] : x[i - half];
    out[i] = Round(t, x[i] * cos[i] + rot * sin[i]);
  }
  return out;
}

struct IndexerResult {
  std::vector<int32_t> topk;
  std::vector<float> ring;
  std::vector<float> pool;
};

// Brute-force transcription of `apply_qsa_indexer_update` (batch 1).
IndexerResult ReferenceIndexer(
    const IndexerConfig& c, float scale, const std::vector<float>& q,
    const std::vector<float>& k_new, const std::vector<float>& ring,
    const std::vector<float>& pool, const std::vector<float>& weight,
    const std::vector<float>& cos, const std::vector<float>& sin,
    const std::vector<bool>& vis) {
  const int S = c.seq_len, D = c.dim, R = c.ratio, MB = c.max_blocks;
  const int T = c.kv_len, pos0 = c.pos0;
  auto visible = [&](int s, int t) {
    return t >= 0 && t < T && vis[static_cast<size_t>(s) * T + t];
  };
  IndexerResult res;
  res.pool = pool;
  res.ring = ring;
  auto pool_block = [&](const std::vector<float>& raw, int blk) {
    std::vector<float> mean(D, 0.0f);
    for (int d = 0; d < D; ++d) {
      float sum = 0.0f;
      for (int r = 0; r < R; ++r) sum += raw[r * D + d];
      mean[d] = sum / R;
    }
    return ReferencePool(c, mean, weight, &cos[blk * c.rotary_dim],
                         &sin[blk * c.rotary_dim]);
  };
  if (S == 1) {
    const bool tok_vis = visible(0, pos0);
    const int slot = ((pos0 % R) + R) % R;
    if (tok_vis) {
      for (int d = 0; d < D; ++d) res.ring[slot * D + d] = k_new[d];
    }
    const int blk = pos0 >= 0 ? pos0 / R : -1;
    if (tok_vis && (pos0 + 1) % R == 0 && blk >= 0 && blk < MB) {
      const std::vector<float> rot = pool_block(res.ring, blk);
      for (int d = 0; d < D; ++d) res.pool[blk * D + d] = rot[d];
    }
  } else {
    std::vector<bool> tok_valid(S);
    int valid_len = 0;
    for (int i = 0; i < S; ++i) {
      tok_valid[i] = visible(i, pos0 + i);
      if (tok_valid[i]) valid_len = i + 1;
    }
    const int total = MB * R;
    std::vector<float> combined(static_cast<size_t>(total) * D, 0.0f);
    const int start_block = pos0 / R;
    for (int t = 0; t < total; ++t) {
      const int i = t - pos0;
      const bool has_new = i >= 0 && i < S && tok_valid[i];
      const bool ring_grid = t / R == start_block && t % R < pos0 % R;
      for (int d = 0; d < D; ++d) {
        if (has_new) {
          combined[static_cast<size_t>(t) * D + d] = k_new[i * D + d];
        } else if (ring_grid) {
          combined[static_cast<size_t>(t) * D + d] = ring[(t % R) * D + d];
        }
      }
    }
    const int end_pos = pos0 + valid_len;
    for (int blk = 0; blk < MB; ++blk) {
      if (blk >= start_block && blk < end_pos / R) {
        std::vector<float> raw(combined.begin() + blk * R * D,
                               combined.begin() + (blk + 1) * R * D);
        const std::vector<float> rot = pool_block(raw, blk);
        std::copy(rot.begin(), rot.end(), res.pool.begin() + blk * D);
      }
    }
    const int tail = end_pos / R;
    for (int i = 0; i < R * D; ++i) {
      res.ring[i] =
          tail < MB ? combined[static_cast<size_t>(tail) * R * D + i] : 0.0f;
    }
  }
  res.topk.assign(static_cast<size_t>(S) * c.block_topk, -1);
  for (int s = 0; s < S; ++s) {
    std::vector<std::pair<float, int>> scored;
    std::vector<bool> complete(MB);
    for (int j = 0; j < MB; ++j) {
      bool all = true;
      for (int r = 0; r < R; ++r) all = all && visible(s, j * R + r);
      complete[j] = all;
      float score = 0.0f;
      for (int h = 0; h < c.num_heads; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot += q[(static_cast<size_t>(s) * c.num_heads + h) * D + d] *
                 res.pool[j * D + d];
        }
        score += std::max(dot, 0.0f);
      }
      scored.push_back({all ? score * scale : kMaskFill, j});
    }
    std::stable_sort(
        scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    const int k_blocks = std::min(c.block_topk, MB);
    for (int i = 0; i < k_blocks; ++i) {
      const int idx = scored[i].second;
      res.topk[static_cast<size_t>(s) * c.block_topk + i] =
          complete[idx] ? idx : -1;
    }
  }
  return res;
}

void RunIndexerTest(const IndexerConfig& c) {
  const int valid_rows = c.valid_rows < 0 ? c.seq_len : c.valid_rows;
  std::mt19937 rng(4321 + c.seq_len * 13 + c.pos0);
  const int S = c.seq_len, D = c.dim, R = c.ratio, MB = c.max_blocks;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  const int B = c.batch;
  CompositeModel model;
  const int q_id = model.AddInput(c.type, {B, S, c.num_heads, D});
  const int k_id = model.AddInput(c.type, {B, S, D});
  const int ring_id = model.AddInput(c.type, {B, R, D});
  const int pool_id = model.AddInput(c.type, {B, MB, D});
  const int weight_id = model.AddInput(kTfLiteFloat32, {D});
  const int cos_id = model.AddInput(kTfLiteFloat32, {1, MB, c.rotary_dim});
  const int sin_id = model.AddInput(kTfLiteFloat32, {1, MB, c.rotary_dim});
  const int pos_id = model.AddInput(kTfLiteInt32, {1});
  const int mask_id = model.AddInput(c.mask_type, {1, 1, S, c.kv_len});
  const int topk_id = model.AddOutput(kTfLiteInt32, {B, S, c.block_topk});
  const int new_ring_id = model.AddOutput(c.type, {B, R, D});
  const int new_pool_id = model.AddOutput(c.type, {B, MB, D});
  model.Build("odml.qsa_indexer_update", IndexerAttributes(c, scale));
  model.Delegate(c.num_threads);
  if (c.alias_state) {
    model.Alias(ring_id, new_ring_id);
    model.Alias(pool_id, new_pool_id);
  }
  model.Allocate();

  const std::vector<float> q =
      RandomValues(rng, Product({B, S, c.num_heads, D}), c.type);
  const std::vector<float> k_new =
      RandomValues(rng, Product({B, S, D}), c.type);
  const std::vector<float> ring = RandomValues(rng, Product({B, R, D}), c.type);
  const std::vector<float> pool =
      RandomValues(rng, Product({B, MB, D}), c.type);
  const std::vector<float> weight =
      RandomValues(rng, D, kTfLiteFloat32, 0.5f, 1.5f);
  std::vector<float> cos(static_cast<size_t>(MB) * c.rotary_dim);
  std::vector<float> sin(cos.size());
  for (int b = 0; b < MB; ++b) {
    for (int i = 0; i < c.rotary_dim; ++i) {
      const float angle =
          b * R *
          std::pow(100.0f, -(i % (c.rotary_dim / 2)) / (c.rotary_dim / 2.0f));
      cos[b * c.rotary_dim + i] = std::cos(angle);
      sin[b * c.rotary_dim + i] = std::sin(angle);
    }
  }
  const std::vector<bool> vis = CausalMask(S, c.kv_len, c.pos0, valid_rows);
  std::vector<float> mask_f(vis.size());
  for (size_t i = 0; i < vis.size(); ++i) mask_f[i] = vis[i] ? 0.0f : kMaskFill;

  model.SetFloat(q_id, q);
  model.SetFloat(k_id, k_new);
  model.SetFloat(ring_id, ring);
  model.SetFloat(pool_id, pool);
  model.SetFloat(weight_id, weight);
  model.SetFloat(cos_id, cos);
  model.SetFloat(sin_id, sin);
  model.SetInt32(pos_id, {c.pos0});
  if (c.mask_type == kTfLiteBool) {
    model.SetBool(mask_id, vis);
  } else {
    model.SetFloat(mask_id, mask_f);
  }
  model.Invoke();

  // Batch entries are independent (the mask has batch 1); run the reference
  // on each slice.
  auto slice = [B](const std::vector<float>& v, int b) {
    const size_t n = v.size() / B;
    return std::vector<float>(v.begin() + b * n, v.begin() + (b + 1) * n);
  };
  IndexerResult expected;
  for (int b = 0; b < B; ++b) {
    const IndexerResult r =
        ReferenceIndexer(c, scale, slice(q, b), slice(k_new, b), slice(ring, b),
                         slice(pool, b), weight, cos, sin, vis);
    expected.topk.insert(expected.topk.end(), r.topk.begin(), r.topk.end());
    expected.ring.insert(expected.ring.end(), r.ring.begin(), r.ring.end());
    expected.pool.insert(expected.pool.end(), r.pool.begin(), r.pool.end());
  }
  EXPECT_EQ(model.GetInt32(topk_id), expected.topk);
  const std::vector<float> actual_ring = model.GetFloat(new_ring_id);
  const std::vector<float> actual_pool = model.GetFloat(new_pool_id);
  ASSERT_EQ(actual_ring.size(), expected.ring.size());
  for (size_t i = 0; i < expected.ring.size(); ++i) {
    EXPECT_EQ(actual_ring[i], expected.ring[i]) << "ring at " << i;
  }
  ASSERT_EQ(actual_pool.size(), expected.pool.size());
  for (size_t i = 0; i < expected.pool.size(); ++i) {
    EXPECT_NEAR(actual_pool[i], expected.pool[i],
                4e-3f * std::max(1.0f, std::fabs(expected.pool[i])))
        << "pool at " << i;
  }
}

TEST(QsaIndexerUpdateTest, DecodeCompletesBlock) {
  IndexerConfig c;
  c.pos0 = 23;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, DecodeBatched) {
  IndexerConfig c;
  c.batch = 2;
  c.pos0 = 27;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, PrefillBatched) {
  IndexerConfig c;
  c.batch = 2;
  c.seq_len = 6;
  c.pos0 = 9;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, DecodeMidBlock) {
  IndexerConfig c;
  c.pos0 = 21;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, DecodeFewerCompleteBlocksThanTopk) {
  IndexerConfig c;
  c.pos0 = 11;
  c.block_topk = 8;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, DecodeInPlaceState) {
  IndexerConfig c;
  c.pos0 = 31;
  c.alias_state = true;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, DecodeLongContextSplitsScoring) {
  IndexerConfig c;
  c.max_blocks = 1024;
  c.kv_len = 4096;
  c.block_topk = 64;
  c.pos0 = 3999;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, PrefillWithRingAndPadding) {
  IndexerConfig c;
  c.seq_len = 7;
  c.pos0 = 10;
  c.valid_rows = 5;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, PrefillInPlaceState) {
  IndexerConfig c;
  c.seq_len = 13;
  c.pos0 = 6;
  c.block_topk = 2;
  c.alias_state = true;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, PrefillFp32FloatMask) {
  IndexerConfig c;
  c.type = kTfLiteFloat32;
  c.mask_type = kTfLiteFloat32;
  c.seq_len = 8;
  c.pos0 = 0;
  RunIndexerTest(c);
}

TEST(QsaIndexerUpdateTest, PrefillFp16MaskAndMaxBlocksSmallerThanTopk) {
  IndexerConfig c;
  c.mask_type = kTfLiteFloat16;
  c.max_blocks = 4;
  c.kv_len = 16;
  c.block_topk = 8;
  c.seq_len = 5;
  c.pos0 = 8;
  RunIndexerTest(c);
}

// Nodes with invalid attributes must be left to the default implementation.
TEST(QsaDelegateKernelTest, InvalidAttributesAreNotDelegated) {
  for (const int k_ts_idx : {1, 4}) {
    SdpaConfig c;
    c.k_ts_idx = k_ts_idx;
    CompositeModel model;
    model.AddInput(c.type, {1, c.num_q_heads, 1, c.head_dim});
    model.AddInput(c.type, {1, c.num_kv_heads, c.kv_len, c.head_dim});
    model.AddInput(c.type, {1, c.num_kv_heads, c.head_dim, c.kv_len});
    model.AddInput(kTfLiteInt32, {1, 1, c.block_topk});
    model.AddInput(kTfLiteBool, {1, 1, 1, c.kv_len});
    model.AddOutput(c.type, {1, 1, c.num_q_heads, c.head_dim});
    model.Build("odml.sparse_sdpa_transposed", SdpaAttributes(c, 0.25f));
    EXPECT_FALSE(model.TryDelegate(/*num_threads=*/1));
  }
}

TEST(QsaDelegateKernelTest, MissingAttributesAreNotDelegated) {
  IndexerConfig c;
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Int("compress_ratio", c.ratio); });
  fbb.Finish();
  CompositeModel model;
  model.AddInput(c.type, {1, 1, c.num_heads, c.dim});
  model.AddInput(c.type, {1, 1, c.dim});
  model.AddInput(c.type, {1, c.ratio, c.dim});
  model.AddInput(c.type, {1, c.max_blocks, c.dim});
  model.AddInput(kTfLiteFloat32, {c.dim});
  model.AddInput(kTfLiteFloat32, {1, c.max_blocks, c.rotary_dim});
  model.AddInput(kTfLiteFloat32, {1, c.max_blocks, c.rotary_dim});
  model.AddInput(kTfLiteInt32, {1});
  model.AddInput(kTfLiteBool, {1, 1, 1, c.kv_len});
  model.AddOutput(kTfLiteInt32, {1, 1, c.block_topk});
  model.AddOutput(c.type, {1, c.ratio, c.dim});
  model.AddOutput(c.type, {1, c.max_blocks, c.dim});
  model.Build("odml.qsa_indexer_update", fbb.GetBuffer());
  EXPECT_FALSE(model.TryDelegate(/*num_threads=*/1));
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
