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
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ynnpack {
namespace {

struct Config {
  int h = 1;   // KV heads.
  int g = 1;   // Query heads per KV head.
  int t = 1;   // New tokens per step (static).
  int d = 16;  // Head dim.
  int w = 16;  // Ring buffer size.
  float softcap = 0.0f;
  bool update_cache = true;
  bool param_int64 = false;
  bool use_custom_op = false;
  int num_threads = 1;
  bool fast_math = false;
  float tol = 1e-4f;
};

TfLiteStatus DummyFusedPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}
TfLiteStatus DummyFusedEval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}
TfLiteRegistration* Register_DummyFusedSdpaCacheUpdate() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/DummyFusedPrepare,
      /*.invoke=*/DummyFusedEval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/BuiltinOperator_CUSTOM,
      /*.custom_name=*/"odml.fused_sdpa_cache_update",
      /*.version=*/1,
  };
  return &reg;
}

class FusedSdpaCacheUpdateModel : public SingleOpModel {
 public:
  explicit FusedSdpaCacheUpdateModel(const Config& c) : c_(c) {
    q_ = AddInput({TensorType_FLOAT32, {1, c.h, c.g * c.t, c.d}});
    kc_ = AddInput({TensorType_FLOAT32, {1, c.h, c.w, c.d}});
    vc_ = AddInput({TensorType_FLOAT32, {1, c.h, c.d, c.w}});
    kn_ = AddInput({TensorType_FLOAT32, {1, c.h, c.t, c.d}});
    vn_ = AddInput({TensorType_FLOAT32, {1, c.h, c.d, c.t}});
    mask_ = AddInput({TensorType_BOOL, {1, 1, c.t, c.w + c.t}});
    param_ = AddInput(
        {c.param_int64 ? TensorType_INT64 : TensorType_INT32, {1, 1, 1, 7}});
    out_ = AddOutput({TensorType_FLOAT32, {1, c.h, c.g * c.t, c.d}});
    if (c.update_cache) {
      kc_out_ = AddOutput({TensorType_FLOAT32, {1, c.h, c.w, c.d}});
      vc_out_ = AddOutput({TensorType_FLOAT32, {1, c.h, c.d, c.w}});
    }

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("cache_size", c.w);
      fbb.Bool("update_cache", c.update_cache);
      if (c.softcap > 0.0f) fbb.Float("softcap", c.softcap);
    });
    fbb.Finish();
    std::vector<uint8_t> attrs = fbb.GetBuffer();
    if (c.use_custom_op) {
      SetCustomOp("odml.fused_sdpa_cache_update", attrs,
                  Register_DummyFusedSdpaCacheUpdate);
    } else {
      flatbuffers::Offset<StableHLOCompositeOptions> options =
          CreateStableHLOCompositeOptionsDirect(
              builder_, "odml.fused_sdpa_cache_update",
              /*decomposition_subgraph_index=*/1, &attrs);
      SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                   BuiltinOptions2_StableHLOCompositeOptions, options.Union());
    }
    BuildInterpreter({GetShape(q_), GetShape(kc_), GetShape(vc_), GetShape(kn_),
                      GetShape(vn_), GetShape(mask_), GetShape(param_)},
                     -1, false, false, /*allocate_and_delegate=*/false);

    TfLiteYNNPackDelegateOptions delegate_options =
        TfLiteYNNPackDelegateOptionsDefault();
    delegate_options.num_threads = c.num_threads;
    delegate_options.fast_math = c.fast_math;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteYNNPackDelegateCreate(&delegate_options),
        TfLiteYNNPackDelegateDelete));
    ApplyDelegate();
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors\n");
    }
  }

  // Returns true if the composite was claimed by the delegate.
  bool FullyDelegated() const {
    return interpreter_->execution_plan().size() == 1 &&
           interpreter_
                   ->node_and_registration(interpreter_->execution_plan()[0])
                   ->first.delegate != nullptr;
  }

  int q() const { return q_; }
  void SetMask(const std::vector<bool>& mask) {
    bool* data = interpreter_->typed_tensor<bool>(mask_);
    for (size_t i = 0; i < mask.size(); ++i) data[i] = mask[i];
  }
  int kc() const { return kc_; }
  int vc() const { return vc_; }
  int kn() const { return kn_; }
  int vn() const { return vn_; }
  int mask() const { return mask_; }
  int param() const { return param_; }
  int out() const { return out_; }
  int kc_out() const { return kc_out_; }
  int vc_out() const { return vc_out_; }

 private:
  Config c_;
  int q_, kc_, vc_, kn_, vn_, mask_, param_, out_;
  int kc_out_ = -1;
  int vc_out_ = -1;
};

constexpr float kMaskFill = -10000.0f;

// One step of state for the reference model and the delegate.
struct Step {
  int start;  // Absolute position of the first new token.
  int valid;  // Number of valid new tokens (<= t).
  std::vector<float> q, kn, vn;
  std::vector<bool> mask;
};

// Absolute position held by ring slot `s` before a step starting at `start`,
// or -1 if the slot has never been written.
int SlotPosition(int s, int start, int w) {
  if (start <= 0) return -1;
  if (start <= w) return s < start ? s : -1;
  // Positions [start - w, start) occupy all slots.
  const int base = start - w;
  const int p = base + ((s - base % w) + w) % w;
  return p;
}

// Builds the combined causal + sliding-window mask the exporter produces:
// columns [0, W) are ring slots, [W, W + T) are the new tokens. `written` is
// the number of positions already in the ring; LiteRT-LM chunked prefill
// re-feeds the previous chunk's last token, so it can exceed `start`.
std::vector<bool> BuildMask(const Config& c, int start, int valid,
                            int written) {
  std::vector<bool> mask(c.t * (c.w + c.t), false);
  for (int i = 0; i < c.t; ++i) {
    const int qpos = start + i;
    for (int s = 0; s < c.w; ++s) {
      const int p = SlotPosition(s, written, c.w);
      if (p >= 0 && p <= qpos && qpos - p < c.w) {
        mask[i * (c.w + c.t) + s] = true;
      }
    }
    for (int j = 0; j < c.t; ++j) {
      const int kpos = start + j;
      if (j < valid && kpos <= qpos && qpos - kpos < c.w) {
        mask[i * (c.w + c.t) + c.w + j] = true;
      }
    }
  }
  return mask;
}

Step MakeStep(const Config& c, int start, int valid, int seed,
              int written = -1) {
  Step st;
  st.start = start;
  st.valid = valid;
  st.q.resize(c.h * c.g * c.t * c.d);
  st.kn.resize(c.h * c.t * c.d);
  st.vn.resize(c.h * c.d * c.t);
  for (size_t i = 0; i < st.q.size(); ++i) {
    st.q[i] = 0.5f * std::sin(0.37f * i + seed);
  }
  for (size_t i = 0; i < st.kn.size(); ++i) {
    st.kn[i] = 0.5f * std::cos(0.23f * i + 2 * seed);
  }
  for (size_t i = 0; i < st.vn.size(); ++i) {
    st.vn[i] = std::sin(0.11f * i + 3 * seed);
  }
  st.mask = BuildMask(c, start, valid, written < 0 ? start : written);
  return st;
}

// Reference attention over [cache | new] followed by the ring write.
void Reference(const Config& c, const Step& st, std::vector<float>& kc,
               std::vector<float>& vc, std::vector<float>* out) {
  const int cols = c.w + c.t;
  out->assign(c.h * c.g * c.t * c.d, 0.0f);
  for (int h = 0; h < c.h; ++h) {
    for (int g = 0; g < c.g; ++g) {
      for (int i = 0; i < c.t; ++i) {
        const int row = g * c.t + i;
        const float* q = &st.q[(h * c.g * c.t + row) * c.d];
        // Masked logits are replaced by -10000 (like the exporter's
        // torch.where), so a row without any valid column averages all
        // columns uniformly.
        std::vector<float> logits(cols, kMaskFill);
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < cols; ++j) {
          if (!st.mask[i * cols + j]) {
            m = std::max(m, kMaskFill);
            continue;
          }
          float dot = 0.0f;
          for (int k = 0; k < c.d; ++k) {
            const float kv = j < c.w ? kc[(h * c.w + j) * c.d + k]
                                     : st.kn[(h * c.t + j - c.w) * c.d + k];
            dot += q[k] * kv;
          }
          if (c.softcap > 0.0f) dot = c.softcap * std::tanh(dot / c.softcap);
          logits[j] = dot;
          m = std::max(m, dot);
        }
        float denom = 0.0f;
        for (int j = 0; j < cols; ++j) {
          logits[j] = std::exp(logits[j] - m);
          denom += logits[j];
        }
        float* o = &(*out)[(h * c.g * c.t + row) * c.d];
        for (int j = 0; j < cols; ++j) {
          const float p = logits[j] / denom;
          for (int k = 0; k < c.d; ++k) {
            const float vv = j < c.w ? vc[(h * c.d + k) * c.w + j]
                                     : st.vn[(h * c.d + k) * c.t + j - c.w];
            o[k] += p * vv;
          }
        }
      }
    }
  }
  if (!c.update_cache) return;
  for (int i = std::max(0, st.valid - c.w); i < st.valid; ++i) {
    const int slot = (st.start + i) % c.w;
    for (int h = 0; h < c.h; ++h) {
      for (int k = 0; k < c.d; ++k) {
        kc[(h * c.w + slot) * c.d + k] = st.kn[(h * c.t + i) * c.d + k];
        vc[(h * c.d + k) * c.w + slot] = st.vn[(h * c.d + k) * c.t + i];
      }
    }
  }
}

// Fills a cache consistent with `start` previous tokens.
void InitCaches(const Config& c, std::vector<float>& kc,
                std::vector<float>& vc) {
  kc.resize(c.h * c.w * c.d);
  vc.resize(c.h * c.d * c.w);
  for (size_t i = 0; i < kc.size(); ++i) kc[i] = 0.4f * std::sin(0.71f * i);
  for (size_t i = 0; i < vc.size(); ++i) vc[i] = std::cos(0.53f * i);
}

void ExpectNear(const std::vector<float>& actual,
                const std::vector<float>& expected, float tol,
                const char* what) {
  ASSERT_EQ(actual.size(), expected.size()) << what;
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_NEAR(actual[i], expected[i], tol) << what << " mismatch at " << i;
  }
}

// Runs `steps` sequentially, feeding the updated caches back in, and checks
// the attention output and the caches after every step.
void RunAndCheck(const Config& c, const std::vector<Step>& steps) {
  FusedSdpaCacheUpdateModel model(c);
  ASSERT_TRUE(model.FullyDelegated());
  std::vector<float> kc, vc;
  InitCaches(c, kc, vc);
  std::vector<float> ref_kc = kc, ref_vc = vc;
  for (const Step& st : steps) {
    model.PopulateTensor(model.q(), st.q);
    model.PopulateTensor(model.kc(), kc);
    model.PopulateTensor(model.vc(), vc);
    model.PopulateTensor(model.kn(), st.kn);
    model.PopulateTensor(model.vn(), st.vn);
    model.SetMask(st.mask);
    if (c.param_int64) {
      model.PopulateTensor<int64_t>(
          model.param(),
          {st.start, st.start + st.valid, st.start + st.valid, 0, 0, 0, 0});
    } else {
      model.PopulateTensor<int32_t>(
          model.param(),
          {st.start, st.start + st.valid, st.start + st.valid, 0, 0, 0, 0});
    }
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    std::vector<float> ref_out;
    Reference(c, st, ref_kc, ref_vc, &ref_out);
    ExpectNear(model.ExtractVector<float>(model.out()), ref_out, c.tol,
               "attention");
    if (c.update_cache) {
      kc = model.ExtractVector<float>(model.kc_out());
      vc = model.ExtractVector<float>(model.vc_out());
      ExpectNear(kc, ref_kc, 0.0f, "key cache");
      ExpectNear(vc, ref_vc, 0.0f, "value cache");
    }
  }
}

TEST(FusedSdpaCacheUpdateTest, DecodePartialCache) {
  Config c{.h = 1, .g = 4, .t = 1, .d = 32, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/5, /*valid=*/1, 1)});
}

TEST(FusedSdpaCacheUpdateTest, DecodeWrappedCache) {
  Config c{.h = 2, .g = 2, .t = 1, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/40, /*valid=*/1, 2)});
}

TEST(FusedSdpaCacheUpdateTest, DecodeSequence) {
  Config c{.h = 1, .g = 4, .t = 1, .d = 16, .w = 8};
  std::vector<Step> steps;
  for (int p = 5; p < 20; ++p) steps.push_back(MakeStep(c, p, 1, p));
  RunAndCheck(c, steps);
}

TEST(FusedSdpaCacheUpdateTest, FirstPrefill) {
  // start == 0: no valid past; the past extent is clamped to one masked slot.
  Config c{.h = 2, .g = 2, .t = 8, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/0, /*valid=*/8, 3)});
}

TEST(FusedSdpaCacheUpdateTest, PrefillWrapsRing) {
  Config c{.h = 2, .g = 2, .t = 8, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/12, /*valid=*/8, 4)});
}

TEST(FusedSdpaCacheUpdateTest, PrefillLongerThanWindow) {
  // G * T = 48 > 32 exercises the prefill (P.V) branch; T > W keeps only the
  // most recent W tokens.
  Config c{.h = 1, .g = 2, .t = 24, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/20, /*valid=*/24, 5)});
}

TEST(FusedSdpaCacheUpdateTest, PrefillWithPadding) {
  Config c{.h = 1, .g = 2, .t = 8, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/14, /*valid=*/5, 6)});
}

TEST(FusedSdpaCacheUpdateTest, ChunkedPrefillThenDecode) {
  Config c{.h = 1, .g = 4, .t = 12, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, 0, 12, 7), MakeStep(c, 12, 12, 8),
                  MakeStep(c, 24, 12, 9), MakeStep(c, 36, 3, 10)});
}

TEST(FusedSdpaCacheUpdateTest, Softcap) {
  Config c{.h = 1, .g = 2, .t = 4, .d = 16, .w = 16, .softcap = 0.5f};
  RunAndCheck(c, {MakeStep(c, /*start=*/9, /*valid=*/4, 11)});
}

TEST(FusedSdpaCacheUpdateTest, ReadOnly) {
  // update_cache = false (shared-KV readers): attention only, one output.
  Config c{.h = 1, .g = 2, .t = 4, .d = 16, .w = 16, .update_cache = false};
  RunAndCheck(c, {MakeStep(c, /*start=*/30, /*valid=*/4, 12)});
}

TEST(FusedSdpaCacheUpdateTest, Int64ParamAndCustomOp) {
  Config c{.h = 1,
           .g = 4,
           .t = 12,
           .d = 16,
           .w = 16,
           .param_int64 = true,
           .use_custom_op = true};
  RunAndCheck(c, {MakeStep(c, /*start=*/4, /*valid=*/12, 22)});
}

TEST(FusedSdpaCacheUpdateTest, OverlappingChunks) {
  // LiteRT-LM re-feeds the last token of each prefill chunk: the next chunk
  // starts at end - 1 and the mask still attends to that token's ring slot.
  Config c{.h = 1, .g = 4, .t = 8, .d = 16, .w = 32};
  RunAndCheck(c, {MakeStep(c, 0, 8, 13), MakeStep(c, 7, 8, 14, /*written=*/8),
                  MakeStep(c, 14, 8, 15, /*written=*/15),
                  MakeStep(c, 21, 8, 16, /*written=*/22)});
}

TEST(FusedSdpaCacheUpdateTest, FullyMaskedPaddingRow) {
  // LiteRT-LM holds the last token of a prefill back as the pending token, so
  // the last row of a chunk can have no valid column at all. Such rows must
  // match the reference (uniform average over all W + T columns), not depend
  // on how many cache slots are read.
  Config c{.h = 1, .g = 4, .t = 8, .d = 16, .w = 32};
  for (int start : {0, 5, 40}) {
    Step st = MakeStep(c, start, 8, start + 20);
    for (int j = 0; j < c.w + c.t; ++j) st.mask[7 * (c.w + c.t) + j] = false;
    RunAndCheck(c, {st});
  }
}

TEST(FusedSdpaCacheUpdateTest, FullyMaskedPaddingRowPrefill) {
  // Same on the prefill path (G * T > 32), which reads only the filled ring
  // slots and corrects such rows for the skipped ones.
  Config c{.h = 2, .g = 4, .t = 16, .d = 16, .w = 32};
  for (int start : {0, 5, 30, 31, 40}) {
    Step st = MakeStep(c, start, 16, start + 30);
    for (int j = 0; j < c.w + c.t; ++j) st.mask[15 * (c.w + c.t) + j] = false;
    RunAndCheck(c, {st});
  }
}

TEST(FusedSdpaCacheUpdateTest, PrefillAroundRingBoundary) {
  // The filled-slot extent min(start + 1, W) at and around the wrap point.
  Config c{.h = 1, .g = 4, .t = 16, .d = 16, .w = 32};
  for (int start : {1, 30, 31, 32, 33}) {
    RunAndCheck(c, {MakeStep(c, start, 16, start + 40)});
  }
}

TEST(FusedSdpaCacheUpdateTest, OverlappingChunksPrefill) {
  // Re-fed tokens on the prefill path: slot `start` is already written and
  // attended, so it must be among the slots read.
  Config c{.h = 1, .g = 4, .t = 16, .d = 16, .w = 64};
  RunAndCheck(c, {MakeStep(c, 0, 16, 17), MakeStep(c, 15, 16, 18, 16),
                  MakeStep(c, 30, 16, 19, 31), MakeStep(c, 45, 16, 20, 46),
                  MakeStep(c, 60, 16, 21, 61)});
}

// Gemma3-1B SWA shapes: 1 KV head, 4 query heads, D = 256, W = 512.
TEST(FusedSdpaCacheUpdateTest, RealisticChunkedPrefill128) {
  Config c{.h = 1, .g = 4, .t = 128, .d = 256, .w = 512, .num_threads = 4};
  std::vector<Step> steps;
  // Chunks as issued by LiteRT-LM: [0, 128), [127, 255), [255, 383), ...
  steps.push_back(MakeStep(c, 0, 128, 0));
  for (int i = 1; i < 7; ++i) {
    const int start = i * 128 - 1;
    steps.push_back(MakeStep(c, start, 128, i, /*written=*/start + 1));
  }
  steps.push_back(MakeStep(c, 895, 77, 7, /*written=*/896));
  RunAndCheck(c, steps);
}

TEST(FusedSdpaCacheUpdateTest, RealisticPrefill1024LongerThanWindow) {
  Config c{.h = 1, .g = 4, .t = 1024, .d = 256, .w = 512, .num_threads = 4};
  RunAndCheck(c, {MakeStep(c, 0, 900, 1), MakeStep(c, 900, 1024, 2)});
}

TEST(FusedSdpaCacheUpdateTest, RealisticPrefill1024ThenPrefill128) {
  Config c1{.h = 1, .g = 4, .t = 1024, .d = 256, .w = 512, .num_threads = 4};
  RunAndCheck(c1, {MakeStep(c1, 0, 700, 3)});
  Config c2{.h = 1, .g = 4, .t = 128, .d = 256, .w = 512, .num_threads = 4};
  RunAndCheck(c2, {MakeStep(c2, 700, 80, 4)});
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
