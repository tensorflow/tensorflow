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
#include "tensorflow/lite/delegates/xnnpack/sdpa_delegate_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {
namespace {

constexpr float kMaskFill = -10000.0f;

TfLiteStatus StubSdpaPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size >= 1);
  TF_LITE_ENSURE(context, node->outputs->size >= 1);
  const TfLiteTensor* q = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, out, TfLiteIntArrayCopy(q->dims)));
  if (node->outputs->size == 3) {
    const TfLiteTensor* kc = &context->tensors[node->inputs->data[1]];
    const TfLiteTensor* vc = &context->tensors[node->inputs->data[2]];
    TfLiteTensor* kc_out = &context->tensors[node->outputs->data[1]];
    TfLiteTensor* vc_out = &context->tensors[node->outputs->data[2]];
    TF_LITE_ENSURE_STATUS(
        context->ResizeTensor(context, kc_out, TfLiteIntArrayCopy(kc->dims)));
    TF_LITE_ENSURE_STATUS(
        context->ResizeTensor(context, vc_out, TfLiteIntArrayCopy(vc->dims)));
  }
  return kTfLiteOk;
}

TfLiteStatus StubSdpaEval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteError;
}

TfLiteRegistration* Register_StubSdpa() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/StubSdpaPrepare,
      /*.invoke=*/StubSdpaEval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/BuiltinOperator_CUSTOM,
      /*.custom_name=*/"stub_sdpa",
      /*.version=*/1,
  };
  return &reg;
}

using UniqueXnnpackDelegate =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

UniqueXnnpackDelegate CreateXnnpackDelegate(int num_threads) {
  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  options.num_threads = num_threads;
  return UniqueXnnpackDelegate(TfLiteXNNPackDelegateCreate(&options),
                               TfLiteXNNPackDelegateDelete);
}

// =============================================================================
// Part 1: odml.sdpa_transposed and odml.scaled_dot_product_attention Tests
// =============================================================================

struct SdpaTestConfig {
  int b = 1;
  int n_q = 2;
  int n_kv = 2;
  int t = 4;
  int s = 8;
  int s_active = 5;
  int d = 16;
  float scale = 0.0f;  // 0 -> 1 / sqrt(d)
  float softcap = 0.0f;
  bool is_seq_major = false;
  bool bool_mask = false;
  bool pack_query = false;
  bool use_composite = true;
  int num_threads = 1;
};

class StandardSdpaModel : public MultiOpModel {
 public:
  explicit StandardSdpaModel(const SdpaTestConfig& c) : c_(c) {
    const int g = c.n_q / c.n_kv;
    const std::vector<int> q_shape =
        c.is_seq_major
            ? std::vector<int>{c.b, c.t, c.n_q, c.d}
            : (c.pack_query ? std::vector<int>{c.b, c.n_kv, g * c.t, c.d}
                            : std::vector<int>{c.b, c.n_q, c.t, c.d});
    const std::vector<int> k_shape =
        c.is_seq_major ? std::vector<int>{c.b, c.s, c.n_kv, c.d}
                       : std::vector<int>{c.b, c.n_kv, c.s, c.d};
    const std::vector<int> v_shape =
        c.is_seq_major ? std::vector<int>{c.b, c.s, c.n_kv, c.d}
                       : std::vector<int>{c.b, c.n_kv, c.d, c.s};
    const std::vector<int> mask_shape = {c.b, 1, c.t, c.s};
    const std::vector<int> param_shape = {1};

    q_ = AddInput({TensorType_FLOAT32, q_shape});
    k_ = AddInput({TensorType_FLOAT32, k_shape});
    v_ = AddInput({TensorType_FLOAT32, v_shape});
    mask_ = AddInput(
        {c.bool_mask ? TensorType_BOOL : TensorType_FLOAT32, mask_shape});
    param_ = AddInput({TensorType_INT32, param_shape});
    out_ = AddOutput({TensorType_FLOAT32, q_shape});

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      if (c.scale > 0.0f) {
        fbb.Float("scale", c.scale);
      }
      if (c.softcap > 0.0f) {
        fbb.Float("softcap", c.softcap);
      }
    });
    fbb.Finish();
    const std::vector<uint8_t>& attrs = fbb.GetBuffer();
    const char* op_name = c.is_seq_major ? "odml.scaled_dot_product_attention"
                                         : "odml.sdpa_transposed";

    if (c.use_composite) {
      SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                   BuiltinOptions2_StableHLOCompositeOptions,
                   CreateStableHLOCompositeOptionsDirect(
                       builder_, op_name,
                       /*decomposition_subgraph_index=*/0, &attrs)
                       .Union());
    } else {
      SetCustomOp(op_name, attrs, Register_StubSdpa);
    }

    BuildInterpreter({q_shape, k_shape, v_shape, mask_shape, param_shape},
                     /*num_threads=*/c.num_threads,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false);

    SetDelegate(CreateXnnpackDelegate(c.num_threads));
    AllocateAndDelegate(true);
  }

  bool FullyDelegated() const {
    return interpreter_->execution_plan().size() == 1 &&
           interpreter_
                   ->node_and_registration(interpreter_->execution_plan()[0])
                   ->first.delegate != nullptr;
  }

  void SetBoolMask(const std::vector<bool>& mask) {
    bool* data = interpreter_->typed_tensor<bool>(mask_);
    for (size_t i = 0; i < mask.size(); ++i) {
      data[i] = mask[i];
    }
  }

  TfLiteStatus ResizeInputTensor(int id, const std::vector<int>& dims) {
    return interpreter_->ResizeInputTensor(id, dims);
  }
  TfLiteStatus AllocateTensors() { return interpreter_->AllocateTensors(); }

  int q() const { return q_; }
  int k() const { return k_; }
  int v() const { return v_; }
  int mask() const { return mask_; }
  int param() const { return param_; }
  int out() const { return out_; }

 private:
  SdpaTestConfig c_;
  int q_, k_, v_, mask_, param_, out_;
};

std::vector<float> ReferenceStandardSdpa(const SdpaTestConfig& c,
                                         const std::vector<float>& q_data,
                                         const std::vector<float>& k_data,
                                         const std::vector<float>& v_data,
                                         const std::vector<float>& mask_f32,
                                         const std::vector<bool>& mask_bool) {
  const int b = c.b;
  const int n_q = c.n_q;
  const int n_kv = c.n_kv;
  const int g = n_q / n_kv;
  const int t = c.t;
  const int s = c.s;
  const int s_active = (c.s_active > 0 && c.s_active <= s) ? c.s_active : s;
  const int d = c.d;
  const float scale =
      (c.scale > 0.0f)
          ? c.scale
          : (c.is_seq_major ? (1.0f / std::sqrt(static_cast<float>(d))) : 1.0f);

  std::vector<float> out(b * n_q * t * d, 0.0f);
  for (int bi = 0; bi < b; ++bi) {
    for (int hi = 0; hi < n_kv; ++hi) {
      for (int gi = 0; gi < g; ++gi) {
        const int n_idx = hi * g + gi;
        for (int ti = 0; ti < t; ++ti) {
          std::vector<float> logits(s_active, 0.0f);
          float max_logit = -std::numeric_limits<float>::infinity();
          for (int si = 0; si < s_active; ++si) {
            float dot = 0.0f;
            for (int di = 0; di < d; ++di) {
              const int q_idx = c.is_seq_major
                                    ? (((bi * t + ti) * n_q + n_idx) * d + di)
                                    : (((bi * n_q + n_idx) * t + ti) * d + di);
              const int k_idx = c.is_seq_major
                                    ? (((bi * s + si) * n_kv + hi) * d + di)
                                    : (((bi * n_kv + hi) * s + si) * d + di);
              dot += (q_data[q_idx] * scale) * k_data[k_idx];
            }
            if (c.softcap > 0.0f) {
              dot = c.softcap * std::tanh(dot / c.softcap);
            }
            const int m_idx = (bi * t + ti) * s + si;
            if (c.bool_mask) {
              dot = mask_bool[m_idx] ? dot : kMaskFill;
            } else {
              dot += mask_f32[m_idx];
            }
            logits[si] = dot;
            max_logit = std::max(max_logit, dot);
          }

          float denom = 0.0f;
          for (int si = 0; si < s_active; ++si) {
            logits[si] = std::exp(logits[si] - max_logit);
            denom += logits[si];
          }
          for (int di = 0; di < d; ++di) {
            float acc = 0.0f;
            for (int si = 0; si < s_active; ++si) {
              const float p = logits[si] / denom;
              const int v_idx = c.is_seq_major
                                    ? (((bi * s + si) * n_kv + hi) * d + di)
                                    : (((bi * n_kv + hi) * d + di) * s + si);
              acc += p * v_data[v_idx];
            }
            const int out_idx = c.is_seq_major
                                    ? (((bi * t + ti) * n_q + n_idx) * d + di)
                                    : (((bi * n_q + n_idx) * t + ti) * d + di);
            out[out_idx] = acc;
          }
        }
      }
    }
  }
  return out;
}

void RunStandardSdpaAndCheck(const SdpaTestConfig& c) {
  std::vector<float> q_data(c.b * c.n_q * c.t * c.d);
  std::vector<float> k_data(c.b * c.n_kv * c.s * c.d);
  std::vector<float> v_data(c.b * c.n_kv * c.d * c.s);
  std::vector<float> mask_f32(c.b * c.t * c.s, 0.0f);
  std::vector<bool> mask_bool(c.b * c.t * c.s, false);

  for (size_t i = 0; i < q_data.size(); ++i) {
    q_data[i] = 0.1f * std::sin(static_cast<float>(i + 1));
  }
  for (size_t i = 0; i < k_data.size(); ++i) {
    k_data[i] = 0.2f * std::cos(static_cast<float>(i + 2));
  }
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = 0.3f * std::sin(static_cast<float>(i + 3));
  }
  for (int i = 0; i < c.b * c.t; ++i) {
    for (int j = 0; j < c.s; ++j) {
      const bool valid = (j < c.s_active) && (j <= i + (c.s_active - c.t));
      mask_bool[i * c.s + j] = valid;
      mask_f32[i * c.s + j] = valid ? 0.0f : -1e9f;
    }
  }

  StandardSdpaModel model(c);
  ASSERT_TRUE(model.FullyDelegated());
  model.PopulateTensor(model.q(), q_data);
  model.PopulateTensor(model.k(), k_data);
  model.PopulateTensor(model.v(), v_data);
  if (c.bool_mask) {
    model.SetBoolMask(mask_bool);
  } else {
    model.PopulateTensor(model.mask(), mask_f32);
  }
  model.PopulateTensor(model.param(), {c.s_active});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  const std::vector<float> actual = model.ExtractVector<float>(model.out());
  const std::vector<float> expected =
      ReferenceStandardSdpa(c, q_data, k_data, v_data, mask_f32, mask_bool);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FALSE(std::isnan(actual[i]));
    EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "mismatch at " << i;
  }
}

TEST(SdpaDelegateKernelTest, SdpaTransposedCorrectness) {
  SdpaTestConfig c;
  c.scale = 1.0f / std::sqrt(static_cast<float>(c.d));
  RunStandardSdpaAndCheck(c);
}

TEST(SdpaDelegateKernelTest, ScaledDotProductAttentionSeqMajorCorrectness) {
  SdpaTestConfig c;
  c.is_seq_major = true;
  c.scale = 1.0f / std::sqrt(static_cast<float>(c.d));
  RunStandardSdpaAndCheck(c);
}

TEST(SdpaDelegateKernelTest, SdpaTransposedLargeShapeBounds) {
  SdpaTestConfig c;
  c.b = 1;
  c.t = 1;
  c.s = 4096;
  c.s_active = 41;
  c.d = 64;
  c.n_q = 32;
  c.n_kv = 32;
  c.scale = 1.0f / std::sqrt(static_cast<float>(c.d));

  std::vector<float> q_data(c.b * c.n_q * c.t * c.d);
  std::vector<float> k_data(c.b * c.n_kv * c.s * c.d);
  std::vector<float> v_data(c.b * c.n_kv * c.d * c.s);
  std::vector<float> mask_f32(c.b * c.t * c.s);
  std::vector<bool> mask_bool;

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = std::sin(i);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = std::sin(i * 2);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = std::sin(i * 3);
  for (int j = 0; j < c.s; ++j) {
    mask_f32[j] =
        (j < c.s_active) ? 0.0f : -std::numeric_limits<float>::infinity();
  }

  StandardSdpaModel model(c);
  ASSERT_TRUE(model.FullyDelegated());
  model.PopulateTensor(model.q(), q_data);
  model.PopulateTensor(model.k(), k_data);
  model.PopulateTensor(model.v(), v_data);
  model.PopulateTensor(model.mask(), mask_f32);
  model.PopulateTensor(model.param(), {c.s_active});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  const std::vector<float> actual = model.ExtractVector<float>(model.out());
  const std::vector<float> expected =
      ReferenceStandardSdpa(c, q_data, k_data, v_data, mask_f32, mask_bool);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FALSE(std::isnan(actual[i]));
    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
  }
}

TEST(SdpaDelegateKernelTest, SdpaTransposedResizeAfterDelegation) {
  SdpaTestConfig c;
  c.scale = 1.0f / std::sqrt(static_cast<float>(c.d));
  StandardSdpaModel model(c);
  ASSERT_TRUE(model.FullyDelegated());

  const int t_new = 2;
  ASSERT_EQ(model.ResizeInputTensor(model.q(), {c.b, c.n_q, t_new, c.d}),
            kTfLiteOk);
  ASSERT_EQ(model.ResizeInputTensor(model.mask(), {c.b, 1, t_new, c.s}),
            kTfLiteOk);
  ASSERT_EQ(model.AllocateTensors(), kTfLiteOk);

  c.t = t_new;
  std::vector<float> q_data(c.b * c.n_q * c.t * c.d);
  std::vector<float> k_data(c.b * c.n_kv * c.s * c.d);
  std::vector<float> v_data(c.b * c.n_kv * c.d * c.s);
  std::vector<float> mask_f32(c.b * c.t * c.s);
  std::vector<bool> mask_bool;

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = 0.15f * (i % 7);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = 0.2f * (i % 10);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = 0.3f * (i % 10);
  for (int i = 0; i < c.b * c.t; ++i) {
    for (int j = 0; j < c.s; ++j) {
      mask_f32[i * c.s + j] = (j < c.s_active) ? 0.0f : -1e9f;
    }
  }

  model.PopulateTensor(model.q(), q_data);
  model.PopulateTensor(model.k(), k_data);
  model.PopulateTensor(model.v(), v_data);
  model.PopulateTensor(model.mask(), mask_f32);
  model.PopulateTensor(model.param(), {c.s_active});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  const std::vector<float> actual = model.ExtractVector<float>(model.out());
  const std::vector<float> expected =
      ReferenceStandardSdpa(c, q_data, k_data, v_data, mask_f32, mask_bool);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
  }
}

TEST(SdpaDelegateKernelTest, SdpaTransposedGqaDecodeAndPrefill) {
  for (int t : {1, 4, 12, 20}) {
    for (int n_kv : {1, 2}) {
      for (bool bool_mask : {false, true}) {
        for (bool pack_query : {false, true}) {
          SdpaTestConfig c;
          c.b = 1;
          c.n_q = 4;
          c.n_kv = n_kv;
          c.t = t;
          c.s = 24;
          c.s_active = 19;
          c.d = 16;
          c.scale = 1.0f / std::sqrt(static_cast<float>(c.d));
          c.softcap = 30.0f;
          c.bool_mask = bool_mask;
          c.pack_query = pack_query;
          RunStandardSdpaAndCheck(c);
        }
      }
    }
  }
}

// =============================================================================
// Part 2: odml.fused_sdpa_cache_update Tests
// =============================================================================

struct Config {
  int h = 1;
  int g = 4;
  int t = 1;
  int d = 16;
  int w = 16;
  float softcap = 0.0f;
  bool update_cache = true;
  bool param_int64 = false;
  bool use_custom_op = false;
  int num_threads = 1;
  float tol = 1e-4f;
};

class FusedSdpaCacheUpdateModel : public MultiOpModel {
 public:
  explicit FusedSdpaCacheUpdateModel(const Config& c) : c_(c) {
    const std::vector<int> q_shape = {1, c.h, c.g * c.t, c.d};
    const std::vector<int> kc_shape = {1, c.h, c.w, c.d};
    const std::vector<int> vc_shape = {1, c.h, c.d, c.w};
    const std::vector<int> kn_shape = {1, c.h, c.t, c.d};
    const std::vector<int> vn_shape = {1, c.h, c.d, c.t};
    const std::vector<int> mask_shape = {1, 1, c.t, c.w + c.t};
    const std::vector<int> param_shape = {1, 1, 1, 7};

    q_ = AddInput({TensorType_FLOAT32, q_shape});
    kc_ = AddInput({TensorType_FLOAT32, kc_shape});
    vc_ = AddInput({TensorType_FLOAT32, vc_shape});
    kn_ = AddInput({TensorType_FLOAT32, kn_shape});
    vn_ = AddInput({TensorType_FLOAT32, vn_shape});
    mask_ = AddInput({TensorType_BOOL, mask_shape});
    param_ = AddInput(
        {c.param_int64 ? TensorType_INT64 : TensorType_INT32, param_shape});

    out_ = AddOutput({TensorType_FLOAT32, q_shape});
    std::vector<int> outputs = {out_};
    if (c.update_cache) {
      kc_out_ = AddOutput({TensorType_FLOAT32, kc_shape});
      vc_out_ = AddOutput({TensorType_FLOAT32, vc_shape});
      outputs.push_back(kc_out_);
      outputs.push_back(vc_out_);
    }

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      if (c.softcap > 0.0f) {
        fbb.Float("softcap", c.softcap);
      }
    });
    fbb.Finish();
    const std::vector<uint8_t>& attrs = fbb.GetBuffer();
    if (c.use_custom_op) {
      SetCustomOp("odml.fused_sdpa_cache_update", attrs, Register_StubSdpa);
    } else {
      SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                   BuiltinOptions2_StableHLOCompositeOptions,
                   CreateStableHLOCompositeOptionsDirect(
                       builder_, "odml.fused_sdpa_cache_update",
                       /*decomposition_subgraph_index=*/0, &attrs)
                       .Union());
    }

    BuildInterpreter({q_shape, kc_shape, vc_shape, kn_shape, vn_shape,
                      mask_shape, param_shape},
                     /*num_threads=*/c.num_threads,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false);

    SetDelegate(CreateXnnpackDelegate(c.num_threads));
    AllocateAndDelegate(true);
  }

  bool FullyDelegated() const {
    return interpreter_->execution_plan().size() == 1 &&
           interpreter_
                   ->node_and_registration(interpreter_->execution_plan()[0])
                   ->first.delegate != nullptr;
  }

  void SetMask(const std::vector<bool>& mask) {
    bool* data = interpreter_->typed_tensor<bool>(mask_);
    for (size_t i = 0; i < mask.size(); ++i) {
      data[i] = mask[i];
    }
  }

  TfLiteTensor* tensor(int id) { return interpreter_->tensor(id); }

  int q() const { return q_; }
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

struct Step {
  int start;
  int valid;
  std::vector<float> q, kn, vn;
  std::vector<bool> mask;
};

int SlotPosition(int s, int start, int w) {
  if (start <= 0) return -1;
  if (start <= w) return s < start ? s : -1;
  const int base = start - w;
  const int p = base + ((s - base % w) + w) % w;
  return p;
}

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

void Reference(const Config& c, const Step& st, std::vector<float>& kc,
               std::vector<float>& vc, std::vector<float>* out) {
  const int cols = c.w + c.t;
  out->assign(c.h * c.g * c.t * c.d, 0.0f);
  for (int h = 0; h < c.h; ++h) {
    for (int g = 0; g < c.g; ++g) {
      for (int i = 0; i < c.t; ++i) {
        const int row = g * c.t + i;
        const float* q = &st.q[(h * c.g * c.t + row) * c.d];
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
  Config c{.h = 2, .g = 2, .t = 8, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/0, /*valid=*/8, 3)});
}

TEST(FusedSdpaCacheUpdateTest, PrefillWrapsRing) {
  Config c{.h = 2, .g = 2, .t = 8, .d = 16, .w = 16};
  RunAndCheck(c, {MakeStep(c, /*start=*/12, /*valid=*/8, 4)});
}

TEST(FusedSdpaCacheUpdateTest, PrefillLongerThanWindow) {
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
  Config c{.h = 1, .g = 4, .t = 8, .d = 16, .w = 32};
  RunAndCheck(c, {MakeStep(c, 0, 8, 13), MakeStep(c, 7, 8, 14, /*written=*/8),
                  MakeStep(c, 14, 8, 15, /*written=*/15),
                  MakeStep(c, 21, 8, 16, /*written=*/22)});
}

TEST(FusedSdpaCacheUpdateTest, FullyMaskedPaddingRow) {
  Config c{.h = 1, .g = 4, .t = 8, .d = 16, .w = 32};
  for (int start : {0, 5, 40}) {
    Step st = MakeStep(c, start, 8, start + 20);
    for (int j = 0; j < c.w + c.t; ++j) st.mask[7 * (c.w + c.t) + j] = false;
    RunAndCheck(c, {st});
  }
}

TEST(FusedSdpaCacheUpdateTest, FullyMaskedPaddingRowPrefill) {
  Config c{.h = 2, .g = 4, .t = 16, .d = 16, .w = 32};
  for (int start : {0, 5, 30, 31, 40}) {
    Step st = MakeStep(c, start, 16, start + 30);
    for (int j = 0; j < c.w + c.t; ++j) st.mask[15 * (c.w + c.t) + j] = false;
    RunAndCheck(c, {st});
  }
}

TEST(FusedSdpaCacheUpdateTest, PrefillAroundRingBoundary) {
  Config c{.h = 1, .g = 4, .t = 16, .d = 16, .w = 32};
  for (int start : {1, 30, 31, 32, 33}) {
    RunAndCheck(c, {MakeStep(c, start, 16, start + 40)});
  }
}

TEST(FusedSdpaCacheUpdateTest, OverlappingChunksPrefill) {
  Config c{.h = 1, .g = 4, .t = 16, .d = 16, .w = 64};
  RunAndCheck(c, {MakeStep(c, 0, 16, 17), MakeStep(c, 15, 16, 18, 16),
                  MakeStep(c, 30, 16, 19, 31), MakeStep(c, 45, 16, 20, 46),
                  MakeStep(c, 60, 16, 21, 61)});
}

// Verifies LiteRT-LM's CPU in-place KV cache aliasing contract (kc_out->data
// == kc->data and vc_out->data == vc->data).
TEST(FusedSdpaCacheUpdateTest, InPlaceBufferAliasing) {
  Config c{.h = 2, .g = 4, .t = 16, .d = 16, .w = 32};
  FusedSdpaCacheUpdateModel model(c);
  ASSERT_TRUE(model.FullyDelegated());
  std::vector<float> kc, vc;
  InitCaches(c, kc, vc);
  std::vector<float> ref_kc = kc, ref_vc = vc;

  // Alias kc_out to kc and vc_out to vc in-place.
  TfLiteTensor* kc_t = model.tensor(model.kc());
  TfLiteTensor* vc_t = model.tensor(model.vc());
  TfLiteTensor* kc_out_t = model.tensor(model.kc_out());
  TfLiteTensor* vc_out_t = model.tensor(model.vc_out());
  char* orig_kc_out = kc_out_t->data.raw;
  char* orig_vc_out = vc_out_t->data.raw;
  kc_out_t->data.raw = kc_t->data.raw;
  vc_out_t->data.raw = vc_t->data.raw;

  model.PopulateTensor(model.kc(), kc);
  model.PopulateTensor(model.vc(), vc);

  for (int step_idx = 0; step_idx < 4; ++step_idx) {
    const int start = step_idx * 15;
    Step st = MakeStep(c, start, 16, step_idx + 50,
                       /*written=*/step_idx == 0 ? 0 : start + 1);
    model.PopulateTensor(model.q(), st.q);
    model.PopulateTensor(model.kn(), st.kn);
    model.PopulateTensor(model.vn(), st.vn);
    model.SetMask(st.mask);
    model.PopulateTensor<int32_t>(
        model.param(),
        {st.start, st.start + st.valid, st.start + st.valid, 0, 0, 0, 0});
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    std::vector<float> ref_out;
    Reference(c, st, ref_kc, ref_vc, &ref_out);
    ExpectNear(model.ExtractVector<float>(model.out()), ref_out, c.tol,
               "attention (in-place)");
    ExpectNear(model.ExtractVector<float>(model.kc_out()), ref_kc, 0.0f,
               "key cache (in-place)");
    ExpectNear(model.ExtractVector<float>(model.vc_out()), ref_vc, 0.0f,
               "value cache (in-place)");
  }

  kc_out_t->data.raw = orig_kc_out;
  vc_out_t->data.raw = orig_vc_out;
}

TEST(FusedSdpaCacheUpdateTest, RealisticChunkedPrefill128) {
  Config c{.h = 1, .g = 4, .t = 128, .d = 256, .w = 512, .num_threads = 4};
  std::vector<Step> steps;
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

class CompositeInliningModel : public SingleOpModel {
 public:
  CompositeInliningModel() { interpreter_ = std::make_unique<Interpreter>(); }
  Interpreter& interpreter() { return *interpreter_; }
  using SingleOpModel::AddSubgraphs;
};

// Verifies that when SetShloCompositeInlining(true) is enabled on a model
// containing both an unsupported STABLEHLO_COMPOSITE op (wrapped around an ADD
// op) and a supported odml.sdpa_transposed STABLEHLO_COMPOSITE op,
// DelegatePrepare inlines the unsupported composite while preserving and
// delegating odml.sdpa_transposed.
TEST(SdpaDelegateKernelTest, ShloCompositeInliningWithUnsupportedComposite) {
  SdpaTestConfig c{.b = 1, .n_q = 2, .n_kv = 2, .t = 4, .s = 8, .d = 16};
  const std::vector<int> q_shape = {c.b, c.n_q, c.t, c.d};
  const std::vector<int> k_shape = {c.b, c.n_kv, c.s, c.d};
  const std::vector<int> v_shape = {c.b, c.n_kv, c.d, c.s};
  const std::vector<int> mask_shape = {c.b, 1, c.t, c.s};
  const std::vector<int> param_shape = {1};

  CompositeInliningModel model;
  Interpreter& interpreter = model.interpreter();
  InterpreterOptions options;
  options.SetShloCompositeInlining(true);
  interpreter.ApplyOptions(&options);
  model.AddSubgraphs(2);

  // Subgraph 1: decomposition of "odml.unsupported_pre_q" -> ADD(in0, in1).
  Subgraph* decomp_add = interpreter.subgraph(1);
  ASSERT_EQ(decomp_add->AddTensors(3), kTfLiteOk);
  for (int t_idx : {0, 1, 2}) {
    ASSERT_EQ(decomp_add->SetTensorParametersReadWrite(
                  t_idx, kTfLiteFloat32, "", q_shape, TfLiteQuantization{}),
              kTfLiteOk);
  }
  ASSERT_EQ(decomp_add->SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(decomp_add->SetOutputs({2}), kTfLiteOk);
  auto* add_params =
      static_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;
  add_params->pot_scale_int16 = false;
  TfLiteRegistration add_reg = *ops::builtin::Register_ADD();
  add_reg.builtin_code = kTfLiteBuiltinAdd;
  int add_node_idx = -1;
  ASSERT_EQ(
      decomp_add->AddNodeWithParameters({0, 1}, {2}, {}, nullptr, 0, add_params,
                                        &add_reg, &add_node_idx),
      kTfLiteOk);

  // Subgraph 2: dummy decomposition of "odml.sdpa_transposed" (must NOT be
  // inlined; if inlined, the output would not match SDPA).
  Subgraph* decomp_sdpa = interpreter.subgraph(2);
  ASSERT_EQ(decomp_sdpa->AddTensors(6), kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", q_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "", k_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetTensorParametersReadWrite(
                2, kTfLiteFloat32, "", v_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetTensorParametersReadWrite(
                3, kTfLiteFloat32, "", mask_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetTensorParametersReadWrite(
                4, kTfLiteInt32, "", param_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetTensorParametersReadWrite(
                5, kTfLiteFloat32, "", q_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetInputs({0, 1, 2, 3, 4}), kTfLiteOk);
  ASSERT_EQ(decomp_sdpa->SetOutputs({5}), kTfLiteOk);

  // Subgraph 0 (primary):
  //   q_sum = STABLEHLO_COMPOSITE["odml.unsupported_pre_q"](q_a, q_b)
  //   out   = STABLEHLO_COMPOSITE["odml.sdpa_transposed"](q_sum, k, v, mask, p)
  Subgraph& primary = interpreter.primary_subgraph();
  ASSERT_EQ(primary.AddTensors(8), kTfLiteOk);
  for (int t_idx : {0, 1, 6, 7}) {
    ASSERT_EQ(primary.SetTensorParametersReadWrite(
                  t_idx, kTfLiteFloat32, "", q_shape, TfLiteQuantization{}),
              kTfLiteOk);
  }
  ASSERT_EQ(primary.SetTensorParametersReadWrite(2, kTfLiteFloat32, "", k_shape,
                                                 TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetTensorParametersReadWrite(3, kTfLiteFloat32, "", v_shape,
                                                 TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetTensorParametersReadWrite(
                4, kTfLiteFloat32, "", mask_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetTensorParametersReadWrite(
                5, kTfLiteInt32, "", param_shape, TfLiteQuantization{}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetInputs({0, 1, 2, 3, 4, 5}), kTfLiteOk);
  ASSERT_EQ(primary.SetOutputs({7}), kTfLiteOk);

  TfLiteRegistration composite_reg =
      *ops::builtin::Register_STABLEHLO_COMPOSITE();
  composite_reg.builtin_code = kTfLiteBuiltinStablehloComposite;

  auto* comp_add_params = static_cast<TfLiteStablehloCompositeParams*>(
      calloc(1, sizeof(TfLiteStablehloCompositeParams)));
  comp_add_params->name = "odml.unsupported_pre_q";
  comp_add_params->subgraph_index = 1;
  comp_add_params->version = 1;
  int node0 = -1;
  ASSERT_EQ(
      primary.AddNodeWithParameters(
          {0, 1}, {6}, {}, reinterpret_cast<const char*>(comp_add_params),
          sizeof(*comp_add_params), comp_add_params, &composite_reg, &node0),
      kTfLiteOk);

  auto* comp_sdpa_params = static_cast<TfLiteStablehloCompositeParams*>(
      calloc(1, sizeof(TfLiteStablehloCompositeParams)));
  comp_sdpa_params->name = "odml.sdpa_transposed";
  comp_sdpa_params->subgraph_index = 2;
  comp_sdpa_params->version = 1;
  int node1 = -1;
  ASSERT_EQ(
      primary.AddNodeWithParameters(
          {6, 2, 3, 4, 5}, {7}, {},
          reinterpret_cast<const char*>(comp_sdpa_params),
          sizeof(*comp_sdpa_params), comp_sdpa_params, &composite_reg, &node1),
      kTfLiteOk);

  EXPECT_FALSE(SdpaDelegateKernel::IsSdpaNode(
      &composite_reg, &primary.node_and_registration(node0)->first));
  EXPECT_TRUE(SdpaDelegateKernel::IsSdpaNode(
      &composite_reg, &primary.node_and_registration(node1)->first));

  auto delegate = CreateXnnpackDelegate(/*num_threads=*/1);
  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Both the inlined ADD and the preserved odml.sdpa_transposed composite
  // should be delegated to XNNPACK (ADD in the XNNPACK subgraph partition, and
  // odml.sdpa_transposed in its standalone SdpaDelegateKernel partition).
  ASSERT_EQ(interpreter.execution_plan().size(), 2u);
  for (int node_idx : interpreter.execution_plan()) {
    const auto* pair = interpreter.node_and_registration(node_idx);
    ASSERT_NE(pair->first.delegate, nullptr);
  }

  const int q_elems = c.b * c.n_q * c.t * c.d;
  const int k_elems = c.b * c.n_kv * c.s * c.d;
  const int mask_elems = c.b * c.t * c.s;
  std::vector<float> q_a(q_elems), q_b(q_elems), q_sum(q_elems);
  std::vector<float> k_data(k_elems), v_data(k_elems), mask_f32(mask_elems);
  for (int i = 0; i < q_elems; ++i) {
    q_a[i] = std::sin(0.11f * (i + 1));
    q_b[i] = std::cos(0.17f * (i + 2));
    q_sum[i] = q_a[i] + q_b[i];
  }
  for (int i = 0; i < k_elems; ++i) {
    k_data[i] = std::sin(0.23f * (i + 3));
    v_data[i] = std::cos(0.31f * (i + 4));
  }
  for (int i = 0; i < mask_elems; ++i) {
    mask_f32[i] = (i % 5 == 0) ? -10000.0f : 0.0f;
  }

  std::memcpy(interpreter.typed_tensor<float>(0), q_a.data(),
              q_a.size() * sizeof(float));
  std::memcpy(interpreter.typed_tensor<float>(1), q_b.data(),
              q_b.size() * sizeof(float));
  std::memcpy(interpreter.typed_tensor<float>(2), k_data.data(),
              k_data.size() * sizeof(float));
  std::memcpy(interpreter.typed_tensor<float>(3), v_data.data(),
              v_data.size() * sizeof(float));
  std::memcpy(interpreter.typed_tensor<float>(4), mask_f32.data(),
              mask_f32.size() * sizeof(float));
  interpreter.typed_tensor<int32_t>(5)[0] = 5;

  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  std::vector<float> ref = ReferenceStandardSdpa(c, q_sum, k_data, v_data,
                                                 mask_f32, /*mask_bool=*/{});
  const float* actual = interpreter.typed_tensor<float>(7);
  for (int i = 0; i < q_elems; ++i) {
    EXPECT_NEAR(actual[i], ref[i], 2e-5f) << "at index " << i;
  }
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
