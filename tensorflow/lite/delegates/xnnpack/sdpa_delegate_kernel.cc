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
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace xnnpack {
namespace {

constexpr char kSdpaTransposedOp[] = "odml.sdpa_transposed";
constexpr char kScaledDotProductAttentionOp[] =
    "odml.scaled_dot_product_attention";
constexpr char kFusedSdpaCacheUpdateOp[] = "odml.fused_sdpa_cache_update";
constexpr int kInvalidTensorId = -1;
constexpr uintptr_t kXnnpackWorkspaceAlignment = 128;

// Upper bound used for valid bool-mask positions in min(logits, mask_bound) so
// that valid logits pass through unchanged while masked positions are clamped
// to kMaskFillValue (-10000.0f).
constexpr float kMaskTrueBound = 1.0e30f;
constexpr float kMaskFillValue = -10000.0f;

// Query rows threshold (G * T <= kDecodeQueryRowThreshold) below which we use
// the rewritten decode BMMs:
//   Scores = (K @ Q^T)^T
//   Out    = (V @ P^T)^T
// so XNNPACK packs the tiny Q / P matrix instead of packing the full KV cache.
constexpr int kDecodeQueryRowThreshold = 16;

enum class SdpaOpKind {
  kUnknown = 0,
  kSdpaTransposed,
  kScaledDotProductAttention,
  kFusedSdpaCacheUpdate,
};

struct SdpaAttributes {
  SdpaOpKind kind = SdpaOpKind::kUnknown;
  bool scale_specified = false;
  float scale = 1.0f;
  float softcap = 0.0f;
  int k_ts_idx = 2;
  int v_ts_idx = 3;
};

struct SdpaScratchBuffers {
  std::vector<float> q_buf;
  std::vector<float> q_scaled;
  std::vector<float> k_buf;
  std::vector<float> v_buf;
  std::vector<float> mask_buf;
  std::vector<float> logits;
  std::vector<float> decode_tmp;
  std::vector<float> out_buf;
  std::vector<uint8_t> row_has_valid;
  std::vector<char> workspace;
};

class SdpaScratchPool {
 public:
  std::unique_ptr<SdpaScratchBuffers> Acquire() {
    std::lock_guard<std::mutex> lock(mutex_);  // NOLINT
    if (!free_list_.empty()) {
      std::unique_ptr<SdpaScratchBuffers> buf = std::move(free_list_.back());
      free_list_.pop_back();
      return buf;
    }
    return std::make_unique<SdpaScratchBuffers>();
  }

  void Release(std::unique_ptr<SdpaScratchBuffers> buf) {
    if (buf == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);  // NOLINT
    free_list_.push_back(std::move(buf));
  }

 private:
  std::mutex mutex_;  // NOLINT
  std::vector<std::unique_ptr<SdpaScratchBuffers>> free_list_;
};

class ScopedSdpaScratchBuffers {
 public:
  explicit ScopedSdpaScratchBuffers(SdpaScratchPool* pool)
      : pool_(pool), buf_(pool->Acquire()) {}

  ~ScopedSdpaScratchBuffers() { pool_->Release(std::move(buf_)); }

  ScopedSdpaScratchBuffers(const ScopedSdpaScratchBuffers&) = delete;
  ScopedSdpaScratchBuffers& operator=(const ScopedSdpaScratchBuffers&) = delete;

  SdpaScratchBuffers* get() const { return buf_.get(); }

 private:
  SdpaScratchPool* pool_;
  std::unique_ptr<SdpaScratchBuffers> buf_;
};

std::shared_ptr<SdpaScratchPool> GetOrCreateSdpaScratchPool(const void* key) {
  struct Registry {
    std::mutex mutex;  // NOLINT
    std::unordered_map<const void*, std::weak_ptr<SdpaScratchPool>> pools;
  };
  static Registry* const registry = new Registry();

  std::lock_guard<std::mutex> lock(registry->mutex);  // NOLINT
  auto it = registry->pools.find(key);
  if (it != registry->pools.end()) {
    if (std::shared_ptr<SdpaScratchPool> existing = it->second.lock()) {
      return existing;
    }
  }
  for (auto iter = registry->pools.begin(); iter != registry->pools.end();) {
    if (iter->second.expired()) {
      iter = registry->pools.erase(iter);
    } else {
      ++iter;
    }
  }
  auto created = std::make_shared<SdpaScratchPool>();
  registry->pools[key] = created;
  return created;
}

const char* GetOpName(const TfLiteRegistration* registration,
                      const TfLiteNode* node) {
  if (registration == nullptr) {
    return nullptr;
  }
  if (registration->builtin_code == kTfLiteBuiltinCustom &&
      registration->custom_name != nullptr) {
    return registration->custom_name;
  }
  if (registration->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node != nullptr && node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    return composite_params->name;
  }
  return nullptr;
}

SdpaOpKind ResolveOpKind(const TfLiteRegistration* registration,
                         const TfLiteNode* node) {
  const char* name = GetOpName(registration, node);
  if (name == nullptr) {
    return SdpaOpKind::kUnknown;
  }
  if (std::strcmp(name, kSdpaTransposedOp) == 0) {
    return SdpaOpKind::kSdpaTransposed;
  }
  if (std::strcmp(name, kScaledDotProductAttentionOp) == 0) {
    return SdpaOpKind::kScaledDotProductAttention;
  }
  if (std::strcmp(name, kFusedSdpaCacheUpdateOp) == 0) {
    return SdpaOpKind::kFusedSdpaCacheUpdate;
  }
  return SdpaOpKind::kUnknown;
}

flexbuffers::Map GetAttributeMap(const TfLiteRegistration* registration,
                                 const TfLiteNode* node) {
  if (registration != nullptr &&
      registration->builtin_code == kTfLiteBuiltinStablehloComposite &&
      node != nullptr && node->builtin_data != nullptr) {
    const auto* composite_params =
        static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
    if (composite_params->attributes != nullptr &&
        composite_params->attributes_size > 0) {
      return flexbuffers::GetRoot(composite_params->attributes,
                                  composite_params->attributes_size)
          .AsMap();
    }
  }
  if (node != nullptr && node->custom_initial_data != nullptr &&
      node->custom_initial_data_size > 0) {
    return flexbuffers::GetRoot(
               static_cast<const uint8_t*>(node->custom_initial_data),
               node->custom_initial_data_size)
        .AsMap();
  }
  return flexbuffers::Map::EmptyMap();
}

bool ReadSdpaAttributes(const TfLiteRegistration* registration,
                        const TfLiteNode* node, SdpaAttributes* attr) {
  attr->kind = ResolveOpKind(registration, node);
  if (attr->kind == SdpaOpKind::kUnknown) {
    return false;
  }
  const flexbuffers::Map map = GetAttributeMap(registration, node);
  if (!map["scale"].IsNull()) {
    attr->scale = map["scale"].AsFloat();
    attr->scale_specified = true;
  }
  if (!map["softcap"].IsNull()) {
    attr->softcap = map["softcap"].AsFloat();
  } else if (!map["logit_cap"].IsNull()) {
    attr->softcap = map["logit_cap"].AsFloat();
  }
  if (!map["k_ts_idx"].IsNull()) {
    attr->k_ts_idx = map["k_ts_idx"].AsInt32();
  }
  if (!map["v_ts_idx"].IsNull()) {
    attr->v_ts_idx = map["v_ts_idx"].AsInt32();
  }
  return true;
}

bool ReadParamElement(const TfLiteTensor& param, size_t index, int64_t* out) {
  if (param.data.raw == nullptr) {
    return false;
  }
  if (param.type == kTfLiteInt32) {
    if (param.bytes < (index + 1) * sizeof(int32_t)) {
      return false;
    }
    *out = GetTensorData<int32_t>(&param)[index];
    return true;
  }
  if (param.type == kTfLiteInt64) {
    if (param.bytes < (index + 1) * sizeof(int64_t)) {
      return false;
    }
    *out = GetTensorData<int64_t>(&param)[index];
    return true;
  }
  return false;
}

// Writes the valid prefix [0, valid) of kn [1, H, T, D] and vn [1, H, D, T]
// into ring buffers kc_out [1, H, W, D] and vc_out [1, H, D, W] starting at
// ring slot `start % W`. Supports in-place updates (kc == kc_out, vc == vc_out)
// as well as out-of-place updates.
void ApplyFusedCacheWrite(int64_t start, int64_t end, size_t h, size_t w,
                          size_t t, size_t d, const float* kc, const float* vc,
                          const float* kn, const float* vn, float* kc_out,
                          float* vc_out) {
  if (kc != kc_out) {
    std::memcpy(kc_out, kc, h * w * d * sizeof(float));
  }
  if (vc != vc_out) {
    std::memcpy(vc_out, vc, h * d * w * sizeof(float));
  }

  const int64_t valid =
      std::clamp<int64_t>(end - start, 0, static_cast<int64_t>(t));
  if (valid <= 0) {
    return;
  }

  // If valid > W, only the last W tokens survive in the ring buffer.
  const int64_t skip = std::max<int64_t>(0, valid - static_cast<int64_t>(w));
  const size_t count = static_cast<size_t>(valid - skip);
  const size_t src_off = static_cast<size_t>(skip);
  const int64_t w_i64 = static_cast<int64_t>(w);
  const size_t dst_off =
      static_cast<size_t>(((start + skip) % w_i64 + w_i64) % w_i64);
  const size_t first = std::min(count, w - dst_off);
  const size_t second = count - first;

  for (size_t hi = 0; hi < h; ++hi) {
    const float* kn_h = kn + (hi * t + src_off) * d;
    float* kc_h = kc_out + hi * w * d;
    std::memcpy(kc_h + dst_off * d, kn_h, first * d * sizeof(float));
    if (second > 0) {
      std::memcpy(kc_h, kn_h + first * d, second * d * sizeof(float));
    }

    for (size_t di = 0; di < d; ++di) {
      const float* vn_hd = vn + (hi * d + di) * t + src_off;
      float* vc_hd = vc_out + (hi * d + di) * w;
      std::memcpy(vc_hd + dst_off, vn_hd, first * sizeof(float));
      if (second > 0) {
        std::memcpy(vc_hd, vn_hd + first, second * sizeof(float));
      }
    }
  }
}

}  // namespace

class SdpaDelegateKernel::Impl {
 public:
  using XnnOperatorPtr =
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>;

  static bool IsSdpaNode(const TfLiteRegistration* registration,
                         const TfLiteNode* node) {
    return ResolveOpKind(registration, node) != SdpaOpKind::kUnknown;
  }

  static TfLiteStatus IsSupported(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLiteRegistration* registration,
                                  int node_index) {
    if (node == nullptr || node->inputs == nullptr ||
        node->outputs == nullptr) {
      return kTfLiteError;
    }
    SdpaAttributes attr;
    if (!ReadSdpaAttributes(registration, node, &attr)) {
      return kTfLiteError;
    }
    if (attr.kind == SdpaOpKind::kFusedSdpaCacheUpdate) {
      return IsFusedSdpaCacheUpdateSupported(context, node, node_index);
    }
    return IsStandardSdpaSupported(context, node, attr, node_index);
  }

  static std::unique_ptr<Impl> Create(TfLiteContext* context,
                                      const TfLiteDelegateParams* params,
                                      pthreadpool_t threadpool) {
    if (params == nullptr || params->nodes_to_replace == nullptr ||
        params->nodes_to_replace->size != 1) {
      return nullptr;
    }
    const int node_index = params->nodes_to_replace->data[0];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      return nullptr;
    }
    if (node == nullptr ||
        IsSupported(context, node, registration, node_index) != kTfLiteOk) {
      return nullptr;
    }

    SdpaAttributes attr;
    if (!ReadSdpaAttributes(registration, node, &attr)) {
      return nullptr;
    }

    XnnOperatorPtr bmm1_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr bmm2_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr softmax_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr transpose_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr binary_add_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr binary_min_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr binary_mul_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr binary_div_op(nullptr, &xnn_delete_operator);
    XnnOperatorPtr unary_tanh_op(nullptr, &xnn_delete_operator);

    xnn_operator_t raw_op = nullptr;
    if (xnn_create_batch_matrix_multiply_nc_f32(
            /*flags=*/XNN_FLAG_TRANSPOSE_B, &raw_op) != xnn_status_success) {
      return nullptr;
    }
    bmm1_op.reset(raw_op);

    raw_op = nullptr;
    if (xnn_create_batch_matrix_multiply_nc_f32(
            /*flags=*/XNN_FLAG_TRANSPOSE_B, &raw_op) != xnn_status_success) {
      return nullptr;
    }
    bmm2_op.reset(raw_op);

    raw_op = nullptr;
    if (xnn_create_softmax_nc_f32(/*flags=*/0, &raw_op) != xnn_status_success) {
      return nullptr;
    }
    softmax_op.reset(raw_op);

    raw_op = nullptr;
    if (xnn_create_transpose_nd_x32(/*flags=*/0, &raw_op) !=
        xnn_status_success) {
      return nullptr;
    }
    transpose_op.reset(raw_op);

    raw_op = nullptr;
    if (xnn_create_binary_elementwise_nd(
            xnn_binary_add, xnn_datatype_fp32, /*a_quantization=*/nullptr,
            /*b_quantization=*/nullptr, /*output_quantization=*/nullptr,
            /*flags=*/0, &raw_op) != xnn_status_success) {
      return nullptr;
    }
    binary_add_op.reset(raw_op);

    raw_op = nullptr;
    if (xnn_create_binary_elementwise_nd(
            xnn_binary_minimum, xnn_datatype_fp32, /*a_quantization=*/nullptr,
            /*b_quantization=*/nullptr, /*output_quantization=*/nullptr,
            /*flags=*/0, &raw_op) != xnn_status_success) {
      return nullptr;
    }
    binary_min_op.reset(raw_op);

    raw_op = nullptr;
    if (xnn_create_binary_elementwise_nd(
            xnn_binary_multiply, xnn_datatype_fp32, /*a_quantization=*/nullptr,
            /*b_quantization=*/nullptr, /*output_quantization=*/nullptr,
            /*flags=*/0, &raw_op) != xnn_status_success) {
      return nullptr;
    }
    binary_mul_op.reset(raw_op);

    if (attr.softcap > 0.0f) {
      raw_op = nullptr;
      if (xnn_create_binary_elementwise_nd(
              xnn_binary_divide, xnn_datatype_fp32, /*a_quantization=*/nullptr,
              /*b_quantization=*/nullptr, /*output_quantization=*/nullptr,
              /*flags=*/0, &raw_op) != xnn_status_success) {
        return nullptr;
      }
      binary_div_op.reset(raw_op);

      raw_op = nullptr;
      if (xnn_create_unary_elementwise_nc(
              xnn_unary_tanh, xnn_datatype_fp32, xnn_datatype_fp32,
              /*params=*/nullptr, /*lut=*/nullptr,
              /*input_quantization=*/nullptr, /*output_quantization=*/nullptr,
              /*flags=*/0, &raw_op) != xnn_status_success) {
        return nullptr;
      }
      unary_tanh_op.reset(raw_op);
    }

    int q_id = kInvalidTensorId;
    int k_id = kInvalidTensorId;
    int v_id = kInvalidTensorId;
    int kn_id = kInvalidTensorId;
    int vn_id = kInvalidTensorId;
    int mask_id = kInvalidTensorId;
    int param_id = kInvalidTensorId;
    int out_id = node->outputs->data[0];
    int kc_out_id = kInvalidTensorId;
    int vc_out_id = kInvalidTensorId;

    if (attr.kind == SdpaOpKind::kFusedSdpaCacheUpdate) {
      q_id = node->inputs->data[0];
      k_id = node->inputs->data[1];
      v_id = node->inputs->data[2];
      kn_id = node->inputs->data[3];
      vn_id = node->inputs->data[4];
      mask_id = node->inputs->data[5];
      param_id = node->inputs->data[6];
      if (node->outputs->size == 3) {
        kc_out_id = node->outputs->data[1];
        vc_out_id = node->outputs->data[2];
      }
    } else {
      q_id = node->inputs->data[0];
      k_id = node->inputs->data[1];
      v_id = node->inputs->data[2];
      if (node->inputs->size >= 4 &&
          node->inputs->data[3] != kTfLiteOptionalTensor) {
        const int idx3 = node->inputs->data[3];
        const TfLiteTensor& t3 = context->tensors[idx3];
        if (t3.type == kTfLiteFloat32 || t3.type == kTfLiteBool) {
          mask_id = idx3;
        } else {
          param_id = idx3;
        }
      }
      if (node->inputs->size >= 5 &&
          node->inputs->data[4] != kTfLiteOptionalTensor) {
        param_id = node->inputs->data[4];
      }
    }

    const void* pool_key = params->delegate != nullptr
                               ? static_cast<const void*>(params->delegate)
                               : static_cast<const void*>(context);
    std::shared_ptr<SdpaScratchPool> scratch_pool =
        GetOrCreateSdpaScratchPool(pool_key);

    return std::unique_ptr<Impl>(new Impl(
        attr, q_id, k_id, v_id, kn_id, vn_id, mask_id, param_id, out_id,
        kc_out_id, vc_out_id, std::move(bmm1_op), std::move(bmm2_op),
        std::move(softmax_op), std::move(transpose_op),
        std::move(binary_add_op), std::move(binary_min_op),
        std::move(binary_mul_op), std::move(binary_div_op),
        std::move(unary_tanh_op), threadpool, std::move(scratch_pool)));
  }

  TfLiteStatus Prepare(TfLiteContext* context) {
    const TfLiteTensor& q = context->tensors[q_id_];
    TfLiteTensor& out = context->tensors[out_id_];
    if (q.dims != nullptr && out.dims != nullptr &&
        !TfLiteIntArrayEqual(q.dims, out.dims)) {
      TfLiteIntArray* new_shape = TfLiteIntArrayCopy(q.dims);
      TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, &out, new_shape));
    }
    if (kc_out_id_ != kInvalidTensorId) {
      const TfLiteTensor& kc = context->tensors[k_id_];
      TfLiteTensor& kc_out = context->tensors[kc_out_id_];
      if (kc.dims != nullptr && kc_out.dims != nullptr &&
          !TfLiteIntArrayEqual(kc.dims, kc_out.dims)) {
        TfLiteIntArray* new_shape = TfLiteIntArrayCopy(kc.dims);
        TF_LITE_ENSURE_STATUS(
            context->ResizeTensor(context, &kc_out, new_shape));
      }
    }
    if (vc_out_id_ != kInvalidTensorId) {
      const TfLiteTensor& vc = context->tensors[v_id_];
      TfLiteTensor& vc_out = context->tensors[vc_out_id_];
      if (vc.dims != nullptr && vc_out.dims != nullptr &&
          !TfLiteIntArrayEqual(vc.dims, vc_out.dims)) {
        TfLiteIntArray* new_shape = TfLiteIntArrayCopy(vc.dims);
        TF_LITE_ENSURE_STATUS(
            context->ResizeTensor(context, &vc_out, new_shape));
      }
    }
    return kTfLiteOk;
  }

  TfLiteStatus Invoke(TfLiteContext* context) {
    ScopedSdpaScratchBuffers scratch_guard(scratch_pool_.get());
    SdpaScratchBuffers* scratch = scratch_guard.get();
    if (attr_.kind == SdpaOpKind::kFusedSdpaCacheUpdate) {
      return InvokeFusedSdpaCacheUpdate(context, scratch);
    }
    return InvokeStandardSdpa(context, scratch);
  }

 private:
  Impl(SdpaAttributes attr, int q_id, int k_id, int v_id, int kn_id, int vn_id,
       int mask_id, int param_id, int out_id, int kc_out_id, int vc_out_id,
       XnnOperatorPtr bmm1_op, XnnOperatorPtr bmm2_op,
       XnnOperatorPtr softmax_op, XnnOperatorPtr transpose_op,
       XnnOperatorPtr binary_add_op, XnnOperatorPtr binary_min_op,
       XnnOperatorPtr binary_mul_op, XnnOperatorPtr binary_div_op,
       XnnOperatorPtr unary_tanh_op, pthreadpool_t threadpool,
       std::shared_ptr<SdpaScratchPool> scratch_pool)
      : attr_(attr),
        q_id_(q_id),
        k_id_(k_id),
        v_id_(v_id),
        kn_id_(kn_id),
        vn_id_(vn_id),
        mask_id_(mask_id),
        param_id_(param_id),
        out_id_(out_id),
        kc_out_id_(kc_out_id),
        vc_out_id_(vc_out_id),
        bmm1_op_(std::move(bmm1_op)),
        bmm2_op_(std::move(bmm2_op)),
        softmax_op_(std::move(softmax_op)),
        transpose_op_(std::move(transpose_op)),
        binary_add_op_(std::move(binary_add_op)),
        binary_min_op_(std::move(binary_min_op)),
        binary_mul_op_(std::move(binary_mul_op)),
        binary_div_op_(std::move(binary_div_op)),
        unary_tanh_op_(std::move(unary_tanh_op)),
        threadpool_(threadpool),
        scratch_pool_(std::move(scratch_pool)) {}

  static bool IsValid4DTensor(const TfLiteTensor& t, TfLiteType expected_type) {
    return t.type == expected_type && t.dims != nullptr && t.dims->size == 4 &&
           t.dims->data[0] > 0 && t.dims->data[1] > 0 && t.dims->data[2] > 0 &&
           t.dims->data[3] > 0;
  }

  static TfLiteStatus IsStandardSdpaSupported(TfLiteContext* context,
                                              const TfLiteNode* node,
                                              const SdpaAttributes& attr,
                                              int node_index) {
    if (node->inputs->size < 3 || node->inputs->size > 5 ||
        node->outputs->size != 1) {
      return kTfLiteError;
    }
    const TfLiteTensor& q = context->tensors[node->inputs->data[0]];
    const TfLiteTensor& k = context->tensors[node->inputs->data[1]];
    const TfLiteTensor& v = context->tensors[node->inputs->data[2]];
    const TfLiteTensor& out = context->tensors[node->outputs->data[0]];

    if (!IsValid4DTensor(q, kTfLiteFloat32) ||
        !IsValid4DTensor(k, kTfLiteFloat32) ||
        !IsValid4DTensor(v, kTfLiteFloat32) ||
        !IsValid4DTensor(out, kTfLiteFloat32)) {
      return kTfLiteError;
    }

    const bool is_seq_major =
        (attr.kind == SdpaOpKind::kScaledDotProductAttention);
    if (is_seq_major) {
      // q: [B, T, N_q, D], k: [B_k, S, N_kv, D], v: [B_k, S, N_kv, D]
      const int b = q.dims->data[0];
      const int n_q = q.dims->data[2];
      const int d = q.dims->data[3];
      const int b_k = k.dims->data[0];
      const int s = k.dims->data[1];
      const int n_kv = k.dims->data[2];
      if ((b_k != b && b_k != 1) || v.dims->data[0] != b_k ||
          v.dims->data[1] != s || v.dims->data[2] != n_kv ||
          k.dims->data[3] != d || v.dims->data[3] != d || n_kv <= 0 ||
          n_q < n_kv || n_q % n_kv != 0) {
        return kTfLiteError;
      }
    } else {
      if (attr.k_ts_idx != 2 || (attr.v_ts_idx != 2 && attr.v_ts_idx != 3)) {
        return kTfLiteError;
      }
      // q: [B, N_q_or_kv, Q_rows, D], k: [B_k, N_kv, S, D],
      // v: [B_k, N_kv, D, S] (if v_ts_idx == 3) or [B_k, N_kv, S, D] (if 2).
      const int b = q.dims->data[0];
      const int d = q.dims->data[3];
      const int b_k = k.dims->data[0];
      const int n_kv = k.dims->data[1];
      const int s = k.dims->data[2];
      if ((b_k != b && b_k != 1) || v.dims->data[0] != b_k ||
          v.dims->data[1] != n_kv || k.dims->data[3] != d || n_kv <= 0 ||
          q.dims->data[1] < n_kv || q.dims->data[1] % n_kv != 0) {
        return kTfLiteError;
      }
      if (attr.v_ts_idx == 3) {
        if (v.dims->data[2] != d || v.dims->data[3] != s) {
          return kTfLiteError;
        }
      } else {
        if (v.dims->data[2] != s || v.dims->data[3] != d) {
          return kTfLiteError;
        }
      }
    }

    if (node->inputs->size >= 4 &&
        node->inputs->data[3] != kTfLiteOptionalTensor) {
      const TfLiteTensor& input3 = context->tensors[node->inputs->data[3]];
      if (input3.type == kTfLiteFloat32 || input3.type == kTfLiteBool) {
        if (input3.dims == nullptr || input3.dims->size != 4) {
          return kTfLiteError;
        }
      } else if (input3.type == kTfLiteInt32 || input3.type == kTfLiteInt64) {
        if (NumElements(&input3) < 1) {
          return kTfLiteError;
        }
      } else {
        return kTfLiteError;
      }
    }

    if (node->inputs->size >= 5 &&
        node->inputs->data[4] != kTfLiteOptionalTensor) {
      const TfLiteTensor& input4 = context->tensors[node->inputs->data[4]];
      if ((input4.type != kTfLiteInt32 && input4.type != kTfLiteInt64) ||
          NumElements(&input4) < 1) {
        return kTfLiteError;
      }
    }
    (void)node_index;
    return kTfLiteOk;
  }

  static TfLiteStatus IsFusedSdpaCacheUpdateSupported(TfLiteContext* context,
                                                      const TfLiteNode* node,
                                                      int node_index) {
    if (node->inputs->size != 7 ||
        (node->outputs->size != 1 && node->outputs->size != 3)) {
      return kTfLiteError;
    }
    const TfLiteTensor& q = context->tensors[node->inputs->data[0]];
    const TfLiteTensor& kc = context->tensors[node->inputs->data[1]];
    const TfLiteTensor& vc = context->tensors[node->inputs->data[2]];
    const TfLiteTensor& kn = context->tensors[node->inputs->data[3]];
    const TfLiteTensor& vn = context->tensors[node->inputs->data[4]];
    const TfLiteTensor& mask = context->tensors[node->inputs->data[5]];
    const TfLiteTensor& param = context->tensors[node->inputs->data[6]];
    const TfLiteTensor& out = context->tensors[node->outputs->data[0]];

    if (!IsValid4DTensor(q, kTfLiteFloat32) ||
        !IsValid4DTensor(kc, kTfLiteFloat32) ||
        !IsValid4DTensor(vc, kTfLiteFloat32) ||
        !IsValid4DTensor(kn, kTfLiteFloat32) ||
        !IsValid4DTensor(vn, kTfLiteFloat32) ||
        !IsValid4DTensor(out, kTfLiteFloat32)) {
      return kTfLiteError;
    }
    if ((mask.type != kTfLiteBool && mask.type != kTfLiteFloat32) ||
        mask.dims == nullptr || mask.dims->size != 4) {
      return kTfLiteError;
    }
    if ((param.type != kTfLiteInt32 && param.type != kTfLiteInt64) ||
        NumElements(&param) < 2) {
      return kTfLiteError;
    }

    const int h = kc.dims->data[1];
    const int w = kc.dims->data[2];
    const int d = kc.dims->data[3];
    const int t = kn.dims->data[2];
    const int gt = q.dims->data[2];

    if (q.dims->data[0] != 1 || kc.dims->data[0] != 1 ||
        vc.dims->data[0] != 1 || kn.dims->data[0] != 1 ||
        vn.dims->data[0] != 1 || mask.dims->data[0] != 1) {
      return kTfLiteError;
    }
    if (q.dims->data[1] != h || vc.dims->data[1] != h ||
        kn.dims->data[1] != h || vn.dims->data[1] != h) {
      return kTfLiteError;
    }
    if (q.dims->data[3] != d || vc.dims->data[2] != d ||
        kn.dims->data[3] != d || vn.dims->data[2] != d) {
      return kTfLiteError;
    }
    if (vc.dims->data[3] != w || vn.dims->data[3] != t) {
      return kTfLiteError;
    }
    if (gt < t || gt % t != 0) {
      return kTfLiteError;
    }
    if (mask.dims->data[1] != 1 || mask.dims->data[2] != t ||
        mask.dims->data[3] != w + t) {
      return kTfLiteError;
    }

    if (node->outputs->size == 3) {
      const TfLiteTensor& kc_out = context->tensors[node->outputs->data[1]];
      const TfLiteTensor& vc_out = context->tensors[node->outputs->data[2]];
      if (!IsValid4DTensor(kc_out, kTfLiteFloat32) ||
          !IsValid4DTensor(vc_out, kTfLiteFloat32)) {
        return kTfLiteError;
      }
    }
    (void)node_index;
    return kTfLiteOk;
  }

  static void* AlignWorkspace(void* ptr) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned = (address + kXnnpackWorkspaceAlignment - 1) &
                              ~(kXnnpackWorkspaceAlignment - 1);
    return reinterpret_cast<void*>(aligned);
  }

  template <typename T>
  static void EnsureSize(std::vector<T>* buffer, size_t size) {
    const size_t extra_elems = (XNN_EXTRA_BYTES + sizeof(T) - 1) / sizeof(T);
    const size_t padded_size = size + extra_elems;
    if (buffer->size() < padded_size) {
      buffer->resize(padded_size);
    }
  }

  bool RunBmm(xnn_operator_t op, size_t num_batch_dims,
              const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m,
              size_t k, size_t n, const float* input_a, const float* input_b,
              float* output, SdpaScratchBuffers* scratch) {
    size_t workspace_size = 0;
    xnn_status status = xnn_reshape_batch_matrix_multiply_nc_f32(
        op, num_batch_dims, batch_dims_a, batch_dims_b, m, k, n,
        &workspace_size, threadpool_);
    if (status != xnn_status_success) {
      return false;
    }
    void* workspace = nullptr;
    if (workspace_size > 0) {
      EnsureSize(&scratch->workspace,
                 workspace_size + kXnnpackWorkspaceAlignment - 1);
      workspace = AlignWorkspace(scratch->workspace.data());
    }
    status = xnn_setup_batch_matrix_multiply_nc_f32(op, workspace, input_a,
                                                    input_b, output);
    if (status != xnn_status_success) {
      return false;
    }
    return xnn_run_operator(op, threadpool_) == xnn_status_success;
  }

  bool RunTranspose(size_t num_dims, const size_t* input_shape,
                    const size_t* perm, const float* input, float* output) {
    if (xnn_reshape_transpose_nd_x32(transpose_op_.get(), num_dims, input_shape,
                                     perm, threadpool_) != xnn_status_success) {
      return false;
    }
    if (xnn_setup_transpose_nd_x32(transpose_op_.get(), input, output) !=
        xnn_status_success) {
      return false;
    }
    return xnn_run_operator(transpose_op_.get(), threadpool_) ==
           xnn_status_success;
  }

  bool RunBinaryNd(xnn_operator_t op, size_t num_a_dims, const size_t* a_shape,
                   size_t num_b_dims, const size_t* b_shape,
                   const float* input_a, const float* input_b, float* output) {
    if (xnn_reshape_binary_elementwise_nd(op, num_a_dims, a_shape, num_b_dims,
                                          b_shape,
                                          threadpool_) != xnn_status_success) {
      return false;
    }
    if (xnn_setup_binary_elementwise_nd(op, input_a, input_b, output) !=
        xnn_status_success) {
      return false;
    }
    return xnn_run_operator(op, threadpool_) == xnn_status_success;
  }

  bool RunSoftmax(size_t batch_size, size_t channels, const float* input,
                  float* output) {
    if (xnn_reshape_softmax_nc_f32(softmax_op_.get(), channels, channels,
                                   channels, batch_size,
                                   threadpool_) != xnn_status_success) {
      return false;
    }
    if (xnn_setup_softmax_nc_f32(softmax_op_.get(), input, output) !=
        xnn_status_success) {
      return false;
    }
    return xnn_run_operator(softmax_op_.get(), threadpool_) ==
           xnn_status_success;
  }

  bool ApplySoftcapInPlace(size_t num_elements, float* logits) {
    if (attr_.softcap <= 0.0f) {
      return true;
    }
    const size_t shape_1d[1] = {num_elements};
    const size_t scalar_shape[1] = {1};
    const float cap = attr_.softcap;
    if (!RunBinaryNd(binary_div_op_.get(), 1, shape_1d, 1, scalar_shape, logits,
                     &cap, logits)) {
      return false;
    }
    if (xnn_reshape_unary_elementwise_nc(
            unary_tanh_op_.get(), /*batch_size=*/1, num_elements, num_elements,
            num_elements, threadpool_) != xnn_status_success) {
      return false;
    }
    if (xnn_setup_unary_elementwise_nc(unary_tanh_op_.get(), logits, logits) !=
        xnn_status_success) {
      return false;
    }
    if (xnn_run_operator(unary_tanh_op_.get(), threadpool_) !=
        xnn_status_success) {
      return false;
    }
    return RunBinaryNd(binary_mul_op_.get(), 1, shape_1d, 1, scalar_shape,
                       logits, &cap, logits);
  }

  void ParallelFor1D(size_t count, size_t min_Count_per_thread,
                     const std::function<void(size_t, size_t)>& fn) {
    if (count == 0) {
      return;
    }
    if (threadpool_ == nullptr || count <= min_Count_per_thread) {
      fn(0, count);
      return;
    }
    struct Context {
      const std::function<void(size_t, size_t)>* fn;
    } ctx{&fn};
    pthreadpool_parallelize_1d_tile_1d(
        threadpool_,
        +[](void* raw_ctx, size_t offset, size_t tile) {
          (*static_cast<const Context*>(raw_ctx)->fn)(offset, tile);
        },
        &ctx, count, min_Count_per_thread, /*flags=*/0);
  }

  // Executes the core SDPA pipeline:
  //   Q: [B, H_kv, G, T, D]
  //   K: [B_k, H_kv, S_active, D] (with head stride k_head_stride)
  //   V: [B_k, H_kv, D, S_active]
  //   mask_ptr: optional 5D float mask [B_m, H_m, G_m, T_m, S_active]
  //   out_ptr: [B, H_kv, G, T, D]
  bool RunCoreSdpa(size_t b, size_t b_k, size_t h_kv, size_t g, size_t t,
                   size_t d, size_t s_active, size_t k_head_stride,
                   const float* q_ptr, const float* k_ptr, const float* v_ptr,
                   const float* mask_ptr, bool mask_is_upper_bound,
                   const size_t mask_shape_5d[5], float* out_ptr,
                   SdpaScratchBuffers* scratch) {
    const size_t gt = g * t;
    const size_t total_logits = b * h_kv * gt * s_active;
    EnsureSize(&scratch->logits, total_logits);
    float* logits_ptr = scratch->logits.data();

    const bool use_decode =
        (b == b_k) && (gt <= static_cast<size_t>(kDecodeQueryRowThreshold));
    const bool k_strided = (k_head_stride != s_active * d) && (b * h_kv > 1);

    if (use_decode) {
      // Decode optimization: compute K @ Q^T -> [B*H_kv, S_active, G*T] so
      // XNNPACK packs the tiny Q [G*T, D] instead of the large KV cache K
      // [S_active, D], then transpose to [B*H_kv, G*T, S_active].
      EnsureSize(&scratch->decode_tmp, total_logits);
      if (k_strided) {
        const size_t single_batch[1] = {1};
        for (size_t hi = 0; hi < b * h_kv; ++hi) {
          if (!RunBmm(bmm1_op_.get(), /*num_batch_dims=*/1, single_batch,
                      single_batch, /*m=*/s_active, /*k=*/d, /*n=*/gt,
                      k_ptr + hi * k_head_stride, q_ptr + hi * gt * d,
                      scratch->decode_tmp.data() + hi * s_active * gt,
                      scratch)) {
            return false;
          }
        }
      } else {
        const size_t batch_dims[1] = {b * h_kv};
        if (!RunBmm(bmm1_op_.get(), /*num_batch_dims=*/1, batch_dims,
                    batch_dims, /*m=*/s_active, /*k=*/d, /*n=*/gt, k_ptr, q_ptr,
                    scratch->decode_tmp.data(), scratch)) {
          return false;
        }
      }
      const size_t scores_t_shape[3] = {b * h_kv, s_active, gt};
      const size_t perm[3] = {0, 2, 1};
      if (!RunTranspose(3, scores_t_shape, perm, scratch->decode_tmp.data(),
                        logits_ptr)) {
        return false;
      }
    } else if (b == b_k) {
      if (k_strided) {
        const size_t single_batch[1] = {1};
        for (size_t hi = 0; hi < b * h_kv; ++hi) {
          if (!RunBmm(bmm1_op_.get(), /*num_batch_dims=*/1, single_batch,
                      single_batch, /*m=*/gt, /*k=*/d, /*n=*/s_active,
                      q_ptr + hi * gt * d, k_ptr + hi * k_head_stride,
                      logits_ptr + hi * gt * s_active, scratch)) {
            return false;
          }
        }
      } else {
        const size_t batch_dims[1] = {b * h_kv};
        if (!RunBmm(bmm1_op_.get(), /*num_batch_dims=*/1, batch_dims,
                    batch_dims, /*m=*/gt, /*k=*/d, /*n=*/s_active, q_ptr, k_ptr,
                    logits_ptr, scratch)) {
          return false;
        }
      }
    } else {
      const size_t batch_dims_a[3] = {b, h_kv, g};
      const size_t batch_dims_b[3] = {b_k, h_kv, 1};
      if (!RunBmm(bmm1_op_.get(), /*num_batch_dims=*/3, batch_dims_a,
                  batch_dims_b, /*m=*/t, /*k=*/d, /*n=*/s_active, q_ptr, k_ptr,
                  logits_ptr, scratch)) {
        return false;
      }
    }

    if (!ApplySoftcapInPlace(total_logits, logits_ptr)) {
      return false;
    }

    if (mask_ptr != nullptr) {
      const size_t logits_shape_5d[5] = {b, h_kv, g, t, s_active};
      xnn_operator_t mask_op =
          mask_is_upper_bound ? binary_min_op_.get() : binary_add_op_.get();
      if (!RunBinaryNd(mask_op, 5, logits_shape_5d, 5, mask_shape_5d,
                       logits_ptr, mask_ptr, logits_ptr)) {
        return false;
      }
    }

    if (!RunSoftmax(b * h_kv * gt, s_active, logits_ptr, logits_ptr)) {
      return false;
    }

    if (use_decode) {
      // Decode optimization: compute V @ probs^T -> [B*H_kv, D, G*T] so
      // XNNPACK packs the small probs [G*T, S_active] instead of the large V
      // cache [D, S_active], then transpose to [B*H_kv, G*T, D].
      EnsureSize(&scratch->decode_tmp, b * h_kv * d * gt);
      const size_t batch_dims[1] = {b * h_kv};
      if (!RunBmm(bmm2_op_.get(), /*num_batch_dims=*/1, batch_dims, batch_dims,
                  /*m=*/d, /*k=*/s_active, /*n=*/gt, v_ptr, logits_ptr,
                  scratch->decode_tmp.data(), scratch)) {
        return false;
      }
      const size_t out_t_shape[3] = {b * h_kv, d, gt};
      const size_t perm[3] = {0, 2, 1};
      if (!RunTranspose(3, out_t_shape, perm, scratch->decode_tmp.data(),
                        out_ptr)) {
        return false;
      }
    } else if (b == b_k) {
      const size_t batch_dims[1] = {b * h_kv};
      if (!RunBmm(bmm2_op_.get(), /*num_batch_dims=*/1, batch_dims, batch_dims,
                  /*m=*/gt, /*k=*/s_active, /*n=*/d, logits_ptr, v_ptr, out_ptr,
                  scratch)) {
        return false;
      }
    } else {
      const size_t batch_dims_a[3] = {b, h_kv, g};
      const size_t batch_dims_b[3] = {b_k, h_kv, 1};
      if (!RunBmm(bmm2_op_.get(), /*num_batch_dims=*/3, batch_dims_a,
                  batch_dims_b, /*m=*/t, /*k=*/s_active, /*n=*/d, logits_ptr,
                  v_ptr, out_ptr, scratch)) {
        return false;
      }
    }

    return true;
  }

  TfLiteStatus InvokeStandardSdpa(TfLiteContext* context,
                                  SdpaScratchBuffers* scratch) {
    const TfLiteTensor& q_tensor = context->tensors[q_id_];
    const TfLiteTensor& k_tensor = context->tensors[k_id_];
    const TfLiteTensor& v_tensor = context->tensors[v_id_];
    TfLiteTensor& out_tensor = context->tensors[out_id_];

    const float* q_raw = GetTensorData<float>(&q_tensor);
    const float* k_raw = GetTensorData<float>(&k_tensor);
    const float* v_raw = GetTensorData<float>(&v_tensor);
    float* out_raw = GetTensorData<float>(&out_tensor);
    if (q_raw == nullptr || k_raw == nullptr || v_raw == nullptr ||
        out_raw == nullptr) {
      return kTfLiteError;
    }

    const bool is_seq_major =
        (attr_.kind == SdpaOpKind::kScaledDotProductAttention);

    size_t b = static_cast<size_t>(q_tensor.dims->data[0]);
    size_t b_k = static_cast<size_t>(k_tensor.dims->data[0]);
    size_t h_kv = 0;
    size_t g = 1;
    size_t t = 0;
    size_t s_max = 0;
    size_t d = static_cast<size_t>(q_tensor.dims->data[3]);

    if (is_seq_major) {
      t = static_cast<size_t>(q_tensor.dims->data[1]);
      const size_t n_q = static_cast<size_t>(q_tensor.dims->data[2]);
      s_max = static_cast<size_t>(k_tensor.dims->data[1]);
      h_kv = static_cast<size_t>(k_tensor.dims->data[2]);
      if (h_kv == 0 || n_q % h_kv != 0) {
        return kTfLiteError;
      }
      g = n_q / h_kv;
    } else {
      h_kv = static_cast<size_t>(k_tensor.dims->data[1]);
      s_max = static_cast<size_t>(k_tensor.dims->data[2]);
      const size_t q_dim1 = static_cast<size_t>(q_tensor.dims->data[1]);
      const size_t q_dim2 = static_cast<size_t>(q_tensor.dims->data[2]);
      if (h_kv == 0 || q_dim1 < h_kv || q_dim1 % h_kv != 0) {
        return kTfLiteError;
      }
      if (q_dim1 > h_kv) {
        g = q_dim1 / h_kv;
        t = q_dim2;
      } else if (mask_id_ != kInvalidTensorId) {
        const TfLiteTensor& mask_tensor = context->tensors[mask_id_];
        const size_t mask_t = static_cast<size_t>(mask_tensor.dims->data[2]);
        if (mask_t > 1 && q_dim2 > mask_t && q_dim2 % mask_t == 0) {
          t = mask_t;
          g = q_dim2 / mask_t;
        } else {
          g = 1;
          t = q_dim2;
        }
      } else {
        g = 1;
        t = q_dim2;
      }
    }

    size_t s_active = s_max;
    if (param_id_ != kInvalidTensorId) {
      const TfLiteTensor& param_tensor = context->tensors[param_id_];
      const int num_elems = NumElements(&param_tensor);
      if (num_elems > 0) {
        const size_t index = (num_elems >= 2) ? 1 : 0;
        int64_t val = 0;
        if (ReadParamElement(param_tensor, index, &val) && val > 0) {
          s_active = std::min<size_t>(static_cast<size_t>(val), s_max);
        }
      }
    }

    const float scale =
        attr_.scale_specified
            ? attr_.scale
            : (is_seq_major ? (1.0f / std::sqrt(static_cast<float>(d))) : 1.0f);

    // 1. Prepare Q in [B, H_kv, G, T, D] layout.
    const float* q_ptr = q_raw;
    const size_t total_q = b * h_kv * g * t * d;
    if (is_seq_major) {
      EnsureSize(&scratch->q_buf, total_q);
      const size_t q_shape[4] = {b, t, h_kv * g, d};
      const size_t perm[4] = {0, 2, 1, 3};
      if (!RunTranspose(4, q_shape, perm, q_raw, scratch->q_buf.data())) {
        return kTfLiteError;
      }
      q_ptr = scratch->q_buf.data();
    }
    if (scale != 1.0f) {
      EnsureSize(&scratch->q_scaled, total_q);
      const size_t q_1d[1] = {total_q};
      const size_t scalar_1d[1] = {1};
      if (!RunBinaryNd(binary_mul_op_.get(), 1, q_1d, 1, scalar_1d, q_ptr,
                       &scale, scratch->q_scaled.data())) {
        return kTfLiteError;
      }
      q_ptr = scratch->q_scaled.data();
    }

    // 2. Prepare K in [B_k, H_kv, S_active, D] layout (with head stride
    // k_head_stride).
    const float* k_ptr = k_raw;
    size_t k_head_stride = s_active * d;
    if (is_seq_major) {
      const float* k_src = k_raw;
      if (s_active < s_max && b_k > 1) {
        EnsureSize(&scratch->decode_tmp, b_k * s_active * h_kv * d);
        float* dst = scratch->decode_tmp.data();
        const size_t slice_elems = s_active * h_kv * d;
        const size_t full_elems = s_max * h_kv * d;
        for (size_t bi = 0; bi < b_k; ++bi) {
          std::memcpy(dst + bi * slice_elems, k_raw + bi * full_elems,
                      slice_elems * sizeof(float));
        }
        k_src = dst;
      }
      EnsureSize(&scratch->k_buf, b_k * h_kv * s_active * d);
      const size_t k_shape[4] = {b_k, s_active, h_kv, d};
      const size_t perm[4] = {0, 2, 1, 3};
      if (!RunTranspose(4, k_shape, perm, k_src, scratch->k_buf.data())) {
        return kTfLiteError;
      }
      k_ptr = scratch->k_buf.data();
    } else if (s_active < s_max && b_k * h_kv > 1) {
      if (b == b_k && s_active >= 256) {
        k_head_stride = s_max * d;
      } else {
        EnsureSize(&scratch->k_buf, b_k * h_kv * s_active * d);
        float* dst = scratch->k_buf.data();
        const size_t total_heads = b_k * h_kv;
        ParallelFor1D(total_heads, /*min_Count_per_thread=*/1,
                      [&](size_t offset, size_t count) {
                        for (size_t hi = offset; hi < offset + count; ++hi) {
                          std::memcpy(dst + hi * s_active * d,
                                      k_raw + hi * s_max * d,
                                      s_active * d * sizeof(float));
                        }
                      });
        k_ptr = dst;
      }
    }

    // 3. Prepare V in [B_k, H_kv, D, S_active] layout.
    const float* v_ptr = v_raw;
    if (is_seq_major) {
      const float* v_src = v_raw;
      if (s_active < s_max && b_k > 1) {
        EnsureSize(&scratch->decode_tmp, b_k * s_active * h_kv * d);
        float* dst = scratch->decode_tmp.data();
        const size_t slice_elems = s_active * h_kv * d;
        const size_t full_elems = s_max * h_kv * d;
        for (size_t bi = 0; bi < b_k; ++bi) {
          std::memcpy(dst + bi * slice_elems, v_raw + bi * full_elems,
                      slice_elems * sizeof(float));
        }
        v_src = dst;
      }
      EnsureSize(&scratch->v_buf, b_k * h_kv * d * s_active);
      const size_t v_shape[4] = {b_k, s_active, h_kv, d};
      const size_t perm[4] = {0, 2, 3, 1};
      if (!RunTranspose(4, v_shape, perm, v_src, scratch->v_buf.data())) {
        return kTfLiteError;
      }
      v_ptr = scratch->v_buf.data();
    } else if (attr_.v_ts_idx == 2) {
      const float* v_src = v_raw;
      if (s_active < s_max && b_k * h_kv > 1) {
        EnsureSize(&scratch->decode_tmp, b_k * h_kv * s_active * d);
        float* dst = scratch->decode_tmp.data();
        const size_t total_heads = b_k * h_kv;
        ParallelFor1D(total_heads, /*min_Count_per_thread=*/1,
                      [&](size_t offset, size_t count) {
                        for (size_t hi = offset; hi < offset + count; ++hi) {
                          std::memcpy(dst + hi * s_active * d,
                                      v_raw + hi * s_max * d,
                                      s_active * d * sizeof(float));
                        }
                      });
        v_src = dst;
      }
      EnsureSize(&scratch->v_buf, b_k * h_kv * d * s_active);
      const size_t v_shape[4] = {b_k, h_kv, s_active, d};
      const size_t perm[4] = {0, 1, 3, 2};
      if (!RunTranspose(4, v_shape, perm, v_src, scratch->v_buf.data())) {
        return kTfLiteError;
      }
      v_ptr = scratch->v_buf.data();
    } else if (s_active < s_max) {
      EnsureSize(&scratch->v_buf, b_k * h_kv * d * s_active);
      float* dst = scratch->v_buf.data();
      const size_t total_rows = b_k * h_kv * d;
      ParallelFor1D(total_rows, /*min_Count_per_thread=*/32,
                    [&](size_t offset, size_t count) {
                      for (size_t r = offset; r < offset + count; ++r) {
                        std::memcpy(dst + r * s_active, v_raw + r * s_max,
                                    s_active * sizeof(float));
                      }
                    });
      v_ptr = dst;
    }

    // 4. Prepare Mask in 5D [B_m, H_m, G_m, T_m, S_active].
    const float* mask_ptr = nullptr;
    bool mask_is_upper_bound = false;
    size_t mask_shape_5d[5] = {1, 1, 1, 1, s_active};
    if (mask_id_ != kInvalidTensorId) {
      const TfLiteTensor& mask_tensor = context->tensors[mask_id_];
      const size_t m_b = static_cast<size_t>(mask_tensor.dims->data[0]);
      const size_t m_d1 = static_cast<size_t>(mask_tensor.dims->data[1]);
      const size_t m_d2 = static_cast<size_t>(mask_tensor.dims->data[2]);
      const size_t m_s = static_cast<size_t>(mask_tensor.dims->data[3]);
      if (m_s < s_active) {
        return kTfLiteError;
      }
      size_t m_h = 1;
      size_t m_g = 1;
      size_t m_t = 1;
      if (is_seq_major && m_d2 == 1 && (m_d1 == t || m_d1 == 1)) {
        m_t = m_d1;
      } else if (m_d1 == 1) {
        if (m_d2 == g * t && g > 1) {
          m_g = g;
          m_t = t;
        } else {
          m_t = m_d2;
        }
      } else if (m_d1 == h_kv * g) {
        m_h = h_kv;
        m_g = g;
        m_t = m_d2;
      } else if (m_d1 == h_kv && g == 1) {
        m_h = h_kv;
        m_t = m_d2;
      } else {
        return kTfLiteError;
      }
      mask_shape_5d[0] = m_b;
      mask_shape_5d[1] = m_h;
      mask_shape_5d[2] = m_g;
      mask_shape_5d[3] = m_t;
      mask_shape_5d[4] = s_active;

      const size_t mask_rows = m_b * m_h * m_g * m_t;
      if (mask_tensor.type == kTfLiteBool) {
        const bool* mask_bool = GetTensorData<bool>(&mask_tensor);
        if (mask_bool == nullptr) {
          return kTfLiteError;
        }
        EnsureSize(&scratch->mask_buf, mask_rows * s_active);
        float* dst = scratch->mask_buf.data();
        ParallelFor1D(mask_rows, /*min_Count_per_thread=*/16,
                      [&](size_t offset, size_t count) {
                        for (size_t r = offset; r < offset + count; ++r) {
                          const bool* src_row = mask_bool + r * m_s;
                          float* dst_row = dst + r * s_active;
                          for (size_t j = 0; j < s_active; ++j) {
                            dst_row[j] =
                                src_row[j] ? kMaskTrueBound : kMaskFillValue;
                          }
                        }
                      });
        mask_ptr = dst;
        mask_is_upper_bound = true;
      } else {
        const float* mask_f32 = GetTensorData<float>(&mask_tensor);
        if (mask_f32 == nullptr) {
          return kTfLiteError;
        }
        if (s_active == m_s) {
          mask_ptr = mask_f32;
        } else {
          EnsureSize(&scratch->mask_buf, mask_rows * s_active);
          float* dst = scratch->mask_buf.data();
          ParallelFor1D(mask_rows, /*min_Count_per_thread=*/16,
                        [&](size_t offset, size_t count) {
                          for (size_t r = offset; r < offset + count; ++r) {
                            std::memcpy(dst + r * s_active, mask_f32 + r * m_s,
                                        s_active * sizeof(float));
                          }
                        });
          mask_ptr = dst;
        }
        mask_is_upper_bound = false;
      }
    }

    float* out_target = out_raw;
    if (is_seq_major) {
      EnsureSize(&scratch->out_buf, total_q);
      out_target = scratch->out_buf.data();
    }

    if (!RunCoreSdpa(b, b_k, h_kv, g, t, d, s_active, k_head_stride, q_ptr,
                     k_ptr, v_ptr, mask_ptr, mask_is_upper_bound, mask_shape_5d,
                     out_target, scratch)) {
      return kTfLiteError;
    }

    if (is_seq_major) {
      const size_t out_shape[4] = {b, h_kv * g, t, d};
      const size_t perm[4] = {0, 2, 1, 3};
      if (!RunTranspose(4, out_shape, perm, out_target, out_raw)) {
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  TfLiteStatus InvokeFusedSdpaCacheUpdate(TfLiteContext* context,
                                          SdpaScratchBuffers* scratch) {
    const TfLiteTensor& q_tensor = context->tensors[q_id_];
    const TfLiteTensor& kc_tensor = context->tensors[k_id_];
    const TfLiteTensor& vc_tensor = context->tensors[v_id_];
    const TfLiteTensor& kn_tensor = context->tensors[kn_id_];
    const TfLiteTensor& vn_tensor = context->tensors[vn_id_];
    const TfLiteTensor& mask_tensor = context->tensors[mask_id_];
    const TfLiteTensor& param_tensor = context->tensors[param_id_];
    TfLiteTensor& out_tensor = context->tensors[out_id_];

    const float* q_ptr = GetTensorData<float>(&q_tensor);
    const float* kc_ptr = GetTensorData<float>(&kc_tensor);
    const float* vc_ptr = GetTensorData<float>(&vc_tensor);
    const float* kn_ptr = GetTensorData<float>(&kn_tensor);
    const float* vn_ptr = GetTensorData<float>(&vn_tensor);
    float* out_ptr = GetTensorData<float>(&out_tensor);
    if (q_ptr == nullptr || kc_ptr == nullptr || vc_ptr == nullptr ||
        kn_ptr == nullptr || vn_ptr == nullptr || out_ptr == nullptr) {
      return kTfLiteError;
    }

    int64_t start = 0;
    int64_t end = 0;
    if (!ReadParamElement(param_tensor, 0, &start) ||
        !ReadParamElement(param_tensor, 1, &end)) {
      return kTfLiteError;
    }

    const size_t h = static_cast<size_t>(kc_tensor.dims->data[1]);
    const size_t w = static_cast<size_t>(kc_tensor.dims->data[2]);
    const size_t d = static_cast<size_t>(kc_tensor.dims->data[3]);
    const size_t t = static_cast<size_t>(kn_tensor.dims->data[2]);
    const size_t gt = static_cast<size_t>(q_tensor.dims->data[2]);
    if (t == 0 || gt < t || gt % t != 0) {
      return kTfLiteError;
    }
    const size_t g = gt / t;

    // Only slots [0, min(start + 1, W)) of the ring buffer can be valid before
    // the cache write; slicing the cache to L avoids reading and multiplying
    // never-written ring slots during prefill and decode.
    const size_t l = (mask_tensor.type == kTfLiteBool)
                         ? static_cast<size_t>(std::clamp<int64_t>(
                               start + 1, 1, static_cast<int64_t>(w)))
                         : w;
    const size_t s_active = l + t;

    // 1. Concatenate K_all [1, H, L + T, D] and V_all [1, H, D, L + T].
    EnsureSize(&scratch->k_buf, h * s_active * d);
    EnsureSize(&scratch->v_buf, h * d * s_active);
    float* k_all = scratch->k_buf.data();
    float* v_all = scratch->v_buf.data();

    for (size_t hi = 0; hi < h; ++hi) {
      float* k_dst = k_all + hi * s_active * d;
      std::memcpy(k_dst, kc_ptr + hi * w * d, l * d * sizeof(float));
      std::memcpy(k_dst + l * d, kn_ptr + hi * t * d, t * d * sizeof(float));

      for (size_t di = 0; di < d; ++di) {
        float* v_dst = v_all + (hi * d + di) * s_active;
        std::memcpy(v_dst, vc_ptr + (hi * d + di) * w, l * sizeof(float));
        std::memcpy(v_dst + l, vn_ptr + (hi * d + di) * t, t * sizeof(float));
      }
    }

    // 2. Build sliced mask [1, 1, 1, T, L + T].
    EnsureSize(&scratch->mask_buf, t * s_active);
    EnsureSize(&scratch->row_has_valid, t);
    float* mask_dst = scratch->mask_buf.data();
    uint8_t* row_has_valid = scratch->row_has_valid.data();
    bool has_fully_masked_row = false;
    bool mask_is_upper_bound = false;

    const size_t full_cols = w + t;
    if (mask_tensor.type == kTfLiteBool) {
      const bool* mask_bool = GetTensorData<bool>(&mask_tensor);
      if (mask_bool == nullptr) {
        return kTfLiteError;
      }
      for (size_t i = 0; i < t; ++i) {
        const bool* src_row = mask_bool + i * full_cols;
        float* dst_row = mask_dst + i * s_active;
        uint8_t any_valid = 0;
        for (size_t j = 0; j < l; ++j) {
          const bool m = src_row[j];
          any_valid |= static_cast<uint8_t>(m);
          dst_row[j] = m ? kMaskTrueBound : kMaskFillValue;
        }
        for (size_t j = 0; j < t; ++j) {
          const bool m = src_row[w + j];
          any_valid |= static_cast<uint8_t>(m);
          dst_row[l + j] = m ? kMaskTrueBound : kMaskFillValue;
        }
        row_has_valid[i] = any_valid;
        if (!any_valid) {
          has_fully_masked_row = true;
        }
      }
      mask_is_upper_bound = true;
    } else {
      const float* mask_f32 = GetTensorData<float>(&mask_tensor);
      if (mask_f32 == nullptr) {
        return kTfLiteError;
      }
      if (l == w) {
        std::memcpy(mask_dst, mask_f32, t * full_cols * sizeof(float));
      } else {
        for (size_t i = 0; i < t; ++i) {
          const float* src_row = mask_f32 + i * full_cols;
          float* dst_row = mask_dst + i * s_active;
          std::memcpy(dst_row, src_row, l * sizeof(float));
          std::memcpy(dst_row + l, src_row + w, t * sizeof(float));
        }
      }
      mask_is_upper_bound = false;
    }

    const size_t mask_shape_5d[5] = {1, 1, 1, t, s_active};
    if (!RunCoreSdpa(/*b=*/1, /*b_k=*/1, h, g, t, d, s_active,
                     /*k_head_stride=*/s_active * d, q_ptr, k_all, v_all,
                     mask_dst, mask_is_upper_bound, mask_shape_5d, out_ptr,
                     scratch)) {
      return kTfLiteError;
    }

    // 3. If L < W and any query row was completely masked (e.g. a trailing
    // padding token in a prefill chunk), its softmax over the full W + T
    // columns is the uniform average over all W + T value columns.
    if (l < w && has_fully_masked_row) {
      const float inv_cols = 1.0f / static_cast<float>(w + t);
      for (size_t hi = 0; hi < h; ++hi) {
        for (size_t di = 0; di < d; ++di) {
          const float* vc_row = vc_ptr + (hi * d + di) * w;
          const float* vn_row = vn_ptr + (hi * d + di) * t;
          float sum = 0.0f;
          for (size_t s = 0; s < w; ++s) {
            sum += vc_row[s];
          }
          for (size_t j = 0; j < t; ++j) {
            sum += vn_row[j];
          }
          const float mean_v = sum * inv_cols;
          for (size_t gi = 0; gi < g; ++gi) {
            for (size_t i = 0; i < t; ++i) {
              if (!row_has_valid[i]) {
                out_ptr[((hi * g + gi) * t + i) * d + di] = mean_v;
              }
            }
          }
        }
      }
    }

    // 4. Update ring KV caches after attention has finished reading kc and vc.
    if (kc_out_id_ != kInvalidTensorId && vc_out_id_ != kInvalidTensorId) {
      float* kc_out_ptr = GetTensorData<float>(&context->tensors[kc_out_id_]);
      float* vc_out_ptr = GetTensorData<float>(&context->tensors[vc_out_id_]);
      if (kc_out_ptr == nullptr || vc_out_ptr == nullptr) {
        return kTfLiteError;
      }
      ApplyFusedCacheWrite(start, end, h, w, t, d, kc_ptr, vc_ptr, kn_ptr,
                           vn_ptr, kc_out_ptr, vc_out_ptr);
    }

    return kTfLiteOk;
  }

  SdpaAttributes attr_;
  int q_id_ = kInvalidTensorId;
  int k_id_ = kInvalidTensorId;
  int v_id_ = kInvalidTensorId;
  int kn_id_ = kInvalidTensorId;
  int vn_id_ = kInvalidTensorId;
  int mask_id_ = kInvalidTensorId;
  int param_id_ = kInvalidTensorId;
  int out_id_ = kInvalidTensorId;
  int kc_out_id_ = kInvalidTensorId;
  int vc_out_id_ = kInvalidTensorId;

  XnnOperatorPtr bmm1_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr bmm2_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr softmax_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr transpose_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr binary_add_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr binary_min_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr binary_mul_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr binary_div_op_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr unary_tanh_op_{nullptr, &xnn_delete_operator};

  pthreadpool_t threadpool_ = nullptr;
  std::shared_ptr<SdpaScratchPool> scratch_pool_;
};

SdpaDelegateKernel::SdpaDelegateKernel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SdpaDelegateKernel::~SdpaDelegateKernel() = default;

bool SdpaDelegateKernel::IsSdpaNode(const TfLiteRegistration* registration,
                                    const TfLiteNode* node) {
  return Impl::IsSdpaNode(registration, node);
}

TfLiteStatus SdpaDelegateKernel::IsSupported(
    TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration, int node_index) {
  return Impl::IsSupported(context, node, registration, node_index);
}

std::unique_ptr<SdpaDelegateKernel> SdpaDelegateKernel::Create(
    TfLiteContext* context, const TfLiteDelegateParams* params,
    pthreadpool_t threadpool) {
  std::unique_ptr<Impl> impl = Impl::Create(context, params, threadpool);
  if (impl == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<SdpaDelegateKernel>(
      new SdpaDelegateKernel(std::move(impl)));
}

TfLiteStatus SdpaDelegateKernel::Prepare(TfLiteContext* context) {
  return impl_->Prepare(context);
}

TfLiteStatus SdpaDelegateKernel::Invoke(TfLiteContext* context) {
  return impl_->Invoke(context);
}

}  // namespace xnnpack
}  // namespace tflite
