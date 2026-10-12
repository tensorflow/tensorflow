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

#include "tensorflow/lite/delegates/ynnpack/fused_sdpa_cache_update.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/attention.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {

namespace {

constexpr int kQuery = 0;
constexpr int kKeyCache = 1;
constexpr int kValueCache = 2;
constexpr int kKeyNew = 3;
constexpr int kValueNew = 4;
constexpr int kMask = 5;
constexpr int kParam = 6;
constexpr int kNumInputs = 7;

// Decode-sized problems (G * T <= kDecodeRows) compute Q.K^T as (K.Q^T)^T and
// P.V as (V.P^T)^T so the long key axis is the one being reduced over without
// packing the cache.
constexpr int kDecodeRows = 32;
// Upper bound for attended logits; far above any real logit.
constexpr float kKeepBound = 1e30f;

size_t ElementSize(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return 4;
    case kTfLiteFloat16:
    case kTfLiteBFloat16:
      return 2;
    default:
      return 0;
  }
}

bool SameDims(const TfLiteTensor& a, const TfLiteTensor& b) {
  if (a.dims->size != b.dims->size) return false;
  for (int i = 0; i < a.dims->size; ++i) {
    if (a.dims->data[i] != b.dims->data[i]) return false;
  }
  return true;
}

// Converts a boolean mask (true = attend) into a bound for `min`: kKeepBound
// where attended, kAttentionMaskFillValue where masked. `min(logits, bound)`
// then *replaces* masked logits by kAttentionMaskFillValue like the reference
// decomposition (torch.where) for any logit >= kAttentionMaskFillValue, in one
// elementwise op. Replacing (rather than adding) matters for rows with no
// valid column, e.g. padding rows: they then average over all W + T columns
// like the reference.
TfLiteStatus BoolMaskToBound(TfLiteContext* context, ynn_subgraph_t subgraph,
                             uint32_t mask_id, uint32_t* bound_id) {
  uint32_t keep = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_convert(subgraph, mask_id, ynn_type_fp32, &keep, 0));
  uint32_t range = YNN_INVALID_VALUE_ID;
  uint32_t fill = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineFp32Scalar(
      context, subgraph, kKeepBound - kAttentionMaskFillValue, &range));
  TF_LITE_ENSURE_STATUS(
      DefineFp32Scalar(context, subgraph, kAttentionMaskFillValue, &fill));
  uint32_t scaled = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                              keep, range, &scaled, 0));
  *bound_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, ynn_binary_add, scaled, fill, bound_id, 0));
  return kTfLiteOk;
}

}  // namespace

bool IsFusedSdpaCacheUpdate(const TfLiteRegistration* registration,
                            const TfLiteNode* node) {
  return IsCustomOrCompositeOpNamed(registration, node,
                                    kFusedSdpaCacheUpdateName);
}

TfLiteStatus IsFusedSdpaCacheUpdateSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context) {
  TF_LITE_ENSURE(context, IsFusedSdpaCacheUpdate(registration, node));
  TF_LITE_ENSURE_EQ(context, node->inputs->size, kNumInputs);
  TF_LITE_ENSURE(context, node->outputs->size == 1 || node->outputs->size == 3);

  const TfLiteTensor& q = context->tensors[node->inputs->data[kQuery]];
  const TfLiteTensor& kc = context->tensors[node->inputs->data[kKeyCache]];
  const TfLiteTensor& vc = context->tensors[node->inputs->data[kValueCache]];
  const TfLiteTensor& kn = context->tensors[node->inputs->data[kKeyNew]];
  const TfLiteTensor& vn = context->tensors[node->inputs->data[kValueNew]];
  const TfLiteTensor& mask = context->tensors[node->inputs->data[kMask]];
  const int param_index = node->inputs->data[kParam];
  const TfLiteTensor& param = context->tensors[param_index];
  const TfLiteTensor& out = context->tensors[node->outputs->data[0]];

  for (const TfLiteTensor* t : {&q, &kc, &vc, &kn, &vn, &out}) {
    TF_LITE_ENSURE(context, IsTensorSupported(*t));
    TF_LITE_ENSURE(context, IsFloatAttentionType(t->type));
    TF_LITE_ENSURE(context, !IsQuantized(*t));
    TF_LITE_ENSURE_EQ(context, t->dims->size, 4);
  }
  // Host-side cache write needs K/V caches and new K/V in the same dtype.
  TF_LITE_ENSURE_EQ(context, kc.type, kn.type);
  TF_LITE_ENSURE_EQ(context, vc.type, vn.type);
  TF_LITE_ENSURE_EQ(context, kc.type, vc.type);

  const int h = kc.dims->data[1];
  const int w = kc.dims->data[2];
  const int d = kc.dims->data[3];
  const int t = kn.dims->data[2];
  TF_LITE_ENSURE(context, h > 0 && w > 0 && d > 0 && t > 0);
  TF_LITE_ENSURE_EQ(context, kc.dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, q.dims->data[1], h);
  TF_LITE_ENSURE_EQ(context, q.dims->data[3], d);
  TF_LITE_ENSURE_EQ(context, q.dims->data[2] % t, 0);
  TF_LITE_ENSURE_EQ(context, vc.dims->data[1], h);
  TF_LITE_ENSURE_EQ(context, vc.dims->data[2], d);
  TF_LITE_ENSURE_EQ(context, vc.dims->data[3], w);
  TF_LITE_ENSURE_EQ(context, kn.dims->data[1], h);
  TF_LITE_ENSURE_EQ(context, kn.dims->data[3], d);
  TF_LITE_ENSURE_EQ(context, vn.dims->data[1], h);
  TF_LITE_ENSURE_EQ(context, vn.dims->data[2], d);
  TF_LITE_ENSURE_EQ(context, vn.dims->data[3], t);
  TF_LITE_ENSURE(context, SameDims(out, q));

  TF_LITE_ENSURE(context,
                 mask.type == kTfLiteBool || IsFloatAttentionType(mask.type));
  TF_LITE_ENSURE_EQ(context, mask.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, mask.dims->data[2], t);
  TF_LITE_ENSURE_EQ(context, mask.dims->data[3], w + t);

  // The param tensor is read on the host before and after the runtime is
  // invoked (ring-fill extent and cache write offsets), so it must not be
  // produced inside a delegated partition.
  TF_LITE_ENSURE(context,
                 param.type == kTfLiteInt32 || param.type == kTfLiteInt64);
  TF_LITE_ENSURE(context, tflite::NumElements(&param) >= 2);
  TF_LITE_ENSURE(context, !IsProducedInSubgraph(context, param_index));

  if (node->outputs->size == 3) {
    TF_LITE_ENSURE(context,
                   SameDims(context->tensors[node->outputs->data[1]], kc));
    TF_LITE_ENSURE(context,
                   SameDims(context->tensors[node->outputs->data[2]], vc));
    TF_LITE_ENSURE_EQ(context, context->tensors[node->outputs->data[1]].type,
                      kc.type);
    TF_LITE_ENSURE_EQ(context, context->tensors[node->outputs->data[2]].type,
                      vc.type);
  }
  return kTfLiteOk;
}

TfLiteStatus DefineFusedSdpaCacheUpdateNode(
    TfLiteContext* context, ynn_subgraph_t subgraph,
    TensorToValueIdMap& tensor_to_value_id, uint32_t& next_external_id,
    std::vector<DummyInputInfo>& dummy_inputs, const NodeInfo& node,
    std::vector<FusedCacheWrite>& cache_writes) {
  const TfLiteTensor& q = context->tensors[node.inputs[kQuery]];
  const TfLiteTensor& kc = context->tensors[node.inputs[kKeyCache]];
  const TfLiteTensor& kn = context->tensors[node.inputs[kKeyNew]];
  const TfLiteTensor& mask = context->tensors[node.inputs[kMask]];

  const int kv_heads = kc.dims->data[1];
  const int cache_size = kc.dims->data[2];
  const int head_dim = kc.dims->data[3];
  const int new_len = kn.dims->data[2];
  const int rows = q.dims->data[2];
  const size_t g = static_cast<size_t>(rows / new_len);
  const bool decode = rows <= kDecodeRows;

  const flexbuffers::Map attrs = GetFlexBufferMap(node);
  float softcap = 0.0f;
  if (!attrs["softcap"].IsNull()) softcap = attrs["softcap"].AsFloat();

  auto value_id = [&](int tensor_index) {
    return GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                              tensor_index);
  };
  const uint32_t q_id = value_id(node.inputs[kQuery]);
  const uint32_t kc_id = value_id(node.inputs[kKeyCache]);
  const uint32_t vc_id = value_id(node.inputs[kValueCache]);
  const uint32_t kn_id = value_id(node.inputs[kKeyNew]);
  const uint32_t vn_id = value_id(node.inputs[kValueNew]);
  const uint32_t mask_id = value_id(node.inputs[kMask]);
  uint32_t out_id = value_id(node.outputs[0]);
  for (uint32_t id : {q_id, kc_id, vc_id, kn_id, vn_id, mask_id, out_id}) {
    TF_LITE_ENSURE(context, id != YNN_INVALID_VALUE_ID);
  }

  // Work in a rank-5 [1, H, G, T, *] layout so the [1, 1, T, S] mask
  // broadcasts over the G query heads that share a KV head. (Applying it to
  // packed [1, H, G * T, S] logits needs split/fuse reshapes around every mask
  // op, which keeps YNNPACK from fusing the elementwise chain and costs ~2x
  // for prefill on Arm.)
  const size_t q_splits[2] = {g, 0};
  uint32_t q5 = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(
      subgraph, /*axis=*/2, /*num_splits=*/2, q_splits, q_id, &q5, 0));
  const int32_t kv_group_axis = 2;
  auto add_group_axis = [&](uint32_t id, uint32_t* out) -> TfLiteStatus {
    *out = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, /*num_new_axes=*/1, &kv_group_axis, id, out, 0));
    return kTfLiteOk;
  };

  // 1. Attend over [cache | new tokens]. The mask alone decides which cache
  //    slots are valid. The concatenated K/V are small next to the
  //    [G * T, W + T] logits.
  //
  //    Until the ring wraps, only slots [0, L), L = min(start + 1, W), can
  //    hold tokens of this sequence (slot `start` holds the token LiteRT-LM
  //    re-feeds at the start of a chunk); the mask excludes the rest for every
  //    row. Prefill therefore attends over [cache[0, L) | new], with L carried
  //    by dummy inputs resized from param[0] at each invoke. This skips up to
  //    W / (W + T) of the first chunk's attention. Rows without any valid
  //    column are kept exact by the correction in step 3.
  //    Decode keeps all W columns: it is bandwidth bound, and its T = 1 mask
  //    slice would be a broadcast dimension. Float (additive) masks and
  //    non-fp32 tensors also keep the full ring.
  const TfLiteTensor& vc = context->tensors[node.inputs[kValueCache]];
  const bool slice_ring = !decode && mask.type == kTfLiteBool &&
                          kc.type == kTfLiteFloat32 &&
                          vc.type == kTfLiteFloat32 && q.type == kTfLiteFloat32;
  uint32_t kc_used = kc_id;
  uint32_t vc_used = vc_id;
  uint32_t mask_used = mask_id;
  uint32_t skipped_count = YNN_INVALID_VALUE_ID;  // scalar W - L
  uint32_t skipped_v_sum = YNN_INVALID_VALUE_ID;  // [1, H, 1, 1, D]
  if (slice_ring) {
    const int param_index = node.inputs[kParam];
    auto ring_fill_dummy = [&](const TfLiteTensor& cache, int seq_axis,
                               uint32_t* id) -> TfLiteStatus {
      size_t full_dims[4];
      for (int i = 0; i < 4; ++i) full_dims[i] = cache.dims->data[i];
      return GetOrCreateDummyInput(context, subgraph, next_external_id,
                                   dummy_inputs, param_index, seq_axis,
                                   /*rank=*/4, full_dims,
                                   GetYnnType(cache.type), id,
                                   DummyExtent::kRingFill);
    };
    uint32_t k_fill = YNN_INVALID_VALUE_ID;  // [1, H, L, D]
    uint32_t v_fill = YNN_INVALID_VALUE_ID;  // [1, H, D, L]
    TF_LITE_ENSURE_STATUS(ring_fill_dummy(kc, /*seq_axis=*/2, &k_fill));
    TF_LITE_ENSURE_STATUS(ring_fill_dummy(vc, /*seq_axis=*/3, &v_fill));
    const int32_t axis2[] = {2};
    const int32_t axis3[] = {3};
    kc_used = YNN_INVALID_VALUE_ID;
    vc_used = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_slice_like(subgraph, 1, axis2, kc_id, k_fill, &kc_used, 0));
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_slice_like(subgraph, 1, axis3, vc_id, v_fill, &vc_used, 0));
    // Mask columns: [0, L) of the cache part, then the T new-token columns.
    uint32_t mask_cache = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(subgraph, 1, axis3, mask_id,
                                                    v_fill, &mask_cache, 0));
    const int64_t begin = cache_size;
    const int64_t end = cache_size + new_len;
    const int64_t stride = 1;
    uint32_t mask_new = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_slice(
        subgraph, 1, axis3, &begin, &end, &stride, mask_id, &mask_new, 0));
    const uint32_t mask_parts[2] = {mask_cache, mask_new};
    mask_used = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_concatenate(
        subgraph, /*axis=*/3, /*num_inputs=*/2, mask_parts, &mask_used, 0));

    // Skipped slots [L, W): their V sum and their count, as x - (x with the
    // slots >= L zeroed), so both are exact zero once the ring is full.
    auto skipped_part = [&](uint32_t x, uint32_t* out) -> TfLiteStatus {
      uint32_t head = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_slice_like(
          subgraph, 1, axis3, x, v_fill, &head, YNN_NODE_FLAG_KEEP_SHAPE));
      *out = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_binary(subgraph, ynn_binary_subtract, x, head, out, 0));
      return kTfLiteOk;
    };
    uint32_t v_skipped = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(skipped_part(vc_id, &v_skipped));
    uint32_t v_sum = YNN_INVALID_VALUE_ID;  // [1, H, D, 1]
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
        subgraph, ynn_reduce_sum, 1, axis3, v_skipped, YNN_INVALID_VALUE_ID,
        &v_sum, YNN_NODE_FLAG_KEEP_DIMS));
    uint32_t v_sum_t = YNN_INVALID_VALUE_ID;  // [1, H, 1, D]
    const int32_t swap4[] = {0, 1, 3, 2};
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_static_transpose(subgraph, 4, swap4, v_sum, &v_sum_t, 0));
    TF_LITE_ENSURE_STATUS(add_group_axis(v_sum_t, &skipped_v_sum));

    const std::vector<float> ones(cache_size, 1.0f);
    const size_t ones_dims[4] = {1, 1, 1, static_cast<size_t>(cache_size)};
    uint32_t ones_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_tensor(subgraph, ynn_type_fp32, 4, ones_dims, ones.data(),
                          YNN_VALUE_FLAG_COPY_DATA_FP32, &ones_id));
    uint32_t ones_skipped = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(skipped_part(ones_id, &ones_skipped));
    const int32_t all_axes[] = {0, 1, 2, 3};
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_reduce(subgraph, ynn_reduce_sum, 4, all_axes, ones_skipped,
                          YNN_INVALID_VALUE_ID, &skipped_count, 0));
  }

  const uint32_t k_parts[2] = {kc_used, kn_id};
  const uint32_t v_parts[2] = {vc_used, vn_id};
  uint32_t k_all = YNN_INVALID_VALUE_ID;
  uint32_t v_all = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_concatenate(
      subgraph, /*axis=*/2, /*num_inputs=*/2, k_parts, &k_all, 0));
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_concatenate(
      subgraph, /*axis=*/3, /*num_inputs=*/2, v_parts, &v_all, 0));
  uint32_t k5 = YNN_INVALID_VALUE_ID;  // [1, H, 1, L + T, D]
  uint32_t v5 = YNN_INVALID_VALUE_ID;  // [1, H, 1, D, L + T]
  TF_LITE_ENSURE_STATUS(add_group_axis(k_all, &k5));
  TF_LITE_ENSURE_STATUS(add_group_axis(v_all, &v5));

  uint32_t logits = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(
      DefineAttentionScores(context, subgraph, q5, k5, decode, &logits));
  TF_LITE_ENSURE_STATUS(ApplyLogitSoftcap(context, subgraph, softcap, &logits));

  // 2. Mask: bool masks replace masked logits (see BoolMaskToBound); float
  //    masks are additive.
  uint32_t mask_term = mask_used;
  if (mask.type == kTfLiteBool) {
    TF_LITE_ENSURE_STATUS(
        BoolMaskToBound(context, subgraph, mask_used, &mask_term));
  }
  // The mask is shared by every KV head: its batch and head dims have extent
  // 1 but are not broadcast dims when shapes are dynamic.
  const int32_t bcast_axes[] = {0, 1};
  uint32_t mask_bcast = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_broadcast(subgraph, 2, bcast_axes, mask_term, &mask_bcast, 0));
  uint32_t mask5 = YNN_INVALID_VALUE_ID;  // [1, 1, 1, T, L + T]
  TF_LITE_ENSURE_STATUS(add_group_axis(mask_bcast, &mask5));
  uint32_t masked = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
      subgraph, mask.type == kTfLiteBool ? ynn_binary_min : ynn_binary_add,
      logits, mask5, &masked, 0));

  // 3. out = softmax(l) . V, normalizing P before the dot like the reference
  //    decomposition and the odml.sdpa kernel. (Normalizing after P.V is as
  //    accurate and no faster here, but reorders the rounding, which the
  //    int8-quantized layers downstream then amplify into visible NLL shifts.)
  const int32_t reduce_axis[] = {-1};
  uint32_t max_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_reduce(
      subgraph, ynn_reduce_max, 1, reduce_axis, masked, YNN_INVALID_VALUE_ID,
      &max_id, YNN_NODE_FLAG_KEEP_DIMS));
  uint32_t centered = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_subtract,
                                              masked, max_id, &centered, 0));
  uint32_t e = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_unary(subgraph, ynn_unary_exp, centered, &e, 0));
  uint32_t denom = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_reduce(subgraph, ynn_reduce_sum, 1, reduce_axis, e,
                        YNN_INVALID_VALUE_ID, &denom, YNN_NODE_FLAG_KEEP_DIMS));
  // With a sliced ring, account for the W - L skipped (always masked) slots
  // as the reference does: each has logit kAttentionMaskFillValue, i.e. weight
  // c = exp(kAttentionMaskFillValue - max). c underflows to exactly 0 for any
  // row with a valid column, and is 1 for a row with none, which thus again
  // averages all W + T columns uniformly.
  uint32_t skipped_weight = YNN_INVALID_VALUE_ID;  // c, [1, H, G, T, 1]
  if (slice_ring) {
    uint32_t fill = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(
        DefineFp32Scalar(context, subgraph, kAttentionMaskFillValue, &fill));
    uint32_t fill_centered = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_subtract, fill, max_id, &fill_centered, 0));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_unary(
        subgraph, ynn_unary_exp, fill_centered, &skipped_weight, 0));
    uint32_t skipped_mass = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                skipped_weight, skipped_count,
                                                &skipped_mass, 0));
    uint32_t full_denom = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_add, denom,
                                                skipped_mass, &full_denom, 0));
    denom = full_denom;
  }
  uint32_t probs = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_binary(subgraph, ynn_binary_divide, e, denom, &probs, 0));
  uint32_t o5 = YNN_INVALID_VALUE_ID;
  const int32_t swap_last_two_perm[] = {-1, -2};
  TF_LITE_ENSURE_STATUS(DefineAttentionWeightedValues(
      context, subgraph, probs, v5, decode, swap_last_two_perm, &o5));
  if (slice_ring) {
    // o += (c / denom) * sum(V[L, W)).
    uint32_t p_skipped = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(
        subgraph, ynn_binary_divide, skipped_weight, denom, &p_skipped, 0));
    uint32_t correction = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_multiply,
                                                p_skipped, skipped_v_sum,
                                                &correction, 0));
    uint32_t corrected = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_binary(subgraph, ynn_binary_add, o5,
                                                correction, &corrected, 0));
    o5 = corrected;
  }
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(
      subgraph, /*axis=*/2, /*axes_count=*/2, o5, &out_id, 0));
  tensor_to_value_id[node.outputs[0]] = out_id;

  // 4. Cache write: stage K/V_new into delegate-owned buffers; the actual
  //    ring-buffer write happens on the host after the runtime has run.
  if (node.outputs.size() == 3) {
    FusedCacheWrite write;
    write.param_tensor_index = node.inputs[kParam];
    write.k_cache_in_index = node.inputs[kKeyCache];
    write.v_cache_in_index = node.inputs[kValueCache];
    write.k_cache_out_index = node.outputs[1];
    write.v_cache_out_index = node.outputs[2];
    write.element_size = ElementSize(kc.type);
    write.kv_heads = kv_heads;
    write.head_dim = head_dim;
    write.cache_size = cache_size;
    write.new_len = new_len;
    TF_LITE_ENSURE(context, write.element_size > 0);
    const size_t bytes =
        static_cast<size_t>(kv_heads) * new_len * head_dim * write.element_size;
    write.k_new_buffer.resize(bytes);
    write.v_new_buffer.resize(bytes);

    const TfLiteTensor& vn = context->tensors[node.inputs[kValueNew]];
    auto stage = [&](const TfLiteTensor& src, uint32_t src_id,
                     uint32_t* ext_id) -> TfLiteStatus {
      size_t dims[4];
      for (int i = 0; i < 4; ++i) dims[i] = src.dims->data[i];
      *ext_id = next_external_id++;
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_tensor(subgraph, GetYnnType(src.type), 4, dims, nullptr,
                            YNN_VALUE_FLAG_EXTERNAL_OUTPUT, ext_id));
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_copy(subgraph, src_id, ext_id, 0));
      return kTfLiteOk;
    };
    TF_LITE_ENSURE_STATUS(stage(kn, kn_id, &write.k_new_ext_id));
    TF_LITE_ENSURE_STATUS(stage(vn, vn_id, &write.v_new_ext_id));
    cache_writes.push_back(std::move(write));
  }
  return kTfLiteOk;
}

TfLiteStatus ApplyFusedCacheWrite(TfLiteContext* context,
                                  const FusedCacheWrite& write) {
  const TfLiteTensor& param = context->tensors[write.param_tensor_index];
  int64_t start = 0;
  int64_t end = 0;
  TF_LITE_ENSURE(context, ReadParamElement(param, 0, &start));
  TF_LITE_ENSURE(context, ReadParamElement(param, 1, &end));
  TF_LITE_ENSURE(context, start >= 0);

  const int64_t w = write.cache_size;
  const int64_t t = write.new_len;
  const int64_t d = write.head_dim;
  const size_t es = write.element_size;
  const int64_t valid = std::clamp<int64_t>(end - start, 0, t);

  struct CachePair {
    int in_index;
    int out_index;
  };
  for (const CachePair& pair :
       {CachePair{write.k_cache_in_index, write.k_cache_out_index},
        CachePair{write.v_cache_in_index, write.v_cache_out_index}}) {
    const TfLiteTensor& in = context->tensors[pair.in_index];
    TfLiteTensor& out = context->tensors[pair.out_index];
    TF_LITE_ENSURE(context, in.data.raw != nullptr && out.data.raw != nullptr);
    TF_LITE_ENSURE_EQ(context, in.bytes, out.bytes);
    // In-place binding (the normal LiteRT-LM setup) makes this a no-op.
    if (out.data.raw != in.data.raw) {
      std::memcpy(out.data.raw, in.data.raw, in.bytes);
    }
  }
  if (valid == 0) return kTfLiteOk;

  // Only the most recent min(valid, W) tokens survive in the ring.
  const int64_t first = std::max<int64_t>(0, valid - w);
  const int64_t count = valid - first;
  const int64_t slot0 = (start + first) % w;
  // The destination range [slot0, slot0 + count) wraps at most once.
  const int64_t run1 = std::min(count, w - slot0);
  const int64_t run2 = count - run1;

  // K: cache [H, W, D] <- new [H, T, D]; each token is a contiguous row.
  auto* k_out = reinterpret_cast<uint8_t*>(
      context->tensors[write.k_cache_out_index].data.raw);
  const uint8_t* k_new = write.k_new_buffer.data();
  for (int64_t h = 0; h < write.kv_heads; ++h) {
    const uint8_t* src = k_new + ((h * t + first) * d) * es;
    uint8_t* dst = k_out + (h * w) * d * es;
    std::memcpy(dst + slot0 * d * es, src, run1 * d * es);
    if (run2 > 0) std::memcpy(dst, src + run1 * d * es, run2 * d * es);
  }

  // V: cache [H, D, W] <- new [H, D, T]; tokens are contiguous per (h, d) row.
  auto* v_out = reinterpret_cast<uint8_t*>(
      context->tensors[write.v_cache_out_index].data.raw);
  const uint8_t* v_new = write.v_new_buffer.data();
  for (int64_t row = 0; row < write.kv_heads * d; ++row) {
    const uint8_t* src = v_new + (row * t + first) * es;
    uint8_t* dst = v_out + row * w * es;
    std::memcpy(dst + slot0 * es, src, run1 * es);
    if (run2 > 0) std::memcpy(dst, src + run1 * es, run2 * es);
  }
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
