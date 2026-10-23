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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_ATTENTION_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_ATTENTION_H_

#include <cstdint>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

// Finite mask fill value matching the exporter's reference decomposition
// (`torch.where(mask, logits, -10000.0)`). This must not be `-inf` or
// `std::numeric_limits<float>::lowest()`:
//   1. `odml.sdpa_transposed` applies bool masks additively
//      (`logits + (1 - mask) * fill`), where `lowest()` overflows float32 to
//      `-inf` for any negative logit; a fully-masked padding row then produces
//      `-inf - (-inf) = NaN` in softmax.
//   2. `odml.fused_sdpa_cache_update` replaces masked logits via
//      `min(logits, keep * (kKeepBound - fill) + fill)` and corrects for
//      sliced-out ring slots using `exp(fill - max_logit)`. With `-10000.0f`,
//      `exp(fill - max_logit)` underflows to 0 when any valid column is
//      present, and equals `exp(0) = 1` on fully-masked padding rows so they
//      average uniformly over all W + T columns like the reference.
inline constexpr float kAttentionMaskFillValue = -10000.0f;

bool IsFloatAttentionType(TfLiteType type);

// Q [..., Q_rows, D] x K [..., S, D] -> scores [..., Q_rows, S] (fp32).
// `k_tensor` provides the quantization parameters when K is quantized int8.
// When `use_decode_transpose` is true, computes (K @ Q^T)^T so the long key
// axis is the one being reduced over without packing the cache.
TfLiteStatus DefineAttentionScores(TfLiteContext* context,
                                   ynn_subgraph_t subgraph, uint32_t q_id,
                                   uint32_t k_id, const TfLiteTensor& k_tensor,
                                   bool use_decode_transpose,
                                   uint32_t* scores_id);

// Optional tanh soft-capping: cap * tanh(x / cap). No-op when `cap_val <= 0`.
TfLiteStatus ApplyLogitSoftcap(TfLiteContext* context, ynn_subgraph_t subgraph,
                               float cap_val, uint32_t* logits_id);

// P [..., Q_rows, S] x V -> out [..., Q_rows, D].
// `v_tensor` provides the quantization parameters when V is quantized int8.
// When `use_decode_transpose` is true, V is [..., D, S] and this computes
// (V @ P^T)^T. Otherwise V is transposed along `v_perm` (2 elements) before
// computing P @ V^T.
TfLiteStatus DefineAttentionWeightedValues(
    TfLiteContext* context, ynn_subgraph_t subgraph, uint32_t probs_id,
    uint32_t v_id, const TfLiteTensor& v_tensor, ynn_type output_type,
    bool use_decode_transpose, const int32_t v_perm[2], uint32_t* out_id);

bool IsSdpa(const TfLiteRegistration* registration, const TfLiteNode* node);

TfLiteStatus IsSdpaSupported(const TfLiteRegistration* registration,
                             const TfLiteNode* node, TfLiteContext* context);

TfLiteStatus DefineSdpaNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& tensor_to_value_id,
                            uint32_t& next_external_id,
                            std::vector<DummyInputInfo>& dummy_inputs,
                            const NodeInfo& node);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_ATTENTION_H_
