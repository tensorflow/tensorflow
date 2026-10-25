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

// Fused sliding-window attention + ring-buffer KV cache update
// (`odml.fused_sdpa_cache_update`).
//
// Signature (all float tensors share the cache dtype):
//   inputs:
//     0 query        [1, Hkv, G*T, D]  pre-scaled, GQA heads packed g-major.
//     1 key_cache    [1, Hkv, W, D]    ring buffer.
//     2 value_cache  [1, Hkv, D, W]    ring buffer (transposed).
//     3 key_new      [1, Hkv, T, D]
//     4 value_new    [1, Hkv, D, T]
//     5 mask         [1, 1, T, W + T]  bool (true = attend) or additive float.
//     6 param        int32, >= 2 elements: [start, end, ...]. The number of
//                    valid new tokens is `end - start`.
//   outputs:
//     0 attention    [1, Hkv, G*T, D]
//     1 key_cache'   (only if update_cache) same shape as input 1.
//     2 value_cache' (only if update_cache) same shape as input 2.
//
// Attention is computed over [key_cache | key_new] (all W + T columns; the
// mask alone decides validity), i.e. against the cache contents *before* this
// step's write. Bool masks replace masked logits with -10000, matching the
// reference decomposition, so rows without any valid column (padding) produce
// exactly the reference output. The cache write (new token i -> slot
// (start + i) % W for the last min(valid, W) valid tokens) is performed via
// memcpy in `ApplyFusedCacheWrite` after `ynn_invoke_runtime` has finished, so
// every read of the old cache happens first.

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_FUSED_SDPA_CACHE_UPDATE_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_FUSED_SDPA_CACHE_UPDATE_H_

#include <cstdint>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"

namespace tflite {
namespace ynnpack {

inline constexpr char kFusedSdpaCacheUpdateName[] =
    "odml.fused_sdpa_cache_update";

// Post-invoke cache-write state for one fused node: where the YNNPACK subgraph
// stages the new K/V rows (`k_new_buffer` / `v_new_buffer`) and which TFLite
// cache tensors `ApplyFusedCacheWrite` copies them into after
// `ynn_invoke_runtime` finishes.
struct FusedCacheWrite {
  int param_tensor_index = -1;
  int k_cache_in_index = -1;
  int v_cache_in_index = -1;
  int k_cache_out_index = -1;
  int v_cache_out_index = -1;
  uint32_t k_new_ext_id = YNN_INVALID_VALUE_ID;
  uint32_t v_new_ext_id = YNN_INVALID_VALUE_ID;
  std::vector<uint8_t> k_new_buffer;
  std::vector<uint8_t> v_new_buffer;
  size_t element_size = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int cache_size = 0;  // W
  int new_len = 0;     // T
};

bool IsFusedSdpaCacheUpdate(const TfLiteRegistration* registration,
                            const TfLiteNode* node);

TfLiteStatus IsFusedSdpaCacheUpdateSupported(
    const TfLiteRegistration* registration, const TfLiteNode* node,
    TfLiteContext* context);

// Number of extra external value ids (K/V staging outputs and the two
// ring-fill length dummies) the node may allocate.
inline constexpr int kFusedSdpaCacheUpdateMaxExternalIds = 4;

// Defines the attention part of the node in `subgraph`. If the node updates
// the cache, `cache_writes` is appended with the post-invoke write descriptor.
TfLiteStatus DefineFusedSdpaCacheUpdateNode(
    TfLiteContext* context, ynn_subgraph_t subgraph,
    TensorToValueIdMap& tensor_to_value_id, uint32_t& next_external_id,
    std::vector<DummyInputInfo>& dummy_inputs, const NodeInfo& node,
    std::vector<FusedCacheWrite>& cache_writes);

// Performs the ring-buffer write after the YNNPACK runtime has run.
TfLiteStatus ApplyFusedCacheWrite(TfLiteContext* context,
                                  const FusedCacheWrite& write);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_FUSED_SDPA_CACHE_UPDATE_H_
