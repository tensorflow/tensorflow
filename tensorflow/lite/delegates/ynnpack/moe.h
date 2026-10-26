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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_MOE_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_MOE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"

namespace tflite {
namespace ynnpack {

// Returns true if the given registration/node represents an `odml.moe_experts`
// composite op.
bool IsMoe(const TfLiteRegistration* registration, const TfLiteNode* node);

// Validates whether the MoE node is supported by the delegate.
TfLiteStatus IsMoeSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context,
                            const TfLiteYNNPackDelegateOptions& options);

// Host-side routing buffers for an MoE node that partitions `(token, expert)`
// selections into `E` contiguous expert groups for `ynn_define_ragged_dot`:
//  - `slot_token` (`[R, 1]` int32): token index `t in [0, T)` for each slot.
//  - `slot_expert` (`[R, 1]` int32): expert index `e in [0, E)` for each slot.
//  - `token_slot` (`[T, K, 1]` int32): slot index in `[0, R)` for each
//    `(t, k)`.
struct MoeDispatch {
  int expert_indices_tensor_index = -1;
  // External partition input (or constant) backing
  // `expert_indices_tensor_index` after tracing through partition-internal
  // shape-only ops (`RESHAPE`, `EXPAND_DIMS`). TFLite does not allocate buffers
  // for delegate-internal tensors, so `SetupMoeDispatch` reads the `int32`
  // indices from this tensor.
  int expert_indices_data_tensor_index = -1;
  // Indices into `nodes_info_` of the partition-internal `RESHAPE` /
  // `EXPAND_DIMS` ops in forward order from `expert_indices_data_tensor_index`
  // to `expert_indices_tensor_index`. Replayed against
  // `expert_indices_data_tensor_index` so `Prepare` and
  // `BuildSubgraphAndRuntime` resolve the live `[B, N, K]` shape without
  // reading unresized internal `TfLiteTensor` dims.
  std::vector<size_t> expert_indices_view_nodes;
  int external_tokens_tensor_index = -1;
  int external_routing_weights_tensor_index = -1;
  int batch_size = 0;
  int top_k = 0;
  // Non-zero when the op fixes `top_k` via attributes (e.g. `CUSTOM` `"moe"`).
  int fixed_top_k = 0;
  int num_experts = 0;

  uint32_t slot_token_id = YNN_INVALID_VALUE_ID;
  uint32_t slot_expert_id = YNN_INVALID_VALUE_ID;
  uint32_t token_slot_id = YNN_INVALID_VALUE_ID;

  std::vector<int32_t> slot_token;
  std::vector<int32_t> slot_expert;
  std::vector<int32_t> group_offsets;
  std::vector<int32_t> token_slot;
};

// Maximum number of external values `DefineMoeNode` defines in `dispatch`.
constexpr int kNumMoeDispatchExternalValues = 3;

// Defines the MoE operation in the YNNPACK subgraph.
TfLiteStatus DefineMoeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           uint32_t& next_external_id, MoeDispatch& dispatch,
                           const NodeInfo& node);

// Sets the external value shapes of `dispatch` in `runtime` for the resolved
// `[B, N, K]` `expert_indices_dims`, updating `dispatch.top_k`.
TfLiteStatus SetMoeDispatchShapes(TfLiteContext* context, ynn_runtime_t runtime,
                                  MoeDispatch& dispatch,
                                  const size_t (&expert_indices_dims)[3]);

// Partitions the `(token, expert)` selections for the current expert indices
// and binds `dispatch`'s buffers in `runtime`.
TfLiteStatus SetupMoeDispatch(TfLiteContext* context, ynn_runtime_t runtime,
                              MoeDispatch& dispatch);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_MOE_H_
