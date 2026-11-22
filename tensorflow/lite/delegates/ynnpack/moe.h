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

#include <cstdint>
#include <memory>
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
                            const TfLiteNode* node, TfLiteContext* context);

// The host-side bucketing of the tokens of an MoE node by expert, which lets
// each expert compute one contiguous run of the tokens routed to it.
//
// The buckets are computed on every invocation from the expert indices, so
// this is only possible when the expert indices are available on the host
// when the runtime is invoked, i.e. when they are an input of the partition.
// The bucketed tokens are passed to the runtime in two external inputs:
//  - `slot_token` [E, C] int32: the token each slot of each expert computes,
//  - `token_slot` [B, N, K] int32: the slot, in [0, E * C), that computes
//    each (token, selected expert) pair,
// where C, the capacity, is the largest number of (token, selected expert)
// pairs routed to one expert.
struct MoeDispatch {
  // The expert indices the buckets are computed from.
  int expert_indices_tensor_index = -1;
  int num_experts = 0;

  // The two external inputs, see above.
  uint32_t slot_token_id = YNN_INVALID_VALUE_ID;
  uint32_t token_slot_id = YNN_INVALID_VALUE_ID;

  // The capacity the external values are currently shaped for.
  size_t capacity = 1;
  // The data of the two external inputs.
  std::vector<int32_t> slot_token;
  std::vector<int32_t> token_slot;
  // Per-expert counts, scratch for `SetupMoeDispatch`; a member so it is
  // not reallocated on every invocation.
  std::vector<int32_t> count;
};

// Returns true if the tokens of the MoE `node` should be bucketed by expert
// on the host (see `MoeDispatch`) for the current shape of its inputs. This
// requires `options.dispatch_moe_on_host` to be enabled and the expert indices
// to be an input of the partition, `partition_inputs`. Either choice computes
// the same result.
bool ShouldDispatchMoeOnHost(TfLiteContext* context, const NodeInfo& node,
                             const std::vector<int>& partition_inputs,
                             const TfLiteYNNPackDelegateOptions& options);

// The number of external values `DefineMoeNode` defines with a `dispatch`.
constexpr int kNumMoeDispatchExternalValues = 2;

// Defines the vectorized MoE operation in the YNNPACK subgraph.
//
// If `dispatch` is not null, the node consumes the tokens bucketed by expert:
// this defines the external values of `dispatch` with the next
// `kNumMoeDispatchExternalValues` ids from `next_external_id`, and fills in
// its other fields; the runtime then needs `SetMoeDispatchShapes` when
// prepared, and `SetupMoeDispatch` and `SetMoeDispatchData` before every
// invocation. Otherwise, the node gathers the selected experts' weights for
// each token.
TfLiteStatus DefineMoeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           uint32_t& next_external_id, MoeDispatch* dispatch,
                           const NodeInfo& node);

// Sets the shapes of the external values of `dispatch` in `runtime`, for the
// current shape of the expert indices and capacity. The runtime has to be
// reshaped afterwards.
TfLiteStatus SetMoeDispatchShapes(TfLiteContext* context, ynn_runtime_t runtime,
                                  const MoeDispatch& dispatch);

// Buckets the tokens by expert for the current expert indices. If the
// capacity changes, this sets the new shapes of the external values of
// `dispatch` in `runtime` and sets `*reshape_required`; the runtime then has
// to be reshaped before it is invoked. Fails if an expert index is out of
// range. A caller that fails between this call and the reshape leaves
// `dispatch` consistent: `capacity`, the external shapes and the two
// vectors are set together, so the next invocation computes with matching
// values; only the reshape's checks of that capacity are skipped.
TfLiteStatus SetupMoeDispatch(TfLiteContext* context, ynn_runtime_t runtime,
                              MoeDispatch& dispatch, bool* reshape_required);

// Sets the data of the external values of `dispatch` in `runtime`. This has
// to be called after `SetupMoeDispatch`, and after the runtime is reshaped.
TfLiteStatus SetMoeDispatchData(TfLiteContext* context, ynn_runtime_t runtime,
                                MoeDispatch& dispatch);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_MOE_H_
