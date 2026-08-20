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

namespace tflite {
namespace ynnpack {

struct MoeInfo {
  int expert_indices_tensor_index = 0;
  int tokens_tensor_index = 0;
  int num_experts = 0;
  int k = 0;
  std::vector<uint32_t> dynamic_routing_val_ids;
  std::vector<std::vector<int32_t>> dynamic_routing_bufs;
};

// Returns true if the given registration/node represents an MoE composite op.
bool IsMoe(const TfLiteRegistration* registration, const TfLiteNode* node);
bool IsMoe(TfLiteContext* context, int node_index);

// Validates whether the MoE composite node is supported by the delegate.
TfLiteStatus IsMoeSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context);

// Defines the unrolled E-way MoE composite operation in the YNNPACK subgraph.
TfLiteStatus DefineMoeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           uint32_t& next_external_id, const NodeInfo& node,
                           std::vector<std::unique_ptr<float[]>>* temp_buffers,
                           MoeInfo* moe_info, bool static_shape);

// Initializes external value shapes for MoE nodes after runtime creation.
TfLiteStatus InitMoeRuntime(TfLiteContext* context, ynn_runtime_t runtime,
                            const std::vector<MoeInfo>& moe_infos);

// Computes dynamic routing and updates external value shapes/data in Eval().
TfLiteStatus EvalMoeNodes(TfLiteContext* context, ynn_runtime_t runtime,
                          std::vector<MoeInfo>& moe_infos);

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_MOE_H_
