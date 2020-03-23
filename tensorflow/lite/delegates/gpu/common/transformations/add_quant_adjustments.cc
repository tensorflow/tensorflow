/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/transformations/add_quant_adjustments.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

class AddQuantAdjustments : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
    if (node->operation.type ==
        ToString(OperationType::QUANTIZE_AND_DEQUANTIZE)) {
      return {TransformStatus::SKIPPED, ""};
    }

    bool transform_applied = false;
    auto node_outputs = graph->FindOutputs(node->id);
    for (auto output_value : node_outputs) {
      // Skip if quantization doesn't apply.
      if (!output_value->quant_params) continue;
      auto consumers = graph->FindConsumers(output_value->id);
      // No need to do anything if this isn't consumed by another node.
      if (consumers.empty()) {
        continue;
      }

      // Add a new QuantizeAndDequantize node.
      auto* quant_and_dequant_node = graph->NewNode();
      quant_and_dequant_node->operation.type =
          ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
      QuantizeAndDequantizeAttributes attr;
      attr.min = output_value->quant_params.value().min;
      attr.max = output_value->quant_params.value().max;
      attr.scale = output_value->quant_params.value().scale;
      quant_and_dequant_node->operation.attributes = attr;

      // Add one output Value for the new node.
      // The tensor information should rename the same.
      Value<TensorRef<BHWC>>* adjusted_value = graph->NewValue();
      adjusted_value->tensor = output_value->tensor;
      Status status =
          graph->SetProducer(quant_and_dequant_node->id, adjusted_value->id);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                "Could not create QuantizeAndDequantize node."};
      }

      // Replace output_value with adjusted_value on all consumers.
      for (auto& consumer : consumers) {
        status = graph->ReplaceInput(consumer->id, output_value->id,
                                     adjusted_value->id);
        if (!status.ok()) {
          return {TransformStatus::INVALID,
                  absl::StrCat(
                      "Failed to associate quant-adjusted value for consumer: ",
                      status.message())};
        }
      }

      // Add QuantizeAndDequantize node as a consumer of output_value.
      status = graph->AddConsumer(quant_and_dequant_node->id, output_value->id);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                absl::StrCat(
                    "Could not associate output to QuantizeAndDequantize: ",
                    status.message())};
      }

      // Remove quant params on output_value, to make the transformation
      // idempotent.
      output_value->quant_params.reset();
      transform_applied = true;
    }

    if (transform_applied) {
      return {TransformStatus::APPLIED, ""};
    }
    return {TransformStatus::SKIPPED, ""};
  }
};

std::unique_ptr<NodeTransformation> NewAddQuantAdjustments() {
  return absl::make_unique<AddQuantAdjustments>();
}

}  // namespace gpu
}  // namespace tflite
