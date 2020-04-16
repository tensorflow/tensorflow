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

#include "tensorflow/lite/delegates/gpu/common/object_reader.h"

#include <cstdint>
#include <unordered_map>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/utils.h"

namespace tflite {
namespace gpu {

absl::Status ObjectReader::ReadNonConstantTensor(
    TfLiteContext* context, std::unordered_map<int, Value*>* tensor_to_value,
    std::unordered_map<int, int>* quant_conversion_map, GraphFloat32* graph,
    uint32_t tensor_idx, Value** value) {
  if (tensor_idx >= context->tensors_size) {
    return absl::OutOfRangeError(
        absl::StrCat("ReadNonConstTensor: input tensor index: ", tensor_idx));
  }

  if (tensor_to_value->find(tensor_idx) == tensor_to_value->end()) {
    const TfLiteTensor& tflite_tensor = context->tensors[tensor_idx];
    if (tflite::IsConstantTensor(&tflite_tensor)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ReadNonConstantTensor: value is a constant tensor: ", tensor_idx));
    }

    if ((tflite_tensor.type == kTfLiteInt8 ||
         tflite_tensor.type == kTfLiteUInt8) &&
        quant_conversion_map) {
      // Quantized case
      if (quant_conversion_map->find(tensor_idx) ==
          quant_conversion_map->end()) {
        // Since the original tensor is fixed-point, add a new float tensor to
        // the TFLite graph to represent the dequantized data.
        int fp_tensor_index = 0;
        TfLiteTensor* fp_tflite_tensor;
        if (delegates::CreateNewTensorWithDifferentType(
                context, tensor_idx, kTfLiteFloat32, &fp_tflite_tensor,
                &fp_tensor_index) != kTfLiteOk) {
          return absl::InternalError("Could not add new tensor to graph");
        }
        // Remember this tensor for later.
        (*quant_conversion_map)[fp_tensor_index] = tensor_idx;
        (*quant_conversion_map)[tensor_idx] = fp_tensor_index;
        // Add a new GPU Value for the new dequantized floating-point tensor.
        Value* value = graph->NewValue();
        RETURN_IF_ERROR(
            ConvertTfLiteTensorToTensorRef(*fp_tflite_tensor, &value->tensor));
        value->tensor.ref = fp_tensor_index;
        value->quant_params.emplace();
        RETURN_IF_ERROR(
            PopulateQuantParams(tflite_tensor, &value->quant_params.value()));
        (*tensor_to_value)[fp_tensor_index] = value;
      }
      // We do not use the original tensor index as reference for the GPU
      // Value, instead pointing at the corresponding float version.
      tensor_idx = quant_conversion_map->at(tensor_idx);
    } else {
      // Floating-point case.
      Value* value = graph->NewValue();
      RETURN_IF_ERROR(
          ConvertTfLiteTensorToTensorRef(tflite_tensor, &value->tensor));
      value->tensor.ref = tensor_idx;
      (*tensor_to_value)[tensor_idx] = value;
    }
  }

  if (value) {
    *value = (*tensor_to_value)[tensor_idx];
  }
  return absl::OkStatus();
}

absl::Status ObjectReader::ReadValue(uint32_t idx, Value** value) {
  if (idx >= node_->inputs->size) {
    return absl::OutOfRangeError(
        absl::StrCat("ReadValue: input tensor index: ", idx));
  }
  return ReadValueByTensorIdx(node_->inputs->data[idx], value);
}

absl::Status ObjectReader::ReadValueByTensorIdx(uint32_t tensor_idx,
                                                Value** value) {
  // Constant tensors should be handled by ReadTensor.
  return ReadNonConstantTensor(context_, tensor_to_value_,
                               quant_conversion_map_, graph_, tensor_idx,
                               value);
}

int ObjectReader::GetNumberOfRuntimeInputs() const {
  return GetNumberOfRuntimeInputsForNode(context_, node_);
}

absl::Status ObjectReader::GetTensorDims(uint32_t idx,
                                         TfLiteIntArray* dimensions) const {
  if (idx >= node_->inputs->size) {
    return absl::OutOfRangeError(absl::StrCat("Input tensor index: ", idx));
  }
  const int tensor_idx = node_->inputs->data[idx];
  if (tensor_idx < 0 || tensor_idx > context_->tensors_size) {
    return absl::OutOfRangeError(absl::StrCat("Tensor index: ", tensor_idx));
  }
  const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
  *dimensions = *tflite_tensor.dims;
  return absl::OkStatus();
}

absl::Status ObjectReader::AddOutput(const Node* node, int id) {
  if (node_->outputs->size <= id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Data id ", id, " must be less than tflite node outputs size ",
        node_->outputs->size));
  }
  int output_tensor_idx = node_->outputs->data[id];
  Value* value;
  RETURN_IF_ERROR(ReadValueByTensorIdx(output_tensor_idx, &value));
  RETURN_IF_ERROR(graph_->SetProducer(node->id, value->id));
  return absl::OkStatus();
}

absl::Status ObjectReader::AddOutputs(const Node* node) {
  for (int i = 0; i < node_->outputs->size; ++i) {
    RETURN_IF_ERROR(AddOutput(node, i));
  }
  return absl::OkStatus();
}

absl::Status ObjectReader::AddInput(const Node* node, uint32_t idx) {
  Value* input;
  RETURN_IF_ERROR(ReadValue(idx, &input));
  return graph_->AddConsumer(node->id, input->id);
}

TfLiteTensor* ObjectReader::GetInputTensor(int index) const {
  return index >= 0 && index < node_->inputs->size
             ? context_->tensors + node_->inputs->data[index]
             : nullptr;
}

TfLiteTensor* ObjectReader::GetOutputTensor(int index) const {
  return index >= 0 && index < node_->outputs->size
             ? context_->tensors + node_->outputs->data[index]
             : nullptr;
}

absl::Status ObjectReader::VerifyInputsConstsOutputs(const TfLiteNode* node,
                                                     int runtime_inputs,
                                                     int const_inputs,
                                                     int outputs) {
  return CheckInputsConstsOutputs(context_, node, runtime_inputs, const_inputs,
                                  outputs);
}

}  // namespace gpu
}  // namespace tflite
