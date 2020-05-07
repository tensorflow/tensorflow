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

#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"

#include <set>
#include <string>
#include <unordered_map>

#include <fp16.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

TfLiteStatus GraphWithDequantPartitionHelper::Partition(
    std::set<std::string>* unsupported_nodes_info) {
  const auto status = GraphPartitionHelper::Partition(unsupported_nodes_info);
  // Clean up those partitions that have a single dequant op. NoteThose
  // removed dequant ops have to be reserved in the graph and should not be
  // delegated.
  RemoveSingleDequantNodePartitions();
  return status;
}

std::vector<int>
GraphWithDequantPartitionHelper::GetNodesOfFirstNLargestPartitions(int n) {
  // We first get partitions to reduce the number of nodes to be checked in
  // deciding which dequant ops could actually be replaced. And then we
  // remap input-tensor to dequant nodes' inputs and remove those
  // to-be-reserved dequant nodes.
  auto first_nps = GetFirstNLargestPartitions(n);
  std::vector<int> ops_to_replace;
  for (const auto p : first_nps) {
    auto nodes = p->nodes_to_replace;
    ops_to_replace.insert(ops_to_replace.end(), nodes->data,
                          nodes->data + nodes->size);
  }
  RemapInputTensors(ops_to_replace);
  RemoveReservedDequantsFromNodes(&ops_to_replace);
  return ops_to_replace;
}

bool GraphWithDequantPartitionHelper::IsNodeSupported(
    TfLiteContext* context, TfLiteNode* node, TfLiteRegistration* registration,
    int node_id, std::string* unsupported_details) {
  // If we need to handle dequant nodes, we have to remap input tensors of
  // this node if some of them come from a dequant node before testing if
  // the node is supported.
  std::vector<int> orig_inputs;
  if (RecordAndRemapInputTensors(registration->builtin_code, node_id, node,
                                 &orig_inputs)) {
    // We have a dequant op here. Note that we retrun an Ok status because a
    // dequant node is first added as supported. Later, this dequant node
    // will be removed if it has to be preserved in the graph which happens
    // when its immediate downstream nodes cannot be supported.
    return true;
  }
  const auto status = GraphPartitionHelper::IsNodeSupported(
      context, node, registration, node_id, unsupported_details);
  RestoreToOrigInputTensors(node, orig_inputs);
  return status;
}

bool GraphWithDequantPartitionHelper::RecordAndRemapInputTensors(
    int32_t op_code, int node_id, TfLiteNode* node,
    std::vector<int>* orig_inputs) {
  orig_inputs->clear();
  // Record the dequant node.
  if (op_code == kTfLiteBuiltinDequantize &&
      context_->tensors[node->inputs->data[0]].type ==
          TfLiteType::kTfLiteFloat16) {
    dequant_nodes_[node->outputs->data[0]] = node->inputs->data[0];
    return true;
  }
  // For a dequantize op, there's no need to remap its input tensors.
  if (dequant_nodes_.empty()) return false;
  RemapInputTensors(node, orig_inputs);
  return false;
}

void GraphWithDequantPartitionHelper::RestoreToOrigInputTensors(
    TfLiteNode* node, const std::vector<int>& orig_inputs) {
  if (node->inputs->size != orig_inputs.size()) return;
  for (int j = 0; j < node->inputs->size; ++j) {
    node->inputs->data[j] = orig_inputs[j];
  }
}

void GraphWithDequantPartitionHelper::RemapInputTensors(
    const std::vector<int>& nodes) const {
  for (int node_id : nodes) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    GetNodeAndRegistration(context_, node_id, &node, &registration)
        .IgnoreError();
    RemapInputTensors(node, nullptr /* orig_inputs*/);
  }
}

void GraphWithDequantPartitionHelper::RemoveSingleDequantNodePartitions() {
  auto it = partitions_.begin();
  while (it != partitions_.end()) {
    auto p = *it;
    if (p->nodes_to_replace->size != 1) {
      ++it;
      continue;
    }
    int node_id = p->nodes_to_replace->data[0];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    GetNodeAndRegistration(context_, node_id, &node, &registration)
        .IgnoreError();
    if (registration->builtin_code != kTfLiteBuiltinDequantize ||
        context_->tensors[node->inputs->data[0]].type !=
            TfLiteType::kTfLiteFloat16) {
      ++it;
      continue;
    }
    // Note such dequant nodes have to be preserved in the graph as dequant
    // ops are not actually supported in the GPU delegate.
    dequant_nodes_to_save_.insert(node_id);
    it = partitions_.erase(it);
  }
}

void GraphWithDequantPartitionHelper::RemoveReservedDequantsFromNodes(
    std::vector<int>* nodes) {
  if (dequant_nodes_to_save_.empty()) return;
  auto it = nodes->begin();
  while (it != nodes->end()) {
    if (dequant_nodes_to_save_.find(*it) == dequant_nodes_to_save_.end()) {
      ++it;
      continue;
    }
    it = nodes->erase(it);
  }
}

void GraphWithDequantPartitionHelper::RemapInputTensors(
    TfLiteNode* node, std::vector<int>* orig_inputs) const {
  TfLiteIntArray* inputs = node->inputs;
  auto inputs_view = TfLiteIntArrayView(inputs);
  // Prepopulate 'orig_inputs' first and clear it if there's no input from a
  // dequant op.
  if (orig_inputs) {
    orig_inputs->clear();
    orig_inputs->reserve(inputs->size);
    for (auto tid : inputs_view) {
      orig_inputs->push_back(tid);
    }
  }
  // Fix this node's inputs (i.e. prune out the preceding dequantize node) in
  // order to test if it is supported.
  bool is_remapped = false;
  for (int j = 0; j < inputs->size; ++j) {
    const int input_tid = inputs->data[j];
    const auto it = dequant_nodes_.find(input_tid);
    if (it != dequant_nodes_.end()) {
      inputs->data[j] = it->second;
      is_remapped = true;
    }
  }
  if (!is_remapped && orig_inputs) orig_inputs->clear();
}

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration) {
  if (context->GetNodeAndRegistration(context, node_id, tflite_node,
                                      registration) != kTfLiteOk) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Couldn't get node and registration info for op: ", node_id));
  }
  return absl::OkStatus();
}

DataType ToDataType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return DataType::FLOAT32;
    case kTfLiteInt32:
      return DataType::INT32;
    case kTfLiteInt64:
      return DataType::INT64;
    case kTfLiteInt8:
      return DataType::INT8;
    case kTfLiteUInt8:
      return DataType::UINT8;
    default:
      return DataType::UNKNOWN;
  }
}

absl::Status ExtractTensorShape(const TfLiteTensor& tflite_tensor, BHWC* bhwc) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  switch (dims->size) {
    case 1:
      *bhwc = BHWC(dims->data[0], 1, 1, 1);
      return absl::OkStatus();
    case 2:
      *bhwc = BHWC(dims->data[0], 1, 1, dims->data[1]);
      return absl::OkStatus();
    case 3:
      *bhwc = BHWC(dims->data[0], 1, dims->data[1], dims->data[2]);
      return absl::OkStatus();
    case 4:
      *bhwc = BHWC(dims->data[0], dims->data[1], dims->data[2], dims->data[3]);
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Tensor \"", tflite_tensor.name ? tflite_tensor.name : "nullptr",
          "\" has bad input dims size: ", dims->size, "."));
  }
}

absl::Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                            TensorRef<BHWC>* tensor_ref) {
  tensor_ref->type = ToDataType(tflite_tensor.type);
  return ExtractTensorShape(tflite_tensor, &tensor_ref->shape);
}

absl::Status PopulateQuantParams(const TfLiteTensor& tensor,
                                 QuantizationParams* quant_params) {
  const TfLiteQuantization& quant = tensor.quantization;
  if (quant.type != TfLiteQuantizationType::kTfLiteAffineQuantization) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tensor not quantized: ", std::string(tensor.name)));
  }
  const TfLiteAffineQuantization* params =
      static_cast<const TfLiteAffineQuantization*>(quant.params);
  if (params->scale->size > 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-constant per-channel quantized tensor: ",
                     std::string(tensor.name)));
  }
  const float scale = params->scale->data[0];
  const float zero_point = static_cast<float>(params->zero_point->data[0]);

  float qmin_value = 0;
  float qmax_value = 0;
  if (tensor.type == kTfLiteUInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<uint8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<uint8_t>::max());
  } else if (tensor.type == kTfLiteInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<int8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<int8_t>::max());
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Type invalid for quantized tensor: ", std::string(tensor.name)));
  }
  quant_params->min = scale * (static_cast<float>(qmin_value) - zero_point);
  quant_params->max = scale * (static_cast<float>(qmax_value) - zero_point);
  quant_params->scale = scale;

  return absl::OkStatus();
}

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node) {
  int number_of_runtime_inputs = 0;
  for (int i = 0; i < tflite_node->inputs->size; i++) {
    if (!IsConstantTensor(&context->tensors[tflite_node->inputs->data[i]])) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node) {
  return tflite_node->inputs->size -
         GetNumberOfRuntimeInputsForNode(context, tflite_node);
}

int GetNumberOfRuntimeOutputsForNode(const TfLiteContext* context,
                                     const TfLiteNode* tflite_node) {
  int number_of_runtime_outputs = 0;
  for (int i = 0; i < tflite_node->outputs->size; i++) {
    if (!IsConstantTensor(&context->tensors[tflite_node->outputs->data[i]])) {
      number_of_runtime_outputs++;
    }
  }
  return number_of_runtime_outputs;
}

absl::Status CheckInputsOutputs(const TfLiteContext* context,
                                const TfLiteNode* tflite_node,
                                int runtime_inputs, int outputs) {
  const int runtime_inputs_from_model =
      GetNumberOfRuntimeInputsForNode(context, tflite_node);
  if (runtime_inputs_from_model != runtime_inputs) {
    return absl::InternalError(absl::StrCat(
        "Expected ", runtime_inputs, " runtime input tensor(s), but node has ",
        runtime_inputs_from_model, " runtime input(s)."));
  }
  const int runtime_outputs =
      GetNumberOfRuntimeOutputsForNode(context, tflite_node);
  if (runtime_outputs != outputs) {
    return absl::InternalError(absl::StrCat("Expected ", outputs,
                                            " output tensor(s), but node has ",
                                            runtime_outputs, " output(s)."));
  }
  return absl::OkStatus();
}

absl::Status CheckInputsConstsOutputs(const TfLiteContext* context,
                                      const TfLiteNode* tflite_node,
                                      int runtime_inputs, int const_inputs,
                                      int outputs) {
  const int const_inputs_from_model =
      GetNumberOfConstInputsForNode(context, tflite_node);
  if (const_inputs_from_model != const_inputs) {
    return absl::InternalError(absl::StrCat(
        "Expected ", const_inputs, " const input tensor(s), but node has ",
        const_inputs_from_model, " const input(s)."));
  }
  return CheckInputsOutputs(context, tflite_node, runtime_inputs, outputs);
}

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst) {
  for (size_t i = 0; i < num_elements; i++) {
    *dst++ = fp16_ieee_to_fp32_value(*src++);
  }
}

template <>
absl::Status CreateVectorCopyData<float>(const TfLiteTensor& tensor,
                                         float* tensor_data) {
  switch (tensor.type) {
    case kTfLiteFloat32:
      std::memcpy(tensor_data, tensor.data.f, tensor.bytes);
      break;
    case kTfLiteFloat16:
      ConvertFloat16ToFloat32(
          NumElements(&tensor),
          reinterpret_cast<uint16_t const*>(tensor.data.f16), tensor_data);
      break;
    case kTfLiteInt8:
      DequantizeConstantTensor(tensor, tensor.data.int8, tensor_data);
      break;
    case kTfLiteUInt8:
      DequantizeConstantTensor(tensor, tensor.data.uint8, tensor_data);
      break;
    case kTfLiteInt32:
      DequantizeConstantTensor(tensor, tensor.data.i32, tensor_data);
      break;
    default:
      return absl::InvalidArgumentError(
          "Unsupported data type for float32 tensor");
  }
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Scalar* shape) {
  if (dimensions->size < 0) {
    return absl::InvalidArgumentError("Invalid Scalar dimensions");
  }
  for (int i = 0; i < dimensions->size; ++i) {
    if (dimensions->data[i] != 1) {
      return absl::InvalidArgumentError(
          "Dimension can not be reduced to scalar.");
    }
  }
  shape->v = 1;
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Linear* shape) {
  if (dimensions->size <= 0) {
    return absl::InvalidArgumentError("Dimension is empty.");
  }
  for (int i = 0; i < dimensions->size - 1; ++i) {
    if (dimensions->data[i] != 1) {
      return absl::InvalidArgumentError(
          "Dimension can not be reduced to linear.");
    }
  }
  shape->v = dimensions->data[dimensions->size - 1];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HWC* shape) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError("Dimensions are not HWC");
  }
  if (dimensions->data[0] != 1) {
    return absl::UnimplementedError("Batch size is not equal to 1.");
  }
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->c = dimensions->data[3];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HW* shape) {
  if (dimensions->size != 2) {
    return absl::InvalidArgumentError("Dimensions are not HW");
  }
  shape->h = dimensions->data[0];
  shape->w = dimensions->data[1];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, OHWI* shape) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Dimensions are not OHWI: ", dimensions->size));
  }
  shape->o = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->i = dimensions->data[3];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, BHWC* shape) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError("Dimensions are not BHWC");
  }
  shape->b = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->c = dimensions->data[3];
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
