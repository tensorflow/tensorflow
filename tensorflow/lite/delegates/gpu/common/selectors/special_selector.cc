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

#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/flops_util.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/depthwise_conv_plus_1x1_conv.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/fc_fc_add.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {
absl::Status TryDepthwiseConvPlus1x1Conv(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  auto* dw_node = graph.GetNode(first_node_id);
  if (dw_node == nullptr) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  if (OperationTypeFromString(dw_node->operation.type) !=
      OperationType::DEPTHWISE_CONVOLUTION) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_inputs = graph.FindInputs(dw_node->id);
  if (dw_inputs.size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_outputs = graph.FindOutputs(dw_node->id);
  auto consumers = graph.FindConsumers(dw_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }

  Node* next_node;
  next_node = consumers[0];
  if (next_node == nullptr) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  if (consumed_nodes->find(next_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  Node* relu_node = nullptr;
  ReLUAttributes relu_attributes;
  if (OperationTypeFromString(next_node->operation.type) ==
      OperationType::RELU) {
    relu_node = next_node;
    auto relu_outputs = graph.FindOutputs(relu_node->id);
    consumers = graph.FindConsumers(relu_outputs[0]->id);
    if (consumers.size() != 1) {
      return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
    }
    relu_attributes =
        absl::any_cast<ReLUAttributes>(relu_node->operation.attributes);
    next_node = consumers[0];
  }

  auto* conv_node = next_node;
  if (OperationTypeFromString(conv_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  if (graph.FindInputs(conv_node->id).size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
      dw_node->operation.attributes);
  auto conv_attr =
      absl::any_cast<Convolution2DAttributes>(conv_node->operation.attributes);
  auto conv_outputs = graph.FindOutputs(conv_node->id);
  OperationDef op_def;
  op_def.precision = precision;
  auto it = tensor_descriptors.find(dw_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(conv_outputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.dst_tensors.push_back(it->second);
  }
  if (!IsDepthwiseConvPlus1x1ConvSupported(op_def, gpu_info, dw_attr, conv_attr,
                                           &conv_outputs[0]->tensor.shape)) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(dw_inputs, conv_outputs, gpu_subgraph);
  ReLUAttributes* relu_attr_ptr = relu_node ? &relu_attributes : nullptr;
  auto operation = CreateDepthwiseConvPlus1x1Conv(op_def, gpu_info, dw_attr,
                                                  conv_attr, relu_attr_ptr);
  *gpu_op = std::make_unique<GPUOperation>(std::move(operation));
  (*gpu_op)->flops_ = GetDepthwiseConvolutionFlops(dw_outputs[0]->tensor.shape,
                                                   dw_attr.weights.shape) +
                      GetConvolutionFlops(conv_outputs[0]->tensor.shape,
                                          conv_attr.weights.shape);
  std::string fused_nodes = std::to_string(dw_node->id);
  if (relu_node) {
    fused_nodes += " " + std::to_string(relu_node->id);
  }
  fused_nodes += " " + std::to_string(conv_node->id);
  gpu_subgraph->operations[0].name =
      "depthwise_conv_plus_1x1_conv " + fused_nodes;
  consumed_nodes->insert(dw_node->id);
  if (relu_node) {
    consumed_nodes->insert(relu_node->id);
  }
  consumed_nodes->insert(conv_node->id);
  return absl::OkStatus();
}

// fully connected + fully connected + add
absl::Status TryFCFCAdd(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  auto* fc0_node = graph.GetNode(first_node_id);
  if (fc0_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto first_op_type = OperationTypeFromString(fc0_node->operation.type);
  if (first_op_type != OperationType::FULLY_CONNECTED &&
      first_op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  const bool first_quantized =
      first_op_type == OperationType::FULLY_CONNECTED_INT8;
  auto fc0_inputs = graph.FindInputs(fc0_node->id);
  if (fc0_inputs.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc0_output_id = graph.FindOutputs(fc0_node->id)[0]->id;
  auto consumers = graph.FindConsumers(fc0_output_id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto* add_node = consumers[0];
  if (add_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (consumed_nodes->find(add_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (OperationTypeFromString(add_node->operation.type) != OperationType::ADD) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto add_inputs = graph.FindInputs(add_node->id);
  if (add_inputs.size() != 2) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc1_output_id = add_inputs[0]->id + add_inputs[1]->id - fc0_output_id;
  auto* fc1_node = graph.FindProducer(fc1_output_id);
  if (fc1_node == nullptr) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto second_op_type = OperationTypeFromString(fc1_node->operation.type);
  if (second_op_type != OperationType::FULLY_CONNECTED &&
      second_op_type != OperationType::FULLY_CONNECTED_INT8) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  const bool second_quantized =
      second_op_type == OperationType::FULLY_CONNECTED_INT8;
  const bool both_quantized = first_quantized && second_quantized;
  const bool both_not_quantized = !first_quantized && !second_quantized;
  if (!(both_quantized || both_not_quantized)) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  if (consumed_nodes->find(fc1_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto fc1_inputs = graph.FindInputs(fc1_node->id);
  if (fc1_inputs.size() != 1) {
    return absl::NotFoundError("FCFCAdd not suitable.");
  }
  auto add_outputs = graph.FindOutputs(add_node->id);

  OperationDef op_def;
  op_def.precision = precision;
  auto it = tensor_descriptors.find(fc0_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(fc1_inputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.src_tensors.push_back(it->second);
  }
  it = tensor_descriptors.find(add_outputs[0]->id);
  if (it != tensor_descriptors.end()) {
    op_def.dst_tensors.push_back(it->second);
  }

  for (int i = 0; i < fc1_inputs.size(); ++i) {
    fc0_inputs.push_back(fc1_inputs[i]);
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(fc0_inputs, add_outputs, gpu_subgraph);
  FCFCAdd fc;
  if (both_not_quantized) {
    auto fc0_attr = absl::any_cast<FullyConnectedAttributes>(
        fc0_node->operation.attributes);
    auto fc1_attr = absl::any_cast<FullyConnectedAttributes>(
        fc1_node->operation.attributes);
    if (fc0_attr.weights.shape.o != fc1_attr.weights.shape.o) {
      return absl::NotFoundError("FCFCAdd not suitable.");
    }
    fc = CreateFCFCAdd(gpu_info, op_def, fc0_attr, fc1_attr);
  } else {
    // both_quantized
    auto fc0_attr = absl::any_cast<FullyConnectedInt8Attributes>(
        fc0_node->operation.attributes);
    auto fc1_attr = absl::any_cast<FullyConnectedInt8Attributes>(
        fc1_node->operation.attributes);
    if (fc0_attr.weights.shape.o != fc1_attr.weights.shape.o) {
      return absl::NotFoundError("FCFCAdd not suitable.");
    }
    fc = CreateFCFCAdd(gpu_info, op_def, fc0_attr, fc1_attr);
  }
  *gpu_op = std::make_unique<FCFCAdd>(std::move(fc));
  const std::string fused_nodes = std::to_string(fc0_node->id) + " " +
                                  std::to_string(fc1_node->id) + " " +
                                  std::to_string(add_node->id);
  gpu_subgraph->operations[0].name =
      "fully_connected_x2_and_add " + fused_nodes;
  consumed_nodes->insert(fc0_node->id);
  consumed_nodes->insert(fc1_node->id);
  consumed_nodes->insert(add_node->id);
  return absl::OkStatus();
}
}  // namespace

absl::Status GPUSubgraphFromGraph(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  if ((gpu_info.IsAdreno() || gpu_info.IsNvidia() || gpu_info.IsMali() ||
       gpu_info.IsApple() || gpu_info.IsAMD()) &&
      TryDepthwiseConvPlus1x1Conv(gpu_info, precision, graph, first_node_id,
                                  tensor_descriptors, consumed_nodes,
                                  gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if ((gpu_info.IsIntel() || gpu_info.IsNvidia() || gpu_info.IsAMD()) &&
      TryFCFCAdd(gpu_info, precision, graph, first_node_id, tensor_descriptors,
                 consumed_nodes, gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (TryFusedPointwiseConv(graph, first_node_id, precision, tensor_descriptors,
                            consumed_nodes, gpu_subgraph)
          .ok()) {
    gpu_subgraph->operations[0].name = "slice_mul_mean_concat";
    return absl::OkStatus();
  }
  return absl::NotFoundError("No special combination.");
}

}  // namespace gpu
}  // namespace tflite
