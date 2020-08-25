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

#include "tensorflow/lite/delegates/gpu/cl/selectors/special_selector.h"

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/special/depthwise_conv_plus_1x1_conv.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
absl::Status TryDepthwiseConvPlus1x1Conv(
    CalculationsPrecision precision, const GraphFloat32& graph,
    NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  auto* dw_node = graph.GetNode(first_node_id);
  if (OperationTypeFromString(dw_node->operation.type) !=
      OperationType::DEPTHWISE_CONVOLUTION) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto dw_outputs = graph.FindOutputs(dw_node->id);
  auto consumers = graph.FindConsumers(dw_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  auto* conv_node = consumers[0];
  if (consumed_nodes->find(conv_node->id) != consumed_nodes->end()) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
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
  auto dw_inputs = graph.FindInputs(dw_node->id);
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
  if (!IsDepthwiseConvPlus1x1ConvSupported(op_def, dw_attr, conv_attr)) {
    return absl::NotFoundError("DepthwiseConvPlus1x1Conv not suitable.");
  }
  std::unique_ptr<GPUOperation>* gpu_op =
      InitSingleOpSubgraph(dw_inputs, conv_outputs, gpu_subgraph);
  auto operation = CreateDepthwiseConvPlus1x1Conv(op_def, dw_attr, conv_attr);
  *gpu_op = absl::make_unique<GPUOperation>(std::move(operation));
  consumed_nodes->insert(dw_node->id);
  consumed_nodes->insert(conv_node->id);
  return absl::OkStatus();
}
}  // namespace

absl::Status GPUSubgraphFromGraph(
    const DeviceInfo& device_info, CalculationsPrecision precision,
    const GraphFloat32& graph, NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  if (!device_info.IsNvidia()) {
    return absl::NotFoundError(
        "Experimental feature, enabled for NVidia only, but device is not "
        "nvidia gpu.");
  }
  if (TryDepthwiseConvPlus1x1Conv(precision, graph, first_node_id,
                                  tensor_descriptors, consumed_nodes,
                                  gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  return absl::NotFoundError("No special combination.");
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
