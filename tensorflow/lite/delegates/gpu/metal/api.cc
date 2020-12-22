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

#include "tensorflow/lite/delegates/gpu/metal/api.h"

#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/metal/selectors/subgraph.h"

namespace tflite {
namespace gpu {
namespace metal {

absl::Status Compile(const GraphFloat32& graph, const GpuInfo& gpu_info,
                     CalculationsPrecision precision,
                     CompiledModel* compiled_model) {
  if (!IsBatchMatchesForAllValues(graph)) {
    return absl::InvalidArgumentError(
        "Only identical batch dimension is supported");
  }
  int last_value_id = 0;
  for (const auto& value : graph.values()) {
    compiled_model->tensor_shapes[value->id] = value->tensor.shape;
    last_value_id = std::max(last_value_id, static_cast<int>(value->id));
  }
  int node_linear_id = 0;
  for (const auto& node : graph.nodes()) {
    auto inputs = graph.FindInputs(node->id);
    auto outputs = graph.FindOutputs(node->id);
    DataType data_type = DeduceDataTypeFromPrecision(precision);
    TensorDescriptor tensor_descriptor =
        TensorDescriptor{data_type, TensorStorageType::BUFFER, Layout::HWC};
    OperationDef op_def;
    op_def.precision = precision;
    for (int j = 0; j < inputs.size(); ++j) {
      op_def.src_tensors.push_back(tensor_descriptor);
    }
    for (int j = 0; j < outputs.size(); ++j) {
      op_def.dst_tensors.push_back(tensor_descriptor);
    }
    GPUOperationsSubgraph gpu_subgraph;
    RETURN_IF_ERROR(GPUOperationFromNode(gpu_info, op_def, inputs, outputs,
                                         *node, &gpu_subgraph));
    std::map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      last_value_id++;
      compiled_model->tensor_shapes[last_value_id] = t.first;
      mapping_to_global_ids[j] = last_value_id;
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      NodeDescriptor metal_node;
      metal_node.task = std::move(gpu_op.operation);
      metal_node.src_tensors_ids.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          metal_node.src_tensors_ids[j] = id;
        } else {
          metal_node.src_tensors_ids[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      metal_node.dst_tensors_ids.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          metal_node.dst_tensors_ids[j] = id;
        } else {
          metal_node.dst_tensors_ids[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      metal_node.description =
          node->operation.type + " " + std::to_string(node->id);
      metal_node.id = node_linear_id++;
      compiled_model->nodes.push_back(std::move(metal_node));
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
