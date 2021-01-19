/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"

#include <map>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/metal/selectors/subgraph.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

bool HasIntersection(const std::vector<ValueId>& vec_ids,
                     const std::set<ValueId>& ids) {
  for (ValueId id : vec_ids) {
    if (ids.find(id) != ids.end()) {
      return true;
    }
  }
  return false;
}

bool IsReady(const std::set<ValueId>& ready_tensors, const MetalNode& node) {
  for (const ValueId in_id : node.inputs) {
    if (ready_tensors.find(in_id) == ready_tensors.end()) {
      return false;
    }
  }
  return true;
}

void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    // initializing start index(.x) and end index(.y)
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    // updating end index(.y)
    (*usage_records)[id].y = task_index;
  }
}

// Generic add is add that have several runtime inputs and they are not
// broadcasted, i.e. pointwise add for N tensors where N > 1.
bool IsGenericAdd(const Node& node, const std::vector<Value*>& inputs,
                  const std::vector<Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const OperationType op_type = OperationTypeFromString(node.operation.type);
  if (op_type != OperationType::ADD) {
    return false;
  }

  const auto dst_shape = outputs[0]->tensor.shape;
  for (int i = 0; i < inputs.size(); ++i) {
    const auto src_shape = inputs[i]->tensor.shape;
    if (dst_shape.b != src_shape.b && src_shape.b == 1) {
      return false;
    }
    if (dst_shape.h != src_shape.h && src_shape.h == 1) {
      return false;
    }
    if (dst_shape.w != src_shape.w && src_shape.w == 1) {
      return false;
    }
    if (dst_shape.c != src_shape.c && src_shape.c == 1) {
      return false;
    }
  }
  return true;
}

absl::Status MergeNodes(MetalNode* src, MetalNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->task.AddTask(&src->task);
}
}  // namespace

absl::Status InferenceContext::InitFromGraph(
    const CreateInferenceInfo& create_info, const GraphFloat32& graph,
    id<MTLDevice> device_id) {
  const auto inputs = graph.inputs();
  for (const auto& input : inputs) {
    input_ids_.push_back(input->id);
  }

  const auto outputs = graph.outputs();
  for (const auto& output : outputs) {
    output_ids_.push_back(output->id);
  }
  precision_ = create_info.precision;

  MetalDevice metal_device(device_id);
  ReserveGraphTensors(create_info, metal_device.GetInfo(), graph);
  RETURN_IF_ERROR(
      Compile(graph, metal_device.GetInfo(), create_info.precision));
  RETURN_IF_ERROR(Merge());
  RETURN_IF_ERROR(CompileOperations(&metal_device));
  RETURN_IF_ERROR(AllocateTensors(&metal_device));
  BindTensorsToOperations();
  RETURN_IF_ERROR(UpdateParams(metal_device.GetInfo()));
  RETURN_IF_ERROR(Tune(TuningType::kFast, &metal_device));
  return absl::OkStatus();
}

void InferenceContext::ReserveGraphTensors(
    const CreateInferenceInfo& create_info, const GpuInfo& gpu_info,
    const GraphFloat32& graph) {
  ValueId max_id = 0;
  auto tensors = graph.values();
  auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
  for (auto& t : tensors) {
    TensorStorageType storage_type = create_info.storage_type;
    const auto shape = graph.GetValue(t->id)->tensor.shape;
    Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
    // Temporary disabled because no support of SINGLE_TEXTURE_2D in Metal
    // Metal supports only BUFFER storage type currently
    // if (graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id)) {
    //   if (false && shape.c < 4 &&
    //       CanCreateTensorWithShape(
    //           gpu_info, shape,
    //           TensorDescriptor{data_type,
    //           TensorStorageType::SINGLE_TEXTURE_2D,
    //                            layout})) {
    //     storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
    //   }
    // }
    storage_type =
        SelectBestStorageType(gpu_info, shape, storage_type, data_type, layout);
    tensor_reserver_.Add(
        t->id, {shape, TensorDescriptor{data_type, storage_type, layout}});
    max_id = std::max(max_id, t->id);
  }
  tensor_reserver_.SetNext(max_id + 1);
}

absl::Status InferenceContext::Compile(const GraphFloat32& graph,
                                       const GpuInfo& gpu_info,
                                       CalculationsPrecision precision) {
  if (!IsBatchMatchesForAllValues(graph)) {
    return absl::InvalidArgumentError(
        "Only identical batch dimension is supported");
  }
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input_id : input_ids_) {
    tensor_usages[input_id] = -1;  // so as inputs "updated" before operation 0,
                                   // we will mark them with -1
  }
  std::vector<Node*> graph_nodes = graph.nodes();
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    auto inputs = graph.FindInputs(node.id);
    auto outputs = graph.FindOutputs(node.id);
    // Reordering of input ids and updating of temporary tensors_usage struct.
    // This stage is necessary because we are building OperationDef that rely
    // on order of input ids. But we also should have input id on first
    // position that potentially can be "linking" tensor and as result
    // eliminated(unused) We apply it only for ADD operation, because of ADD
    // associativity and ADD can be linked. In current approach "linking"
    // tensor can be only latest written tensor(during linear order of
    // execution) among input tensors.
    if (IsGenericAdd(node, inputs, outputs)) {
      int latest_written_tensor_index = 0;
      int last_usage = tensor_usages[inputs[0]->id];
      for (int j = 1; j < inputs.size(); ++j) {
        if (tensor_usages[inputs[j]->id] > last_usage) {
          last_usage = tensor_usages[inputs[j]->id];
          latest_written_tensor_index = j;
        }
      }
      std::swap(inputs[0], inputs[latest_written_tensor_index]);
    }
    OperationDef op_def;
    op_def.precision = precision;
    for (int j = 0; j < inputs.size(); ++j) {
      op_def.src_tensors.push_back(
          tensor_reserver_.Get(inputs[j]->id).descriptor);
    }
    for (int j = 0; j < outputs.size(); ++j) {
      op_def.dst_tensors.push_back(
          tensor_reserver_.Get(outputs[j]->id).descriptor);
    }
    GPUOperationsSubgraph gpu_subgraph;
    RETURN_IF_ERROR(GPUOperationFromNode(gpu_info, op_def, inputs, outputs,
                                         node, &gpu_subgraph));
    std::map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      auto global_id = tensor_reserver_.Add({t.first, t.second});
      mapping_to_global_ids[j] = global_id;
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      MetalNode metal_node;
      if (gpu_op.task_desc) {
        metal_node.task.Init(std::move(gpu_op.task_desc));
      } else {
        metal_node.task.Init(std::move(gpu_op.operation));
      }
      metal_node.inputs.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          metal_node.inputs[j] = id;
        } else {
          metal_node.inputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      metal_node.outputs.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          metal_node.outputs[j] = id;
          tensor_usages[id] = i;
        } else {
          metal_node.outputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      metal_node.name = node.operation.type + " " + std::to_string(node.id);
      nodes_.push_back(std::move(metal_node));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Merge() {
  std::set<ValueId> ready_tensors;
  for (const auto& input_id : input_ids_) {
    ready_tensors.insert(input_id);
  }
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    for (const auto& out_id : node.outputs) {
      ready_tensors.insert(out_id);
    }
    if (node.outputs.size() != 1) {
      continue;
    }
    std::vector<int> next_nodes;
    int link_index = 0;
    for (int j = i + 1; j < nodes_.size(); ++j) {
      for (int k = 0; k < nodes_[j].inputs.size(); ++k) {
        if (nodes_[j].inputs[k] == node.outputs[0]) {
          next_nodes.push_back(j);
          link_index = k;
        }
      }
    }
    if (next_nodes.size() != 1 || link_index != 0) {
      continue;
    }
    auto& linkable_node = nodes_[next_nodes[0]];
    if (!linkable_node.task.IsLinkable() || linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node)) {
      continue;
    }
    const auto& original_dst_def = node.task.GetDefinition().dst_tensors[0];
    const auto& link_dst_def =
        linkable_node.task.GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def) {
      continue;
    }
    RETURN_IF_ERROR(MergeNodes(&linkable_node, &node));
    nodes_.erase(nodes_.begin() + next_nodes[0]);
    i -= 1;
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::CompileOperations(MetalDevice* device) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.task.Compile(device));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateTensors(MetalDevice* device) {
  std::set<ValueId> preallocated_ids;
  for (auto tensor_id : input_ids_) {
    preallocated_ids.insert(tensor_id);
  }
  for (const auto& outputId : output_ids_) {
    preallocated_ids.insert(outputId);
  }
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    if (HasIntersection(node.inputs, preallocated_ids) ||
        HasIntersection(node.outputs, preallocated_ids)) {
      task_ids_with_preallocated_tensors_.push_back(i);
    }
  }

  const bool f32_storage = precision_ == CalculationsPrecision::F32;
  for (auto& tensor_id : preallocated_ids) {
    const auto& t = tensor_reserver_.Get(tensor_id);
    preallocated_tensors_[tensor_id] =
        CreateSharedBufferTensor(nil, t.shape, t.descriptor);
  }

  RETURN_IF_ERROR(AllocateMemoryForBuffers(device));
  return absl::OkStatus();
}

MetalSpatialTensor* InferenceContext::GetTensor(ValueId tensor_id) {
  if (preallocated_tensors_.find(tensor_id) != preallocated_tensors_.end()) {
    return &preallocated_tensors_[tensor_id];
  } else if (graph_ids_to_shared_buffer_tensors_.find(tensor_id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_
        [graph_ids_to_shared_buffer_tensors_[tensor_id]];
  }
  return nullptr;
}

void InferenceContext::BindTensorsToOperations() {
  for (auto& node : nodes_) {
    const auto& src_ids = node.inputs;
    for (int i = 0; i < src_ids.size(); ++i) {
      node.task.SetSrcTensor(GetTensor(src_ids[i]), i);
    }
    const auto& dst_ids = node.outputs;
    for (int i = 0; i < dst_ids.size(); ++i) {
      node.task.SetDstTensor(GetTensor(dst_ids[i]), i);
    }
  }
}

absl::Status InferenceContext::UpdateParams(const GpuInfo& gpu_info) {
  for (auto& node : nodes_) {
    std::vector<BHWC> src_shapes;
    std::vector<BHWC> dst_shapes;
    for (const auto& in_id : node.inputs) {
      src_shapes.push_back(tensor_reserver_.Get(in_id).shape);
    }
    for (const auto& out_id : node.outputs) {
      dst_shapes.push_back(tensor_reserver_.Get(out_id).shape);
    }
    RETURN_IF_ERROR(node.task.UpdateParams(gpu_info, src_shapes, dst_shapes));
  }
  return absl::OkStatus();
}

void InferenceContext::GetUsages(std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (preallocated_tensors_.find(in_id) == preallocated_tensors_.end()) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < nodes_.size(); ++op_index) {
    for (auto& tensor_id : nodes_[op_index].inputs) {
      if (preallocated_tensors_.find(tensor_id) ==
          preallocated_tensors_.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
    for (auto& tensor_id : nodes_[op_index].outputs) {
      if (preallocated_tensors_.find(tensor_id) ==
          preallocated_tensors_.end()) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (preallocated_tensors_.find(out_id) == preallocated_tensors_.end()) {
      AddUsage(out_id, nodes_.size(), usages);
    }
  }
}

absl::Status InferenceContext::AllocateMemoryForBuffers(MetalDevice* device) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(&buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& shape = tensor_reserver_.Get(usage.first).shape;
    const size_t buffer_size =
        shape.b * shape.w * shape.h * AlignByN(shape.c, 4);
    graph_ids_to_shared_buffer_tensors_[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  const bool f32_storage = precision_ == CalculationsPrecision::F32;
  size_t dataTypeSize = f32_storage ? sizeof(float) : sizeof(HalfBits);
  shared_buffers_.resize(buffer_assignment.object_sizes.size());
  for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
    // Initialize metal buffer
    NSUInteger bufferSize = dataTypeSize * buffer_assignment.object_sizes[i];

    if (bufferSize > device->GetInfo().GetMaxBufferSize()) {
      std::string error("Tensor id: ");
      error += std::to_string(buffer_assignment.object_ids[i]) +
               " with size: " + std::to_string(bufferSize) +
               " exceeds MTLDevice maxBufferLength: " +
               std::to_string(device->GetInfo().GetMaxBufferSize());
      return absl::ResourceExhaustedError(error);
    }

    shared_buffers_[i] =
        [device->device() newBufferWithLength:bufferSize
                                      options:MTLResourceStorageModeShared];
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  TensorDescriptor descriptor;
  descriptor.storage_type = TensorStorageType::BUFFER;
  descriptor.data_type = f32_storage ? DataType::FLOAT32 : DataType::FLOAT16;
  descriptor.layout = Layout::HWC;
  for (auto& node : nodes_) {
    std::vector<ValueId> all_ids = node.inputs;
    all_ids.insert(all_ids.end(), node.outputs.begin(), node.outputs.end());
    for (auto& tensor_id : all_ids) {
      if (preallocated_tensors_.find(tensor_id) != preallocated_tensors_.end())
        continue;
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const auto& shape = tensor_reserver_.Get(tensor_id).shape;
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      shared_buffer_tensors_[tensor_index] = CreateSharedBufferTensor(
          shared_buffers_[buffer_index], shape, descriptor);
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Tune(TuningType tuning_type,
                                    MetalDevice* device) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.task.Tune(tuning_type, device));
  }
  return absl::OkStatus();
}

void InferenceContext::EncodeWithEncoder(
    id<MTLComputeCommandEncoder> command_encoder,
    const std::map<ValueId, id<MTLBuffer>>& in_out_buffers) {
  UpdatePreallocatedTensors(in_out_buffers);
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& task = nodes_[i].task;
    task.Encode(command_encoder);
  }
}

void InferenceContext::EncodeWithCommandBuffer(
    id<MTLCommandBuffer> command_buffer,
    const std::map<ValueId, id<MTLBuffer>>& in_out_buffers) {
  UpdatePreallocatedTensors(in_out_buffers);
  for (int i = 0; i < nodes_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    auto& task = nodes_[i].task;
    task.Encode(encoder);
    [encoder endEncoding];
  }
}

void InferenceContext::EncodeWithCommandQueue(
    id<MTLCommandQueue> command_queue,
    const std::map<ValueId, id<MTLBuffer>>& in_out_buffers, int flush_period) {
  UpdatePreallocatedTensors(in_out_buffers);
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  for (int i = 0; i < nodes_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    auto& task = nodes_[i].task;
    task.Encode(encoder);
    [encoder endEncoding];
    if (i % flush_period == (flush_period - 1)) {
      [command_buffer commit];
      command_buffer = [command_queue commandBuffer];
    }
  }
  [command_buffer commit];
}

void InferenceContext::UpdatePreallocatedTensors(
    const std::map<ValueId, id<MTLBuffer>>& preallocated) {
  for (const auto& it : preallocated) {
    preallocated_tensors_[it.first].SetBufferHandle(it.second);
  }
  for (auto& task_index : task_ids_with_preallocated_tensors_) {
    auto& task = nodes_[task_index].task;
    const auto& src_ids = nodes_[task_index].inputs;
    for (int i = 0; i < src_ids.size(); ++i) {
      const auto& it = preallocated_tensors_.find(src_ids[i]);
      if (it != preallocated_tensors_.end()) {
        task.SetSrcTensor(&it->second, i);
      }
    }
    const auto& dst_ids = nodes_[task_index].outputs;
    for (int i = 0; i < dst_ids.size(); ++i) {
      const auto& it = preallocated_tensors_.find(dst_ids[i]);
      if (it != preallocated_tensors_.end()) {
        task.SetDstTensor(&it->second, i);
      }
    }
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
