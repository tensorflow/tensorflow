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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

// returns true if actual memory for this storage type is buffer
bool IsBufferBased(const TensorStorageType& type) {
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER ||
         type == TensorStorageType::TEXTURE_2D ||
         type == TensorStorageType::SINGLE_TEXTURE_2D;
}

bool HasIntersection(const std::vector<ValueId>& vec_ids,
                     const std::set<ValueId>& ids) {
  for (ValueId id : vec_ids) {
    if (ids.find(id) != ids.end()) {
      return true;
    }
  }
  return false;
}

bool IsReady(const std::set<ValueId>& ready_tensors, const GpuNode& node) {
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

bool IsAssociativeLinkableOp(const tflite::gpu::Node& node,
                             const std::vector<tflite::gpu::Value*>& inputs,
                             const std::vector<tflite::gpu::Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const tflite::gpu::OperationType op_type =
      tflite::gpu::OperationTypeFromString(node.operation.type);
  if (op_type != tflite::gpu::OperationType::ADD &&
      op_type != tflite::gpu::OperationType::MUL) {
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

absl::Status MergeGpuNodes(GpuNode* src, GpuNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->gpu_operation->AddOperation(src->gpu_operation.get());
}

// Helper class for creating descriptors for appropriate tensors from
// GraphFloat32
// Also allows to create descriptors for new tensors(not present in
// GraphFloat32)
class TensorReserver {
 public:
  TensorReserver() : next_(0) {}
  ValueId Add(const TensorDescriptor& dummy) {
    reservations_[next_] = dummy;
    return next_++;
  }
  void Add(ValueId id, const TensorDescriptor& dummy) {
    reservations_[id] = dummy;
  }
  void SetNext(ValueId id) { next_ = id; }
  TensorDescriptor Get(ValueId id) { return reservations_[id]; }

 public:
  absl::flat_hash_map<ValueId, TensorDescriptor> reservations_;
  ValueId next_;
};

absl::Status CheckExternalTensorDescription(const GpuInfo& gpu_info,
                                            const TensorDescriptor& tensor_desc,
                                            const BHWC& shape,
                                            DataType data_type) {
  if (tensor_desc.data_type != data_type) {
    return absl::InvalidArgumentError(
        "Global precision and precision of predefined/external tensors must be "
        "synchronized.");
  }
  const bool tensor_supported_layout = tensor_desc.layout == Layout::HWDC ||
                                       tensor_desc.layout == Layout::BHWDC ||
                                       tensor_desc.layout == Layout::HWC ||
                                       tensor_desc.layout == Layout::BHWC;
  if (!tensor_supported_layout) {
    return absl::InvalidArgumentError(
        "Currently no support of this layouts for spatial tensors.");
  }
  const bool has_depth =
      tensor_desc.layout == Layout::HWDC || tensor_desc.layout == Layout::BHWDC;
  if (has_depth) {
    return absl::InvalidArgumentError(
        "Currently no support of Depth dimension in predefined/external "
        "tensors.");
  }
  const bool has_batch =
      tensor_desc.layout == Layout::BHWC || tensor_desc.layout == Layout::BHWDC;
  if (has_batch && shape.b == 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!has_batch && shape.b != 1) {
    return absl::InvalidArgumentError("Wrong layout, batch mismatch.");
  }
  if (!CanCreateTensorWithShape(gpu_info, shape, tensor_desc).ok()) {
    return absl::UnavailableError(
        "Current device can not allocate tensor with this shape for "
        "predefined/external descriptor.");
  }
  return absl::OkStatus();
}

absl::Status ReserveGraphTensors(
    const InferenceContext::CreateInferenceInfo& create_info,
    const GpuInfo& gpu_info, const GraphFloat32& graph,
    TensorReserver* tensor_reserver) {
  ValueId max_id = 0;
  auto tensors = graph.values();
  auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
  for (auto& t : tensors) {
    const auto shape = graph.GetValue(t->id)->tensor.shape;
    auto it_immutable_external =
        create_info.external_immutable_tensors.find(t->id);
    auto it_mutable_external = create_info.external_mutable_tensors.find(t->id);
    TensorDescriptor tensor_desc;
    if (it_immutable_external != create_info.external_immutable_tensors.end()) {
      if (!(graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id))) {
        return absl::InvalidArgumentError(
            "Currently external tensors can be used only for graph "
            "inputs/outputs");
      }
      tensor_desc = it_immutable_external->second->GetDescriptor();
      RETURN_IF_ERROR(CheckExternalTensorDescription(gpu_info, tensor_desc,
                                                     shape, data_type));
    } else if (it_mutable_external !=
               create_info.external_mutable_tensors.end()) {
      if (!(graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id))) {
        return absl::InvalidArgumentError(
            "Currently external tensors can be used only for graph "
            "inputs/outputs");
      }
      tensor_desc = it_mutable_external->second;
      RETURN_IF_ERROR(CheckExternalTensorDescription(
          gpu_info, it_mutable_external->second, shape, data_type));
    } else {
      TensorStorageType storage_type = create_info.storage_type;
      Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
      if (graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id)) {
        if (shape.c < 4 &&
            CanCreateTensorWithShape(
                gpu_info, shape,
                TensorDescriptor{data_type,
                                 TensorStorageType::SINGLE_TEXTURE_2D, layout})
                .ok()) {
          storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
        }
      }
      RETURN_IF_ERROR(SelectBestStorageType(gpu_info, shape, storage_type,
                                            data_type, layout, &storage_type));
      tensor_desc = TensorDescriptor{data_type, storage_type, layout};
      if (storage_type == TensorStorageType::TEXTURE_2D) {
        tensor_desc.use_buffer_for_write_only_2d_texture = true;
      }
    }
    tensor_desc.shape = BHWDC(shape.b, shape.h, shape.w, 1, shape.c);
    tensor_reserver->Add(t->id, tensor_desc);
    max_id = std::max(max_id, t->id);
  }
  tensor_reserver->SetNext(max_id + 1);
  return absl::OkStatus();
}

absl::Status ConvertOperations(
    const GpuInfo& gpu_info, const GraphFloat32& graph,
    const InferenceContext::CreateInferenceInfo& create_info,
    TensorReserver* tensor_reserver, InferenceContext::GpuModel* gpu_model) {
  std::map<ValueId, TensorDescriptor> tensor_descriptors;
  const auto values = graph.values();
  for (auto value : values) {
    tensor_descriptors[value->id] = tensor_reserver->Get(value->id);
  }
  std::set<NodeId> consumed_nodes;
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input : gpu_model->input_ids_and_refs) {
    tensor_usages[input.first] = -1;  // so as inputs "updated" before operation
                                      // 0, we will mark them with -1
  }
  std::vector<Node*> graph_nodes = graph.nodes();
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    if (consumed_nodes.find(node.id) != consumed_nodes.end()) {
      continue;
    }
    auto op_type = OperationTypeFromString(node.operation.type);
    if (op_type == OperationType::CONSTANT) {
      auto attr =
          absl::any_cast<ConstTensorAttributes>(node.operation.attributes);
      auto outputs = graph.FindOutputs(node.id);
      gpu_model->const_tensors[outputs[0]->id] =
          tensor_reserver->Get(outputs[0]->id);
      gpu_model->const_tensors[outputs[0]->id].UploadData(attr.tensor);
      continue;
    }
    GPUOperationsSubgraph gpu_subgraph;
    if (create_info.hints.Check(ModelHints::kAllowSpecialKernels) &&
        GPUSubgraphFromGraph(gpu_info, create_info.precision, graph, node.id,
                             tensor_descriptors, &consumed_nodes, &gpu_subgraph)
            .ok()) {
      // Mapping of subgraph (set of nodes) to GPU operations. Should happen
      // before straigtforward mapping.
    } else {
      // Straigtforward mapping of one graph node to GPU operations.
      auto inputs = graph.FindInputs(node.id);
      auto outputs = graph.FindOutputs(node.id);
      // Reordering of input ids and updating of temporary tensors_usage struct.
      // To have better linking we need linking tensor(latest written during
      // linear execution) on first position.
      if (IsAssociativeLinkableOp(node, inputs, outputs)) {
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
      consumed_nodes.insert(node.id);
      OperationDef op_def;
      op_def.precision = create_info.precision;
      for (int j = 0; j < inputs.size(); ++j) {
        op_def.src_tensors.push_back(tensor_reserver->Get(inputs[j]->id));
      }
      for (int j = 0; j < outputs.size(); ++j) {
        op_def.dst_tensors.push_back(tensor_reserver->Get(outputs[j]->id));
      }
      RETURN_IF_ERROR(GPUOperationFromNode(gpu_info, op_def, create_info.hints,
                                           inputs, outputs, node,
                                           &gpu_subgraph));
    }
    std::map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      TensorDescriptor td = t.second;
      td.shape = BHWDC(t.first.b, t.first.h, t.first.w, 1, t.first.c);
      auto global_id = tensor_reserver->Add(td);
      mapping_to_global_ids[j] = global_id;
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      GpuNode gpu_node;
      gpu_node.gpu_operation = std::move(gpu_op.operation);
      gpu_node.inputs.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          gpu_node.inputs[j] = id;
        } else {
          gpu_node.inputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      gpu_node.outputs.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          gpu_node.outputs[j] = id;
          tensor_usages[id] = i;
        } else {
          gpu_node.outputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      gpu_node.name = gpu_op.name;
      gpu_model->nodes.push_back(std::move(gpu_node));
    }
  }
  return absl::OkStatus();
}

absl::Status Merge(InferenceContext::GpuModel* gpu_model) {
  std::set<ValueId> ready_tensors;
  for (const auto& input : gpu_model->input_ids_and_refs) {
    ready_tensors.insert(input.first);
  }
  auto& nodes = gpu_model->nodes;
  for (int i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    for (const auto& out_id : node.outputs) {
      ready_tensors.insert(out_id);
    }
    if (node.outputs.size() != 1) {
      continue;
    }
    std::vector<int> next_nodes;
    int link_index = 0;
    for (int j = i + 1; j < nodes.size(); ++j) {
      for (int k = 0; k < nodes[j].inputs.size(); ++k) {
        if (nodes[j].inputs[k] == node.outputs[0]) {
          next_nodes.push_back(j);
          link_index = k;
        }
      }
    }
    if (next_nodes.size() != 1 || link_index != 0) {
      continue;
    }
    auto& linkable_node = nodes[next_nodes[0]];
    if (!linkable_node.gpu_operation->IsLinkable() ||
        linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node)) {
      continue;
    }
    const auto& original_dst_def =
        node.gpu_operation->GetDefinition().dst_tensors[0];
    const auto& link_dst_def =
        linkable_node.gpu_operation->GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def) {
      continue;
    }
    RETURN_IF_ERROR(MergeGpuNodes(&linkable_node, &node));
    nodes.erase(nodes.begin() + next_nodes[0]);
    i -= 1;
  }
  return absl::OkStatus();
}

void CopyExternals(const GraphFloat32& graph,
                   InferenceContext::GpuModel* gpu_model) {
  const auto inputs = graph.inputs();
  for (const auto& value : inputs) {
    gpu_model->input_ids_and_refs.push_back({value->id, value->tensor.ref});
  }

  const auto variable_inputs = graph.variable_inputs();
  for (const auto& value : variable_inputs) {
    gpu_model->variable_ids_and_refs.push_back({value->id, value->tensor.ref});
  }

  const auto outputs = graph.outputs();
  for (const auto& value : outputs) {
    gpu_model->output_ids_and_refs.push_back({value->id, value->tensor.ref});
  }
}

absl::Status GraphToGpuModel(
    const InferenceContext::CreateInferenceInfo& create_info,
    const GraphFloat32& graph, const GpuInfo& gpu_info,
    InferenceContext::GpuModel* gpu_model) {
  TensorReserver tensor_reserver;
  RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, gpu_info, graph, &tensor_reserver));
  CopyExternals(graph, gpu_model);
  RETURN_IF_ERROR(ConvertOperations(gpu_info, graph, create_info,
                                    &tensor_reserver, gpu_model));
  RETURN_IF_ERROR(Merge(gpu_model));
  gpu_model->tensors = std::move(tensor_reserver.reservations_);

  for (auto& node : gpu_model->nodes) {
    RETURN_IF_ERROR(node.gpu_operation->AssembleCode(gpu_info));
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status InferenceContext::InitFromGraphWithTransforms(
    const CreateInferenceInfo& create_info, GraphFloat32* graph,
    id<MTLDevice> device_id) {
  RETURN_IF_ERROR(RunGraphTransforms(graph));
  RETURN_IF_ERROR(InitFromGraph(create_info, *graph, device_id));
  return absl::OkStatus();
}

absl::Status InferenceContext::InitFromGraph(
    const CreateInferenceInfo& create_info, const GraphFloat32& graph,
    id<MTLDevice> device_id) {
  device_ = device_id;
  MetalDevice metal_device(device_id);
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      GraphToGpuModel(create_info, graph, metal_device.GetInfo(), &gpu_model));

  for (const auto& input : gpu_model.input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& output : gpu_model.output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model.nodes.size());
  for (int i = 0; i < gpu_model.nodes.size(); ++i) {
    nodes_[i].task.Init(std::move(gpu_model.nodes[i].gpu_operation));
    nodes_[i].inputs = gpu_model.nodes[i].inputs;
    nodes_[i].outputs = gpu_model.nodes[i].outputs;
    nodes_[i].name = gpu_model.nodes[i].name;
  }
  const_tensors_descs_ = std::move(gpu_model.const_tensors);
  tensors_descs_ = std::move(gpu_model.tensors);

  for (const auto& external_tensor : create_info.external_immutable_tensors) {
    auto* metal_spatial_tensor =
        dynamic_cast<MetalSpatialTensor*>(external_tensor.second);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    external_immutable_tensors_[external_tensor.first] = metal_spatial_tensor;
  }
  std::map<ValueId, MetalSpatialTensor> temp_external_tensors;
  for (const auto& external_tensor : create_info.external_mutable_tensors) {
    RETURN_IF_ERROR(
        CreateTensor(device_id, tensors_descs_[external_tensor.first].shape,
                     tensors_descs_[external_tensor.first],
                     &temp_external_tensors[external_tensor.first]));
    external_mutable_tensors_[external_tensor.first] =
        &temp_external_tensors[external_tensor.first];
  }
  PrepareExternal();
  RETURN_IF_ERROR(CompileOperations(&metal_device));
  RETURN_IF_ERROR(AllocateTensors(&metal_device));
  BindTensorsToOperations();
  RETURN_IF_ERROR(UpdateParams(metal_device.GetInfo()));
  RETURN_IF_ERROR(Tune(TuningType::kFast, &metal_device));

  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }

  bool add_icb_support = false && external_mutable_tensors_.empty();
  if (add_icb_support) {
    if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
      MTLIndirectCommandBufferDescriptor* icb_desc =
          [[MTLIndirectCommandBufferDescriptor alloc] init];
      icb_desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
      icb_desc.inheritBuffers = NO;
      icb_desc.inheritPipelineState = NO;
      icb_desc.maxKernelBufferBindCount = 1;

      icb_ = [device_id newIndirectCommandBufferWithDescriptor:icb_desc
                                               maxCommandCount:nodes_.size()
                                                       options:0];

      for (int i = 0; i < nodes_.size(); ++i) {
        id<MTLIndirectComputeCommand> icb_command =
            [icb_ indirectComputeCommandAtIndex:i];
        auto& node = nodes_[i];
        node.task.EncodeToICB(icb_command);
      }
    }
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
  RETURN_IF_ERROR(AllocateMemoryForConstTensors(device));
  RETURN_IF_ERROR(AllocateMemoryForBuffers(device));
  RETURN_IF_ERROR(AllocateMemoryForStrongShapes(device));
  return absl::OkStatus();
}

MetalSpatialTensor* InferenceContext::GetTensor(ValueId tensor_id) {
  if (external_immutable_tensors_.find(tensor_id) !=
      external_immutable_tensors_.end()) {
    return external_immutable_tensors_[tensor_id];
  } else if (external_mutable_tensors_.find(tensor_id) !=
             external_mutable_tensors_.end()) {
    return external_mutable_tensors_[tensor_id];
  } else if (const_tensors_.find(tensor_id) != const_tensors_.end()) {
    return &const_tensors_[tensor_id];
  } else if (graph_ids_to_shared_buffer_tensors_.find(tensor_id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_
        [graph_ids_to_shared_buffer_tensors_[tensor_id]];
  } else if (graph_ids_to_strong_shape_tensors_.find(tensor_id) !=
             graph_ids_to_strong_shape_tensors_.end()) {
    return &strong_shape_tensors_
        [graph_ids_to_strong_shape_tensors_[tensor_id]];
  }
  return nullptr;
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorFloat32& tensor) {
  return GetTensor(id)->WriteData(device_, tensor);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               TensorFloat32* result) {
  const auto& gpu_tensor = *GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor.Batch(), gpu_tensor.Height(),
                              gpu_tensor.Width(), gpu_tensor.Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());
  return gpu_tensor.ReadData(device_, result);
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
      const auto& shape = tensors_descs_[in_id].shape;
      src_shapes.push_back(BHWC(shape.b, shape.h, shape.w, shape.c));
    }
    for (const auto& out_id : node.outputs) {
      const auto& shape = tensors_descs_[out_id].shape;
      dst_shapes.push_back(BHWC(shape.b, shape.h, shape.w, shape.c));
    }
    RETURN_IF_ERROR(node.task.UpdateParams());
  }
  return absl::OkStatus();
}

InferenceContext::TensorMemoryType InferenceContext::GetTensorMemoryType(
    ValueId id) {
  if (external_immutable_tensors_.find(id) !=
      external_immutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (external_mutable_tensors_.find(id) !=
             external_mutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (IsBufferBased(tensors_descs_[id].storage_type)) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kStrongShape;
  }
}

void InferenceContext::GetUsages(const std::function<bool(ValueId)>& functor,
                                 std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (functor(in_id)) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < nodes_.size(); ++op_index) {
    for (auto& tensor_id : nodes_[op_index].inputs) {
      if (functor(tensor_id)) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
    for (auto& tensor_id : nodes_[op_index].outputs) {
      if (functor(tensor_id)) {
        AddUsage(tensor_id, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (functor(out_id)) {
      AddUsage(out_id, nodes_.size(), usages);
    }
  }
}

absl::Status InferenceContext::AllocateMemoryForConstTensors(
    MetalDevice* device) {
  for (auto& description : const_tensors_descs_) {
    RETURN_IF_ERROR(const_tensors_[description.first].CreateFromDescriptor(
        description.second, device->device()));
  }
  const_tensors_descs_.clear();
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForBuffers(MetalDevice* device) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(
      [this](ValueId id) {
        return GetTensorMemoryType(id) == TensorMemoryType::kBuffer;
      },
      &buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& t = tensors_descs_[usage.first];
    const auto& shape = t.shape;
    const auto& descriptor = t;
    const size_t element_size =
        descriptor.data_type == DataType::FLOAT32 ? 4 : 2;
    size_t buffer_size;
    size_t row_bytes_alignment = [device->device()
        minimumLinearTextureAlignmentForPixelFormat:DataTypeToRGBAPixelFormat(
                                                        descriptor.data_type,
                                                        false)];
    if (descriptor.storage_type == TensorStorageType::TEXTURE_2D) {
      const size_t bytes_per_row = element_size * shape.b * shape.w * 4;
      const size_t height = shape.h * DivideRoundUp(shape.c, 4);
      buffer_size = AlignByN(bytes_per_row, row_bytes_alignment) * height;
    } else if (descriptor.storage_type ==
               TensorStorageType::SINGLE_TEXTURE_2D) {
      const size_t bytes_per_row = element_size * shape.b * shape.w * shape.c;
      const size_t height = shape.h;
      buffer_size = AlignByN(bytes_per_row, row_bytes_alignment) * height;
    } else {
      buffer_size =
          shape.b * shape.w * shape.h * AlignByN(shape.c, 4) * element_size;
    }
    graph_ids_to_shared_buffer_tensors_[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  shared_buffers_.resize(buffer_assignment.object_sizes.size());
  for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
    // Initialize metal buffer
    NSUInteger bufferSize = buffer_assignment.object_sizes[i];

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
  for (auto& node : nodes_) {
    std::vector<ValueId> all_ids = node.inputs;
    all_ids.insert(all_ids.end(), node.outputs.begin(), node.outputs.end());
    for (auto& tensor_id : all_ids) {
      if (GetTensorMemoryType(tensor_id) != TensorMemoryType::kBuffer) {
        continue;
      }
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const auto& tensor_dummy = tensors_descs_[tensor_id];
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      if (tensor_dummy.storage_type == TensorStorageType::TEXTURE_2D ||
          tensor_dummy.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
        size_t row_bytes_alignment = [device->device()
            minimumLinearTextureAlignmentForPixelFormat:
                DataTypeToRGBAPixelFormat(tensor_dummy.data_type, false)];
        RETURN_IF_ERROR(CreateSharedImage2DBufferTensor(
            shared_buffers_[buffer_index], tensor_dummy.shape, tensor_dummy,
            row_bytes_alignment, &shared_buffer_tensors_[tensor_index]));
      } else {
        RETURN_IF_ERROR(CreateSharedBufferTensor(
            shared_buffers_[buffer_index], tensor_dummy.shape, tensor_dummy,
            &shared_buffer_tensors_[tensor_index]));
      }
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForStrongShapes(
    MetalDevice* device) {
  std::map<ValueId, int2> usages;
  GetUsages(
      [this](ValueId id) {
        return GetTensorMemoryType(id) == TensorMemoryType::kStrongShape;
      },
      &usages);

  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      return tensor_desc.data_type == t.tensor_desc.data_type &&
             tensor_desc.storage_type == t.tensor_desc.storage_type &&
             tensor_desc.layout == t.tensor_desc.layout &&
             tensor_desc.shape == t.tensor_desc.shape;
    }
  };

  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  std::map<ValueId, ValueId> remap_from_graph_ids;
  for (auto& usage : usages) {
    remap_from_graph_ids[usage.first] = usage_records.size();
    usage_records.push_back({{tensors_descs_[usage.first]},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  for (auto& node : nodes_) {
    std::vector<ValueId> all_ids = node.inputs;
    all_ids.insert(all_ids.end(), node.outputs.begin(), node.outputs.end());
    for (auto& tensor_id : all_ids) {
      const auto& tensor_dummy = tensors_descs_[tensor_id];
      if (GetTensorMemoryType(tensor_id) != TensorMemoryType::kStrongShape) {
        continue;
      }
      const auto id = assignment.object_ids[remap_from_graph_ids[tensor_id]];
      graph_ids_to_strong_shape_tensors_[tensor_id] = id;
      const auto& it = strong_shape_tensors_.find(id);
      if (it == strong_shape_tensors_.end()) {
        RETURN_IF_ERROR(CreateTensor(device->device(), tensor_dummy.shape,
                                     tensor_dummy, &strong_shape_tensors_[id]));
      }
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
    id<MTLComputeCommandEncoder> command_encoder) {
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& task = nodes_[i].task;
    task.Encode(command_encoder);
  }
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void InferenceContext::AddResources(
    id<MTLComputeCommandEncoder> command_encoder) {
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& task = nodes_[i].task;
    task.AddResourcesToEncoder(command_encoder);
  }
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void InferenceContext::EncodeWithICB(
    id<MTLComputeCommandEncoder> command_encoder) {
  [command_encoder executeCommandsInBuffer:icb_
                                 withRange:NSMakeRange(0, nodes_.size())];
}

void InferenceContext::Profile(id<MTLDevice> device, ProfilingInfo* result) {
  result->dispatches.resize(nodes_.size());
  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  for (int k = 0; k < nodes_.size(); ++k) {
    @autoreleasepool {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      auto& task = nodes_[k].task;
      const int kRuns = 500;
      for (int i = 0; i < kRuns; ++i) {
        task.Encode(encoder);
      }
      [encoder endEncoding];
      auto start = absl::Now();
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      auto end = absl::Now();
      auto& dispatch_info = result->dispatches[k];
      dispatch_info.label = nodes_[k].name;
      dispatch_info.duration = (end - start) / static_cast<float>(kRuns);

      uint64_t read_size = 0;
      for (auto& src_id : nodes_[k].inputs) {
        read_size += GetTensor(src_id)->GetMemorySizeInBytes();
      }
      const auto& gpu_op = nodes_[k].task.GetGpuOperation();
      read_size += gpu_op.const_args_size_;
      uint64_t write_size = 0;
      for (auto& dst_id : nodes_[k].outputs) {
        write_size += GetTensor(dst_id)->GetMemorySizeInBytes();
      }
      dispatch_info.flops = gpu_op.flops_;
      dispatch_info.read_mem_size = read_size;
      dispatch_info.write_mem_size = write_size;
    }
  }
}

uint64_t InferenceContext::GetIntermediateTensorsSize() const {
  uint64_t total_memory = 0;
  for (const auto& t : strong_shape_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    total_memory += [b length];
  }

  return total_memory;
}

void InferenceContext::EncodeWithCommandBuffer(
    id<MTLCommandBuffer> command_buffer) {
  for (int i = 0; i < nodes_.size(); ++i) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    auto& task = nodes_[i].task;
    task.Encode(encoder);
    [encoder endEncoding];
  }
}

void InferenceContext::EncodeWithCommandQueue(id<MTLCommandQueue> command_queue,
                                              int flush_period) {
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

absl::Status InferenceContext::SetTensor(const ValueId& tensor_id,
                                         MetalSpatialTensor* tensor_ptr) {
  auto it = external_mutable_tensors_.find(tensor_id);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  external_mutable_tensors_[tensor_id] = tensor_ptr;
  for (int node_index : external_tensor_to_nodes_[tensor_id]) {
    auto& node = nodes_[node_index];
    for (int i = 0; i < node.inputs.size(); ++i) {
      if (node.inputs[i] == tensor_id) {
        node.task.SetSrcTensor(tensor_ptr, i);
      }
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      if (node.outputs[i] == tensor_id) {
        node.task.SetDstTensor(tensor_ptr, i);
      }
    }
  }
  return absl::OkStatus();
}

void InferenceContext::PrepareExternal() {
  for (auto& external : external_mutable_tensors_) {
    for (int i = 0; i < nodes_.size(); ++i) {
      bool has_tensor = false;
      const auto& src_ids = nodes_[i].inputs;
      for (int i = 0; i < src_ids.size(); ++i) {
        if (src_ids[i] == external.first) {
          has_tensor = true;
        }
      }
      const auto& dst_ids = nodes_[i].outputs;
      for (int i = 0; i < dst_ids.size(); ++i) {
        if (dst_ids[i] == external.first) {
          has_tensor = true;
        }
      }
      if (has_tensor) {
        external_tensor_to_nodes_[external.first].push_back(i);
      }
    }
  }
}

absl::Status RunGraphTransforms(GraphFloat32* graph) {
  auto merge_padding_transform = NewMergePaddingWithAdd();
  auto add_bias_transform = NewAddBias();
  auto pooling_to_reduce_op = NewGlobalPoolingToReduceOp();
  ModelTransformer transformer(graph);
  if (!transformer.Apply("add_bias", add_bias_transform.get())) {
    return absl::InternalError("Invalid add_bias transform");
  }
  if (!transformer.Apply("merge_padding", merge_padding_transform.get())) {
    return absl::InternalError("Invalid merge_padding transform");
  }
  if (!transformer.Apply("global pooling to mean",
                         pooling_to_reduce_op.get())) {
    return absl::InternalError("Invalid global pooling to mean transform");
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
