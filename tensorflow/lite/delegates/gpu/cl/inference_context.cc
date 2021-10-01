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

#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/common/selectors/special_selector.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {
bool IsReady(const absl::flat_hash_set<ValueId>& ready_tensors,
             const CLNode& node) {
  for (const ValueId in_id : node.inputs) {
    if (ready_tensors.find(in_id) == ready_tensors.end()) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<ValueId, TensorDescriptor>> GetCLNodeTensors(
    const CLNode& node) {
  std::vector<std::pair<ValueId, TensorDescriptor>> result;
  result.reserve(node.inputs.size() + node.outputs.size());
  const OperationDef op_def = node.cl_operation.GetDefinition();
  for (int j = 0; j < node.inputs.size(); ++j) {
    result.push_back({node.inputs[j], op_def.src_tensors[j]});
  }
  for (int j = 0; j < node.outputs.size(); ++j) {
    result.push_back({node.outputs[j], op_def.dst_tensors[j]});
  }

  return result;
}

absl::Status MergeCLNodes(CLNode* src, CLNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->cl_operation.AddOperation(&src->cl_operation);
}

void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    (*usage_records)[id].y = task_index;
  }
}

// returns true if actual memory for this storage type will be allocated with
// clCreateBuffer.
bool IsBufferBased(const GpuInfo& gpu_info, const TensorStorageType& type) {
  const bool image2d_based_buffer =
      (type == TensorStorageType::TEXTURE_2D ||
       type == TensorStorageType::SINGLE_TEXTURE_2D) &&
      gpu_info.opencl_info.IsImage2dFromBufferSupported();
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER || image2d_based_buffer;
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

// Calculates the total size of the assignment.
size_t TotalSize(const ObjectsAssignment<size_t>& assignment) {
  return std::accumulate(assignment.object_sizes.begin(),
                         assignment.object_sizes.end(), static_cast<size_t>(0));
}

// Checks if sub-buffer image 2D mapping is supported.
bool CanUseSubBuffer(const GpuInfo& gpu_info) {
  if (!gpu_info.IsCL11OrHigher()) {
    return false;
  }
  if (gpu_info.IsPowerVR()) {
    return false;
  }
  if (gpu_info.IsMali() &&
      (gpu_info.mali_info.IsBifrost() || gpu_info.mali_info.IsMidgard())) {
    // Known driver issue on some G72 (Bifrost), G76 (Bifrost), T830 (Midgard),
    // and T880 (Midgard) devices.
    return false;
  }
  return true;
}

}  // namespace

absl::Status InferenceContext::InitFromGraph(
    const CreateInferenceInfo& create_info, const GraphFloat32& graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();

  RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, creation_context.GetGpuInfo(), graph));
  precision_ = create_info.precision;
  storage_type_ = create_info.storage_type;
  if (env->device().GetInfo().IsMali()) {
    need_flush_ = true;
    need_manual_release_ =
        env->device().GetInfo().mali_info.IsValhall() ? false : true;

    flush_periodically_ = true;
    flush_period_ = 24;
  }
  if (env->device().GetInfo().IsPowerVR()) {
    need_flush_ = true;
  }
  CopyInAndOutIds(graph);
  RETURN_IF_ERROR(ConvertOperations(creation_context.GetGpuInfo(), graph,
                                    create_info.hints));
  RETURN_IF_ERROR(Merge());
  RETURN_IF_ERROR(
      AllocateMemory(creation_context.GetGpuInfo(), creation_context.context));
  BindMemoryToOperations();
  RETURN_IF_ERROR(Compile(creation_context));
  RETURN_IF_ERROR(UpdateParams());

  TuningType tuning_type = TuningType::kExhaustive;
  if (create_info.hints.Check(ModelHints::kFastTuning)) {
    tuning_type = TuningType::kFast;
  }
  if (env->device().GetInfo().IsMali()) {
    const MaliInfo& info = env->device().GetInfo().mali_info;
    if (info.IsMaliT6xx()) {
      // Mali T628 hangs forever in clFinish when used profiling queue
      // TuningType::FAST does not use profiling queue.
      tuning_type = TuningType::kFast;
    }
  }
  RETURN_IF_ERROR(
      Tune(tuning_type, env->device().GetInfo(), env->profiling_queue()));
  InitRecordableQueue(env);

  if (serialized_model) {
    for (auto& node : nodes_) {
      node.cl_operation.MoveObjectRefsFromCLToGeneric();
      node.cl_operation.SyncScalarValues();
    }
    const auto inputs = graph.inputs();
    const auto outputs = graph.outputs();
    std::vector<int64_t> in_refs(inputs.size());
    std::vector<int64_t> out_refs(outputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      in_refs[i] = inputs[i]->tensor.ref;
    }
    for (int i = 0; i < outputs.size(); ++i) {
      out_refs[i] = outputs[i]->tensor.ref;
    }
    flatbuffers::FlatBufferBuilder builder;
    auto encoded_fb = Encode(*env->GetDevicePtr(), *this, *env->program_cache(),
                             in_refs, out_refs, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
    for (auto& node : nodes_) {
      node.cl_operation.MoveObjectRefsFromGenericToCL();
    }
  }
  ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status InferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model, Environment* env) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  RETURN_IF_ERROR(Decode(env->context(), *env->GetDevicePtr(),
                         env->program_cache(), decoded_fb, this));

  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();

  RETURN_IF_ERROR(
      AllocateMemory(creation_context.GetGpuInfo(), creation_context.context));
  BindMemoryToOperations();
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.RestoreDeserialized(creation_context));
  }
  RETURN_IF_ERROR(UpdateParams());
  InitRecordableQueue(env);
  ReleaseCPURepresentation();
  return absl::OkStatus();
}

void InferenceContext::InitRecordableQueue(Environment* env) {
  std::vector<ClOperation*> ops(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    ops[i] = &nodes_[i].cl_operation;
  }
  recordable_queue_ = CreateRecordableQueue(ops, env->device(), env->context());
}

absl::Status InferenceContext::InitFromGraphWithTransforms(
    const CreateInferenceInfo& create_info, GraphFloat32* graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  RETURN_IF_ERROR(RunGraphTransforms(graph));
  RETURN_IF_ERROR(InitFromGraph(create_info, *graph, env, serialized_model));
  return absl::OkStatus();
}

void InferenceContext::CopyInAndOutIds(const GraphFloat32& graph) {
  const auto inputs = graph.inputs();
  for (const auto& input : inputs) {
    input_ids_.push_back(input->id);
  }

  const auto variable_inputs = graph.variable_inputs();
  for (const auto& variable_input : variable_inputs) {
    variable_ids_and_refs_[variable_input->id] = variable_input->tensor.ref;
  }

  const auto outputs = graph.outputs();
  for (const auto& output : outputs) {
    output_ids_.push_back(output->id);
  }
}

absl::Status InferenceContext::ReserveGraphTensors(
    const CreateInferenceInfo& create_info, const GpuInfo& gpu_info,
    const GraphFloat32& graph) {
  ValueId max_id = 0;
  auto tensors = graph.values();
  auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
  for (auto& t : tensors) {
    TensorStorageType storage_type = create_info.storage_type;
    const auto shape = graph.GetValue(t->id)->tensor.shape;
    Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
    if (graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id)) {
      if (shape.c < 4 &&
          CanCreateTensorWithShape(
              gpu_info, shape,
              TensorDescriptor{data_type, TensorStorageType::SINGLE_TEXTURE_2D,
                               layout})
              .ok()) {
        storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
      }
    }
    RETURN_IF_ERROR(SelectBestStorageType(gpu_info, shape, storage_type,
                                          data_type, layout, &storage_type));
    tensor_reserver_.Add(
        t->id, {shape, TensorDescriptor{data_type, storage_type, layout}});
    max_id = std::max(max_id, t->id);
  }
  tensor_reserver_.SetNext(max_id + 1);
  return absl::OkStatus();
}

absl::Status InferenceContext::ConvertOperations(const GpuInfo& gpu_info,
                                                 const GraphFloat32& graph,
                                                 ModelHints hints) {
  std::map<ValueId, TensorDescriptor> tensor_descriptors;
  const auto values = graph.values();
  for (auto value : values) {
    tensor_descriptors[value->id] = tensor_reserver_.Get(value->id).descriptor;
  }
  std::set<NodeId> consumed_nodes;
  std::vector<Node*> graph_nodes = graph.nodes();
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input_id : input_ids_) {
    tensor_usages[input_id] = -1;  // so as inputs "updated" before operation 0,
                                   // we will mark them with -1
  }
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
      const_tensors_descs_[outputs[0]->id] =
          tensor_reserver_.Get(outputs[0]->id).descriptor;
      const_tensors_descs_[outputs[0]->id].UploadData(attr.tensor);
      continue;
    }
    std::string op_name = node.operation.type + " " + std::to_string(node.id);
    GPUOperationsSubgraph gpu_subgraph;
    if (hints.Check(ModelHints::kAllowSpecialKernels) &&
        GPUSubgraphFromGraph(gpu_info, precision_, graph, node.id,
                             tensor_descriptors, &consumed_nodes, &gpu_subgraph,
                             &op_name)
            .ok()) {
      // Mapping of subgraph (set of nodes) to GPU operations. Should happen
      // before straigtforward mapping.
    } else {
      // Straigtforward mapping of one graph node to GPU operations.
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
      consumed_nodes.insert(node.id);
      OperationDef op_def;
      op_def.precision = precision_;
      for (int j = 0; j < inputs.size(); ++j) {
        op_def.src_tensors.push_back(
            tensor_reserver_.Get(inputs[j]->id).descriptor);
      }
      for (int j = 0; j < outputs.size(); ++j) {
        op_def.dst_tensors.push_back(
            tensor_reserver_.Get(outputs[j]->id).descriptor);
      }
      RETURN_IF_ERROR(GPUOperationFromNode(gpu_info, op_def, hints, inputs,
                                           outputs, node, &gpu_subgraph));
    }
    absl::flat_hash_map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      auto global_id = tensor_reserver_.Add({t.first, t.second});
      mapping_to_global_ids[j] = global_id;
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      CLNode cl_node;
      cl_node.cl_operation.Init(std::move(gpu_op.operation));
      cl_node.inputs.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          cl_node.inputs[j] = id;
        } else {
          cl_node.inputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      cl_node.outputs.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          cl_node.outputs[j] = id;
          tensor_usages[id] = i;
        } else {
          cl_node.outputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      cl_node.name = op_name;
      nodes_.push_back(std::move(cl_node));
    }
  }

  return absl::OkStatus();
}

absl::Status InferenceContext::Merge() {
  absl::flat_hash_set<ValueId> ready_tensors;
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
    if (!linkable_node.cl_operation.GetGpuOperation().IsLinkable() ||
        linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node)) {
      continue;
    }
    const auto& original_dst_def =
        node.cl_operation.GetDefinition().dst_tensors[0];
    const auto& link_dst_def =
        linkable_node.cl_operation.GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def) {
      continue;
    }
    RETURN_IF_ERROR(MergeCLNodes(&linkable_node, &node));
    nodes_.erase(nodes_.begin() + next_nodes[0]);
    i -= 1;
  }
  return absl::OkStatus();
}

void InferenceContext::GetUsages(const std::function<bool(ValueId)>& functor,
                                 std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (functor(in_id)) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < nodes_.size(); ++op_index) {
    auto tensors = GetCLNodeTensors(nodes_[op_index]);
    for (auto& tensor : tensors) {
      if (functor(tensor.first)) {
        AddUsage(tensor.first, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (functor(out_id)) {
      AddUsage(out_id, nodes_.size(), usages);
    }
  }
}

InferenceContext::TensorMemoryType InferenceContext::GetTensorMemoryType(
    const GpuInfo& gpu_info, ValueId id) {
  if (const_tensors_.find(id) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return TensorMemoryType::kVariable;
  } else if (IsBufferBased(gpu_info,
                           tensor_reserver_.Get(id).descriptor.storage_type)) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kStrongShape;
  }
}

absl::Status InferenceContext::AllocateMemory(const GpuInfo& gpu_info,
                                              CLContext* context) {
  RETURN_IF_ERROR(AllocateMemoryForConstTensors(context));
  RETURN_IF_ERROR(AllocateMemoryForVariableTensors(context));
  RETURN_IF_ERROR(AllocateMemoryForBuffers(gpu_info, context));
  RETURN_IF_ERROR(AllocateMemoryForStrongShapes(gpu_info, context));
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForConstTensors(
    CLContext* context) {
  for (auto& description : const_tensors_descs_) {
    RETURN_IF_ERROR(const_tensors_[description.first].CreateFromDescriptor(
        description.second, context));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForVariableTensors(
    CLContext* context) {
  std::map<ValueId, int> ref_value_to_tensor_index;

  for (auto value_and_ref_value : variable_ids_and_refs_) {
    if (ref_value_to_tensor_index.find(value_and_ref_value.second) ==
        ref_value_to_tensor_index.end()) {
      const auto& t = tensor_reserver_.Get(value_and_ref_value.first);
      const auto& shape = t.shape;
      const auto& descriptor = t.descriptor;

      RETURN_IF_ERROR(
          CreateTensor(*context, shape, descriptor,
                       &variable_tensors_[value_and_ref_value.second]));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForBuffers(const GpuInfo& gpu_info,
                                                        CLContext* context) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(
      [this, &gpu_info](ValueId id) {
        return GetTensorMemoryType(gpu_info, id) == TensorMemoryType::kBuffer;
      },
      &buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& t = tensor_reserver_.Get(usage.first);
    const auto& shape = t.shape;
    const auto& descriptor = t.descriptor;
    const size_t element_size =
        descriptor.data_type == DataType::FLOAT32 ? 4 : 2;
    size_t buffer_size;
    if (descriptor.storage_type == TensorStorageType::TEXTURE_2D) {
      const size_t bytes_per_pixel = element_size * 4;
      const size_t width = shape.b * shape.w;
      const size_t height = shape.h * DivideRoundUp(shape.c, 4);
      const int row_bytes_alignment =
          gpu_info.IsAdreno()
              ? gpu_info.opencl_info.image_pitch_alignment
              : gpu_info.opencl_info.image_pitch_alignment * bytes_per_pixel;
      buffer_size =
          AlignByN(width * bytes_per_pixel, row_bytes_alignment) * height;
    } else if (descriptor.storage_type ==
               TensorStorageType::SINGLE_TEXTURE_2D) {
      const size_t bytes_per_pixel = element_size * shape.c;
      const size_t width = shape.b * shape.w;
      const size_t height = shape.h;
      const int row_bytes_alignment =
          gpu_info.IsAdreno()
              ? gpu_info.opencl_info.image_pitch_alignment
              : gpu_info.opencl_info.image_pitch_alignment * bytes_per_pixel;
      buffer_size =
          AlignByN(width * bytes_per_pixel, row_bytes_alignment) * height;
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

  size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);
  bool use_offset_assignment = false;

  OffsetsAssignment offset_assignment;
  if (CanUseSubBuffer(gpu_info)) {
    RETURN_IF_ERROR(AssignOffsetsToTensors(
        buffer_usage_records, MemoryStrategy::GREEDY_BY_SIZE,
        &offset_assignment, base_align_bytes));
    if (offset_assignment.total_size < TotalSize(buffer_assignment) &&
        offset_assignment.total_size <= gpu_info.GetMaxBufferSize()) {
      use_offset_assignment = true;
    }
  }

  if (use_offset_assignment) {
    shared_buffers_.resize(offset_assignment.offsets.size());
    RETURN_IF_ERROR(CreateReadWriteBuffer(offset_assignment.total_size, context,
                                          &shared_buffers_parent_));
    for (int i = 0; i < offset_assignment.offsets.size(); ++i) {
      RETURN_IF_ERROR(CreateReadWriteSubBuffer(
          shared_buffers_parent_, offset_assignment.offsets[i],
          buffer_usage_records[i].tensor_size, context, &shared_buffers_[i]));
    }
  } else {
    shared_buffers_.resize(buffer_assignment.object_sizes.size());
    for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
      RETURN_IF_ERROR(CreateReadWriteBuffer(buffer_assignment.object_sizes[i],
                                            context, &shared_buffers_[i]));
    }
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  for (auto& node : nodes_) {
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(gpu_info, t.first) != TensorMemoryType::kBuffer)
        continue;
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[t.first];
      if (created_tensors[tensor_index]) continue;
      const auto& shape = tensor_reserver_.Get(t.first).shape;
      const int buffer_index = use_offset_assignment
                                   ? tensor_index
                                   : buffer_assignment.object_ids[tensor_index];
      if (t.second.storage_type == TensorStorageType::TEXTURE_2D ||
          t.second.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
        const int row_bytes_alignment =
            gpu_info.IsAdreno() ? gpu_info.opencl_info.image_pitch_alignment
                                : 0;
        RETURN_IF_ERROR(CreateSharedImage2DBufferTensor(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), shape,
            t.second, row_bytes_alignment,
            &shared_buffer_tensors_[tensor_index]));
      } else {
        RETURN_IF_ERROR(CreateSharedTensor(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), shape,
            t.second, &shared_buffer_tensors_[tensor_index]));
      }
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForStrongShapes(
    const GpuInfo& gpu_info, CLContext* context) {
  std::map<ValueId, int2> usages;
  GetUsages(
      [this, &gpu_info](ValueId id) {
        return GetTensorMemoryType(gpu_info, id) ==
               TensorMemoryType::kStrongShape;
      },
      &usages);

  std::vector<TensorUsageRecord<DummyTensor>> usage_records;
  std::map<ValueId, ValueId> remap_from_graph_ids;
  for (auto& usage : usages) {
    remap_from_graph_ids[usage.first] = usage_records.size();
    usage_records.push_back({tensor_reserver_.Get(usage.first),
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<DummyTensor> assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  for (auto& node : nodes_) {
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(gpu_info, t.first) !=
          TensorMemoryType::kStrongShape) {
        continue;
      }
      const auto& shape = tensor_reserver_.Get(t.first).shape;
      const auto id = assignment.object_ids[remap_from_graph_ids[t.first]];
      graph_ids_to_strong_shape_tensors_[t.first] = id;
      const auto& it = strong_shape_tensors_.find(id);
      if (it == strong_shape_tensors_.end()) {
        RETURN_IF_ERROR(CreateTensor(*context, shape, t.second,
                                     &strong_shape_tensors_[id]));
      }
    }
  }
  return absl::OkStatus();
}

void InferenceContext::BindMemoryToOperations() {
  for (auto& node : nodes_) {
    for (int i = 0; i < node.inputs.size(); ++i) {
      node.cl_operation.GetGpuOperation().SetSrc(GetTensor(node.inputs[i]), i);
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      node.cl_operation.GetGpuOperation().SetDst(GetTensor(node.outputs[i]), i);
    }
  }
}

absl::Status InferenceContext::Compile(
    const CreationContext& creation_context) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.Compile(creation_context));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Tune(TuningType tuning_type,
                                    const GpuInfo& gpu_info,
                                    ProfilingCommandQueue* profiling_queue) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(
        node.cl_operation.Tune(tuning_type, gpu_info, profiling_queue));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::UpdateParams() {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.UpdateParams());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AddToQueue(CLCommandQueue* queue) {
  if (recordable_queue_->IsSupported()) {
    return recordable_queue_->Execute(queue);
  }
  if (need_manual_release_) {
    if (prev_enqueue_start_point_.is_valid()) {
      prev_enqueue_start_point_.Wait();
    }
    RETURN_IF_ERROR(queue->EnqueueEvent(&prev_enqueue_start_point_));
  }
  int counter = 0;
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
    counter++;
    if (flush_periodically_ && counter % flush_period_ == 0) {
      clFlush(queue->queue());
    }
  }
  if (need_flush_) {
    clFlush(queue->queue());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Profile(ProfilingCommandQueue* queue,
                                       ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (auto& node : nodes_) {
    queue->SetEventsLabel(node.name);
    RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
  }
  RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();
  return absl::OkStatus();
}

uint64_t InferenceContext::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  uint64_t total_memory = 0;
  for (const auto& t : strong_shape_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    // Sub-buffers do not allocate memory. Count the size of the parent buffer
    // object instead.
    if (!b.IsSubBuffer()) {
      total_memory += b.GetMemorySizeInBytes();
    }
  }
  for (const auto& t : variable_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }
  total_memory += shared_buffers_parent_.GetMemorySizeInBytes();

  return total_memory;
}

Tensor* InferenceContext::GetTensor(ValueId id) {
  if (const_tensors_.find(id) != const_tensors_.end()) {
    return &const_tensors_[id];
  } else if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return &variable_tensors_[variable_ids_and_refs_[id]];
  } else if (graph_ids_to_shared_buffer_tensors_.find(id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_[graph_ids_to_shared_buffer_tensors_[id]];
  } else {
    return &strong_shape_tensors_[graph_ids_to_strong_shape_tensors_[id]];
  }
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorFloat32& tensor,
                                              CLCommandQueue* queue) {
  return GetTensor(id)->WriteData(queue, tensor);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorFloat32* result) {
  const auto& gpu_tensor = *GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor.Batch(), gpu_tensor.Height(),
                              gpu_tensor.Width(), gpu_tensor.Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());
  return gpu_tensor.ReadData(queue, result);
}

void InferenceContext::ReleaseCPURepresentation() {
  for (auto& node : nodes_) {
    node.cl_operation.GetGpuOperation().args_.ReleaseCPURepresentation();
  }
  const_tensors_descs_.clear();
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
