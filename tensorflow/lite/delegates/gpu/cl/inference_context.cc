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
#include "tensorflow/lite/delegates/gpu/cl/serialization.h"
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
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {

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

void InferenceContext::ExecutionHints::Init(const GpuInfo& gpu_info) {
  if (gpu_info.IsMali()) {
    need_flush = true;
    need_manual_release = gpu_info.mali_info.IsValhall() ? false : true;

    flush_periodically = true;
    flush_period = 24;
  }
  if (gpu_info.IsPowerVR()) {
    need_flush = true;
    flush_periodically = true;
    flush_period = 16;
  }
}

absl::Status InferenceContext::InitFromGraph(
    const CreateGpuModelInfo& create_info, const GraphFloat32& graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  GpuModel gpu_model;
  RETURN_IF_ERROR(GraphToGpuModel(graph, create_info,
                                  env->GetDevicePtr()->GetInfo(), &gpu_model));
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = Encode(gpu_model, &builder);
  }
  CopyFromGpuModel(&gpu_model);

  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();
  for (const auto& external_tensor : create_info.external_immutable_tensors) {
    auto* cl_spatial_tensor = dynamic_cast<Tensor*>(external_tensor.second);
    if (!cl_spatial_tensor) {
      return absl::InvalidArgumentError("Expected CLSpatialTensor.");
    }
    external_immutable_tensors_[external_tensor.first] = cl_spatial_tensor;
  }
  std::map<ValueId, Tensor> temp_external_tensors;
  for (const auto& external_tensor : create_info.external_mutable_tensors) {
    RETURN_IF_ERROR(CreateTensor(
        env->context(), tensors_descs_[external_tensor.first].shape,
        tensors_descs_[external_tensor.first],
        &temp_external_tensors[external_tensor.first]));
    external_mutable_tensors_[external_tensor.first] =
        &temp_external_tensors[external_tensor.first];
  }
  PrepareExternal();
  execution_hints_.Init(env->device().GetInfo());
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
  if (external_mutable_tensors_.empty()) {
    // using recordable queue only when no mutable external tensors
    InitRecordableQueue(env);
  }

  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }

  gpu_info_ = env->device().GetInfo();

  if (serialized_model) {
    auto encoded_fb = Encode(*env->GetDevicePtr(), *this, *env->program_cache(),
                             gpu_model_fb, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }
  ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status InferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model, Environment* env,
    CreateGpuModelInfo* create_info) {
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
  std::map<ValueId, Tensor> temp_external_tensors;
  if (create_info) {
    for (const auto& external_tensor :
         create_info->external_immutable_tensors) {
      auto* cl_spatial_tensor = dynamic_cast<Tensor*>(external_tensor.second);
      if (!cl_spatial_tensor) {
        return absl::InvalidArgumentError("Expected CLSpatialTensor.");
      }
      external_immutable_tensors_[external_tensor.first] = cl_spatial_tensor;
    }
    for (const auto& external_tensor : create_info->external_mutable_tensors) {
      RETURN_IF_ERROR(CreateTensor(
          env->context(), tensors_descs_[external_tensor.first].shape,
          tensors_descs_[external_tensor.first],
          &temp_external_tensors[external_tensor.first]));
      external_mutable_tensors_[external_tensor.first] =
          &temp_external_tensors[external_tensor.first];
    }
  }
  PrepareExternal();

  execution_hints_.Init(env->device().GetInfo());

  RETURN_IF_ERROR(
      AllocateMemory(creation_context.GetGpuInfo(), creation_context.context));
  BindMemoryToOperations();
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.RestoreDeserialized(creation_context));
  }
  RETURN_IF_ERROR(UpdateParams());
  if (external_mutable_tensors_.empty()) {
    // using recordable queue only when no mutable external tensors
    InitRecordableQueue(env);
  }
  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }
  ReleaseCPURepresentation();
  return absl::OkStatus();
}

void InferenceContext::CopyFromGpuModel(GpuModel* gpu_model) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& variable_input : gpu_model->variable_ids_and_refs) {
    variable_ids_and_refs_[variable_input.first] = variable_input.second;
  }
  for (const auto& output : gpu_model->output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model->nodes.size());
  for (int i = 0; i < gpu_model->nodes.size(); ++i) {
    nodes_[i].cl_operation.Init(std::move(gpu_model->nodes[i].gpu_operation));
    nodes_[i].inputs = gpu_model->nodes[i].inputs;
    nodes_[i].outputs = gpu_model->nodes[i].outputs;
    nodes_[i].name = gpu_model->nodes[i].name;
  }
  const_tensors_descs_ = std::move(gpu_model->const_tensors);
  tensors_descs_ = std::move(gpu_model->tensors);
}

void InferenceContext::InitRecordableQueue(Environment* env) {
  std::vector<ClOperation*> ops(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    ops[i] = &nodes_[i].cl_operation;
  }
  recordable_queue_ = CreateRecordableQueue(ops, env->device(), env->context());
}

absl::Status InferenceContext::InitFromGraphWithTransforms(
    const CreateGpuModelInfo& create_info, GraphFloat32* graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  RETURN_IF_ERROR(RunGraphTransformsForGpuModel(graph));
  RETURN_IF_ERROR(InitFromGraph(create_info, *graph, env, serialized_model));
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
  if (external_immutable_tensors_.find(id) !=
      external_immutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (external_mutable_tensors_.find(id) !=
             external_mutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return TensorMemoryType::kVariable;
  } else if (IsBufferBased(gpu_info, tensors_descs_[id].storage_type)) {
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
      const auto& t = tensors_descs_[value_and_ref_value.first];
      const auto& shape = t.shape;
      const auto& descriptor = t;

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
    const auto& t = tensors_descs_[usage.first];
    const auto& shape = t.shape;
    const auto& descriptor = t;
    const size_t element_size = SizeOf(descriptor.data_type);
    size_t buffer_size;
    if (descriptor.storage_type == TensorStorageType::TEXTURE_2D ||
        descriptor.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
      const size_t bytes_per_pixel =
          element_size *
          (descriptor.storage_type == TensorStorageType::TEXTURE_2D ? 4
                                                                    : shape.c);
      const size_t width = shape.b * shape.w;
      const size_t height = shape.h * DivideRoundUp(shape.c, 4);
      size_t width_pixel_alignment = gpu_info.opencl_info.image_pitch_alignment;
      if (gpu_info.IsAdreno() && width_pixel_alignment % bytes_per_pixel == 0) {
        width_pixel_alignment /= bytes_per_pixel;
      }
      const size_t width_aligned = AlignByN(width, width_pixel_alignment);
      buffer_size = width_aligned * bytes_per_pixel * height;
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
      const auto& shape_5d = tensors_descs_[t.first].shape;
      const auto shape = BHWC(shape_5d.b, shape_5d.h, shape_5d.w, shape_5d.c);
      const int buffer_index = use_offset_assignment
                                   ? tensor_index
                                   : buffer_assignment.object_ids[tensor_index];
      if (t.second.storage_type == TensorStorageType::TEXTURE_2D ||
          t.second.storage_type == TensorStorageType::SINGLE_TEXTURE_2D) {
        const size_t bytes_per_pixel =
            SizeOf(t.second.data_type) *
            (t.second.storage_type == TensorStorageType::TEXTURE_2D ? 4
                                                                    : shape.c);
        size_t width_pixel_alignment =
            gpu_info.opencl_info.image_pitch_alignment;
        if (gpu_info.IsAdreno() &&
            width_pixel_alignment % bytes_per_pixel == 0) {
          width_pixel_alignment /= bytes_per_pixel;
        }
        RETURN_IF_ERROR(CreateSharedImage2DBufferTensor(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), shape,
            t.second, width_pixel_alignment,
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
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(gpu_info, t.first) !=
          TensorMemoryType::kStrongShape) {
        continue;
      }
      const auto& shape = tensors_descs_[t.first].shape;
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

absl::Status InferenceContext::SetTensor(const ValueId& tensor_id,
                                         Tensor* tensor_ptr) {
  auto it = external_mutable_tensors_.find(tensor_id);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  external_mutable_tensors_[tensor_id] = tensor_ptr;
  for (int node_index : external_tensor_to_nodes_[tensor_id]) {
    auto& node = nodes_[node_index];
    for (int i = 0; i < node.inputs.size(); ++i) {
      if (node.inputs[i] == tensor_id) {
        RETURN_IF_ERROR(node.cl_operation.SetSrcTensor(i, tensor_ptr));
      }
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      if (node.outputs[i] == tensor_id) {
        RETURN_IF_ERROR(node.cl_operation.SetDstTensor(i, tensor_ptr));
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

absl::Status InferenceContext::AddToQueue(CLCommandQueue* queue) {
  if (recordable_queue_ && recordable_queue_->IsSupported()) {
    return recordable_queue_->Execute(queue);
  }
  if (execution_hints_.need_manual_release) {
    if (execution_hints_.prev_enqueue_start_point.is_valid()) {
      execution_hints_.prev_enqueue_start_point.Wait();
    }
    RETURN_IF_ERROR(
        queue->EnqueueEvent(&execution_hints_.prev_enqueue_start_point));
  }
  int counter = 0;
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
    counter++;
    if (execution_hints_.flush_periodically &&
        counter % execution_hints_.flush_period == 0) {
      clFlush(queue->queue());
    }
  }
  if (execution_hints_.need_flush) {
    clFlush(queue->queue());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::ProfileTime(ProfilingCommandQueue* queue,
                                           ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (auto& node : nodes_) {
    queue->SetEventsLabel(node.name);
    RETURN_IF_ERROR(node.cl_operation.AddToQueue(queue));
  }
  RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();

  if (!(gpu_info_.IsMali() || gpu_info_.IsPowerVR())) {
    return absl::OkStatus();
  }

  if (gpu_info_.IsMali()) {
    queue->ResetMeasurements();
    for (int i = 0; i < nodes_.size(); ++i) {
      queue->SetEventsLabel(nodes_[i].name);
      const double times =
          16.0 / absl::ToDoubleMilliseconds(result->dispatches[i].duration);
      const int n = std::min(256.0, std::max(2.0, times));
      RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
    }
    RETURN_IF_ERROR(queue->WaitForCompletion());
    *result = queue->GetProfilingInfo();
    return absl::OkStatus();
  }

  if (gpu_info_.IsPowerVR()) {
    queue->ResetMeasurements();
    for (int i = 0; i < nodes_.size(); ++i) {
      queue->SetEventsLabel(nodes_[i].name);
      const double times =
          32.0 / absl::ToDoubleMilliseconds(result->dispatches[i].duration);
      const int n = std::min(64.0, std::max(4.0, times));
      RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
    }
    RETURN_IF_ERROR(queue->WaitForCompletion());
    *result = queue->GetProfilingInfo();

    queue->ResetMeasurements();
    for (int i = 0; i < nodes_.size(); ++i) {
      queue->SetEventsLabel(nodes_[i].name);
      const double times =
          128.0 / absl::ToDoubleMilliseconds(result->dispatches[i].duration);
      const int n = std::min(1024.0, std::max(4.0, times));
      RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
    }
    RETURN_IF_ERROR(queue->WaitForCompletion());
    *result = queue->GetProfilingInfo();
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::Status InferenceContext::Profile(ProfilingCommandQueue* queue,
                                       ProfilingInfo* result) {
  RETURN_IF_ERROR(ProfileTime(queue, result));
  for (int i = 0; i < nodes_.size(); ++i) {
    uint64_t read_size = 0;
    for (auto& src_id : nodes_[i].inputs) {
      read_size += GetTensor(src_id)->GetMemorySizeInBytes();
    }
    const auto& gpu_op = nodes_[i].cl_operation.GetGpuOperation();
    read_size += gpu_op.const_args_size_;
    uint64_t write_size = 0;
    for (auto& dst_id : nodes_[i].outputs) {
      write_size += GetTensor(dst_id)->GetMemorySizeInBytes();
    }
    result->dispatches[i].flops = gpu_op.flops_;
    result->dispatches[i].read_mem_size = read_size;
    result->dispatches[i].write_mem_size = write_size;
  }

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

uint64_t InferenceContext::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& node : nodes_) {
    total_size += node.cl_operation.GetGpuOperation().const_args_size_;
  }
  return total_size;
}

Tensor* InferenceContext::GetTensor(ValueId id) {
  if (external_immutable_tensors_.find(id) !=
      external_immutable_tensors_.end()) {
    return external_immutable_tensors_[id];
  } else if (external_mutable_tensors_.find(id) !=
             external_mutable_tensors_.end()) {
    return external_mutable_tensors_[id];
  } else if (const_tensors_.find(id) != const_tensors_.end()) {
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
