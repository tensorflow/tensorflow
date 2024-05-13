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
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/serialization_generated.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model_generated.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {
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
size_t TotalSize(const ObjectsAssignment<size_t>& assignment,
                 size_t alignment = 1) {
  size_t total_size = 0;
  for (auto object_size : assignment.object_sizes) {
    total_size += AlignByN(object_size, alignment);
  }
  return total_size;
}

TensorType GetTensorType(const GpuModel& gpu_model,
                         const CreateGpuModelInfo* create_info,
                         const GpuInfo& gpu_info, ValueId id) {
  bool is_variable = false;
  for (int i = 0; i < gpu_model.variable_ids_and_refs.size(); ++i) {
    if (gpu_model.variable_ids_and_refs[i].first == id) {
      is_variable = true;
      break;
    }
  }
  if (is_variable) {
    return TensorType::kVariable;
  } else if (create_info &&
             (create_info->external_immutable_tensors.find(id) !=
                  create_info->external_immutable_tensors.end() ||
              create_info->external_mutable_tensors.find(id) !=
                  create_info->external_mutable_tensors.end())) {
    return TensorType::kExternal;
  } else if (gpu_model.const_tensors.find(id) !=
             gpu_model.const_tensors.end()) {
    return TensorType::kConst;
  } else {
    return TensorType::kRuntime;
  }
}

void GetUsages(const GpuModel& model,
               const std::function<bool(ValueId)>& functor,
               std::map<ValueId, int2>* usages) {
  for (const auto& in_id : model.input_ids_and_refs) {
    if (functor(in_id.first)) {
      AddUsage(in_id.first, 0, usages);
    }
  }
  for (int op_index = 0; op_index < model.nodes.size(); ++op_index) {
    for (auto input_id : model.nodes[op_index].inputs) {
      if (functor(input_id)) {
        AddUsage(input_id, op_index, usages);
      }
    }
    for (auto output_id : model.nodes[op_index].outputs) {
      if (functor(output_id)) {
        AddUsage(output_id, op_index, usages);
      }
    }
  }
  for (const auto& out_id : model.output_ids_and_refs) {
    if (functor(out_id.first)) {
      AddUsage(out_id.first, model.nodes.size(), usages);
    }
  }
}

absl::Status GetBufferAsignment(
    const GpuModel& gpu_model, const CreateGpuModelInfo* create_info,
    const GpuInfo& gpu_info,
    std::vector<TensorUsageRecord<size_t>>* buffer_usage_records,
    std::map<ValueId, int>* graph_ids_to_shared_buffer_tensors,
    ObjectsAssignment<size_t>* buffer_assignment,
    OffsetsAssignment* offset_assignment, bool* use_offset_assignment,
    bool* is_sub_buffers_supported) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(
      gpu_model,
      [&gpu_model, &gpu_info, &create_info](ValueId id) {
        return GetTensorType(gpu_model, create_info, gpu_info, id) ==
                   TensorType::kRuntime &&
               IsBufferBased(gpu_info,
                             gpu_model.tensors.at(id).GetStorageType());
      },
      &buffer_usages);

  bool has_buffer_based_images = false;
  for (auto& usage : buffer_usages) {
    const auto& t = gpu_model.tensors.at(usage.first);
    const auto& shape = t.GetBHWDCShape();
    const auto& descriptor = t;
    const size_t element_size = SizeOf(descriptor.GetDataType());
    size_t buffer_size;
    if (descriptor.GetStorageType() == TensorStorageType::TEXTURE_2D ||
        descriptor.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D) {
      has_buffer_based_images = true;
      const size_t bytes_per_pixel =
          element_size *
          (descriptor.GetStorageType() == TensorStorageType::TEXTURE_2D
               ? 4
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
      if (descriptor.GetStorageType() == TensorStorageType::IMAGE_BUFFER) {
        has_buffer_based_images = true;
      }
      buffer_size =
          shape.b * shape.w * shape.h * AlignByN(shape.c, 4) * element_size;
    }
    if (graph_ids_to_shared_buffer_tensors) {
      (*graph_ids_to_shared_buffer_tensors)[usage.first] =
          buffer_usage_records->size();
    }
    buffer_usage_records->push_back({buffer_size,
                                     static_cast<TaskId>(usage.second.x),
                                     static_cast<TaskId>(usage.second.y)});
  }

  RETURN_IF_ERROR(AssignObjectsToTensors(
      *buffer_usage_records, MemoryStrategy::GREEDY_BEST, buffer_assignment));

  *is_sub_buffers_supported =
      (!has_buffer_based_images && gpu_info.IsCL11OrHigher()) ||
      CanUseSubBufferForImage2d(gpu_info);
  const size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);

  *use_offset_assignment = false;
  if (*is_sub_buffers_supported) {
    RETURN_IF_ERROR(AssignOffsetsToTensors(
        *buffer_usage_records, MemoryStrategy::GREEDY_BY_SIZE,
        offset_assignment, base_align_bytes));
    if (offset_assignment->total_size <= TotalSize(*buffer_assignment) &&
        offset_assignment->total_size <= gpu_info.GetMaxBufferSize()) {
      *use_offset_assignment = true;
    }
  }
  return absl::OkStatus();
}

absl::Status ClarifyWithCommandBuffer(ProfilingCommandQueue* queue,
                                      int num_tries, double cb_duration_ms,
                                      const std::vector<CLNode*>& nodes,
                                      std::vector<double>* time_ns) {
  auto get_tasks_count = [&](int node_index) {
    const int tasks_count = cb_duration_ms / ((*time_ns)[node_index] * 1e-6f);
    return std::min(256, std::max(1, tasks_count));
  };

  std::vector<CLCommandBuffer> cbs(nodes.size() * num_tries);
  for (int t = 0; t < num_tries; ++t) {
    for (int node_index = 0; node_index < nodes.size(); ++node_index) {
      const int index = t * nodes.size() + node_index;
      auto& cb = cbs[index];
      RETURN_IF_ERROR(cb.Init(queue, /*simultaneous_use=*/false));
      const int num_kernels_in_cb = get_tasks_count(node_index);
      for (int j = 0; j < num_kernels_in_cb; ++j) {
        RETURN_IF_ERROR(nodes[node_index]->cl_operation.AddToCommanBuffer(
            cb.GetCommandBuffer()));
      }
      RETURN_IF_ERROR(cb.Finalize());
    }
  }
  std::vector<CLEvent> events(nodes.size() * num_tries);
  for (int t = 0; t < num_tries; ++t) {
    for (int node_index = 0; node_index < nodes.size(); ++node_index) {
      const int index = t * nodes.size() + node_index;
      RETURN_IF_ERROR(cbs[index].Enqueue(queue, &events[index]));
    }
  }
  clFinish(queue->queue());
  for (int node_index = 0; node_index < nodes.size(); ++node_index) {
    double min_time_ns = std::numeric_limits<double>::max();
    for (int t = 0; t < num_tries; ++t) {
      const int num_kernels_in_cb = get_tasks_count(node_index);
      double time_ns = events[t * nodes.size() + node_index].GetEventTimeNs() /
                       num_kernels_in_cb;
      min_time_ns = std::min(min_time_ns, time_ns);
    }
    (*time_ns)[node_index] = min_time_ns;
  }
  return absl::OkStatus();
}

}  // namespace

void InferenceContext::ExecutionHints::Init(const GpuInfo& gpu_info) {
  if (gpu_info.IsMali()) {
    need_flush = true;
    need_manual_release = gpu_info.mali_info.IsValhall() ? false : true;

    flush_periodically = true;
    flush_period = 24;
  } else if (gpu_info.IsPowerVR()) {
    need_flush = true;
    flush_periodically = true;
    // Some Ge8xxx devices are slower without frequent periodic flushing.
    flush_period =
        gpu_info.powervr_info.IsBetterThan(PowerVRGpu::kRogueGm9xxx) ? 16 : 4;
  } else if (gpu_info.IsAdreno() &&
             !gpu_info.adreno_info.IsBetterThan(AdrenoGpu::kAdreno630)) {
    // Adreno620 or lower devices has smaller GPU buffer.
    flush_periodically = true;
    flush_period = 16;
  }
  // clvk has inside to know when to flush, do not do it at the application
  // level.
  if (gpu_info.IsApiOpenCl() && gpu_info.opencl_info.IsCLVK()) {
    need_flush = false;
    flush_periodically = false;
  }
}

absl::Status InferenceContext::InitFromGraph(
    const CreateGpuModelInfo& create_info, const GraphFloat32& graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  GpuModel gpu_model;
  RETURN_IF_ERROR(GraphToGpuModel(graph, create_info,
                                  env->GetDevicePtr()->GetInfo(), &gpu_model));
  return InitFromGpuModel(create_info, &gpu_model, env, serialized_model);
}

absl::Status InferenceContext::InitFromGpuModel(
    const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
    Environment* env, std::vector<uint8_t>* serialized_model,
    Buffer* shared_buffer) {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = tflite::gpu::Encode(*gpu_model, &builder);
  }
  shared_buffers_parent_ptr_ = shared_buffer;
  RETURN_IF_ERROR(AllocateMemory(*gpu_model, env->GetDevicePtr()->GetInfo(),
                                 &create_info, &env->context()));
  InitFromGpuModel(gpu_model);

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
    RETURN_IF_ERROR(
        CreateTensor(env->context(),
                     gpu_model->tensors[external_tensor.first],
                     &temp_external_tensors[external_tensor.first]));
    external_mutable_tensors_[external_tensor.first] =
        &temp_external_tensors[external_tensor.first];
  }
  PrepareExternal();
  execution_hints_.Init(env->device().GetInfo());
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
    auto encoded_fb = Encode(*env->GetDevicePtr(), *env->program_cache(),
                             gpu_model_fb, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AddToCommanBuffer(cl_command_buffer_khr cb) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.cl_operation.AddToCommanBuffer(cb));
  }
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
  std::string platform_version(decoded_fb->driver_version()->c_str(),
                               decoded_fb->driver_version()->size());
  if (env->GetDevicePtr()->GetPlatformVersion() != platform_version) {
    return absl::InvalidArgumentError(
        "OpenCL driver changed, model respresentation invalid, must be "
        "regenerated.");
  }
  GpuModel gpu_model;
  RETURN_IF_ERROR(tflite::gpu::Decode(decoded_fb->gpu_model(), &gpu_model));
  RETURN_IF_ERROR(AllocateMemory(gpu_model, env->GetDevicePtr()->GetInfo(),
                                 create_info, &env->context()));
  InitFromGpuModel(&gpu_model);

  // deserializing kernels into program_cache
  for (auto binary_program_fb : *decoded_fb->binary_programs()) {
    RETURN_IF_ERROR(env->program_cache()->AddProgramBinary(
        env->context(), *env->GetDevicePtr(), binary_program_fb->fingerprint(),
        absl::MakeSpan(binary_program_fb->binary()->data(),
                       binary_program_fb->binary()->size())));
  }

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
      RETURN_IF_ERROR(
          CreateTensor(env->context(),
                       gpu_model.tensors[external_tensor.first],
                       &temp_external_tensors[external_tensor.first]));
      external_mutable_tensors_[external_tensor.first] =
          &temp_external_tensors[external_tensor.first];
    }
  }
  PrepareExternal();

  execution_hints_.Init(env->device().GetInfo());

  BindMemoryToOperations();
  for (int i = 0; i < nodes_.size(); ++i) {
    uint64_t fingerprint = (*decoded_fb->fingerprints_per_node())[i];
    int3 wg_size;
    wg_size.x = (*decoded_fb->tuned_work_group_sizes_per_node())[i]->x();
    wg_size.y = (*decoded_fb->tuned_work_group_sizes_per_node())[i]->y();
    wg_size.z = (*decoded_fb->tuned_work_group_sizes_per_node())[i]->z();
    RETURN_IF_ERROR(nodes_[i].cl_operation.RestoreDeserialized(
        *env->program_cache(), fingerprint, env->GetDevicePtr()->GetInfo(),
        wg_size, &env->context()));
  }
  RETURN_IF_ERROR(UpdateParams());
  if (external_mutable_tensors_.empty()) {
    // using recordable queue only when no mutable external tensors
    InitRecordableQueue(env);
  }
  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }
  return absl::OkStatus();
}

void InferenceContext::InitFromGpuModel(GpuModel* gpu_model) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
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

absl::Status InferenceContext::AllocateMemory(
    const GpuModel& gpu_model, const GpuInfo& gpu_info,
    const CreateGpuModelInfo* create_info, CLContext* context) {
  RETURN_IF_ERROR(AllocateConstTensors(gpu_model, context));
  RETURN_IF_ERROR(AllocateVariableTensors(gpu_model, context));
  RETURN_IF_ERROR(
      AllocateBufferBasedTensors(gpu_model, gpu_info, create_info, context));
  RETURN_IF_ERROR(
      AllocateStrongShapesTensors(gpu_model, gpu_info, create_info, context));
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateConstTensors(const GpuModel& gpu_model,
                                                    CLContext* context) {
  for (auto& description : gpu_model.const_tensors) {
    RETURN_IF_ERROR(const_tensors_[description.first].CreateFromDescriptor(
        description.second, context));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateVariableTensors(
    const GpuModel& gpu_model, CLContext* context) {
  for (const auto& variable_input : gpu_model.variable_ids_and_refs) {
    variable_ids_and_refs_[variable_input.first] = variable_input.second;
  }

  std::map<ValueId, int> ref_value_to_tensor_index;

  for (auto value_and_ref_value : variable_ids_and_refs_) {
    if (ref_value_to_tensor_index.find(value_and_ref_value.second) ==
        ref_value_to_tensor_index.end()) {
      auto it = gpu_model.tensors.find(value_and_ref_value.first);
      if (it == gpu_model.tensors.end()) {
        return absl::InternalError("No variable tensor with this id.");
      }
      RETURN_IF_ERROR(
          CreateTensor(*context, it->second,
                       &variable_tensors_[value_and_ref_value.second]));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateBufferBasedTensors(
    const GpuModel& gpu_model, const GpuInfo& gpu_info,
    const CreateGpuModelInfo* create_info, CLContext* context) {
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  ObjectsAssignment<size_t> buffer_assignment;
  OffsetsAssignment offset_assignment;
  bool use_offset_assignment;
  bool is_sub_buffers_supported;
  RETURN_IF_ERROR(GetBufferAsignment(
      gpu_model, create_info, gpu_info, &buffer_usage_records,
      &graph_ids_to_shared_buffer_tensors_, &buffer_assignment,
      &offset_assignment, &use_offset_assignment, &is_sub_buffers_supported));
  const size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);

  if (buffer_usage_records.empty()) {
    return absl::OkStatus();
  }

  if (use_offset_assignment) {
    if (!shared_buffers_parent_ptr_) {
      Buffer shared_buffer;
      RETURN_IF_ERROR(CreateReadWriteBuffer(offset_assignment.total_size,
                                            context, &shared_buffer));
      shared_buffers_parent_ =
          std::make_unique<Buffer>(std::move(shared_buffer));
      shared_buffers_parent_ptr_ = shared_buffers_parent_.get();
    } else if (shared_buffers_parent_ptr_->GetMemorySizeInBytes() <
               offset_assignment.total_size) {
      return absl::FailedPreconditionError(
          "Externally provided buffer not big enough.");
    }
    shared_buffers_.resize(offset_assignment.offsets.size());
    for (int i = 0; i < offset_assignment.offsets.size(); ++i) {
      RETURN_IF_ERROR(CreateReadWriteSubBuffer(
          *shared_buffers_parent_ptr_, offset_assignment.offsets[i],
          buffer_usage_records[i].tensor_size, context, &shared_buffers_[i]));
    }
  } else {
    const size_t total_size = TotalSize(buffer_assignment, base_align_bytes);
    if (is_sub_buffers_supported && total_size <= gpu_info.GetMaxBufferSize()) {
      // use single parent buffer:
      if (!shared_buffers_parent_ptr_) {
        Buffer shared_buffer;
        RETURN_IF_ERROR(
            CreateReadWriteBuffer(total_size, context, &shared_buffer));
        shared_buffers_parent_ =
            std::make_unique<Buffer>(std::move(shared_buffer));
        shared_buffers_parent_ptr_ = shared_buffers_parent_.get();
      } else if (shared_buffers_parent_ptr_->GetMemorySizeInBytes() <
                 total_size) {
        return absl::FailedPreconditionError(
            "Externally provided buffer not big enough.");
      }

      shared_buffers_.resize(buffer_assignment.object_sizes.size());
      size_t offset = 0;
      for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
        const size_t aligned_size =
            AlignByN(buffer_assignment.object_sizes[i], base_align_bytes);
        RETURN_IF_ERROR(CreateReadWriteSubBuffer(*shared_buffers_parent_ptr_,
                                                 offset, aligned_size, context,
                                                 &shared_buffers_[i]));
        offset += aligned_size;
      }
    } else {
      shared_buffers_.resize(buffer_assignment.object_sizes.size());
      for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
        RETURN_IF_ERROR(CreateReadWriteBuffer(buffer_assignment.object_sizes[i],
                                              context, &shared_buffers_[i]));
      }
    }
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  bool create_model_output_tensors = false;
  for (auto& node : gpu_model.nodes) {
    // Handle node input tensors.
    std::vector<ValueId> node_tensor_ids = node.inputs;
    // Handle node output tensors.
    node_tensor_ids.insert(node_tensor_ids.end(), node.outputs.begin(),
                           node.outputs.end());
    if (!create_model_output_tensors) {
      // Handle graph output tensors.
      for (const auto& output : gpu_model.output_ids_and_refs) {
        node_tensor_ids.push_back(output.first);
      }
      create_model_output_tensors = true;
    }
    for (auto& tensor_id : node_tensor_ids) {
      if (GetTensorType(gpu_model, create_info, gpu_info, tensor_id) !=
          TensorType::kRuntime) {
        continue;
      }
      const auto& tensor_desc = gpu_model.tensors.at(tensor_id);
      if (!IsBufferBased(gpu_info, tensor_desc.GetStorageType())) {
        continue;
      }
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const int buffer_index = use_offset_assignment
                                   ? tensor_index
                                   : buffer_assignment.object_ids[tensor_index];
      if (tensor_desc.GetStorageType() == TensorStorageType::TEXTURE_2D ||
          tensor_desc.GetStorageType() ==
              TensorStorageType::SINGLE_TEXTURE_2D) {
        const size_t bytes_per_pixel =
            SizeOf(tensor_desc.GetDataType()) *
            (tensor_desc.GetStorageType() == TensorStorageType::TEXTURE_2D
                 ? 4
                 : tensor_desc.GetBHWCShape().c);
        size_t width_pixel_alignment =
            gpu_info.opencl_info.image_pitch_alignment;
        if (gpu_info.IsAdreno() &&
            width_pixel_alignment % bytes_per_pixel == 0) {
          width_pixel_alignment /= bytes_per_pixel;
        }
        RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), tensor_desc,
            width_pixel_alignment, &shared_buffer_tensors_[tensor_index]));
      } else {
        RETURN_IF_ERROR(CreateTensorShared(
            *context, shared_buffers_[buffer_index].GetMemoryPtr(), tensor_desc,
            &shared_buffer_tensors_[tensor_index]));
      }
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateStrongShapesTensors(
    const GpuModel& gpu_model, const GpuInfo& gpu_info,
    const CreateGpuModelInfo* create_info, CLContext* context) {
  std::map<ValueId, int2> usages;
  GetUsages(
      gpu_model,
      [&gpu_model, &gpu_info, &create_info](ValueId id) {
        return GetTensorType(gpu_model, create_info, gpu_info, id) ==
                   TensorType::kRuntime &&
               !IsBufferBased(gpu_info,
                              gpu_model.tensors.at(id).GetStorageType());
      },
      &usages);

  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      return tensor_desc == t.tensor_desc &&
             tensor_desc.GetBHWDCShape() == t.tensor_desc.GetBHWDCShape();
    }
  };

  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  std::map<ValueId, ValueId> remap_from_graph_ids;
  for (auto& usage : usages) {
    remap_from_graph_ids[usage.first] = usage_records.size();
    usage_records.push_back({{gpu_model.tensors.at(usage.first)},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  for (auto& node : gpu_model.nodes) {
    std::vector<ValueId> node_tensor_ids = node.inputs;
    node_tensor_ids.insert(node_tensor_ids.end(), node.outputs.begin(),
                           node.outputs.end());
    for (auto& tensor_id : node_tensor_ids) {
      if (GetTensorType(gpu_model, create_info, gpu_info, tensor_id) !=
          TensorType::kRuntime) {
        continue;
      }
      const auto& tensor_desc = gpu_model.tensors.at(tensor_id);
      if (IsBufferBased(gpu_info, tensor_desc.GetStorageType())) {
        continue;
      }
      const auto id = assignment.object_ids[remap_from_graph_ids[tensor_id]];
      graph_ids_to_strong_shape_tensors_[tensor_id] = id;
      const auto& it = strong_shape_tensors_.find(id);
      if (it == strong_shape_tensors_.end()) {
        RETURN_IF_ERROR(
            CreateTensor(*context, tensor_desc, &strong_shape_tensors_[id]));
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
  // Cache tuned CL operations. Multiple CL operations might share the
  // same kernel but use different inputs, which might require different working
  // group setups. Therefore, we store a vector of tuned cl operations for each
  // kernel and match in a second stage based on equal CL arguments.
  typedef std::reference_wrapper<const ClOperation> ClOperationRef;
  absl::flat_hash_map<uint64_t, std::vector<ClOperationRef>> tuned_ops;

  for (auto& node : nodes_) {
    uint64_t fingerprint = node.cl_operation.GetKernelFingerprint();
    auto cl_ops_it = tuned_ops.find(fingerprint);
    bool found_cached_cl_op = false;
    if (cl_ops_it != tuned_ops.end()) {
      for (const auto& cl_op : cl_ops_it->second) {
        if (!node.cl_operation.HasEqualScalarArguments(cl_op)) {
          continue;
        }
        // Fingerprint and CLArguments match, so we reuse the work group size.
        node.cl_operation.GetGpuOperation().work_group_size_ =
            cl_op.get().GetGpuOperation().work_group_size_;
        node.cl_operation.GetGpuOperation().RecalculateWorkGroupsCount();
        found_cached_cl_op = true;
      }
    }
    if (found_cached_cl_op) {
      continue;
    }
    RETURN_IF_ERROR(
        node.cl_operation.Tune(tuning_type, gpu_info, profiling_queue));
    tuned_ops[fingerprint].emplace_back(std::cref(node.cl_operation));
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

absl::Status InferenceContext::ClarifyTimeWithCommandBuffer(
    ProfilingCommandQueue* queue, ProfilingInfo* result) {
  const int num_tries = 3;
  const double cb_duration_ms = 10.0;  // looks like enough
  const int node_group_count = 8;  // Current PowerVR drivers have issues with
                                   // big amount of CB or big CB.
  for (int node_index = 0; node_index < nodes_.size();
       node_index += node_group_count) {
    std::vector<CLNode*> nodes_to_clarify;
    std::vector<double> times_ns;
    for (int i = 0; i < node_group_count && node_index + i < nodes_.size();
         ++i) {
      nodes_to_clarify.push_back(&nodes_[node_index + i]);
      times_ns.push_back(absl::ToDoubleNanoseconds(
          result->dispatches[node_index + i].duration));
    }
    RETURN_IF_ERROR(ClarifyWithCommandBuffer(queue, num_tries, cb_duration_ms,
                                             nodes_to_clarify, &times_ns));
    for (int i = 0; i < node_group_count && node_index + i < nodes_.size();
         ++i) {
      result->dispatches[node_index + i].duration =
          absl::Nanoseconds(times_ns[i]);
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::ClarifyTimeMultipleEnqueue(
    double ops_total_duration_ms, int min_ops, int max_ops,
    ProfilingCommandQueue* queue, ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (int i = 0; i < nodes_.size(); ++i) {
    queue->SetEventsLabel(nodes_[i].name);
    const int times =
        ops_total_duration_ms /
        absl::ToDoubleMilliseconds(result->dispatches[i].duration);
    const int n = std::min(max_ops, std::max(min_ops, times));
    RETURN_IF_ERROR(nodes_[i].cl_operation.AddToQueueNTimes(queue, n));
  }
  RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();
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
    return ClarifyTimeMultipleEnqueue(/*ops_total_duration_ms=*/16.0,
                                      /*min_ops=*/2, /*max_ops=*/256, queue,
                                      result);
  }

  if (gpu_info_.IsPowerVR()) {
    if (gpu_info_.SupportsExtension("cl_khr_command_buffer")) {
      RETURN_IF_ERROR(ClarifyTimeWithCommandBuffer(queue, result));
      RETURN_IF_ERROR(ClarifyTimeWithCommandBuffer(queue, result));
    } else {
      RETURN_IF_ERROR(ClarifyTimeMultipleEnqueue(/*ops_total_duration_ms=*/32.0,
                                                 /*min_ops=*/4, /*max_ops=*/64,
                                                 queue, result));
      return ClarifyTimeMultipleEnqueue(/*ops_total_duration_ms=*/128.0,
                                        /*min_ops=*/4, /*max_ops=*/1024, queue,
                                        result);
    }
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
  if (shared_buffers_parent_) {
    total_memory += shared_buffers_parent_->GetMemorySizeInBytes();
  }

  return total_memory;
}

uint64_t InferenceContext::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& node : nodes_) {
    total_size += node.cl_operation.GetGpuOperation().const_args_size_;
  }
  for (const auto& t : const_tensors_) {
    total_size += t.second.GetMemorySizeInBytes();
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
  Tensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, queue);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorFloat32* result) {
  const Tensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, queue));
  desc.DownloadData(result);
  return absl::OkStatus();
}

flatbuffers::Offset<data::InferenceContext> InferenceContext::Encode(
    const CLDevice& device, const ProgramCache& program_cache,
    flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<tflite::gpu::data::Int3>> work_groups_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto work_group_fb =
        tflite::gpu::Encode(nodes_[i].cl_operation.GetWorkGroupSize(), builder);
    work_groups_fb.push_back(work_group_fb);
  }
  auto work_groups_fb_vec = builder->CreateVector(work_groups_fb);
  std::vector<uint64_t> node_fingerprints(nodes_.size());
  for (int i = 0; i < nodes_.size(); ++i) {
    node_fingerprints[i] = nodes_[i].cl_operation.GetKernelFingerprint();
  }
  auto node_fingerprints_fb = builder->CreateVector(node_fingerprints);

  std::set<uint64_t> fingerprints;
  for (const auto& node : nodes_) {
    fingerprints.insert(node.cl_operation.GetKernelFingerprint());
  }
  std::vector<flatbuffers::Offset<data::BinaryProgram>> binary_programs_fb;
  for (auto fingerprint : fingerprints) {
    std::vector<uint8_t> program_binary;
    program_cache.GetProgramBinary(fingerprint, &program_binary).IgnoreError();
    auto binary_fb = builder->CreateVector(program_binary);
    data::BinaryProgramBuilder program_builder(*builder);
    program_builder.add_fingerprint(fingerprint);
    program_builder.add_binary(binary_fb);
    binary_programs_fb.push_back(program_builder.Finish());
  }
  auto binary_programs_fb_vec = builder->CreateVector(binary_programs_fb);
  auto driver_version = builder->CreateString(device.GetPlatformVersion());

  data::InferenceContextBuilder inf_builder(*builder);
  inf_builder.add_gpu_model(gpu_model_fb);
  inf_builder.add_driver_version(driver_version);
  inf_builder.add_binary_programs(binary_programs_fb_vec);
  inf_builder.add_tuned_work_group_sizes_per_node(work_groups_fb_vec);
  inf_builder.add_fingerprints_per_node(node_fingerprints_fb);
  return inf_builder.Finish();
}

absl::Status GetInOutRefs(const absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto fb_inference = data::GetInferenceContext(serialized_model.data());
  if (in_refs) {
    in_refs->clear();
    for (auto in_fb : *fb_inference->gpu_model()->input_refs()) {
      in_refs->push_back(in_fb);
    }
  }
  if (out_refs) {
    out_refs->clear();
    for (auto out_fb : *fb_inference->gpu_model()->output_refs()) {
      out_refs->push_back(out_fb);
    }
  }
  return absl::OkStatus();
}

absl::Status GetTotalBufferSizeForTensors(const GpuModel& gpu_model,
                                          const CreateGpuModelInfo& create_info,
                                          const GpuInfo& gpu_info,
                                          uint64_t* result) {
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  ObjectsAssignment<size_t> buffer_assignment;
  OffsetsAssignment offset_assignment;
  bool use_offset_assignment;
  bool is_sub_buffers_supported;
  RETURN_IF_ERROR(GetBufferAsignment(
      gpu_model, &create_info, gpu_info, &buffer_usage_records, nullptr,
      &buffer_assignment, &offset_assignment, &use_offset_assignment,
      &is_sub_buffers_supported));
  if (use_offset_assignment) {
    *result = offset_assignment.total_size;
    return absl::OkStatus();
  }

  const size_t base_align_bytes =
      std::max<size_t>(gpu_info.opencl_info.base_addr_align_in_bits >> 3, 1);
  *result = TotalSize(buffer_assignment, base_align_bytes);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
