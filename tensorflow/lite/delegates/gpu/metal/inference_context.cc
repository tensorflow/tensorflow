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
#include <numeric>
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
#include "tensorflow/lite/delegates/gpu/common/task/serialization_base.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

// returns true if actual memory for this storage type is buffer
bool IsBufferBased(const GpuInfo& gpu_info, const TensorStorageType& type) {
  const bool family_apple1 =
      gpu_info.IsApple() && gpu_info.apple_info.IsFamilyApple1();
  if (!family_apple1 && (type == TensorStorageType::TEXTURE_2D ||
                         type == TensorStorageType::SINGLE_TEXTURE_2D)) {
    return true;
  }
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER;
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

// Calculates the total size of the assignment.
size_t TotalSize(const ObjectsAssignment<size_t>& assignment,
                 size_t alignment = 1) {
  size_t total_size = 0;
  for (auto object_size : assignment.object_sizes) {
    total_size += AlignByN(object_size, alignment);
  }
  return total_size;
}

flatbuffers::Offset<data::MetalProgram> EncodeProgram(
    const std::string& code, const std::map<std::string, std::string>& defines,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> names_fb;
  std::vector<flatbuffers::Offset<flatbuffers::String>> expressions_fb;
  for (auto& define : defines) {
    names_fb.push_back(builder->CreateString(define.first));
    expressions_fb.push_back(builder->CreateString(define.second));
  }
  auto names_fb_vec = builder->CreateVector(names_fb);
  auto expressions_fb_vec = builder->CreateVector(expressions_fb);
  auto code_fb = builder->CreateString(code);
  data::MetalProgramBuilder program_builder(*builder);
  program_builder.add_define_names(names_fb_vec);
  program_builder.add_define_expressions(expressions_fb_vec);
  program_builder.add_code(code_fb);
  return program_builder.Finish();
}

void DecodeProgram(const data::MetalProgram* metal_program, std::string* code,
                   std::map<std::string, std::string>* defines) {
  *code = std::string(metal_program->code()->c_str(),
                      metal_program->code()->size());
  for (int i = 0; i < metal_program->define_names()->size(); ++i) {
    std::string key((*metal_program->define_names())[i]->c_str(),
                    (*metal_program->define_names())[i]->size());
    std::string value((*metal_program->define_expressions())[i]->c_str(),
                      (*metal_program->define_expressions())[i]->size());
    (*defines)[key] = value;
  }
}
}  // namespace

absl::Status InferenceContext::InitFromGraphWithTransforms(
    const CreateGpuModelInfo& create_info, GraphFloat32* graph,
    id<MTLDevice> device_id, std::vector<uint8_t>* serialized_model) {
  RETURN_IF_ERROR(RunGraphTransformsForGpuModel(graph));
  RETURN_IF_ERROR(
      InitFromGraph(create_info, *graph, device_id, serialized_model));
  return absl::OkStatus();
}

void InferenceContext::CopyFromGpuModel(GpuModel* gpu_model) {
  for (const auto& input : gpu_model->input_ids_and_refs) {
    input_ids_.push_back(input.first);
  }
  for (const auto& output : gpu_model->output_ids_and_refs) {
    output_ids_.push_back(output.first);
  }
  nodes_.resize(gpu_model->nodes.size());
  for (int i = 0; i < gpu_model->nodes.size(); ++i) {
    nodes_[i].task.Init(std::move(gpu_model->nodes[i].gpu_operation));
    nodes_[i].inputs = gpu_model->nodes[i].inputs;
    nodes_[i].outputs = gpu_model->nodes[i].outputs;
    nodes_[i].name = gpu_model->nodes[i].name;
  }
  const_tensors_descs_ = std::move(gpu_model->const_tensors);
  tensors_descs_ = std::move(gpu_model->tensors);
}

absl::Status InferenceContext::InitFromGraph(
    const CreateGpuModelInfo& create_info, const GraphFloat32& graph,
    id<MTLDevice> device_id, std::vector<uint8_t>* serialized_model) {
  device_ = device_id;
  MetalDevice metal_device(device_id);
  GpuModel gpu_model;
  RETURN_IF_ERROR(
      GraphToGpuModel(graph, create_info, metal_device.GetInfo(), &gpu_model));
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb;
  if (serialized_model) {
    gpu_model_fb = tflite::gpu::Encode(gpu_model, &builder);
  }
  CopyFromGpuModel(&gpu_model);

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
        CreateTensor(device_id, tensors_descs_[external_tensor.first],
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

  if (serialized_model) {
    auto encoded_fb = Encode(&metal_device, gpu_model_fb, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
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

absl::Status InferenceContext::RestoreDeserialized(
    const absl::Span<const uint8_t> serialized_model, id<MTLDevice> device_id,
    CreateGpuModelInfo* create_info) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  device_ = device_id;
  MetalDevice metal_device(device_id);
  RETURN_IF_ERROR(Decode(&metal_device, decoded_fb));

  std::map<ValueId, MetalSpatialTensor> temp_external_tensors;
  if (create_info) {
    for (const auto& external_tensor :
         create_info->external_immutable_tensors) {
      auto* cl_spatial_tensor =
          dynamic_cast<MetalSpatialTensor*>(external_tensor.second);
      if (!cl_spatial_tensor) {
        return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
      }
      external_immutable_tensors_[external_tensor.first] = cl_spatial_tensor;
    }
    for (const auto& external_tensor : create_info->external_mutable_tensors) {
      RETURN_IF_ERROR(
          CreateTensor(device_id, tensors_descs_[external_tensor.first],
                       &temp_external_tensors[external_tensor.first]));
      external_mutable_tensors_[external_tensor.first] =
          &temp_external_tensors[external_tensor.first];
    }
  }
  PrepareExternal();

  RETURN_IF_ERROR(AllocateTensors(&metal_device));
  BindTensorsToOperations();

  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.task.RestoreDeserialized(&metal_device));
  }
  RETURN_IF_ERROR(UpdateParams(metal_device.GetInfo()));
  for (auto& external_tensor : external_mutable_tensors_) {
    external_tensor.second = nullptr;
  }
  return absl::OkStatus();
}

flatbuffers::Offset<data::InferenceContext> InferenceContext::Encode(
    MetalDevice* device,
    flatbuffers::Offset<tflite::gpu::data::GpuModel> gpu_model_fb,
    flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<tflite::gpu::data::Int3>> work_groups_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto work_group_fb =
        tflite::gpu::Encode(nodes_[i].task.GetWorkGroupSize(), builder);
    work_groups_fb.push_back(work_group_fb);
  }
  auto work_groups_fb_vec = builder->CreateVector(work_groups_fb);

  std::vector<flatbuffers::Offset<data::MetalProgram>> programs_fb;
  for (int i = 0; i < nodes_.size(); ++i) {
    auto program_fb = EncodeProgram(nodes_[i].task.GetCode(),
                                    nodes_[i].task.GetDefines(), builder);
    programs_fb.push_back(program_fb);
  }
  auto programs_fb_vec = builder->CreateVector(programs_fb);

  data::InferenceContextBuilder inf_builder(*builder);
  inf_builder.add_gpu_model(gpu_model_fb);
  inf_builder.add_tuned_work_group_sizes_per_node(work_groups_fb_vec);
  inf_builder.add_metal_programs(programs_fb_vec);
  return inf_builder.Finish();
}

absl::Status InferenceContext::Decode(
    MetalDevice* device, const data::InferenceContext* fb_inference) {
  GpuModel gpu_model;
  RETURN_IF_ERROR(tflite::gpu::Decode(fb_inference->gpu_model(), &gpu_model));
  CopyFromGpuModel(&gpu_model);

  for (int i = 0; i < nodes_.size(); ++i) {
    std::string code;
    std::map<std::string, std::string> defines;
    DecodeProgram((*fb_inference->metal_programs())[i], &code, &defines);
    RETURN_IF_ERROR(nodes_[i].task.Init(device, code, defines));

    int3 wg_size;
    wg_size.x = (*fb_inference->tuned_work_group_sizes_per_node())[i]->x();
    wg_size.y = (*fb_inference->tuned_work_group_sizes_per_node())[i]->y();
    wg_size.z = (*fb_inference->tuned_work_group_sizes_per_node())[i]->z();
    nodes_[i].task.SetWorkGroupSize(wg_size);
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
  MetalSpatialTensor* gpu_tensor = GetTensor(id);
  TensorDescriptor descriptor_with_data = gpu_tensor->GetDescriptor();
  descriptor_with_data.UploadData(tensor);
  return gpu_tensor->UploadDescriptorData(descriptor_with_data, device_);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               TensorFloat32* result) {
  const MetalSpatialTensor* gpu_tensor = GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor->Batch(), gpu_tensor->Height(),
                              gpu_tensor->Width(), gpu_tensor->Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());

  TensorDescriptor desc;
  RETURN_IF_ERROR(gpu_tensor->ToDescriptor(&desc, device_));
  desc.DownloadData(result);
  return absl::OkStatus();
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
      const auto& shape = tensors_descs_[in_id].GetBHWDCShape();
      src_shapes.push_back(BHWC(shape.b, shape.h, shape.w, shape.c));
    }
    for (const auto& out_id : node.outputs) {
      const auto& shape = tensors_descs_[out_id].GetBHWDCShape();
      dst_shapes.push_back(BHWC(shape.b, shape.h, shape.w, shape.c));
    }
    RETURN_IF_ERROR(node.task.UpdateParams());
  }
  return absl::OkStatus();
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
  } else if (IsBufferBased(gpu_info, tensors_descs_[id].GetStorageType())) {
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
      [this, device](ValueId id) {
        return GetTensorMemoryType(device->GetInfo(), id) ==
               TensorMemoryType::kBuffer;
      },
      &buffer_usages);

  if (buffer_usages.empty()) {
    return absl::OkStatus();
  }

  // From Apple documentation:
  // For buffers in the device address space, align the offset to the data type
  // consumed by the compute function (which is always less than or equal to 16
  // bytes).
  // For buffers in the constant address space, align the offset to 256
  // bytes in macOS. In iOS, align the offset to the maximum of either the data
  // type consumed by the compute function, or 4 bytes. A 16-byte alignment is
  // safe in iOS if you don't need to consider the data type.
#if defined(TARGET_IOS) || defined(TARGET_TVOS)
  const size_t kConstAlignment = 16;
#elif defined(TARGET_MACOS)
  const size_t kConstAlignment = 256;
#else
  const size_t kConstAlignment = 256;
#endif
  size_t min_common_alignment = kConstAlignment;
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& t = tensors_descs_[usage.first];
    const auto& shape = t.GetBHWDCShape();
    const auto& descriptor = t;
    const size_t element_size = SizeOf(descriptor.GetDataType());
    size_t buffer_size;
    size_t row_bytes_alignment = [device->device()
        minimumLinearTextureAlignmentForPixelFormat:DataTypeToRGBAPixelFormat(
                                                        descriptor
                                                            .GetDataType(),
                                                        false)];
    if (descriptor.GetStorageType() == TensorStorageType::TEXTURE_2D) {
      min_common_alignment =
          std::lcm(min_common_alignment, row_bytes_alignment);
      const size_t bytes_per_row = element_size * shape.b * shape.w * 4;
      const size_t height = shape.h * DivideRoundUp(shape.c, 4);
      buffer_size = AlignByN(bytes_per_row, row_bytes_alignment) * height;
    } else if (descriptor.GetStorageType() ==
               TensorStorageType::SINGLE_TEXTURE_2D) {
      min_common_alignment =
          std::lcm(min_common_alignment, row_bytes_alignment);
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

  OffsetsAssignment offset_assignment;
  RETURN_IF_ERROR(AssignOffsetsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BY_SIZE, &offset_assignment,
      min_common_alignment));

  bool use_offset_assignment = false;
  if (offset_assignment.total_size <= TotalSize(buffer_assignment) &&
      offset_assignment.total_size <= device->GetInfo().GetMaxBufferSize()) {
    use_offset_assignment = true;
  }

  if (use_offset_assignment) {
    shared_buffers_.resize(1);
    shared_buffers_[0] =
        [device->device() newBufferWithLength:offset_assignment.total_size
                                      options:MTLResourceStorageModeShared];
  } else {
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
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  for (auto& node : nodes_) {
    std::vector<ValueId> all_ids = node.inputs;
    all_ids.insert(all_ids.end(), node.outputs.begin(), node.outputs.end());
    for (auto& tensor_id : all_ids) {
      if (GetTensorMemoryType(device->GetInfo(), tensor_id) !=
          TensorMemoryType::kBuffer) {
        continue;
      }
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[tensor_id];
      if (created_tensors[tensor_index]) continue;
      const auto& tensor_dummy = tensors_descs_[tensor_id];
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      uint64_t base_buffer_offset = 0;
      id<MTLBuffer> base_buffer;
      if (use_offset_assignment) {
        base_buffer = shared_buffers_[0];
        base_buffer_offset = offset_assignment.offsets[tensor_index];
      } else {
        base_buffer = shared_buffers_[buffer_index];
        base_buffer_offset = 0;
      }
      if (tensor_dummy.GetStorageType() == TensorStorageType::TEXTURE_2D ||
          tensor_dummy.GetStorageType() ==
              TensorStorageType::SINGLE_TEXTURE_2D) {
        size_t row_bytes_alignment = [device->device()
            minimumLinearTextureAlignmentForPixelFormat:
                DataTypeToRGBAPixelFormat(tensor_dummy.GetDataType(), false)];
        RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
            base_buffer, tensor_dummy, row_bytes_alignment,
            &shared_buffer_tensors_[tensor_index], base_buffer_offset));
      } else {
        RETURN_IF_ERROR(CreateTensorSharedBuffer(
            base_buffer, tensor_dummy, &shared_buffer_tensors_[tensor_index],
            base_buffer_offset));
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
      [this, device](ValueId id) {
        return GetTensorMemoryType(device->GetInfo(), id) ==
               TensorMemoryType::kStrongShape;
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
      if (GetTensorMemoryType(device->GetInfo(), tensor_id) !=
          TensorMemoryType::kStrongShape) {
        continue;
      }
      const auto id = assignment.object_ids[remap_from_graph_ids[tensor_id]];
      graph_ids_to_strong_shape_tensors_[tensor_id] = id;
      const auto& it = strong_shape_tensors_.find(id);
      if (it == strong_shape_tensors_.end()) {
        RETURN_IF_ERROR(CreateTensor(device->device(), tensor_dummy,
                                     &strong_shape_tensors_[id]));
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

uint64_t InferenceContext::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& node : nodes_) {
    total_size += node.task.GetGpuOperation().const_args_size_;
  }
  for (const auto& t : const_tensors_) {
    total_size += t.second.GetMemorySizeInBytes();
  }
  return total_size;
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

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
