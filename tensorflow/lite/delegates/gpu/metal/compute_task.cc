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

#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"

#include <Availability.h>

#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/kernel_info.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
  size_t position = str->find(old_word);
  while (position != std::string::npos) {
    const char prev = position == 0 ? ' ' : (*str)[position - 1];
    const char next = position + old_word.size() < str->size()
                          ? (*str)[position + old_word.size()]
                          : ' ';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position = str->find(old_word, position + 1);
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    position = str->find(old_word, position + new_word.size());
  }
}

std::map<std::string, std::string> GetMetalDefines(
    MetalDevice* device, CalculationsPrecision precision) {
  std::string simdgroup_barrier;
  // simdgroup_barrier is supported since Metal shading language version 2.0
  if (device->IsLanguageVersion2orHigher()) {
    simdgroup_barrier = "simdgroup_barrier";
  } else {
    simdgroup_barrier = "threadgroup_barrier";
  }
  std::string storage_type;
  std::string accumulator_type;
  std::string to_accumulator_type4;
  if (precision == CalculationsPrecision::F32) {
    storage_type = "float";
    accumulator_type = "float";
  } else {
    // FP16
    storage_type = "half";
    if (precision == CalculationsPrecision::F32_F16) {
      accumulator_type = "float";
      to_accumulator_type4 = "float4";
    } else {
      accumulator_type = "half";
    }
  }
  return {
      {"FLT16_0123(V)", "V[0]"},
      {"FLT16_4567(V)", "V[1]"},
      {"FLT16_89ab(V)", "V[2]"},
      {"FLT16_cdef(V)", "V[3]"},
      {"FLT", storage_type},
      {"FLT2", storage_type + "2"},
      {"FLT3", storage_type + "3"},
      {"FLT4", storage_type + "4"},
      {"ACCUM_FLT", accumulator_type},
      {"ACCUM_FLT2", accumulator_type + "2"},
      {"ACCUM_FLT3", accumulator_type + "3"},
      {"ACCUM_FLT4", accumulator_type + "4"},
      {"INIT_ACCUM_FLT4(value)", accumulator_type + "4(value)"},
      {"TO_ACCUM_TYPE", to_accumulator_type4},
      {"TO_ACCUM_FLT", accumulator_type},
      {"TO_ACCUM_FLT2", accumulator_type + "2"},
      {"TO_ACCUM_FLT3", accumulator_type + "3"},
      {"TO_ACCUM_FLT4", accumulator_type + "4"},
      {"TO_FLT4", storage_type + "4"},
      {"SIMDGROUP_BARRIER", simdgroup_barrier},
      {"SIMD_LOCAL_MEM_BARRIER", simdgroup_barrier},
      {"MAIN_FUNCTION", "kernel void ComputeFunction"},
      {"GLOBAL_ID_0", "static_cast<int>(reserved_gid.x)"},
      {"GLOBAL_ID_1", "static_cast<int>(reserved_gid.y)"},
      {"GLOBAL_ID_2", "static_cast<int>(reserved_gid.z)"},
      {"LOCAL_ID_0", "static_cast<int>(reserved_lid.x)"},
      {"LOCAL_ID_1", "static_cast<int>(reserved_lid.y)"},
      {"LOCAL_ID_2", "static_cast<int>(reserved_lid.z)"},
      {"GROUP_ID_0", "static_cast<int>(reserved_group_id.x)"},
      {"GROUP_ID_1", "static_cast<int>(reserved_group_id.y)"},
      {"GROUP_ID_2", "static_cast<int>(reserved_group_id.z)"},
      {"GROUP_SIZE_0", "static_cast<int>(reserved_group_size.x)"},
      {"GROUP_SIZE_1", "static_cast<int>(reserved_group_size.y)"},
      {"GROUP_SIZE_2", "static_cast<int>(reserved_group_size.z)"},
      {"SUB_GROUP_LOCAL_ID", "static_cast<int>(reserved_simd_id)"},
      {"SUB_GROUP_BROADCAST(V, ID)", "simd_broadcast(V, ID)"},
      {"__local", "threadgroup"},
      {"__global", "device"},
      {"__constant", "constant"},
      {"LOCAL_MEM_BARRIER", "threadgroup_barrier(mem_flags::mem_threadgroup)"},
      {"INIT_FLT(value)", storage_type + "(value)"},
      {"INIT_FLT4(value)", storage_type + "4(value)"},
      {"INIT_FLT4v4(v0, v1, v2, v3)", storage_type + "4(v0, v1, v2, v3)"},
      {"INIT_FLOAT(value)", "float(value)"},
      {"INIT_FLOAT2(value)", "float2(value)"},
      {"INIT_FLOAT2v2(v0, v1)", "float2(v0, v1)"},
      {"INIT_FLOAT3(value)", "float3(value)"},
      {"INIT_FLOAT3v3(v0, v1, v2)", "float3(v0, v1, v2)"},
      {"INIT_FLOAT4(value)", "float4(value)"},
      {"INIT_FLOAT4v4(v0, v1, v2, v3)", "float4(v0, v1, v2, v3)"},
      {"INIT_INT(value)", "int(value)"},
      {"INIT_INT2v2(v0, v1)", "int2(v0, v1)"},
      {"INIT_INT4v4(v0, v1, v2, v3)", "int4(v0, v1, v2, v3)"},
      {"CONVERT_TO_INT4(value)", "int4(value)"},
  };
}
}  // namespace

ComputeTask::ComputeTask(ComputeTask&& task)
    : operation_(std::move(task.operation_)),
      program_(task.program_),
      metal_args_(std::move(task.metal_args_)),
      use_arguments_buffer_(task.use_arguments_buffer_),
      need_icb_support_(task.need_icb_support_),
      arguments_encoder_(task.arguments_encoder_),
      arg_buffer_(task.arg_buffer_) {
  task.program_ = nullptr;
  task.arguments_encoder_ = nullptr;
  task.arg_buffer_ = nullptr;
}

ComputeTask& ComputeTask::operator=(ComputeTask&& task) {
  if (this != &task) {
    Release();
    operation_ = std::move(task.operation_);
    std::swap(program_, task.program_);
    metal_args_ = std::move(task.metal_args_);
    std::swap(use_arguments_buffer_, task.use_arguments_buffer_);
    std::swap(need_icb_support_, task.need_icb_support_);
    std::swap(arguments_encoder_, task.arguments_encoder_);
    std::swap(arg_buffer_, task.arg_buffer_);
  }
  return *this;
}

ComputeTask::~ComputeTask() { Release(); }

void ComputeTask::Release() {
  if (program_) {
    program_ = nullptr;
  }
  if (arguments_encoder_) {
    arguments_encoder_ = nullptr;
  }
  if (arg_buffer_) {
    arg_buffer_ = nullptr;
  }
}

void ComputeTask::Init(std::unique_ptr<GPUOperation>&& operation) {
  operation_ = std::move(operation);
}

const OperationDef& ComputeTask::GetDefinition() const {
  return operation_->GetDefinition();
}

absl::Status ComputeTask::Compile(MetalDevice* device) {
  RETURN_IF_ERROR(metal_args_.Init(use_arguments_buffer_, device,
                                   &operation_->args_, &operation_->code_));

  operation_->args_.ReleaseCPURepresentation();

  // manually resolving this defines, so as Metal has reserved words for them
  ReplaceAllWords("float16", "float4x4", &operation_->code_);
  ReplaceAllWords("half16", "half4x4", &operation_->code_);
  ReplaceAllWords("float8", "float2x4", &operation_->code_);
  ReplaceAllWords("half8", "half2x4", &operation_->code_);
  defines_ = GetMetalDefines(device, operation_->GetDefinition().precision);
  return CompileProgram(device, operation_->code_, defines_);
}

absl::Status ComputeTask::CompileProgram(
    MetalDevice* device, const std::string& code,
    const std::map<std::string, std::string>& defines) {
  id<MTLComputePipelineState> program;
  if (use_arguments_buffer_) {
    id<MTLArgumentEncoder> arguments_encoder;
    if (need_icb_support_) {
      RETURN_IF_ERROR(CreateComputeProgramWithICBSupport(
          device->device(), code, "ComputeFunction", defines, &program,
          &arguments_encoder));
    } else {
      RETURN_IF_ERROR(CreateComputeProgramWithArgumentBuffer(
          device->device(), code, "ComputeFunction", defines, &program,
          &arguments_encoder));
    }
    arguments_encoder_ = arguments_encoder;
    arg_buffer_ =
        [device->device() newBufferWithLength:arguments_encoder_.encodedLength
                                      options:0];
    if (!arg_buffer_) {
      return absl::InternalError("Failed to create MTLBuffer.");
    }
  } else {
    RETURN_IF_ERROR(CreateComputeProgram(device->device(), code,
                                         "ComputeFunction", defines, &program));
  }
  program_ = program;
  return absl::OkStatus();
}

absl::Status ComputeTask::Init(
    MetalDevice* device, const std::string& code,
    const std::map<std::string, std::string>& defines) {
  return CompileProgram(device, code, defines);
}

absl::Status ComputeTask::RestoreDeserialized(MetalDevice* device) {
  RETURN_IF_ERROR(
      metal_args_.Init(use_arguments_buffer_, device, &operation_->args_));

  operation_->args_.ReleaseCPURepresentation();
  return absl::OkStatus();
}

absl::Status ComputeTask::UpdateParams() {
  for (int i = 0; i < operation_->GetSrcTensorsNames().size(); ++i) {
    const auto* metal_spatial_tensor =
        dynamic_cast<const MetalSpatialTensor*>(operation_->GetSrcTensors()[i]);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    RETURN_IF_ERROR(metal_args_.SetObjectRef(
        operation_->GetSrcTensorsNames()[i], *metal_spatial_tensor));
  }
  for (int i = 0; i < operation_->GetDstTensorsNames().size(); ++i) {
    const auto* metal_spatial_tensor =
        dynamic_cast<const MetalSpatialTensor*>(operation_->GetDstTensors()[i]);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    RETURN_IF_ERROR(metal_args_.SetObjectRef(
        operation_->GetDstTensorsNames()[i], *metal_spatial_tensor));
  }
  RETURN_IF_ERROR(operation_->BindArguments(&metal_args_));
  operation_->RecalculateGridSize();
  operation_->RecalculateWorkGroupsCount();
  Update();
  return absl::OkStatus();
}

API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
void ComputeTask::EncodeToICB(id<MTLIndirectComputeCommand> icb_command) {
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->GetWorkGroupsCount().x;
  groupsCount.height = operation_->GetWorkGroupsCount().y;
  groupsCount.depth = operation_->GetWorkGroupsCount().z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [icb_command setComputePipelineState:program_];
  [icb_command setKernelBuffer:arg_buffer_ offset:0 atIndex:0];
  [icb_command concurrentDispatchThreadgroups:groupsCount
                        threadsPerThreadgroup:groupsSize];
  [icb_command setBarrier];
}

API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
void ComputeTask::AddResourcesToEncoder(
    id<MTLComputeCommandEncoder> encoder) const {
  metal_args_.AddResourcesToEncoder(encoder);
}

void ComputeTask::Update() {
  if (use_arguments_buffer_) {
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
      [arguments_encoder_ setArgumentBuffer:arg_buffer_ offset:0];
      metal_args_.EncodeArguments(arguments_encoder_);
    }
  }
}

void ComputeTask::Encode(id<MTLComputeCommandEncoder> encoder) {
  [encoder setComputePipelineState:program_];
  if (use_arguments_buffer_) {
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
      metal_args_.AddResourcesToEncoder(encoder);
      [encoder setBuffer:arg_buffer_ offset:0 atIndex:0];
    }
  } else {
    metal_args_.Encode(encoder, 0);
  }
  MTLSize groupsCount, groupsSize;
  groupsCount.width = operation_->GetWorkGroupsCount().x;
  groupsCount.height = operation_->GetWorkGroupsCount().y;
  groupsCount.depth = operation_->GetWorkGroupsCount().z;
  groupsSize.width = operation_->work_group_size_.x;
  groupsSize.height = operation_->work_group_size_.y;
  groupsSize.depth = operation_->work_group_size_.z;
  [encoder dispatchThreadgroups:groupsCount threadsPerThreadgroup:groupsSize];
}

void ComputeTask::SetSrcTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetSrc(tensor, index);
  auto status = metal_args_.SetObjectRef(
      operation_->GetSrcTensorsNames()[index], *tensor);
}

void ComputeTask::SetDstTensor(MetalSpatialTensor* tensor, int index) {
  operation_->SetDst(tensor, index);
  auto status = metal_args_.SetObjectRef(
      operation_->GetDstTensorsNames()[index], *tensor);
}

absl::Status ComputeTask::Tune(TuningType tuning_type, MetalDevice* device) {
  KernelInfo kernel_info;
  kernel_info.max_work_group_size = [program_ maxTotalThreadsPerThreadgroup];
  kernel_info.private_memory_size = 0;
  std::vector<GPUOperation::DispatchInfo> possible_dispatches;
  operation_->GetPossibleDispatches(tuning_type, device->GetInfo(), kernel_info,
                                    &possible_dispatches);
  if (possible_dispatches.empty()) {
    return absl::NotFoundError("No dispatch parameters to launch kernel");
  }
  operation_->work_group_size_ = possible_dispatches[0].work_group_size;
  operation_->RecalculateWorkGroupsCount();
  return absl::OkStatus();
}

void ComputeTask::SetWorkGroupSize(const int3& work_group_size) {
  operation_->work_group_size_ = work_group_size;
  operation_->RecalculateWorkGroupsCount();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
