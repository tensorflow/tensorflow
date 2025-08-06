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
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"

#include <cctype>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/canonicalize_value.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/database_generated.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility_binary.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/variables.h"

namespace tflite {
namespace acceleration {
namespace {

void CanonicalizeValues(std::map<std::string, std::string>* variable_values) {
  for (auto& i : *variable_values) {
    i.second = CanonicalizeValueWithKey(i.first, i.second);
  }
}

}  // namespace

GPUCompatibilityList::GPUCompatibilityList(
    const unsigned char* compatibility_list_flatbuffer) {
  if (!compatibility_list_flatbuffer) return;
  database_ =
      flatbuffers::GetRoot<DeviceDatabase>(compatibility_list_flatbuffer);
}

GPUCompatibilityList::GPUCompatibilityList(
    std::string compatibility_list_flatbuffer)
    : fbcontent_(std::move(compatibility_list_flatbuffer)) {
  database_ = flatbuffers::GetRoot<DeviceDatabase>(fbcontent_.data());
}

std::unique_ptr<GPUCompatibilityList> GPUCompatibilityList::Create() {
  return Create(g_tflite_acceleration_gpu_compatibility_binary,
                g_tflite_acceleration_gpu_compatibility_binary_len);
}

std::unique_ptr<GPUCompatibilityList> GPUCompatibilityList::Create(
    const unsigned char* compatibility_list_flatbuffer, int length) {
  if (!compatibility_list_flatbuffer ||
      !IsValidFlatbuffer(compatibility_list_flatbuffer, length)) {
    return nullptr;
  }
  return std::unique_ptr<GPUCompatibilityList>(
      new GPUCompatibilityList(compatibility_list_flatbuffer));
}

std::unique_ptr<GPUCompatibilityList> GPUCompatibilityList::Create(
    std::string compatibility_list_flatbuffer) {
  if (!IsValidFlatbuffer(reinterpret_cast<const unsigned char*>(
                             compatibility_list_flatbuffer.data()),
                         compatibility_list_flatbuffer.size())) {
    return nullptr;
  }
  return std::unique_ptr<GPUCompatibilityList>(
      new GPUCompatibilityList(std::move(compatibility_list_flatbuffer)));
}

std::map<std::string, std::string> GPUCompatibilityList::CalculateVariables(
    const AndroidInfo& android_info,
    const ::tflite::gpu::GpuInfo& gpu_info) const {
  std::map<std::string, std::string> variables =
      InfosToMap(android_info, gpu_info);

  CanonicalizeValues(&variables);

  if (!database_) return variables;
  UpdateVariablesFromDatabase(&variables, *database_);
  return variables;
}

bool GPUCompatibilityList::Includes(
    const AndroidInfo& android_info,
    const ::tflite::gpu::GpuInfo& gpu_info) const {
  auto variables = CalculateVariables(android_info, gpu_info);
  return variables[gpu::kStatus] == std::string(gpu::kStatusSupported);
}

gpu::CompatibilityStatus GPUCompatibilityList::GetStatus(
    const AndroidInfo& android_info,
    const ::tflite::gpu::GpuInfo& gpu_info) const {
  std::map<std::string, std::string> variables =
      InfosToMap(android_info, gpu_info);
  return GetStatus(variables);
}

gpu::CompatibilityStatus GPUCompatibilityList::GetStatus(
    std::map<std::string, std::string>& variables) const {
  CanonicalizeValues(&variables);
  if (!database_) return gpu::CompatibilityStatus::kUnknown;
  UpdateVariablesFromDatabase(&variables, *database_);
  return StringToCompatibilityStatus(variables[gpu::kStatus]);
}

TfLiteGpuDelegateOptionsV2 GPUCompatibilityList::GetBestOptionsFor(
    const AndroidInfo& /* android_info */,
    const ::tflite::gpu::GpuInfo& /* gpu_info */) const {
  // This method is for forwards-compatibility: the list may later include
  // information about which backend to choose (OpenGL/OpenCL/Vulkan) or other
  // options.
  return TfLiteGpuDelegateOptionsV2Default();
}

// static
bool GPUCompatibilityList::IsValidFlatbuffer(const unsigned char* data,
                                             int len) {
  // Verify opensource db.
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(data), len);
  return tflite::acceleration::VerifyDeviceDatabaseBuffer(verifier);
}

std::map<std::string, std::string> GPUCompatibilityList::InfosToMap(
    const AndroidInfo& android_info,
    const ::tflite::gpu::GpuInfo& gpu_info) const {
  std::map<std::string, std::string> variables;
  variables[kAndroidSdkVersion] = android_info.android_sdk_version;
  variables[kDeviceModel] = android_info.model;
  variables[kDeviceName] = android_info.device;
  variables[kManufacturer] = android_info.manufacturer;
  const auto& gl_info = gpu_info.opengl_info;
  variables[kGPUModel] = gl_info.renderer_name;

  char buffer[128];
  snprintf(buffer, 128 - 1, "%d.%d", gl_info.major_version,
           gl_info.minor_version);
  variables[kOpenGLESVersion] = std::string(buffer);
  return variables;
}

// static
std::string GPUCompatibilityList::CompatibilityStatusToString(
    gpu::CompatibilityStatus status) {
  switch (status) {
    case gpu::CompatibilityStatus::kSupported:
      return gpu::kStatusSupported;
    case gpu::CompatibilityStatus::kUnsupported:
      return gpu::kStatusUnsupported;
    case gpu::CompatibilityStatus::kUnknown:
      return gpu::kStatusUnknown;
  }
}

// static
gpu::CompatibilityStatus GPUCompatibilityList::StringToCompatibilityStatus(
    absl::string_view status) {
  if (status == gpu::kStatusSupported) {
    return gpu::CompatibilityStatus::kSupported;
  } else if (status == gpu::kStatusUnsupported) {
    return gpu::CompatibilityStatus::kUnsupported;
  }
  return gpu::CompatibilityStatus::kUnknown;
}

}  // namespace acceleration
}  // namespace tflite
