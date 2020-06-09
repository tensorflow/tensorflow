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
#include "tensorflow/lite/experimental/acceleration/whitelist/gpu_whitelist.h"

#include <cctype>
#include <map>
#include <string>

#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/whitelist/database_generated.h"
#include "tensorflow/lite/experimental/acceleration/whitelist/devicedb.h"
#include "tensorflow/lite/experimental/acceleration/whitelist/gpu_whitelist_binary.h"
#include "tensorflow/lite/experimental/acceleration/whitelist/variables.h"

namespace tflite {
namespace acceleration {
namespace {

std::string CanonicalizeValue(absl::string_view input) {
  // This assumes ASCII, which holds for all values we have in the whitelist.
  std::string output(input);
  for (int i = 0; i < output.size(); i++) {
    char c = output[i];
    if (c == ' ' || c == '-') {
      output[i] = '_';
    } else if (isalpha(c)) {
      output[i] = tolower(c);
    }
  }
  return output;
}

void CanonicalizeValues(std::map<std::string, std::string>* variable_values) {
  for (auto& i : *variable_values) {
    i.second = CanonicalizeValue(i.second);
  }
}

}  // namespace

GPUWhitelist::GPUWhitelist()
    : GPUWhitelist(g_tflite_acceleration_gpu_whitelist_binary) {}

GPUWhitelist::GPUWhitelist(const unsigned char* whitelist_flatbuffer) {
  if (!whitelist_flatbuffer) return;
  database_ = flatbuffers::GetRoot<DeviceDatabase>(whitelist_flatbuffer);
}

std::map<std::string, std::string> GPUWhitelist::CalculateVariables(
    const AndroidInfo& android_info,
    const ::tflite::gpu::GpuInfo& gpu_info) const {
  std::map<std::string, std::string> variables;

  variables[kAndroidSdkVersion] = android_info.android_sdk_version;
  variables[kDeviceModel] = android_info.model;
  variables[kDeviceName] = android_info.device;
  variables[kManufacturer] = android_info.manufacturer;
  variables[kGPUModel] = gpu_info.renderer_name;
  char buffer[128];
  int len = snprintf(buffer, 128 - 1, "%d.%d", gpu_info.major_version,
                     gpu_info.minor_version);
  buffer[len] = '\0';
  variables[kOpenGLESVersion] = std::string(buffer);
  CanonicalizeValues(&variables);
  if (!database_) return variables;
  UpdateVariablesFromDatabase(&variables, *database_);
  return variables;
}

bool GPUWhitelist::Includes(const AndroidInfo& android_info,
                            const ::tflite::gpu::GpuInfo& gpu_info) const {
  auto variables = CalculateVariables(android_info, gpu_info);
  return variables[gpu::kStatus] == std::string(gpu::kStatusWhitelisted);
}

TfLiteGpuDelegateOptionsV2 GPUWhitelist::GetBestOptionsFor(
    const AndroidInfo& /* android_info */,
    const ::tflite::gpu::GpuInfo& /* gpu_info */) const {
  // This method is for forwards-compatibility: the whitelist may later include
  // information about which backend to choose (OpenGL/OpenCL/Vulkan) or other
  // options.
  return TfLiteGpuDelegateOptionsV2Default();
}

}  // namespace acceleration
}  // namespace tflite
