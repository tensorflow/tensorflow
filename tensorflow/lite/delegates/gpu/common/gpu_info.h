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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_

#include <string>
#include <vector>

namespace tflite {
namespace gpu {

enum class GpuType { UNKNOWN, MALI, ADRENO, POWERVR, INTEL, NVIDIA };
enum class GpuModel {
  UNKNOWN,
  // Adreno 6xx series
  ADRENO640,
  ADRENO630,
  ADRENO616,
  ADRENO615,
  ADRENO612,
  ADRENO605,
  // Adreno 5xx series
  ADRENO540,
  ADRENO530,
  ADRENO512,
  ADRENO510,
  ADRENO509,
  ADRENO508,
  ADRENO506,
  ADRENO505,
  ADRENO504,
  // Adreno 4xx series
  ADRENO430,
  ADRENO420,
  ADRENO418,
  ADRENO405,
  // Adreno 3xx series
  ADRENO330,
  ADRENO320,
  ADRENO308,
  ADRENO306,
  ADRENO305,
  ADRENO304,
  // Adreno 2xx series
  ADRENO225,
  ADRENO220,
  ADRENO205,
  ADRENO203,
  ADRENO200,
  // Adreno 1xx series
  ADRENO130,
};

struct GpuInfo {
  GpuType type = GpuType::UNKNOWN;
  std::string renderer_name;
  std::string vendor_name;
  std::string version;
  GpuModel gpu_model;
  int major_version = -1;
  int minor_version = -1;
  std::vector<std::string> extensions;
  int max_ssbo_bindings = 0;
  int max_image_bindings = 0;
  std::vector<int> max_work_group_size;
  int max_work_group_invocations;
  int max_texture_size = 0;
  int max_image_units = 0;
  int max_array_texture_layers = 0;
};

inline bool IsOpenGl31OrAbove(const GpuInfo& gpu_info) {
  return (gpu_info.major_version == 3 && gpu_info.minor_version >= 1) ||
         gpu_info.major_version > 3;
}

// Analyzes `renderer` and returns matching `GpuType` and `GpuModel`.
void GetGpuModelAndType(const std::string& renderer, GpuModel* gpu_model,
                        GpuType* gpu_type);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_
