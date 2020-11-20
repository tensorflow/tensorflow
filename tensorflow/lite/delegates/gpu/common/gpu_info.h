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

// The VendorID returned by the GPU driver.
enum class GpuVendor {
  kApple,
  kQualcomm,
  kMali,
  kPowerVR,
  kNvidia,
  kAMD,
  kIntel,
  kUnknown
};

enum class GpuApi {
  kUnknown,
  kOpenCl,
  kMetal,
  kVulkan,
  kOpenGl,
};

enum class AdrenoGpu {
  // Adreno 6xx series
  kAdreno685,
  kAdreno680,
  kAdreno675,
  kAdreno650,
  kAdreno640,
  kAdreno630,
  kAdreno620,
  kAdreno618,
  kAdreno616,
  kAdreno615,
  kAdreno612,
  kAdreno610,
  kAdreno605,
  // Adreno 5xx series
  kAdreno540,
  kAdreno530,
  kAdreno512,
  kAdreno510,
  kAdreno509,
  kAdreno508,
  kAdreno506,
  kAdreno505,
  kAdreno504,
  // Adreno 4xx series
  kAdreno430,
  kAdreno420,
  kAdreno418,
  kAdreno405,
  // Adreno 3xx series
  kAdreno330,
  kAdreno320,
  kAdreno308,
  kAdreno306,
  kAdreno305,
  kAdreno304,
  // Adreno 2xx series
  kAdreno225,
  kAdreno220,
  kAdreno205,
  kAdreno203,
  kAdreno200,
  // Adreno 1xx series
  kAdreno130,
  kAdreno120,
  kUnknown
};

struct AdrenoInfo {
  AdrenoInfo() = default;
  explicit AdrenoInfo(const std::string& device_version);

  AdrenoGpu adreno_gpu;

  bool IsAdreno1xx() const;
  bool IsAdreno2xx() const;
  bool IsAdreno3xx() const;
  bool IsAdreno4xx() const;
  bool IsAdreno5xx() const;
  bool IsAdreno6xx() const;
  bool IsAdreno6xxOrHigher() const;

  // This function returns some not very documented physical parameter of
  // Adreno6xx GPU.
  // We obtained it using Snapdragon Profiler.
  int GetMaximumWavesCount() const;

  // returns amount of register memory per CU(Compute Unit) in bytes.
  int GetRegisterMemorySizePerComputeUnit() const;

  // returns maximum possible amount of waves based on register usage.
  int GetMaximumWavesCount(int register_footprint_per_tread,
                           bool full_wave = true) const;

  int GetWaveSize(bool full_wave) const;

  // Not supported on some Adreno devices with specific driver version.
  // b/131099086
  bool support_one_layer_texture_array = true;
};

enum class AppleGpu {
  kUnknown,
  kA7,
  kA8,
  kA8X,
  kA9,
  kA9X,
  kA10,
  kA10X,
  kA11,
  kA12,
  kA12X,
  kA12Z,
  kA13,
  kA14,
};

struct AppleInfo {
  AppleInfo() = default;
  explicit AppleInfo(const std::string& gpu_description);
  AppleGpu gpu_type;

  bool IsLocalMemoryPreferredOverGlobal() const;

  bool IsBionic() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  int GetComputeUnitsCount() const;
};

enum class MaliGpu {
  kUnknown,
  kT604,
  kT622,
  kT624,
  kT628,
  kT658,
  kT678,
  kT720,
  kT760,
  kT820,
  kT830,
  kT860,
  kT880,
  kG31,
  kG51,
  kG71,
  kG52,
  kG72,
  kG76,
  kG57,
  kG77,
  kG68,
  kG78,
};

struct MaliInfo {
  MaliInfo() = default;
  explicit MaliInfo(const std::string& gpu_description);
  MaliGpu gpu_version;

  bool IsMaliT6xx() const;
  bool IsMaliT7xx() const;
  bool IsMaliT8xx() const;
  bool IsMidgard() const;
  bool IsBifrostGen1() const;
  bool IsBifrostGen2() const;
  bool IsBifrostGen3() const;
  bool IsBifrost() const;
  bool IsValhall() const;
};

struct OpenGlInfo {
  std::string renderer_name;
  std::string vendor_name;
  std::string version;
  int major_version = -1;
  int minor_version = -1;

  int max_image_units = 0;
  int max_ssbo_bindings = 0;
  int max_image_bindings = 0;
  int max_work_group_invocations = 0;
  int max_texture_size = 0;
  int max_array_texture_layers = 0;
};

struct VulkanInfo {
  std::string vendor_name;
  uint32_t api_version = -1;
  uint32_t api_version_major = -1;
  uint32_t api_version_minor = -1;
  uint32_t api_version_patch = -1;

  uint32_t max_per_stage_descriptor_sampled_images = 0;
  uint32_t max_compute_work_group_invocations;
  uint32_t max_image_dimension_2d;
  uint32_t max_image_array_layers;
};

struct GpuInfo {
  bool IsAdreno() const;
  bool IsApple() const;
  bool IsMali() const;
  bool IsPowerVR() const;
  bool IsNvidia() const;
  bool IsAMD() const;
  bool IsIntel() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  // returns true if device have fixed wave size equal to 32
  bool IsWaveSizeEqualTo32() const;

  int GetComputeUnitsCount() const;

  int GetMaxImageArguments() const;

  int GetMaxWorkGroupSizeForX() const;
  int GetMaxWorkGroupSizeForY() const;
  int GetMaxWorkGroupSizeForZ() const;
  int GetMaxWorkGroupTotalSize() const;

  uint64_t GetMaxImage2DWidth() const;
  uint64_t GetMaxImage2DHeight() const;
  uint64_t GetMaxImage2DArrayLayers() const;

  GpuVendor vendor = GpuVendor::kUnknown;
  GpuApi gpu_api = GpuApi::kUnknown;

  std::vector<std::string> extensions;
  std::vector<int> max_work_group_size;

  std::vector<int> supported_subgroup_sizes;

  AdrenoInfo adreno_info;
  AppleInfo apple_info;
  MaliInfo mali_info;

  // OpenGL specific, gpu_api should be kOpenGl
  OpenGlInfo opengl_info;
  bool IsApiOpenGl() const;
  bool IsApiOpenGl31OrAbove() const;

  // Vulkan specific, gpu_api should be kVulkan
  VulkanInfo vulkan_info;
  bool IsApiVulkan() const;

  bool IsApiMetal() const;

  bool IsApiOpenCl() const;
};

inline bool IsOpenGl31OrAbove(const GpuInfo& gpu_info) {
  return (gpu_info.opengl_info.major_version == 3 &&
          gpu_info.opengl_info.minor_version >= 1) ||
         gpu_info.opengl_info.major_version > 3;
}

// Currently it initializes:
// vendor
// AdrenoInfo if vendor is kQualcomm
// AppleInfo if vendor is kApple
// MaliInfo if vendor is kMali
void GetGpuInfoFromDeviceDescription(const std::string& gpu_description,
                                     GpuApi gpu_api, GpuInfo* gpu_info);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_GPU_INFO_H_
