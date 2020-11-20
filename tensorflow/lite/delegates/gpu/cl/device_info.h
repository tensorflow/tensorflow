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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_DEVICE_INFO_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_DEVICE_INFO_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

// for use only in device_info.cc, but keep here to make tests
int GetAdrenoGPUVersion(const std::string& gpu_version);

namespace tflite {
namespace gpu {
namespace cl {

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

std::string GpuVendorToString(GpuVendor v);

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

enum class OpenClVersion {
  kCl1_0,
  kCl1_1,
  kCl1_2,
  kCl2_0,
  kCl2_1,
  kCl2_2,
  kCl3_0,
  kUnknown,
};
std::string OpenClVersionToString(OpenClVersion version);

struct OpenClInfo {
  OpenClVersion cl_version;

  std::vector<std::string> extensions;
  bool supports_fp16;
  bool supports_image3d_writes;
  int compute_units_count;
  uint64_t buffer_max_size;
  uint64_t image2d_max_width;
  uint64_t image2d_max_height;
  uint64_t image_buffer_max_size;
  uint64_t image_array_max_layers;
  uint64_t image3d_max_width;
  uint64_t image3d_max_height;
  uint64_t image3d_max_depth;
  int max_work_group_size_x;
  int max_work_group_size_y;
  int max_work_group_size_z;

  // rtn is ROUND_TO_NEAREST
  // with rtn precision is much better then with rtz (ROUND_TO_ZERO)
  // Adreno 3xx supports only rtz, Adreno 4xx and more support rtn
  // Mali from T6xx supports rtn
  // PowerVR supports only rtz
  bool supports_fp32_rtn;
  bool supports_fp16_rtn;

  bool supports_r_f16_tex2d = false;
  bool supports_rg_f16_tex2d = false;
  bool supports_rgb_f16_tex2d = false;
  bool supports_rgba_f16_tex2d = false;

  bool supports_r_f32_tex2d = false;
  bool supports_rg_f32_tex2d = false;
  bool supports_rgb_f32_tex2d = false;
  bool supports_rgba_f32_tex2d = false;
};

struct GpuInfo {
  GpuInfo() = default;

  bool IsAdreno() const;
  bool IsApple() const;
  bool IsMali() const;
  bool IsPowerVR() const;
  bool IsNvidia() const;
  bool IsAMD() const;
  bool IsIntel() const;

  bool SupportsFP16() const;

  bool SupportsTextureArray() const;
  bool SupportsImageBuffer() const;
  bool SupportsImage3D() const;

  bool SupportsFloatImage2D(DataType data_type, int channels) const;

  bool SupportsExtension(const std::string& extension) const;
  bool IsCL20OrHigher() const;
  bool IsCL30OrHigher() const;
  bool SupportsSubGroupWithSize(int sub_group_size) const;

  int GetComputeUnitsCount() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  int GetMaxWorkGroupSizeForX() const;
  int GetMaxWorkGroupSizeForY() const;
  int GetMaxWorkGroupSizeForZ() const;

  uint64_t GetMaxImage2DWidth() const;
  uint64_t GetMaxImage2DHeight() const;
  uint64_t GetMaxImage3DWidth() const;
  uint64_t GetMaxImage3DHeight() const;
  uint64_t GetMaxImage3DDepth() const;

  uint64_t GetMaxBufferSize() const;
  uint64_t GetMaxImageBufferWidth() const;
  uint64_t GetMaxImage2DArrayLayers() const;

  std::vector<int> supported_subgroup_sizes;

  GpuVendor gpu_vendor;

  AdrenoInfo adreno_info;
  MaliInfo mali_info;

  OpenClInfo opencl_info;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_DEVICE_INFO_H_
