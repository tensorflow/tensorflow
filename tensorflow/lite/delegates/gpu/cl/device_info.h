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

enum class Vendor {
  kQualcomm,
  kMali,
  kPowerVR,
  kNvidia,
  kAMD,
  kIntel,
  kUnknown
};
std::string VendorToString(Vendor v);

enum class OpenCLVersion {
  CL_1_0,
  CL_1_1,
  CL_1_2,
  CL_2_0,
  CL_2_1,
  CL_2_2,
  CL_3_0
};
std::string OpenCLVersionToString(OpenCLVersion version);

struct AdrenoInfo {
  AdrenoInfo() = default;
  explicit AdrenoInfo(const std::string& device_version);
  int gpu_version = -1;  // can be, for example, 405/430/540/530/630 etc.

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

enum class MaliGPU {
  T604,
  T622,
  T624,
  T628,
  T658,
  T678,
  T720,
  T760,
  T820,
  T830,
  T860,
  T880,
  G31,
  G51,
  G71,
  G52,
  G72,
  G76,
  G57,
  G77,
  G68,
  G78,
  UNKNOWN
};

struct MaliInfo {
  MaliInfo() = default;
  explicit MaliInfo(const std::string& device_name);
  MaliGPU gpu_version;

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

struct DeviceInfo {
  DeviceInfo() = default;

  bool IsAdreno() const;
  bool IsAdreno3xx() const;
  bool IsAdreno4xx() const;
  bool IsAdreno5xx() const;
  bool IsAdreno6xx() const;
  bool IsAdreno6xxOrHigher() const;
  bool IsPowerVR() const;
  bool IsNvidia() const;
  bool IsMali() const;
  bool IsAMD() const;
  bool IsIntel() const;

  bool SupportsTextureArray() const;
  bool SupportsImageBuffer() const;
  bool SupportsImage3D() const;

  bool SupportsFloatImage2D(DataType data_type, int channels) const;

  // To track bug on some Adreno. b/131099086
  bool SupportsOneLayerTextureArray() const;

  bool SupportsExtension(const std::string& extension) const;
  bool IsCL20OrHigher() const;
  bool SupportsSubGroupWithSize(int sub_group_size) const;

  std::vector<std::string> extensions;
  bool supports_fp16;
  bool supports_image3d_writes;
  Vendor vendor;
  OpenCLVersion cl_version;
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
  std::vector<int> supported_subgroup_sizes;

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

  AdrenoInfo adreno_info;
  MaliInfo mali_info;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_DEVICE_INFO_H_
