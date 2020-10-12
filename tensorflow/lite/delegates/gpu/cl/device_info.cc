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

#include "tensorflow/lite/delegates/gpu/cl/device_info.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
// check that gpu_version belong to range min_version-max_version
// min_version is included and max_version is excluded.
bool IsGPUVersionInRange(int gpu_version, int min_version, int max_version) {
  return gpu_version >= min_version && gpu_version < max_version;
}

MaliGPU GetMaliGPUVersion(const std::string& device_name) {
  const std::map<std::string, MaliGPU> kMapping = {
      {"T604", MaliGPU::T604}, {"T622", MaliGPU::T622}, {"T624", MaliGPU::T624},
      {"T628", MaliGPU::T628}, {"T658", MaliGPU::T658}, {"T678", MaliGPU::T678},
      {"T720", MaliGPU::T720}, {"T760", MaliGPU::T760}, {"T820", MaliGPU::T820},
      {"T830", MaliGPU::T830}, {"T860", MaliGPU::T860}, {"T880", MaliGPU::T880},
      {"G31", MaliGPU::G31},   {"G51", MaliGPU::G51},   {"G71", MaliGPU::G71},
      {"G52", MaliGPU::G52},   {"G72", MaliGPU::G72},   {"G76", MaliGPU::G76},
      {"G57", MaliGPU::G57},   {"G77", MaliGPU::G77},   {"G68", MaliGPU::G68},
      {"G78", MaliGPU::G78},
  };
  for (const auto& v : kMapping) {
    if (device_name.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return MaliGPU::UNKNOWN;
}

}  // namespace

// There is no rule for gpu version encoding, but we found these samples:
// Version: OpenCL C 2.0 Adreno(TM) 540   // Pixel 2
// Version: OpenCL C 2.0 Adreno(TM) 630   // Sony Compact XZ2
// Version: OpenCL C 2.0 Adreno(TM) 630   // Pixel 3
// Version: OpenCL C 2.0 Adreno(TM) 540   // Samsung S8
// Version: OpenCL C 1.2 Adreno(TM) 430   // HTC One M9
// Version: OpenCL C 2.0 Adreno(TM) 530   // Samsung S7 Edge
// Version: OpenCL C 1.2 Adreno(TM) 405   // Motorola Moto G(4)
// After the number string ends.
// It is assumed that the <vendor-specific information> for Adreno GPUs has
// the following format:
// <text?><space?>Adreno(TM)<space><text?><version>
// Returns -1 if vendor-specific information cannot be parsed
int GetAdrenoGPUVersion(const std::string& gpu_version) {
  const std::string gpu = absl::AsciiStrToLower(gpu_version);
  const std::vector<absl::string_view> words = absl::StrSplit(gpu, ' ');
  int i = 0;
  for (; i < words.size(); ++i) {
    if (words[i].find("adreno") != words[i].npos) {
      break;
    }
  }
  i += 1;
  for (; i < words.size(); ++i) {
    int number;
    bool is_number = absl::SimpleAtoi(words[i], &number);
    // Adreno GPUs starts from 2xx, but opencl support should be only from 3xx
    if (is_number && number >= 300) {
      return number;
    }
  }
  return -1;
}

std::string VendorToString(Vendor v) {
  switch (v) {
    case Vendor::kQualcomm:
      return "Qualcomm";
    case Vendor::kMali:
      return "Mali";
    case Vendor::kPowerVR:
      return "PowerVR";
    case Vendor::kNvidia:
      return "NVIDIA";
    case Vendor::kAMD:
      return "AMD";
    case Vendor::kIntel:
      return "Intel";
    case Vendor::kUnknown:
      return "unknown vendor";
  }
}

std::string OpenCLVersionToString(OpenCLVersion version) {
  switch (version) {
    case OpenCLVersion::CL_1_0:
      return "1.0";
    case OpenCLVersion::CL_1_1:
      return "1.1";
    case OpenCLVersion::CL_1_2:
      return "1.2";
    case OpenCLVersion::CL_2_0:
      return "2.0";
    case OpenCLVersion::CL_2_1:
      return "2.1";
    case OpenCLVersion::CL_2_2:
      return "2.2";
    case OpenCLVersion::CL_3_0:
      return "3.0";
  }
}

AdrenoInfo::AdrenoInfo(const std::string& device_version)
    : gpu_version(GetAdrenoGPUVersion(device_version)) {}

int AdrenoInfo::GetMaximumWavesCount() const {
  if (gpu_version < 400) {
    return -1;  // Adreno 3xx does not support it currently
  } else if (gpu_version >= 400 && gpu_version < 500) {
    return -1;  // Adreno 4xx does not support it currently
  } else if (gpu_version >= 500 && gpu_version < 600) {
    return -1;  // Adreno 5xx does not support it currently
  } else if (gpu_version >= 600 && gpu_version < 700) {
    return gpu_version == 640 ? 30 : 16;
  } else {
    return -1;  //  Adreno 7xx and higher does not exist yet
  }
}

int AdrenoInfo::GetRegisterMemorySizePerComputeUnit() const {
  if (gpu_version < 400) {
    return -1;  // Adreno 3xx does not support it currently
  } else if (gpu_version >= 400 && gpu_version < 500) {
    return -1;  // Adreno 4xx does not support it currently
  } else if (gpu_version >= 500 && gpu_version < 600) {
    return -1;  // Adreno 5xx does not support it currently
  } else if (gpu_version >= 600 && gpu_version < 700) {
    return gpu_version == 640 ? 128 * 144 * 16 : 128 * 96 * 16;
  } else {
    return -1;  //  Adreno 7xx and higher does not exist yet
  }
}

int AdrenoInfo::GetMaximumWavesCount(int register_footprint_per_tread,
                                     bool full_wave) const {
  const int register_usage_per_wave =
      GetWaveSize(full_wave) * register_footprint_per_tread;
  const int possible_waves_count =
      GetRegisterMemorySizePerComputeUnit() / register_usage_per_wave;
  return std::min(possible_waves_count, GetMaximumWavesCount());
}

int AdrenoInfo::GetWaveSize(bool full_wave) const {
  if (gpu_version < 400) {
    return -1;  // Adreno 3xx does not support it currently
  } else if (gpu_version < 600) {
    return full_wave ? 64 : 32;
  } else {
    return full_wave ? 128 : 64;
  }
}

MaliInfo::MaliInfo(const std::string& device_name)
    : gpu_version(GetMaliGPUVersion(device_name)) {}

bool MaliInfo::IsMaliT6xx() const {
  return gpu_version == MaliGPU::T604 || gpu_version == MaliGPU::T622 ||
         gpu_version == MaliGPU::T624 || gpu_version == MaliGPU::T628 ||
         gpu_version == MaliGPU::T658 || gpu_version == MaliGPU::T678;
}

bool MaliInfo::IsMaliT7xx() const {
  return gpu_version == MaliGPU::T720 || gpu_version == MaliGPU::T760;
}

bool MaliInfo::IsMaliT8xx() const {
  return gpu_version == MaliGPU::T820 || gpu_version == MaliGPU::T830 ||
         gpu_version == MaliGPU::T860 || gpu_version == MaliGPU::T880;
}

bool MaliInfo::IsMidgard() const {
  return IsMaliT6xx() || IsMaliT7xx() || IsMaliT8xx();
}

bool MaliInfo::IsBifrostGen1() const {
  return gpu_version == MaliGPU::G31 || gpu_version == MaliGPU::G51 ||
         gpu_version == MaliGPU::G71;
}

bool MaliInfo::IsBifrostGen2() const {
  return gpu_version == MaliGPU::G52 || gpu_version == MaliGPU::G72;
}

bool MaliInfo::IsBifrostGen3() const { return gpu_version == MaliGPU::G76; }

bool MaliInfo::IsBifrost() const {
  return IsBifrostGen1() || IsBifrostGen2() || IsBifrostGen3();
}

bool MaliInfo::IsValhall() const {
  return gpu_version == MaliGPU::G57 || gpu_version == MaliGPU::G77 ||
         gpu_version == MaliGPU::G68 || gpu_version == MaliGPU::G78;
}

bool DeviceInfo::SupportsTextureArray() const {
  return cl_version >= OpenCLVersion::CL_1_2;
}

bool DeviceInfo::SupportsImageBuffer() const {
  return cl_version >= OpenCLVersion::CL_1_2;
}

bool DeviceInfo::SupportsImage3D() const {
  if (vendor == Vendor::kMali) {
    // On Mali T880 read_imageh doesn't compile with image3d_t
    return false;
  }
  return supports_image3d_writes;
}

bool DeviceInfo::SupportsFloatImage2D(DataType data_type, int channels) const {
  if (channels == 1) {
    return data_type == DataType::FLOAT32 ? supports_r_f32_tex2d
                                          : supports_r_f16_tex2d;
  } else if (channels == 2) {
    return data_type == DataType::FLOAT32 ? supports_rg_f32_tex2d
                                          : supports_rg_f16_tex2d;
  } else if (channels == 3) {
    return data_type == DataType::FLOAT32 ? supports_rgb_f32_tex2d
                                          : supports_rgb_f16_tex2d;
  } else if (channels == 4) {
    return data_type == DataType::FLOAT32 ? supports_rgba_f32_tex2d
                                          : supports_rgba_f16_tex2d;
  } else {
    return false;
  }
}

bool DeviceInfo::SupportsOneLayerTextureArray() const {
  return !IsAdreno() || adreno_info.support_one_layer_texture_array;
}

bool DeviceInfo::SupportsExtension(const std::string& extension) const {
  for (const auto& ext : extensions) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

bool DeviceInfo::IsCL20OrHigher() const {
  return cl_version != OpenCLVersion::CL_1_0 &&
         cl_version != OpenCLVersion::CL_1_1 &&
         cl_version != OpenCLVersion::CL_1_2;
}

bool DeviceInfo::SupportsSubGroupWithSize(int sub_group_size) const {
  for (auto subgroup_size : supported_subgroup_sizes) {
    if (sub_group_size == subgroup_size) {
      return true;
    }
  }
  return false;
}

bool DeviceInfo::IsAdreno() const { return vendor == Vendor::kQualcomm; }

bool DeviceInfo::IsAdreno3xx() const {
  return IsAdreno() && IsGPUVersionInRange(adreno_info.gpu_version, 300, 400);
}

bool DeviceInfo::IsAdreno4xx() const {
  return IsAdreno() && IsGPUVersionInRange(adreno_info.gpu_version, 400, 500);
}

bool DeviceInfo::IsAdreno5xx() const {
  return IsAdreno() && IsGPUVersionInRange(adreno_info.gpu_version, 500, 600);
}

bool DeviceInfo::IsAdreno6xx() const {
  return IsAdreno() && IsGPUVersionInRange(adreno_info.gpu_version, 600, 700);
}

bool DeviceInfo::IsAdreno6xxOrHigher() const {
  return IsAdreno() && adreno_info.gpu_version >= 600;
}

bool DeviceInfo::IsPowerVR() const { return vendor == Vendor::kPowerVR; }

bool DeviceInfo::IsNvidia() const { return vendor == Vendor::kNvidia; }

bool DeviceInfo::IsMali() const { return vendor == Vendor::kMali; }

bool DeviceInfo::IsAMD() const { return vendor == Vendor::kAMD; }

bool DeviceInfo::IsIntel() const { return vendor == Vendor::kIntel; }

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
