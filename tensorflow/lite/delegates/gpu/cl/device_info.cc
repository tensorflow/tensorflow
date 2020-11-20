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
AdrenoGpu GetAdrenoGpuVersion(const std::string& device_name) {
  const std::map<std::string, AdrenoGpu> kMapping = {
      // Adreno 6xx series
      {"685", AdrenoGpu::kAdreno685},
      {"680", AdrenoGpu::kAdreno680},
      {"675", AdrenoGpu::kAdreno675},
      {"650", AdrenoGpu::kAdreno650},
      {"640", AdrenoGpu::kAdreno640},
      {"630", AdrenoGpu::kAdreno630},
      {"620", AdrenoGpu::kAdreno620},
      {"616", AdrenoGpu::kAdreno618},
      {"616", AdrenoGpu::kAdreno616},
      {"615", AdrenoGpu::kAdreno615},
      {"612", AdrenoGpu::kAdreno612},
      {"610", AdrenoGpu::kAdreno610},
      {"605", AdrenoGpu::kAdreno605},
      // Adreno 5xx series
      {"540", AdrenoGpu::kAdreno540},
      {"530", AdrenoGpu::kAdreno530},
      {"512", AdrenoGpu::kAdreno512},
      {"510", AdrenoGpu::kAdreno510},
      {"509", AdrenoGpu::kAdreno509},
      {"508", AdrenoGpu::kAdreno508},
      {"506", AdrenoGpu::kAdreno506},
      {"505", AdrenoGpu::kAdreno505},
      {"504", AdrenoGpu::kAdreno504},
      // Adreno 4xx series
      {"430", AdrenoGpu::kAdreno430},
      {"420", AdrenoGpu::kAdreno420},
      {"418", AdrenoGpu::kAdreno418},
      {"405", AdrenoGpu::kAdreno405},
      // Adreno 3xx series
      {"330", AdrenoGpu::kAdreno330},
      {"320", AdrenoGpu::kAdreno320},
      {"308", AdrenoGpu::kAdreno308},
      {"306", AdrenoGpu::kAdreno306},
      {"305", AdrenoGpu::kAdreno305},
      {"304", AdrenoGpu::kAdreno304},
      // Adreno 2xx series
      {"225", AdrenoGpu::kAdreno225},
      {"220", AdrenoGpu::kAdreno220},
      {"205", AdrenoGpu::kAdreno205},
      {"203", AdrenoGpu::kAdreno203},
      {"200", AdrenoGpu::kAdreno200},
      // Adreno 1xx series
      {"130", AdrenoGpu::kAdreno130},
      {"120", AdrenoGpu::kAdreno120},
  };

  for (const auto& v : kMapping) {
    if (device_name.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return AdrenoGpu::kUnknown;
}

MaliGpu GetMaliGpuVersion(const std::string& gpu_description) {
  const std::map<std::string, MaliGpu> kMapping = {
      {"t604", MaliGpu::kT604}, {"t622", MaliGpu::kT622},
      {"t624", MaliGpu::kT624}, {"t628", MaliGpu::kT628},
      {"t658", MaliGpu::kT658}, {"t678", MaliGpu::kT678},
      {"t720", MaliGpu::kT720}, {"t760", MaliGpu::kT760},
      {"t820", MaliGpu::kT820}, {"t830", MaliGpu::kT830},
      {"t860", MaliGpu::kT860}, {"t880", MaliGpu::kT880},
      {"g31", MaliGpu::kG31},   {"g51", MaliGpu::kG51},
      {"g71", MaliGpu::kG71},   {"g52", MaliGpu::kG52},
      {"g72", MaliGpu::kG72},   {"g76", MaliGpu::kG76},
      {"g57", MaliGpu::kG57},   {"g77", MaliGpu::kG77},
      {"g68", MaliGpu::kG68},   {"g78", MaliGpu::kG78},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return MaliGpu::kUnknown;
}

}  // namespace

std::string GpuVendorToString(GpuVendor v) {
  switch (v) {
    case GpuVendor::kApple:
      return "Apple";
    case GpuVendor::kQualcomm:
      return "Qualcomm";
    case GpuVendor::kMali:
      return "Mali";
    case GpuVendor::kPowerVR:
      return "PowerVR";
    case GpuVendor::kNvidia:
      return "NVIDIA";
    case GpuVendor::kAMD:
      return "AMD";
    case GpuVendor::kIntel:
      return "Intel";
    case GpuVendor::kUnknown:
      return "unknown vendor";
  }
}

std::string OpenClVersionToString(OpenClVersion version) {
  switch (version) {
    case OpenClVersion::kCl1_0:
      return "1.0";
    case OpenClVersion::kCl1_1:
      return "1.1";
    case OpenClVersion::kCl1_2:
      return "1.2";
    case OpenClVersion::kCl2_0:
      return "2.0";
    case OpenClVersion::kCl2_1:
      return "2.1";
    case OpenClVersion::kCl2_2:
      return "2.2";
    case OpenClVersion::kCl3_0:
      return "3.0";
    default:
      return "Unknown OpenCL version";
  }
}

AdrenoInfo::AdrenoInfo(const std::string& device_version)
    : adreno_gpu(GetAdrenoGpuVersion(device_version)) {}

bool AdrenoInfo::IsAdreno1xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno120 ||
         adreno_gpu == AdrenoGpu::kAdreno130;
}

bool AdrenoInfo::IsAdreno2xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno200 ||
         adreno_gpu == AdrenoGpu::kAdreno203 ||
         adreno_gpu == AdrenoGpu::kAdreno205 ||
         adreno_gpu == AdrenoGpu::kAdreno220 ||
         adreno_gpu == AdrenoGpu::kAdreno225;
}

bool AdrenoInfo::IsAdreno3xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno304 ||
         adreno_gpu == AdrenoGpu::kAdreno305 ||
         adreno_gpu == AdrenoGpu::kAdreno306 ||
         adreno_gpu == AdrenoGpu::kAdreno308 ||
         adreno_gpu == AdrenoGpu::kAdreno320 ||
         adreno_gpu == AdrenoGpu::kAdreno330;
}

bool AdrenoInfo::IsAdreno4xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno405 ||
         adreno_gpu == AdrenoGpu::kAdreno418 ||
         adreno_gpu == AdrenoGpu::kAdreno420 ||
         adreno_gpu == AdrenoGpu::kAdreno430;
}

bool AdrenoInfo::IsAdreno5xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno504 ||
         adreno_gpu == AdrenoGpu::kAdreno505 ||
         adreno_gpu == AdrenoGpu::kAdreno506 ||
         adreno_gpu == AdrenoGpu::kAdreno508 ||
         adreno_gpu == AdrenoGpu::kAdreno509 ||
         adreno_gpu == AdrenoGpu::kAdreno510 ||
         adreno_gpu == AdrenoGpu::kAdreno512 ||
         adreno_gpu == AdrenoGpu::kAdreno530 ||
         adreno_gpu == AdrenoGpu::kAdreno540;
}

bool AdrenoInfo::IsAdreno6xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno605 ||
         adreno_gpu == AdrenoGpu::kAdreno610 ||
         adreno_gpu == AdrenoGpu::kAdreno612 ||
         adreno_gpu == AdrenoGpu::kAdreno615 ||
         adreno_gpu == AdrenoGpu::kAdreno616 ||
         adreno_gpu == AdrenoGpu::kAdreno618 ||
         adreno_gpu == AdrenoGpu::kAdreno620 ||
         adreno_gpu == AdrenoGpu::kAdreno630 ||
         adreno_gpu == AdrenoGpu::kAdreno640 ||
         adreno_gpu == AdrenoGpu::kAdreno650 ||
         adreno_gpu == AdrenoGpu::kAdreno675 ||
         adreno_gpu == AdrenoGpu::kAdreno680 ||
         adreno_gpu == AdrenoGpu::kAdreno685;
}

bool AdrenoInfo::IsAdreno6xxOrHigher() const { return IsAdreno6xx(); }

int AdrenoInfo::GetMaximumWavesCount() const {
  if (IsAdreno6xx()) {
    if (adreno_gpu == AdrenoGpu::kAdreno640) {
      return 30;
    } else {
      return 16;
    }
  } else {
    // all other versions not supported
    return 1;
  }
}

int AdrenoInfo::GetRegisterMemorySizePerComputeUnit() const {
  if (IsAdreno6xx()) {
    if (adreno_gpu == AdrenoGpu::kAdreno640) {
      return 128 * 144 * 16;
    } else if (adreno_gpu == AdrenoGpu::kAdreno650) {
      return 128 * 64 * 16;
    } else {
      return 128 * 96 * 16;
    }
  } else {
    // all other versions not supported
    return 1;
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
  if (IsAdreno6xx()) {
    return full_wave ? 128 : 64;
  } else if (IsAdreno5xx() || IsAdreno4xx()) {
    return full_wave ? 64 : 32;
  } else {
    // all other versions not supported
    return 1;
  }
}

MaliInfo::MaliInfo(const std::string& gpu_description)
    : gpu_version(GetMaliGpuVersion(gpu_description)) {}

bool MaliInfo::IsMaliT6xx() const {
  return gpu_version == MaliGpu::kT604 || gpu_version == MaliGpu::kT622 ||
         gpu_version == MaliGpu::kT624 || gpu_version == MaliGpu::kT628 ||
         gpu_version == MaliGpu::kT658 || gpu_version == MaliGpu::kT678;
}

bool MaliInfo::IsMaliT7xx() const {
  return gpu_version == MaliGpu::kT720 || gpu_version == MaliGpu::kT760;
}

bool MaliInfo::IsMaliT8xx() const {
  return gpu_version == MaliGpu::kT820 || gpu_version == MaliGpu::kT830 ||
         gpu_version == MaliGpu::kT860 || gpu_version == MaliGpu::kT880;
}

bool MaliInfo::IsMidgard() const {
  return IsMaliT6xx() || IsMaliT7xx() || IsMaliT8xx();
}

bool MaliInfo::IsBifrostGen1() const {
  return gpu_version == MaliGpu::kG31 || gpu_version == MaliGpu::kG51 ||
         gpu_version == MaliGpu::kG71;
}

bool MaliInfo::IsBifrostGen2() const {
  return gpu_version == MaliGpu::kG52 || gpu_version == MaliGpu::kG72;
}

bool MaliInfo::IsBifrostGen3() const { return gpu_version == MaliGpu::kG76; }

bool MaliInfo::IsBifrost() const {
  return IsBifrostGen1() || IsBifrostGen2() || IsBifrostGen3();
}

bool MaliInfo::IsValhall() const {
  return gpu_version == MaliGpu::kG57 || gpu_version == MaliGpu::kG77 ||
         gpu_version == MaliGpu::kG68 || gpu_version == MaliGpu::kG78;
}

bool GpuInfo::SupportsTextureArray() const {
  return opencl_info.cl_version >= OpenClVersion::kCl1_2;
}

bool GpuInfo::SupportsImageBuffer() const {
  return opencl_info.cl_version >= OpenClVersion::kCl1_2;
}

bool GpuInfo::SupportsImage3D() const {
  if (IsMali() && mali_info.IsMidgard()) {
    // On Mali T880 read_imageh doesn't compile with image3d_t
    return false;
  }
  return supports_image3d_writes;
}

bool GpuInfo::SupportsFloatImage2D(DataType data_type, int channels) const {
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

bool GpuInfo::SupportsExtension(const std::string& extension) const {
  for (const auto& ext : extensions) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::IsCL20OrHigher() const {
  return opencl_info.cl_version != OpenClVersion::kCl1_0 &&
         opencl_info.cl_version != OpenClVersion::kCl1_1 &&
         opencl_info.cl_version != OpenClVersion::kCl1_2;
}

bool GpuInfo::IsCL30OrHigher() const {
  return IsCL20OrHigher() && opencl_info.cl_version != OpenClVersion::kCl2_0 &&
         opencl_info.cl_version != OpenClVersion::kCl2_1 &&
         opencl_info.cl_version != OpenClVersion::kCl2_2;
}

bool GpuInfo::SupportsSubGroupWithSize(int sub_group_size) const {
  for (auto subgroup_size : supported_subgroup_sizes) {
    if (sub_group_size == subgroup_size) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::IsAdreno() const { return gpu_vendor == GpuVendor::kQualcomm; }

bool GpuInfo::IsApple() const { return gpu_vendor == GpuVendor::kApple; }

bool GpuInfo::IsMali() const { return gpu_vendor == GpuVendor::kMali; }

bool GpuInfo::IsPowerVR() const { return gpu_vendor == GpuVendor::kPowerVR; }

bool GpuInfo::IsNvidia() const { return gpu_vendor == GpuVendor::kNvidia; }

bool GpuInfo::IsAMD() const { return gpu_vendor == GpuVendor::kAMD; }

bool GpuInfo::IsIntel() const { return gpu_vendor == GpuVendor::kIntel; }

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
