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

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"

#include <map>
#include <string>

#include "absl/strings/ascii.h"

namespace tflite {
namespace gpu {
namespace {

GpuVendor GetGpuVendor(const std::string& gpu_description) {
  const std::map<std::string, GpuVendor> kMapping = {
      {"adreno", GpuVendor::kQualcomm},
      {"apple", GpuVendor::kApple},
      {"qualcomm", GpuVendor::kQualcomm},
      {"mali", GpuVendor::kMali},
      {"powervr", GpuVendor::kPowerVR},
      {"advanced micro devices", GpuVendor::kAMD},
      {"intel", GpuVendor::kIntel},
      {"nvidia", GpuVendor::kNvidia},
      {"amd", GpuVendor::kAMD},
      {"power", GpuVendor::kPowerVR},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return GpuVendor::kUnknown;
}

AdrenoGpu GetAdrenoGpuVersion(const std::string& gpu_description) {
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
    if (gpu_description.find(v.first) != std::string::npos) {
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

AppleInfo::AppleInfo(const std::string& gpu_description) {
  const std::map<std::string, AppleGpu> kMapping = {
      {"apple a7 gpu", AppleGpu::kA7},     {"apple a8 gpu", AppleGpu::kA8},
      {"apple a8x gpu", AppleGpu::kA8X},   {"apple a9 gpu", AppleGpu::kA9},
      {"apple a9x gpu", AppleGpu::kA9X},   {"apple a10 gpu", AppleGpu::kA10},
      {"apple a10x gpu", AppleGpu::kA10X}, {"apple a11 gpu", AppleGpu::kA11},
      {"apple a12 gpu", AppleGpu::kA12},   {"apple a12x gpu", AppleGpu::kA12X},
      {"apple a12z gpu", AppleGpu::kA12Z}, {"apple a13 gpu", AppleGpu::kA13},
      {"apple a14 gpu", AppleGpu::kA14},
  };
  auto it = kMapping.find(gpu_description);
  if (it != kMapping.end()) {
    gpu_type = it->second;
  } else {
    gpu_type = AppleGpu::kUnknown;
  }
}

bool AppleInfo::IsLocalMemoryPreferredOverGlobal() const {
  return gpu_type == AppleGpu::kA7 || gpu_type == AppleGpu::kA8 ||
         gpu_type == AppleGpu::kA8X;
}

bool AppleInfo::IsBionic() const {
  return gpu_type == AppleGpu::kA11 || gpu_type == AppleGpu::kA12 ||
         gpu_type == AppleGpu::kA12X || gpu_type == AppleGpu::kA12Z ||
         gpu_type == AppleGpu::kA13 || gpu_type == AppleGpu::kA14;
}

bool AppleInfo::IsRoundToNearestSupported() const { return IsBionic(); }

int AppleInfo::GetComputeUnitsCount() const {
  switch (gpu_type) {
    case AppleGpu::kA7:
      return 4;
    case AppleGpu::kA8:
      return 4;
    case AppleGpu::kA8X:
      return 8;
    case AppleGpu::kA9:
      return 6;
    case AppleGpu::kA9X:
      return 12;
    case AppleGpu::kA10:
      return 6;
    case AppleGpu::kA10X:
      return 12;
    case AppleGpu::kA11:
      return 3;
    case AppleGpu::kA12:
      return 4;
    case AppleGpu::kA12X:
      return 7;
    case AppleGpu::kA12Z:
      return 8;
    case AppleGpu::kA13:
      return 4;
    case AppleGpu::kA14:
      return 4;
    case AppleGpu::kUnknown:
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

void GetGpuInfoFromDeviceDescription(const std::string& gpu_description,
                                     GpuApi gpu_api, GpuInfo* gpu_info) {
  gpu_info->gpu_api = gpu_api;
  std::string lowered = gpu_description;
  absl::AsciiStrToLower(&lowered);
  gpu_info->vendor = GetGpuVendor(lowered);
  if (gpu_info->IsAdreno()) {
    gpu_info->adreno_info = AdrenoInfo(lowered);
  } else if (gpu_info->IsApple()) {
    gpu_info->apple_info = AppleInfo(lowered);
    gpu_info->supported_subgroup_sizes = {32};
  } else if (gpu_info->IsMali()) {
    gpu_info->mali_info = MaliInfo(lowered);
  }
}

bool GpuInfo::IsAdreno() const { return vendor == GpuVendor::kQualcomm; }

bool GpuInfo::IsApple() const { return vendor == GpuVendor::kApple; }

bool GpuInfo::IsMali() const { return vendor == GpuVendor::kMali; }

bool GpuInfo::IsPowerVR() const { return vendor == GpuVendor::kPowerVR; }

bool GpuInfo::IsNvidia() const { return vendor == GpuVendor::kNvidia; }

bool GpuInfo::IsAMD() const { return vendor == GpuVendor::kAMD; }

bool GpuInfo::IsIntel() const { return vendor == GpuVendor::kIntel; }

bool GpuInfo::IsRoundToNearestSupported() const {
  if (IsApple()) {
    return apple_info.IsRoundToNearestSupported();
  } else {
    return true;
  }
}

bool GpuInfo::IsWaveSizeEqualTo32() const {
  return supported_subgroup_sizes.size() == 1 &&
         supported_subgroup_sizes[0] == 32;
}

int GpuInfo::GetComputeUnitsCount() const {
  if (IsApple()) {
    return apple_info.GetComputeUnitsCount();
  } else {
    return 1;
  }
}

int GpuInfo::GetMaxImageArguments() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_image_units;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_per_stage_descriptor_sampled_images;
  } else if (IsApiMetal()) {
    return 32;
  } else if (IsApiOpenCl()) {
    return 128;
  } else {
    return 1;
  }
}

bool GpuInfo::IsApiOpenGl() const { return gpu_api == GpuApi::kOpenGl; }

bool GpuInfo::IsApiVulkan() const { return gpu_api == GpuApi::kVulkan; }

bool GpuInfo::IsApiMetal() const { return gpu_api == GpuApi::kMetal; }

bool GpuInfo::IsApiOpenCl() const { return gpu_api == GpuApi::kOpenCl; }

bool GpuInfo::IsApiOpenGl31OrAbove() const {
  if (!IsApiOpenGl()) {
    return false;
  }
  return (opengl_info.major_version == 3 && opengl_info.minor_version >= 1) ||
         opengl_info.major_version > 3;
}

}  // namespace gpu
}  // namespace tflite
