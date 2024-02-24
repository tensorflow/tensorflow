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

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

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
      {"radeon", GpuVendor::kAMD},
      {"xclipse", GpuVendor::kAMD},
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
      // Adreno 7xx series
      {"740", AdrenoGpu::kAdreno740},
      {"730", AdrenoGpu::kAdreno730},
      // Adreno 6xx series
      {"685", AdrenoGpu::kAdreno685},
      {"680", AdrenoGpu::kAdreno680},
      {"675", AdrenoGpu::kAdreno675},
      {"660", AdrenoGpu::kAdreno660},
      {"650", AdrenoGpu::kAdreno650},
      {"640", AdrenoGpu::kAdreno640},
      {"630", AdrenoGpu::kAdreno630},
      {"620", AdrenoGpu::kAdreno620},
      {"618", AdrenoGpu::kAdreno618},
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
  // Order must be preserved
  const std::vector<std::pair<std::string, MaliGpu>> kMapping = {
      {"t604", MaliGpu::kT604}, {"t622", MaliGpu::kT622},
      {"t624", MaliGpu::kT624}, {"t628", MaliGpu::kT628},
      {"t658", MaliGpu::kT658}, {"t678", MaliGpu::kT678},
      {"t720", MaliGpu::kT720}, {"t760", MaliGpu::kT760},
      {"t820", MaliGpu::kT820}, {"t830", MaliGpu::kT830},
      {"t860", MaliGpu::kT860}, {"t880", MaliGpu::kT880},
      {"g310", MaliGpu::kG310}, {"g31", MaliGpu::kG31},
      {"g510", MaliGpu::kG510}, {"g51", MaliGpu::kG51},
      {"g52", MaliGpu::kG52},   {"g57", MaliGpu::kG57},
      {"g610", MaliGpu::kG610}, {"g68", MaliGpu::kG68},
      {"g710", MaliGpu::kG710}, {"g715", MaliGpu::kG715},
      {"g71", MaliGpu::kG71},   {"g72", MaliGpu::kG72},
      {"g76", MaliGpu::kG76},   {"g77", MaliGpu::kG77},
      {"g78", MaliGpu::kG78},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return MaliGpu::kUnknown;
}

PowerVRGpu GetPowerVRGpuVersion(const std::string& gpu_description) {
  // Order must be preserved
  const std::vector<std::pair<std::string, PowerVRGpu>> kMapping = {
      {"rogue gm9", PowerVRGpu::kRogueGm9xxx},    // From OpenGL
      {"powervr gm9", PowerVRGpu::kRogueGm9xxx},  /// From OpenCL
      {"rogue ge8", PowerVRGpu::kRogueGe8xxx},    // From OpenGL
      {"powervr ge8", PowerVRGpu::kRogueGe8xxx},  /// From OpenCL
      {"rogue", PowerVRGpu::kRogue},
      {"axe", PowerVRGpu::kAXE},
      {"axm", PowerVRGpu::kAXM},
      {"axt", PowerVRGpu::kAXT},
      {"bxe", PowerVRGpu::kBXE},
      {"bxm", PowerVRGpu::kBXM},
      {"bxs", PowerVRGpu::kBXS},
      {"bxt", PowerVRGpu::kBXT},
      {"cxt", PowerVRGpu::kCXT},
      {"dxt", PowerVRGpu::kDXT},
      {"powervr g", PowerVRGpu::kRogue},
  };
  for (const auto& v : kMapping) {
    if (gpu_description.find(v.first) != std::string::npos) {
      return v.second;
    }
  }

  return PowerVRGpu::kUnknown;
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
         adreno_gpu == AdrenoGpu::kAdreno660 ||
         adreno_gpu == AdrenoGpu::kAdreno675 ||
         adreno_gpu == AdrenoGpu::kAdreno680 ||
         adreno_gpu == AdrenoGpu::kAdreno685;
}

bool AdrenoInfo::IsAdreno7xx() const {
  return adreno_gpu == AdrenoGpu::kAdreno730 ||
         adreno_gpu == AdrenoGpu::kAdreno740;
}

bool AdrenoInfo::IsBetterThan(AdrenoGpu gpu) const {
  // Smaller value is better (recent) version.
  return (adreno_gpu <= gpu);
}

bool AdrenoInfo::IsAdreno6xxOrHigher() const {
  return (!compiler_bugs_in_a6xx && IsAdreno6xx()) || IsAdreno7xx();
}

int AdrenoInfo::GetMaximumWavesCount() const {
  if (IsAdreno7xx()) {
    return 16;
  } else if (IsAdreno6xx()) {
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
  if (IsAdreno7xx()) {
    return 128 * 96 * 16;
  } else if (IsAdreno6xx()) {
    if (adreno_gpu == AdrenoGpu::kAdreno640) {
      return 128 * 144 * 16;
    } else if (adreno_gpu == AdrenoGpu::kAdreno620 ||
               adreno_gpu == AdrenoGpu::kAdreno650 ||
               adreno_gpu == AdrenoGpu::kAdreno660) {
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
  if (IsAdreno7xx()) {
    return full_wave ? 128 : 64;
  } else if (IsAdreno6xx()) {
    return full_wave ? 128 : 64;
  } else if (IsAdreno5xx() || IsAdreno4xx()) {
    return full_wave ? 64 : 32;
  } else {
    return full_wave ? 32 : 16;
  }
}

int AdrenoInfo::GetComputeUnitsCount() const {
  // can provide not correct numbers.
  switch (adreno_gpu) {
    // Adreno 7xx series
    case AdrenoGpu::kAdreno740:
      return 6;
    case AdrenoGpu::kAdreno730:
      return 4;
    // Adreno 6xx series
    case AdrenoGpu::kAdreno685:
      return 4;
    case AdrenoGpu::kAdreno680:
      return 4;
    case AdrenoGpu::kAdreno675:
      return 4;
    case AdrenoGpu::kAdreno660:
      return 3;
    case AdrenoGpu::kAdreno650:
      return 3;
    case AdrenoGpu::kAdreno640:
      return 2;
    case AdrenoGpu::kAdreno630:
      return 2;
    case AdrenoGpu::kAdreno620:
      return 1;
    case AdrenoGpu::kAdreno618:
      return 1;
    case AdrenoGpu::kAdreno616:
      return 1;
    case AdrenoGpu::kAdreno615:
      return 1;
    case AdrenoGpu::kAdreno612:
      return 1;
    case AdrenoGpu::kAdreno610:
      return 1;
    case AdrenoGpu::kAdreno605:
      return 1;
    // Adreno 5xx series
    case AdrenoGpu::kAdreno540:
      return 4;
    case AdrenoGpu::kAdreno530:
      return 4;
    case AdrenoGpu::kAdreno512:
      return 2;
    case AdrenoGpu::kAdreno510:
      return 2;
    case AdrenoGpu::kAdreno509:
      return 2;
    case AdrenoGpu::kAdreno508:
      return 1;
    case AdrenoGpu::kAdreno506:
      return 1;
    case AdrenoGpu::kAdreno505:
      return 1;
    case AdrenoGpu::kAdreno504:
      return 1;
    // Adreno 4xx series
    case AdrenoGpu::kAdreno430:
      return 4;
    case AdrenoGpu::kAdreno420:
      return 4;
    case AdrenoGpu::kAdreno418:
      return 2;
    case AdrenoGpu::kAdreno405:
      return 1;
    // Adreno 3xx series
    case AdrenoGpu::kAdreno330:
      return 4;
    case AdrenoGpu::kAdreno320:
      return 2;
    case AdrenoGpu::kAdreno308:
      return 1;
    case AdrenoGpu::kAdreno306:
      return 1;
    case AdrenoGpu::kAdreno305:
      return 1;
    case AdrenoGpu::kAdreno304:
      return 1;
    default:
      return 1;
  }
}

AppleInfo::AppleInfo(const std::string& gpu_description) {
  const std::map<std::string, AppleGpu> kMapping = {
      {"apple a7 gpu", AppleGpu::kA7},
      {"apple a8 gpu", AppleGpu::kA8},
      {"apple a8x gpu", AppleGpu::kA8X},
      {"apple a9 gpu", AppleGpu::kA9},
      {"apple a9x gpu", AppleGpu::kA9X},
      {"apple a10 gpu", AppleGpu::kA10},
      {"apple a10x gpu", AppleGpu::kA10X},
      {"apple a11 gpu", AppleGpu::kA11},
      {"apple a12 gpu", AppleGpu::kA12},
      {"apple a12x gpu", AppleGpu::kA12X},
      {"apple a12z gpu", AppleGpu::kA12Z},
      {"apple a13 gpu", AppleGpu::kA13},
      {"apple a14 gpu", AppleGpu::kA14},
      {"apple a15 gpu", AppleGpu::kA15},
      {"apple a16 gpu", AppleGpu::kA16},
      {"apple a17 pro gpu", AppleGpu::kA17Pro},
      // on tablets we have metal device name "apple m1 gpu"
      // and on notebooks "apple m1"
      {"apple m1 gpu", AppleGpu::kM1},
      {"apple m1", AppleGpu::kM1},
      {"apple m1 pro", AppleGpu::kM1Pro},
      {"apple m1 max", AppleGpu::kM1Max},
      {"apple m1 ultra", AppleGpu::kM1Ultra},
      {"apple m2", AppleGpu::kM2},
  };
  auto it = kMapping.find(gpu_description);
  if (it != kMapping.end()) {
    gpu_type = it->second;
  } else {
    gpu_type = AppleGpu::kUnknown;
  }
  gpu_family = GetGpuFamily();
}

AppleInfo::Family AppleInfo::GetGpuFamily() const {
  if (gpu_type == AppleGpu::kA7) {
    return AppleInfo::Family::kApple1;
  } else if (gpu_type == AppleGpu::kA8 || gpu_type == AppleGpu::kA8X) {
    return AppleInfo::Family::kApple2;
  } else if (gpu_type == AppleGpu::kA9 || gpu_type == AppleGpu::kA9X ||
             gpu_type == AppleGpu::kA10 || gpu_type == AppleGpu::kA10X) {
    return AppleInfo::Family::kApple3;
  } else if (gpu_type == AppleGpu::kA11) {
    return AppleInfo::Family::kApple4;
  } else if (gpu_type == AppleGpu::kA12 || gpu_type == AppleGpu::kA12X ||
             gpu_type == AppleGpu::kA12Z) {
    return AppleInfo::Family::kApple5;
  } else if (gpu_type == AppleGpu::kA13) {
    return AppleInfo::Family::kApple6;
  } else if (gpu_type == AppleGpu::kA14 || IsM1Series()) {
    return AppleInfo::Family::kApple7;
  } else if (gpu_type == AppleGpu::kA15 || gpu_type == AppleGpu::kA16 ||
             gpu_type == AppleGpu::kM2) {
    return AppleInfo::Family::kApple8;
  } else if (gpu_type == AppleGpu::kA17Pro) {
    return AppleInfo::Family::kApple9;
  }
  return AppleInfo::Family::kApple1;
}

bool AppleInfo::IsFamilyApple1() const {
  return gpu_family == AppleInfo::Family::kApple1;
}

bool AppleInfo::IsFamilyApple2() const {
  return gpu_family == AppleInfo::Family::kApple2;
}

bool AppleInfo::IsFamilyApple3() const {
  return gpu_family == AppleInfo::Family::kApple3;
}

bool AppleInfo::IsFamilyApple4() const {
  return gpu_family == AppleInfo::Family::kApple4;
}

bool AppleInfo::IsFamilyApple5() const {
  return gpu_family == AppleInfo::Family::kApple5;
}

bool AppleInfo::IsFamilyApple6() const {
  return gpu_family == AppleInfo::Family::kApple6;
}

bool AppleInfo::IsFamilyApple7() const {
  return gpu_family == AppleInfo::Family::kApple7;
}

bool AppleInfo::IsFamilyApple8() const {
  return gpu_family == AppleInfo::Family::kApple8;
}

bool AppleInfo::IsFamilyApple9() const {
  return gpu_family == AppleInfo::Family::kApple9;
}

bool AppleInfo::IsFamilyOrLower(AppleInfo::Family family) const {
  return gpu_family <= family;
}

bool AppleInfo::IsLocalMemoryPreferredOverGlobal() const {
  return IsFamilyOrLower(AppleInfo::Family::kApple2);
}

bool AppleInfo::IsM1Series() const {
  return gpu_type == AppleGpu::kM1 || gpu_type == AppleGpu::kM1Pro ||
         gpu_type == AppleGpu::kM1Max || gpu_type == AppleGpu::kM1Ultra;
}

bool AppleInfo::IsBionic() const {
  return gpu_type == AppleGpu::kA11 || gpu_type == AppleGpu::kA12 ||
         gpu_type == AppleGpu::kA12X || gpu_type == AppleGpu::kA12Z ||
         gpu_type == AppleGpu::kA13 || gpu_type == AppleGpu::kA14 ||
         gpu_type == AppleGpu::kA15 || gpu_type == AppleGpu::kA16 ||
         gpu_type == AppleGpu::kA17Pro || gpu_type == AppleGpu::kM1 ||
         gpu_type == AppleGpu::kM1Pro || gpu_type == AppleGpu::kM1Max ||
         gpu_type == AppleGpu::kM1Ultra || gpu_type == AppleGpu::kM2;
}

bool AppleInfo::IsSIMDMatMulSupported() const {
  return gpu_type == AppleGpu::kA14 || gpu_type == AppleGpu::kA15 ||
         gpu_type == AppleGpu::kA16 || gpu_type == AppleGpu::kA17Pro ||
         gpu_type == AppleGpu::kM1 || gpu_type == AppleGpu::kM1Pro ||
         gpu_type == AppleGpu::kM1Max || gpu_type == AppleGpu::kM1Ultra ||
         gpu_type == AppleGpu::kM2;
}

bool AppleInfo::IsSIMDMatMulFp32Perf2x() const {
  return gpu_type == AppleGpu::kA15 || gpu_type == AppleGpu::kA16 ||
         gpu_type == AppleGpu::kA17Pro || gpu_type == AppleGpu::kM2;
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
    // For some apple GPUs we can not receive exact CU count from name.
    // No official Metal API to receive this info.
    case AppleGpu::kA15:
      if (compute_units != -1) {
        return compute_units;
      }
      return 5;
    case AppleGpu::kA16:
      return 5;
    case AppleGpu::kA17Pro:
      return 6;
    case AppleGpu::kM1:
      // approximate, can be 7 or 8
      return 8;
    case AppleGpu::kM1Pro:
      // approximate, can be 14 or 16
      return 16;
    case AppleGpu::kM1Max:
      // approximate, can be 24 or 32
      return 32;
    case AppleGpu::kM1Ultra:
      // approximate, 64 is max possible
      return 64;
    case AppleGpu::kM2:
      // approximate, 10 is max possible
      return 10;
    case AppleGpu::kUnknown:
      return 4;
  }
}

void AppleInfo::SetComputeUnits(int compute_units_count) {
  compute_units = compute_units_count;
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

bool MaliInfo::IsValhallGen1() const {
  return gpu_version == MaliGpu::kG57 || gpu_version == MaliGpu::kG77;
}

bool MaliInfo::IsValhallGen2() const {
  return gpu_version == MaliGpu::kG68 || gpu_version == MaliGpu::kG78;
}

bool MaliInfo::IsValhallGen3() const {
  return gpu_version == MaliGpu::kG310 || gpu_version == MaliGpu::kG510 ||
         gpu_version == MaliGpu::kG610 || gpu_version == MaliGpu::kG710;
}

bool MaliInfo::IsValhallGen4() const { return gpu_version == MaliGpu::kG715; }

bool MaliInfo::IsValhall() const {
  return IsValhallGen1() || IsValhallGen2() || IsValhallGen3() ||
         IsValhallGen4();
}

int MaliInfo::GetApproximateComputeUnitsCount() const {
  if (IsMidgard()) {
    // Mali Midgard can have 1-16 cores
    return 8;
  } else if (IsBifrost()) {
    // Mali Bifrost can have 1-32 cores
    return 16;
  } else if (IsValhall()) {
    if (gpu_version == MaliGpu::kG57) {
      return 6;  // Mali-G57 can have 1-6 cores
    } else if (gpu_version == MaliGpu::kG77) {
      return 16;  // Mali-G77 can have 7-16 cores
    } else if (gpu_version == MaliGpu::kG68) {
      return 6;  // Mali-G68 can have 4-6 cores
    } else if (gpu_version == MaliGpu::kG78) {
      return 16;  // Mali-G78 can have 7-24 cores
    } else if (gpu_version == MaliGpu::kG310 || gpu_version == MaliGpu::kG510 ||
               gpu_version == MaliGpu::kG610) {
      return 6;  // Mali-G310/G510/G610 can have up to 6 cores
    } else if (gpu_version == MaliGpu::kG710 || gpu_version == MaliGpu::kG715) {
      return 10;  // Mali-G710/G715 can have 7–16 cores
    }
  }
  return 4;
}

PowerVRInfo::PowerVRInfo(const std::string& gpu_description)
    : gpu_version(GetPowerVRGpuVersion(gpu_description)) {}

bool PowerVRInfo::IsRogue() const {
  return gpu_version == PowerVRGpu::kRogue ||
         gpu_version == PowerVRGpu::kRogueGe8xxx ||
         gpu_version == PowerVRGpu::kRogueGm9xxx;
}

bool PowerVRInfo::IsImgAxx() const {
  return gpu_version == PowerVRGpu::kAXE || gpu_version == PowerVRGpu::kAXM ||
         gpu_version == PowerVRGpu::kAXT;
}

bool PowerVRInfo::IsImgBxx() const {
  return gpu_version == PowerVRGpu::kBXE || gpu_version == PowerVRGpu::kBXM ||
         gpu_version == PowerVRGpu::kBXS || gpu_version == PowerVRGpu::kBXT;
}

bool PowerVRInfo::IsImgCxx() const { return gpu_version == PowerVRGpu::kCXT; }

bool PowerVRInfo::IsImgDxx() const { return gpu_version == PowerVRGpu::kDXT; }

bool PowerVRInfo::IsBetterThan(PowerVRGpu gpu) const {
  // Smaller value is better (recent) version.
  return (gpu_version <= gpu);
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
  } else if (gpu_info->IsPowerVR()) {
    gpu_info->powervr_info = PowerVRInfo(lowered);
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

bool OpenGlInfo::SupportsExplicitFp16() const {
  bool supports_f16_alu = false;
  bool supports_f16_storage = false;
  for (const auto& ext : extensions) {
    if (ext == "GL_EXT_shader_explicit_arithmetic_types_float16") {
      supports_f16_alu = true;
    }
    if (ext == "GL_EXT_shader_16bit_storage") {
      supports_f16_storage = true;
    }
  }
  return supports_f16_alu && supports_f16_storage;
}

bool OpenGlInfo::IsApiOpenGl31OrAbove() const {
  return (major_version == 3 && minor_version >= 1) || major_version > 3;
}

bool OpenGlInfo::IsApiOpenGl32OrAbove() const {
  return (major_version == 3 && minor_version >= 2) || major_version > 3;
}

bool VulkanInfo::SupportsExplicitFp16() const {
  bool supports_f16_alu = false;
  bool supports_f16_storage = false;
  for (const auto& ext : extensions) {
    if (ext == "VK_KHR_shader_float16_int8") {
      supports_f16_alu = true;
    }
    if (ext == "VK_KHR_16bit_storage") {
      supports_f16_storage = true;
    }
  }
  return supports_f16_alu && supports_f16_storage;
}

bool OpenClInfo::SupportedImage2dTypes::SupportsImage2D(DataType data_type,
                                                        int channels) const {
  if (channels == 1) {
    return r_layout.find(data_type) != r_layout.end();
  } else if (channels == 2) {
    return rg_layout.find(data_type) != rg_layout.end();
  } else if (channels == 3) {
    return rgb_layout.find(data_type) != rgb_layout.end();
  } else if (channels == 4) {
    return rgba_layout.find(data_type) != rgba_layout.end();
  } else {
    return false;
  }
}

bool OpenClInfo::IsImage2dFromBufferSupported() const {
  if (image_pitch_alignment == 0) {
    return false;
  }
  if (image_base_address_alignment == 0) {
    return false;
  }
  if (cl_version == OpenClVersion::kCl2_0 ||
      cl_version == OpenClVersion::kCl2_1 ||
      cl_version == OpenClVersion::kCl2_2) {
    return true;
  }
  for (const auto& ext : extensions) {
    if (ext == "cl_khr_image2d_from_buffer") {
      return true;
    }
  }
  return false;
}

bool MetalInfo::IsSIMDMatMulSupported() const {
  if (language_version == MetalLanguageVersion::kUnknown ||
      language_version == MetalLanguageVersion::kMetal1_0 ||
      language_version == MetalLanguageVersion::kMetal1_1 ||
      language_version == MetalLanguageVersion::kMetal1_2 ||
      language_version == MetalLanguageVersion::kMetal2_0 ||
      language_version == MetalLanguageVersion::kMetal2_1 ||
      language_version == MetalLanguageVersion::kMetal2_2) {
    return false;
  }
  return true;
}

bool MetalInfo::IsMslVersionEqualOrHigher(int major, int minor) const {
  const std::map<MetalLanguageVersion, std::pair<int, int>> kMapping = {
      {MetalLanguageVersion::kUnknown, {1, 0}},
      {MetalLanguageVersion::kMetal1_0, {1, 0}},
      {MetalLanguageVersion::kMetal1_1, {1, 1}},
      {MetalLanguageVersion::kMetal1_2, {1, 2}},
      {MetalLanguageVersion::kMetal2_0, {2, 0}},
      {MetalLanguageVersion::kMetal2_1, {2, 1}},
      {MetalLanguageVersion::kMetal2_2, {2, 2}},
      {MetalLanguageVersion::kMetal2_3, {2, 3}},
      {MetalLanguageVersion::kMetal2_4, {2, 4}},
      {MetalLanguageVersion::kMetal3_0, {3, 0}}};
  auto version = kMapping.at(language_version);
  if (major > version.first) {
    return true;
  } else if (major == version.first && minor >= version.second) {
    return true;
  } else {
    return false;
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
  if (IsApiOpenCl()) {
    return opencl_info.supports_fp16_rtn || opencl_info.supports_fp32_rtn;
  }
  if (IsApple()) {
    return apple_info.IsRoundToNearestSupported();
  }
  if (IsAdreno()) {
    if (adreno_info.IsAdreno1xx() || adreno_info.IsAdreno2xx() ||
        adreno_info.IsAdreno3xx()) {
      return false;
    }
  }
  if (IsPowerVR()) {
    return false;
  }
  return true;
}

bool GpuInfo::SupportsFP16() const {
  if (IsApiOpenCl()) {
    return opencl_info.supports_fp16;
  }
  return true;
}

bool GpuInfo::SupportsTextureArray() const {
  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    return opencl_info.cl_version >= OpenClVersion::kCl1_2;
  }
  return true;
}

bool GpuInfo::SupportsImageBuffer() const {
  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    return opencl_info.cl_version >= OpenClVersion::kCl1_2;
  }
  return true;
}

bool GpuInfo::SupportsImage3D() const {
  if (!SupportsImages()) {
    return false;
  }
  if (IsApiOpenCl()) {
    if (IsMali() && mali_info.IsMidgard()) {
      // On Mali T880 read_imageh doesn't compile with image3d_t
      return false;
    }
    return opencl_info.supports_image3d_writes;
  }
  return true;
}

bool GpuInfo::SupportsImages() const {
  if (IsApiOpenCl()) {
    return opencl_info.supports_images;
  }
  return true;
}

bool GpuInfo::SupportsPointersInKernels() const {
  return IsApiOpenCl() || IsApiMetal();
}

bool GpuInfo::SupportsZeroClampForImageBuffer() const {
  if (IsApiMetal() || IsApiOpenCl()) {
    return true;
  } else {
    return false;
  }
}

bool GpuInfo::SupportsZeroClampForImages() const {
  if (IsApiMetal()) {
    return true;
  } else if (IsApiOpenCl()) {
    return true;
  } else if (IsApiVulkan()) {
    return false;
  } else if (IsApiOpenGl()) {
    return false;
  } else {
    return false;
  }
}

bool GpuInfo::IsWaveSizeEqualTo32() const {
  return supported_subgroup_sizes.size() == 1 &&
         supported_subgroup_sizes[0] == 32;
}

bool GpuInfo::SupportsExtension(const std::string& extension) const {
  const std::vector<std::string>* extensions = nullptr;
  if (IsApiOpenGl()) {
    extensions = &opengl_info.extensions;
  } else if (IsApiVulkan()) {
    extensions = &vulkan_info.extensions;
  } else if (IsApiOpenCl()) {
    extensions = &opencl_info.extensions;
  }
  if (!extensions) {
    return false;
  }
  for (const auto& ext : *extensions) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::SupportsSubGroupWithSize(int sub_group_size) const {
  for (auto subgroup_size : supported_subgroup_sizes) {
    if (sub_group_size == subgroup_size) {
      return true;
    }
  }
  return false;
}

bool GpuInfo::SupportsFloatImage2D(DataType data_type, int channels) const {
  if (IsApiOpenCl()) {
    return opencl_info.supported_images_2d.SupportsImage2D(data_type, channels);
  }
  return false;
}

int GpuInfo::GetComputeUnitsCount() const {
  if (IsApiOpenCl()) {
    return opencl_info.compute_units_count;
  }
  if (IsApple()) {
    return apple_info.GetComputeUnitsCount();
  }
  if (IsAMD()) {
    if (amd_info.GetComputeUnitsCount() != 0) {
      return amd_info.GetComputeUnitsCount();
    } else {
      // approximate number
      return 16;
    }
  }
  if (IsAdreno()) {
    return adreno_info.GetComputeUnitsCount();
  }
  if (IsMali()) {
    return mali_info.GetApproximateComputeUnitsCount();
  }
  return 4;
}

int GpuInfo::GetMaxWorkGroupSizeForX() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_compute_work_group_size_x;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_size_x;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_size_x;
  }
  if (IsApiMetal()) {
    return metal_info.max_work_group_size_x;
  }
  return 256;
}

int GpuInfo::GetMaxWorkGroupSizeForY() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_compute_work_group_size_y;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_size_y;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_size_y;
  }
  if (IsApiMetal()) {
    return metal_info.max_work_group_size_y;
  }
  return 256;
}

int GpuInfo::GetMaxWorkGroupSizeForZ() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_compute_work_group_size_z;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_size_z;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_size_z;
  }
  if (IsApiMetal()) {
    return metal_info.max_work_group_size_z;
  }
  return 64;
}

int GpuInfo::GetMaxWorkGroupTotalSize() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_work_group_invocations;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_compute_work_group_invocations;
  }
  if (IsApiOpenCl()) {
    return opencl_info.max_work_group_total_size;
  }
  if (IsApiMetal()) {
    int max_size = metal_info.max_work_group_size_x;
    max_size = std::max(max_size, metal_info.max_work_group_size_y);
    max_size = std::max(max_size, metal_info.max_work_group_size_z);
    return max_size;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage2DWidth() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_texture_size;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_2d;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image2d_max_width;
  }
  if (IsApiMetal()) {
    return metal_info.image2d_max_width;
  }
  return 2048;
}

uint64_t GpuInfo::GetMaxImage2DHeight() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_texture_size;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_2d;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image2d_max_height;
  }
  if (IsApiMetal()) {
    return metal_info.image2d_max_height;
  }
  return 2048;
}

uint64_t GpuInfo::GetMaxImage2DArrayLayers() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_array_texture_layers;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_image_array_layers;
  }
  if (IsApiOpenCl()) {
    return opencl_info.image_array_max_layers;
  }
  if (IsApiMetal()) {
    return metal_info.image_array_max_layers;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DWidth() const {
  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_width;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_width;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DHeight() const {
  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_height;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_height;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxImage3DDepth() const {
  if (IsApiOpenCl()) {
    return opencl_info.image3d_max_depth;
  } else if (IsApiMetal()) {
    return metal_info.image3d_max_depth;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_image_dimension_3d;
  }
  return 256;
}

uint64_t GpuInfo::GetMaxBufferSize() const {
  if (IsApiOpenCl()) {
    return opencl_info.buffer_max_size;
  } else if (IsApiMetal()) {
    return metal_info.buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_storage_buffer_range;
  }
  return 128 * 1024 * 1024;
}

uint64_t GpuInfo::GetMaxMemoryAllocationSize() const {
  if (IsApiOpenCl()) {
    return opencl_info.max_allocation_size;
  } else if (IsApiMetal()) {
    return metal_info.buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_storage_buffer_range;
  }
  return 128 * 1024 * 1024;
}

uint64_t GpuInfo::GetMaxImageBufferWidth() const {
  if (IsApiOpenCl()) {
    return opencl_info.image_buffer_max_size;
  } else if (IsApiVulkan()) {
    return vulkan_info.max_texel_buffer_elements;
  }
  return 64 * 1024;
}

int GpuInfo::GetMaxImageArguments() const {
  if (IsApiOpenGl()) {
    return opengl_info.max_image_units;
  }
  if (IsApiVulkan()) {
    return vulkan_info.max_per_stage_descriptor_sampled_images;
  }
  if (IsApiMetal()) {
    return 32;
  }
  if (IsApiOpenCl()) {
    return 128;
  }
  return 1;
}

bool GpuInfo::IsApiOpenGl() const { return gpu_api == GpuApi::kOpenGl; }

bool GpuInfo::IsApiOpenGl31OrAbove() const {
  if (!IsApiOpenGl()) {
    return false;
  }
  return opengl_info.IsApiOpenGl31OrAbove();
}

bool GpuInfo::IsApiVulkan() const { return gpu_api == GpuApi::kVulkan; }

bool GpuInfo::IsApiMetal() const { return gpu_api == GpuApi::kMetal; }

bool GpuInfo::IsApiOpenCl() const { return gpu_api == GpuApi::kOpenCl; }

bool GpuInfo::IsGlsl() const { return IsApiOpenGl() || IsApiVulkan(); }

bool GpuInfo::IsGlslSupportsExplicitFp16() const {
  if (IsApiOpenGl() && opengl_info.SupportsExplicitFp16()) {
    return true;
  }
  if (IsApiVulkan() && vulkan_info.SupportsExplicitFp16()) {
    return true;
  }
  return false;
}

bool GpuInfo::IsCL11OrHigher() const {
  if (!IsApiOpenCl()) {
    return false;
  }
  return opencl_info.cl_version != OpenClVersion::kCl1_0;
}

bool GpuInfo::IsCL20OrHigher() const {
  if (!IsApiOpenCl()) {
    return false;
  }
  return opencl_info.cl_version != OpenClVersion::kCl1_0 &&
         opencl_info.cl_version != OpenClVersion::kCl1_1 &&
         opencl_info.cl_version != OpenClVersion::kCl1_2;
}

bool GpuInfo::IsCL30OrHigher() const {
  if (!IsApiOpenCl()) {
    return false;
  }
  return IsCL20OrHigher() && opencl_info.cl_version != OpenClVersion::kCl2_0 &&
         opencl_info.cl_version != OpenClVersion::kCl2_1 &&
         opencl_info.cl_version != OpenClVersion::kCl2_2;
}

}  // namespace gpu
}  // namespace tflite
