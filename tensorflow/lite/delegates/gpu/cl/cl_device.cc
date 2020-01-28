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

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

template <>
std::string GetDeviceInfo<std::string>(cl_device_id id, cl_device_info info) {
  size_t size;
  cl_int error = clGetDeviceInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetDeviceInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

namespace {
template <typename T>
T GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
  T result;
  cl_int error = clGetPlatformInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS) {
    return -1;
  }
  return result;
}

std::string GetPlatformInfo(cl_platform_id id, cl_platform_info info) {
  size_t size;
  cl_int error = clGetPlatformInfo(id, info, 0, nullptr, &size);
  if (error != CL_SUCCESS) {
    return "";
  }

  std::string result(size - 1, 0);
  error = clGetPlatformInfo(id, info, size, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return "";
  }
  return result;
}

void GetDeviceWorkDimsSizes(cl_device_id id, int3* result) {
  int dims_count =
      GetDeviceInfo<cl_uint>(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
  if (dims_count < 3) {
    return;
  }
  std::vector<size_t> limits(dims_count);
  cl_int error =
      clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(size_t) * dims_count, limits.data(), nullptr);
  if (error != CL_SUCCESS) {
    return;
  }
  // dims_count must be at least 3 according to spec
  result->x = limits[0];
  result->y = limits[1];
  result->z = limits[2];
}

OpenCLVersion ParseCLVersion(const std::string& version) {
  const auto first_dot_pos = version.find_first_of('.');
  if (first_dot_pos == std::string::npos) {
    return OpenCLVersion::CL_1_0;
  }
  const int major = version[first_dot_pos - 1] - '0';
  const int minor = version[first_dot_pos + 1] - '0';

  if (major == 1) {
    if (minor == 2) {
      return OpenCLVersion::CL_1_2;
    } else if (minor == 1) {
      return OpenCLVersion::CL_1_1;
    } else {
      return OpenCLVersion::CL_1_0;
    }
  } else {
    return OpenCLVersion::CL_2_0;
  }
}

Vendor ParseVendor(const std::string& device_name,
                   const std::string& vendor_name) {
  std::string d_name = device_name;
  std::string v_name = vendor_name;
  std::transform(d_name.begin(), d_name.end(), d_name.begin(), ::tolower);
  std::transform(v_name.begin(), v_name.end(), v_name.begin(), ::tolower);
  if (d_name.find("qualcomm") != std::string::npos ||
      v_name.find("qualcomm") != std::string::npos) {
    return Vendor::QUALCOMM;
  } else if (d_name.find("mali") != std::string::npos ||
             v_name.find("mali") != std::string::npos) {
    return Vendor::MALI;
  } else if (d_name.find("power") != std::string::npos ||
             v_name.find("power") != std::string::npos) {
    return Vendor::POWERVR;
  } else if (d_name.find("nvidia") != std::string::npos ||
             v_name.find("nvidia") != std::string::npos) {
    return Vendor::NVIDIA;
  } else {
    return Vendor::UNKNOWN;
  }
}

// check that gpu_version belong to range min_version-max_version
// min_version is included and max_version is excluded.
bool IsGPUVersionInRange(int gpu_version, int min_version, int max_version) {
  return gpu_version >= min_version && gpu_version < max_version;
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
    case Vendor::QUALCOMM:
      return "Qualcomm";
    case Vendor::MALI:
      return "Mali";
    case Vendor::POWERVR:
      return "PowerVR";
    case Vendor::NVIDIA:
      return "NVIDIA";
    case Vendor::UNKNOWN:
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

DeviceInfo::DeviceInfo(cl_device_id id)
    : adreno_info(GetDeviceInfo<std::string>(id, CL_DEVICE_OPENCL_C_VERSION)) {
  const auto device_name = GetDeviceInfo<std::string>(id, CL_DEVICE_NAME);
  const auto vendor_name = GetDeviceInfo<std::string>(id, CL_DEVICE_VENDOR);
  vendor = ParseVendor(device_name, vendor_name);
  cl_version = ParseCLVersion(
      GetDeviceInfo<std::string>(id, CL_DEVICE_OPENCL_C_VERSION));
  extensions =
      absl::StrSplit(GetDeviceInfo<std::string>(id, CL_DEVICE_EXTENSIONS), ' ');
  supports_fp16 = false;
  supports_image3d_writes = false;
  for (const auto& ext : extensions) {
    if (ext == "cl_khr_fp16") {
      supports_fp16 = true;
    }
    if (ext == "cl_khr_3d_image_writes") {
      supports_image3d_writes = true;
    }
  }

  f32_config =
      GetDeviceInfo<cl_device_fp_config>(id, CL_DEVICE_SINGLE_FP_CONFIG);
  supports_fp32_rtn = f32_config & CL_FP_ROUND_TO_NEAREST;

  if (supports_fp16) {
    auto status = GetDeviceInfo<cl_device_fp_config>(
        id, CL_DEVICE_HALF_FP_CONFIG, &f16_config);
    if (status.ok()) {
      supports_fp16_rtn = f16_config & CL_FP_ROUND_TO_NEAREST;
    } else {  // happens on PowerVR
      f16_config = f32_config;
      supports_fp16_rtn = supports_fp32_rtn;
    }
  } else {
    f16_config = 0;
    supports_fp16_rtn = false;
  }

  if (vendor == Vendor::POWERVR && !supports_fp16) {
    // PowerVR doesn't have full support of fp16 and so doesn't list this
    // extension. But it can support fp16 in MADs and as buffers/textures types,
    // so we will use it.
    supports_fp16 = true;
    f16_config = f32_config;
    supports_fp16_rtn = supports_fp32_rtn;
  }

  if (!supports_image3d_writes &&
      ((vendor == Vendor::QUALCOMM &&
        IsGPUVersionInRange(adreno_info.gpu_version, 400, 500)) ||
       vendor == Vendor::NVIDIA)) {
    // in local tests Adreno 430 can write in image 3d, at least on small sizes,
    // but it doesn't have cl_khr_3d_image_writes in list of available
    // extensions
    // The same for NVidia
    supports_image3d_writes = true;
  }
  compute_units_count = GetDeviceInfo<cl_uint>(id, CL_DEVICE_MAX_COMPUTE_UNITS);
  image2d_max_width = GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_WIDTH);
  image2d_max_height = GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  buffer_max_size = GetDeviceInfo<cl_ulong>(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
  if (cl_version >= OpenCLVersion::CL_1_2) {
    image_buffer_max_size =
        GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
    image_array_max_layers =
        GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE);
  }
  image3d_max_width = GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE3D_MAX_WIDTH);
  image3d_max_height = GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
  image3d_max_depth = GetDeviceInfo<size_t>(id, CL_DEVICE_IMAGE3D_MAX_DEPTH);
  GetDeviceWorkDimsSizes(id, &max_work_group_sizes);
}

bool DeviceInfo::SupportsTextureArray() const {
  return cl_version >= OpenCLVersion::CL_1_2;
}

bool DeviceInfo::SupportsImageBuffer() const {
  return cl_version >= OpenCLVersion::CL_1_2;
}

bool DeviceInfo::SupportsImage3D() const {
  if (vendor == Vendor::MALI) {
    // On Mali T880 read_imageh doesn't compile with image3d_t
    return false;
  }
  return supports_image3d_writes;
}

CLDevice::CLDevice(cl_device_id id, cl_platform_id platform_id)
    : id_(id), platform_id_(platform_id), info_(id) {}

CLDevice::CLDevice(const CLDevice& device)
    : id_(device.id_), platform_id_(device.platform_id_), info_(device.info_) {}

CLDevice& CLDevice::operator=(const CLDevice& device) {
  if (this != &device) {
    id_ = device.id_;
    platform_id_ = device.platform_id_;
    info_ = device.info_;
  }
  return *this;
}

CLDevice::CLDevice(CLDevice&& device)
    : id_(device.id_),
      platform_id_(device.platform_id_),
      info_(std::move(device.info_)) {
  device.id_ = nullptr;
  device.platform_id_ = nullptr;
}

CLDevice& CLDevice::operator=(CLDevice&& device) {
  if (this != &device) {
    id_ = nullptr;
    platform_id_ = nullptr;
    std::swap(id_, device.id_);
    std::swap(platform_id_, device.platform_id_);
    info_ = std::move(device.info_);
  }
  return *this;
}

bool CLDevice::SupportsFP16() const { return info_.supports_fp16; }

bool CLDevice::SupportsExtension(const std::string& extension) const {
  for (const auto& ext : info_.extensions) {
    if (ext == extension) {
      return true;
    }
  }
  return false;
}

bool CLDevice::SupportsTextureArray() const {
  return info_.SupportsTextureArray();
}

bool CLDevice::SupportsImageBuffer() const {
  return info_.SupportsImageBuffer();
}

bool CLDevice::SupportsImage3D() const { return info_.SupportsImage3D(); }

bool CLDevice::SupportsFP32RTN() const { return info_.supports_fp32_rtn; }

bool CLDevice::SupportsFP16RTN() const { return info_.supports_fp16_rtn; }

std::string CLDevice::GetPlatformVersion() const {
  return GetPlatformInfo(platform_id_, CL_PLATFORM_VERSION);
}

bool CLDevice::IsAdreno() const { return info_.vendor == Vendor::QUALCOMM; }

bool CLDevice::IsAdreno3xx() const {
  return IsAdreno() &&
         IsGPUVersionInRange(info_.adreno_info.gpu_version, 300, 400);
}

bool CLDevice::IsAdreno4xx() const {
  return IsAdreno() &&
         IsGPUVersionInRange(info_.adreno_info.gpu_version, 400, 500);
}

bool CLDevice::IsAdreno5xx() const {
  return IsAdreno() &&
         IsGPUVersionInRange(info_.adreno_info.gpu_version, 500, 600);
}

bool CLDevice::IsAdreno6xx() const {
  return IsAdreno() &&
         IsGPUVersionInRange(info_.adreno_info.gpu_version, 600, 700);
}

bool CLDevice::IsAdreno6xxOrHigher() const {
  return IsAdreno() && info_.adreno_info.gpu_version >= 600;
}

bool CLDevice::IsPowerVR() const { return info_.vendor == Vendor::POWERVR; }

bool CLDevice::IsNvidia() const { return info_.vendor == Vendor::NVIDIA; }

bool CLDevice::IsMali() const { return info_.vendor == Vendor::MALI; }

bool CLDevice::SupportsOneLayerTextureArray() const {
  return !IsAdreno() || info_.adreno_info.support_one_layer_texture_array;
}

void CLDevice::DisableOneLayerTextureArray() {
  info_.adreno_info.support_one_layer_texture_array = false;
}

Status CreateDefaultGPUDevice(CLDevice* result) {
  cl_uint num_platforms;
  clGetPlatformIDs(0, nullptr, &num_platforms);
  if (num_platforms == 0) {
    return UnknownError("No supported OpenCL platform.");
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  cl_uint num_devices;
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (num_devices == 0) {
    return UnknownError("No GPU on current platform.");
  }

  std::vector<cl_device_id> devices(num_devices);
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices.data(),
                 nullptr);

  *result = CLDevice(devices[0], platforms[0]);
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
