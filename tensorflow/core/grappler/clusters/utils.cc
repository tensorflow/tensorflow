/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/clusters/utils.h"

#include "Eigen/Core"  // from @eigen_archive

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cudnn/cudnn.h"
#endif

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace grappler {

DeviceProperties GetLocalCPUInfo() {
  DeviceProperties device;
  device.set_type("CPU");

  device.set_vendor(port::CPUVendorIDString());
  // Combine cpu family and model into the model string.
  device.set_model(
      strings::StrCat((port::CPUFamily() << 4) + port::CPUModelNum()));
  device.set_frequency(port::NominalCPUFrequency() * 1e-6);
  device.set_num_cores(port::NumSchedulableCPUs());
  device.set_l1_cache_size(Eigen::l1CacheSize());
  device.set_l2_cache_size(Eigen::l2CacheSize());
  device.set_l3_cache_size(Eigen::l3CacheSize());

  int64_t free_mem = port::AvailableRam();
  if (free_mem < INT64_MAX) {
    device.set_memory_size(free_mem);
  }

  (*device.mutable_environment())["cpu_instruction_set"] =
      Eigen::SimdInstructionSetsInUse();

  (*device.mutable_environment())["eigen"] = strings::StrCat(
      EIGEN_WORLD_VERSION, ".", EIGEN_MAJOR_VERSION, ".", EIGEN_MINOR_VERSION);

  return device;
}

DeviceProperties GetLocalGPUInfo(PlatformDeviceId platform_device_id) {
  DeviceProperties device;
  device.set_type("GPU");

#if GOOGLE_CUDA
  cudaDeviceProp properties;
  cudaError_t error =
      cudaGetDeviceProperties(&properties, platform_device_id.value());
  if (error != cudaSuccess) {
    device.set_type("UNKNOWN");
    LOG(ERROR) << "Failed to get device properties, error code: " << error;
    return device;
  }

  device.set_vendor("NVIDIA");
  device.set_model(properties.name);
  device.set_frequency(properties.clockRate * 1e-3);
  device.set_num_cores(properties.multiProcessorCount);
  device.set_num_registers(properties.regsPerMultiprocessor);
  // For compute capability less than 5, l1 cache size is configurable to
  // either 16 KB or 48 KB. We use the initial configuration 16 KB here. For
  // compute capability larger or equal to 5, l1 cache (unified with texture
  // cache) size is 24 KB. This number may need to be updated for future
  // compute capabilities.
  device.set_l1_cache_size((properties.major < 5) ? 16 * 1024 : 24 * 1024);
  device.set_l2_cache_size(properties.l2CacheSize);
  device.set_l3_cache_size(0);
  device.set_shared_memory_size_per_multiprocessor(
      properties.sharedMemPerMultiprocessor);
  device.set_memory_size(properties.totalGlobalMem);
  // 8 is the number of bits per byte. 2 is accounted for
  // double data rate (DDR).
  device.set_bandwidth(properties.memoryBusWidth / 8 *
                       properties.memoryClockRate * 2);

  (*device.mutable_environment())["architecture"] =
      strings::StrCat(properties.major, ".", properties.minor);
  (*device.mutable_environment())["cuda"] = strings::StrCat(CUDA_VERSION);
  (*device.mutable_environment())["cudnn"] = strings::StrCat(CUDNN_VERSION);

#elif TENSORFLOW_USE_ROCM
  hipDeviceProp_t properties;
  hipError_t error =
      hipGetDeviceProperties(&properties, platform_device_id.value());
  if (error != hipSuccess) {
    device.set_type("UNKNOWN");
    LOG(ERROR) << "Failed to get device properties, error code: " << error;
    return device;
  }

  // ROCM TODO review if numbers here are valid
  device.set_vendor("Advanced Micro Devices, Inc");
  device.set_model(properties.name);
  device.set_frequency(properties.clockRate * 1e-3);
  device.set_num_cores(properties.multiProcessorCount);
  device.set_num_registers(properties.regsPerBlock);
  device.set_l1_cache_size(16 * 1024);
  device.set_l2_cache_size(properties.l2CacheSize);
  device.set_l3_cache_size(0);
  device.set_shared_memory_size_per_multiprocessor(
      properties.maxSharedMemoryPerMultiProcessor);
  device.set_memory_size(properties.totalGlobalMem);
  // 8 is the number of bits per byte. 2 is accounted for
  // double data rate (DDR).
  device.set_bandwidth(properties.memoryBusWidth / 8 *
                       properties.memoryClockRate * 2);

  (*device.mutable_environment())["architecture"] =
      strings::StrCat("gfx", properties.gcnArchName);
#endif

  return device;
}

DeviceProperties GetDeviceInfo(const DeviceNameUtils::ParsedName& device) {
  DeviceProperties unknown;
  unknown.set_type("UNKNOWN");

  if (device.type == "CPU") {
    return GetLocalCPUInfo();
  } else if (device.type == "GPU") {
    if (device.has_id) {
      TfDeviceId tf_device_id(device.id);
      PlatformDeviceId platform_device_id;
      Status s =
          GpuIdManager::TfToPlatformDeviceId(tf_device_id, &platform_device_id);
      if (!s.ok()) {
        LOG(ERROR) << s;
        return unknown;
      }
      return GetLocalGPUInfo(platform_device_id);
    } else {
      return GetLocalGPUInfo(PlatformDeviceId(0));
    }
  }
  return unknown;
}

}  // end namespace grappler
}  // end namespace tensorflow
