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

#include "tensorflow/lite/delegates/gpu/cl/environment.h"

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
absl::Status CreateEnvironment(Environment* result, bool shared,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display) {
  CLDevice gpu;
  RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContext context;
  if (shared) {
    RETURN_IF_ERROR(CreateCLGLContext(gpu, egl_context, egl_display, &context));
  } else {
    RETURN_IF_ERROR(CreateCLContext(gpu, &context));
  }
  CLCommandQueue queue;
  RETURN_IF_ERROR(CreateCLCommandQueue(gpu, context, &queue));
  ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(gpu, context, &profiling_queue));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));

  return result->Init();
}

bool IsGpuSupportsStorageType(const GpuInfo& gpu_info,
                              TensorStorageType storage_type) {
  switch (storage_type) {
    case TensorStorageType::TEXTURE_2D:
      return !gpu_info.IsAMD();
    case TensorStorageType::BUFFER:
      return true;
    case TensorStorageType::TEXTURE_ARRAY:
      return !gpu_info.IsAMD() && gpu_info.SupportsTextureArray();
    case TensorStorageType::IMAGE_BUFFER:
      return (gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsNvidia()) &&
             gpu_info.SupportsImageBuffer();
    case TensorStorageType::TEXTURE_3D:
      return !gpu_info.IsAMD() && gpu_info.SupportsImage3D();
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return false;
    case TensorStorageType::UNKNOWN:
      return false;
  }
  return false;
}

bool IsGpuSupportsPrecision(const GpuInfo& gpu_info,
                            CalculationsPrecision precision) {
  switch (precision) {
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      return gpu_info.SupportsFP16();
    case CalculationsPrecision::F32:
      return true;
  }
}

}  // namespace

Environment::Environment(CLDevice&& device, CLContext&& context,
                         CLCommandQueue&& queue,
                         ProfilingCommandQueue&& profiling_queue)
    : device_(std::move(device)),
      context_(std::move(context)),
      queue_(std::move(queue)),
      profiling_queue_(std::move(profiling_queue)) {}

Environment::Environment(Environment&& environment)
    : device_(std::move(environment.device_)),
      context_(std::move(environment.context_)),
      queue_(std::move(environment.queue_)),
      profiling_queue_(std::move(environment.profiling_queue_)),
      program_cache_(std::move(environment.program_cache_)) {}

Environment& Environment::operator=(Environment&& environment) {
  if (this != &environment) {
    device_ = std::move(environment.device_);
    context_ = std::move(environment.context_);
    queue_ = std::move(environment.queue_);
    profiling_queue_ = std::move(environment.profiling_queue_);
    program_cache_ = std::move(environment.program_cache_);
  }
  return *this;
}

absl::Status Environment::Init() {
  if (device().GetInfo().IsAdreno() &&
      device().GetInfo().SupportsTextureArray()) {
    const auto& adreno_info = device().info_.adreno_info;
    // Some Adreno < 600 have bug with one layer texture array. b/131099086
    // If we have one layer texture array and will write smt from kernel to this
    // texture, we will get zeroes instead of actual values.
    // The same kernel will work, if we use texture array with more than one
    // layer.
    if (adreno_info.IsAdreno3xx() || adreno_info.IsAdreno4xx() ||
        adreno_info.IsAdreno5xx()) {
      GetDevicePtr()->DisableOneLayerTextureArray();
    }
  }
  return absl::OkStatus();
}

void Environment::SetHighPerformance() const {
  // TODO(sorokin) use cl_perf_hint if available
}

void Environment::SetDefaultPerformance() const {
  // TODO(sorokin) use cl_perf_hint if available
}

void Environment::SetLowPerformance() const {
  // TODO(sorokin) use cl_perf_hint if available
}

std::vector<CalculationsPrecision> Environment::GetSupportedPrecisions() const {
  std::vector<CalculationsPrecision> precisions;
  for (CalculationsPrecision precision :
       {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
        CalculationsPrecision::F16}) {
    if (IsSupported(precision)) {
      precisions.push_back(precision);
    }
  }
  return precisions;
}

bool Environment::IsSupported(CalculationsPrecision precision) const {
  return IsGpuSupportsPrecision(device_.GetInfo(), precision);
}

std::vector<TensorStorageType> Environment::GetSupportedStorages() const {
  std::vector<TensorStorageType> storage_types;
  for (auto storage_type :
       {TensorStorageType::TEXTURE_2D, TensorStorageType::BUFFER,
        TensorStorageType::TEXTURE_ARRAY, TensorStorageType::IMAGE_BUFFER,
        TensorStorageType::TEXTURE_3D}) {
    if (IsSupported(storage_type)) {
      storage_types.push_back(storage_type);
    }
  }
  return storage_types;
}

std::vector<TensorStorageType>
Environment::GetSupportedStoragesWithHWZeroClampSupport() const {
  std::vector<TensorStorageType> storage_types;
  for (auto storage_type :
       {TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_ARRAY,
        TensorStorageType::TEXTURE_3D}) {
    if (IsSupported(storage_type)) {
      storage_types.push_back(storage_type);
    }
  }
  return storage_types;
}

bool Environment::IsSupported(TensorStorageType storage_type) const {
  return IsGpuSupportsStorageType(device_.GetInfo(), storage_type);
}

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno6xxOrHigher() &&
        !gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
      return TensorStorageType::TEXTURE_ARRAY;
    } else {
      return TensorStorageType::TEXTURE_2D;
    }
  } else if (gpu_info.IsPowerVR()) {
    return TensorStorageType::TEXTURE_2D;
  } else if (gpu_info.IsMali()) {
    const MaliInfo mali_info = gpu_info.mali_info;
    if (mali_info.IsMaliT8xx() || mali_info.IsBifrostGen3() ||
        mali_info.IsValhall()) {
      return TensorStorageType::TEXTURE_2D;
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::BUFFER;
  }
  return TensorStorageType::BUFFER;
}

TensorStorageType GetStorageTypeWithMinimalMemoryConsumption(
    const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx() ||
        gpu_info.adreno_info.IsAdreno4xx()) {
      return TensorStorageType::BUFFER;
    } else {
      if (gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
        return TensorStorageType::TEXTURE_2D;
      } else {
        return TensorStorageType::IMAGE_BUFFER;
      }
    }
  } else if (gpu_info.IsPowerVR()) {
    return TensorStorageType::BUFFER;
  } else if (gpu_info.IsMali()) {
    const MaliInfo mali_info = gpu_info.mali_info;
    if (mali_info.IsMaliT8xx() || mali_info.IsBifrostGen3() ||
        mali_info.IsValhall()) {
      if (gpu_info.opencl_info.IsImage2dFromBufferSupported()) {
        return TensorStorageType::TEXTURE_2D;
      } else {
        return TensorStorageType::BUFFER;
      }
    } else {
      return TensorStorageType::BUFFER;
    }
  } else if (gpu_info.IsNvidia()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsAMD()) {
    return gpu_info.SupportsImageBuffer() ? TensorStorageType::IMAGE_BUFFER
                                          : TensorStorageType::BUFFER;
  } else if (gpu_info.IsIntel()) {
    return TensorStorageType::BUFFER;
  }
  return TensorStorageType::BUFFER;
}

absl::Status CreateEnvironment(Environment* result) {
  CLDevice gpu;
  RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContext context;
  RETURN_IF_ERROR(CreateCLContext(gpu, &context));
  CLCommandQueue queue;
  RETURN_IF_ERROR(CreateCLCommandQueue(gpu, context, &queue));
  ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(gpu, context, &profiling_queue));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));
  return result->Init();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
