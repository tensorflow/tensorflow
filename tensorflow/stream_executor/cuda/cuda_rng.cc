/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cuda_rng.h"

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"
// clang-format off
#include "cuda/include/curand.h"
// clang-format on

// Formats curandStatus_t to output prettified values into a log stream.
std::ostream &operator<<(std::ostream &in, const curandStatus_t &status) {
#define OSTREAM_CURAND_STATUS(__name) \
  case CURAND_STATUS_##__name:        \
    in << "CURAND_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_CURAND_STATUS(SUCCESS)
    OSTREAM_CURAND_STATUS(VERSION_MISMATCH)
    OSTREAM_CURAND_STATUS(NOT_INITIALIZED)
    OSTREAM_CURAND_STATUS(ALLOCATION_FAILED)
    OSTREAM_CURAND_STATUS(TYPE_ERROR)
    OSTREAM_CURAND_STATUS(OUT_OF_RANGE)
    OSTREAM_CURAND_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_CURAND_STATUS(LAUNCH_FAILURE)
    OSTREAM_CURAND_STATUS(PREEXISTING_FAILURE)
    OSTREAM_CURAND_STATUS(INITIALIZATION_FAILED)
    OSTREAM_CURAND_STATUS(ARCH_MISMATCH)
    OSTREAM_CURAND_STATUS(INTERNAL_ERROR)
    default:
      in << "curandStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kGpuRandPlugin);

GpuRng::GpuRng(GpuExecutor* parent) : parent_(parent), rng_(nullptr) {}

GpuRng::~GpuRng() {
  if (rng_ != nullptr) {
    cuda::ScopedActivateExecutorContext sac(parent_);
    curandDestroyGenerator(rng_);
  }
}

bool GpuRng::Init() {
  mutex_lock lock(mu_);
  CHECK(rng_ == nullptr);

  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret = curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT);
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool GpuRng::SetStream(Stream* stream) {
  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret = curandSetStream(rng_, AsGpuStreamValue(stream));
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for random generation: " << ret;
    return false;
  }

  return true;
}

// Returns true if std::complex stores its contents as two consecutive
// elements. Tests int, float and double, as the last two are independent
// specializations.
constexpr bool ComplexIsConsecutiveFloats() {
  return sizeof(std::complex<int>) == 8 && sizeof(std::complex<float>) == 8 &&
      sizeof(std::complex<double>) == 16;
}

template <typename T>
bool GpuRng::DoPopulateRandUniformInternal(Stream* stream, DeviceMemory<T>* v) {
  mutex_lock lock(mu_);
  static_assert(ComplexIsConsecutiveFloats(),
                "std::complex values are not stored as consecutive values");

  if (!SetStream(stream)) {
    return false;
  }

  // std::complex<T> is currently implemented as two consecutive T variables.
  uint64 element_count = v->ElementCount();
  if (std::is_same<T, std::complex<float>>::value ||
      std::is_same<T, std::complex<double>>::value) {
    element_count *= 2;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  curandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = curandGenerateUniform(
        rng_, reinterpret_cast<float*>(GpuMemoryMutable(v)), element_count);
  } else {
    ret = curandGenerateUniformDouble(
        rng_, reinterpret_cast<double*>(GpuMemoryMutable(v)), element_count);
  }
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<float>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream, DeviceMemory<double>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<float>>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool GpuRng::DoPopulateRandUniform(Stream* stream,
                                   DeviceMemory<std::complex<double>>* v) {
  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
bool GpuRng::DoPopulateRandGaussianInternal(Stream* stream, ElemT mean,
                                            ElemT stddev,
                                            DeviceMemory<ElemT>* v,
                                            FuncT func) {
  mutex_lock lock(mu_);

  if (!SetStream(stream)) {
    return false;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  uint64 element_count = v->ElementCount();
  curandStatus_t ret =
      func(rng_, GpuMemoryMutable(v), element_count, mean, stddev);

  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                                    DeviceMemory<float>* v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        curandGenerateNormal);
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                                    DeviceMemory<double>* v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        curandGenerateNormalDouble);
}

bool GpuRng::SetSeed(Stream* stream, const uint8* seed, uint64 seed_bytes) {
  mutex_lock lock(mu_);
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  curandStatus_t ret = curandSetPseudoRandomGeneratorSeed(
      rng_, *(reinterpret_cast<const uint64*>(seed)));
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = curandSetGeneratorOffset(rng_, 0);
  if (ret != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace gpu

void initialize_curand() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::RngFactory>(
          cuda::kCudaPlatformId, gpu::kGpuRandPlugin, "cuRAND",
          [](internal::StreamExecutorInterface* parent) -> rng::RngSupport* {
            gpu::GpuExecutor* cuda_executor =
                dynamic_cast<gpu::GpuExecutor*>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuRAND "
                  << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::GpuRng* rng = new gpu::GpuRng(cuda_executor);
            if (!rng->Init()) {
              // Note: Init() will log a more specific error.
              delete rng;
              return nullptr;
            }
            return rng;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuRAND factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kRng, gpu::kGpuRandPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_curand,
                            { stream_executor::initialize_curand(); });
