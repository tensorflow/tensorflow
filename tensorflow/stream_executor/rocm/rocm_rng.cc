/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "rocm/include/hiprand/hiprand.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/stream_executor/gpu/gpu_rng.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"

// Formats hiprandStatus_t to output prettified values into a log stream.
std::ostream& operator<<(std::ostream& in, const hiprandStatus_t& status) {
#define OSTREAM_HIPRAND_STATUS(__name) \
  case HIPRAND_STATUS_##__name:        \
    in << "HIPRAND_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_HIPRAND_STATUS(SUCCESS)
    OSTREAM_HIPRAND_STATUS(VERSION_MISMATCH)
    OSTREAM_HIPRAND_STATUS(NOT_INITIALIZED)
    OSTREAM_HIPRAND_STATUS(ALLOCATION_FAILED)
    OSTREAM_HIPRAND_STATUS(TYPE_ERROR)
    OSTREAM_HIPRAND_STATUS(OUT_OF_RANGE)
    OSTREAM_HIPRAND_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_HIPRAND_STATUS(LAUNCH_FAILURE)
    OSTREAM_HIPRAND_STATUS(PREEXISTING_FAILURE)
    OSTREAM_HIPRAND_STATUS(INITIALIZATION_FAILED)
    OSTREAM_HIPRAND_STATUS(ARCH_MISMATCH)
    OSTREAM_HIPRAND_STATUS(INTERNAL_ERROR)
    default:
      in << "hiprandStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kGpuRandPlugin);

namespace wrap {

#define PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(__name)                     \
  struct WrapperShim__##__name {                                    \
    template <typename... Args>                                     \
    hiprandStatus_t operator()(GpuExecutor* parent, Args... args) { \
      gpu::ScopedActivateExecutorContext sac{parent};               \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;

PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandCreateGenerator);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandDestroyGenerator);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandSetStream);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandGenerateUniform);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandGenerateUniformDouble);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandSetPseudoRandomGeneratorSeed);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandSetGeneratorOffset);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandGenerateNormal);
PERFTOOLS_GPUTOOLS_HIPRAND_WRAP(hiprandGenerateNormalDouble);

}  // namespace wrap

GpuRng::GpuRng(GpuExecutor* parent) : parent_(parent), rng_(nullptr) {}

GpuRng::~GpuRng() {
  if (rng_ != nullptr) {
    wrap::hiprandDestroyGenerator(parent_, rng_);
  }
}

bool GpuRng::Init() {
  mutex_lock lock{mu_};
  CHECK(rng_ == nullptr);

  hiprandStatus_t ret =
      wrap::hiprandCreateGenerator(parent_, &rng_, HIPRAND_RNG_PSEUDO_DEFAULT);
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool GpuRng::SetStream(Stream* stream) {
  hiprandStatus_t ret =
      wrap::hiprandSetStream(parent_, rng_, AsGpuStreamValue(stream));
  if (ret != HIPRAND_STATUS_SUCCESS) {
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
  mutex_lock lock{mu_};
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

  hiprandStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = wrap::hiprandGenerateUniform(
        parent_, rng_, reinterpret_cast<float*>(GpuMemoryMutable(v)),
        element_count);
  } else {
    ret = wrap::hiprandGenerateUniformDouble(
        parent_, rng_, reinterpret_cast<double*>(GpuMemoryMutable(v)),
        element_count);
  }
  if (ret != HIPRAND_STATUS_SUCCESS) {
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
  mutex_lock lock{mu_};

  if (!SetStream(stream)) {
    return false;
  }

  uint64 element_count = v->ElementCount();
  hiprandStatus_t ret =
      func(parent_, rng_, GpuMemoryMutable(v), element_count, mean, stddev);

  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                                    DeviceMemory<float>* v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        wrap::hiprandGenerateNormal);
}

bool GpuRng::DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                                    DeviceMemory<double>* v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        wrap::hiprandGenerateNormalDouble);
}

bool GpuRng::SetSeed(Stream* stream, const uint8* seed, uint64 seed_bytes) {
  mutex_lock lock{mu_};
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  hiprandStatus_t ret = wrap::hiprandSetPseudoRandomGeneratorSeed(
      parent_, rng_, *(reinterpret_cast<const uint64*>(seed)));
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = wrap::hiprandSetGeneratorOffset(parent_, rng_, 0);
  if (ret != HIPRAND_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace gpu
}  // namespace stream_executor

namespace se = ::stream_executor;

REGISTER_MODULE_INITIALIZER(register_hiprand, {
  se::port::Status status =
      se::PluginRegistry::Instance()
          ->RegisterFactory<se::PluginRegistry::RngFactory>(
              se::gpu::kROCmPlatformId, se::gpu::kGpuRandPlugin, "hipRAND",
              [](se::internal::StreamExecutorInterface* parent)
                  -> se::rng::RngSupport* {
                se::gpu::GpuExecutor* rocm_executor =
                    dynamic_cast<se::gpu::GpuExecutor*>(parent);
                if (rocm_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the hipRAND "
                      << "support library with a non-ROCM StreamExecutor";
                  return nullptr;
                }

                se::gpu::GpuRng* rng = new se::gpu::GpuRng(rocm_executor);
                if (!rng->Init()) {
                  // Note: Init() will log a more specific error.
                  delete rng;
                  return nullptr;
                }
                return rng;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register hipRAND factory: "
               << status.error_message();
  }

  se::PluginRegistry::Instance()->SetDefaultFactory(
      se::gpu::kROCmPlatformId, se::PluginKind::kRng, se::gpu::kGpuRandPlugin);
});
