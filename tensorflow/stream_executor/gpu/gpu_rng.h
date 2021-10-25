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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_RNG_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_RNG_H_

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rng.h"

namespace stream_executor {

class Stream;
template <typename ElemT>
class DeviceMemory;

namespace gpu {

// Opaque and unique identifier for the GPU RNG plugin.
extern const PluginId kGpuRandPlugin;

class GpuExecutor;

// GPU-platform implementation of the random number generation support
// interface.
//
// Thread-safe post-initialization.
class GpuRng : public rng::RngSupport {
 public:
  explicit GpuRng(GpuExecutor* parent);

  // Retrieves a gpu rng library generator handle. This is necessary for
  // enqueuing random number generation work onto the device.
  // TODO(leary) provide a way for users to select the RNG algorithm.
  bool Init();

  // Releases a gpu rng library generator handle, if one was acquired.
  ~GpuRng() override;

  // See rng::RngSupport for details on the following overrides.
  bool DoPopulateRandUniform(Stream* stream, DeviceMemory<float>* v) override;
  bool DoPopulateRandUniform(Stream* stream, DeviceMemory<double>* v) override;
  bool DoPopulateRandUniform(Stream* stream,
                             DeviceMemory<std::complex<float>>* v) override;
  bool DoPopulateRandUniform(Stream* stream,
                             DeviceMemory<std::complex<double>>* v) override;
  bool DoPopulateRandGaussian(Stream* stream, float mean, float stddev,
                              DeviceMemory<float>* v) override;
  bool DoPopulateRandGaussian(Stream* stream, double mean, double stddev,
                              DeviceMemory<double>* v) override;

  bool SetSeed(Stream* stream, const uint8* seed, uint64_t seed_bytes) override;

 private:
  // Actually performs the work of generating random numbers - the public
  // methods are thin wrappers to this interface.
  template <typename T>
  bool DoPopulateRandUniformInternal(Stream* stream, DeviceMemory<T>* v);
  template <typename ElemT, typename FuncT>
  bool DoPopulateRandGaussianInternal(Stream* stream, ElemT mean, ElemT stddev,
                                      DeviceMemory<ElemT>* v, FuncT func);

  // Sets the stream for the internal gpu rng generator.
  //
  // This is a stateful operation, as the handle can only have one stream set at
  // a given time, so it is usually performed right before enqueuing work to do
  // with random number generation.
  bool SetStream(Stream* stream) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Guards the gpu rng library handle for this device.
  absl::Mutex mu_;

  // GpuExecutor which instantiated this GpuRng.
  // Immutable post-initialization.
  GpuExecutor* parent_;

  // gpu rng library handle on the device.
  GpuRngHandle rng_ TF_GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(GpuRng);
};

template <typename T>
std::string TypeString();

template <>
std::string TypeString<float>() {
  return "float";
}

template <>
std::string TypeString<double>() {
  return "double";
}

template <>
std::string TypeString<std::complex<float>>() {
  return "std::complex<float>";
}

template <>
std::string TypeString<std::complex<double>>() {
  return "std::complex<double>";
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_RNG_H_
