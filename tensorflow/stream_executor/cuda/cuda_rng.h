/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_RNG_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_RNG_H_

#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rng.h"

typedef struct curandGenerator_st *curandGenerator_t;

namespace perftools {
namespace gputools {

class Stream;
template <typename ElemT>
class DeviceMemory;

namespace cuda {

// Opaque and unique identifier for the cuRAND plugin.
extern const PluginId kCuRandPlugin;

class CUDAExecutor;

// CUDA-platform implementation of the random number generation support
// interface.
//
// Thread-safe post-initialization.
class CUDARng : public rng::RngSupport {
 public:
  explicit CUDARng(CUDAExecutor *parent);

  // Retrieves a curand library generator handle. This is necessary for
  // enqueuing random number generation work onto the device.
  // TODO(leary) provide a way for users to select the RNG algorithm.
  bool Init();

  // Releases a curand library generator handle, if one was acquired.
  ~CUDARng() override;

  // See rng::RngSupport for details on the following overrides.
  bool DoPopulateRandUniform(Stream *stream, DeviceMemory<float> *v) override;
  bool DoPopulateRandUniform(Stream *stream, DeviceMemory<double> *v) override;
  bool DoPopulateRandUniform(Stream *stream,
                             DeviceMemory<std::complex<float>> *v) override;
  bool DoPopulateRandUniform(Stream *stream,
                             DeviceMemory<std::complex<double>> *v) override;
  bool DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                              DeviceMemory<float> *v) override;
  bool DoPopulateRandGaussian(Stream *stream, double mean, double stddev,
                              DeviceMemory<double> *v) override;

  bool SetSeed(Stream *stream, const uint8 *seed, uint64 seed_bytes) override;

 private:
  // Actually performs the work of generating random numbers - the public
  // methods are thin wrappers to this interface.
  template <typename T>
  bool DoPopulateRandUniformInternal(Stream *stream, DeviceMemory<T> *v);
  template <typename ElemT, typename FuncT>
  bool DoPopulateRandGaussianInternal(Stream *stream, ElemT mean, ElemT stddev,
                                      DeviceMemory<ElemT> *v, FuncT func);

  // Sets the stream for the internal curand generator.
  //
  // This is a stateful operation, as the handle can only have one stream set at
  // a given time, so it is usually performed right before enqueuing work to do
  // with random number generation.
  bool SetStream(Stream *stream) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // mutex that guards the cuRAND handle for this device.
  mutex mu_;

  // CUDAExecutor which instantiated this CUDARng.
  // Immutable post-initialization.
  CUDAExecutor *parent_;

  // cuRANDalibrary handle on the device.
  curandGenerator_t rng_ GUARDED_BY(mu_);

  SE_DISALLOW_COPY_AND_ASSIGN(CUDARng);
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_RNG_H_
