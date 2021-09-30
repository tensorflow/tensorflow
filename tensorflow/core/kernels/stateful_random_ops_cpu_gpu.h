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

#ifndef TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_
#define TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_

#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/stateful_random_ops.h"

namespace tensorflow {

PHILOX_DEVICE_INLINE PhiloxRandom
GetPhiloxRandomFromMem(StateElementType const* ptr) {
  auto ptr_ = reinterpret_cast<uint64 const*>(ptr);
  return GetPhiloxRandomFromCounterKeyMem(ptr_, ptr_ + 2);
}

PHILOX_DEVICE_INLINE void WritePhiloxRandomToMem(PhiloxRandom const& philox,
                                                 StateElementType* ptr) {
  auto ptr_ = reinterpret_cast<uint64*>(ptr);
  WriteCounterToMem(philox.counter(), ptr_);
  WriteKeyToMem(philox.key(), ptr_ + 2);
}

PHILOX_DEVICE_INLINE PhiloxRandom SkipPhiloxRandom(PhiloxRandom const& philox,
                                                   uint64 output_size) {
  auto new_philox = philox;
  // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change it
  // just here.
  auto delta = output_size * 256;
  new_philox.Skip(delta);  // do the actual increasing
  return new_philox;
}

PHILOX_DEVICE_INLINE void UpdateMemWithPhiloxRandom(PhiloxRandom const& philox,
                                                    uint64 output_size,
                                                    StateElementType* ptr) {
  auto new_philox = SkipPhiloxRandom(philox, output_size);
  WritePhiloxRandomToMem(new_philox, ptr);
}

PHILOX_DEVICE_INLINE void UpdateCounterMemWithPhiloxRandom(
    PhiloxRandom::ResultType const& counter, uint64 output_size,
    StateElementType* ptr) {
  auto philox = PhiloxRandom(counter, PhiloxRandom::Key() /*dummy*/);
  auto new_philox = SkipPhiloxRandom(philox, output_size);
  WriteCounterToMem(new_philox.counter(), reinterpret_cast<uint64*>(ptr));
}

namespace functor {

// A per-device helper function that does the actual work for
// `UpdateVariableAndFill`.
// Reason to use functor: C++ doesn't allow function-template partial
// specialization.
template <typename Device, typename Distribution>
struct UpdateVariableAndFill_Philox;

template <typename Device>
struct RngSkip_Philox;

}  // end namespace functor

using CPUDevice = Eigen::ThreadPoolDevice;

class ScopedUnlockUnrefVar;

struct UpdateVariableAndFill_Philox_Arg {
  int64_t output_size;
  int64_t alg_tag_skip;
  // TODO(b/201572028): Rename `not_used`.
  ScopedUnlockUnrefVar* not_used;
  Tensor* state_tensor;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using GPUDevice = Eigen::GpuDevice;

namespace functor {

// Declares the partially GPU-specialized functor structs.
// must be kept at <=6 arguments because of a gcc/clang ABI incompatibility bug
template <typename Distribution>
struct UpdateVariableAndFill_Philox<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& device,
                  Distribution dist, UpdateVariableAndFill_Philox_Arg* arg,
                  typename Distribution::ResultElementType* output_data);
};

template <>
struct RngSkip_Philox<GPUDevice> {
  void operator()(const GPUDevice& device, const StateElementType* in_data,
                  uint64 delta, StateElementType* out_data);
};

}  // end namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_
