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

#ifndef TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_
#define TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_

#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {

// 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained
// in b/111604096 and cl/171681867), so I use signed int here. I choose int64
// instead of int32 because `VarHandleOp` doesn't support int32 on GPU.
using StateElementType = int64;
static constexpr DataType STATE_ELEMENT_DTYPE = DT_INT64;

using Algorithm = StateElementType;
static constexpr Algorithm RNG_ALG_PHILOX = 1;

using random::PhiloxRandom;

static constexpr int64 PHILOX_MIN_STATE_SIZE =
    (PhiloxRandom::ResultType::kElementCount +
     PhiloxRandom::Key::kElementCount) /
    2;

// The following two functions use the contract "lower 32 bits for the first
// uint32, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
PHILOX_DEVICE_FUNC void Int64ToUint32s(int64 input, uint32* output1,
                                       uint32* output2) {
  auto u64 = static_cast<uint64>(input);
  *output1 = static_cast<uint32>(u64);
  *output2 = static_cast<uint32>(u64 >> 32);
}

PHILOX_DEVICE_FUNC int64 Uint32sToInt64(uint32 input1, uint32 input2) {
  auto u64_1 = static_cast<uint64>(input1);
  auto u64_2 = static_cast<uint64>(input2);
  return static_cast<int64>(u64_1 | (u64_2 << 32));
}

PHILOX_DEVICE_FUNC PhiloxRandom
GetPhiloxRandomFromMem(StateElementType const* ptr) {
  PhiloxRandom::ResultType counter;
  PhiloxRandom::Key key;
  Int64ToUint32s(ptr[0], &counter[0], &counter[1]);
  Int64ToUint32s(ptr[1], &counter[2], &counter[3]);
  Int64ToUint32s(ptr[2], &key[0], &key[1]);
  return PhiloxRandom(counter, key);
}

PHILOX_DEVICE_FUNC void WritePhiloxRandomToMem(PhiloxRandom const& philox,
                                               StateElementType* ptr) {
  PhiloxRandom::ResultType const& counter = philox.counter();
  PhiloxRandom::Key const& key = philox.key();
  ptr[0] = Uint32sToInt64(counter[0], counter[1]);
  ptr[1] = Uint32sToInt64(counter[2], counter[3]);
  ptr[2] = Uint32sToInt64(key[0], key[1]);
}

PHILOX_DEVICE_FUNC void UpdateMemWithPhiloxRandom(PhiloxRandom const& philox,
                                                  int64 output_size,
                                                  StateElementType* ptr) {
  auto new_philox = philox;
  // Multiplier 256 is the same as in `FillPhiloxRandomTask`; do not change
  // it just here.
  auto delta = output_size * 256;
  new_philox.Skip(delta);  // do the actual increasing
  WritePhiloxRandomToMem(new_philox, ptr);
}

// Does unlock and unref automatically when going out of scope, and also
// supports early manual release.
class ScopedUnlockUnref {
 public:
  explicit ScopedUnlockUnref(Var* var) : var_(var) {
    if (var_) {
      var_->mu()->lock();
    }
  }
  void Release() {
    if (var_) {
      var_->mu()->unlock();
      var_->Unref();
      var_ = nullptr;
    }
  }
  ~ScopedUnlockUnref() { Release(); }

 private:
  Var* var_;

  ScopedUnlockUnref(const ScopedUnlockUnref&) = delete;
  void operator=(const ScopedUnlockUnref&) = delete;
};

// A per-device helper function that does the actual work for
// `UpdateVariableAndFill`.
// Reason to use functor: C++ doesn't allow function-template partial
// specialization.
template <typename Device, typename Distribution>
struct UpdateVariableAndFill_Philox;

using CPUDevice = Eigen::ThreadPoolDevice;

#if GOOGLE_CUDA

using GPUDevice = Eigen::GpuDevice;

// Declares the partially GPU-specialized functor struct.
template <typename Distribution>
struct UpdateVariableAndFill_Philox<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& device,
                  int64 output_size, int64 alg_tag_skip,
                  ScopedUnlockUnref* not_used, Tensor* state_tensor,
                  typename Distribution::ResultElementType* output_data);
};

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_
