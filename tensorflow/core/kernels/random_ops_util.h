/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_RANDOM_OPS_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_RANDOM_OPS_UTIL_H_

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using random::PhiloxRandom;

// The following 2 functions use the contract "lower 32 bits for the first
// uint32, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
PHILOX_DEVICE_INLINE void Uint64ToUint32s(uint64_t input, uint32_t* output1,
                                          uint32_t* output2) {
  *output1 = static_cast<uint32_t>(input);
  *output2 = static_cast<uint32_t>(input >> 32);
}

PHILOX_DEVICE_INLINE uint64_t Uint32sToUint64(uint32_t input1,
                                              uint32_t input2) {
  auto u64_1 = static_cast<uint64_t>(input1);
  auto u64_2 = static_cast<uint64_t>(input2);
  return u64_1 | (u64_2 << 32);
}

PHILOX_DEVICE_INLINE PhiloxRandom::ResultType GetCounterFromMem(
    const uint64_t* ptr) {
  PhiloxRandom::ResultType counter;
  Uint64ToUint32s(ptr[0], &counter[0], &counter[1]);
  Uint64ToUint32s(ptr[1], &counter[2], &counter[3]);
  return counter;
}

PHILOX_DEVICE_INLINE void WriteCounterToMem(
    PhiloxRandom::ResultType const& counter, uint64_t* ptr) {
  ptr[0] = Uint32sToUint64(counter[0], counter[1]);
  ptr[1] = Uint32sToUint64(counter[2], counter[3]);
}

PHILOX_DEVICE_INLINE PhiloxRandom::Key GetKeyFromMem(const uint64_t* ptr) {
  PhiloxRandom::Key key;
  Uint64ToUint32s(ptr[0], &key[0], &key[1]);
  return key;
}

PHILOX_DEVICE_INLINE void WriteKeyToMem(PhiloxRandom::Key const& key,
                                        uint64_t* ptr) {
  *ptr = Uint32sToUint64(key[0], key[1]);
}

PHILOX_DEVICE_INLINE PhiloxRandom GetPhiloxRandomFromCounterKeyMem(
    const uint64_t* counter_ptr, const uint64_t* key_ptr) {
  return PhiloxRandom(GetCounterFromMem(counter_ptr), GetKeyFromMem(key_ptr));
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RANDOM_OPS_UTIL_H_
