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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_CLIENT_IGNITE_BYTE_SWAPPER_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_CLIENT_IGNITE_BYTE_SWAPPER_H_

#include <stdint.h>
#include "tensorflow/core/platform/byte_order.h"

namespace tensorflow {

class ByteSwapper {
 public:
  ByteSwapper(bool big_endian) { swap_ = big_endian == port::kLittleEndian; }

  void SwapIfRequiredInt16(int16_t *x) const {
    if (swap_) {
      Swap16(x);
    }
  }

  void SwapIfRequiredUnsignedInt16(uint16_t *x) const {
    if (swap_) {
      Swap16(reinterpret_cast<int16_t *>(x));
    }
  }

  void SwapIfRequiredInt32(int32_t *x) const {
    if (swap_) {
      Swap32(x);
    }
  }

  void SwapIfRequiredFloat(float *x) const {
    if (swap_) {
      Swap32(reinterpret_cast<int32_t *>(x));
    }
  }

  void SwapIfRequiredInt64(int64_t *x) const {
    if (swap_) {
      Swap64(x);
    }
  }

  void SwapIfRequiredDouble(double *x) const {
    if (swap_) {
      Swap64(reinterpret_cast<int64_t *>(x));
    }
  }

  void SwapIfRequiredInt16Arr(int16_t *x, int32_t length) const {
    if (swap_) {
      for (int32_t i = 0; i < length; i++) Swap16(&x[i]);
    }
  }

  void SwapIfRequiredUnsignedInt16Arr(uint16_t *x, int32_t length) const {
    if (swap_) {
      for (int32_t i = 0; i < length; i++)
        Swap16(reinterpret_cast<int16_t *>(&x[i]));
    }
  }

  void SwapIfRequiredInt32Arr(int32_t *x, int32_t length) const {
    if (swap_) {
      for (int32_t i = 0; i < length; i++) Swap32(&x[i]);
    }
  }

  void SwapIfRequiredFloatArr(float *x, int32_t length) const {
    if (swap_) {
      for (int32_t i = 0; i < length; i++)
        Swap32(reinterpret_cast<int32_t *>(&x[i]));
    }
  }

  void SwapIfRequiredInt64Arr(int64_t *x, int32_t length) const {
    if (swap_) {
      for (int32_t i = 0; i < length; i++) Swap64(&x[i]);
    }
  }

  void SwapIfRequiredDoubleArr(double *x, int32_t length) const {
    if (swap_) {
      for (int32_t i = 0; i < length; i++)
        Swap64(reinterpret_cast<int64_t *>(&x[i]));
    }
  }

 private:
  void Swap16(int16_t *x) const {
    *x = ((*x & 0xFF) << 8) | ((*x >> 8) & 0xFF);
  }

  void Swap32(int32_t *x) const {
    *x = ((*x & 0xFF) << 24) | (((*x >> 8) & 0xFF) << 16) |
         (((*x >> 16) & 0xFF) << 8) | ((*x >> 24) & 0xFF);
  }

  void Swap64(int64_t *x) const {
    *x = ((*x & 0xFF) << 56) | (((*x >> 8) & 0xFF) << 48) |
         (((*x >> 16) & 0xFF) << 40) | (((*x >> 24) & 0xFF) << 32) |
         (((*x >> 32) & 0xFF) << 24) | (((*x >> 40) & 0xFF) << 16) |
         (((*x >> 48) & 0xFF) << 8) | ((*x >> 56) & 0xFF);
  }

  bool swap_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_CLIENT_IGNITE_BYTE_SWAPPER_H_
