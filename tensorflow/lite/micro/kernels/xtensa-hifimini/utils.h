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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_UTILS_H_

#include <xtensa/tie/xt_hifi2.h>

// INT24 MIN/MAX
#define INT24_MIN -8388608
#define INT24_MAX 8388607

// Converts an int32 value into a 2x24bit PR register file. If the int32 value
// is outside the numerical limits of a 24bit integer, the "fractional" or lower
// 8bits are discarded. If the value is within the range of a 24 bit integer,
// the "signed" or upper 8bits are discarded.
inline ae_p24x2s AE_CONVERT_INT32_24x2(int32_t v) {
  if (v > INT24_MIN && v < INT24_MAX) {
    return *((ae_p24s*)&v);
  } else {
    return (ae_p24s) * ((ae_p24f*)&v);
  }
}

// Shifts a 48bit accumulator value into 32bit space and returns the value.
#define AE_CONVERT_Q56_INT32(v) AE_TRUNCA32Q48(AE_Q56S_SLAI(v, 16))

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_HIFIMINI_UTILS_H_
