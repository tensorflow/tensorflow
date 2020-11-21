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

#ifndef TENSORFLOW_CORE_FRAMEWORK_BFLOAT16_H_
#define TENSORFLOW_CORE_FRAMEWORK_BFLOAT16_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"

// Compact 16-bit encoding of floating point numbers. This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.  It
// is assumed that floats are in IEEE 754 format so the representation is just
// bits 16-31 of a single precision float.
//
// NOTE: The IEEE floating point standard defines a float16 format that
// is different than this format (it has fewer bits of exponent and more
// bits of mantissa).  We don't use that format here because conversion
// to/from 32-bit floats is more complex for that format, and the
// conversion for this format is very simple.
//
// Because of the existing IEEE float16 type, we do not name our representation
// "float16" but just use "uint16".
//
// <-----our 16bits float------->
// s e e e e e e e e f f f f f f f f f f f f f f f f f f f f f f f
// <------------------------------float-------------------------->
// 3 3             2 2             1 1                           0
// 1 0             3 2             5 4                           0
//
//
// This type only supports conversion back and forth with float.
//
// This file must be compilable by nvcc.
//
// The type is defined in framework/numeric_types.h.

namespace tensorflow {

// Convert from float to bfloat16 with rounding-to-nearest-even.
void RoundFloatToBFloat16(const float* src, bfloat16* dst, int64 size);
// Convert from float to bfloat16 with truncation. Notice this conversion is
// lossy since it truncates the float to 7 mantissa bits without rounding.
void FloatToBFloat16(const float* src, bfloat16* dst, int64 size);
// Convert from bfloat16 to float. This conversion is lossless.
void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_BFLOAT16_H_
