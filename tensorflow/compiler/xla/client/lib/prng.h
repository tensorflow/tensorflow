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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_PRNG_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_PRNG_H_

#include <array>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Implements the ThreeFry counter-based PRNG algorithm.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
using ThreeFry2x32State = std::array<XlaOp, 2>;
ThreeFry2x32State ThreeFry2x32(ThreeFry2x32State input, ThreeFry2x32State key);

// Returns a tensor containing 'shape' random values uniformly distributed in
// the range [minval, maxval). Requires 2 32-bit integer seeds.
// Currently only 'shape's of type F32, S32 and S64 are implemented.
XlaOp StatelessRngUniform(std::array<XlaOp, 2> seeds, const Shape& shape,
                          XlaOp minval, XlaOp maxval);

// Converts a 32-bit (signed or unsigned) integer random number `bits` into a
// float32 in the range [minval, maxval).
XlaOp StatelessRngUniformF32(XlaOp bits, XlaOp minval, XlaOp maxval);

// Converts an integer random number 'bits' of type 'type' to a random number
// in the range [minval, maxval), of the same type. 'unsigned_type' is the
// unsigned version of 'type' (could be the same) with the same bit width.
// The algorithm is the same one that TF uses right now, but it's
// uniform only when maxval - minval is a divisor of the range that bits is
// generated from.
// TODO(b/72573764): Generate real uniform integer distribution.
XlaOp StatelessRngUniformInt(XlaOp bits, XlaOp minval, XlaOp maxval,
                             PrimitiveType type, PrimitiveType unsigned_type);

// The following 2 functions, for converting between one uint64 and two uint32s,
// use the contract "lower 32 bits for the first uint32, higher 32 bits for the
// second".
ThreeFry2x32State Uint64ToUint32s(XlaOp u64);
XlaOp Uint32sToUint64(ThreeFry2x32State u32s);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_PRNG_H_
