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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("Invert")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, uint32, uint64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Flips all bits elementwise.

The result will have exactly those bits set, that are not set in `x`. The
computation is performed on the underlying representation of x.
)doc");

#define BINARY_BITWISE()                                                     \
  Input("x: T")                                                              \
      .Input("y: T")                                                         \
      .Output("z: T")                                                        \
      .SetIsCommutative()                                                    \
      .Attr("T: {int8, int16, int32, int64, uint8, uint16, uint32, uint64}") \
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)

REGISTER_OP("PopulationCount")
    .Input("x: T")
    .Output("y: uint8")
    .Attr("T: {int8, int16, int32, int64, uint8, uint16, uint32, uint64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).

For each entry in `x`, calculates the number of `1` (on) bits in the binary
representation of that entry.

**NOTE**: It is more efficient to first `tf.bitcast` your tensors into
`int32` or `int64` and perform the bitcount on the result, than to feed in
8- or 16-bit inputs and then aggregate the resulting counts.
)doc");

REGISTER_OP("BitwiseAnd")
    .BINARY_BITWISE()
    .Doc(R"doc(
Elementwise computes the bitwise AND of `x` and `y`.

The result will have those bits set, that are set in both `x` and `y`. The
computation is performed on the underlying representations of `x` and `y`.
)doc");

REGISTER_OP("BitwiseOr")
    .BINARY_BITWISE()
    .Doc(R"doc(
Elementwise computes the bitwise OR of `x` and `y`.

The result will have those bits set, that are set in `x`, `y` or both. The
computation is performed on the underlying representations of `x` and `y`.
)doc");

REGISTER_OP("BitwiseXor")
    .BINARY_BITWISE()
    .Doc(R"doc(
Elementwise computes the bitwise XOR of `x` and `y`.

The result will have those bits set, that are different in `x` and `y`. The
computation is performed on the underlying representations of `x` and `y`.
)doc");

REGISTER_OP("LeftShift")
    .BINARY_BITWISE()
    .Doc(R"doc(
Elementwise computes the bitwise left-shift of `x` and `y`.

If `y` is negative, or greater than or equal to the width of `x` in bits the
result is implementation defined.
)doc");

REGISTER_OP("RightShift")
    .BINARY_BITWISE()
    .Doc(R"doc(
Elementwise computes the bitwise right-shift of `x` and `y`.

Performs a logical shift for unsigned integer types, and an arithmetic shift
for signed integer types.

If `y` is negative, or greater than or equal to than the width of `x` in bits
the result is implementation defined.
)doc");

}  // namespace tensorflow
