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
    .SetShapeFn(shape_inference::UnchangedShape);

#define BINARY_BITWISE()                                                     \
  Input("x: T")                                                              \
      .Input("y: T")                                                         \
      .Output("z: T")                                                        \
      .Attr("T: {int8, int16, int32, int64, uint8, uint16, uint32, uint64}") \
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)

#define BINARY_BITWISE_COMMUTATIVE()                                         \
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
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("BitwiseAnd").BINARY_BITWISE_COMMUTATIVE();

REGISTER_OP("BitwiseOr").BINARY_BITWISE_COMMUTATIVE();

REGISTER_OP("BitwiseXor").BINARY_BITWISE_COMMUTATIVE();

REGISTER_OP("LeftShift").BINARY_BITWISE();

REGISTER_OP("RightShift").BINARY_BITWISE();

}  // namespace tensorflow
