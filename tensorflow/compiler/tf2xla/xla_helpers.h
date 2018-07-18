/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This file defines helper routines for the XLA device.

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_

#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Helper methods for building XLA computations.
class XlaHelpers {
 public:
  // Returns a handle representing the zero value of a scalar
  // element of data_type.
  static xla::XlaOp Zero(xla::XlaBuilder* b, DataType data_type);

  // Returns a handle representing the one value of a scalar
  // element of data_type.
  static xla::XlaOp One(xla::XlaBuilder* b, DataType data_type);

  // Returns a handle representing the given value of an integer scalar
  // element of data_type.
  // Note that unlike One and Zero, does not work on boolean types.
  static xla::XlaOp IntegerLiteral(xla::XlaBuilder* b, DataType data_type,
                                   int64 value);

  // Returns a handle representing the given value of a floating-point scalar
  // element of data_type.
  static xla::XlaOp FloatLiteral(xla::XlaBuilder* b, DataType data_type,
                                 double value);

  // Reshapes literal 'input' to have 'shape'. Both the original shape and
  // 'shape' must contain the same number of elements.
  static Status ReshapeLiteral(const xla::Literal& input,
                               gtl::ArraySlice<int64> shape,
                               xla::Literal* output);

  // Returns the argmax of `input` along `axis`. `output_type` is the type to
  // use for the output.
  static xla::XlaOp ArgMax(xla::XlaOp input, xla::PrimitiveType output_type,
                           int axis);

  // Returns the argmin of `input` along `axis`. `output_type` is the type to
  // use for the output.
  static xla::XlaOp ArgMin(xla::XlaOp input, xla::PrimitiveType output_type,
                           int axis);

  // Converts `indices` into a one-hot representation. `depth` is the size
  // of the new axis to add. `axis` is the position at which to add the new
  // axis. `indices_shape` is the shape of `indices`. `on_value` and
  // `off_value` represent the values to use for the on and off positions,
  // respectively.
  static Status OneHot(xla::XlaBuilder* builder, int64 depth, int axis,
                       DataType index_type, const TensorShape& indices_shape,
                       const xla::XlaOp& indices, const xla::XlaOp& on_value,
                       const xla::XlaOp& off_value, xla::XlaOp* one_hot);

  // Certain DataTypes should use increased precision DataTypes when performing
  // reductions.  This function remaps a given DataType to a higher precision
  // DataType if needed.
  static DataType SumAccumulationType(const DataType& dtype);

  // A helper for creating a ConvertElementType xla op given a DataType rather
  // than the xla::PrimitiveType.
  static xla::XlaOp ConvertElementType(xla::XlaBuilder* const builder,
                                       const xla::XlaOp& operand,
                                       const DataType new_element_type);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_
