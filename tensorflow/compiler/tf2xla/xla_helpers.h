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
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Helper methods for building XLA computations.
class XlaHelpers {
 public:
  // Returns a handle representing the minimum value of a scalar
  // element of data_type.
  static xla::ComputationDataHandle MinValue(xla::ComputationBuilder* b,
                                             DataType data_type);

  // Returns a handle representing the maximum value of a scalar
  // element of data_type.
  static xla::ComputationDataHandle MaxValue(xla::ComputationBuilder* b,
                                             DataType data_type);

  // Returns a handle representing the zero value of a scalar
  // element of data_type.
  static xla::ComputationDataHandle Zero(xla::ComputationBuilder* b,
                                         DataType data_type);

  // Returns a handle representing the one value of a scalar
  // element of data_type.
  static xla::ComputationDataHandle One(xla::ComputationBuilder* b,
                                        DataType data_type);

  // Returns the machine epsilon for floating-point type `data_type`, i.e.,
  // the difference between 1.0 and the next representable value.
  static xla::ComputationDataHandle Epsilon(xla::ComputationBuilder* b,
                                            DataType data_type);

  // Returns a handle representing the given value of an integer scalar
  // element of data_type.
  // Note that unlike One and Zero, does not work on boolean types.
  static xla::ComputationDataHandle IntegerLiteral(xla::ComputationBuilder* b,
                                                   DataType data_type,
                                                   int64 value);

  // Returns a handle representing the given value of a floating-point scalar
  // element of data_type.
  static xla::ComputationDataHandle FloatLiteral(xla::ComputationBuilder* b,
                                                 DataType data_type,
                                                 double value);

  // Reshapes literal 'input' to have 'shape'. Both the original shape and
  // 'shape' must contain the same number of elements.
  static Status ReshapeLiteral(const xla::Literal& input,
                               gtl::ArraySlice<int64> shape,
                               xla::Literal* output);

  // Sets `argmax` to the argmax of `input` along `axis`. `input_shape` and
  // `input_dtype` are the shape and dtype of `input` respectively, and
  // `output_type` is the dtype to use for `argmax`.
  static Status ArgMax(xla::ComputationBuilder* builder,
                       XlaOpKernelContext* ctx,
                       const xla::ComputationDataHandle& input,
                       const TensorShape& input_shape, DataType input_type,
                       DataType output_type, int axis,
                       xla::ComputationDataHandle* argmax);

  // Sets `argmin` to the argmin of `input` along `axis`. `input_shape` and
  // `input_dtype` are the shape and dtype of `input` respectively, and
  // `output_type` is the dtype to use for `argmin`.
  static Status ArgMin(xla::ComputationBuilder* builder,
                       XlaOpKernelContext* ctx,
                       const xla::ComputationDataHandle& input,
                       const TensorShape& input_shape, DataType input_type,
                       DataType output_type, int axis,
                       xla::ComputationDataHandle* argmin);

  // Sets *iota to a rank 1 tensor with values [0, 1, 2, ...] of `dtype`.
  static Status Iota(xla::ComputationBuilder* builder, DataType dtype,
                     int64 size, xla::ComputationDataHandle* iota);

  // Converts `indices` into a one-hot representation. `depth` is the size
  // of the new axis to add. `axis` is the position at which to add the new
  // axis. `indices_shape` is the shape of `indices`. `on_value` and
  // `off_value` represent the values to use for the on and off positions,
  // respectively.
  static Status OneHot(xla::ComputationBuilder* builder, int64 depth, int axis,
                       DataType index_type, const TensorShape& indices_shape,
                       const xla::ComputationDataHandle& indices,
                       const xla::ComputationDataHandle& on_value,
                       const xla::ComputationDataHandle& off_value,
                       xla::ComputationDataHandle* one_hot);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_HELPERS_H_
