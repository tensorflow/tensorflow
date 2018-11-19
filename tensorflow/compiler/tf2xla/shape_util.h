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

// Utilities for working with XLA shapes.

#ifndef TENSORFLOW_COMPILER_TF2XLA_SHAPE_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_SHAPE_UTIL_H_

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// Convert an XLA Shape into the equivalent TensorFlow shape. May fail since
// not all XLA shapes can be represented as TensorShapes.
Status XLAShapeToTensorShape(const xla::Shape& shape,
                             TensorShape* tensor_shape);

// Convert a TensorShape into the equivalent XLA Shape proto. Unlike Tensorflow,
// XLA shapes include the type. Not all `dtype` values can be represented by
// XLA, so this conversion may fail.
Status TensorShapeToXLAShape(DataType dtype, const TensorShape& tensor_shape,
                             xla::Shape* shape);

// Converts a TensorShape into the equivalent XLA Shape proto, taking an
// xla::PrimitiveType to specify the element type.  This never fails.
xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const TensorShape& tensor_shape);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_SHAPE_UTIL_H_
