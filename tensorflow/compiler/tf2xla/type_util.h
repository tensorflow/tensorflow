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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TYPE_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_TYPE_UTIL_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Converts a Tensorflow DataType to an XLA PrimitiveType.
Status DataTypeToPrimitiveType(DataType data_type, xla::PrimitiveType* type);

// Converts an XLA PrimitiveType to a TensorFlow DataType.
// Caution: The mapping from TF types to XLA types is not one-to-one: for
// example, both DT_INT8 and DT_QINT8 map to xla::S8. So the inverse is not a
// uniquely defined function. This is fine if you want a way to encode an XLA
// object as a TensorFlow object (e.g., in XRT); whereas if you started with a
// TensorFlow object in the first place, you most likely should preserve the
// original TensorFlow type, rather than trying to convert an XLA type back into
// a TensorFlow type.
StatusOr<DataType> EncodePrimitiveTypeAsDataType(xla::PrimitiveType type);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TYPE_UTIL_H_
