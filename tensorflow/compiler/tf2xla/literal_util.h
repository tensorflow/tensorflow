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

// Utilities for working with XLA Literals.

#ifndef TENSORFLOW_COMPILER_TF2XLA_LITERAL_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_LITERAL_UTIL_H_

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

// Returns a BorrowingLiteral that utilizes the same underlying buffer owned by
// 'host_tensor'.
Status HostTensorToBorrowingLiteral(const Tensor& host_tensor,
                                    xla::BorrowingLiteral* literal);

// Returns a BorrowingLiteral tuple that utilizes the same underlying buffers
// owned by 'host_tensors'.
Status HostTensorsToBorrowingLiteralTuple(
    tensorflow::gtl::ArraySlice<Tensor> host_tensors,
    xla::BorrowingLiteral* literal);

// Copies 'literal' to freshly allocated 'host_tensor', which is allocated of
// type <target_type>.
// Fails if the literal's primitive type !=
// DataTypeToPrimitiveType(target_type). Note that <target_type> is not
// derivable from the type of <literal>, because multiple tensorflow types map
// to the same XLA type (e.g. INT32 and QINT32 both map to INT32 in
// XLA).
Status LiteralToHostTensor(const xla::LiteralSlice& literal,
                           DataType target_type, Tensor* host_tensor);

// Copies the contents of 'literal' to a previously allocated tensor
// 'host_tensor'. The tensor and the literal must have the same number of
// elements and the same type.
Status CopyLiteralToHostTensor(const xla::LiteralSlice& literal,
                               Tensor* host_tensor);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LITERAL_UTIL_H_
