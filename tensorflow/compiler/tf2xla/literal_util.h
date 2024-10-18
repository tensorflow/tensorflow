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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Returns a BorrowingLiteral that utilizes the same underlying buffer owned by
// 'host_tensor'.
absl::Status HostTensorToBorrowingLiteral(const Tensor& host_tensor,
                                          xla::BorrowingLiteral* literal);
// Similar as above, except the literal shape is explicitly provided and used
// instead of obtaining it from the 'host_tensor'. The provided literal shape
// 'xla_shape' must be compatible with the shape of 'host_tensor'.
absl::Status HostTensorToBorrowingLiteral(const xla::Shape& xla_shape,
                                          const Tensor& host_tensor,
                                          xla::BorrowingLiteral* literal);

// Returns a Literal with the contents of 'host_tensor', backed by its own
// storage (i.e., not reusing 'host_tensor's buffers.)
absl::StatusOr<xla::Literal> HostTensorToLiteral(const Tensor& host_tensor);

// Returns a MutableBorrowingLiteral that utilizes the same underlying buffer
// owned by 'host_tensor', but is mutable via the xla::Literal methods.
absl::Status HostTensorToMutableBorrowingLiteral(
    Tensor* host_tensor, xla::MutableBorrowingLiteral* literal);
// Similar as above, except the literal shape is explicitly provided and used
// instead of obtaining it from the 'host_tensor'. The provided literal shape
// 'xla_shape' must be compatible with the shape of 'host_tensor'.
absl::Status HostTensorToMutableBorrowingLiteral(
    const xla::Shape& xla_shape, Tensor* host_tensor,
    xla::MutableBorrowingLiteral* literal);

// Returns a BorrowingLiteral tuple that utilizes the same underlying buffers
// owned by 'host_tensors'.
absl::Status HostTensorsToBorrowingLiteralTuple(
    absl::Span<const Tensor> host_tensors, xla::BorrowingLiteral* literal);

// Copies 'literal' to freshly allocated 'host_tensor', which is allocated of
// type <target_type>.
// Fails if the literal's primitive type !=
// DataTypeToPrimitiveType(target_type). Note that <target_type> is not
// derivable from the type of <literal>, because multiple tensorflow types map
// to the same XLA type (e.g. INT32 and QINT32 both map to INT32 in
// XLA).
absl::Status LiteralToHostTensor(const xla::LiteralSlice& literal,
                                 DataType target_type, Tensor* host_tensor);

// Copies the contents of 'literal' to a previously allocated tensor
// 'host_tensor'. The tensor and the literal must have the same number of
// elements and the same type.
absl::Status CopyLiteralToHostTensor(const xla::LiteralSlice& literal,
                                     Tensor* host_tensor);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LITERAL_UTIL_H_
