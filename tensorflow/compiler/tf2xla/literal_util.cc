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

#include "tensorflow/compiler/tf2xla/literal_util.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

absl::Status HostTensorToBorrowingLiteral(const Tensor& host_tensor,
                                          xla::BorrowingLiteral* literal) {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(host_tensor.dtype(),
                                           host_tensor.shape(), &xla_shape));
  return HostTensorToBorrowingLiteral(xla_shape, host_tensor, literal);
}

absl::Status HostTensorToBorrowingLiteral(const xla::Shape& xla_shape,
                                          const Tensor& host_tensor,
                                          xla::BorrowingLiteral* literal) {
  const auto& tshape = host_tensor.shape();
  TF_RET_CHECK(tshape.IsFullyDefined() &&
               tshape.dims() == xla_shape.dimensions_size() &&
               tshape.dim_sizes() == xla_shape.dimensions())
      << "Provided xla::Shape must have the same dims as the Tensor shape.";
  *literal = xla::BorrowingLiteral(
      static_cast<const char*>(DMAHelper::base(&host_tensor)), xla_shape);
  return absl::OkStatus();
}

absl::StatusOr<xla::Literal> HostTensorToLiteral(const Tensor& host_tensor) {
  xla::BorrowingLiteral literal;
  TF_RETURN_IF_ERROR(HostTensorToBorrowingLiteral(host_tensor, &literal));
  return literal.Clone();
}

absl::Status HostTensorToMutableBorrowingLiteral(
    Tensor* host_tensor, xla::MutableBorrowingLiteral* literal) {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(host_tensor->dtype(),
                                           host_tensor->shape(), &xla_shape));
  return HostTensorToMutableBorrowingLiteral(xla_shape, host_tensor, literal);
}

absl::Status HostTensorToMutableBorrowingLiteral(
    const xla::Shape& xla_shape, Tensor* host_tensor,
    xla::MutableBorrowingLiteral* literal) {
  *literal = xla::MutableBorrowingLiteral(
      static_cast<const char*>(DMAHelper::base(host_tensor)), xla_shape);

  return absl::OkStatus();
}

absl::Status HostTensorsToBorrowingLiteralTuple(
    absl::Span<const Tensor> host_tensors, xla::BorrowingLiteral* literal) {
  std::vector<const char*> buf_ptrs;
  buf_ptrs.reserve(host_tensors.size());
  std::vector<xla::Shape> tensor_shapes(host_tensors.size());

  for (int i = 0, end = host_tensors.size(); i < end; i++) {
    // Validate runtime shapes and fail if it doesn't match the contract.
    const Tensor* tensor = &host_tensors[i];
    buf_ptrs.emplace_back(static_cast<const char*>(DMAHelper::base(tensor)));
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(tensor->dtype(), tensor->shape(),
                                             &tensor_shapes[i]));
  }

  *literal = xla::BorrowingLiteral(
      buf_ptrs, xla::ShapeUtil::MakeTupleShape(tensor_shapes));

  return absl::OkStatus();
}

absl::Status CopyLiteralToHostTensor(const xla::LiteralSlice& literal,
                                     Tensor* host_tensor) {
  TF_RET_CHECK(literal.shape().IsArray() &&
               xla::ShapeUtil::ElementsIn(literal.shape()) ==
                   host_tensor->NumElements());
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(
      DataTypeToPrimitiveType(host_tensor->dtype(), &primitive_type));
  if (literal.shape().element_type() != primitive_type) {
    return errors::InvalidArgument(
        "Cannot convert literal of type ",
        xla::PrimitiveType_Name(literal.shape().element_type()),
        " to tensor of type ", DataTypeString(host_tensor->dtype()));
  }
  size_t total_bytes = host_tensor->TotalBytes();
  if (total_bytes > 0) {
    const void* src_ptr = literal.untyped_data();
    void* dst_ptr = DMAHelper::base(host_tensor);
    memcpy(dst_ptr, src_ptr, total_bytes);
  }
  return absl::OkStatus();
}

absl::Status LiteralToHostTensor(const xla::LiteralSlice& literal,
                                 DataType target_type, Tensor* host_tensor) {
  TensorShape shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(literal.shape(), &shape));
  *host_tensor = Tensor(target_type, shape);
  return CopyLiteralToHostTensor(literal, host_tensor);
}

}  // namespace tensorflow
