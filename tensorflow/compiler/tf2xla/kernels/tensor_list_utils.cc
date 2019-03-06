/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

bool IsTensorListInput(XlaOpKernelContext* ctx, int index) {
  return ctx->InputExpression(index).kind() == XlaExpression::Kind::kTensorList;
}

Status BuildTensorList(const xla::XlaOp& buffer, const xla::XlaOp& push_index,
                       xla::XlaOp* output_list) {
  TF_RET_CHECK(buffer.builder());
  *output_list = xla::Tuple(buffer.builder(), {buffer, push_index});
  return Status::OK();
}

Status GetTensorListBuffer(const xla::XlaOp& op, xla::XlaOp* buffer) {
  TF_RET_CHECK(op.builder());
  *buffer = xla::GetTupleElement(op, 0);
  return Status::OK();
}

Status GetTensorListPushIndex(const xla::XlaOp& op, xla::XlaOp* push_index) {
  TF_RET_CHECK(op.builder());
  *push_index = xla::GetTupleElement(op, 1);
  return Status::OK();
}

Status GetTensorListBufferShape(const xla::XlaOp& op,
                                TensorShape* buffer_shape) {
  TF_RET_CHECK(op.builder());
  TensorShape shape;
  TF_ASSIGN_OR_RETURN(const xla::Shape& list_tuple_shape,
                      op.builder()->GetShape(op));
  return GetTensorListBufferShape(list_tuple_shape, buffer_shape);
}

Status GetTensorListBufferShape(const xla::Shape& list_shape,
                                TensorShape* buffer_shape) {
  TF_RET_CHECK(list_shape.IsTuple());
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(
      xla::ShapeUtil::GetTupleElementShape(list_shape, 0), buffer_shape));
  return Status::OK();
}

Status IsTensorListInitialized(const xla::XlaOp& op, bool* is_initialized) {
  TensorShape list_shape;
  TF_RETURN_IF_ERROR(GetTensorListBufferShape(op, &list_shape));
  *is_initialized = !(list_shape.dims() == 2 && list_shape.dim_size(1) == 0);
  return Status::OK();
}

Status InitializeTensorList(const xla::XlaOp& uninitialized_list,
                            const TensorShape& buffer_shape,
                            xla::XlaOp* output_list) {
  TensorShape input_buffer_shape;
  TF_RETURN_IF_ERROR(
      GetTensorListBufferShape(uninitialized_list, &input_buffer_shape));
  if (input_buffer_shape.dim_size(0) != buffer_shape.dim_size(0)) {
    return errors::InvalidArgument(
        "Number of elements in input list does not match buffer size. ",
        "input list size: ", input_buffer_shape.dim_size(0),
        "buffer size: ", buffer_shape.dim_size(0));
  }
  xla::XlaBuilder* builder = uninitialized_list.builder();
  xla::XlaOp input_buffer;
  TF_RETURN_IF_ERROR(GetTensorListBuffer(uninitialized_list, &input_buffer));
  TF_ASSIGN_OR_RETURN(const xla::Shape& input_buffer_xla_shape,
                      builder->GetShape(input_buffer));
  auto new_buffer = xla::Broadcast(
      xla::ConstantLiteral(builder, xla::LiteralUtil::Zero(
                                        input_buffer_xla_shape.element_type())),
      buffer_shape.dim_sizes());
  xla::XlaOp push_index;
  TF_RETURN_IF_ERROR(GetTensorListPushIndex(uninitialized_list, &push_index));
  return BuildTensorList(new_buffer, push_index, output_list);
}

}  // namespace tensorflow
