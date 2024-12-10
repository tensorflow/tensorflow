/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {

absl::StatusOr<tensorflow::DataType> ToTensorDataType(
    xla::ifrt::DType ifrt_dtype) {
  if (ifrt_dtype.kind() == xla::ifrt::DType::kString) {
    return tensorflow::DataType::DT_STRING;
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(ifrt_dtype));
  return tensorflow::EncodePrimitiveTypeAsDataType(primitive_type);
}

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(
    tensorflow::DataType tensor_dtype) {
  if (tensor_dtype == tensorflow::DataType::DT_STRING) {
    return xla::ifrt::DType(xla::ifrt::DType::kString);
  }
  xla::PrimitiveType primitive_type;
  TF_RETURN_IF_ERROR(
      tensorflow::DataTypeToPrimitiveType(tensor_dtype, &primitive_type));
  return xla::ifrt::ToDType(primitive_type);
}

xla::ifrt::Shape ToIfrtShape(const tensorflow::TensorShape& shape) {
  return xla::ifrt::Shape(shape.dim_sizes());
}

tensorflow::TensorShape ToTensorShape(const xla::ifrt::Shape& shape) {
  return tensorflow::TensorShape(shape.dims());
}
}  // namespace ifrt_serving
}  // namespace tensorflow
