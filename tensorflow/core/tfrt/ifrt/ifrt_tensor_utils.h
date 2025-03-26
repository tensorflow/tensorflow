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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_TENSOR_UTILS_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_TENSOR_UTILS_H_

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {
namespace ifrt_serving {

absl::StatusOr<tensorflow::DataType> ToTensorDataType(
    xla::ifrt::DType ifrt_dtype);

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(tensorflow::DataType tensor_dtype);

xla::ifrt::Shape ToIfrtShape(const tensorflow::TensorShape& shape);

tensorflow::TensorShape ToTensorShape(const xla::ifrt::Shape& shape);

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_TENSOR_UTILS_H_
