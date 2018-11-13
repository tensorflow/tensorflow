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

#include "tensorflow/compiler/tf2xla/type_util.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status DataTypeToPrimitiveType(DataType data_type, xla::PrimitiveType* type) {
  switch (data_type) {
    case tensorflow::DT_BOOL:
      *type = xla::PRED;
      return Status::OK();
    case tensorflow::DT_INT8:
    case tensorflow::DT_QINT8:
      *type = xla::S8;
      return Status::OK();
    case tensorflow::DT_INT16:
    case tensorflow::DT_QINT16:
      *type = xla::S16;
      return Status::OK();
    case tensorflow::DT_INT32:
    case tensorflow::DT_QINT32:
      *type = xla::S32;
      return Status::OK();
    case tensorflow::DT_INT64:
      *type = xla::S64;
      return Status::OK();
    case tensorflow::DT_UINT8:
    case tensorflow::DT_QUINT8:
      *type = xla::U8;
      return Status::OK();
    case tensorflow::DT_UINT16:
    case tensorflow::DT_QUINT16:
      *type = xla::U16;
      return Status::OK();
    case tensorflow::DT_UINT32:
      *type = xla::U32;
      return Status::OK();
    case tensorflow::DT_UINT64:
      *type = xla::U64;
      return Status::OK();
    case tensorflow::DT_BFLOAT16:
      *type = xla::BF16;
      return Status::OK();
    case tensorflow::DT_HALF:
      *type = xla::F16;
      return Status::OK();
    case tensorflow::DT_FLOAT:
      *type = xla::F32;
      return Status::OK();
    case tensorflow::DT_DOUBLE:
      *type = xla::F64;
      return Status::OK();
    case tensorflow::DT_COMPLEX64:
      *type = xla::C64;
      return Status::OK();
    default:
      return errors::InvalidArgument(
          "Unsupported type in DataTypeToPrimitiveType ",
          DataTypeString(data_type));
  }
}

}  // namespace tensorflow
