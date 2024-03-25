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

#include "absl/container/flat_hash_map.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

Status DataTypeToPrimitiveType(DataType data_type, xla::PrimitiveType* type) {
  switch (data_type) {
    case tensorflow::DT_BOOL:
      *type = xla::PRED;
      return absl::OkStatus();
    case tensorflow::DT_INT4:
      *type = xla::S4;
      return absl::OkStatus();
    case tensorflow::DT_INT8:
    case tensorflow::DT_QINT8:
      *type = xla::S8;
      return absl::OkStatus();
    case tensorflow::DT_INT16:
    case tensorflow::DT_QINT16:
      *type = xla::S16;
      return absl::OkStatus();
    case tensorflow::DT_INT32:
    case tensorflow::DT_QINT32:
      *type = xla::S32;
      return absl::OkStatus();
    case tensorflow::DT_INT64:
      *type = xla::S64;
      return absl::OkStatus();
    case tensorflow::DT_UINT4:
      *type = xla::U4;
      return absl::OkStatus();
    case tensorflow::DT_UINT8:
    case tensorflow::DT_QUINT8:
      *type = xla::U8;
      return absl::OkStatus();
    case tensorflow::DT_UINT16:
    case tensorflow::DT_QUINT16:
      *type = xla::U16;
      return absl::OkStatus();
    case tensorflow::DT_UINT32:
      *type = xla::U32;
      return absl::OkStatus();
    case tensorflow::DT_UINT64:
      *type = xla::U64;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT8_E5M2:
      *type = xla::F8E5M2;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT8_E4M3FN:
      *type = xla::F8E4M3FN;
      return absl::OkStatus();
    case tensorflow::DT_BFLOAT16:
      *type = xla::BF16;
      return absl::OkStatus();
    case tensorflow::DT_HALF:
      *type = xla::F16;
      return absl::OkStatus();
    case tensorflow::DT_FLOAT:
      *type = xla::F32;
      return absl::OkStatus();
    case tensorflow::DT_DOUBLE:
      *type = xla::F64;
      return absl::OkStatus();
    case tensorflow::DT_COMPLEX64:
      *type = xla::C64;
      return absl::OkStatus();
    case tensorflow::DT_COMPLEX128:
      *type = xla::C128;
      return absl::OkStatus();
    default:
      return errors::InvalidArgument(
          "Unsupported type in DataTypeToPrimitiveType: '",
          DataTypeString(data_type), "'");
  }
}

absl::StatusOr<DataType> EncodePrimitiveTypeAsDataType(
    xla::PrimitiveType type) {
  static const absl::flat_hash_map<xla::PrimitiveType, DataType>&
      data_type_map = *new absl::flat_hash_map<xla::PrimitiveType, DataType>({
          {xla::PRED, DT_BOOL},
          {xla::F8E5M2, DT_FLOAT8_E5M2},
          {xla::F8E4M3FN, DT_FLOAT8_E4M3FN},
          {xla::BF16, DT_BFLOAT16},
          {xla::F16, DT_HALF},
          {xla::F32, DT_FLOAT},
          {xla::F64, DT_DOUBLE},
          {xla::C64, DT_COMPLEX64},
          {xla::S4, DT_INT4},
          {xla::S8, DT_INT8},
          {xla::S16, DT_INT16},
          {xla::S32, DT_INT32},
          {xla::S64, DT_INT64},
          {xla::U4, DT_UINT4},
          {xla::U8, DT_UINT8},
          {xla::U16, DT_UINT16},
          {xla::U32, DT_UINT32},
          {xla::U64, DT_UINT64},
          {xla::C128, DT_COMPLEX128},
      });

  auto it = data_type_map.find(type);
  if (it == data_type_map.end()) {
    return errors::InvalidArgument(
        "Unsupported type in PrimitiveTypeToDataType ", type);
  }
  return it->second;
}

}  // namespace tensorflow
