/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/protobuf.h"

#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow {
namespace proto_utils {

using tensorflow::protobuf::FieldDescriptor;
using tensorflow::protobuf::internal::WireFormatLite;

bool IsCompatibleType(FieldDescriptor::Type field_type, DataType dtype) {
  switch (field_type) {
    case WireFormatLite::TYPE_DOUBLE:
      return dtype == tensorflow::DT_DOUBLE;
    case WireFormatLite::TYPE_FLOAT:
      return dtype == tensorflow::DT_FLOAT || dtype == tensorflow::DT_DOUBLE;
    case WireFormatLite::TYPE_INT64:
      return dtype == tensorflow::DT_INT64;
    case WireFormatLite::TYPE_UINT64:
      return dtype == tensorflow::DT_UINT64;
    case WireFormatLite::TYPE_INT32:
      return dtype == tensorflow::DT_INT32 || dtype == tensorflow::DT_INT64;
    case WireFormatLite::TYPE_FIXED64:
      return dtype == tensorflow::DT_UINT64;
    case WireFormatLite::TYPE_FIXED32:
      return dtype == tensorflow::DT_UINT32 || dtype == tensorflow::DT_UINT64;
    case WireFormatLite::TYPE_BOOL:
      return dtype == tensorflow::DT_BOOL;
    case WireFormatLite::TYPE_STRING:
      return dtype == tensorflow::DT_STRING;
    case WireFormatLite::TYPE_GROUP:
      return dtype == tensorflow::DT_STRING;
    case WireFormatLite::TYPE_MESSAGE:
      return dtype == tensorflow::DT_STRING;
    case WireFormatLite::TYPE_BYTES:
      return dtype == tensorflow::DT_STRING;
    case WireFormatLite::TYPE_UINT32:
      return dtype == tensorflow::DT_UINT32 || dtype == tensorflow::DT_UINT64;
    case WireFormatLite::TYPE_ENUM:
      return dtype == tensorflow::DT_INT32;
    case WireFormatLite::TYPE_SFIXED32:
      return dtype == tensorflow::DT_INT32 || dtype == tensorflow::DT_INT64;
    case WireFormatLite::TYPE_SFIXED64:
      return dtype == tensorflow::DT_INT64;
    case WireFormatLite::TYPE_SINT32:
      return dtype == tensorflow::DT_INT32 || dtype == tensorflow::DT_INT64;
    case WireFormatLite::TYPE_SINT64:
      return dtype == tensorflow::DT_INT64;
      // default: intentionally omitted in order to enable static checking.
  }
}

}  // namespace proto_utils
}  // namespace tensorflow
