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

#include "tensorflow/core/util/proto/proto_utils.h"

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

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

Status ParseTextFormatFromString(absl::string_view input,
                                 protobuf::Message* output) {
  DCHECK(output != nullptr) << "output must be non NULL";
  // When checks are disabled, instead log the error and return an error status.
  if (output == nullptr) {
    LOG(ERROR) << "output must be non NULL";
    return Status(absl::StatusCode::kInvalidArgument,
                  "output must be non NULL");
  }
  string err;
  StringErrorCollector err_collector(&err, /*one-indexing=*/true);
  protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&err_collector);
  if (!parser.ParseFromString(string(input), output)) {
    return Status(absl::StatusCode::kInvalidArgument, err);
  }
  return OkStatus();
}

StringErrorCollector::StringErrorCollector(string* error_text)
    : StringErrorCollector(error_text, false) {}

StringErrorCollector::StringErrorCollector(string* error_text,
                                           bool one_indexing)
    : error_text_(error_text), index_offset_(one_indexing ? 1 : 0) {
  DCHECK(error_text_ != nullptr) << "error_text must be non NULL";
  // When checks are disabled, just log and then ignore added errors/warnings.
  if (error_text_ == nullptr) {
    LOG(ERROR) << "error_text must be non NULL";
  }
}

void StringErrorCollector::AddError(int line, int column,
                                    const string& message) {
  if (error_text_ != nullptr) {
    absl::SubstituteAndAppend(error_text_, "$0($1): $2\n", line + index_offset_,
                              column + index_offset_, message);
  }
}

void StringErrorCollector::AddWarning(int line, int column,
                                      const string& message) {
  AddError(line, column, message);
}

}  // namespace proto_utils
}  // namespace tensorflow
