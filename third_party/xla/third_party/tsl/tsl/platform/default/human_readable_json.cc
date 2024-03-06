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

#include "tsl/platform/human_readable_json.h"

#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/strcat.h"

namespace tsl {

absl::Status ProtoToHumanReadableJson(const protobuf::Message& proto,
                                      string* result,
                                      bool ignore_accuracy_loss) {
  result->clear();

  protobuf::util::JsonPrintOptions json_options;
  json_options.preserve_proto_field_names = true;
  json_options.always_print_primitive_fields = true;
  auto status =
      protobuf::util::MessageToJsonString(proto, result, json_options);
  if (!status.ok()) {
    // Convert error_msg google::protobuf::StringPiece to
    // tsl::StringPiece.
    auto error_msg = status.message();
    return errors::Internal(
        strings::StrCat("Could not convert proto to JSON string: ",
                        StringPiece(error_msg.data(), error_msg.length())));
  }
  return absl::OkStatus();
}

absl::Status ProtoToHumanReadableJson(const protobuf::MessageLite& proto,
                                      string* result,
                                      bool ignore_accuracy_loss) {
  *result = "[human readable output not available for lite protos]";
  return absl::OkStatus();
}

absl::Status HumanReadableJsonToProto(const string& str,
                                      protobuf::Message* proto) {
  proto->Clear();
  auto status = protobuf::util::JsonStringToMessage(str, proto);
  if (!status.ok()) {
    // Convert error_msg google::protobuf::StringPiece to
    // tsl::StringPiece.
    auto error_msg = status.message();
    return errors::Internal(
        strings::StrCat("Could not convert JSON string to proto: ",
                        StringPiece(error_msg.data(), error_msg.length())));
  }
  return absl::OkStatus();
}

absl::Status HumanReadableJsonToProto(const string& str,
                                      protobuf::MessageLite* proto) {
  return errors::Internal("Cannot parse JSON protos on Android");
}

}  // namespace tsl
