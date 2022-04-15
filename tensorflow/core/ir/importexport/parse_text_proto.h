/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_PARSE_TEXT_PROTO_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_PARSE_TEXT_PROTO_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace mlir {
namespace tfg {

// Sets output to the given input with `prefix` stripped, or returns an error if
// the prefix doesn't exist.
tensorflow::Status ConsumePrefix(absl::string_view str,
                                 absl::string_view prefix,
                                 absl::string_view* output);

// Strips `prefix_to_strip` from `text_proto`, parses, and returns the parsed
// proto.
tensorflow::Status ParseTextProto(absl::string_view text_proto,
                                  absl::string_view prefix_to_strip,
                                  tensorflow::protobuf::Message* parsed_proto);
inline tensorflow::Status ParseTextProto(
    absl::string_view /* text_proto */, absl::string_view /* prefix_to_strip */,
    tensorflow::protobuf::MessageLite* /* parsed_proto */) {
  return tensorflow::errors::Unavailable("Cannot parse text protos on mobile.");
}

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_PARSE_TEXT_PROTO_H_
