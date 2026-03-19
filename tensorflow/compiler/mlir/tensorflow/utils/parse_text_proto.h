/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_PARSE_TEXT_PROTO_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_PARSE_TEXT_PROTO_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Sets output to the given input with `prefix` stripped, or returns an error if
// the prefix doesn't exist.
absl::Status ConsumePrefix(absl::string_view str, absl::string_view prefix,
                           absl::string_view* output);

// Strips `prefix_to_strip` from `text_proto`, parses, and returns the parsed
// proto.
absl::Status ParseTextProto(absl::string_view text_proto,
                            absl::string_view prefix_to_strip,
                            protobuf::Message* parsed_proto);
inline absl::Status ParseTextProto(absl::string_view /* text_proto */,
                                   absl::string_view /* prefix_to_strip */,
                                   protobuf::MessageLite* /* parsed_proto */) {
  return errors::Unavailable("Cannot parse text protos on mobile.");
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_PARSE_TEXT_PROTO_H_
