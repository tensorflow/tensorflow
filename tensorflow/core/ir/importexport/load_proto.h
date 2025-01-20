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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_LOAD_PROTO_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_LOAD_PROTO_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Reads text (.pbtext) or binary (.pb) format of a proto message from the given
// buffer. Returns error status of the file is not found or malformed proto.
// Note that text protos can only be parsed when full protobuf::Message protos
// are used, and will fail for protobuf::MessageLite protos.
absl::Status LoadProtoFromBuffer(absl::string_view input,
                                 protobuf::Message* proto);
absl::Status LoadProtoFromBuffer(absl::string_view input,
                                 protobuf::MessageLite* proto);

// Reads text (.pbtext) or binary (.pb) format of a proto message from the given
// file path. Returns error status of the file is not found or malformed proto.
// Note that text protos can only be parsed when full protobuf::Message protos
// are used, and will fail for protobuf::MessageLite protos.
absl::Status LoadProtoFromFile(absl::string_view input_filename,
                               protobuf::Message* proto);
absl::Status LoadProtoFromFile(absl::string_view input_filename,
                               protobuf::MessageLite* proto);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_LOAD_PROTO_H_
