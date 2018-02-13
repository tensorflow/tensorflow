/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_CREATE_PROTO_DEBUG_STRING_LIB_H_
#define TENSORFLOW_CORE_UTIL_CREATE_PROTO_DEBUG_STRING_LIB_H_

#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

struct ProtoTextFunctionCode {
  string header;       // for a file named proto_name + ".pb_text.h"
  string header_impl;  // for a file named proto_name + ".pb_text-impl.h"
  string cc;           // for a file named proto_name + ".pb_text.cc"
};

// Returns the generated source code for a proto file descriptor.
//
// <tf_header_prefix> is used as the prefix for #include paths, when including
// tensorflow library headers.
//
// Only works for proto3 messages.
//
// The generated API has, for enums and messages defined in the proto file:
// 1. For each message:
//    * ProtoDebugString(m): same as msg.DebugString(), except that google.any
//      is not expanded.
//    * ProtoShortDebugString(m): same as msg.ShorDebugString(), except that
//      google.any is not expanded.
//    * ProtoParseFromString(s, m): same as TextFormat.ParseFromString(s, &m);
// 2. For each enum:
//    * EnumName_<EnumTypeName>(enum_value): same as <EnumTypeName>(enum_value)
//      in proto.
ProtoTextFunctionCode GetProtoTextFunctionCode(
    const tensorflow::protobuf::FileDescriptor& fd,
    const string& tf_header_prefix);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CREATE_PROTO_DEBUG_STRING_LIB_H_
