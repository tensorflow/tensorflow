/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_PROTOBUF_INTERNAL_H_
#define TENSORFLOW_PLATFORM_PROTOBUF_INTERNAL_H_

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Returns the DebugString when available, or a stub message otherwise. Useful
// for messages that are incompatible with proto_text (e.g. those using Any).
#ifdef TENSORFLOW_LITE_PROTOS
template <class T>
string DebugStringIfAvailable(T proto) {
  return "[DebugString not available with lite protos]";
}
#else
template <class T>
auto DebugStringIfAvailable(T proto) -> decltype(proto.DebugString()) {
  return proto.DebugString();
}
#endif  // defined(TENSORFLOW_LITE_PROTOS)

// Utility for parsing an Any value with full or lite protos.
template <class T>
Status ParseAny(const google::protobuf::Any& any, T* message,
                const string& type_name) {
#ifdef TENSORFLOW_LITE_PROTOS
  if (any.type_url() != strings::StrCat("type.googleapis.com/", type_name)) {
    return errors::FailedPrecondition(
        "Expected Any type_url for: ", type_name, ". Got: ",
        string(any.type_url().data(), any.type_url().size()), ".");
  }
  if (!message->ParseFromString(any.value())) {
    return errors::FailedPrecondition("Failed to unpack: ",
                                      DebugStringIfAvailable(any));
  }
#else
  CHECK_EQ(type_name, message->descriptor()->full_name());
  if (!any.Is<T>()) {
    return errors::FailedPrecondition(
        "Expected Any type_url for: ", message->descriptor()->full_name(),
        ". Got: ", string(any.type_url().data(), any.type_url().size()), ".");
  }
  if (!any.UnpackTo(message)) {
    return errors::FailedPrecondition("Failed to unpack: ",
                                      DebugStringIfAvailable(any));
  }
#endif
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_PROTOBUF_INTERNAL_H_
