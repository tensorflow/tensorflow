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

#ifndef TENSORFLOW_CORE_PLATFORM_PROTOBUF_INTERNAL_H_
#define TENSORFLOW_CORE_PLATFORM_PROTOBUF_INTERNAL_H_

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Utility for parsing an Any value with full or lite protos.
template <class T>
Status ParseAny(const google::protobuf::Any& any, T* message,
                const string& type_name) {
  CHECK_EQ(type_name, message->GetTypeName());
  if (!any.Is<T>()) {
    return errors::FailedPrecondition(
        "Expected Any type_url for: ", message->GetTypeName(),
        ". Got: ", string(any.type_url().data(), any.type_url().size()), ".");
  }
  if (!any.UnpackTo(message)) {
    return errors::FailedPrecondition("Failed to unpack: ", any.DebugString());
  }
  return OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROTOBUF_INTERNAL_H_
