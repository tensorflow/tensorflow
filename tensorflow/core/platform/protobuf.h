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

#ifndef TENSORFLOW_PLATFORM_PROTOBUF_H_
#define TENSORFLOW_PLATFORM_PROTOBUF_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

// Import whatever namespace protobuf comes from into the
// ::tensorflow::protobuf namespace.
//
// TensorFlow code should use the ::tensorflow::protobuf namespace to
// refer to all protobuf APIs.

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/build_config/protobuf.h"
#else
#include "tensorflow/core/platform/default/protobuf.h"
#endif

namespace tensorflow {
// Parses a protocol buffer contained in a string in the binary wire format.
// Returns true on success. Note: Unlike protobuf's builtin ParseFromString,
// this function has no size restrictions on the total size of the encoded
// protocol buffer.
bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                         const string& serialized);
bool ParseProtoUnlimited(protobuf::MessageLite* proto, const void* serialized,
                         size_t size);

// Returns the string value for the value of a string or bytes protobuf field.
inline const string& ProtobufStringToString(const string& s) { return s; }

// Set <dest> to <src>. Swapping is allowed, as <src> does not need to be
// preserved.
inline void SetProtobufStringSwapAllowed(string* src, string* dest) {
  dest->swap(*src);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_PROTOBUF_H_
