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
#ifndef TENSORFLOW_CORE_LIB_STRINGS_PROTO_SERIALIZATION_H_
#define TENSORFLOW_CORE_LIB_STRINGS_PROTO_SERIALIZATION_H_

#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Wrapper around protocol buffer serialization that requests deterministic
// serialization, in particular for Map fields, which serialize in a random
// order by default. Returns true on success.
// Serialization is guaranteed to be deterministic for a given binary only.
// See the following for more details:
// https://github.com/google/protobuf/blob/a1bb147e96b6f74db6cdf3c3fcb00492472dbbfa/src/google/protobuf/io/coded_stream.h#L834
bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    string* result);

// As above, but serialize to a pre-allocated `buffer` of length `size`.
// PRECONDITION: size == msg.ByteSizeLong().
bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
                                    char* buffer, int size);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_STRINGS_PROTO_SERIALIZATION_H_
