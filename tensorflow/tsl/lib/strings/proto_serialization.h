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
#ifndef TENSORFLOW_TSL_LIB_STRINGS_PROTO_SERIALIZATION_H_
#define TENSORFLOW_TSL_LIB_STRINGS_PROTO_SERIALIZATION_H_

#include "tensorflow/tsl/platform/protobuf.h"

namespace tsl {

// Wrapper around protocol buffer serialization that requests deterministic
// serialization, in particular for Map fields, which serialize in a random
// order by default. Returns true on success.
// Serialization is guaranteed to be deterministic for a given binary only.
// See the following for more details:
// https://github.com/google/protobuf/blob/a1bb147e96b6f74db6cdf3c3fcb00492472dbbfa/src/google/protobuf/io/coded_stream.h#L834
bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    string* result);

// As above, but takes a pre-allocated buffer wrapped by result.
// PRECONDITION: size == msg.ByteSizeLong() && size <= INT_MAX.
bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
                                    char* buffer, size_t size);

// Returns true if serializing x and y using
// SerializeToBufferDeterministic() yields identical strings.
bool AreSerializedProtosEqual(const protobuf::MessageLite& x,
                              const protobuf::MessageLite& y);

// Computes Hash64 of the output of SerializeToBufferDeterministic().
uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto);
uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto,
                                uint64 seed);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_STRINGS_PROTO_SERIALIZATION_H_
