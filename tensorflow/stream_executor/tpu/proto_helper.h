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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_PROTO_HELPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_PROTO_HELPER_H_

#include <cstddef>

#include "tensorflow/core/platform/logging.h"

extern "C" {

typedef struct TpuSerializedProto {
  const char* bytes;
  size_t size;
} TpuSerializedProto;

void StreamExecutor_Tpu_FreeSerializedProto(const TpuSerializedProto* proto);

}  // extern "C"

namespace stream_executor {
namespace tpu {

using SerializedProto = TpuSerializedProto;

// Serializes a proto and put the result in the given SerializedProto* argument.
//
// Users should call SerializedProto_Free on `serialized_proto` afterwards.
template <class Proto>
inline void SerializeProto(const Proto& proto,
                           SerializedProto* serialized_proto) {
  auto size = proto.ByteSizeLong();
  auto bytes = new char[size];
  CHECK(proto.SerializeToArray(bytes, size));
  serialized_proto->size = size;
  serialized_proto->bytes = bytes;
}

// Serializes a proto and return the result as a SerializedProto value.
//
// Users should call SerializedProto_Free on the return value afterwards.
template <class Proto>
inline SerializedProto SerializeProto(const Proto& proto) {
  SerializedProto serialized_proto;
  SerializeProto(proto, &serialized_proto);
  return serialized_proto;
}

// Deserializes a buffer and return the corresponding proto. If the buffer is
// empty, return an empty proto.
template <class Proto>
inline Proto DeserializeProto(const SerializedProto& serialized_proto) {
  Proto proto;
  if (serialized_proto.bytes != nullptr) {
    CHECK_GT(serialized_proto.size, 0);
    CHECK(proto.ParseFromArray(serialized_proto.bytes, serialized_proto.size))
        << "Invalid buffer, failed to deserialize buffer.";
  }
  return proto;
}

// Releases the memory allocated for serialized protos.
inline void SerializedProto_Free(const SerializedProto& serialized_proto) {
  CHECK_NE(serialized_proto.bytes, nullptr);
  CHECK_GT(serialized_proto.size, 0);
  delete[] serialized_proto.bytes;
}

}  // namespace tpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_PROTO_HELPER_H_
