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
#include "tensorflow/core/lib/strings/proto_serialization.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    string* result) {
  DCHECK_LE(msg.ByteSizeLong(), static_cast<size_t>(INT_MAX));
  *result = string(msg.ByteSizeLong(), '\0');
  return SerializeToBufferDeterministic(msg, const_cast<char*>(result->data()),
                                        result->size());
}

// As above, but takes a pre-allocated buffer wrapped by result.
// PRECONDITION: result.size() == msg.ByteSizeLong().
bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
                                    char* buffer, int size) {
  DCHECK_LE(msg.ByteSizeLong(), static_cast<size_t>(INT_MAX));
  DCHECK_EQ(msg.ByteSizeLong(), size);
  protobuf::io::ArrayOutputStream array_stream(buffer, size);
  protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  msg.SerializeWithCachedSizes(&output_stream);
  return !output_stream.HadError() && size == output_stream.ByteCount();
}

}  // namespace tensorflow
