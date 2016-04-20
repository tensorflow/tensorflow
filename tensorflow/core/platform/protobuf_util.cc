/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

bool ParseProtoUnlimited(protobuf::Message* proto, const string& serialized) {
  return ParseProtoUnlimited(proto, serialized.data(), serialized.size());
}

bool ParseProtoUnlimited(protobuf::Message* proto, const void* serialized,
                         size_t size) {
  protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8*>(serialized), size);
  coded_stream.SetTotalBytesLimit(INT_MAX, INT_MAX);
  return proto->ParseFromCodedStream(&coded_stream);
}

}  // namespace tensorflow
