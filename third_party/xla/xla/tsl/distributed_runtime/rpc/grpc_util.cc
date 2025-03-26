/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"

#include <algorithm>
#include <vector>

#include "grpcpp/impl/codegen/proto_utils.h"
#include "tsl/platform/protobuf.h"

namespace tsl {

::grpc::Status GrpcMaybeUnparseProto(const protobuf::Message& src,
                                     grpc::ByteBuffer* dst) {
  bool own_buffer;
  // grpc::ProtoBufferWriter
  return ::grpc::SerializationTraits<protobuf::Message>::Serialize(src, dst,
                                                                   &own_buffer);
}

bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, protobuf::Message* dst) {
  // grpc::ProtoBufferReader
  return ::grpc::SerializationTraits<protobuf::Message>::Deserialize(src, dst)
      .ok();
}

// GrpcMaybeUnparseProto from a string simply copies the string to the
// ByteBuffer.
::grpc::Status GrpcMaybeUnparseProto(const string& src, grpc::ByteBuffer* dst) {
  ::grpc::Slice s(src.data(), src.size());
  ::grpc::ByteBuffer buffer(&s, 1);
  dst->Swap(&buffer);
  return ::grpc::Status::OK;
}

// GrpcMaybeParseProto simply copies bytes into the string.
bool GrpcMaybeParseProto(grpc::ByteBuffer* src, string* dst) {
  dst->clear();
  dst->reserve(src->Length());
  std::vector<::grpc::Slice> slices;
  if (!src->Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}

// GrpcMaybeParseProto simply copies bytes into the tstring.
bool GrpcMaybeParseProto(grpc::ByteBuffer* src, tstring* dst) {
  dst->clear();
  dst->reserve(src->Length());
  std::vector<::grpc::Slice> slices;
  if (!src->Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}
}  // namespace tsl
