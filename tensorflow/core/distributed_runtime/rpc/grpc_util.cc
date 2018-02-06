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

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"

namespace tensorflow {

GrpcByteBufferSource::GrpcByteBufferSource() {}

bool GrpcByteBufferSource::Init(const grpc::ByteBuffer& src) {
  cur_ = -1;
  left_ = 0;
  ptr_ = nullptr;
  byte_count_ = 0;
  bool ok = src.Dump(&slices_).ok();
  if (!ok) {
    slices_.clear();
  }
  return ok;
}

bool GrpcByteBufferSource::Next(const void** data, int* size) {
  // Use loop instead of if in case buffer contained empty slices.
  while (left_ == 0) {
    // Advance to next slice.
    cur_++;
    if (cur_ >= slices_.size()) {
      return false;
    }
    const ::grpc::Slice& s = slices_[cur_];
    left_ = s.size();
    ptr_ = reinterpret_cast<const char*>(s.begin());
  }

  *data = ptr_;
  *size = left_;
  byte_count_ += left_;
  ptr_ += left_;
  left_ = 0;
  return true;
}

void GrpcByteBufferSource::BackUp(int count) {
  ptr_ -= count;
  left_ += count;
  byte_count_ -= count;
}

bool GrpcByteBufferSource::Skip(int count) {
  const void* data;
  int size;
  while (Next(&data, &size)) {
    if (size >= count) {
      BackUp(size - count);
      return true;
    }
    // size < count;
    count -= size;
  }
  // error or we have too large count;
  return false;
}

grpc::protobuf::int64 GrpcByteBufferSource::ByteCount() const {
  return byte_count_;
}

void GrpcMaybeUnparseProto(const protobuf::Message& src,
                           grpc::ByteBuffer* dst) {
  // TODO(sanjay): For bigger protos, serialize into a ZeroCopyOutputStream.
  ::grpc::Slice s(src.ByteSizeLong());
  src.SerializeWithCachedSizesToArray(
      const_cast<uint8*>(reinterpret_cast<const uint8*>(s.begin())));
  ::grpc::ByteBuffer buffer(&s, 1);
  dst->Swap(&buffer);
}

// GrpcMaybeUnparseProto from a string simply copies the string to the
// ByteBuffer.
void GrpcMaybeUnparseProto(const string& src, grpc::ByteBuffer* dst) {
  ::grpc::Slice s(src.data(), src.size());
  ::grpc::ByteBuffer buffer(&s, 1);
  dst->Swap(&buffer);
}

bool GrpcMaybeParseProto(const grpc::ByteBuffer& src, protobuf::Message* dst) {
  GrpcByteBufferSource stream;
  if (!stream.Init(src)) return false;
  return dst->ParseFromZeroCopyStream(&stream);
}

// Overload of GrpcParseProto so we can decode a TensorResponse without
// extra copying.  This overload is used by the RPCState class in
// grpc_state.h.
bool GrpcMaybeParseProto(const ::grpc::ByteBuffer& src, TensorResponse* dst) {
  struct ByteSource : public TensorResponse::Source {
    const ::grpc::ByteBuffer* buffer;
    GrpcByteBufferSource src;
    bool ok;

    ::tensorflow::protobuf::io::ZeroCopyInputStream* contents() override {
      ok = src.Init(*buffer);
      return &src;
    }
  };
  ByteSource bs;
  bs.buffer = &src;
  return dst->ParseFrom(&bs).ok() && bs.ok;
}

// GrpcMaybeParseProto into a string simply copies bytes into the string.
bool GrpcMaybeParseProto(const grpc::ByteBuffer& src, string* dst) {
  dst->clear();
  dst->reserve(src.Length());
  std::vector<::grpc::Slice> slices;
  if (!src.Dump(&slices).ok()) {
    return false;
  }
  for (const ::grpc::Slice& s : slices) {
    dst->append(reinterpret_cast<const char*>(s.begin()), s.size());
  }
  return true;
}

}  // namespace tensorflow
