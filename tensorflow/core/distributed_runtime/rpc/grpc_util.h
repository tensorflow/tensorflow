/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_

#include <memory>
#include <string>

#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/support/byte_buffer.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::FromGrpcStatus;
using tsl::SharedGrpcChannelPtr;
using tsl::ToGrpcStatus;
// NOLINTEND(misc-unused-using-decls)

// Thin wrapper around ::grpc::ProtoBufferReader to give TensorResponse
// an efficient byte reader from which to decode a RecvTensorResponse.
class GrpcByteSource : public TensorResponse::Source {
 public:
  explicit GrpcByteSource(::grpc::ByteBuffer* buffer) : buffer_(buffer) {}
  ~GrpcByteSource() override { DeleteStream(); }

  typedef ::grpc::ProtoBufferReader Reader;

  protobuf::io::ZeroCopyInputStream* contents() override {
    DeleteStream();
    stream_ = new (&space_) Reader(buffer_);
    return stream_;
  }

 private:
  void DeleteStream() {
    if (stream_) {
      stream_->~Reader();
    }
  }

  ::grpc::ByteBuffer* buffer_;  // Not owned
  Reader* stream_ = nullptr;    // Points into space_ if non-nullptr
  char space_[sizeof(Reader)];
};

inline string GrpcIdKey() { return "tf-rpc"; }

// Decode a TensorResponse without extra copying. This function is an optimized
// variant of tsl::GrpcMaybeParseProto.
bool GrpcMaybeParseTensorResponse(::grpc::ByteBuffer* src, TensorResponse* dst);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
