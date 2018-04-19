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

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/support/byte_buffer.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

constexpr char kStreamRemovedMessage[] = "Stream removed";

// Identify if the given grpc::Status corresponds to an HTTP stream removed
// error (see chttp2_transport.cc).
//
// When auto-reconnecting to a remote TensorFlow worker after it restarts, gRPC
// can return an UNKNOWN error code with a "Stream removed" error message.
// This should not be treated as an unrecoverable error.
//
// N.B. This is dependent on the error message from grpc remaining consistent.
inline bool IsStreamRemovedError(const ::grpc::Status& s) {
  return !s.ok() && s.error_code() == ::grpc::StatusCode::UNKNOWN &&
         s.error_message() == kStreamRemovedMessage;
}

inline Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return Status::OK();
  } else {
    // Convert "UNKNOWN" stream removed errors into unavailable, to allow
    // for retry upstream.
    if (IsStreamRemovedError(s)) {
      return Status(tensorflow::error::UNAVAILABLE, s.error_message());
    }
    return Status(static_cast<tensorflow::error::Code>(s.error_code()),
                  s.error_message());
  }
}

inline ::grpc::Status ToGrpcStatus(const ::tensorflow::Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    if (s.error_message().size() > 3072 /* 3k bytes */) {
      // TODO(b/62947679): Remove truncation once the gRPC issue is resolved.
      string scratch =
          strings::Printf("%.3072s ... [truncated]", s.error_message().c_str());
      LOG(ERROR) << "Truncated error message: " << s;
      return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()), scratch);
    }
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          s.error_message());
  }
}

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

inline string GrpcIdKey() { return "tf-rpc"; }

// Serialize src and store in *dst.
void GrpcMaybeUnparseProto(const protobuf::Message& src,
                           ::grpc::ByteBuffer* dst);

// Parse contents of src and initialize *dst with them.
bool GrpcMaybeParseProto(const ::grpc::ByteBuffer& src, protobuf::Message* dst);

// Specialization for TensorResponse
bool GrpcMaybeParseProto(const ::grpc::ByteBuffer& src, TensorResponse* dst);

// Copy string src to grpc buffer *dst.
void GrpcMaybeUnparseProto(const string& src, ::grpc::ByteBuffer* dst);

// Copy grpc buffer src to string *dst.
bool GrpcMaybeParseProto(const ::grpc::ByteBuffer& src, string* dst);

// A ZeroCopyInputStream that reads from a grpc::ByteBuffer.
class GrpcByteBufferSource : public ::grpc::protobuf::io::ZeroCopyInputStream {
 public:
  GrpcByteBufferSource();
  bool Init(const ::grpc::ByteBuffer& src);  // Can be called multiple times.
  bool Next(const void** data, int* size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  ::grpc::protobuf::int64 ByteCount() const override;

 private:
  std::vector<::grpc::Slice> slices_;
  int cur_;          // Current slice index.
  int left_;         // Number of bytes in slices_[cur_] left to yield.
  const char* ptr_;  // Address of next byte in slices_[cur_] to yield.
  ::grpc::protobuf::int64 byte_count_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
