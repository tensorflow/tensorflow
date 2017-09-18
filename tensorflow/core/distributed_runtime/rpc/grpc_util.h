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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_

#include <memory>

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/support/byte_buffer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

inline Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return Status::OK();
  } else {
    return Status(static_cast<tensorflow::error::Code>(s.error_code()),
                  s.error_message());
  }
}

inline ::grpc::Status ToGrpcStatus(const ::tensorflow::Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          s.error_message());
  }
}

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

inline string GrpcIdKey() { return "tf-rpc"; }

// Serialize src and store in *dst.
void GrpcUnparseProto(const protobuf::Message& src, ::grpc::ByteBuffer* dst);

// Parse contents of src and initialize *dst with them.
bool GrpcParseProto(const ::grpc::ByteBuffer& src, protobuf::Message* dst);

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

// GrpcCounter is used to delay shutdown until all active RPCs are done.
class GrpcCounter {
 public:
  GrpcCounter() {}

  GrpcCounter(const GrpcCounter&) = delete;
  GrpcCounter& operator=(const GrpcCounter&) = delete;

  // Increment the count of live RPCs.
  void Increment();

  // Decrement the count of live RPCs.
  void Decrement();

  // Wait until count of live RPCs is zero.
  void WaitUntilUnused();

 private:
  mutex mu_;
  condition_variable empty_;
  int counter_ = 0;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
