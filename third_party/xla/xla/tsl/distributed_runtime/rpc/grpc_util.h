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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/support/byte_buffer.h"
#include "xla/tsl/protobuf/distributed_runtime_payloads.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/stringpiece.h"
#include "tsl/platform/stringprintf.h"

namespace tsl {

// Proto: tensorflow::distributed_runtime::GrpcPayloadsLost
// Location: tsl/protobuf/distributed_runtime_payloads.proto
// Usage: Flags the absl::Status to have lost payloads during GRPC conversion.
constexpr char kGrpcPayloadsLost[] =
    "type.googleapis.com/tensorflow.distributed_runtime.GrpcPayloadsLost";

constexpr char kStreamRemovedMessage[] = "Stream removed";

// Identify if the given grpc::Status corresponds to an HTTP stream removed
// error (see chttp2_transport.cc).
//
// When auto-reconnecting to a remote worker after it restarts, gRPC can return
// an UNKNOWN error code with a "Stream removed" error message. This should not
// be treated as an unrecoverable error.
//
// N.B. This is dependent on the error message from grpc remaining consistent.
inline bool IsStreamRemovedError(const ::grpc::Status& s) {
  return !s.ok() && s.error_code() == ::grpc::StatusCode::UNKNOWN &&
         s.error_message() == kStreamRemovedMessage;
}

inline std::string SerializePayloads(const absl::Status& s) {
  tensorflow::distributed_runtime::GrpcPayloadContainer container;
  s.ForEachPayload(
      [&container](absl::string_view key, const absl::Cord& value) {
        (*container.mutable_payloads())[std::string(key)] = std::string(value);
      });
  return container.SerializeAsString();
}

inline void InsertSerializedPayloads(absl::Status& s, std::string payloads) {
  tensorflow::distributed_runtime::GrpcPayloadContainer container;
  if (container.ParseFromString(payloads)) {
    for (const auto& key_val : container.payloads()) {
      s.SetPayload(key_val.first, absl::Cord(key_val.second));
    }
  } else {
    s.SetPayload(kGrpcPayloadsLost,
                 absl::Cord(tensorflow::distributed_runtime::GrpcPayloadsLost()
                                .SerializeAsString()));
  }
}

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return absl::OkStatus();
  } else {
    absl::Status converted;
    // Convert "UNKNOWN" stream removed errors into unavailable, to allow
    // for retry upstream.
    if (IsStreamRemovedError(s)) {
      converted =
          absl::Status(absl::StatusCode::kUnavailable, s.error_message());
    }
    converted = absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                             s.error_message());
    InsertSerializedPayloads(converted, s.error_details());
    return converted;
  }
}

inline ::grpc::Status ToGrpcStatus(const absl::Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  } else {
    if (s.message().size() > 3072 /* 3k bytes */) {
      // TODO(b/62947679): Remove truncation once the gRPC issue is resolved.
      string scratch = strings::Printf("%.3072s ... [truncated]",
                                       absl::StatusMessageAsCStr(s));
      LOG(ERROR) << "Truncated error message: " << s;
      return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()), scratch,
                            SerializePayloads(s));
    }
    return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                          std::string(s.message()), SerializePayloads(s));
  }
}

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;

// Serialize src and store in *dst.
::grpc::Status GrpcMaybeUnparseProto(const protobuf::Message& src,
                                     ::grpc::ByteBuffer* dst);

// Parse contents of src and initialize *dst with them.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, protobuf::Message* dst);

// Copy string src to grpc buffer *dst.
::grpc::Status GrpcMaybeUnparseProto(const string& src,
                                     ::grpc::ByteBuffer* dst);

// Copy grpc buffer src to string *dst.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, string* dst);

// Copy grpc buffer src to tstring *dst.
bool GrpcMaybeParseProto(::grpc::ByteBuffer* src, tstring* dst);
}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_UTIL_H_
