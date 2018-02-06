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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/impl/codegen/rpc_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/status.h"
#include "grpc++/impl/codegen/stub_options.h"
#include "grpc++/impl/codegen/sync_stream.h"
#include "grpc++/support/byte_buffer.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_serialization_traits.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/protobuf/worker.pb.h"

// Contains potentially large GraphDef.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RegisterGraphRequest);
// Contains potentially large TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RunGraphRequest);
// Contains potentially large StepStats, TensorProto.
TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(tensorflow::RunGraphResponse);

namespace tensorflow {
class GrpcByteSource : public TensorResponse::Source {
 public:
  explicit GrpcByteSource(grpc_byte_buffer* buffer) : buffer_(buffer) {}
  ~GrpcByteSource() override { DeleteStream(); }

  typedef ::grpc::tensorflow_helper::GrpcBufferReader Reader;

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

  grpc_byte_buffer* buffer_;  // Not owned
  Reader* stream_ = nullptr;  // Points into space_ if non-nullptr
  char space_[sizeof(Reader)];
};
}  // namespace tensorflow

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;

// Support parsing/unparsing of tensorflow::TensorResponse.
// Wire-format is identical to RecvTensorResponse.
template <>
class SerializationTraits<tensorflow::TensorResponse>
    : public UnlimitedSizeProtoSerializationTraits<tensorflow::TensorResponse> {
 public:
  static Status Serialize(const tensorflow::TensorResponse& msg,
                          grpc_byte_buffer** bp, bool* own_buffer) {
    LOG(FATAL) << "TODO(sanjay,jeff): Implement";
    return Status();
  }
  static Status Deserialize(grpc_byte_buffer* buffer,
                            tensorflow::TensorResponse* msg,
                            int max_message_size = INT_MAX) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }
    Status result = g_core_codegen_interface->ok();
    if (result.ok()) {
      ::tensorflow::GrpcByteSource source(buffer);
      auto s = msg->ParseFrom(&source);
      if (!s.ok()) {
        result = Status(StatusCode::INTERNAL,
                        ::tensorflow::strings::StrCat(
                            "TensorResponse parse error", s.ToString()));
      }
    }
    g_core_codegen_interface->grpc_byte_buffer_destroy(buffer);
    return result;
  }
};
}  // namespace grpc

namespace tensorflow {

// Names of worker methods.
enum class GrpcWorkerMethod {
  kGetStatus,
  kCreateWorkerSession,
  kDeleteWorkerSession,
  kRegisterGraph,
  kDeregisterGraph,
  kRunGraph,
  kCleanupGraph,
  kCleanupAll,
  kRecvTensor,
  kLogging,
  kTracing,
};
static const int kGrpcNumWorkerMethods =
    static_cast<int>(GrpcWorkerMethod::kTracing) + 1;

const char* GrpcWorkerMethodName(GrpcWorkerMethod id);

namespace grpc {

// Implementation of `tensorflow.WorkerService`, based on the
// definition in "//tensorflow/core/protobuf/worker_service.proto",
// and the gRPC generated stub and service classes.
// See the proto file for the definition of methods and messages.
class WorkerService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();

    // Make RequestAsyncUnary public for grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace grpc

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_IMPL_H_
