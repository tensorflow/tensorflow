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
// Copied from auto-generated gRPC code in order to enable using grpc_call.h
// for raw message handling.
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_GRPC_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_GRPC_H_

#include <functional>

#include "grpcpp/impl/codegen/async_generic_service.h"
#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/client_callback.h"
#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/completion_queue.h"
#include "grpcpp/impl/codegen/method_handler.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/server_callback.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"

#if defined(LIBTPU_ON_GCE)
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache.pb.h"
#else
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache.pb.h"  // copybara"
#endif
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"

namespace tensorflow {
namespace tpu {
namespace grpc {
class TpuCompilationCacheService final {
 public:
  using RequestType = ::tensorflow::tpu::GetTpuProgramRequest;
#if defined(LIBTPU_ON_GCE)
  using ResponseType = ::tensorflow::tpu::GetTpuProgramResponseExternal;
#else
  using ResponseType = ::tensorflow::tpu::GetTpuProgramResponse;
#endif

  // N.B. This must be synchronized with the method order in
  // tpu_compilation_cache.proto.
  enum class MethodId { kGetTpuProgram = 0 };

  static constexpr char const* service_full_name() {
#if defined(LIBTPU_ON_GCE)
    return "tensorflow.tpu.TpuCompilationCacheServiceExternal";
#else
    return "tensorflow.tpu.TpuCompilationCacheService";
#endif
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    // This method requests the cached proto that the TPU execute op has
    // been instructed to execute.
    virtual ::grpc::Status GetTpuProgram(::grpc::ClientContext* context,
                                         const RequestType& request,
                                         ResponseType* response) = 0;
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<ResponseType>>
    AsyncGetTpuProgram(::grpc::ClientContext* context,
                       const RequestType& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReaderInterface<ResponseType>>(
          AsyncGetTpuProgramRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<ResponseType>>
    PrepareAsyncGetTpuProgram(::grpc::ClientContext* context,
                              const RequestType& request,
                              ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReaderInterface<ResponseType>>(
          PrepareAsyncGetTpuProgramRaw(context, request, cq));
    }

   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface<ResponseType>*
    AsyncGetTpuProgramRaw(::grpc::ClientContext* context,
                          const RequestType& request,
                          ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<ResponseType>*
    PrepareAsyncGetTpuProgramRaw(::grpc::ClientContext* context,
                                 const RequestType& request,
                                 ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    explicit Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    ::grpc::Status GetTpuProgram(::grpc::ClientContext* context,
                                 const RequestType& request,
                                 ResponseType* response) override;
    std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>
    AsyncGetTpuProgram(::grpc::ClientContext* context,
                       const RequestType& request,
                       ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>(
          AsyncGetTpuProgramRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>
    PrepareAsyncGetTpuProgram(::grpc::ClientContext* context,
                              const RequestType& request,
                              ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReader<ResponseType>>(
          PrepareAsyncGetTpuProgramRaw(context, request, cq));
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader<ResponseType>* AsyncGetTpuProgramRaw(
        ::grpc::ClientContext* context, const RequestType& request,
        ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader<ResponseType>*
    PrepareAsyncGetTpuProgramRaw(::grpc::ClientContext* context,
                                 const RequestType& request,
                                 ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_get_tpu_program_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    ~Service() override;
    // This method requests the cached proto that the TPU execute op has
    // been instructed to execute.
    virtual ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                         const RequestType* request,
                                         ResponseType* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_GetTpuProgram : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* service) {}

   public:
    WithAsyncMethod_GetTpuProgram() { ::grpc::Service::MarkMethodAsync(0); }
    ~WithAsyncMethod_GetTpuProgram() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                 const RequestType* request,
                                 ResponseType* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetTpuProgram(
        ::grpc::ServerContext* context, RequestType* request,
        ::grpc::ServerAsyncResponseWriter<ResponseType>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }

    // Make RequestAsyncUnary accessible to grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
  typedef WithAsyncMethod_GetTpuProgram<Service> AsyncService;
  template <class BaseClass>
  class WithGenericMethod_GetTpuProgram : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* service) {}

   public:
    WithGenericMethod_GetTpuProgram() { ::grpc::Service::MarkMethodGeneric(0); }
    ~WithGenericMethod_GetTpuProgram() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                 const RequestType* request,
                                 ResponseType* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GetTpuProgram : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* service) {}

   public:
    WithStreamedUnaryMethod_GetTpuProgram() {
      ::grpc::Service::MarkMethodStreamed(
          0,
          new ::grpc::internal::StreamedUnaryHandler<RequestType, ResponseType>(
              std::bind(&WithStreamedUnaryMethod_GetTpuProgram<
                            BaseClass>::StreamedGetTpuProgram,
                        this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_GetTpuProgram() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetTpuProgram(::grpc::ServerContext* context,
                                 const RequestType* request,
                                 ResponseType* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGetTpuProgram(
        ::grpc::ServerContext* context,
        ::grpc::ServerUnaryStreamer<RequestType, ResponseType>*
            server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_GetTpuProgram<Service> StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_GetTpuProgram<Service> StreamedService;
};
}  // namespace grpc
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_GRPC_H_
