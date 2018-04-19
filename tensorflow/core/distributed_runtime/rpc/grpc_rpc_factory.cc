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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/util/rpc/call_container.h"
#include "tensorflow/core/util/rpc/rpc_factory.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_rpc_factory.h"

namespace tensorflow {

namespace {
class GrpcCall {
 public:
  explicit GrpcCall(CallContainer<GrpcCall>* container, int index, bool try_rpc,
                    const string* request_msg, string* response_msg,
                    int32* status_code, string* status_message)
      : container_(container),
        index_(index),
        try_rpc_(try_rpc),
        request_msg_(request_msg),
        response_msg_(response_msg),
        status_code_(status_code),
        status_message_(status_message) {}

  void StartCancel() { call_opts_.StartCancel(); }

  void Done(const Status& s) {
    DCHECK(container_ != nullptr);
    if (!s.ok() && try_rpc_) {
      DCHECK(status_code_ != nullptr);
      DCHECK(status_message_ != nullptr);
      *status_code_ = s.code();
      *status_message_ = s.error_message();
    }
    container_->Done(s, index_);
  }

  const string& request() const { return *request_msg_; }
  string* response() const { return response_msg_; }
  CallOptions* call_opts() { return &call_opts_; }

 private:
  CallContainer<GrpcCall>* const container_;
  const int index_;
  bool try_rpc_;
  CallOptions call_opts_;
  const string* request_msg_;
  string* response_msg_;
  int* status_code_;
  string* status_message_;
};

}  // namespace

GrpcRPCFactory::GrpcRPCFactory(OpKernelConstruction* ctx, bool fail_fast,
                               int64 timeout_in_ms)
    : RPCFactory(), fail_fast_(fail_fast), timeout_in_ms_(timeout_in_ms) {
  // TODO(ebrevdo): Investigate possible performance improvements by
  // replacing this thread with a threadpool.
  polling_thread_ =
      ctx->env()->StartThread(ThreadOptions(), "rpc_op_grpc_factory", [this]() {
        void* tag;
        bool ok;
        while (completion_queue_.Next(&tag, &ok)) {
          GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
          callback_tag->OnCompleted(ok);
        }
      });
}

GrpcRPCFactory::~GrpcRPCFactory() {
  // The amount of time we wait depends on several parameters, including:
  //   - the value of the fail_fast attribute.
  //   - the timeout option of the rpc call in the proto declaration.
  //   - the network roundtrip time and service's execution time.
  //
  // If a connection is made but the service doesn't ever respond, and
  // there is no timeout option set for this rpc call, then it is
  // possible the RPC request will wait forever.
  //
  completion_queue_.Shutdown();
  delete polling_thread_;
}

void GrpcRPCFactory::Call(OpKernelContext* ctx, int64 num_elements,
                          const Tensor& address_t, const Tensor& method_t,
                          const Tensor& request_t, const bool try_rpc,
                          Tensor* response_t, Tensor* status_code_t,
                          Tensor* status_message_t,
                          AsyncOpKernel::DoneCallback done) {
  auto address = address_t.flat<string>();
  auto method = method_t.flat<string>();
  auto request = request_t.flat<string>();

  // Stubs are maintained by the GrpcRPCFactory class and will be
  // deleted when the class is destroyed.
  ::grpc::GenericStub* singleton_stub = nullptr;
  if (address.size() == 1) {
    singleton_stub = GetOrCreateStubForAddress(address(0));
  }
  auto get_stub = [&address, this,
                   singleton_stub](int64 ix) -> ::grpc::GenericStub* {
    return (address.size() > 1) ? GetOrCreateStubForAddress(address(ix))
                                : singleton_stub;
  };
  auto get_method_ptr = [&method](int64 ix) -> const string* {
    return (method.size() > 1) ? &(method(ix)) : &(method(0));
  };
  auto get_request_ptr = [&request](int64 ix) -> const string* {
    return (request.size() > 1) ? &(request(ix)) : &(request(0));
  };

  if (try_rpc) {
    // In this case status_code will never be set in the response,
    // so we just set it to OK.
    DCHECK(status_code_t != nullptr);
    status_code_t->flat<int32>().setConstant(
        static_cast<int>(errors::Code::OK));
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken cancellation_token = cm->get_cancellation_token();

  // This object will delete itself when done.
  auto* container =
      new CallContainer<GrpcCall>(ctx, num_elements, fail_fast_, try_rpc,
                                  std::move(done), cancellation_token);

  auto response = response_t->flat<string>();
  int32* status_code_ptr = nullptr;
  string* status_message_ptr = nullptr;
  if (try_rpc) {
    status_code_ptr = status_code_t->flat<int32>().data();
    status_message_ptr = status_message_t->flat<string>().data();
  }
  for (int i = 0; i < num_elements; ++i) {
    container->calls()->emplace_back(
        container, i, try_rpc, get_request_ptr(i), &response(i),
        (try_rpc) ? &status_code_ptr[i] : nullptr,
        (try_rpc) ? &status_message_ptr[i] : nullptr);
  }

  int i = 0;
  for (GrpcCall& call : *(container->calls())) {
    // This object will delete itself when done.
    new RPCState<string>(get_stub(i), &completion_queue_, *get_method_ptr(i),
                         call.request(), call.response(),
                         /*done=*/[&call](const Status& s) { call.Done(s); },
                         call.call_opts(), fail_fast_, timeout_in_ms_);
    ++i;
  }

  // Need to register this callback after all the RPCs are in
  // flight; otherwise we may try to cancel an RPC *before* it
  // launches, which is a no-op, and then fall into a deadlock.
  bool is_cancelled = !cm->RegisterCallback(
      cancellation_token, [container]() { container->StartCancel(); });

  if (is_cancelled) {
    ctx->SetStatus(errors::Cancelled("Operation has been cancelled."));
    // container's reference counter will take care of calling done().
    container->StartCancel();
  }
}

::grpc::GenericStub* GrpcRPCFactory::GetOrCreateStubForAddress(
    const string& address) {
  mutex_lock lock(mu_);

  auto stub = stubs_.find(address);
  if (stub != stubs_.end()) return stub->second.get();

  ChannelPtr channel = CreateChannelForAddress(address);
  auto* created = new ::grpc::GenericStub(channel);
  stubs_[address].reset(created);
  return created;
}

GrpcRPCFactory::ChannelPtr GrpcRPCFactory::CreateChannelForAddress(
    const string& address) {
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());

  // Set a standard backoff timeout of 1s instead of the
  // (sometimes default) 20s.
  args.SetInt("grpc.testing.fixed_reconnect_backoff_ms", 1000);
  return ::grpc::CreateCustomChannel(
      /*target=*/address, ::grpc::InsecureChannelCredentials(), args);
}

}  // namespace tensorflow
