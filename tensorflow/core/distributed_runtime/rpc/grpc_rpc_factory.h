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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RPC_FACTORY_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RPC_FACTORY_H_

#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/rpc/call_container.h"
#include "tensorflow/core/util/rpc/rpc_factory.h"

namespace tensorflow {

// Forward declaration of GrpcCall.
namespace internal {
class GrpcCall;
}  // namespace internal

class GrpcRPCFactory : public RPCFactory {
 public:
  explicit GrpcRPCFactory(OpKernelConstruction* ctx, bool fail_fast,
                          int64 timeout_in_ms);

  // Explicit destructor to control destruction order.
  ~GrpcRPCFactory() override;

  void Call(OpKernelContext* ctx, int64 num_elements, const Tensor& address_t,
            const Tensor& method_t, const Tensor& request_t, const bool try_rpc,
            Tensor* response_t, Tensor* status_code_t, Tensor* status_message_t,
            AsyncOpKernel::DoneCallback done) override;

 protected:
  typedef std::shared_ptr<::grpc::Channel> ChannelPtr;
  virtual ChannelPtr CreateChannelForAddress(const string& address);

 private:
  // Creates a call and registers it with given `container`. The `index` is used
  // to index into the tensor arguments.
  void CreateCall(const Tensor& request_t, const bool try_rpc, int index,
                  CallContainer<internal::GrpcCall>* container,
                  Tensor* response_t, Tensor* status_code_t,
                  Tensor* status_message_t);

  // Asynchronously invokes the given `call`. The call completion is handled
  // by the call container the call was previously registered with.
  void StartCall(const Tensor& address_t, const Tensor& method_t,
                 internal::GrpcCall* call);

  ::grpc::GenericStub* GetOrCreateStubForAddress(const string& address);

  bool fail_fast_;
  int64 timeout_in_ms_;
  ::grpc::CompletionQueue completion_queue_;
  Thread* polling_thread_;  // Owned.

  mutex mu_;
  typedef std::unique_ptr<::grpc::GenericStub> StubPtr;
  std::unordered_map<string, StubPtr> stubs_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_RPC_FACTORY_H_
