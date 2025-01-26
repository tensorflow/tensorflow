/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
// Needed for encoding and decoding ResourceDeleter Variant.
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/distribute/experimental/rpc/kernels/grpc_credentials.h"
#include "tensorflow/distribute/experimental/rpc/kernels/grpc_rpc_service.h"
#include "tensorflow/distribute/experimental/rpc/proto/tf_rpc_service.pb.h"

namespace tensorflow {
namespace rpc {

// Register a function to local built in server or RPC server
class RpcServerRegisterOp : public OpKernel {
 public:
  explicit RpcServerRegisterOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  NameAttrList func_;
  StructuredValue output_specs_;
  StructuredValue input_specs_;
  RpcServerRegisterOp(const RpcServerRegisterOp&) = delete;
  void operator=(const RpcServerRegisterOp&) = delete;
};

// Create a server resource to store registered functions
class RpcServerOp : public OpKernel {
 public:
  explicit RpcServerOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  RpcServerOp(const RpcServerOp&) = delete;
  void operator=(const RpcServerOp&) = delete;
};

// Start GRPC server with registered methods
class RpcServerStartOp : public OpKernel {
 public:
  explicit RpcServerStartOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  RpcServerStartOp(const RpcServerStartOp&) = delete;
  void operator=(const RpcServerStartOp&) = delete;
};

// Create a client resource to store registered functions.
class RpcClientOp : public AsyncOpKernel {
 public:
  explicit RpcClientOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::string name_;
  bool list_registered_methods_;
  RpcClientOp(const RpcClientOp&) = delete;
  void operator=(const RpcClientOp&) = delete;
};

// Remote RPC using client handle passed and returns a future Resource handle to
// get Status and value.
class RpcCallOp : public OpKernel {
 public:
  explicit RpcCallOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  RpcCallOp(const RpcCallOp&) = delete;
  void operator=(const RpcCallOp&) = delete;
};

// Remote Check Status Op waits till the RPC issued by Call Op is finished.
class RpcCheckStatusOp : public AsyncOpKernel {
 public:
  explicit RpcCheckStatusOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  RpcCheckStatusOp(const RpcCheckStatusOp&) = delete;
  void operator=(const RpcCheckStatusOp&) = delete;
};

// Op to get response output after RPC Call.
class RpcGetValueOp : public AsyncOpKernel {
 public:
  explicit RpcGetValueOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  RpcGetValueOp(const RpcGetValueOp&) = delete;
  void operator=(const RpcGetValueOp&) = delete;
};

class DeleteRpcFutureResourceOp : public OpKernel {
 public:
  explicit DeleteRpcFutureResourceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

 protected:
  void Compute(OpKernelContext* ctx) override {
    const ResourceHandle& handle = ctx->input(0).flat<ResourceHandle>()(0);
    // The resource is guaranteed to exist because the variant tensor
    // wrapping the deleter is provided as an unused input to this op, which
    // guarantees that it has not run yet.
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete(handle));
  }
};

struct FunctionMetadata {
  FunctionLibraryRuntime::Handle handle;
  FunctionLibraryRuntime* lib;
  std::vector<Tensor> captured_inputs;
  StructuredValue input_specs;
  StructuredValue output_specs;
};

class FunctionRegistry {
 public:
  std::string DebugString() const {
    mutex_lock l(mu_);
    std::string debug_string = "Registered methods: [";
    debug_string.append(absl::StrJoin(
        registered_methods_, ", ",
        [](std::string* out, const auto& pair) { return pair.first; }));

    debug_string.append("]");
    return debug_string;
  }

  absl::Status Register(const std::string& method, FunctionLibraryRuntime* lib,
                        FunctionLibraryRuntime::Handle fn_handle,
                        std::vector<Tensor> captured_inputs,
                        const StructuredValue& input_specs,
                        const StructuredValue& output_specs) {
    mutex_lock l(mu_);
    FunctionMetadata fn_metadata;
    fn_metadata.handle = fn_handle;
    fn_metadata.lib = lib;
    fn_metadata.captured_inputs = std::move(captured_inputs);
    fn_metadata.input_specs = input_specs;
    fn_metadata.output_specs = output_specs;
    auto result = registered_methods_.insert(
        std::pair<std::string, FunctionMetadata>(method, fn_metadata));
    if (!result.second) {
      return tensorflow::errors::InvalidArgument(
          absl::StrCat(method, " is already registered."));
    }
    return absl::OkStatus();
  }

  absl::Status LookUp(const std::string& method,
                      FunctionMetadata* output) const {
    mutex_lock l(mu_);
    auto it = registered_methods_.find(method);
    if (it == registered_methods_.end()) {
      return tensorflow::errors::InvalidArgument(
          absl::StrCat(method, " is not registered."));
    }

    *output = it->second;
    return absl::OkStatus();
  }

  const gtl::FlatMap<std::string, FunctionMetadata>& List() const {
    return registered_methods_;
  }

 private:
  mutable mutex mu_;
  gtl::FlatMap<std::string, FunctionMetadata> registered_methods_
      TF_GUARDED_BY(mu_);
};

class RpcServiceImpl : public grpc::RpcService::Service {
 public:
  explicit RpcServiceImpl(const FunctionRegistry& registry)
      : registry_(registry) {}

  ::grpc::Status Call(::grpc::ServerContext* context,
                      const CallRequest* request,
                      CallResponse* response) override {
    const auto& method_name = request->method();

    FunctionLibraryRuntime::Options opts;

    FunctionMetadata fn_metadata;
    auto status = registry_.LookUp(method_name, &fn_metadata);
    FunctionLibraryRuntime::Handle handle = fn_metadata.handle;
    FunctionLibraryRuntime* fn_lib = fn_metadata.lib;
    std::vector<Tensor> captured_inputs =
        std::move(fn_metadata.captured_inputs);

    if (!status.ok()) {
      return ToGrpcStatus(status);
    }

    std::vector<Tensor> args;
    for (const auto& t : request->input_tensors()) {
      Tensor tensor;
      if (tensor.FromProto(t)) {
        args.push_back(std::move(tensor));
      } else {
        return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                              "Failed to parse input tensor from proto.");
      }
    }

    // Add captured args as well
    for (const auto& t : captured_inputs) {
      args.push_back(std::move(t));
    }

    std::vector<Tensor>* rets = new std::vector<Tensor>;
    Notification notification;
    fn_lib->Run(
        opts, handle, args, rets,
        [rets, response, &notification, &status](const absl::Status& st) {
          status = st;
          if (status.ok()) {
            for (size_t i = 0; i < rets->size(); ++i) {
              auto t = response->add_output_tensors();
              (*rets)[i].AsProtoField(t);
            }
          }
          delete rets;
          notification.Notify();
        });

    notification.WaitForNotification();
    return ToGrpcStatus(status);
  }

  ::grpc::Status List(::grpc::ServerContext* context,
                      const rpc::ListRequest* request,
                      rpc::ListResponse* response) override {
    auto methods = registry_.List();
    for (auto it : methods) {
      auto* registered_method = response->add_registered_methods();
      registered_method->set_method(it.first);
      *registered_method->mutable_output_specs() = it.second.output_specs;
      *registered_method->mutable_input_specs() = it.second.input_specs;
    }
    return ::grpc::Status(::grpc::Status::OK);
  }

 private:
  const FunctionRegistry& registry_;
};

class RpcServer : public ResourceBase {
 public:
  explicit RpcServer(std::string server_address)
      : server_address_(server_address),
        server_(nullptr),
        server_started_(false) {
    service_ = std::make_unique<RpcServiceImpl>(registry_);
  }

  ~RpcServer() override {
    if (server_) {
      LOG(INFO) << "Shutting down server listening on: " << server_address_;
      server_->Shutdown();
    }
  }

  std::string DebugString() const override {
    return absl::StrCat("RpcServer resource with ", registry_.DebugString());
  }

  absl::Status Register(const std::string& method, FunctionLibraryRuntime* lib,
                        FunctionLibraryRuntime::Handle fn_handle,
                        std::vector<Tensor> captured_inputs,
                        const StructuredValue& input_specs,
                        const StructuredValue& output_specs) {
    mutex_lock m(mu_);
    if (server_started_) {
      return tensorflow::errors::FailedPrecondition(
          "All methods must be registered before starting the server. Method "
          "registration after starting the server is not supported.");
    }
    return registry_.Register(method, lib, fn_handle, captured_inputs,
                              input_specs, output_specs);
  }

  void StartServer() {
    mutex_lock l(mu_);
    ::grpc::ServerBuilder builder;
    std::shared_ptr<::grpc::ServerCredentials> creds =
        GetDefaultServerCredentials();
    builder.AddListeningPort(server_address_, creds);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on: " << server_address_;
    server_started_ = true;
  }

 private:
  FunctionRegistry registry_;
  std::unique_ptr<RpcServiceImpl> service_;
  std::string server_address_;
  std::unique_ptr<::grpc::Server> server_;
  bool server_started_ TF_GUARDED_BY(mu_);
  mutex mu_;
};

class GrpcPollingThread {
 public:
  explicit GrpcPollingThread(std::string thread_name) {
    // Thread name can only have alpha numeric characters. Remove special
    // characters from input thread_name.
    thread_name.erase(
        std::remove_if(thread_name.begin(), thread_name.end(),
                       [](auto const c) -> bool { return !std::isalnum(c); }),
        thread_name.end());
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), absl::StrCat("GrpcPollingThread", thread_name),
        [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
          }
        }));
  }

  ~GrpcPollingThread() {
    completion_queue_.Shutdown();
    thread_.reset();
  }

  ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};

class RpcClient : public ResourceBase {
 public:
  explicit RpcClient(std::string address, std::string resource_name,
                     int64 timeout_in_ms)
      : server_address_(address),
        thread_(resource_name),
        timeout_in_ms_(timeout_in_ms) {
    std::shared_ptr<::grpc::ChannelCredentials> creds =
        GetDefaultChannelCredentials();

    channel_ = ::grpc::CreateChannel(address, creds);

    stub_ = std::make_unique<::grpc::GenericStub>(channel_);
    cq_ = thread_.completion_queue();
    callback_threadpool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), ThreadOptions(), "RPC_Client_threadpool", 5,
        /*low_latency_hint=*/false, /*allocator=*/nullptr);
  }

  std::string DebugString() const override {
    return absl::StrCat("Rpc client for address: ", server_address_);
  }

  void CallAsync(const std::string& method_name,
                 const std::vector<Tensor>& inputs, CallResponse* response,
                 StatusCallback callback, int64 timeout_in_ms) {
    CallRequest request;
    request.set_method(method_name);
    for (const auto& t : inputs) {
      t.AsProtoField(request.add_input_tensors());
    }
    ::grpc::ClientContext context;
    // Use per call timeout if specified, otherwise use default client timeout.
    int64 timeout = timeout_in_ms > 0 ? timeout_in_ms : timeout_in_ms_;
    new RPCState<CallResponse>(
        stub_.get(), cq_, "/tensorflow.rpc.RpcService/Call", request, response,
        /*done=*/std::move(callback),
        /*call_opts=*/nullptr,
        /*threadpool=*/callback_threadpool_.get(),
        /*fail_fast=*/false, /*timeout_in_ms=*/timeout,
        /*max_retries=*/0, /*target=*/nullptr);
  }

  void ListAsync(rpc::ListResponse* response, StatusCallback callback) {
    rpc::ListRequest request;
    ::grpc::ClientContext context;
    // fail_fast=false sets wait_for_ready to true in GRPC call.
    // ListAsync is called during Client creation thus, we want to wait till
    // server is ready for issuing RPC.
    new RPCState<rpc::ListResponse>(
        stub_.get(), cq_, "/tensorflow.rpc.RpcService/List", request, response,
        /*done=*/std::move(callback),
        /*call_opts=*/nullptr,
        /*threadpool=*/callback_threadpool_.get(),
        /*fail_fast=*/false, /*timeout_in_ms=*/timeout_in_ms_,
        /*max_retries=*/0, /*target=*/nullptr);
  }

 private:
  std::shared_ptr<::grpc::Channel> channel_;
  std::string server_address_;
  std::unique_ptr<::grpc::GenericStub> stub_;
  ::grpc::CompletionQueue* cq_;
  GrpcPollingThread thread_;
  std::unique_ptr<thread::ThreadPool> callback_threadpool_;
  int64 timeout_in_ms_;
};

class RpcFutureResource : public ResourceBase {
  typedef std::function<void(const absl::Status&, const CallResponse&)>
      FutureCallBack;

 public:
  RpcFutureResource() : done_(false) {}
  std::string DebugString() const override { return "Wait Resource"; }

  void AddDoneCallback(FutureCallBack cb) {
    mutex_lock l(mu_);
    if (!done_) {
      call_backs_.push_back(cb);
    } else {
      cb(status_, response_);
    }
  }

  void OperationFinished() {
    mutex_lock l(mu_);
    for (const auto& cb : call_backs_) {
      cb(status_, response_);
    }
    done_ = true;
  }

  void set_status(absl::Status status) { status_.Update(status); }
  absl::Status get_status() { return status_; }
  CallResponse* get_response() { return &response_; }

 private:
  CallResponse response_;
  bool done_ TF_GUARDED_BY(mu_);
  absl::Status status_;
  std::vector<FutureCallBack> call_backs_ TF_GUARDED_BY(mu_);
  mutable mutex mu_;
};

absl::Status ExtractServerAddressFromInput(OpKernelContext* ctx,
                                           std::string* address) {
  const Tensor* server_address;
  auto status = ctx->input("server_address", &server_address);
  if (status.ok()) {
    *address = server_address->scalar<tstring>()();
  }
  return status;
}

RpcServerOp::RpcServerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void RpcServerOp::Compute(OpKernelContext* ctx) {
  std::string address = "";
  OP_REQUIRES_OK(ctx, ExtractServerAddressFromInput(ctx, &address));

  // Create resource handle
  AllocatorAttributes attr;
  attr.set_on_host(true);

  ResourceHandle resource_handle =
      MakeResourceHandle<RpcServer>(ctx, "rpc_server", address);
  Tensor handle;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
  handle.scalar<ResourceHandle>()() = resource_handle;

  // Create resource
  auto creator = [address](RpcServer** server) {
    *server = new RpcServer(address);
    return absl::OkStatus();
  };
  core::RefCountPtr<RpcServer> server;
  OP_REQUIRES_OK(ctx, LookupOrCreateResource<RpcServer>(ctx, resource_handle,
                                                        &server, creator));
  ctx->set_output(0, handle);
}

RpcClientOp::RpcClientOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("list_registered_methods", &list_registered_methods_));
}

void RpcClientOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  std::string address = "";
  OP_REQUIRES_OK_ASYNC(ctx, ExtractServerAddressFromInput(ctx, &address), done);

  const Tensor* timeout;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input("timeout_in_ms", &timeout), done);
  auto timeout_in_ms = timeout->scalar<int64_t>()();

  // Create resource handle
  AllocatorAttributes attr;
  attr.set_on_host(true);
  auto resource_name = absl::StrCat(name_, address);

  ResourceHandle resource_handle =
      MakeResourceHandle<RpcClient>(ctx, "rpc_client", resource_name);
  Tensor handle;
  OP_REQUIRES_OK_ASYNC(
      ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr),
      done);
  handle.scalar<ResourceHandle>()() = resource_handle;

  // Delete old client handle if exists, to clear old client resource state.
  DeleteResource(ctx, resource_handle).IgnoreError();

  // Create resource
  auto creator = [&address, &resource_name, timeout_in_ms](RpcClient** client) {
    *client = new RpcClient(address, resource_name, timeout_in_ms);
    return absl::OkStatus();
  };

  core::RefCountPtr<RpcClient> client;
  OP_REQUIRES_OK_ASYNC(
      ctx,
      LookupOrCreateResource<RpcClient>(ctx, resource_handle, &client, creator),
      done);
  ctx->set_output(0, handle);

  if (!list_registered_methods_) {
    Tensor* method_output_t;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(1, TensorShape({}), &method_output_t), done);
    method_output_t->scalar<tstring>()() = "";
    done();
    return;
  }
  auto* response = new ListResponse();
  client->ListAsync(
      response, [ctx, response, done](const absl::Status& status) {
        if (!status.ok()) {
          ctx->SetStatus(status);
        } else {
          Tensor* method_output_signatures_t;
          auto method_output_shape = TensorShape(
              {static_cast<int64_t>(response->registered_methods_size())});
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_output(1, method_output_shape,
                                   &method_output_signatures_t),
              done);
          auto method_output_signatures =
              method_output_signatures_t->vec<tstring>();
          for (int i = 0; i < response->registered_methods_size(); ++i) {
            method_output_signatures(i) =
                response->registered_methods(i).SerializeAsString();
          }
        }
        delete response;
        done();
      });
}

RpcServerStartOp::RpcServerStartOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void RpcServerStartOp::Compute(OpKernelContext* ctx) {
  core::RefCountPtr<RpcServer> server;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &server));

  server->StartServer();
  ctx->SetStatus(absl::OkStatus());
}

RpcServerRegisterOp::RpcServerRegisterOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(FunctionLibraryDefinition::kFuncAttr, &func_));
  std::string output_specs_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_specs", &output_specs_string));

  OP_REQUIRES(ctx, output_specs_.ParseFromString(output_specs_string),
              tensorflow::errors::InvalidArgument(
                  "Unable to parse StructuredValue output_spec string: ",
                  output_specs_string));

  std::string input_specs_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_specs", &input_specs_string));

  OP_REQUIRES(ctx, input_specs_.ParseFromString(input_specs_string),
              tensorflow::errors::InvalidArgument(
                  "Unable to parse StructuredValue output_spec string: ",
                  input_specs_string));
}

void RpcServerRegisterOp::Compute(OpKernelContext* ctx) {
  FunctionLibraryRuntime* lib = ctx->function_library();
  OP_REQUIRES(ctx, lib != nullptr,
              errors::Internal("No function library is provided"));

  const Tensor* method_name;
  OP_REQUIRES_OK(ctx, ctx->input("method_name", &method_name));

  std::string method = method_name->scalar<tstring>()();

  OpInputList captured_inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list("captured_inputs", &captured_inputs));
  std::vector<Tensor> captured(captured_inputs.begin(), captured_inputs.end());

  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = ctx->device()->name();
  instantiate_opts.lib_def = lib->GetFunctionLibraryDefinition();
  // In case captured inputs are on different device.
  instantiate_opts.is_multi_device_function = true;

  const FunctionDef* fdef =
      lib->GetFunctionLibraryDefinition()->Find(func_.name());
  OP_REQUIRES(ctx, fdef != nullptr,
              errors::Internal("Failed to find function."));
  int num_args = fdef->signature().input_arg_size();

  const int num_non_captured_inputs = num_args - captured.size();
  for (int i = 0; i < num_non_captured_inputs; ++i) {
    instantiate_opts.input_devices.push_back(ctx->device()->name());
  }

  absl::flat_hash_map<string, std::vector<string>> composite_devices;
  for (int i = 0; i < captured.size(); ++i) {
    if (captured[i].dtype() == DT_RESOURCE) {
      instantiate_opts.input_devices.push_back(GetFunctionResourceInputDevice(
          captured[i], num_non_captured_inputs + i, *fdef, &composite_devices));
    } else {
      instantiate_opts.input_devices.push_back(ctx->device()->name());
    }
  }

  for (const auto& it : composite_devices) {
    instantiate_opts.composite_devices[it.first] = &it.second;
  }

  FunctionLibraryRuntime::Handle handle;
  OP_REQUIRES_OK(ctx, lib->Instantiate(func_.name(), AttrSlice(&func_.attr()),
                                       instantiate_opts, &handle));

  core::RefCountPtr<RpcServer> server;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &server));

  OP_REQUIRES_OK(ctx, server->Register(method, lib, handle, std::move(captured),
                                       input_specs_, output_specs_));
}

RpcCallOp::RpcCallOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void RpcCallOp::Compute(OpKernelContext* ctx) {
  const Tensor* method_name;
  OP_REQUIRES_OK(ctx, ctx->input("method_name", &method_name));
  std::string method = method_name->scalar<tstring>()();

  const Tensor* timeout;
  OP_REQUIRES_OK(ctx, ctx->input("timeout_in_ms", &timeout));
  auto timeout_in_ms = timeout->scalar<int64_t>()();

  OpInputList arguments;
  OP_REQUIRES_OK(ctx, ctx->input_list("args", &arguments));
  std::vector<Tensor> args(arguments.begin(), arguments.end());

  core::RefCountPtr<RpcClient> client;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client));

  ResourceHandle resource_handle = MakeResourceHandle<RpcFutureResource>(
      ctx, "rpc_future_resource", absl::StrFormat("%d", random::New64()));

  AllocatorAttributes attr;
  attr.set_on_host(true);
  Tensor handle;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
  handle.scalar<ResourceHandle>()() = resource_handle;

  // Create resource
  auto creator = [](RpcFutureResource** resource) {
    *resource = new RpcFutureResource();
    return absl::OkStatus();
  };
  core::RefCountPtr<RpcFutureResource> future_resource;
  OP_REQUIRES_OK(ctx, LookupOrCreateResource<RpcFutureResource>(
                          ctx, resource_handle, &future_resource, creator));

  Tensor deleter_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_VARIANT, TensorShape({}), &deleter_t, attr));
  deleter_t.scalar<Variant>()() =
      ResourceDeleter(resource_handle, ctx->resource_manager());
  ctx->set_output(0, handle);
  ctx->set_output(1, deleter_t);

  CallResponse* response = future_resource->get_response();
  auto* future_resource_ptr = future_resource.release();

  client->CallAsync(
      method, args, response,
      [future_resource_ptr](const absl::Status& status) {
        future_resource_ptr->set_status(status);
        future_resource_ptr->OperationFinished();
        future_resource_ptr->Unref();
      },
      timeout_in_ms);
}

RpcCheckStatusOp::RpcCheckStatusOp(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx) {}

void RpcCheckStatusOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  core::RefCountPtr<RpcFutureResource> future_resource;
  auto handle = HandleFromInput(ctx, 0);
  {
    auto status = LookupResource(ctx, handle, &future_resource);
    if (!status.ok()) {
      if (absl::IsNotFound(status)) {
        ctx->SetStatus(tensorflow::errors::NotFound(
            absl::StrCat("Future resource no longer exists. Please make sure "
                         "resource is not already deleted.")));
        done();
        return;
      } else {
        ctx->SetStatus(status);
      }
    }
  }

  future_resource->AddDoneCallback(
      [ctx, done, handle](const absl::Status& status,
                          const CallResponse& response) {
        Tensor error_code(DT_INT64, TensorShape({})),
            error_message(DT_STRING, TensorShape({}));
        error_code.scalar<int64_t>()() = status.raw_code();
        error_message.scalar<tstring>()() = status.message();

        ctx->set_output(0, error_code);
        ctx->set_output(1, error_message);

        done();
      });
}

RpcGetValueOp::RpcGetValueOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}

void RpcGetValueOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  core::RefCountPtr<RpcFutureResource> future_resource;
  auto handle = HandleFromInput(ctx, 0);
  {
    auto status = LookupResource(ctx, handle, &future_resource);
    if (!status.ok()) {
      if (absl::IsNotFound(status)) {
        ctx->SetStatus(tensorflow::errors::NotFound(
            absl::StrCat("Future resource no longer exists. Please ensure "
                         "resource is not already deleted.")));
        done();
        return;
      } else {
        ctx->SetStatus(status);
      }
    }
  }

  future_resource->AddDoneCallback(
      [ctx, done, handle](const absl::Status& status,
                          const CallResponse& response) {
        if (!status.ok()) {
          ctx->SetStatus(status);
        } else {
          if (ctx->num_outputs() != response.output_tensors().size()) {
            ctx->SetStatus(tensorflow::errors::InvalidArgument(absl::StrCat(
                "Incorrect number of output types specified.",
                ctx->num_outputs(), " ", response.output_tensors().size())));
          } else {
            int i = 0;
            for (const auto& t_proto : response.output_tensors()) {
              Tensor t;
              if (!t.FromProto(t_proto)) {
                ctx->SetStatus(tensorflow::errors::Internal(
                    absl::StrCat("Invalid Tensor Proto response returned.")));
              }
              ctx->set_output(i++, std::move(t));
            }
          }
        }
        done();
      });
}

REGISTER_KERNEL_BUILDER(Name("RpcServer").Device(DEVICE_CPU), RpcServerOp);
REGISTER_KERNEL_BUILDER(Name("RpcClient").Device(DEVICE_CPU), RpcClientOp);
REGISTER_KERNEL_BUILDER(Name("RpcServerStart").Device(DEVICE_CPU),
                        RpcServerStartOp);
REGISTER_KERNEL_BUILDER(Name("RpcServerRegister").Device(DEVICE_CPU),
                        RpcServerRegisterOp);
REGISTER_KERNEL_BUILDER(Name("RpcCall").Device(DEVICE_CPU), RpcCallOp);
REGISTER_KERNEL_BUILDER(Name("RpcCheckStatus").Device(DEVICE_CPU),
                        RpcCheckStatusOp);
REGISTER_KERNEL_BUILDER(Name("RpcGetValue").Device(DEVICE_CPU), RpcGetValueOp);
REGISTER_KERNEL_BUILDER(Name("DeleteRpcFutureResource").Device(DEVICE_CPU),
                        DeleteRpcFutureResourceOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("RpcServerRegister");

}  // namespace rpc
}  // namespace tensorflow
