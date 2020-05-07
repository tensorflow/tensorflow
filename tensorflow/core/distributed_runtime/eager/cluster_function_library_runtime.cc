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
#include "tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h"

#include <map>
#include <memory>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {
namespace eager {
namespace {
void StripDefaultAttributesInRegisterFunctionOp(
    RegisterFunctionOp* register_function) {
  StripDefaultAttributes(
      *OpRegistry::Global(),
      register_function->mutable_function_def()->mutable_node_def());
  for (auto& function :
       *register_function->mutable_library()->mutable_function()) {
    StripDefaultAttributes(*OpRegistry::Global(), function.mutable_node_def());
  }
}
}  // namespace

void EagerClusterFunctionLibraryRuntime::Instantiate(
    const string& function_name, const FunctionLibraryDefinition& lib_def,
    AttrSlice attrs, const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::LocalHandle* handle,
    FunctionLibraryRuntime::DoneCallback done) {
  auto target = options.target;
  auto released_op = std::make_unique<EagerOperation>(ctx_);
  Status s =
      released_op->Reset(function_name.c_str(), target.c_str(), true, nullptr);
  if (!s.ok()) {
    done(s);
    return;
  }
  if (!released_op->is_function()) {
    done(errors::Internal(function_name, " is not a function."));
    return;
  }

  VLOG(1) << "CFLR::Instantiate: " << function_name << " on " << target
          << " (this: " << this << ")";
  core::RefCountPtr<eager::EagerClient> eager_client;
  Device* device;
  s = ctx_->FindDeviceFromName(target.c_str(), &device);
  if (!s.ok()) {
    done(s);
    return;
  }
  s = ctx_->GetClient(device, &eager_client);
  if (!s.ok()) {
    done(s);
    return;
  }

  if (eager_client == nullptr) {
    done(errors::InvalidArgument("Could not find eager client for target: ",
                                 target));
    return;
  }

  const FunctionLibraryDefinition& func_lib_def =
      options.lib_def ? *options.lib_def : lib_def;

  EnqueueRequest* request = new EnqueueRequest;
  EnqueueResponse* response = new EnqueueResponse;

  request->set_context_id(context_id_);

  RegisterFunctionOp* register_function =
      request->add_queue()->mutable_register_function();
  *register_function->mutable_function_def() =
      *func_lib_def.Find(function_name);
  register_function->set_is_component_function(true);
  *register_function->mutable_library() =
      func_lib_def.ReachableDefinitions(register_function->function_def())
          .ToProto();
  StripDefaultAttributesInRegisterFunctionOp(register_function);

  eager_client->EnqueueAsync(
      request, response,
      [this, request, response, handle, released_op = released_op.release(),
       target, eager_client = eager_client.get(), done](const Status& s) {
        {
          mutex_lock l(mu_);
          *handle = function_data_.size();
          function_data_.emplace_back(target, eager_client,
                                      absl::WrapUnique(released_op));
        }
        done(s);
        delete request;
        delete response;
      });
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
  std::vector<FunctionArg> function_args;
  for (const auto& tensor : args) {
    function_args.push_back(tensor);
  }
  Run(opts, handle, function_args, rets, std::move(done));
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle,
    gtl::ArraySlice<FunctionArg> args, std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done) {
  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    DCHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  EagerClient* eager_client = function_data->eager_client.get();
  if (eager_client == nullptr) {
    done(errors::Internal("Could not find eager client"));
    return;
  }

  Device* device;
  Status s = ctx_->FindDeviceFromName(function_data->target.c_str(), &device);
  if (!s.ok()) {
    done(errors::Internal("Failed to get device"));
    return;
  }

  EagerOperation* op = function_data->op.get();

  if (!op->Inputs().empty()) {
    done(errors::Internal("Inputs should not be set during instantiation."));
    return;
  }

  auto request = std::make_shared<RunComponentFunctionRequest>();
  auto response = std::make_shared<RunComponentFunctionResponse>();
  request->set_context_id(context_id_);
  eager::Operation* remote_op = request->mutable_operation();

  for (const auto& arg : args) {
    if (arg.index() == 0) {
      absl::get<Tensor>(arg).AsProtoTensorContent(
          remote_op->add_op_inputs()->mutable_tensor());
    } else {
      remote_op->add_op_inputs()->mutable_remote_handle()->Swap(
          absl::get<RemoteTensorHandle*>(arg));
    }
  }

  // The remote component function should use the same op_id as its parent
  // multi-device function's in order to get the global unique op_id generated
  // by the master context.
  if (opts.op_id.has_value()) {
    remote_op->set_id(opts.op_id.value());
  } else {
    remote_op->set_id(kInvalidRemoteOpId);
  }
  remote_op->set_is_function(true);
  remote_op->set_is_component_function(true);
  remote_op->set_func_step_id(opts.step_id);
  remote_op->set_name(op->Name());
  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(function_data->target);

  // Execute component function on remote worker using RunComponentFunction RPC.
  // Different from executing remote functions with Enqueue, this method runs
  // a function on remote worker without tying up a thread (i.e., pure
  // asynchronously).
  eager_client->RunComponentFunctionAsync(
      request.get(), response.get(),
      [request, response, rets, done = std::move(done)](const Status& s) {
        if (!s.ok()) {
          done(s);
          return;
        }
        for (const auto& tensor_proto : response->tensor()) {
          Tensor t;
          if (t.FromProto(tensor_proto)) {
            rets->push_back(std::move(t));
          } else {
            done(errors::Internal("Could not convert tensor proto: ",
                                  tensor_proto.DebugString()));
            return;
          }
        }
        done(Status::OK());
      });
}

void EagerClusterFunctionLibraryRuntime::CleanUp(
    uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
    FunctionLibraryRuntime::DoneCallback done) {
  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    DCHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  EagerClient* eager_client = function_data->eager_client.get();
  if (eager_client == nullptr) {
    done(errors::Internal("Could not find eager client"));
    return;
  }

  eager::EnqueueRequest* request = new eager::EnqueueRequest;
  EnqueueResponse* response = new EnqueueResponse;
  request->set_context_id(context_id_);
  CleanupFunctionOp* cleanup_function =
      request->add_queue()->mutable_cleanup_function();
  cleanup_function->set_step_id(step_id);
  // StreamingEnqueueAsync could be blocking when streaming RPC is disabled.
  // CleanUp() needs to be non-blocking since it would be invoked inside the
  // enqueue done callback of Run(). So we don't use StreamingEnqueueAsync here.
  eager_client->EnqueueAsync(request, response,
                             [request, response, done](const Status& status) {
                               done(status);
                               delete request;
                               delete response;
                             });
}

DistributedFunctionLibraryRuntime* CreateClusterFLR(
    const uint64 context_id, EagerContext* ctx, WorkerSession* worker_session) {
  if (ctx->LazyCopyFunctionRemoteInputs()) {
    return new EagerClusterFunctionLibraryRuntime(
        context_id, ctx, worker_session->remote_device_mgr());
  } else {
    return worker_session->cluster_flr();
  }
}

}  // namespace eager
}  // namespace tensorflow
