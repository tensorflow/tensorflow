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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {
namespace eager {

void EagerClusterFunctionLibraryRuntime::Instantiate(
    const string& function_name, const FunctionLibraryDefinition& lib_def,
    AttrSlice attrs, const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::LocalHandle* handle,
    FunctionLibraryRuntime::DoneCallback done) {
  const tensorflow::AttrTypeMap* attr_types;
  bool is_function = false;
  Status s;
  s = tensorflow::AttrTypeMapForOp(function_name.c_str(), &attr_types,
                                   &is_function);
  if (!s.ok()) {
    done(s);
    return;
  }
  if (!is_function) {
    done(errors::Internal(function_name, " is not a function."));
    return;
  }
  auto target = options.target;
  auto* released_op =
      new EagerOperation(ctx_, function_name.c_str(), is_function, attr_types);
  s = released_op->SetDeviceName(target.c_str());
  if (!s.ok()) {
    done(s);
    return;
  }

  VLOG(1) << "CFLR::Instantiate: " << function_name << " on " << target
          << " (this: " << this << ")";
  eager::EagerClient* eager_client = nullptr;
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

  eager_client->EnqueueAsync(request, response,
                             [this, request, response, handle, released_op,
                              target, eager_client, done](const Status& s) {
                               {
                                 mutex_lock l(mu_);
                                 *handle = function_data_.size();
                                 function_data_.emplace_back(
                                     target, eager_client,
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
  done(errors::Unimplemented("Not implemented"));
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle,
    std::vector<eager::RemoteTensorHandle>* args,
    FunctionLibraryRuntime::DoneCallback done) {
  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    DCHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  EagerClient* eager_client = function_data->eager_client;
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

  if (!opts.op_id.has_value()) {
    done(
        errors::Internal("op_id is not set for remote function: ", op->Name()));
  }

  eager::EnqueueRequest* request = new eager::EnqueueRequest;
  request->set_context_id(context_id_);
  eager::Operation* remote_op = request->add_queue()->mutable_operation();
  for (size_t i = 0; i < args->size(); ++i) {
    remote_op->add_inputs()->Swap(&(*args)[i]);
  }
  // The remote component function should use the same op_id as its parent
  // multi-device function's in order to get the global unqiue op_id generated
  // by the master context.
  remote_op->set_id(opts.op_id.value());
  remote_op->set_is_function(true);
  remote_op->set_is_component_function(true);
  remote_op->set_func_step_id(opts.step_id);
  remote_op->set_name(op->Name());
  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(function_data->target);

  for (auto handle : op->Inputs()) {
    handle->Ref();
  }

  // TODO(yujingzhang): Use RemoteExecuteNode once we enable async execution.
  EnqueueResponse* response = new EnqueueResponse;
  eager_client->EnqueueAsync(request, response,
                             [op, request, response, done](const Status& s) {
                               for (auto handle : op->Inputs()) {
                                 handle->Unref();
                               }
                               done(s);
                               delete request;
                               delete response;
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

  EagerClient* eager_client = function_data->eager_client;
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
  eager_client->StreamingEnqueueAsync(
      request, response, [request, response, done](const Status& status) {
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
