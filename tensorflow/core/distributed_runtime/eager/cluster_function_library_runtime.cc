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

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {
namespace eager {

Status EagerClusterFunctionLibraryRuntime::Instantiate(
    const string& function_name, const FunctionLibraryDefinition& lib_def,
    AttrSlice attrs, const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::LocalHandle* handle) {
  const tensorflow::AttrTypeMap* attr_types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(tensorflow::AttrTypeMapForOp(function_name.c_str(),
                                                  &attr_types, &is_function));
  if (!is_function) {
    return errors::Internal(function_name, " is not a function.");
  }
  auto op = absl::make_unique<EagerOperation>(ctx_, function_name.c_str(),
                                              is_function, attr_types);
  TF_RETURN_IF_ERROR(op->SetDeviceName(options.target.c_str()));

  VLOG(1) << "CFLR::Instantiate: " << function_name << " on " << options.target
          << " (this: " << this << ")";
  eager::EagerClient* eager_client = nullptr;
  Device* device;
  TF_RETURN_IF_ERROR(ctx_->FindDeviceFromName(options.target.c_str(), &device));
  TF_RETURN_IF_ERROR(ctx_->GetClient(device, &eager_client));

  if (eager_client == nullptr) {
    return errors::InvalidArgument("Could not find eager client for target: ",
                                   options.target);
  }

  const FunctionLibraryDefinition& func_lib_def =
      options.lib_def ? *options.lib_def : lib_def;

  RegisterFunctionRequest request;
  const uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  // TODO(yujingzhang): add FunctionDefLibrary to RegisterFunctionRequest to
  // support nested functions.
  *request.mutable_function_def() = *func_lib_def.Find(function_name);
  request.set_is_component_function(true);

  Status status;
  Notification done;
  RegisterFunctionResponse response;
  eager_client->RegisterFunctionAsync(&request, &response, [&](Status s) {
    status = s;
    done.Notify();
  });
  done.WaitForNotification();
  TF_RETURN_IF_ERROR(status);

  mutex_lock l(mu_);
  *handle = function_data_.size();
  function_data_.emplace_back(options.target, context_id, eager_client,
                              std::move(op));
  return Status::OK();
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
  done(errors::Unimplemented("Not implemented"));
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle, const int64 op_id,
    absl::Span<eager::RemoteTensorHandle* const> args,
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

  eager::EnqueueRequest* request = new eager::EnqueueRequest;
  request->set_context_id(function_data->context_id);
  eager::Operation* remote_op = request->add_queue()->mutable_operation();
  for (size_t i = 0; i < args.size(); ++i) {
    remote_op->add_inputs()->Swap(args[i]);
  }
  // TODO(yujingzhang): add step_id to eager::Operation to make sure that all
  // component functions use the same step id.
  // The remote component function should use the same op_id as its parent
  // multi-device function's in order to get the global unqiue op_id generated
  // by the master context.
  remote_op->set_id(op_id);
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
  done(Status::OK());
}

}  // namespace eager
}  // namespace tensorflow
