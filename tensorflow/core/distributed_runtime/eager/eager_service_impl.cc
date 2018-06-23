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

#include "tensorflow/core/distributed_runtime/eager/eager_service_impl.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace eager {

namespace {
Status GetNumRetvals(tensorflow::EagerContext* context, const string& op_name,
                     const google::protobuf::Map<string, tensorflow::AttrValue>& attrs,
                     int* num_retvals) {
  const tensorflow::OpRegistrationData* op_reg_data = nullptr;
  auto status = tensorflow::OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (errors::IsNotFound(status)) {
    status = context->FindFunctionOpData(op_name, &op_reg_data);
  }
  TF_RETURN_IF_ERROR(status);

  const tensorflow::OpDef& op_def = op_reg_data->op_def;

  for (const auto& output_arg : op_def.output_arg()) {
    if (!output_arg.number_attr().empty()) {
      auto iter = attrs.find(output_arg.number_attr());
      if (iter == attrs.end()) {
        return errors::InvalidArgument("Unable to find number_attr ",
                                       output_arg.number_attr(),
                                       " for Op: ", op_name);
      }
      *num_retvals += iter->second.i();
    } else if (!output_arg.type_list_attr().empty()) {
      auto iter = attrs.find(output_arg.number_attr());
      if (iter == attrs.end()) {
        return errors::InvalidArgument("Unable to find number_attr ",
                                       output_arg.number_attr(),
                                       " for Op: ", op_name);
      }
      *num_retvals += iter->second.list().type_size();
    } else {
      *num_retvals += 1;
    }
  }

  return Status::OK();
}
}  // namespace

Status EagerServiceImpl::CreateContext(const CreateContextRequest* request,
                                       CreateContextResponse* response) {
  //make sure env_ , env_->rendezvous_mgr available
  if (env_ == nullptr || env_->rendezvous_mgr == nullptr) {
    return tensorflow::errors::Internal("invalid eager env_ or env_->rendezvous_mgr.");
  } 
  tensorflow::RemoteRendezvous* r = env_->rendezvous_mgr->Find(0);
  std::vector<tensorflow::Device*> devices;

  TF_RETURN_IF_ERROR(tensorflow::DeviceFactory::AddDevices(
      // TODO(nareshmodi): Correctly set the SessionOptions.
      SessionOptions(),
      strings::Printf("/job:%s/replica:0/task:%d",
                      request->server_def().job_name().data(),
                      request->server_def().task_index()),
      &devices));
  response->mutable_device_attributes()->Reserve(devices.size());
  for (auto& d : devices) {
    *response->add_device_attributes() = d->attributes();
  }

  std::unique_ptr<tensorflow::DeviceMgr> device_mgr(
      new tensorflow::DeviceMgr(devices));

  auto* r = env_->rendezvous_mgr->Find(request->rendezvous_id());
  auto session_name = strings::StrCat("eager_", request->rendezvous_id());
  TF_RETURN_IF_ERROR(env_->session_mgr->CreateSession(
      session_name, request->server_def(), true));

  std::shared_ptr<WorkerSession> worker_session;
  TF_RETURN_IF_ERROR(env_->session_mgr->WorkerSessionForSession(
      session_name, &worker_session));

  // Initialize remote tensor communication based on worker session.
  TF_RETURN_IF_ERROR(r->Initialize(worker_session.get()));

  std::unique_ptr<tensorflow::EagerContext> ctx(new tensorflow::EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      request->async(), std::move(device_mgr), r));

  uint64 context_id;
  {
    mutex_lock l(contexts_mu_);
    do {
      context_id = random::New64();
    } while (contexts_.find(context_id) != contexts_.end());
    contexts_.emplace(context_id, new ServerContext(std::move(ctx)));
  }
  response->set_context_id(context_id);

  return Status::OK();
}

Status EagerServiceImpl::ExecuteOp(const Operation& operation,
                                   ServerContext* server_context) {
  std::unique_ptr<tensorflow::EagerOperation> op;
  const char* name = operation.name().c_str();  // Shorthand
  const tensorflow::AttrTypeMap* types;
  auto status = tensorflow::AttrTypeMapForOp(name, &types);
  if (status.ok()) {
    op.reset(
        new tensorflow::EagerOperation(server_context->Context(), name, types));
  } else if (errors::IsNotFound(status)) {
    if (server_context->Context()->FindFunctionByName(name)) {
      op.reset(new tensorflow::EagerOperation(server_context->Context(), name,
                                              nullptr));
    } else {
      return status;
    }
  } else {
    return status;
  }

  TF_RETURN_IF_ERROR(op->SetDevice(operation.device().c_str()));

  for (const auto& remote_handle : operation.inputs()) {
    tensorflow::TensorHandle* handle;
    TF_RETURN_IF_ERROR(server_context->GetTensorHandle(
        RemoteTensorHandleInternal(remote_handle), &handle));

    op->AddInput(handle);
  }

  for (const auto& attr : operation.attrs()) {
    op->MutableAttrs()->Set(attr.first, attr.second);
  }

  int num_retvals = 0;
  // TODO(nareshmodi): Consider caching this.
  TF_RETURN_IF_ERROR(GetNumRetvals(server_context->Context(), operation.name(),
                                   operation.attrs(), &num_retvals));

  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> retvals;
  TF_RETURN_IF_ERROR(EagerExecute(op.get(), &retvals, &num_retvals));

  server_context->AddOperationOutputs(retvals, operation.id());

  return Status::OK();
}

Status EagerServiceImpl::Enqueue(const EnqueueRequest* request,
                                 EnqueueResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  for (const auto& item : request->queue()) {
    if (item.has_operation()) {
      TF_RETURN_IF_ERROR(ExecuteOp(item.operation(), context));
    } else {
      TF_RETURN_IF_ERROR(context->DeleteTensorHandle(
          RemoteTensorHandleInternal(item.handle_to_decref())));
    }
  }

  return Status::OK();
}

Status EagerServiceImpl::WaitQueueDone(const WaitQueueDoneRequest* request,
                                       WaitQueueDoneResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  if (request->op_id_size() > 0) {
    return errors::Unimplemented(
        "EagerServiceImpl::WaitQueueDone is not "
        "implemented for particular op IDs.");
  }
  return context->Context()->AsyncWait();
}

Status EagerServiceImpl::KeepAlive(const KeepAliveRequest* request,
                                   KeepAliveResponse* response) {
  // TODO(nareshmodi): Automated context_id cleaning is not implemented
  return errors::Unimplemented(
      "EagerServiceImpl::KeepAlive is not implemented.");
}

Status EagerServiceImpl::CloseContext(const CloseContextRequest* request,
                                      CloseContextResponse* response) {
  ServerContext* context = nullptr;
  if (!GetServerContext(request->context_id(), &context).ok()) {
    // Swallow the error here.
    return Status::OK();
  }

  core::ScopedUnref context_unref(context);

  mutex_lock l(contexts_mu_);
  contexts_.erase(request->context_id());

  // GetServerContext returns a newly Reffed copy of ServerContext, which is
  // unreffed by context_unref. Additionally, we need to unref it one time since
  // we are releasing it from the map.
  context->Unref();

  return Status::OK();
}

Status EagerServiceImpl::RegisterFunction(
    const RegisterFunctionRequest* request,
    RegisterFunctionResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  return context->Context()->AddFunctionDef(request->function_def());
}

tensorflow::Status EagerServiceImpl::GetServerContext(
    uint64 context_id, ServerContext** server_context) {
  mutex_lock l(contexts_mu_);
  auto iter = contexts_.find(context_id);
  if (iter == contexts_.end()) {
    *server_context = nullptr;
    return errors::InvalidArgument(strings::Printf(
        "Unable to find a context_id matching the specified one "
        "(%lld). Perhaps the worker was restarted?",
        context_id));
  }

  *server_context = iter->second;
  (*server_context)->Ref();
  return Status::OK();
}

}  // namespace eager
}  // namespace tensorflow
