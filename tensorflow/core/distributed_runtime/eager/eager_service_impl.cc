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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_distributed_manager.h"
#include "xla/tsl/distributed_runtime/preemption/preemption_notifier.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
namespace tensorflow {
namespace eager {

namespace {
absl::Status GetNumRetvals(
    FunctionLibraryDefinition* func_lib_def, const string& op_name,
    const google::protobuf::Map<string, tensorflow::AttrValue>& attrs, int* num_retvals) {
  const tensorflow::OpRegistrationData* op_reg_data = nullptr;
  auto status = tensorflow::OpRegistry::Global()->LookUp(op_name, &op_reg_data);
  if (absl::IsNotFound(status)) {
    status = func_lib_def->LookUp(op_name, &op_reg_data);
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
      auto iter = attrs.find(output_arg.type_list_attr());
      if (iter == attrs.end()) {
        return errors::InvalidArgument("Unable to find type_list_attr ",
                                       output_arg.type_list_attr(),
                                       " for Op: ", op_name);
      }
      *num_retvals += iter->second.list().type_size();
    } else {
      *num_retvals += 1;
    }
  }

  return absl::OkStatus();
}

absl::Status GetEagerOperationAndNumRetvals(const Operation& operation,
                                            EagerContext* eager_context,
                                            EagerExecutor* eager_executor,
                                            EagerOperation* eager_op,
                                            int* num_retvals) {
  const char* name = operation.name().c_str();  // Shorthand
  std::optional<tensorflow::EagerFunctionParams> remote_func_params =
      std::nullopt;
  FunctionLibraryDefinition* func_lib_def;
  if (operation.is_function()) {
    if (operation.is_component_function()) {
      func_lib_def =
          eager_context->GetComponentFunctionFunctionLibraryDefinition(
              operation.name());
      if (func_lib_def == nullptr) {
        return absl::InternalError(
            absl::StrCat("Could not find function library for registered "
                         "component function: ",
                         operation.name()));
      }
      remote_func_params = {operation.id(), /*is_component_function=*/true,
                            operation.func_step_id(), func_lib_def};
    } else {
      func_lib_def = eager_context->FuncLibDef();
      remote_func_params = {operation.id(), /*is_component_function=*/false,
                            std::nullopt, /*func_lib_def=*/nullptr};
    }
  } else {
    func_lib_def = eager_context->FuncLibDef();
  }
  TF_RETURN_IF_ERROR(eager_op->Reset(name, operation.device().c_str(), false,
                                     eager_executor, remote_func_params));

  {
    tsl::profiler::TraceMe activity("EagerService:RemoteTensorHandleInternal",
                                    tsl::profiler::TraceMeLevel::kVerbose);
    for (const auto& input : operation.op_inputs()) {
      tensorflow::TensorHandle* handle;
      if (input.has_remote_handle()) {
        TF_RETURN_IF_ERROR(
            eager_context->RemoteMgr()->DeserializeRemoteTensorHandle(
                input.remote_handle(), &handle));
        TF_RETURN_IF_ERROR(eager_op->AddInput(handle));
      } else {
        Tensor tensor;
        if (!ParseTensorProtoToTensor(input.tensor(), &tensor)) {
          return errors::InvalidArgument("Invalid TensorProto: ",
                                         input.tensor().DebugString());
        } else {
          handle = TensorHandle::CreateLocalHandle(std::move(tensor), nullptr,
                                                   nullptr, eager_context);
          TF_RETURN_IF_ERROR(eager_op->AddInput(handle));
        }
      }
      // Unref handle since it has a ref as an input now.
      handle->Unref();
    }
  }

  for (const auto& attr : operation.attrs()) {
    eager_op->MutableAttrs()->Set(attr.first, attr.second);
  }

  // TODO(nareshmodi): Consider caching this.
  return GetNumRetvals(func_lib_def, operation.name(), operation.attrs(),
                       num_retvals);
}

absl::Status TensorHandleProto(TensorHandle* handle, TensorProto* proto) {
  const tensorflow::Tensor* t = nullptr;
  TF_RETURN_IF_ERROR(handle->Tensor(&t));
  t->AsProtoTensorContent(proto);
  return absl::OkStatus();
}

absl::Status TensorHandleShape(TensorHandle* handle, TensorShapeProto* proto) {
  const tensorflow::Tensor* t = nullptr;

  // TODO(nareshmodi): This call makes async calls sync calls. Fix this.
  if (handle->Type() == TensorHandle::LOCAL) {
    TF_RETURN_IF_ERROR(handle->Tensor(&t));

    t->shape().AsProto(proto);
  } else {
    TensorShape shape;
    TF_RETURN_IF_ERROR(handle->Shape(&shape));
    shape.AsProto(proto);
  }

  return absl::OkStatus();
}

absl::Status AddOpRetvalsToResponse(
    EagerContext* eager_context, int op_id, int num_retvals,
    const std::vector<int32>& output_nums, TensorHandle** retvals,
    std::function<TensorProto*()> add_tensor_proto_fn,
    std::function<TensorShapeProto*()> add_shape_proto_fn,
    std::function<string*()> add_device_fn = nullptr) {
  // retvals hold references to the allocated output tensor handles. If errors
  // happen with adding some results to the response, aggregate the status in sg
  // instead of directly returning the error, to make sure unref or ownership
  // transfer completes for the rest of output tensor handles.
  StatusGroup sg;
  if (op_id == kInvalidOpId) {
    // Copy the output tensors back along with the response, since the op id
    // is invalid which cannot be added to RemoteMgr.
    for (int i = 0; i < num_retvals; i++) {
      sg.Update(TensorHandleProto(retvals[i], add_tensor_proto_fn()));
      retvals[i]->Unref();
    }
  } else {
    for (int i = 0; i < num_retvals; i++) {
      sg.Update(TensorHandleShape(retvals[i], add_shape_proto_fn()));
      if (add_device_fn) {
        Device* device = retvals[i]->device();
        *add_device_fn() = device ? device->name() : "";
      }
      if (retvals[i]->Type() == TensorHandle::REMOTE) {
        retvals[i]->Unref();
      } else {
        const int output_num = output_nums.empty() ? i : output_nums.at(i);
        eager_context->RemoteMgr()->AddOperationOutput(retvals[i], op_id,
                                                       output_num);
      }
    }
  }
  return sg.as_summary_status();
}

absl::Status ResetAgentAndConnectToCoordinationService(
    tsl::CoordinationServiceAgent* coord_agent) {
  // The error state should already be consumed when a new context is
  // created. It should be fine to reset the agent.
  if (coord_agent->IsError()) {
    const absl::Status s = coord_agent->Reset();
    if (!s.ok()) {
      LOG(ERROR) << "Coordination Service agent reset failed " << s;
      return s;
    }
  }
  // In the scenario of PS strategy, the setup is single client and the error
  // cannot be propagated. As a result, Coordination Service agent can still
  // have the status of being connected. We should not let it connect again.
  if (!coord_agent->IsConnected()) {
    const absl::Status s = coord_agent->Connect();
    if (!s.ok()) {
      LOG(ERROR) << "Coordination Service agent connect failed " << s;
      return s;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status EagerServiceImpl::CreateContext(
    const CreateContextRequest* request, CreateContextResponse* response) {
  bool update_collective_executor_mgr = false;
  {
    mutex_lock l(contexts_mu_);
    if (contexts_.empty()) {
      update_collective_executor_mgr = true;
    } else {
      auto context_it = contexts_.find(request->context_id());
      if (context_it != contexts_.end()) {
        if (request->context_view_id() <
            context_it->second->Context()->GetContextViewId()) {
          return errors::InvalidArgument("EagerService:CreateContext failed. ",
                                         "Context id: <", request->context_id(),
                                         "> already exists.");
        } else {
          // For existing context with a stale context_view_id, close the old
          // one and recreate with new view id. This is likely due to the worker
          // disconnected and then reconnected after one or more cluster
          // updates.
          context_it->second->Unref();
          contexts_.erase(context_it);
        }
      }
    }
  }
  // make sure env_ , env_->rendezvous_mgr available
  if (env_ == nullptr || env_->rendezvous_mgr == nullptr) {
    return tensorflow::errors::Internal(
        "invalid eager env_ or env_->rendezvous_mgr.");
  }
  if (request->clear_existing_contexts()) {
    // Cleanup state from WorkerEnv
    for (auto* device : env_->device_mgr->ListDevices()) {
      device->ClearResourceMgr();
    }
    env_->rendezvous_mgr->CleanupAll();
    env_->collective_executor_mgr->CleanupAll();
    TF_RETURN_IF_ERROR(env_->session_mgr->DeleteAllSessions());

    // Cleanup existing contexts if any.
    std::unordered_map<uint64, ServerContext*> tmp_contexts;
    {
      mutex_lock l(contexts_mu_);
      if (!contexts_.empty()) {
        std::swap(tmp_contexts, contexts_);
      }
    }
    for (auto& context : tmp_contexts) {
      context.second->Unref();
    }
  }

  tsl::core::RefCountPtr<RemoteRendezvous> r =
      env_->rendezvous_mgr->Find(request->context_id());
  auto session_name =
      tensorflow::strings::StrCat("eager_", request->context_id());
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Creating context on /job:" << request->server_def().job_name()
            << "/task:" << request->server_def().task_index();
    for (const auto& da : request->cluster_device_attributes()) {
      VLOG(2) << "    " << da.name();
    }
  }
  TF_RETURN_IF_ERROR(env_->session_mgr->CreateSession(
      session_name, request->server_def(), request->cluster_device_attributes(),
      request->server_def().default_session_config().isolate_session_state()));
  int64_t context_id = request->context_id();
  std::function<void()> session_destroyer = [this, context_id, session_name]() {
    env_->rendezvous_mgr->Cleanup(context_id);
    auto s = env_->session_mgr->DeleteSession(session_name);
    if (!s.ok()) {
      LOG(WARNING) << "Failed to destroy worker session '" << session_name
                   << "' due to " << s.message();
    }
  };

  std::shared_ptr<WorkerSession> worker_session;
  TF_RETURN_IF_ERROR(env_->session_mgr->WorkerSessionForSession(
      session_name, &worker_session));

  tensorflow::DeviceMgr* device_mgr = worker_session->device_mgr();

  // Initialize remote tensor communication based on worker session.
  TF_RETURN_IF_ERROR(r->Initialize(worker_session.get()));
  // Set the rendezvous as context-global instance for eager op-by-op execution.
  r->SetRemoteEagerContextDefault();

  std::function<tsl::core::RefCountPtr<Rendezvous>(const int64_t)>
      rendezvous_creator = [worker_session, this](const int64_t step_id) {
        tsl::core::RefCountPtr<RemoteRendezvous> r =
            env_->rendezvous_mgr->Find(step_id);
        r->Initialize(worker_session.get()).IgnoreError();
        return r;
      };

  LOG(INFO) << "Creating " << (request->async() ? "async" : "sync")
            << " eager service context with rendezvous_id on host "
            << port::Hostname() << " " << worker_session->worker_name();
  SessionOptions opts;
  opts.config = request->server_def().default_session_config();

  LOG(INFO) << "SessionOptions: " << opts.config.DebugString();

  if (update_collective_executor_mgr) {
    // Replace the collective execution manager in worker env. This ensures
    // this newly create EagerContext and the worker service agrees about the
    // leader and type (RPC / local) of the collectives.
    // Other EagerContexts are broken if they disagree with the worker service.
    env_->collective_executor_mgr = CreateProdRpcCollectiveExecutorMgr(
        opts.config, device_mgr, MaybeCreateNcclCommunicator(opts.config),
        worker_session->worker_cache(), worker_session->worker_name());
  }

  tensorflow::EagerContext* ctx = new tensorflow::EagerContext(
      opts, tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      request->async(), device_mgr, false, std::move(r),
      worker_session->cluster_flr(), env_->collective_executor_mgr.get());

  // Ownership will be transferred to the ServerContext, or else in an error
  // case ctx will be deleted by this unref.
  core::ScopedUnref unref_ctx(ctx);

  std::vector<string> remote_workers;
  worker_session->worker_cache()->ListWorkers(&remote_workers);
  remote_workers.erase(std::remove(remote_workers.begin(), remote_workers.end(),
                                   worker_session->worker_name()),
                       remote_workers.end());

  std::unique_ptr<tensorflow::eager::EagerClientCache> remote_eager_workers;
  TF_RETURN_IF_ERROR(worker_session->worker_cache()->GetEagerClientCache(
      &remote_eager_workers));
  DistributedFunctionLibraryRuntime* cluster_flr =
      eager::CreateClusterFLR(request->context_id(), ctx, worker_session.get());

  auto remote_mgr =
      std::make_unique<tensorflow::eager::RemoteMgr>(/*is_master=*/false, ctx);
  absl::Status s = ctx->InitializeRemoteWorker(
      std::move(remote_eager_workers), worker_session->remote_device_mgr(),
      remote_workers, request->context_id(), request->context_view_id(),
      std::move(rendezvous_creator), cluster_flr, std::move(remote_mgr),
      std::move(session_destroyer));
  if (!s.ok()) {
    VLOG(1) << "EagerContext::InitializeRemoteWorker failed with "
            << s.ToString();
    return s;
  }

#if !defined(IS_MOBILE_PLATFORM)
  const auto& config = request->server_def().default_session_config();
  const bool enable_coordination =
      !config.experimental().coordination_config().service_type().empty();
  if (enable_coordination) {
    auto dist_mgr = std::make_unique<EagerContextDistributedManager>(ctx);
    auto coord_agent = env_->session_mgr->GetCoordinationServiceAgent();
    dist_mgr->SetCoordinationServiceAgent(coord_agent);
    // TODO(b/254356090): See if enabling health check needs to be inside the
    // Coordination Service.
    if (config.experimental().coordination_config().enable_health_check()) {
      TF_RETURN_IF_ERROR(
          ResetAgentAndConnectToCoordinationService(coord_agent));
    }
    auto preemption_notifier =
        tsl::PreemptionNotifier::CreatePreemptionNotifier("sigterm",
                                                          Env::Default());
    preemption_notifier->WillBePreemptedAtAsync(
        [coord_agent](absl::StatusOr<absl::Time> time_or_status) {
          if (time_or_status.ok()) {
            const auto coord_task = coord_agent->GetOwnTask().value();
            absl::Status s = coord_agent->InsertKeyValue(
                "TF_DEFAULT_PREEMPTION_NOTICE_KEY",
                absl::StrCat("/job:", coord_task.job_name(),
                             "/task:", coord_task.task_id()));
            if (!s.ok()) {
              // Dev note: `ALREADY_EXISTS` errors are expected if multiple
              // workers receive a SIGTERM.
              VLOG(3) << "Preemption not exported to coordination service: "
                      << s;
            }
          }
        });
    dist_mgr->SetPreemptionNotifier(std::move(preemption_notifier));
    ctx->SetDistributedManager(std::move(dist_mgr));
  }
#endif  // !IS_MOBILE_PLATFORM

  std::vector<DeviceAttributes> device_attributes;
  device_mgr->ListDeviceAttributes(&device_attributes);

  for (const auto& da : device_attributes) {
    *response->add_device_attributes() = da;
  }
  {
    mutex_lock l(contexts_mu_);
    auto context_it = contexts_.find(request->context_id());
    if (context_it != contexts_.end()) {
      return errors::InvalidArgument("EagerService:CreateContext failed. ",
                                     "Context id: <", request->context_id(),
                                     "> already exists.");
    }
    contexts_.emplace(request->context_id(),
                      new ServerContext(ctx, request->keep_alive_secs(), env_));
  }

  return absl::OkStatus();
}

absl::Status EagerServiceImpl::UpdateContext(
    const UpdateContextRequest* request, UpdateContextResponse* response) {
  // make sure env_ , env_->rendezvous_mgr available
  if (env_ == nullptr || env_->rendezvous_mgr == nullptr) {
    return tensorflow::errors::Internal(
        "invalid eager env_ or env_->rendezvous_mgr.");
  }

  // Find the context to update by the requested context_id
  ServerContext* server_context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &server_context));
  core::ScopedUnref context_unref(server_context);

  tensorflow::EagerContext* ctx = server_context->Context();
  if (request->context_view_id() != ctx->GetContextViewId() + 1) {
    return errors::InvalidArgument(
        "EagerService:UpdateContext failed. Context id: <",
        request->context_id(), "> currently at view #", ctx->GetContextViewId(),
        " but received update request at view #", request->context_view_id(),
        ". View id should only be continuously incremented.");
  }
  if (request->cluster_device_attributes_size() == 0) {
    // In this case, the client indicates that the updated `server_def` and
    // device info is irrelevant to this worker, since it is not connected to
    // the updated ones (likely due to device filter settings). The worker
    // simply needs to update view ID and does not update other internal state.
    ctx->IncrementContextViewId();
    VLOG(1) << "Processing simplified UpdateContextRequest on "
            << ctx->HostCPU()->name();
    return absl::OkStatus();
  }

  auto session_name =
      tensorflow::strings::StrCat("eager_", request->context_id());

  TF_RETURN_IF_ERROR(
      env_->session_mgr->UpdateSession(session_name, request->server_def(),
                                       request->cluster_device_attributes()));

  std::shared_ptr<WorkerSession> worker_session;
  TF_RETURN_IF_ERROR(env_->session_mgr->WorkerSessionForSession(
      session_name, &worker_session));

  const tensorflow::DeviceMgr* device_mgr = worker_session->device_mgr();

  std::vector<string> remote_workers;
  worker_session->worker_cache()->ListWorkers(&remote_workers);
  remote_workers.erase(std::remove(remote_workers.begin(), remote_workers.end(),
                                   worker_session->worker_name()),
                       remote_workers.end());
  VLOG(1) << "On existing server " << worker_session->worker_name()
          << " updating remote workers";
  if (VLOG_IS_ON(2)) {
    for (const string& rw : remote_workers) {
      VLOG(2) << "Remote worker " << rw;
    }
  }

  std::unique_ptr<tensorflow::eager::EagerClientCache> remote_eager_workers;
  TF_RETURN_IF_ERROR(worker_session->worker_cache()->GetEagerClientCache(
      &remote_eager_workers));

  ctx->ClearCachesAndThreadExecutors();
  absl::Status s = ctx->UpdateRemoteWorker(
      std::move(remote_eager_workers), remote_workers, request->context_id());
  if (!s.ok()) {
    VLOG(1) << "EagerContext::UpdateRemoteWorker failed with " << s.ToString();
    return s;
  }

#if !defined(IS_MOBILE_PLATFORM)
  const auto& config = request->server_def().default_session_config();
  const bool should_connect =
      !config.experimental().coordination_config().service_type().empty() &&
      config.experimental().coordination_config().enable_health_check();
  if (should_connect) {
    auto coord_agent = env_->session_mgr->GetCoordinationServiceAgent();
    TF_RETURN_IF_ERROR(ResetAgentAndConnectToCoordinationService(coord_agent));
  }
#endif  // !IS_MOBILE_PLATFORM

  std::vector<DeviceAttributes> device_attributes;
  device_mgr->ListDeviceAttributes(&device_attributes);

  for (const auto& da : device_attributes) {
    *response->add_device_attributes() = da;
  }

  return absl::OkStatus();
}

absl::Status EagerServiceImpl::CreateMasterContext(
    const tensorflow::uint64 context_id, EagerContext* context) {
  {
    mutex_lock l(contexts_mu_);
    auto iter = contexts_.find(context_id);
    if (iter != contexts_.end()) {
      return errors::InvalidArgument(
          "EagerService:CreateMasterContext failed. ", "Context id: <",
          context_id, "> already exists.");
    }
  }
  ServerContext* server_context =
      ServerContext::CreateMasterContext(context, env_);
  mutex_lock l(contexts_mu_);
  contexts_.emplace(context_id, server_context);
  return absl::OkStatus();
}

void EagerServiceImpl::RunComponentFunction(
    CallOptions* call_opts, const RunComponentFunctionRequest* request,
    RunComponentFunctionResponse* response, StatusCallback done) {
  ServerContext* context = nullptr;
  absl::Status s = GetServerContext(request->context_id(), &context);
  if (!s.ok()) {
    done(s);
    return;
  }
  core::ScopedUnref context_unref(context);

  auto& operation = request->operation();
  // This codepath should only be triggered for executing component function
  if (!operation.is_function() || !operation.is_component_function()) {
    done(errors::Internal(
        "RunComponentFunction request can only be used to execute "
        "component functions."));
    return;
  }

  EagerContext* eager_context = context->Context();
  EagerExecutor* eager_executor = &eager_context->Executor();

  EagerOperation* op = new EagerOperation(eager_context);
  int* num_retvals = new int(0);
  s = GetEagerOperationAndNumRetvals(operation, eager_context, eager_executor,
                                     op, num_retvals);
  if (!s.ok()) {
    delete num_retvals;
    delete op;
    done(s);
    return;
  }
  if (!op->IsLocal()) {
    delete num_retvals;
    delete op;
    done(errors::Internal(
        "Received RunComponentFunction request with remote function device. "));
    return;
  }
  s = op->SetAttrBool("is_component_function", true);
  if (!s.ok()) {
    delete num_retvals;
    delete op;
    done(errors::Internal("Error setting is_component_function attribute: ",
                          s.message()));
    return;
  }

  auto* retvals = new absl::FixedArray<TensorHandle*>(*num_retvals);
  VLOG(3) << "ServerContext: Calling EagerLocalExecuteAsync for op "
          << operation.id();
  std::vector<int32> output_nums;
  for (const int32_t output_num : request->output_num()) {
    output_nums.push_back(output_num);
  }

  auto cm = std::make_shared<CancellationManager>();
  op->SetCancellationManager(cm.get());
  call_opts->SetCancelCallback([cm] { cm->StartCancel(); });

  context->Ref();
  EagerLocalExecuteAsync(
      op, retvals->data(), num_retvals,
      [op, op_id = operation.id(), num_retvals, retvals, output_nums, cm,
       call_opts, response, eager_context, context,
       done = std::move(done)](const absl::Status& status) {
        call_opts->ClearCancelCallback();
        auto wrapped_done = [&](const absl::Status& status) {
          context->Unref();
          done(status);
          delete op;
          delete num_retvals;
          delete retvals;
        };
        if (!status.ok()) {
          wrapped_done(status);
          return;
        }
        // The output device of a component function is the component device
        // which is known on the default device of it's parent function.
        wrapped_done(AddOpRetvalsToResponse(
            eager_context, op_id, *num_retvals, output_nums, retvals->data(),
            [response] { return response->add_tensor(); },
            [response] { return response->add_shape(); }));
      });
}

absl::Status EagerServiceImpl::ExecuteOp(CallOptions* call_opts,
                                         const Operation& operation,
                                         EagerContext* eager_context,
                                         EagerExecutor* eager_executor,
                                         QueueResponse* queue_response) {
  tensorflow::EagerOperation op(eager_context);
  int num_retvals = 0;
  TF_RETURN_IF_ERROR(GetEagerOperationAndNumRetvals(
      operation, eager_context, eager_executor, &op, &num_retvals));

  auto cm = std::make_shared<CancellationManager>();
  if (call_opts) {
    op.SetCancellationManager(cm.get());
    call_opts->SetCancelCallback([cm] { cm->StartCancel(); });
  }

  absl::FixedArray<tensorflow::TensorHandle*> retvals(num_retvals);
  VLOG(3) << "ServerContext: Calling EagerExecute for op " << operation.id();
  TF_RETURN_IF_ERROR(op.Execute(
      absl::MakeSpan(
          reinterpret_cast<tensorflow::AbstractTensorHandle**>(retvals.data()),
          num_retvals),
      &num_retvals));

  std::function<string*()> add_device_fn = nullptr;
  // Send the output devices of a function back to let a client know where the
  // outputs are. For a primitive op, an output devics is the op device which is
  // known on a client.
  if (op.is_function()) {
    add_device_fn = [queue_response] { return queue_response->add_device(); };
  }

  return AddOpRetvalsToResponse(
      eager_context, operation.id(), num_retvals, /*output_nums=*/{},
      retvals.data(), [queue_response] { return queue_response->add_tensor(); },
      [queue_response] { return queue_response->add_shape(); },
      std::move(add_device_fn));
}

absl::Status EagerServiceImpl::Enqueue(CallOptions* call_opts,
                                       const EnqueueRequest* request,
                                       EnqueueResponse* response,
                                       uint64 stream_id) {
  tsl::profiler::TraceMe activity(
      [&] {
        return absl::StrCat(
            "EagerService:Enqueue#debug_str=", request->DebugString(), "#");
      },
      tsl::profiler::TraceMeLevel::kInfo);
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  EagerExecutor& executor =
      stream_id == kInvalidStreamId
          ? context->Context()->Executor()
          : context->Context()->RemoteMgr()->GetOrCreateExecutorForStream(
                stream_id);
  absl::Status s;
  for (const auto& item : request->queue()) {
    auto* queue_response = response->add_queue_response();
    if (item.has_operation()) {
      s = ExecuteOp(call_opts, item.operation(), context->Context(), &executor,
                    queue_response);
    } else if (item.has_handle_to_decref()) {
      auto handle_to_decref = std::make_unique<RemoteTensorHandleInternal>(
          item.handle_to_decref());
      auto node = std::make_unique<ClientTensorHandleDeleteNode>(
          context, std::move(handle_to_decref));
      s = context->Context()->Executor().AddOrExecute(std::move(node));
    } else if (item.has_send_tensor()) {
      s = SendTensor(item.send_tensor(), context->Context());
    } else if (item.has_send_packed_handle()) {
      s = SendPackedHandle(item.send_packed_handle(), context->Context());
    } else if (item.has_register_function()) {
      s = RegisterFunction(item.register_function(), context->Context());
    } else if (item.has_remove_function()) {
      s = RemoveFunction(item.remove_function(), context->Context());
    } else if (item.has_cleanup_function()) {
      s = CleanupFunction(item.cleanup_function());
    } else {
      DCHECK(item.has_sync_remote_executor_for_stream());
      s = executor.WaitForAllPendingNodes();
    }

    if (!s.ok()) {
      if (stream_id != kInvalidStreamId) {
        context->Context()->RemoteMgr()->DeleteExecutorForStream(stream_id);
      }
      return s;
    }
  }

  return absl::OkStatus();
}

absl::Status EagerServiceImpl::WaitQueueDone(
    const WaitQueueDoneRequest* request, WaitQueueDoneResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  if (request->op_id_size() > 0) {
    return errors::Unimplemented(
        "EagerServiceImpl::WaitQueueDone is not "
        "implemented for particular op IDs.");
  }
  return context->Context()->Executor().WaitForAllPendingNodes();
}

absl::Status EagerServiceImpl::KeepAlive(const KeepAliveRequest* request,
                                         KeepAliveResponse* response) {
  ServerContext* context = nullptr;
  TF_RETURN_IF_ERROR(GetServerContext(request->context_id(), &context));
  core::ScopedUnref context_unref(context);

  tensorflow::EagerContext* ctx = context->Context();
  response->set_context_view_id(ctx->GetContextViewId());
  return absl::OkStatus();
}

absl::Status EagerServiceImpl::CloseContext(const CloseContextRequest* request,
                                            CloseContextResponse* response) {
  ServerContext* context = nullptr;
  if (!GetServerContext(request->context_id(), &context).ok()) {
    // Swallow the error here.
    return absl::OkStatus();
  }
  core::ScopedUnref context_unref(context);

  if (request->context_view_id() < context->Context()->GetContextViewId()) {
    // Swallow the error here.
    LOG(INFO) << "Ignoring CloseContext request with a stale context_view_id "
              << request->context_view_id() << "  for context_id "
              << request->context_id() << ". The current context_view_id is "
              << context->Context()->GetContextViewId() << ".";
    return absl::OkStatus();
  }

  mutex_lock l(contexts_mu_);
  contexts_.erase(request->context_id());

  // GetServerContext returns a newly Reffed copy of ServerContext, which is
  // unreffed by context_unref. Additionally, we need to unref it one time since
  // we are releasing it from the map.
  context->Unref();

  return absl::OkStatus();
}

absl::Status EagerServiceImpl::RegisterFunction(
    const RegisterFunctionOp& register_function, EagerContext* eager_context) {
  // If the function is a component of a multi-device function, we only need to
  // register it locally.
  if (register_function.is_component_function()) {
    return eager_context->AddComponentFunction(register_function.function_def(),
                                               register_function.library());
  } else {
    return eager_context->AddFunctionDef(register_function.function_def(),
                                         register_function.library(),
                                         /*add_to_local_only=*/false);
  }
}

absl::Status EagerServiceImpl::RemoveFunction(
    const RemoveFunctionOp& remove_function, EagerContext* eager_context) {
  return eager_context->RemoveFunction(remove_function.function_name());
}

absl::Status EagerServiceImpl::CleanupFunction(
    const CleanupFunctionOp& cleanup_function) {
  env_->rendezvous_mgr->Cleanup(cleanup_function.step_id());
  return absl::OkStatus();
}

absl::Status EagerServiceImpl::SendTensor(const SendTensorOp& send_tensor,
                                          EagerContext* eager_context) {
  absl::InlinedVector<tensorflow::TensorHandle*, 2UL> tensors;
  for (const auto& tensor_proto : send_tensor.tensors()) {
    Tensor tensor;
    if (!tensor.FromProto(tensor_proto)) {
      return errors::InvalidArgument("Unable to parse tensor proto");
    }

    TensorHandle* tensor_handle = TensorHandle::CreateLocalHandle(
        std::move(tensor), nullptr, nullptr, eager_context);
    TensorHandle* copied_handle = nullptr;
    Device* device;
    TF_RETURN_IF_ERROR(eager_context->FindDeviceFromName(
        send_tensor.device_name().c_str(), &device));
    TF_RETURN_IF_ERROR(EagerCopyToDevice(tensor_handle, eager_context,
                                         &eager_context->Executor(), device,
                                         false, &copied_handle));
    tensors.push_back(copied_handle);
    tensor_handle->Unref();
  }

  eager_context->RemoteMgr()->AddOperationOutputs(tensors, send_tensor.op_id());

  return absl::OkStatus();
}

absl::Status EagerServiceImpl::SendPackedHandle(
    const SendPackedHandleOp& send_packed_handle, EagerContext* eager_context) {
  if (send_packed_handle.handles().empty()) {
    return errors::InvalidArgument("Handles should not be empty.");
  }

  std::vector<tensorflow::TensorHandle*> handles;
  handles.resize(send_packed_handle.handles_size());
  for (int i = 0; i < send_packed_handle.handles_size(); ++i) {
    const auto& item = send_packed_handle.handles(i);
    if (item.has_local_handle()) {
      Tensor tensor;
      if (!ParseTensorProtoToTensor(item.local_handle().tensor(), &tensor)) {
        return errors::InvalidArgument(
            "Invalid TensorProto: ",
            item.local_handle().tensor().DebugString());
      }
      Device* op_device = nullptr;
      TF_RETURN_IF_ERROR(eager_context->FindDeviceFromName(
          item.local_handle().device().c_str(), &op_device));
      handles[i] = TensorHandle::CreateLocalHandle(
          std::move(tensor), /*d=*/nullptr, op_device, eager_context);
    } else {
      TF_RETURN_IF_ERROR(
          eager_context->RemoteMgr()->DeserializeRemoteTensorHandle(
              item.remote_handle(), &handles[i]));
    }
  }

  tensorflow::TensorHandle* packed_handle = nullptr;
  std::vector<tensorflow::TensorHandle*> handles_to_pack = handles;
  // Create a unshaped packed TensorHandle.
  TF_RETURN_IF_ERROR(TensorHandle::CreatePackedHandle(
      std::move(handles_to_pack), handles.at(0)->dtype, TensorShape(),
      send_packed_handle.device_name(), eager_context, &packed_handle));

  for (auto* h : handles) {
    // Unref handle since it has a ref in the packed handle now.
    h->Unref();
  }

  eager_context->RemoteMgr()->AddOperationOutputs({packed_handle},
                                                  send_packed_handle.op_id());
  return absl::OkStatus();
}

absl::Status EagerServiceImpl::GetServerContext(
    uint64 context_id, ServerContext** server_context) {
  tf_shared_lock l(contexts_mu_);
  auto iter = contexts_.find(context_id);
  if (iter == contexts_.end()) {
    *server_context = nullptr;
    return errors::Aborted(strings::Printf(
        "Unable to find a context_id matching the specified one "
        "(%llu). Perhaps the worker was restarted, or the context was GC'd?",
        static_cast<unsigned long long>(context_id)));
  }

  *server_context = iter->second;
  (*server_context)->Ref();

  (*server_context)->RecordAccess();

  return absl::OkStatus();
}

}  // namespace eager
}  // namespace tensorflow
