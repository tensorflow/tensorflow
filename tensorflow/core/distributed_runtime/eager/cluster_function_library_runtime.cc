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
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"

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
  absl::Status s =
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
  s = ctx_->GetClient(target, &eager_client);
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

  auto request = std::make_shared<EnqueueRequest>();
  auto response = std::make_shared<EnqueueResponse>();

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

  if (options.function_runs_at_most_once) {
    const auto& fdef_attrs = register_function->function_def().attr();
    auto iter =
        fdef_attrs.find(FunctionLibraryDefinition::kFunctionRunsAtMostOnce);
    if (iter == fdef_attrs.end()) {
      done(errors::Internal("Missing function_runs_at_most_once attribute."));
      return;
    }
    if (!iter->second.b()) {
      done(
          errors::Internal("Unexpected `false` value for "
                           "function_runs_at_most_once attribute."));
      return;
    }
  }

  const absl::optional<std::vector<int>>& ret_indices = options.ret_indices;
  eager_client->EnqueueAsync(
      /*call_opts=*/nullptr, request.get(), response.get(),
      [this, request, response, handle, released_op = released_op.release(),
       target, ret_indices, eager_client = eager_client.get(),
       done](const absl::Status& s) {
        {
          mutex_lock l(mu_);
          *handle = function_data_.size();
          function_data_.emplace_back(target, ret_indices, eager_client,
                                      absl::WrapUnique(released_op));
        }
        done(s);
      });
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle, absl::Span<const Tensor> args,
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
  std::vector<FunctionArg> function_args;
  for (const auto& tensor : args) {
    function_args.push_back(tensor);
  }
  std::vector<FunctionRet>* function_rets = new std::vector<FunctionRet>;
  Run(opts, handle, function_args, function_rets,
      [rets, function_rets, done = std::move(done)](const absl::Status& s) {
        absl::Status status = s;
        if (status.ok()) {
          for (const auto& t : *function_rets) {
            if (t.index() == 0) {
              rets->push_back(std::get<Tensor>(t));
            } else {
              status.Update(
                  errors::Internal("Expect a Tensor as a remote function "
                                   "output but got a TensorShape."));
              break;
            }
          }
        }
        delete function_rets;
        done(status);
      });
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle,
    absl::Span<const FunctionArg> args, std::vector<FunctionRet>* rets,
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

  EagerOperation* op = function_data->op.get();
  if (!op->Inputs().empty()) {
    done(errors::Internal("Inputs should not be set during instantiation."));
    return;
  }

  auto request = std::make_shared<RunComponentFunctionRequest>();
  auto response = std::make_shared<RunComponentFunctionResponse>();
  request->set_context_id(context_id_);
  eager::Operation* remote_op = request->mutable_operation();

  if (function_data->ret_indices.has_value()) {
    for (const int ret_index : function_data->ret_indices.value()) {
      request->add_output_num(ret_index);
    }
  }

  for (const auto& arg : args) {
    if (arg.index() == 0) {
      std::get<Tensor>(arg).AsProtoTensorContent(
          remote_op->add_op_inputs()->mutable_tensor());
    } else {
      remote_op->add_op_inputs()->mutable_remote_handle()->Swap(
          std::get<RemoteTensorHandle*>(arg));
    }
  }

  // The remote component function should use the same op_id as its parent
  // multi-device function's in order to get the global unique op_id generated
  // by the master context.
  if (opts.op_id.has_value()) {
    remote_op->set_id(opts.op_id.value());
  } else {
    remote_op->set_id(kInvalidOpId);
  }
  remote_op->set_is_function(true);
  remote_op->set_is_component_function(true);
  remote_op->set_func_step_id(opts.step_id);
  remote_op->set_name(op->Name());
  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(function_data->target);

  CancellationManager* cm = opts.cancellation_manager;
  CancellationToken token = 0;
  auto call_opts = std::make_shared<CallOptions>();
  call_opts->SetTimeout(
      ctx_->session_options().config.operation_timeout_in_ms());
  if (cm != nullptr) {
    token = cm->get_cancellation_token();
    const bool already_cancelled = !cm->RegisterCallback(
        token,
        [call_opts, request, response, done]() { call_opts->StartCancel(); });
    if (already_cancelled) {
      done(errors::Cancelled("EagerClusterFunctionLibraryRuntime::Run"));
      return;
    }
  }

  // Execute component function on remote worker using RunComponentFunction RPC.
  // Different from executing remote functions with Enqueue, this method runs
  // a function on remote worker without tying up a thread (i.e., pure
  // asynchronously).
  eager_client->RunComponentFunctionAsync(
      call_opts.get(), request.get(), response.get(),
      [request, response, rets, call_opts, cm, token,
       done = std::move(done)](const absl::Status& s) {
        if (cm != nullptr) {
          cm->TryDeregisterCallback(token);
        }
        if (!s.ok()) {
          done(s);
          return;
        }
        if (!response->shape().empty() && !response->tensor().empty()) {
          done(errors::Internal(
              "Both shape and tensor are specified in the same response"));
          return;
        }
        for (const auto& shape : response->shape()) {
          rets->push_back(shape);
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
        done(absl::OkStatus());
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

  auto request = std::make_shared<EnqueueRequest>();
  auto response = std::make_shared<EnqueueResponse>();
  request->set_context_id(context_id_);
  CleanupFunctionOp* cleanup_function =
      request->add_queue()->mutable_cleanup_function();
  cleanup_function->set_step_id(step_id);
  // StreamingEnqueueAsync could be blocking when streaming RPC is disabled.
  // CleanUp() needs to be non-blocking since it would be invoked inside the
  // enqueue done callback of Run(). So we don't use StreamingEnqueueAsync here.
  eager_client->EnqueueAsync(
      /*call_opts=*/nullptr, request.get(), response.get(),
      [request, response, done](const absl::Status& status) { done(status); });
}

DistributedFunctionLibraryRuntime* CreateClusterFLR(
    const uint64 context_id, EagerContext* ctx, WorkerSession* worker_session) {
  return new EagerClusterFunctionLibraryRuntime(
      context_id, ctx, worker_session->remote_device_mgr());
}

}  // namespace eager
}  // namespace tensorflow
