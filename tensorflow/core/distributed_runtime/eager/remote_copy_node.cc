/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/eager/remote_copy_node.h"

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace eager {

namespace {

void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(op->DeviceName());
}

absl::Status CreateUncachedKernelAndDeviceOp(
    EagerOperation* op, core::RefCountPtr<KernelAndDevice>* kernel) {
  EagerContext& ctx = op->EagerContext();
  Device* device = std::get<Device*>(op->Device());

  FunctionLibraryRuntime* flr = ctx.func_lib(device);
  if (flr == nullptr) {
    return errors::Unavailable(
        "Unable to find a FunctionLibraryRuntime corresponding to device ",
        device->name());
  }

  auto runner = (flr->runner() != nullptr) ? flr->runner() : ctx.runner();
  kernel->reset(new KernelAndDeviceOp(ctx.GetRendezvous(), ctx.LogMemory(), flr,
                                      runner, ctx.GetCollectiveExecutorHandle(),
                                      ctx.HostCPU()));

  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  return kernel->get()->Init(ctx.LogDevicePlacement(), ndef,
                             /*graph_collector=*/nullptr,
                             /*eager_func_params=*/std::nullopt);
}

// This gets a unique wire ID. We add a random identifier so that if the
// worker has other clients that it is servicing, we don't have any collision.
string GetUniqueWireID() {
  static tensorflow::uint64 random_seed = random::New64();
  static tensorflow::mutex wireid_mutex(tensorflow::LINKER_INITIALIZED);
  static std::atomic<int64_t> wire_id;
  return strings::StrCat(random_seed, "_", wire_id++);
}

}  // namespace

RemoteCopyNode::RemoteCopyNode(EagerContext* ctx, EagerExecutor* executor,
                               TensorHandle* src, TensorHandle* dst,
                               Device* recv_device, uint64 recv_op_id)
    : AsyncEagerNode(),
      src_(src),
      ctx_(ctx),
      executor_(executor),
      send_device_(src->DeviceOrHostCPU(*ctx)),
      recv_device_(recv_device),
      wire_id_(GetUniqueWireID()),
      recv_op_id_(recv_op_id),
      captured_state_(std::make_shared<CapturedSharedState>(dst)),
      started_(false) {
  DCHECK(!send_device_->IsLocal() || !recv_device_->IsLocal());
  src_->Ref();
  ctx_->Ref();
}

RemoteCopyNode::~RemoteCopyNode() {
  src_->Unref();
  ctx_->Unref();
}

absl::Status RemoteCopyNode::RunLocalSend(EagerOperation* op) {
  TF_RETURN_IF_ERROR(executor_->status());

  TF_RETURN_IF_ERROR(op->AddInput(src_));

  core::RefCountPtr<KernelAndDevice> kernel;
  TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(op, &kernel));

  EagerKernelArgs args(1);
  Device* d = ctx_->CanonicalDevice(std::get<Device*>(op->Device()));
  TF_RETURN_IF_ERROR(src_->TensorValue(d, args.MutableInput(0)));
  tsl::CoordinationServiceAgent* coord_agent = nullptr;
  if (ctx_->GetDistributedManager() != nullptr)
    coord_agent = ctx_->GetDistributedManager()->GetCoordinationServiceAgent();

  return kernel->Run(/*step_container=*/nullptr, args, /*outputs=*/nullptr,
                     /*cancellation_manager=*/nullptr,
                     /*eager_func_params=*/std::nullopt,
                     /*stack_trace=*/std::nullopt, coord_agent);
}

void RemoteCopyNode::StartSend() {
  // TODO(gjn): We should consider just using the low-level SendOp::Compute()
  // functionality here instead of constructing an Op.
  EagerOperation op(ctx_);
  absl::Status status = op.Reset("_Send", /*device_name=*/nullptr,
                                 /*remote=*/false, /*executor=*/nullptr);
  if (!status.ok()) {
    captured_state_->SetSendStatus(status);
    return;
  }

  op.SetDevice(send_device_);

  op.MutableAttrs()->Set("tensor_name", wire_id_);
  op.MutableAttrs()->Set("send_device", send_device_->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64_t>(send_device_->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device_->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("T", src_->dtype);

  DCHECK(send_device_ != nullptr);

  if (send_device_->IsLocal()) {
    status = RunLocalSend(&op);
    captured_state_->SetSendStatus(status);
    return;
  } else {
    // Prepare the request
    EnqueueRequest request;
    request.set_context_id(ctx_->GetContextId());
    auto* remote_op = request.add_queue()->mutable_operation();
    status = ctx_->RemoteMgr()->SerializeRemoteTensorHandle(
        src_, /*wait_until_ready=*/false,
        remote_op->add_op_inputs()->mutable_remote_handle(), src_->device());
    if (!status.ok()) {
      captured_state_->SetSendStatus(status);
      return;
    }

    PrepareRemoteOp(remote_op, &op);
    remote_op->set_id(ctx_->RemoteMgr()->NextOpId());

    // Issue the RPC
    core::RefCountPtr<eager::EagerClient> eager_client;
    status = ctx_->GetClient(send_device_, &eager_client);
    if (!status.ok()) {
      captured_state_->SetSendStatus(status);
      return;
    }

    const std::shared_ptr<CapturedSharedState>& captured_state =
        captured_state_;
    EnqueueResponse* response = new EnqueueResponse;
    // If StartRecv fails very quickly, `this` can be destroyed before the
    // callback below is executed. So, we can't capture `this`.
    eager_client->StreamingEnqueueAsync(
        ctx_->Executor().StreamingEnqueue(),
        /*call_opts=*/nullptr, &request, response,
        [response, captured_state](const absl::Status& s) {
          captured_state->SetSendStatus(s);
          if (!s.ok()) {
            captured_state->recv_cancellation()->StartCancel();
          }
          delete response;
        });
  }
}

absl::Status RemoteCopyNode::RunLocalRecv(EagerOperation* op,
                                          std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(executor_->status());

  core::RefCountPtr<KernelAndDevice> kernel;
  TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(op, &kernel));

  EagerKernelArgs args;
  std::vector<EagerKernelRet> rets;
  tsl::CoordinationServiceAgent* coord_agent = nullptr;
  if (ctx_->GetDistributedManager() != nullptr)
    coord_agent = ctx_->GetDistributedManager()->GetCoordinationServiceAgent();
  TF_RETURN_IF_ERROR(kernel->Run(/*step_container*/ nullptr, args, &rets,
                                 captured_state_->recv_cancellation(),
                                 /*eager_func_params=*/std::nullopt,
                                 /*stack_trace=*/std::nullopt, coord_agent));
  outputs->clear();
  for (const auto& ret : rets) {
    if (ret.index() == 0) {
      outputs->push_back(std::get<Tensor>(ret));
    } else {
      return errors::Internal(
          "Expect to receive a Tensor but got a TensorShape.");
    }
  }
  return absl::OkStatus();
}

void RemoteCopyNode::RunRemoteRecv(EagerOperation* op, StatusCallback done) {
  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  auto* remote_op = request.add_queue()->mutable_operation();
  PrepareRemoteOp(remote_op, op);
  remote_op->set_id(recv_op_id_);
  uint64 context_view_id = ctx_->GetContextViewId();

  core::RefCountPtr<eager::EagerClient> eager_client;
  absl::Status status = ctx_->GetClient(recv_device_, &eager_client);
  if (!status.ok()) {
    captured_state_->dst()->PoisonRemote(status, recv_device_, context_view_id);
    done(status);
    return;
  }

  // Don't issue the recv until send has completed.
  //  - local send will complete very quickly.
  //  - remote send will take some time, but remote->remote copy is
  //    probably rare enough that we don't care much.
  // Blocks until send has completed.
  absl::Status send_status = captured_state_->GetSendStatus();
  if (!send_status.ok()) {
    captured_state_->dst()->PoisonRemote(status, recv_device_, context_view_id);
    done(send_status);
    return;
  }

  EnqueueResponse* response = new EnqueueResponse;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  Device* recv_device = recv_device_;
  eager_client->StreamingEnqueueAsync(
      ctx_->Executor().StreamingEnqueue(),
      /*call_opts=*/nullptr, &request, response,
      [captured_state, response, recv_device, context_view_id,
       done](const absl::Status& s) {
        if (s.ok()) {
          absl::Status status = captured_state->dst()->SetRemoteShape(
              response->queue_response(0).shape(0), recv_device,
              context_view_id);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by remote Recv op: "
                       << status.ToString()
                       << "\nThis should never happen. "
                          "Please file an issue with the TensorFlow Team.";
          }
        } else {
          captured_state->dst()->PoisonRemote(s, recv_device, context_view_id);
        }
        done(s);
        delete response;
      });
}

void RemoteCopyNode::StartRecv(StatusCallback done) {
  // TODO(gjn): We should consider just using the low-level RecvOp::Compute()
  // functionality here instead of constructing an Op.
  EagerOperation op(ctx_);
  absl::Status status = op.Reset("_Recv", /*device_name=*/nullptr,
                                 /*remote=*/false, /*executor=*/nullptr);
  Device* recv_device = ctx_->CanonicalDevice(recv_device_);
  if (!status.ok()) {
    captured_state_->dst()->Poison(status, recv_device);
    done(status);
    return;
  }

  op.SetDevice(recv_device_);

  op.MutableAttrs()->Set("tensor_name", wire_id_);
  op.MutableAttrs()->Set("send_device", send_device_->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64_t>(send_device_->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device_->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("tensor_type", src_->dtype);

  if (recv_device_->IsLocal()) {
    std::vector<Tensor> outputs(1);
    status = RunLocalRecv(&op, &outputs);
    if (!status.ok()) {
      captured_state_->dst()->Poison(status, recv_device);
      done(status);
      return;
    }
    status =
        captured_state_->dst()->SetTensor(std::move(outputs[0]), recv_device);
    done(status);
  } else {
    // Handles captured_state_->dst_ internally.
    RunRemoteRecv(&op, std::move(done));
  }
}

absl::Status SerializePackedHandle(const uint64 op_id,
                                   TensorHandle* packed_handle,
                                   const Device* target_device,
                                   EagerContext* ctx, SendPackedHandleOp* op) {
  op->set_op_id(op_id);
  op->set_device_name(packed_handle->DeviceOrHostCPU(*ctx)->name());
  for (int i = 0; i < packed_handle->NumPackedHandles(); ++i) {
    TensorHandle* h = nullptr;
    TF_RETURN_IF_ERROR(packed_handle->ExtractPackedHandle(i, &h));
    if (h->Type() == TensorHandle::LOCAL) {
      // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence
      // copy it to the CPU before copying it out.
      Tensor tensor;
      TF_RETURN_IF_ERROR(h->CopyToDevice(*ctx, ctx->HostCPU(), &tensor));
      auto* local_handle = op->add_handles()->mutable_local_handle();
      local_handle->set_device(h->op_device() ? h->op_device()->name()
                                              : ctx->HostCPU()->name());
      tensor.AsProtoTensorContent(local_handle->mutable_tensor());
    } else if (h->Type() == TensorHandle::REMOTE) {
      // Only serialize the resource dtype and shape of the first handle, since
      // all handles are of the same resource dtype and shape.
      // If src_device is on the same task of target_device, the handle is a
      // local handle on the target device, which means the resource dtype and
      // shape are known on the target device.
      Device* src_device = h->device();
      const bool serialize_resource_dtype_and_shape =
          (i == 0) && (h->dtype == DT_RESOURCE) &&
          (!ctx->OnSameTask(src_device, target_device));
      // For a remote component function, a function execution request and an
      // input generation request may come from different workers. We need to
      // guarantee that the input generation request is processed before the
      // function execution request, so wait until the underlying remote handles
      // are ready before sending a packed handle to the function device.
      TF_RETURN_IF_ERROR(ctx->RemoteMgr()->SerializeRemoteTensorHandle(
          h, /*wait_until_ready=*/true,
          op->add_handles()->mutable_remote_handle(), src_device, "",
          serialize_resource_dtype_and_shape));
    } else {
      return errors::InvalidArgument("Nested packed handles are not supported");
    }
  }
  return absl::OkStatus();
}

void RemoteCopyNode::StartSendPackedHandle(StatusCallback done) {
  absl::Status s;
  const uint64 context_view_id = ctx_->GetContextViewId();
  if (!send_device_->IsLocal()) {
    s = errors::InvalidArgument(
        "Copy a packed handle from a remote device is not supported");
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }

  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  s = SerializePackedHandle(recv_op_id_, src_, recv_device_, ctx_,
                            request.add_queue()->mutable_send_packed_handle());
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }

  TensorShape shape;
  s = src_->Shape(&shape);
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }
  captured_state_->SetSrcShape(shape);

  core::RefCountPtr<eager::EagerClient> eager_client;
  s = ctx_->GetClient(recv_device_, &eager_client);
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }

  EnqueueResponse* response = new EnqueueResponse;
  Device* recv_device = recv_device_;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  eager_client->StreamingEnqueueAsync(
      ctx_->Executor().StreamingEnqueue(),
      /*call_opts=*/nullptr, &request, response,
      [captured_state, response, recv_device, context_view_id,
       done](const absl::Status& s) {
        if (s.ok()) {
          absl::Status status = captured_state->dst()->SetRemoteShape(
              captured_state->GetSrcShape(), recv_device, context_view_id);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by SendPackedHadnle rpc: "
                       << status.ToString();
          }
        } else {
          captured_state->dst()->PoisonRemote(s, recv_device, context_view_id);
        }
        done(s);
        delete response;
      });
}

void RemoteCopyNode::StartRemoteSendTensor(StatusCallback done) {
  absl::Status s;
  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  auto* send_tensor = request.add_queue()->mutable_send_tensor();
  send_tensor->set_op_id(recv_op_id_);
  send_tensor->set_device_name(recv_device_->name());
  uint64 context_view_id = ctx_->GetContextViewId();

  // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence
  // copy it to the CPU before copying it out.
  // TODO(fishx): Make CopyToDevice asynchronous.
  Tensor tensor;
  s = src_->CopyToDevice(*ctx_, ctx_->HostCPU(), &tensor);
  if (!s.ok()) {
    done(s);
    return;
  }
  tensor.AsProtoTensorContent(send_tensor->add_tensors());

  core::RefCountPtr<eager::EagerClient> eager_client;
  s = ctx_->GetClient(recv_device_, &eager_client);
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }
  EnqueueResponse* response = new EnqueueResponse;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  captured_state->SetSrcShape(tensor.shape());
  Device* recv_device = recv_device_;
  eager_client->StreamingEnqueueAsync(
      ctx_->Executor().StreamingEnqueue(),
      /*call_opts=*/nullptr, &request, response,
      [captured_state, response, recv_device, context_view_id,
       done](const absl::Status& s) {
        if (s.ok()) {
          absl::Status status = captured_state->dst()->SetRemoteShape(
              captured_state->GetSrcShape(), recv_device, context_view_id);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by SendTensor rpc: "
                       << status.ToString();
          }
        } else {
          captured_state->dst()->PoisonRemote(s, recv_device, context_view_id);
        }
        done(s);
        delete response;
      });
}

absl::Status RemoteCopyNode::Prepare() {
  TF_RETURN_IF_ERROR(captured_state_->dst()->CopyInferenceShape(src_));
  return absl::OkStatus();
}

void RemoteCopyNode::RunAsync(StatusCallback done) {
  started_ = true;
  if (src_->Type() == TensorHandle::PACKED) {
    return StartSendPackedHandle(std::move(done));
  }

  if ((ctx_->UseSendTensorRPC()) && send_device_->IsLocal() &&
      !recv_device_->IsLocal()) {
    return StartRemoteSendTensor(std::move(done));
  }
  StartSend();

  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  auto done_wrapper = [captured_state,
                       done = std::move(done)](const absl::Status& s) {
    if (!s.ok() && errors::IsCancelled(s)) {
      absl::Status send_status = captured_state->GetSendStatus();
      if (!send_status.ok()) {
        // In this case, Recv is cancelled because the Send op failed.
        // Return the status of the Send op instead.
        done(send_status);
      }
    } else {
      done(s);
    }
  };

  // StartRecv() takes care of doing the right thing to dst handle.
  // No need to poison it after this point.
  StartRecv(std::move(done_wrapper));
}

void RemoteCopyNode::Abort(absl::Status status) {
  if (!started_) {
    uint64 context_view_id = ctx_->GetContextViewId();
    captured_state_->dst()->PoisonRemote(status, recv_device_, context_view_id);
  }
}

}  // namespace eager
}  // namespace tensorflow
