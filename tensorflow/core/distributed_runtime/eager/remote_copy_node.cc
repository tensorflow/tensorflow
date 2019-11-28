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

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace eager {

namespace {

void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(op->Device()->name());
}

Status CreateUncachedKernelAndDeviceOp(
    EagerOperation* op, core::RefCountPtr<KernelAndDevice>* kernel) {
  EagerContext* ctx = op->EagerContext();
  Device* device = op->Device();

  FunctionLibraryRuntime* flr = ctx->func_lib(device);
  if (flr == nullptr) {
    return errors::Unavailable(
        "Unable to find a FunctionLibraryRuntime corresponding to device ",
        device->name());
  }

  auto runner = (flr->runner() != nullptr) ? flr->runner() : ctx->runner();
  kernel->reset(new KernelAndDeviceOp(
      ctx->GetRendezvous(), ctx->LogMemory(), flr, runner,
      ctx->GetCollectiveExecutorHandle(), ctx->HostCPU()));

  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  return kernel->get()->Init(ndef, nullptr);
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
      send_device_(src->DeviceOrHostCPU(ctx)),
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

Status RemoteCopyNode::RunLocalSend(EagerOperation* op) {
  TF_RETURN_IF_ERROR(executor_->status());

  op->AddInput(src_);

  core::RefCountPtr<KernelAndDevice> kernel;
  TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(op, &kernel));

  gtl::InlinedVector<TensorValue, 4> input_vector(1);
  TF_RETURN_IF_ERROR(src_->TensorValue(&input_vector[0]));

  EagerKernelArgs args(std::move(input_vector));
  return kernel->Run(args, nullptr, nullptr, absl::nullopt);
}

void RemoteCopyNode::StartSend() {
  // TODO(gjn): We should consider just using the low-level SendOp::Compute()
  // functionality here instead of constructing an Op.
  const AttrTypeMap* types;
  bool is_function = false;
  Status status = AttrTypeMapForOp("_Send", &types, &is_function);
  if (!status.ok()) {
    captured_state_->SetSendStatus(status);
    return;
  }
  DCHECK(!is_function);
  EagerOperation op(ctx_, "_Send", /*is_function=*/false, types);

  op.SetDevice(send_device_);

  op.MutableAttrs()->Set("tensor_name", wire_id_);
  op.MutableAttrs()->Set("send_device", send_device_->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64>(send_device_->attributes().incarnation()));
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
        src_, remote_op->add_inputs(), src_->device(),
        src_->DeviceOrHostCPU(ctx_)->name());
    if (!status.ok()) {
      captured_state_->SetSendStatus(status);
      return;
    }

    PrepareRemoteOp(remote_op, &op);
    remote_op->set_id(ctx_->RemoteMgr()->NextOpId());

    // Issue the RPC
    eager::EagerClient* eager_client;
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
        &request, response, [response, captured_state](const Status& s) {
          captured_state->SetSendStatus(s);
          if (!s.ok()) {
            captured_state->recv_cancellation()->StartCancel();
          }
          delete response;
        });
  }
}

Status RemoteCopyNode::RunLocalRecv(EagerOperation* op,
                                    std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(executor_->status());

  core::RefCountPtr<KernelAndDevice> kernel;
  TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(op, &kernel));

  EagerKernelArgs args;
  return kernel->Run(args, outputs, captured_state_->recv_cancellation(),
                     absl::nullopt);
}

void RemoteCopyNode::RunRemoteRecv(EagerOperation* op, StatusCallback done) {
  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  auto* remote_op = request.add_queue()->mutable_operation();
  PrepareRemoteOp(remote_op, op);
  remote_op->set_id(recv_op_id_);

  eager::EagerClient* eager_client;
  Status status = ctx_->GetClient(recv_device_, &eager_client);
  if (!status.ok()) {
    captured_state_->dst()->Poison(status);
    done(status);
    return;
  }

  // Don't issue the recv until send has completed.
  //  - local send will complete very quickly.
  //  - remote send will take some time, but remote->remote copy is
  //    probably rare enough that we don't care much.
  // Blocks until send has completed.
  Status send_status = captured_state_->GetSendStatus();
  if (!send_status.ok()) {
    captured_state_->dst()->Poison(send_status);
    done(send_status);
    return;
  }

  EnqueueResponse* response = new EnqueueResponse;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  Device* recv_device = recv_device_;
  eager_client->StreamingEnqueueAsync(
      &request, response,
      [captured_state, response, recv_device, done](const Status& s) {
        if (s.ok()) {
          Status status = captured_state->dst()->SetRemoteShape(
              response->queue_response(0).shape(0), recv_device);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by remote Recv op: "
                       << status.ToString()
                       << "\nThis should never happen. "
                          "Please file an issue with the TensorFlow Team.";
          }
        } else {
          captured_state->dst()->Poison(s);
        }
        done(s);
        delete response;
      });
}

void RemoteCopyNode::StartRecv(StatusCallback done) {
  // TODO(gjn): We should consider just using the low-level RecvOp::Compute()
  // functionality here instead of constructing an Op.
  const AttrTypeMap* types;
  bool is_function = false;
  Status status = AttrTypeMapForOp("_Recv", &types, &is_function);
  if (!status.ok()) {
    captured_state_->dst()->Poison(status);
    done(status);
    return;
  }
  DCHECK(!is_function);
  EagerOperation op(ctx_, "_Recv", /*is_function=*/false, types);

  op.SetDevice(recv_device_);

  op.MutableAttrs()->Set("tensor_name", wire_id_);
  op.MutableAttrs()->Set("send_device", send_device_->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64>(send_device_->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device_->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("tensor_type", src_->dtype);

  if (recv_device_->IsLocal()) {
    std::vector<Tensor> outputs(1);
    status = RunLocalRecv(&op, &outputs);
    if (!status.ok()) {
      captured_state_->dst()->Poison(status);
      done(status);
      return;
    }
    status = captured_state_->dst()->SetTensor(std::move(outputs[0]));
    done(status);
  } else {
    // Handles captured_state_->dst_ internally.
    RunRemoteRecv(&op, std::move(done));
  }
}

void RemoteCopyNode::StartRemoteSendTensor(StatusCallback done) {
  Status s;
  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  auto* send_tensor = request.add_queue()->mutable_send_tensor();
  send_tensor->set_op_id(recv_op_id_);
  send_tensor->set_device_name(recv_device_->name());

  // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence
  // copy it to the CPU before copying it out.
  // TODO(b/110044833): this is currently slow, but can be fixed by making
  // tensor handles aware of more than one device.
  // TODO(fishx): Make CopyToDevice asynchronous.
  Tensor tensor;
  s = src_->CopyToDevice(ctx_, ctx_->HostCPU(), &tensor);
  if (!s.ok()) {
    done(s);
    return;
  }
  tensor.AsProtoTensorContent(send_tensor->add_tensors());

  eager::EagerClient* eager_client;
  s = ctx_->GetClient(recv_device_, &eager_client);
  if (!s.ok()) {
    captured_state_->dst()->Poison(s);
    done(s);
    return;
  }
  EnqueueResponse* response = new EnqueueResponse;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  captured_state->SetSrcShape(tensor.shape());
  Device* recv_device = recv_device_;
  eager_client->StreamingEnqueueAsync(
      &request, response,
      [captured_state, response, recv_device, done](const Status& s) {
        if (s.ok()) {
          Status status = captured_state->dst()->SetRemoteShape(
              captured_state->GetSrcShape(), recv_device);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by SendTensor rpc: "
                       << status.ToString();
          }
        } else {
          captured_state->dst()->Poison(s);
        }
        done(s);
        delete response;
      });
}

Status RemoteCopyNode::Prepare() {
  TF_RETURN_IF_ERROR(captured_state_->dst()->CopyInferenceShape(src_));
  return Status::OK();
}

void RemoteCopyNode::RunAsync(StatusCallback done) {
  started_ = true;
  if (ctx_->UseSendTensorRPC() && send_device_->IsLocal() &&
      !recv_device_->IsLocal()) {
    return StartRemoteSendTensor(std::move(done));
  }
  StartSend();

  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  auto done_wrapper = [captured_state,
                       done = std::move(done)](const Status& s) {
    if (!s.ok() && errors::IsCancelled(s)) {
      Status send_status = captured_state->GetSendStatus();
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

void RemoteCopyNode::Abort(Status status) {
  if (!started_) {
    captured_state_->dst()->Poison(status);
  }
}

}  // namespace eager
}  // namespace tensorflow
