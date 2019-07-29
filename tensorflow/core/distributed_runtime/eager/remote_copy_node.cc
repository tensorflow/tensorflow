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

#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/errors.h"

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
  static tensorflow::int64 wireid GUARDED_BY(wireid_mutex) = 0;
  tensorflow::mutex_lock l(wireid_mutex);
  return strings::StrCat(random_seed, "_", wireid++);
}

}  // namespace

RemoteCopyNode::RemoteCopyNode(EagerContext* ctx, TensorHandle* src,
                               TensorHandle* dst, Device* recv_device,
                               uint64 recv_op_id)
    : EagerNode(),
      src_(src),
      dst_(dst),
      ctx_(ctx),
      send_device_(src->DeviceOrHostCPU(ctx)),
      recv_device_(recv_device),
      wire_id_(GetUniqueWireID()),
      recv_op_id_(recv_op_id) {
  DCHECK(!send_device_->IsLocal() || !recv_device_->IsLocal());
  src_->Ref();
  dst_->Ref();
  ctx_->Ref();
}

Status RemoteCopyNode::RunSend() {
  // TODO(gjn): We should consider just using the low-level SendOp::Compute()
  // functionality here instead of constructing an Op.
  const AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp("_Send", &types, &is_function));
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
    TF_RETURN_IF_ERROR(ctx_->GetStatus());

    op.AddInput(src_);

    core::RefCountPtr<KernelAndDevice> kernel;
    TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(&op, &kernel));

    gtl::InlinedVector<TensorValue, 4> input_vector(1);
    TF_RETURN_IF_ERROR(src_->TensorValue(&input_vector[0]));

    TF_RETURN_IF_ERROR(
        kernel->Run(input_vector, nullptr, nullptr, nullptr, nullptr, nullptr));
  } else {
    eager::EagerClient* eager_client;
    uint64 context_id = ctx_->GetContextId();
    TF_RETURN_IF_ERROR(ctx_->GetClient(send_device_, &eager_client));

    std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
    request->set_context_id(context_id);

    auto* remote_op = request->add_queue()->mutable_operation();
    TF_RETURN_IF_ERROR(ctx_->RemoteMgr()->SerializeRemoteTensorHandle(
        src_, remote_op->add_inputs(), src_->device()));

    PrepareRemoteOp(remote_op, &op);
    remote_op->set_id(ctx_->RemoteMgr()->NextOpId());

    auto* response = new EnqueueResponse;
    eager_client->EnqueueAsync(request.get(), response,
                               [this, response](const Status& s) {
                                 send_status_.Update(s);
                                 if (!s.ok()) {
                                   recv_cancellation_.StartCancel();
                                 }
                                 delete response;
                               });
  }
  return Status::OK();
}

Status RemoteCopyNode::RunRecv() {
  // TODO(gjn): We should consider just using the low-level RecvOp::Compute()
  // functionality here instead of constructing an Op.
  const AttrTypeMap* types;
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp("_Recv", &types, &is_function));
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
    TF_RETURN_IF_ERROR(ctx_->GetStatus());

    core::RefCountPtr<KernelAndDevice> kernel;
    TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(&op, &kernel));

    std::vector<Tensor> outputs;
    gtl::InlinedVector<TensorValue, 4> input_vector;
    TF_RETURN_IF_ERROR(kernel->Run(input_vector, &outputs, nullptr, nullptr,
                                   nullptr, &recv_cancellation_));
    return dst_->SetTensor(outputs[0]);
  } else {
    eager::EagerClient* eager_client;
    uint64 context_id = ctx_->GetContextId();
    TF_RETURN_IF_ERROR(ctx_->GetClient(recv_device_, &eager_client));

    std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);

    request->set_context_id(context_id);

    auto* remote_op = request->add_queue()->mutable_operation();
    PrepareRemoteOp(remote_op, &op);
    remote_op->set_id(recv_op_id_);

    EnqueueResponse response;
    Status status;
    Notification n;

    CancellationToken token = recv_cancellation_.get_cancellation_token();
    bool already_cancelled =
        !recv_cancellation_.RegisterCallback(token, [&n, &status] {
          status.Update(errors::Cancelled(
              "Recv op is cancelled due to an error in Send op."));
          n.Notify();
        });

    if (already_cancelled) {
      status =
          errors::Cancelled("Recv op is cancelled due to an error in Send op.");
    } else {
      // Note(fishx): When the recv op is cancelled, we doesn't clean up the
      // state on remote server. So the recv op may ran successfully on the
      // remote server even though we cancel it on client.
      eager_client->EnqueueAsync(request.get(), &response,
                                 [this, &n, &status](const Status& s) {
                                   if (recv_cancellation_.IsCancelled()) return;
                                   status.Update(s);
                                   n.Notify();
                                 });
      n.WaitForNotification();
      recv_cancellation_.DeregisterCallback(token);
    }

    TF_RETURN_IF_ERROR(status);

    return dst_->SetRemoteShape(response.queue_response(0).shape(0),
                                recv_device_);
  }
}

Status RemoteCopyNode::Run() {
  Status s = RunSend();
  if (!s.ok()) {
    Abort(s);
    return s;
  }

  s = RunRecv();
  if (!s.ok() && errors::IsCancelled(s) && !send_status_.ok()) {
    // In this case, Recv is cancel because Send op failed. Return the status of
    // send op instead.
    Abort(send_status_);
    return send_status_;
  }
  if (!s.ok()) {
    Abort(s);
  }

  src_->Unref();
  dst_->Unref();
  ctx_->Unref();
  return s;
}

void RemoteCopyNode::Abort(Status status) {
  dst_->Poison(status);
  src_->Unref();
  dst_->Unref();
  ctx_->Unref();
}

}  // namespace eager
}  // namespace tensorflow
