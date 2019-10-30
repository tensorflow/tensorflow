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
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"

#include "tensorflow/core/distributed_runtime/eager/destroy_tensor_handle_node.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace {

void DestroyRemoteTensorHandle(EagerContext* ctx, const string& remote_task,
                               uint64 context_id, uint64 op_id, int output_num,
                               bool ready) {
  if (ctx->GetContextId() != context_id) {
    // This means that this tensor was pointing to a remote device, which
    // has been changed out from under us. Simply return since there is
    // nothing we can do.
    return;
  }

  eager::EagerClient* eager_client;
  Status status = ctx->GetClient(remote_task, &eager_client);
  if (!status.ok()) {
    LOG_EVERY_N_SEC(INFO, 60)
        << "Unable to destroy remote tensor handle because the target "
        << remote_task << " is no longer available.";
    return;
  }

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(context_id);

  auto* handle_to_decref = request->add_queue()->mutable_handle_to_decref();
  handle_to_decref->set_op_id(op_id);
  handle_to_decref->set_output_num(output_num);

  VLOG(3) << "Sending request to delete " << request->DebugString();
  std::unique_ptr<EagerNode> node(
      absl::make_unique<eager::DestroyTensorHandleNode>(std::move(request), ctx,
                                                        remote_task, ready));
  auto& executor = ctx->Executor();
  if (executor.Async()) {
    Status status = executor.AddOrExecute(std::move(node));
    if (!status.ok()) {
      LOG_EVERY_N_SEC(WARNING, 60)
          << "Unable to destroy remote tensor handles. If you are "
             "running a tf.function, it usually indicates some op in "
             "the graph gets an error: "
          << status.error_message();
    }
  } else {
    // This thread may still hold tensorflow::StreamingRPCState::mu_. We need
    // to send out the destroy request in a new thread to avoid deadlock.
    auto* released_node = node.release();
    (*ctx->runner())([ctx, released_node] {
      Status status =
          ctx->Executor().AddOrExecute(absl::WrapUnique(released_node));
      if (!status.ok()) {
        LOG_EVERY_N_SEC(WARNING, 60)
            << "Unable to destroy remote tensor handles. If you are "
               "running a tf.function, it usually indicates some op in "
               "the graph gets an error: "
            << status.error_message();
      }
    });
  }
}
}  // namespace

RemoteTensorHandleData::RemoteTensorHandleData(int64 op_id, int output_num,
                                               const TensorShape& shape,
                                               const string& remote_task,
                                               uint64 context_id,
                                               EagerContext* ctx)
    : op_id_(op_id),
      output_num_(output_num),
      shape_(shape),
      remote_task_(remote_task),
      context_id_(context_id),
      ctx_(ctx) {
  DCHECK(op_id_ >= 0 && output_num_ >= 0)
      << "Op ID and output num should be >= 0. Op ID: " << op_id
      << ", Output num: " << output_num;
  ctx->Ref();
}

RemoteTensorHandleData::~RemoteTensorHandleData() {
  DestroyRemoteTensorHandle(ctx_, remote_task_, context_id_, op_id_,
                            output_num_, /*ready=*/true);
  ctx_->Unref();
}

Status RemoteTensorHandleData::Tensor(const tensorflow::Tensor** t) const {
  return errors::Unavailable(
      "Unable to get a tensor for a remote device. Please copy the tensor "
      "handle to a local device using TFE_TensorHandleCopyToDevice");
}

Status RemoteTensorHandleData::TensorValue(tensorflow::TensorValue* t) {
  return errors::Unavailable(
      "Unable to get a tensor for a remote device. Please copy the tensor "
      "handle to a local device using TFE_TensorHandleCopyToDevice");
}

Status RemoteTensorHandleData::Shape(TensorShape* shape) const {
  *shape = shape_;

  return Status::OK();
}

Status RemoteTensorHandleData::NumDims(int* num_dims) const {
  *num_dims = shape_.dims();

  return Status::OK();
}

Status RemoteTensorHandleData::Dim(int dim_index, int64* dim) const {
  *dim = shape_.dim_size(dim_index);

  return Status::OK();
}

Status RemoteTensorHandleData::NumElements(int64* num_elements) const {
  *num_elements = shape_.num_elements();

  return Status::OK();
}

string RemoteTensorHandleData::DebugString() const {
  return strings::StrCat("RemoteTensorHandleData:", " op_id: ", op_id_,
                         " output_num: ", output_num_);
}

UnshapedRemoteTensorHandleData::UnshapedRemoteTensorHandleData(
    int64 op_id, int32 output_num, const string& remote_task, uint64 context_id,
    EagerContext* ctx)
    : op_id_(op_id),
      output_num_(output_num),
      delete_remote_tensor_(true),
      remote_task_(remote_task),
      context_id_(context_id),
      ctx_(ctx) {
  DCHECK(op_id_ >= 0 && output_num_ >= 0)
      << "Op ID and output num should be >= 0. Op ID: " << op_id
      << ", Output num: " << output_num;
  ctx->Ref();
}

UnshapedRemoteTensorHandleData::~UnshapedRemoteTensorHandleData() {
  if (delete_remote_tensor_) {
    DestroyRemoteTensorHandle(ctx_, remote_task_, context_id_, op_id_,
                              output_num_, /*ready=*/false);
  }
  ctx_->Unref();
}

Status UnshapedRemoteTensorHandleData::Tensor(
    const tensorflow::Tensor** t) const {
  return errors::Unavailable(
      "Unable to get a tensor for a remote handle. Please copy the tensor "
      "handle to a local device using TFE_TensorHandleCopyToDevice");
}

Status UnshapedRemoteTensorHandleData::TensorValue(tensorflow::TensorValue* t) {
  return errors::Unavailable(
      "Unable to get a tensor for a remote handle. Please copy the tensor "
      "handle to a local device using TFE_TensorHandleCopyToDevice");
}

Status UnshapedRemoteTensorHandleData::Shape(TensorShape* shape) const {
  return errors::Unavailable(
      "Unable to get shape information for an async remote handle. Please wait "
      "until it is ready");
}

Status UnshapedRemoteTensorHandleData::NumDims(int* num_dims) const {
  return errors::Unavailable(
      "Unable to get shape information for an async remote handle. Please wait "
      "until it is ready");
}

Status UnshapedRemoteTensorHandleData::Dim(int dim_index, int64* dim) const {
  return errors::Unavailable(
      "Unable to get shape information for an async remote handle. Please wait "
      "until it is ready");
}

Status UnshapedRemoteTensorHandleData::NumElements(int64* num_elements) const {
  return errors::Unavailable(
      "Unable to get shape information for an async remote handle. Please wait "
      "until it is ready");
}

string UnshapedRemoteTensorHandleData::DebugString() const {
  return strings::StrCat("UnshapedRemoteTensorHandleDat:", " op_id: ", op_id_,
                         " output_num: ", output_num_);
}

}  // namespace tensorflow
