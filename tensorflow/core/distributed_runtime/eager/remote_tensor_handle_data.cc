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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace {

void DestoryRemoteTensorHandle(EagerContext* ctx,
                               eager::EagerClient* eager_client,
                               uint64 context_id, uint64 op_id,
                               int output_num) {
  auto cleanup = gtl::MakeCleanup([ctx]() { ctx->Unref(); });

  if (ctx->GetContextId() != context_id) {
    // This means that this tensor was pointing to a remote device, which
    // has been changed out from under us. Simply return since there is
    // nothing we can do.
    return;
  }

  std::unique_ptr<eager::EnqueueRequest> request(new eager::EnqueueRequest);
  request->set_context_id(context_id);

  auto* handle_to_decref = request->add_queue()->mutable_handle_to_decref();
  handle_to_decref->set_op_id(op_id);
  handle_to_decref->set_output_num(output_num);

  std::unique_ptr<EagerNode> node(
      absl::make_unique<eager::DestroyTensorHandleNode>(std::move(request),
                                                        eager_client));
  Status s = ctx->Async() ? ctx->ExecutorAdd(std::move(node)) : node->Run();
  if (!s.ok()) {
    LOG(ERROR) << "Unable to destroy remote tensor handles: "
               << s.error_message();
  }
}

}  // namespace

RemoteTensorHandleData::RemoteTensorHandleData(int64 op_id, int output_num,
                                               const TensorShape& shape,
                                               eager::EagerClient* eager_client,
                                               uint64 context_id,
                                               EagerContext* ctx)
    : op_id_(op_id),
      output_num_(output_num),
      shape_(shape),
      eager_client_(eager_client),
      context_id_(context_id),
      ctx_(ctx) {
  DCHECK(op_id_ >= 0 && output_num_ >= 0)
      << "Op ID and output num should be >= 0. Op ID: " << op_id
      << ", Output num: " << output_num;
  ctx->Ref();
}

RemoteTensorHandleData::~RemoteTensorHandleData() {
  DestoryRemoteTensorHandle(ctx_, eager_client_, context_id_, op_id_,
                            output_num_);
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
    int64 op_id, int32 output_num, eager::EagerClient* eager_client,
    uint64 context_id, EagerContext* ctx)
    : op_id_(op_id),
      output_num_(output_num),
      delete_remote_tensor_(true),
      eager_client_(eager_client),
      context_id_(context_id),
      ctx_(ctx) {
  DCHECK(op_id_ >= 0 && output_num_ >= 0)
      << "Op ID and output num should be >= 0. Op ID: " << op_id
      << ", Output num: " << output_num;
  ctx->Ref();
}

UnshapedRemoteTensorHandleData::~UnshapedRemoteTensorHandleData() {
  if (delete_remote_tensor_) {
    DestoryRemoteTensorHandle(ctx_, eager_client_, context_id_, op_id_,
                              output_num_);
  }
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
