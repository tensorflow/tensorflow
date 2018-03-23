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
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

bool TensorHandle::IsReady() {
  if (node_id == 0) return true;
  mutex_lock l(ctx_mutex_);
  return ctx_ == nullptr;
}

Status TensorHandle::WaitReady() {
  if (node_id == 0) return Status::OK();
  EagerExecutor* executor = nullptr;
  {
    mutex_lock l(ctx_mutex_);
    if (ctx_ == nullptr) return Status::OK();
    executor = ctx_->Executor();
  }
  return executor->WaitFor(node_id);
}

Status TensorHandle::Tensor(const tensorflow::Tensor** t) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *t = &tensor_;
  return Status::OK();
}

Status TensorHandle::Device(tensorflow::Device** d) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *d = device_;
  return Status::OK();
}

Status TensorHandle::OpDevice(tensorflow::Device** d) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *d = op_device_;
  return Status::OK();
}

Status TensorHandle::TensorAndDevice(const tensorflow::Tensor** tensor,
                                     tensorflow::Device** device,
                                     tensorflow::Device** op_device) {
  TF_RETURN_IF_ERROR(WaitReady());
  DCHECK(IsReady());
  *tensor = &tensor_;
  *device = device_;
  *op_device = op_device_;
  return Status::OK();
}

void TensorHandle::SetTensorAndDevice(const tensorflow::Tensor& tensor,
                                      tensorflow::Device* device,
                                      tensorflow::Device* op_device) {
  mutex_lock l(ctx_mutex_);
  DCHECK(node_id > 0 && ctx_) << "SetTensorAndDevice should be only called  "
                              << "on non-ready handles.";
  ctx_ = nullptr;
  tensor_ = tensor;
  device_ = device;
  op_device_ = op_device;
}

}  // namespace tensorflow
