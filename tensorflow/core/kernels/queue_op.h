/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_QUEUE_OP_H_
#define TENSORFLOW_KERNELS_QUEUE_OP_H_

#include <deque>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Defines a QueueOp, an abstract class for Queue construction ops.
class QueueOp : public OpKernel {
 public:
  QueueOp(OpKernelConstruction* context)
      : OpKernel(context), queue_handle_set_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &queue_handle_, nullptr));
    if (capacity_ < 0) {
      capacity_ = QueueBase::kUnbounded;
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_types", &component_types_));
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!queue_handle_set_) {
      OP_REQUIRES_OK(ctx, SetQueueHandle(ctx));
    }
    ctx->set_output_ref(0, &mu_, queue_handle_.AccessTensor(ctx));
  }

 protected:
  ~QueueOp() override {
    // If the queue object was not shared, delete it.
    if (queue_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(cinfo_.resource_manager()->Delete<QueueInterface>(
          cinfo_.container(), cinfo_.name()));
    }
  }

 protected:
  typedef std::function<Status(QueueInterface**)> CreatorCallback;

  // Subclasses must override this
  virtual CreatorCallback GetCreator() const = 0;

  // Variables accessible by subclasses
  int32 capacity_;
  DataTypeVector component_types_;
  ContainerInfo cinfo_;

 private:
  Status SetQueueHandle(OpKernelContext* ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
    CreatorCallback creator = GetCreator();
    QueueInterface* queue;
    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<QueueInterface>(
            cinfo_.container(), cinfo_.name(), &queue, creator));
    core::ScopedUnref unref_me(queue);
    // Verify that the shared queue is compatible with the requested arguments.
    TF_RETURN_IF_ERROR(queue->MatchesNodeDef(def()));
    auto h = queue_handle_.AccessTensor(ctx)->flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    queue_handle_set_ = true;
    return Status::OK();
  }

  mutex mu_;
  PersistentTensor queue_handle_ GUARDED_BY(mu_);
  bool queue_handle_set_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_QUEUE_OP_H_
