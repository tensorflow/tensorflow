/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Defines a QueueOp, an abstract class for Queue construction ops.
class QueueOp : public ResourceOpKernel<QueueInterface> {
 public:
  QueueOp(OpKernelConstruction* context) : ResourceOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
    if (capacity_ < 0) {
      capacity_ = QueueBase::kUnbounded;
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_types", &component_types_));
  }

  void Compute(OpKernelContext* context) override {
    ResourceOpKernel<QueueInterface>::Compute(context);
    if (resource_ && context->track_allocations()) {
      context->record_persistent_memory_allocation(resource_->MemoryUsed());
    }
  }

 protected:
  // Variables accessible by subclasses
  int32 capacity_;
  DataTypeVector component_types_;

 private:
  Status VerifyResource(QueueInterface* queue) override {
    return queue->MatchesNodeDef(def());
  }
};

class TypedQueueOp : public QueueOp {
 public:
  using QueueOp::QueueOp;

 protected:
  template <typename TypedQueue>
  Status CreateTypedQueue(TypedQueue* queue, QueueInterface** ret) {
    if (queue == nullptr) {
      return errors::ResourceExhausted("Failed to allocate queue.");
    }
    *ret = queue;
    return queue->Initialize();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_QUEUE_OP_H_
