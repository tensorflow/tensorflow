/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/priority_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Defines a PriorityQueueOp, which produces a Queue (specifically, one
// backed by PriorityQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class PriorityQueueOp : public TypedQueueOp {
 public:
  explicit PriorityQueueOp(OpKernelConstruction* context)
      : TypedQueueOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
    component_types_.insert(component_types_.begin(), DT_INT64);
    if (!component_shapes_.empty()) {
      component_shapes_.insert(component_shapes_.begin(), TensorShape({}));
    }
  }

 private:
  Status CreateResource(QueueInterface** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    PriorityQueue* queue = new PriorityQueue(capacity_, component_types_,
                                             component_shapes_, cinfo_.name());
    return CreateTypedQueue(queue, ret);
  }

  std::vector<TensorShape> component_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(PriorityQueueOp);
};

REGISTER_KERNEL_BUILDER(Name("PriorityQueue").Device(DEVICE_CPU),
                        PriorityQueueOp);

}  // namespace tensorflow
