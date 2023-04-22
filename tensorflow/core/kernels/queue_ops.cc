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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("QueueEnqueue").Device(DEVICE_CPU), EnqueueOp);
REGISTER_KERNEL_BUILDER(Name("QueueEnqueueV2").Device(DEVICE_CPU), EnqueueOp);

REGISTER_KERNEL_BUILDER(Name("QueueEnqueueMany").Device(DEVICE_CPU),
                        EnqueueManyOp);
REGISTER_KERNEL_BUILDER(Name("QueueEnqueueManyV2").Device(DEVICE_CPU),
                        EnqueueManyOp);

REGISTER_KERNEL_BUILDER(Name("QueueDequeue").Device(DEVICE_CPU), DequeueOp);
REGISTER_KERNEL_BUILDER(Name("QueueDequeueV2").Device(DEVICE_CPU), DequeueOp);

REGISTER_KERNEL_BUILDER(Name("QueueDequeueMany").Device(DEVICE_CPU),
                        DequeueManyOp);
REGISTER_KERNEL_BUILDER(Name("QueueDequeueManyV2").Device(DEVICE_CPU),
                        DequeueManyOp);

REGISTER_KERNEL_BUILDER(Name("QueueDequeueUpTo").Device(DEVICE_CPU),
                        DequeueUpToOp);
REGISTER_KERNEL_BUILDER(Name("QueueDequeueUpToV2").Device(DEVICE_CPU),
                        DequeueUpToOp);

REGISTER_KERNEL_BUILDER(Name("QueueClose").Device(DEVICE_CPU), QueueCloseOp);
REGISTER_KERNEL_BUILDER(Name("QueueCloseV2").Device(DEVICE_CPU), QueueCloseOp);

REGISTER_KERNEL_BUILDER(Name("QueueSize").Device(DEVICE_CPU), QueueSizeOp);
REGISTER_KERNEL_BUILDER(Name("QueueSizeV2").Device(DEVICE_CPU), QueueSizeOp);

REGISTER_KERNEL_BUILDER(Name("QueueIsClosed").Device(DEVICE_CPU),
                        QueueIsClosedOp);
REGISTER_KERNEL_BUILDER(Name("QueueIsClosedV2").Device(DEVICE_CPU),
                        QueueIsClosedOp);

REGISTER_KERNEL_BUILDER(
    Name("QueueEnqueueV2").Device(DEVICE_DEFAULT).HostMemory("handle"),
    EnqueueOp);
REGISTER_KERNEL_BUILDER(
    Name("QueueDequeueV2").Device(DEVICE_DEFAULT).HostMemory("handle"),
    DequeueOp);
REGISTER_KERNEL_BUILDER(
    Name("QueueCloseV2").Device(DEVICE_DEFAULT).HostMemory("handle"),
    QueueCloseOp);
REGISTER_KERNEL_BUILDER(Name("QueueSizeV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("size")
                            .HostMemory("handle"),
                        QueueSizeOp);
REGISTER_KERNEL_BUILDER(
    Name("QueueIsClosedV2").Device(DEVICE_DEFAULT).HostMemory("handle"),
    QueueIsClosedOp);

class FakeQueueOp : public OpKernel {
 public:
  explicit FakeQueueOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_STRING, TensorShape({2}), &tensor_));
  }

  void Compute(OpKernelContext* context) override {
    const ResourceHandle& ref = context->input(0).flat<ResourceHandle>()(0);
    tensor_.flat<tstring>()(0) = ref.container();
    tensor_.flat<tstring>()(1) = ref.name();
    context->set_output_ref(0, &mu_, &tensor_);
  }

 private:
  mutex mu_;
  Tensor tensor_;
};

REGISTER_KERNEL_BUILDER(Name("FakeQueue").Device(DEVICE_CPU), FakeQueueOp);

}  // namespace tensorflow
