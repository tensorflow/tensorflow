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

#ifndef TENSORFLOW_CORE_KERNELS_QUEUE_OP_H_
#define TENSORFLOW_CORE_KERNELS_QUEUE_OP_H_

#include <deque>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
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
  QueueOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 protected:
  // Variables accessible by subclasses
  int32 capacity_;
  DataTypeVector component_types_;

 private:
  absl::Status VerifyResource(QueueInterface* queue) override;
};

class TypedQueueOp : public QueueOp {
 public:
  using QueueOp::QueueOp;

 protected:
  template <typename TypedQueue>
  absl::Status CreateTypedQueue(TypedQueue* queue, QueueInterface** ret) {
    if (queue == nullptr) {
      return errors::ResourceExhausted("Failed to allocate queue.");
    }
    *ret = queue;
    return queue->Initialize();
  }
};

// Queue manipulator kernels

class QueueOpKernel : public AsyncOpKernel {
 public:
  explicit QueueOpKernel(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final;

 protected:
  virtual void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                            DoneCallback callback) = 0;
};

class QueueAccessOpKernel : public QueueOpKernel {
 public:
  explicit QueueAccessOpKernel(OpKernelConstruction* context);

 protected:
  int64_t timeout_;
};

// Defines an EnqueueOp, the execution of which enqueues a tuple of
// tensors in the given Queue.
//
// The op has 1 + k inputs, where k is the number of components in the
// tuples stored in the given Queue:
// - Input 0: queue handle.
// - Input 1: 0th element of the tuple.
// - ...
// - Input (1+k): kth element of the tuple.
class EnqueueOp : public QueueAccessOpKernel {
 public:
  explicit EnqueueOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  EnqueueOp(const EnqueueOp&) = delete;
  void operator=(const EnqueueOp&) = delete;
};

// Defines an EnqueueManyOp, the execution of which slices each
// component of a tuple of tensors along the 0th dimension, and
// enqueues tuples of slices in the given Queue.
//
// The op has 1 + k inputs, where k is the number of components in the
// tuples stored in the given Queue:
// - Input 0: queue handle.
// - Input 1: 0th element of the tuple.
// - ...
// - Input (1+k): kth element of the tuple.
//
// N.B. All tuple components must have the same size in the 0th
// dimension.
class EnqueueManyOp : public QueueAccessOpKernel {
 public:
  explicit EnqueueManyOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~EnqueueManyOp() override;

 private:
  EnqueueManyOp(const EnqueueManyOp&) = delete;
  void operator=(const EnqueueManyOp&) = delete;
};

// Defines a DequeueOp, the execution of which dequeues a tuple of
// tensors from the given Queue.
//
// The op has one input, which is the handle of the appropriate
// Queue. The op has k outputs, where k is the number of components in
// the tuples stored in the given Queue, and output i is the ith
// component of the dequeued tuple.
class DequeueOp : public QueueAccessOpKernel {
 public:
  explicit DequeueOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~DequeueOp() override;

 private:
  DequeueOp(const DequeueOp&) = delete;
  void operator=(const DequeueOp&) = delete;
};

// Defines a DequeueManyOp, the execution of which concatenates the
// requested number of elements from the given Queue along the 0th
// dimension, and emits the result as a single tuple of tensors.
//
// The op has two inputs:
// - Input 0: the handle to a queue.
// - Input 1: the number of elements to dequeue.
//
// The op has k outputs, where k is the number of components in the
// tuples stored in the given Queue, and output i is the ith component
// of the dequeued tuple.
class DequeueManyOp : public QueueAccessOpKernel {
 public:
  explicit DequeueManyOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~DequeueManyOp() override;

 private:
  DequeueManyOp(const DequeueManyOp&) = delete;
  void operator=(const DequeueManyOp&) = delete;
};

// Defines a DequeueUpToOp, the execution of which concatenates the
// requested number of elements from the given Queue along the 0th
// dimension, and emits the result as a single tuple of tensors.
//
// The difference between this op and DequeueMany is the handling when
// the Queue is closed.  While the DequeueMany op will return if there
// an error when there are less than num_elements elements left in the
// closed queue, this op will return between 1 and
// min(num_elements, elements_remaining_in_queue), and will not block.
// If there are no elements left, then the standard DequeueMany error
// is returned.
//
// This op only works if the underlying Queue implementation accepts
// the allow_small_batch = true parameter to TryDequeueMany.
// If it does not, an errors::Unimplemented exception is returned.
//
// The op has two inputs:
// - Input 0: the handle to a queue.
// - Input 1: the number of elements to dequeue.
//
// The op has k outputs, where k is the number of components in the
// tuples stored in the given Queue, and output i is the ith component
// of the dequeued tuple.
//
// The op has one attribute: allow_small_batch.  If the Queue supports
// it, setting this to true causes the queue to return smaller
// (possibly zero length) batches when it is closed, up to however
// many elements are available when the op executes.  In this case,
// the Queue does not block when closed.
class DequeueUpToOp : public QueueAccessOpKernel {
 public:
  explicit DequeueUpToOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

  ~DequeueUpToOp() override;

 private:
  DequeueUpToOp(const DequeueUpToOp&) = delete;
  void operator=(const DequeueUpToOp&) = delete;
};

// Defines a QueueCloseOp, which closes the given Queue. Closing a
// Queue signals that no more elements will be enqueued in it.
//
// The op has one input, which is the handle of the appropriate Queue.
class QueueCloseOp : public QueueOpKernel {
 public:
  explicit QueueCloseOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  bool cancel_pending_enqueues_;
  QueueCloseOp(const QueueCloseOp&) = delete;
  void operator=(const QueueCloseOp&) = delete;
};

// Defines a QueueSizeOp, which computes the number of elements in the
// given Queue, and emits it as an output tensor.
//
// The op has one input, which is the handle of the appropriate Queue;
// and one output, which is a single-element tensor containing the current
// size of that Queue.
class QueueSizeOp : public QueueOpKernel {
 public:
  explicit QueueSizeOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  QueueSizeOp(const QueueSizeOp&) = delete;
  void operator=(const QueueSizeOp&) = delete;
};

class QueueIsClosedOp : public QueueOpKernel {
 public:
  explicit QueueIsClosedOp(OpKernelConstruction* context);

 protected:
  void ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                    DoneCallback callback) override;

 private:
  QueueIsClosedOp(const QueueIsClosedOp&) = delete;
  void operator=(const QueueIsClosedOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUEUE_OP_H_
