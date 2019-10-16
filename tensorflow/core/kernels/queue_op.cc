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

#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

QueueOp::QueueOp(OpKernelConstruction* context) : ResourceOpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
  if (capacity_ < 0) {
    capacity_ = QueueBase::kUnbounded;
  }
  OP_REQUIRES_OK(context,
                 context->GetAttr("component_types", &component_types_));
}

void QueueOp::Compute(OpKernelContext* context) {
  ResourceOpKernel<QueueInterface>::Compute(context);
  mutex_lock l(mu_);
  if (resource_ && context->track_allocations()) {
    context->record_persistent_memory_allocation(resource_->MemoryUsed());
  }
}

Status QueueOp::VerifyResource(QueueInterface* queue) {
  return queue->MatchesNodeDef(def());
}


QueueOpKernel::QueueOpKernel(OpKernelConstruction* context)
    : AsyncOpKernel(context) {}

void QueueOpKernel::ComputeAsync(OpKernelContext* ctx, DoneCallback callback) {
  QueueInterface* queue;
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &queue), callback);
  } else {
    OP_REQUIRES_OK_ASYNC(ctx, GetResourceFromContext(ctx, "handle", &queue),
                         callback);
  }
  ComputeAsync(ctx, queue, [callback, queue]() {
    queue->Unref();
    callback();
  });
}

QueueAccessOpKernel::QueueAccessOpKernel(OpKernelConstruction* context)
    : QueueOpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("timeout_ms", &timeout_));
  // TODO(keveman): Enable timeout.
  OP_REQUIRES(context, timeout_ == -1,
              errors::InvalidArgument("Timeout not supported yet."));
}

// Defines an EnqueueOp, the execution of which enqueues a tuple of
// tensors in the given Queue.
//
// The op has 1 + k inputs, where k is the number of components in the
// tuples stored in the given Queue:
// - Input 0: queue handle.
// - Input 1: 0th element of the tuple.
// - ...
// - Input (1+k): kth element of the tuple.
EnqueueOp::EnqueueOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {}

void EnqueueOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                             DoneCallback callback) {
  DataTypeVector expected_inputs;
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    expected_inputs.push_back(DT_RESOURCE);
  } else {
    expected_inputs.push_back(DT_STRING_REF);
  }
  for (DataType dt : queue->component_dtypes()) {
    expected_inputs.push_back(dt);
  }
  OP_REQUIRES_OK_ASYNC(ctx, ctx->MatchSignature(expected_inputs, {}), callback);

  QueueInterface::Tuple tuple;
  OpInputList components;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("components", &components),
                       callback);
  for (const Tensor& Tcomponent : components) {
    tuple.push_back(Tcomponent);
  }

  OP_REQUIRES_OK_ASYNC(ctx, queue->ValidateTuple(tuple), callback);
  queue->TryEnqueue(tuple, ctx, callback);
}

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
EnqueueManyOp::EnqueueManyOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {}

void EnqueueManyOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                 DoneCallback callback) {
  DataTypeVector expected_inputs;
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    expected_inputs.push_back(DT_RESOURCE);
  } else {
    expected_inputs.push_back(DT_STRING_REF);
  }
  for (DataType dt : queue->component_dtypes()) {
    expected_inputs.push_back(dt);
  }
  OP_REQUIRES_OK_ASYNC(ctx, ctx->MatchSignature(expected_inputs, {}), callback);

  QueueInterface::Tuple tuple;
  OpInputList components;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("components", &components),
                       callback);
  for (const Tensor& Tcomponent : components) {
    tuple.push_back(Tcomponent);
  }

  OP_REQUIRES_OK_ASYNC(ctx, queue->ValidateManyTuple(tuple), callback);
  queue->TryEnqueueMany(tuple, ctx, callback);
}

EnqueueManyOp::~EnqueueManyOp() = default;

// Defines a DequeueOp, the execution of which dequeues a tuple of
// tensors from the given Queue.
//
// The op has one input, which is the handle of the appropriate
// Queue. The op has k outputs, where k is the number of components in
// the tuples stored in the given Queue, and output i is the ith
// component of the dequeued tuple.
DequeueOp::DequeueOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {}

void DequeueOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                             DoneCallback callback) {
  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->MatchSignature({DT_RESOURCE}, queue->component_dtypes()),
        callback);
  } else {
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->MatchSignature({DT_STRING_REF}, queue->component_dtypes()),
        callback);
  }

  queue->TryDequeue(ctx, [ctx, callback](const QueueInterface::Tuple& tuple) {
    if (!ctx->status().ok()) {
      callback();
      return;
    }
    OpOutputList output_components;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->output_list("components", &output_components), callback);
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      output_components.set(i, tuple[i]);
    }
    callback();
  });
}

DequeueOp::~DequeueOp() = default;

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
DequeueManyOp::DequeueManyOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {}

void DequeueManyOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                 DoneCallback callback) {
  const Tensor& Tnum_elements = ctx->input(1);
  int32 num_elements = Tnum_elements.flat<int32>()(0);

  OP_REQUIRES_ASYNC(ctx, num_elements >= 0,
                    errors::InvalidArgument("DequeueManyOp requested ",
                                            num_elements, " < 0 elements"),
                    callback);

  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->MatchSignature({DT_RESOURCE, DT_INT32}, queue->component_dtypes()),
        callback);
  } else {
    OP_REQUIRES_OK_ASYNC(ctx,
                         ctx->MatchSignature({DT_STRING_REF, DT_INT32},
                                             queue->component_dtypes()),
                         callback);
  }

  queue->TryDequeueMany(
      num_elements, ctx, false /* allow_small_batch */,
      [ctx, callback](const QueueInterface::Tuple& tuple) {
        if (!ctx->status().ok()) {
          callback();
          return;
        }
        OpOutputList output_components;
        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->output_list("components", &output_components), callback);
        for (int i = 0; i < ctx->num_outputs(); ++i) {
          output_components.set(i, tuple[i]);
        }
        callback();
      });
}

DequeueManyOp::~DequeueManyOp() = default;

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
DequeueUpToOp::DequeueUpToOp(OpKernelConstruction* context)
    : QueueAccessOpKernel(context) {}

void DequeueUpToOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                 DoneCallback callback) {
  const Tensor& Tnum_elements = ctx->input(1);
  int32 num_elements = Tnum_elements.flat<int32>()(0);

  OP_REQUIRES_ASYNC(ctx, num_elements >= 0,
                    errors::InvalidArgument("DequeueUpToOp requested ",
                                            num_elements, " < 0 elements"),
                    callback);

  if (ctx->input_dtype(0) == DT_RESOURCE) {
    OP_REQUIRES_OK_ASYNC(
        ctx,
        ctx->MatchSignature({DT_RESOURCE, DT_INT32}, queue->component_dtypes()),
        callback);
  } else {
    OP_REQUIRES_OK_ASYNC(ctx,
                         ctx->MatchSignature({DT_STRING_REF, DT_INT32},
                                             queue->component_dtypes()),
                         callback);
  }

  queue->TryDequeueMany(
      num_elements, ctx, true /* allow_small_batch */,
      [ctx, callback](const QueueInterface::Tuple& tuple) {
        if (!ctx->status().ok()) {
          callback();
          return;
        }
        OpOutputList output_components;
        OP_REQUIRES_OK_ASYNC(
            ctx, ctx->output_list("components", &output_components), callback);
        for (int i = 0; i < ctx->num_outputs(); ++i) {
          output_components.set(i, tuple[i]);
        }
        callback();
      });
}

DequeueUpToOp::~DequeueUpToOp() = default;

// Defines a QueueCloseOp, which closes the given Queue. Closing a
// Queue signals that no more elements will be enqueued in it.
//
// The op has one input, which is the handle of the appropriate Queue.
QueueCloseOp::QueueCloseOp(OpKernelConstruction* context)
    : QueueOpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("cancel_pending_enqueues",
                                           &cancel_pending_enqueues_));
}

void QueueCloseOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                DoneCallback callback) {
  queue->Close(ctx, cancel_pending_enqueues_, callback);
}

// Defines a QueueSizeOp, which computes the number of elements in the
// given Queue, and emits it as an output tensor.
//
// The op has one input, which is the handle of the appropriate Queue;
// and one output, which is a single-element tensor containing the current
// size of that Queue.
QueueSizeOp::QueueSizeOp(OpKernelConstruction* context)
    : QueueOpKernel(context) {}

void QueueSizeOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                               DoneCallback callback) {
  Tensor* Tqueue_size = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &Tqueue_size));
  Tqueue_size->flat<int32>().setConstant(queue->size());
  callback();
}

QueueIsClosedOp::QueueIsClosedOp(OpKernelConstruction* context)
    : QueueOpKernel(context) {}

void QueueIsClosedOp::ComputeAsync(OpKernelContext* ctx, QueueInterface* queue,
                                   DoneCallback callback) {
  Tensor* Tqueue_is_closed = nullptr;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({}), &Tqueue_is_closed));
  Tqueue_is_closed->flat<bool>().setConstant(queue->is_closed());
  callback();
}

}  // namespace tensorflow
