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

// See docs in ../ops/data_flow_ops.cc.

#include <algorithm>
#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

FIFOQueue::FIFOQueue(int capacity, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name)
    : TypedQueue(capacity, component_dtypes, component_shapes, name) {}

void FIFOQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(queues_[0].size(), size_t{0});
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(*queues_[i][0].AccessTensor(ctx));
    queues_[i].pop_front();
  }
}

void FIFOQueue::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                           DoneCallback callback) {
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kEnqueue, cm, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          1, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", name_, "' is closed."));
              return kComplete;
            }
            if (queues_[0].size() < static_cast<size_t>(capacity_)) {
              for (int i = 0; i < num_components(); ++i) {
                queues_[i].push_back(PersistentTensor(tuple[i]));
              }
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

/* static */
Status FIFOQueue::GetElementComponentFromBatch(const FIFOQueue::Tuple& tuple,
                                               int64 index, int component,
                                               OpKernelContext* ctx,
                                               PersistentTensor* out_tensor) {
  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  Tensor* element_access = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_persistent(
      tuple[component].dtype(), element_shape, out_tensor, &element_access));
  TF_RETURN_IF_ERROR(
      CopySliceToElement(tuple[component], element_access, index));
  return Status::OK();
}

void FIFOQueue::TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                               DoneCallback callback) {
  const int64 batch_size = tuple[0].dim_size(0);
  if (batch_size == 0) {
    callback();
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kEnqueue, cm, token); });
    if (!already_cancelled) {
      enqueue_attempts_.emplace_back(
          batch_size, callback, ctx, cm, token,
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("FIFOQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
              const int64 index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                PersistentTensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
                if (!attempt->context->status().ok()) return kComplete;
                queues_[i].push_back(element);
              }
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                return kComplete;
              }
            }
            return result;
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

void FIFOQueue::TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) {
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          1, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            const int64 queue_size = queues_[0].size();
            if (closed_ && queue_size == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ",
                  queue_size, ")"));
              return kComplete;
            }
            if (queue_size > 0) {
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              attempt->done_callback = [callback, tuple]() { callback(tuple); };
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

void FIFOQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                               bool allow_small_batch,
                               CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(errors::InvalidArgument(
        "FIFOQueue's DequeueMany and DequeueUpTo require the "
        "components to have specified shapes."));
    callback(Tuple());
    return;
  }
  if (num_elements == 0) {
    Tuple tuple;
    tuple.reserve(num_components());
    for (int i = 0; i < num_components(); ++i) {
      // TODO(josh11b,misard): Switch to allocate_output().  Problem is
      // this breaks the abstraction boundary since we don't *really*
      // know if and how the Tensors in the tuple we pass to callback
      // correspond to the outputs of *ctx.  For example, the
      // ReaderRead Op uses TryDequeue() to get a filename out of a
      // queue that is used internally by the reader and is not
      // associated with any output of the ReaderRead.
      // mrry@ adds:
      // Maybe we need to pass a std::function<Tensor*(...)> (or
      // better signature) that calls the appropriate allocator
      // function in addition to ctx?  (Or support a shim Allocator
      // that has an internal OpKernelContext*, and dispatches to the
      // appropriate method?)
      // misard@ adds:
      // I don't see that a std::function would help. The problem is
      // that at this point (allocation time) the system doesn't know
      // what is going to happen to the element read out of the
      // queue. As long as we keep the generality that TensorFlow Ops
      // do their own dynamic allocation in arbitrary C++ code, we
      // need to preserve robustness to allocating output Tensors with
      // the 'wrong' attributes, and fixing up with a copy. The only
      // improvement I can see here in the future would be to support
      // an optimized case where the queue 'knows' what attributes to
      // use, and plumbs them through here.
      Tensor element;
      Status status = ctx->allocate_temp(component_dtypes_[i],
                                         ManyOutShape(i, 0), &element);
      if (!status.ok()) {
        ctx->SetStatus(status);
        callback(Tuple());
        return;
      }
      tuple.emplace_back(element);
    }
    callback(tuple);
    return;
  }

  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { Cancel(kDequeue, cm, token); });
    if (!already_cancelled) {
      // TODO(josh11b): This makes two copies of callback, avoid this if possible.
      dequeue_attempts_.emplace_back(
          num_elements, [callback]() { callback(Tuple()); }, ctx, cm, token,
          [callback, allow_small_batch, this](Attempt* attempt)
              EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                int64 queue_size = queues_[0].size();

                if (closed_ && queue_size < attempt->elements_requested) {
                  // If we don't have enough for a full dequeue, we have
                  // to reset the attempt tuple.
                  if (!attempt->tuple.empty()) {
                    // Restore already-dequeued elements to the front of the
                    // queue.
                    for (int64 i = attempt->tuple[0].dim_size(0) -
                                   attempt->elements_requested - 1;
                         i >= 0; --i) {
                      for (int j = 0; j < num_components(); ++j) {
                        PersistentTensor element;
                        Status s = GetElementComponentFromBatch(
                            attempt->tuple, i, j, attempt->context, &element);
                        if (!s.ok()) {
                          attempt->context->SetStatus(
                              errors::DataLoss("Failed to restore element from "
                                               "partially-dequeued batch "
                                               "to FIFOQueue: ",
                                               s.error_message()));
                        }
                        queues_[j].push_front(element);
                      }
                    }
                  }
                  if (allow_small_batch && !queues_[0].empty()) {
                    // Request all remaining elements in the queue.
                    queue_size = queues_[0].size();
                    attempt->tuple.clear();
                    attempt->elements_requested = queue_size;
                  } else {
                    if (allow_small_batch) {
                      // There may be some other attempts containing
                      // values.  If so, we'll yield and wait for them
                      // to add elements to the queue.
                      if (!enqueue_attempts_.empty()) return kProgress;
                    }
                    if (attempt->context->status().ok()) {
                      attempt->context->SetStatus(errors::OutOfRange(
                          "FIFOQueue '", name_, "' is closed and has ",
                          "insufficient elements (requested ",
                          attempt->elements_requested, ", current size ",
                          queue_size, ")"));
                    }
                    return kComplete;
                  }
                }

                RunResult result = kNoProgress;
                for (; queue_size > 0; --queue_size) {
                  if (attempt->tuple.empty()) {
                    // Only allocate tuple when we have something to dequeue
                    // so we don't use excessive memory when there are many
                    // blocked dequeue attempts waiting.
                    attempt->tuple.reserve(num_components());
                    for (int i = 0; i < num_components(); ++i) {
                      const TensorShape shape =
                          ManyOutShape(i, attempt->elements_requested);
                      Tensor element;
                      attempt->context->SetStatus(
                          attempt->context->allocate_temp(component_dtypes_[i],
                                                          shape, &element));
                      if (!attempt->context->status().ok()) return kComplete;
                      attempt->tuple.emplace_back(element);
                    }
                  }
                  result = kProgress;
                  Tuple tuple;
                  DequeueLocked(attempt->context, &tuple);
                  const int64 index = attempt->tuple[0].dim_size(0) -
                                      attempt->elements_requested;
                  for (int i = 0; i < num_components(); ++i) {
                    attempt->context->SetStatus(CopyElementToSlice(
                        tuple[i], &attempt->tuple[i], index));
                    if (!attempt->context->status().ok()) return kComplete;
                  }
                  tuple.clear();
                  --attempt->elements_requested;
                  if (attempt->elements_requested == 0) {
                    tuple = attempt->tuple;
                    attempt->done_callback = [callback, tuple]() {
                      callback(tuple);
                    };
                    return kComplete;
                  }
                }
                return result;
              });
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
    callback(Tuple());
  }
}

Status FIFOQueue::MatchesNodeDef(const NodeDef& node_def) {
  if (!MatchesNodeDefOp(node_def, "FIFOQueue").ok() &&
      !MatchesNodeDefOp(node_def, "FIFOQueueV2").ok()) {
    return errors::InvalidArgument("Expected FIFOQueue, found ", node_def.op());
  }
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
  TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
  return Status::OK();
}

}  // namespace tensorflow
