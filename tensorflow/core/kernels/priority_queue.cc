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
#include <queue>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/batch_util.h"
#include "tensorflow/core/kernels/priority_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/priority_queue_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

PriorityQueue::PriorityQueue(int32 capacity,
                             const DataTypeVector& component_dtypes,
                             const std::vector<TensorShape>& component_shapes,
                             const string& name)
    : TypedQueue(capacity, component_dtypes, component_shapes, name) {}

Status PriorityQueue::Initialize() {
  Status s = TypedQueue::Initialize();
  if (!s.ok()) return s;

  mutex_lock lock(mu_);
  if (component_dtypes_[0] != DT_INT64) {
    return errors::InvalidArgument(
        "PriorityQueue priority index component must be type int64, but "
        "dtype is: ",
        DataTypeString(component_dtypes_[0]));
  }
  if (specified_shapes() && !TensorShapeUtils::IsScalar(component_shapes_[0])) {
    return errors::InvalidArgument(
        "PriorityQueue priority index component must be a scalar, but shape "
        "is: ",
        component_shapes_[0].DebugString());
  }
  return Status::OK();
}

void PriorityQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(queues_[0].size(), 0);
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    PersistentTensor persistent_tensor = gtl::ConsumeTop(&queues_[i]).second;
    (*tuple).push_back(*persistent_tensor.AccessTensor(ctx));
  }
}

void PriorityQueue::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
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
                  errors::Cancelled("PriorityQueue '", name_, "' is closed."));
              return kComplete;
            }
            if (queues_[0].size() < static_cast<size_t>(capacity_)) {
              if (!TensorShapeUtils::IsScalar(tuple[0].shape())) {
                attempt->context->SetStatus(errors::InvalidArgument(
                    "Expected the priority element to be a scalar, but "
                    "received shape: ",
                    tuple[0].shape().DebugString()));
                return kComplete;
              }
              const int64 priority = tuple[0].scalar<int64>()();
              for (int i = 0; i < num_components(); ++i) {
                queues_[i].emplace(priority, PersistentTensor(tuple[i]));
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
Status PriorityQueue::GetElementComponentFromBatch(
    const PriorityQueue::Tuple& tuple, int index, int component,
    OpKernelContext* ctx, PersistentTensor* out_tensor) {
  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  Tensor* element_access = nullptr;
  TF_RETURN_IF_ERROR(ctx->allocate_persistent(
      tuple[component].dtype(), element_shape, out_tensor, &element_access));
  TF_RETURN_IF_ERROR(
      batch_util::CopySliceToElement(tuple[component], element_access, index));
  return Status::OK();
}

void PriorityQueue::TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
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
          [tuple, this, ctx](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("PriorityQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
              const int index =
                  tuple[0].dim_size(0) - attempt->elements_requested;

              PersistentTensor priority_element;
              attempt->context->SetStatus(GetElementComponentFromBatch(
                  tuple, index, 0, attempt->context, &priority_element));
              if (!attempt->context->status().ok()) return kComplete;
              Tensor* priority_tensor = priority_element.AccessTensor(ctx);
              if (!TensorShapeUtils::IsScalar(priority_tensor->shape())) {
                attempt->context->SetStatus(errors::InvalidArgument(
                    "Expected the priority element to be a scalar, but "
                    "received shape: ",
                    priority_tensor->shape().DebugString()));
                return kComplete;
              }
              const int64 priority = priority_tensor->scalar<int64>()();
              for (int i = 0; i < num_components(); ++i) {
                PersistentTensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
                if (!attempt->context->status().ok()) return kComplete;
                queues_[i].emplace(priority, element);
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

void PriorityQueue::TryDequeue(OpKernelContext* ctx,
                               CallbackWithTuple callback) {
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
            const int32 s = queues_[0].size();
            if (closed_ && s == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "PriorityQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ", s,
                  ")"));
              return kComplete;
            }
            if (s > 0) {
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

void PriorityQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                                   bool allow_small_batch,
                                   CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(
        errors::InvalidArgument("PriorityQueue's DequeueMany requires the "
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
          [callback, this,
           allow_small_batch](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32 s = queues_[0].size();
            // Return OutOfRange if closed and there are fewer elements
            // available than requested.  *Unless* allow_small_batch
            // is true, in which case we return as many elements as
            // possible.
            if (closed_) {
              if (s == 0 ||
                  (!allow_small_batch && s < attempt->elements_requested)) {
                attempt->context->SetStatus(errors::OutOfRange(
                    "PriorityQueue '", name_, "' is closed and has ",
                    "insufficient elements (requested ",
                    attempt->elements_requested, ", current size ", s, ")"));
                return kComplete;
              }
            }

            // The PriorityQueue is expected to always return a
            // sorted set of entries.  In order to do this, the underlying
            // queue must have at least this many entries already.
            // Doing the dynamic thing and pulling out a portion at a
            // time leads to unordered output in calls to DequeueMany.
            //
            // An alternative solution is to store the attempt tuple
            // entries in an identical priority_queue and push onto
            // this queue dynamically, then when it is full, do all
            // the Tensor concatenation at the very end.
            // TODO(ebrevdo): Change approach if this leads to locking issues.
            if (s < attempt->elements_requested) {
              // If we have no elements at all, then wait.
              // Otherwise proceed if closed and allow small batch is true.
              // Otherwise wait until we have more enqueued elements.
              if (s == 0 || !(closed_ && allow_small_batch)) {
                return kNoProgress;
              }
            }

            RunResult result = kNoProgress;
            for (; s > 0; --s) {
              if (attempt->tuple.empty()) {
                // Only allocate tuple when we have something to dequeue
                // so we don't use excessive memory when there are many
                // blocked dequeue attempts waiting.
                attempt->tuple.reserve(num_components());
                for (int i = 0; i < num_components(); ++i) {
                  const TensorShape shape =
                      ManyOutShape(i, attempt->elements_requested);
                  Tensor element;
                  attempt->context->SetStatus(attempt->context->allocate_temp(
                      component_dtypes_[i], shape, &element));
                  if (!attempt->context->status().ok()) return kComplete;
                  attempt->tuple.emplace_back(element);
                }
              }
              result = kProgress;
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              const int index =
                  attempt->tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                attempt->context->SetStatus(batch_util::CopyElementToSlice(
                    std::move(tuple[i]), &attempt->tuple[i], index));
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

Status PriorityQueue::MatchesNodeDef(const NodeDef& node_def) {
  if (!MatchesNodeDefOp(node_def, "PriorityQueue").ok() &&
      !MatchesNodeDefOp(node_def, "PriorityQueueV2").ok()) {
    return errors::InvalidArgument("Expected PriorityQueue, found ",
                                   node_def.op());
  }
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
  TF_RETURN_IF_ERROR(MatchesPriorityNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesPriorityNodeDefShapes(node_def));
  return Status::OK();
}

Status PriorityQueue::MatchesPriorityNodeDefTypes(
    const NodeDef& node_def) const {
  DataTypeVector requested_dtypes;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "component_types", &requested_dtypes));
  requested_dtypes.insert(requested_dtypes.begin(), DT_INT64);
  if (requested_dtypes != component_dtypes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component types ",
                                   DataTypeSliceString(component_dtypes_),
                                   " but requested component types were ",
                                   DataTypeSliceString(requested_dtypes));
  }
  return Status::OK();
}

Status PriorityQueue::MatchesPriorityNodeDefShapes(
    const NodeDef& node_def) const {
  std::vector<TensorShape> requested_shapes;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "shapes", &requested_shapes));
  requested_shapes.insert(requested_shapes.begin(), TensorShape({}));
  if (requested_shapes != component_shapes_) {
    return errors::InvalidArgument("Shared queue '", name_,
                                   "' has component shapes ",
                                   ShapeListString(component_shapes_),
                                   " but requested component shapes were ",
                                   ShapeListString(requested_shapes));
  }
  return Status::OK();
}

}  // namespace tensorflow
