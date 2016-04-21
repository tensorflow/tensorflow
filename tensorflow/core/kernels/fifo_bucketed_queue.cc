#include <deque>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_bucketed_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

FIFOBucketedQueue::FIFOBucketedQueue(
    int buckets, int capacity, int batch_size, const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
  : BucketedTypedQueue(buckets, capacity, component_dtypes, component_shapes,
                       name), batch_size_(batch_size) {}

void FIFOBucketedQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(size(), size_t{0});
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(*queues_[i][0].AccessTensor(ctx));
    queues_[i].pop_front();
  }
}

void FIFOBucketedQueue::TryEnqueue(
    const Tuple& tuple, OpKernelContext* ctx, DoneCallback callback) {
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
          [tuple, ctx, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Aborted("FIFOBucketedQueue '", name_, "' is closed."));
              return kComplete;
            }
            if (size() < capacity_) {
              // Get the bucket id.
              auto bucket_id_ptensor =
                  PersistentTensor(tuple[0]);
              Tensor* bucket_id_tensor = bucket_id_ptensor.AccessTensor(ctx);
              int b = bucket_id_tensor->scalar<int>()();

              bucketed_queues_[b][0].push_back(bucket_id_ptensor);
              for (int i = 1; i < num_components(); ++i) {
                bucketed_queues_[b][i].push_back(PersistentTensor(tuple[i]));
              }
              return kComplete;
            } else {
              return kNoProgress;
            }
          });
      BatchBucketedQueuesToQueuesLocked();
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
Status FIFOBucketedQueue::GetElementComponentFromBatch(const FIFOBucketedQueue::Tuple& tuple,
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

void FIFOBucketedQueue::TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
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
          [tuple, ctx, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Aborted("FIFOBucketedQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (size() < capacity_) {
              result = kProgress;
              const int64 index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              PersistentTensor bucket_id_ptensor;
              attempt->context->SetStatus(GetElementComponentFromBatch(
                  tuple, index, 0, attempt->context, &bucket_id_ptensor));
              if (!attempt->context->status().ok()) return kComplete;
              Tensor* bucket_id_tensor = bucket_id_ptensor.AccessTensor(ctx);
              int b = bucket_id_tensor->scalar<int>()();

              bucketed_queues_[b][0].push_back(bucket_id_ptensor);
              for (int i = 1; i < num_components(); ++i) {
                PersistentTensor element;
                attempt->context->SetStatus(GetElementComponentFromBatch(
                    tuple, index, i, attempt->context, &element));
                if (!attempt->context->status().ok()) return kComplete;
                bucketed_queues_[b][i].push_back(element);
              }
              --attempt->elements_requested;
              if (attempt->elements_requested == 0) {
                return kComplete;
              }
            }
            return result;
          });
      BatchBucketedQueuesToQueuesLocked();
    }
  }
  if (!already_cancelled) {
    FlushUnlocked();
  } else {
    ctx->SetStatus(errors::Cancelled("Enqueue operation was cancelled"));
    callback();
  }
}

void FIFOBucketedQueue::TryDequeue(OpKernelContext* ctx,
                                   CallbackWithTuple callback) {
  ctx->SetStatus(errors::InvalidArgument(
        "TryDequeue is invalid, maybe you want TryDequeueMany with batch_size",
        batch_size_, " ?"));
}

void FIFOBucketedQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                                       CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(
        errors::InvalidArgument("FIFOBucketedQueue's DequeueMany requires the "
                                "components to have specified shapes."));
    callback(Tuple());
    return;
  }
  OP_REQUIRES(ctx, num_elements == batch_size_, errors::InvalidArgument(
        "num_elements (", num_elements, ") must equal batch_size (",
        batch_size_, ")"));

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
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int64 s = size();
            if (closed_ && s < attempt->elements_requested) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "FIFOBucketedQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ",
                  attempt->elements_requested, ", current size ", s, ")"));

              // TODO(mrry): Add support for producing a partial batch as
              // output when the queue is closed.
              if (!attempt->tuple.empty()) {
                // Restore already-dequeued elements to the front of the queue.
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
                                           "to FIFOBucketedQueue: ",
                                           s.error_message()));
                    }
                    queues_[j].push_front(element);
                  }
                }
              }
              return kComplete;
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
                  attempt->context->allocate_temp(component_dtypes_[i], shape,
                                                  &element);
                  attempt->tuple.emplace_back(element);
                }
              }
              result = kProgress;
              Tuple tuple;
              DequeueLocked(attempt->context, &tuple);
              const int64 index =
                  attempt->tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                attempt->context->SetStatus(
                    CopyElementToSlice(tuple[i], &attempt->tuple[i], index));
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

Status FIFOBucketedQueue::MatchesNodeDef(const NodeDef& node_def) {
  TF_RETURN_IF_ERROR(MatchesNodeDefOp(node_def, "FIFOBucketedQueue"));
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));
  TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
  return Status::OK();
}

void FIFOBucketedQueue::BatchBucketedQueuesToQueuesLocked() {
  mutex_lock l(mu_);

  for (int b = 0; b < buckets_; ++b) {
    if (bucketed_queues_[b][0].size() > batch_size_) {
      // Move from bucketed_queues_ to queues_.
      for (int i = 0; i < this->num_components(); ++i) {
        for (int j = 0; j < batch_size_; ++j) {
          queues_[i].emplace_back(bucketed_queues_[b][i].front());
          bucketed_queues_[b][i].pop_front();
        }
      }
    }
  }
}

}  // namespace tensorflow
