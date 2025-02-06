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

#include <cstddef>
#include <deque>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

class RandomShuffleQueue : public TypedQueue<std::vector<Tensor> > {
 public:
  RandomShuffleQueue(int32_t capacity, int32_t min_after_dequeue, int64_t seed,
                     int64_t seed2, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name);

  absl::Status Initialize()
      override;  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------
  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  absl::Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() const override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 private:
  ~RandomShuffleQueue() override {}

  // Helper for dequeuing a single random element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static absl::Status GetElementComponentFromBatch(const Tuple& tuple,
                                                   int64_t index, int component,
                                                   OpKernelContext* ctx,
                                                   Tensor* out_tensor);

  const int32 min_after_dequeue_;
  const int64_t original_seed_;
  const int64_t original_seed2_;

  random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_
      TF_GUARDED_BY(mu_);

  RandomShuffleQueue(const RandomShuffleQueue&) = delete;
  void operator=(const RandomShuffleQueue&) = delete;
};

RandomShuffleQueue::RandomShuffleQueue(
    int32_t capacity, int32_t min_after_dequeue, int64_t seed, int64_t seed2,
    const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : TypedQueue(capacity, component_dtypes, component_shapes, name),
      min_after_dequeue_(min_after_dequeue),
      original_seed_(seed),
      original_seed2_(seed2),
      generator_(&parent_generator_) {
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = random::New64();
    seed2 = random::New64();
  }
  parent_generator_ = random::PhiloxRandom(seed, seed2);
}

absl::Status RandomShuffleQueue::Initialize() {
  TF_RETURN_IF_ERROR(TypedQueue::Initialize());

  mutex_lock lock(mu_);
  for (int i = 0; i < num_components(); ++i) {
    queues_[i].reserve(min_after_dequeue_);
  }
  return absl::OkStatus();
}

void RandomShuffleQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(queues_[0].size(), size_t{0});
  int64_t index = generator_() % queues_[0].size();
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(queues_[i][index]);
    queues_[i][index] = queues_[i].back();
    queues_[i].pop_back();
  }
}

void RandomShuffleQueue::TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
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
          [tuple, this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(errors::Cancelled(
                  "RandomShuffleQueue '", name_, "' is closed."));
              return kComplete;
            }
            if (queues_[0].size() < static_cast<size_t>(capacity_)) {
              for (int i = 0; i < num_components(); ++i) {
                queues_[i].push_back(tuple[i]);
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
absl::Status RandomShuffleQueue::GetElementComponentFromBatch(
    const Tuple& tuple, int64_t index, int component, OpKernelContext* ctx,
    Tensor* out_tensor) {
  TensorShape element_shape(tuple[component].shape());
  element_shape.RemoveDim(0);
  TF_RETURN_IF_ERROR(
      ctx->allocate_temp(tuple[component].dtype(), element_shape, out_tensor));
  TF_RETURN_IF_ERROR(
      batch_util::CopySliceToElement(tuple[component], out_tensor, index));
  return absl::OkStatus();
}

void RandomShuffleQueue::TryEnqueueMany(const Tuple& tuple,
                                        OpKernelContext* ctx,
                                        DoneCallback callback) {
  const int64_t batch_size = tuple[0].dim_size(0);
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
          [tuple, this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(errors::Cancelled(
                  "RandomShuffleQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
              const int index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                Tensor element;
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

void RandomShuffleQueue::TryDequeue(OpKernelContext* ctx,
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
          [callback, this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32_t queue_size = queues_[0].size();
            if (closed_ && queue_size == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "RandomShuffleQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ",
                  queue_size, ")"));
              return kComplete;
            }
            if (!closed_) queue_size -= min_after_dequeue_;
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

void RandomShuffleQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                                        bool allow_small_batch,
                                        CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(errors::InvalidArgument(
        "RandomShuffleQueue's DequeueMany and DequeueUpTo require the "
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
      absl::Status s = ctx->allocate_temp(component_dtypes_[i],
                                          ManyOutShape(i, 0), &element);
      if (!s.ok()) {
        ctx->SetStatus(s);
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
          [callback, allow_small_batch,
           this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32_t queue_size = queues_[0].size();
            if (closed_ && queue_size < attempt->elements_requested) {
              // If we don't have enough for a full dequeue, we have
              // to reset the attempt tuple.
              if (!attempt->tuple.empty()) {
                // Restore already-dequeued elements to the queue.
                for (int64_t i = attempt->tuple[0].dim_size(0) -
                                 attempt->elements_requested - 1;
                     i >= 0; --i) {
                  for (int j = 0; j < num_components(); ++j) {
                    Tensor element;
                    absl::Status s = GetElementComponentFromBatch(
                        attempt->tuple, i, j, attempt->context, &element);
                    if (!s.ok()) {
                      attempt->context->SetStatus(
                          errors::DataLoss("Failed to restore element from "
                                           "partially-dequeued batch "
                                           "to RandomShuffleQueue: ",
                                           s.message()));
                    }
                    queues_[j].push_back(element);
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
                      "RandomShuffleQueue '", name_, "' is closed and has ",
                      "insufficient elements (requested ",
                      attempt->elements_requested, ", current size ",
                      queue_size, ")"));
                }
                return kComplete;
              }
            }

            RunResult result = kNoProgress;
            if (!closed_) queue_size -= min_after_dequeue_;
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

absl::Status RandomShuffleQueue::MatchesNodeDef(const NodeDef& node_def) {
  if (!MatchesNodeDefOp(node_def, "RandomShuffleQueue").ok() &&
      !MatchesNodeDefOp(node_def, "RandomShuffleQueueV2").ok()) {
    return errors::InvalidArgument("Expected RandomShuffleQueue, found ",
                                   node_def.op());
  }
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));

  int32_t min_after_dequeue = -1;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "min_after_dequeue", &min_after_dequeue));
  if (min_after_dequeue != min_after_dequeue_) {
    return errors::InvalidArgument(
        "Shared queue '", name_, "' has min_after_dequeue ", min_after_dequeue_,
        " but requested min_after_dequeue was ", min_after_dequeue, ".");
  }

  int64_t seed = -1;
  int64_t seed2 = -1;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "seed", &seed));
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "seed2", &seed2));
  if ((seed != 0 || seed2 != 0) &&
      (seed != original_seed_ || seed2 != original_seed2_)) {
    return errors::InvalidArgument(
        "Shared queue '", name_, "' has random seeds (", original_seed_, ", ",
        original_seed2_, ") but requested seeds are (", seed, ", ", seed2,
        ").");
  }

  TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
  TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));

  return absl::OkStatus();
}

// Defines a RandomShuffleQueueOp, which produces a Queue (specifically, one
// backed by RandomShuffleQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class RandomShuffleQueueOp : public TypedQueueOp {
 public:
  explicit RandomShuffleQueueOp(OpKernelConstruction* context)
      : TypedQueueOp(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_after_dequeue", &min_after_dequeue_));
    OP_REQUIRES(context, min_after_dequeue_ >= 0,
                errors::InvalidArgument("min_after_dequeue ",
                                        min_after_dequeue_, " must be >= 0"));
    OP_REQUIRES(
        context, min_after_dequeue_ < capacity_,
        errors::InvalidArgument("min_after_dequeue ", min_after_dequeue_,
                                " must be < capacity ", capacity_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(context, context->GetAttr("seed2", &seed2_));

    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
  }

 private:
  absl::Status CreateResource(QueueInterface** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    RandomShuffleQueue* queue = new RandomShuffleQueue(
        capacity_, min_after_dequeue_, seed_, seed2_, component_types_,
        component_shapes_, cinfo_.name());
    return CreateTypedQueue(queue, ret);
  }

  int32 min_after_dequeue_;
  int64_t seed_;
  int64_t seed2_;
  std::vector<TensorShape> component_shapes_;

  RandomShuffleQueueOp(const RandomShuffleQueueOp&) = delete;
  void operator=(const RandomShuffleQueueOp&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("RandomShuffleQueue").Device(DEVICE_CPU),
                        RandomShuffleQueueOp);
REGISTER_KERNEL_BUILDER(Name("RandomShuffleQueueV2").Device(DEVICE_CPU),
                        RandomShuffleQueueOp);

}  // namespace tensorflow
