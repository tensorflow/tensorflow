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

// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>

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

namespace tensorflow {

class RandomShuffleQueue : public TypedQueue<std::vector<PersistentTensor> > {
 public:
  RandomShuffleQueue(int32 capacity, int32 min_after_dequeue, int64 seed,
                     int64 seed2, const DataTypeVector& component_dtypes,
                     const std::vector<TensorShape>& component_shapes,
                     const string& name);

  Status Initialize() override;  // Must be called before any other method.

  // Implementations of QueueInterface methods --------------------------------
  void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                  DoneCallback callback) override;
  void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                      DoneCallback callback) override;
  void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) override;
  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

  int32 size() override {
    mutex_lock lock(mu_);
    return queues_[0].size();
  }

 private:
  ~RandomShuffleQueue() override {}

  // Helper for dequeuing a single random element from queues_.
  void DequeueLocked(OpKernelContext* ctx, Tuple* tuple)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const int32 min_after_dequeue_;
  const int64 original_seed_;
  const int64 original_seed2_;

  random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RandomShuffleQueue);
};

RandomShuffleQueue::RandomShuffleQueue(
    int32 capacity, int32 min_after_dequeue, int64 seed, int64 seed2,
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

Status RandomShuffleQueue::Initialize() {
  Status s = TypedQueue::Initialize();
  if (!s.ok()) return s;

  mutex_lock lock(mu_);
  for (int i = 0; i < num_components(); ++i) {
    queues_.back().reserve(min_after_dequeue_);
  }
  return Status::OK();
}

void RandomShuffleQueue::DequeueLocked(OpKernelContext* ctx, Tuple* tuple) {
  DCHECK_GT(queues_[0].size(), size_t{0});
  int64 index = generator_() % queues_[0].size();
  (*tuple).reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    (*tuple).push_back(*queues_[i][index].AccessTensor(ctx));
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
          [tuple, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(errors::Aborted(
                  "RandomShuffleQueue '", name_, "' is closed."));
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

void RandomShuffleQueue::TryEnqueueMany(const Tuple& tuple,
                                        OpKernelContext* ctx,
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
              attempt->context->SetStatus(errors::Aborted(
                  "RandomShuffleQueue '", name_, "' is closed."));
              return kComplete;
            }
            RunResult result = kNoProgress;
            while (queues_[0].size() < static_cast<size_t>(capacity_)) {
              result = kProgress;
              const int index =
                  tuple[0].dim_size(0) - attempt->elements_requested;
              for (int i = 0; i < num_components(); ++i) {
                TensorShape element_shape(tuple[i].shape());
                element_shape.RemoveDim(0);
                PersistentTensor element;
                Tensor* element_access = nullptr;
                attempt->context->allocate_persistent(
                    tuple[i].dtype(), element_shape, &element, &element_access);
                attempt->context->SetStatus(
                    CopySliceToElement(tuple[i], element_access, index));
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
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32 s = queues_[0].size();
            if (closed_ && s == 0) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "RandomShuffleQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ", 1, ", current size ", s,
                  ")"));
              return kComplete;
            }
            if (!closed_) s -= min_after_dequeue_;
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

void RandomShuffleQueue::TryDequeueMany(int num_elements, OpKernelContext* ctx,
                                        CallbackWithTuple callback) {
  if (!specified_shapes()) {
    ctx->SetStatus(
        errors::InvalidArgument("RandomShuffleQueue's DequeueMany requires the "
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
      ctx->allocate_temp(component_dtypes_[i], ManyOutShape(i, 0), &element);
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
          [callback, this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            int32 s = queues_[0].size();
            if (closed_ && s < attempt->elements_requested) {
              attempt->context->SetStatus(errors::OutOfRange(
                  "RandomShuffleQueue '", name_, "' is closed and has ",
                  "insufficient elements (requested ",
                  attempt->elements_requested, ", current size ", s, ")"));
              return kComplete;
            }

            RunResult result = kNoProgress;
            if (!closed_) s -= min_after_dequeue_;
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
              const int index =
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

Status RandomShuffleQueue::MatchesNodeDef(const NodeDef& node_def) {
  TF_RETURN_IF_ERROR(MatchesNodeDefOp(node_def, "RandomShuffleQueue"));
  TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def, capacity_));

  int32 min_after_dequeue = -1;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "min_after_dequeue", &min_after_dequeue));
  if (min_after_dequeue != min_after_dequeue_) {
    return errors::InvalidArgument(
        "Shared queue '", name_, "' has min_after_dequeue ", min_after_dequeue_,
        " but requested min_after_dequeue was ", min_after_dequeue, ".");
  }

  int64 seed = -1;
  int64 seed2 = -1;
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

  return Status::OK();
}

// Defines a RandomShuffleQueueOp, which produces a Queue (specifically, one
// backed by RandomShuffleQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class RandomShuffleQueueOp : public QueueOp {
 public:
  explicit RandomShuffleQueueOp(OpKernelConstruction* context)
      : QueueOp(context) {
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

 protected:
  CreatorCallback GetCreator() const override {
    return [this](QueueInterface** ret) {
      auto* q = new RandomShuffleQueue(capacity_, min_after_dequeue_, seed_,
                                       seed2_, component_types_,
                                       component_shapes_, cinfo_.name());
      Status s = q->Initialize();
      if (s.ok()) {
        *ret = q;
      } else {
        q->Unref();
      }
      return s;
    };
  }

 private:
  int32 min_after_dequeue_;
  int64 seed_;
  int64 seed2_;
  std::vector<TensorShape> component_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(RandomShuffleQueueOp);
};

REGISTER_KERNEL_BUILDER(Name("RandomShuffleQueue").Device(DEVICE_CPU),
                        RandomShuffleQueueOp);

}  // namespace tensorflow
